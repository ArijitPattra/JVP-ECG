"""Models: classical ML on physiology features, ROCKET and 1D-CNNs on raw waveforms.

All models expose fit(train_subjects) / predict_proba(test_subjects) on subject IDs so the
same subject-level cross-validation loop can drive every model. Waveform models are trained
on individual cycles and averaged to a subject-level probability.
"""
import warnings

import numpy as np

import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

torch.set_num_threads(4)


# ------------------------------------------------------------------ tabular models
def tabular_estimator(name, seed=0):
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    pre = [SimpleImputer(strategy="median"), StandardScaler()]
    if name == "LR":
        est = make_pipeline(*pre, LogisticRegression(class_weight="balanced", max_iter=5000))
        grid = {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}
    elif name == "L1LR":  # sparse logistic regression: feature selection embedded in each training fold
        est = make_pipeline(*pre, LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=5000))
        grid = {"logisticregression__C": [0.1, 0.3, 1.0, 3.0]}
    elif name == "SVM":
        est = make_pipeline(*pre, SVC(class_weight="balanced", probability=True, random_state=seed))
        grid = {"svc__C": [0.1, 1.0, 10.0], "svc__gamma": ["scale", 0.01]}
    elif name == "RF":
        est = make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                                   min_samples_leaf=2, random_state=seed, n_jobs=1))
        grid = {"randomforestclassifier__max_depth": [2, 4, None]}
    elif name == "GBM":  # gradient-boosted trees (sklearn; XGBoost deadlocks next to PyTorch's OpenMP)
        est = make_pipeline(SimpleImputer(strategy="median"),
                            GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, subsample=0.8,
                                                       random_state=seed))
        grid = {"gradientboostingclassifier__max_depth": [1, 2, 3]}
    else:
        raise ValueError(name)
    return GridSearchCV(est, grid, scoring="roc_auc", cv=inner, refit=True, error_score=0.5)


class TabularModel:
    """Nested-CV tuned tabular model. The decision threshold is chosen inside the training fold
    (Youden index on inner out-of-fold scores) and folded into the output by a monotone shift,
    so 0.5 on the returned score corresponds to that threshold and AUROC is unchanged."""

    def __init__(self, name, X, seed=0):
        self.name, self.X, self.seed = name, X, seed  # X: DataFrame indexed by subject

    def _score(self, est, X):
        if self.name == "SVM":  # Platt scaling is unreliable with ~4 positives; rank by margin
            return 1 / (1 + np.exp(-est.decision_function(X)))
        return est.predict_proba(X)[:, 1]

    def fit(self, subj, y):
        X = self.X.loc[subj].values
        self.est = tabular_estimator(self.name, self.seed).fit(X, y)
        inner = StratifiedKFold(3, shuffle=True, random_state=self.seed + 7)
        oof = np.zeros(len(y))
        for tr, te in inner.split(X, y):
            e = clone(self.est.best_estimator_).fit(X[tr], y[tr])
            oof[te] = self._score(e, X[te])
        cand = np.unique(oof)
        youden = [((oof[y == 1] >= c).mean() + (oof[y == 0] < c).mean()) for c in cand]
        self.thr = float(np.clip(cand[int(np.argmax(youden))], 1e-4, 1 - 1e-4))
        return self

    def predict_proba(self, subj):
        p = np.clip(self._score(self.est, self.X.loc[subj].values), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)) - np.log(self.thr / (1 - self.thr))
        return 1 / (1 + np.exp(-logit))


class RuleModel:
    """Untrained clinical rule: diseased if v/c ratio > 1 (writeup hypothesis)."""

    def __init__(self, score):
        self.score = score

    def fit(self, subj, y):
        return self

    def predict_proba(self, subj):
        s = self.score.loc[subj].values
        return 1 / (1 + np.exp(-8 * (s - 1.0)))  # monotone map; 0.5 at v/c = 1


# ------------------------------------------------------------------ ROCKET
class Rocket:
    """Random convolutional kernels (Dempster et al., 2020) for short univariate series."""

    def __init__(self, n_kernels=1000, length=64, seed=0):
        rng = np.random.default_rng(seed)
        self.k = []
        for _ in range(n_kernels):
            L = rng.choice([5, 7, 9])
            w = rng.normal(size=L)
            w -= w.mean()
            b = rng.uniform(-1, 1)
            dmax = max(0, np.log2((length - 1) / (L - 1)))
            d = int(2 ** rng.uniform(0, dmax))
            pad = ((L - 1) * d) // 2 if rng.integers(2) else 0
            self.k.append((w, b, d, pad))

    def transform(self, X):
        """Circular convolution, so PPV and max features are invariant to circular shifts of X."""
        X = np.asarray(X, float)
        L = X.shape[1]
        out = np.zeros((len(X), 2 * len(self.k)))
        for j, (w, b, d, _pad) in enumerate(self.k):
            idx = (np.arange(L)[:, None] + np.arange(len(w))[None, :] * d) % L
            c = X[:, idx] @ w + b  # (N, L)
            out[:, 2 * j] = (c > 0).mean(1)
            out[:, 2 * j + 1] = c.max(1)
        return out


_ROCKET_CACHE = {}


class RocketModel:
    def __init__(self, cycles, seed=0, n_kernels=1000):
        # cycles: list of (subject, cycle_id, waveform[64])
        self.cyc = cycles
        self.seed = seed
        key = (id(cycles), seed, n_kernels)
        if key not in _ROCKET_CACHE:  # transform is label-free: compute once per kernel set
            W = np.array([c[2] for c in cycles])
            rk = Rocket(n_kernels, W.shape[1], seed)
            _ROCKET_CACHE[key] = rk.transform((W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-9))
        self.F = _ROCKET_CACHE[key]
        self.subj = np.array([c[0] for c in cycles])

    def fit(self, subj, y):
        lab = dict(zip(subj, y))
        m = np.isin(self.subj, subj)
        yy = np.array([lab[s] for s in self.subj[m]])
        self.clf = make_pipeline(StandardScaler(),
                                 LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000))
        self.clf.fit(self.F[m], yy)
        return self

    def predict_proba(self, subj):
        p = self.clf.predict_proba(self.F[np.isin(self.subj, subj)])[:, 1]
        s = self.subj[np.isin(self.subj, subj)]
        return np.array([p[s == q].mean() for q in subj])


# ------------------------------------------------------------------ 1D CNN
class CNN1D(nn.Module):
    def __init__(self, c_in=2, width=16, n_extra=0, dropout=0.3, circular=False):
        super().__init__()
        pm = "circular" if circular else "zeros"  # circular padding for periodic JVP windows
        self.f = nn.Sequential(
            nn.Conv1d(c_in, width, 7, padding=3, padding_mode=pm), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, 2 * width, 5, padding=2, padding_mode=pm), nn.BatchNorm1d(2 * width), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(2 * width, 2 * width, 3, padding=1, padding_mode=pm), nn.BatchNorm1d(2 * width), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * width + n_extra, 1))

    def forward(self, x, extra=None, return_map=False):
        h = self.f(x)
        z = h.mean(-1)
        if extra is not None:
            z = torch.cat([z, extra], 1)
        out = self.head(z).squeeze(-1)
        return (out, h) if return_map else out


def focal_loss(alpha=0.5, gamma=2.0):
    """Binary focal loss (Lin et al., 2017) with class-balancing weight alpha for the positive class."""
    def f(logits, target):
        p = torch.sigmoid(logits)
        pt = torch.where(target == 1, p, 1 - p)
        a = torch.where(target == 1, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (a * (1 - pt) ** gamma * bce).mean()
    return f


def augment(x, rng):
    """x: (B, 64) min-max normalised cycles. Amplitude, offset, time-warp, jitter, wander."""
    B, L = x.shape
    t = np.linspace(0, 1, L)
    out = np.empty_like(x)
    for i in range(B):
        k = rng.uniform(-0.08, 0.08)  # smooth monotone time warp
        tw = np.clip(t + k * np.sin(np.pi * t), 0, 1)
        xi = np.interp(tw, t, x[i])
        xi = xi * rng.uniform(0.85, 1.15) + rng.normal(0, 0.03, L)
        xi += rng.uniform(-0.1, 0.1) * np.sin(2 * np.pi * (t * rng.uniform(0.3, 1.0) + rng.uniform()))
        out[i] = xi
    return out


def two_channel(x):
    d = (np.roll(x, -1, axis=-1) - np.roll(x, 1, axis=-1)) * x.shape[-1] / 16.0  # circular derivative
    return np.stack([x - 0.5, d], 1).astype(np.float32)


class CNNModel:
    """Cycle-level 1D-CNN, subject probability = mean over cycles; small seed ensemble."""

    def __init__(self, cycles, extra=None, epochs=120, seeds=(0, 1, 2), lr=3e-3, wd=1e-3,
                 aug=True, c_in=2, width=16, imbalance="weighted", circular=None):
        self.cyc = cycles
        self.subj = np.array([c[0] for c in cycles])
        self.W = np.array([c[2] for c in cycles], dtype=np.float32)
        self.extra = extra  # optional DataFrame of subject-level features (hybrid model)
        self.epochs, self.seeds, self.lr, self.wd, self.aug = epochs, seeds, lr, wd, aug
        self.c_in, self.width = c_in, width
        # imbalance handling: "weighted" (positive-class weighted BCE; default), "none" (plain BCE),
        # "focal" (class-balanced focal loss, gamma = 2), "oversample" (positives resampled to parity)
        self.imbalance = imbalance
        # JVP windows are treated as periodic with unknown start: circular convolutions, random circular
        # shifts during training and averaging over shifts at test time make the model start-agnostic
        self.circular = (self.W.ndim == 2) if circular is None else circular

    def _inputs(self, idx, rng=None):
        x = self.W[idx]
        if self.c_in == 2 and x.ndim == 2:
            if rng is not None and self.circular:
                x = np.stack([np.roll(xi, rng.integers(x.shape[1])) for xi in x])
            if rng is not None and self.aug:
                x = augment(x, rng)
            return torch.from_numpy(two_channel(x))
        if rng is not None and self.aug:
            x = x * rng.uniform(0.9, 1.1, (len(x), 1, 1)) + rng.normal(0, 0.02, x.shape)
        return torch.from_numpy(x.astype(np.float32))

    def _extra(self, idx):
        if self.extra is None:
            return None
        e = (self.extra.loc[self.subj[idx]].values - self.mu) / self.sd
        return torch.from_numpy(np.nan_to_num(e).astype(np.float32))

    def fit(self, subj, y):
        lab = dict(zip(subj, y))
        idx = np.where(np.isin(self.subj, subj))[0]
        yy = torch.tensor([lab[s] for s in self.subj[idx]], dtype=torch.float32)
        if self.extra is not None:
            E = self.extra.loc[subj].values
            self.mu, self.sd = np.nanmean(E, 0), np.nanstd(E, 0) + 1e-6
        pos = float(yy.sum())
        ratio = (len(yy) - pos) / max(pos, 1.0)
        pw = torch.tensor(ratio if self.imbalance == "weighted" else 1.0)
        self.nets = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = CNN1D(self.c_in, self.width, 0 if self.extra is None else self.extra.shape[1], circular=self.circular)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
            lossf = focal_loss(alpha=ratio / (1 + ratio)) if self.imbalance == "focal" else nn.BCEWithLogitsLoss(pos_weight=pw)
            ylocal = yy.numpy()
            for ep in range(self.epochs):
                net.train()
                if self.imbalance == "oversample":  # draw positives as often as negatives
                    p_draw = np.where(ylocal == 1, 1.0 / max(ylocal.sum(), 1), 1.0 / max((1 - ylocal).sum(), 1))
                    perm = rng.choice(len(idx), len(idx), replace=True, p=p_draw / p_draw.sum())
                else:
                    perm = rng.permutation(len(idx))
                for b in range(0, len(perm), 32):
                    bi = idx[perm[b: b + 32]]
                    if len(bi) < 2:
                        continue
                    xb = self._inputs(bi, rng)
                    loss = lossf(net(xb, self._extra(bi)), yy[perm[b: b + 32]])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sched.step()
            net.eval()
            self.nets.append(net)
        return self

    @torch.no_grad()
    def cycle_proba(self, idx, W=None):
        base = self.W[idx] if W is None else W
        shifts = range(0, base.shape[-1], base.shape[-1] // 8) if (self.circular and base.ndim == 2) else [0]
        ps = []
        for sh in shifts:  # test-time averaging over circular shifts
            xb = np.roll(base, sh, axis=-1)
            x = torch.from_numpy(two_channel(xb)) if (self.c_in == 2 and xb.ndim == 2) else torch.from_numpy(xb.astype(np.float32))
            ps += [torch.sigmoid(n(x, self._extra(idx))).numpy() for n in self.nets]
        return np.mean(ps, 0)

    def predict_proba(self, subj, W_override=None):
        idx = np.where(np.isin(self.subj, subj))[0]
        p = self.cycle_proba(idx, None if W_override is None else W_override[idx])
        s = self.subj[idx]
        return np.array([p[s == q].mean() for q in subj])

    def saliency(self, idx):
        """Gradient x input saliency on the waveform channel, per cycle."""
        x = self._inputs(idx).requires_grad_(True)
        tot = 0
        for n in self.nets:
            out = n(x, self._extra(idx)).sum()
            g, = torch.autograd.grad(out, x)
            tot = tot + (g * x).detach().numpy()[:, 0]
        return tot / len(self.nets)
