"""Class-imbalance analysis (6 diseased vs 46 non-diseased; prevalence 0.115).

1. Correction strategies, applied strictly inside the training folds (also inside the inner tuning folds):
   tabular LR / RF on JVP and ECG features: none, class weighting, SMOTE, undersampling ensemble (EasyEnsemble);
   start-agnostic 1D-CNN (primary sensor model): none, weighted BCE, class-balanced focal loss, positive oversampling.
   For each: AUROC, AUPRC, MCC, sensitivity, specificity, balanced accuracy at a fixed 0.5 threshold and at a
   threshold chosen inside the training fold (Youden), Brier score and calibration-in-the-large.
2. Operating points of the main models: sensitivity at 90/95 % specificity, specificity at 100 % sensitivity.
3. PPV/NPV at realistic screening prevalences (Bayes' theorem on the cross-validated sensitivity/specificity).
4. Sample size for a validation study (Hanley-McNeil AUROC standard error; binomial CI for sensitivity).
Outputs: results/imbalance_*.csv, results/imbalance.json, figures/figS3_imbalance.{pdf,png}
Usage: python imbalance.py [--repeats 20] [--cnn-repeats 6] [--quick]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models import CNNModel
from run_benchmark import HYBRID_FEATS, JVP_FEATS, load

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
MAIN = {"JVP rule v/c>1": "JVP | rule v/c>1 (no training)", "JVP LR": "JVP | LR physiology features",
        "JVP 1D-CNN": "JVP | 1D-CNN waveform", "JVP L1-LR": "JVP | L1-LR physiology features (embedded selection)",
        "ECG LR": "ECG | LR V1/V2 features"}


# ------------------------------------------------------------------ resampling estimators
def smote(X, y, rng, k=3):
    """SMOTE (Chawla et al., 2002): interpolate each synthetic positive between a positive and one of
    its k nearest positive neighbours until the classes are balanced."""
    P = X[y == 1]
    n_new = int((y == 0).sum() - (y == 1).sum())
    if len(P) < 2 or n_new <= 0:
        return X, y
    k = min(k, len(P) - 1)
    D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(D, 1)[:, 1:k + 1]
    i = rng.integers(0, len(P), n_new)
    j = nn[i, rng.integers(0, k, n_new)]
    S = P[i] + rng.uniform(0, 1, (n_new, 1)) * (P[j] - P[i])
    return np.vstack([X, S]), np.r_[y, np.ones(n_new, int)]


class SMOTEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base=None, seed=0):
        self.base, self.seed = base, seed

    def fit(self, X, y):
        Xi = SimpleImputer(strategy="median").fit(X); self.imp_ = Xi
        Xr, yr = smote(Xi.transform(X), np.asarray(y), np.random.default_rng(self.seed))
        self.model_ = clone(self.base).fit(Xr, yr); self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.imp_.transform(X))


class EasyEnsemble(BaseEstimator, ClassifierMixin):
    """Average of B models, each trained on all positives and an equal-sized random subset of negatives
    (Liu, Wu & Zhou, 2009)."""

    def __init__(self, base=None, B=10, seed=0):
        self.base, self.B, self.seed = base, B, seed

    def fit(self, X, y):
        y = np.asarray(y); rng = np.random.default_rng(self.seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        self.imp_ = SimpleImputer(strategy="median").fit(X); X = self.imp_.transform(X)
        self.models_ = []
        for _ in range(self.B):
            idx = np.r_[pos, rng.choice(neg, len(pos), replace=False)]
            self.models_.append(clone(self.base).fit(X[idx], y[idx]))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = self.imp_.transform(X)
        return np.mean([m.predict_proba(X) for m in self.models_], 0)


def tabular(learner, strategy, seed):
    lr = lambda cw: make_pipeline(StandardScaler(), LogisticRegression(class_weight=cw, max_iter=5000))
    rf = lambda cw: RandomForestClassifier(n_estimators=200, class_weight=cw, min_samples_leaf=2,
                                           random_state=seed, n_jobs=1)
    if learner == "LR":
        base, key, grid = lr, "logisticregression__C", [0.01, 0.1, 1.0, 10.0]
    else:
        base, key, grid = rf, "max_depth", [2, 4, None]
    if strategy == "none":
        est, g = make_pipeline(SimpleImputer(strategy="median"), base(None)), {f"{('pipeline__' if learner == 'LR' else 'randomforestclassifier__')}{key}": grid}
    elif strategy == "class weight":
        est, g = make_pipeline(SimpleImputer(strategy="median"), base("balanced")), {f"{('pipeline__' if learner == 'LR' else 'randomforestclassifier__')}{key}": grid}
    elif strategy == "SMOTE":
        est, g = SMOTEClassifier(base(None), seed), {f"base__{key}": grid}
    elif strategy == "undersampling ensemble":
        est, g = EasyEnsemble(base(None), seed=seed), {f"base__{key}": grid}
    return GridSearchCV(est, g, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                        refit=True, error_score=0.5)


def youden(y, s):
    c = np.unique(s)
    j = [((s[y == 1] >= t).mean() + (s[y == 0] < t).mean()) for t in c]
    return c[int(np.argmax(j))]


def binary_metrics(y, yhat):
    tp = ((yhat == 1) & (y == 1)).sum(); tn = ((yhat == 0) & (y == 0)).sum()
    sens = tp / max((y == 1).sum(), 1); spec = tn / max((y == 0).sum(), 1)
    return dict(sens=sens, spec=spec, bal_acc=(sens + spec) / 2, mcc=matthews_corrcoef(y, yhat) if len(set(yhat)) > 1 else 0.0)


def run_tabular(X, y, learner, strategy, R):
    rows, Pall = [], []
    for r in range(R):
        P = np.zeros(len(y)); Yh05 = np.zeros(len(y), int); YhJ = np.zeros(len(y), int)
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=r).split(X, y):
            gs = tabular(learner, strategy, r).fit(X[tr], y[tr])
            P[te] = gs.predict_proba(X[te])[:, 1]
            # in-fold threshold from inner out-of-fold scores of the selected configuration
            oof = np.zeros(len(tr))
            for itr, ite in StratifiedKFold(3, shuffle=True, random_state=r + 7).split(X[tr], y[tr]):
                oof[ite] = clone(gs.best_estimator_).fit(X[tr][itr], y[tr][itr]).predict_proba(X[tr][ite])[:, 1]
            thr = youden(y[tr], oof)
            Yh05[te] = P[te] >= 0.5; YhJ[te] = P[te] >= thr
        Pall.append(P)
        rows.append(dict(repeat=r, auroc=roc_auc_score(y, P), auprc=average_precision_score(y, P),
                         brier=np.mean((P - y) ** 2), citl=P.mean() - y.mean(),
                         **{f"{k}@0.5": v for k, v in binary_metrics(y, Yh05).items()},
                         **{f"{k}@youden": v for k, v in binary_metrics(y, YhJ).items()}))
    return pd.DataFrame(rows), np.array(Pall)


def run_cnn(S, y, waves, J, strategy, R, epochs):
    rows = []
    for r in range(R):
        P = np.zeros(len(y))
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=r).split(S, y):
            m = CNNModel(waves, epochs=epochs, seeds=(r, r + 100), imbalance=strategy).fit(S[tr], y[tr])
            P[te] = m.predict_proba(S[te])
        Yh = (P >= 0.5).astype(int)
        rows.append(dict(repeat=r, auroc=roc_auc_score(y, P), auprc=average_precision_score(y, P),
                         brier=np.mean((P - y) ** 2), citl=P.mean() - y.mean(),
                         **{f"{k}@0.5": v for k, v in binary_metrics(y, Yh).items()}))
        print(f"   CNN {strategy} repeat {r}: AUROC {rows[-1]['auroc']:.3f}", flush=True)
    return pd.DataFrame(rows)


def operating_points(y, P_all):
    out = []
    for short, name in MAIN.items():
        P = P_all[name]; v = {"sens@spec90": [], "sens@spec95": [], "spec@sens100": [], "mcc@0.5": []}
        for p in P:
            neg = np.sort(p[y == 0]); pos = p[y == 1]
            for q, k in [(0.90, "sens@spec90"), (0.95, "sens@spec95")]:
                thr = neg[int(np.ceil(q * len(neg))) - 1]  # smallest threshold giving >= q specificity
                v[k].append((pos > thr).mean())
            v["spec@sens100"].append((p[y == 0] < pos.min()).mean())
            v["mcc@0.5"].append(matthews_corrcoef(y, (p >= 0.5).astype(int)))
        out.append(dict(model=short, **{k: np.mean(x) for k, x in v.items()}))
    return pd.DataFrame(out)


def ppv_npv(sens, spec, prev):
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev)
    return ppv, npv


def hanley_mcneil_se(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))


def sample_size(auc=0.90, prev=0.115, half_width=0.05):
    for n_pos in range(5, 2000):
        n_neg = int(round(n_pos * (1 - prev) / prev))
        if 1.96 * hanley_mcneil_se(auc, n_pos, n_neg) <= half_width:
            return n_pos, n_neg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--cnn-repeats", type=int, default=6); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figure-only", action="store_true", help="redraw Fig. S3 from saved results")
    a = ap.parse_args()
    if a.figure_only:
        return figure(pd.read_csv(RES / "imbalance_strategies.csv"), pd.read_csv(RES / "imbalance_ppv_npv.csv"))
    R, RC, EP = (2, 1, 30) if a.quick else (a.repeats, a.cnn_repeats, 120)
    sub, J, E, waves, _ = load(); S, y = sub.subject.values, sub.y.values
    sets = {"JVP features": J[JVP_FEATS].values, "ECG features": E.values}
    strategies = ["none", "class weight", "SMOTE", "undersampling ensemble"]
    rows = []
    for fs, X in sets.items():
        for learner in ("LR", "RF"):
            for st in strategies:
                df, _ = run_tabular(X, y, learner, st, R if learner == "LR" else max(2, R // 4))
                m = df.drop(columns="repeat").mean().to_dict()
                rows.append(dict(features=fs, model=learner, strategy=st, **m))
                print(f"{fs:13s} {learner} {st:24s} AUROC {m['auroc']:.3f} AUPRC {m['auprc']:.3f} "
                      f"sens/spec@0.5 {m['sens@0.5']:.2f}/{m['spec@0.5']:.2f} @Youden {m['sens@youden']:.2f}/{m['spec@youden']:.2f} "
                      f"MCC@Y {m['mcc@youden']:.2f} CITL {m['citl']:+.3f}", flush=True)
    for st in ["none", "weighted", "focal", "oversample"]:
        df = run_cnn(S, y, waves, J, st, RC, EP)
        m = df.drop(columns="repeat").mean().to_dict()
        rows.append(dict(features="JVP waveform", model="1D-CNN", strategy=st, **m))
        print(f"JVP 1D-CNN {st:12s} AUROC {m['auroc']:.3f} AUPRC {m['auprc']:.3f} sens/spec@0.5 {m['sens@0.5']:.2f}/{m['spec@0.5']:.2f} "
              f"MCC {m['mcc@0.5']:.2f} CITL {m['citl']:+.3f}", flush=True)
    T = pd.DataFrame(rows); T.to_csv(RES / "imbalance_strategies.csv", index=False)

    P_all = pd.read_pickle(RES / "oof_all.pkl")
    OP = operating_points(y, P_all); OP.to_csv(RES / "imbalance_operating_points.csv", index=False)
    print(OP.round(3).to_string())
    tb = pd.read_csv(RES / "table_benchmark.csv").set_index("model")
    prevs = [0.02, 0.05, 0.115, 0.20]
    pv = []
    for short, name in MAIN.items():
        se, sp = tb.loc[name, "sens"], tb.loc[name, "spec"]
        for p in prevs:
            ppv, npv = ppv_npv(se, sp, p); pv.append(dict(model=short, prevalence=p, sens=se, spec=sp, ppv=ppv, npv=npv))
    PV = pd.DataFrame(pv); PV.to_csv(RES / "imbalance_ppv_npv.csv", index=False)
    print(PV.round(3).to_string())
    ss = {}
    for auc in (0.85, 0.90, 0.95):
        npos, nneg = sample_size(auc); ss[f"auc{auc}"] = dict(n_pos=npos, n_neg=nneg, n_total=npos + nneg)
    n_sens = int(np.ceil(1.96 ** 2 * 0.9 * 0.1 / 0.1 ** 2))
    se_now = hanley_mcneil_se(0.90, 6, 46)
    out = dict(sample_size_auc_halfwidth_0p05_prev0p115=ss, n_pos_for_sens0p9_halfwidth0p1=n_sens,
               current_halfwidth_auc0p90=1.96 * se_now, prevalence=float(y.mean()))
    json.dump(out, open(RES / "imbalance.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    figure(T, PV)


def figure(T, PV):
    from make_figures import C_ACC, C_MOD, GRID, INK2, MM, panel, plt, save
    fig, axg = plt.subplots(2, 3, figsize=(180 * MM, 125 * MM), gridspec_kw=dict(wspace=0.45, hspace=1.05))
    axs = axg.ravel(); axs[5].set_axis_off()
    order = ["none", "class weight", "SMOTE", "undersampling ensemble"]
    lab = ["none", "class weight", "SMOTE", "undersampling"]
    cols = {"JVP features": C_MOD["JVP"], "ECG features": C_MOD["ECG"]}
    sty = dict(ms=3.5, lw=1.2)
    leg = dict(frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2, handlelength=1.8)
    d = T[T.model == "LR"]
    for fs, mk in [("JVP features", "o"), ("ECG features", "s")]:
        dd = d[d.features == fs].set_index("strategy").loc[order]
        axs[0].plot(range(4), dd.auroc, marker=mk, color=cols[fs], label=f"{fs.split()[0]} AUROC", **sty)
        axs[0].plot(range(4), dd.auprc, marker=mk, color=cols[fs], ls="--", label=f"{fs.split()[0]} AUPRC", **sty)
    axs[0].axhline(0.115, color=INK2, lw=0.6, ls=":"); axs[0].text(3.2, 0.13, "chance AUPRC", fontsize=5, color=INK2, ha="right")
    axs[0].set_ylim(0, 1.03); axs[0].set_ylabel("Score"); axs[0].set_title("Discrimination (LR)")
    for fs, mk in [("JVP features", "o"), ("ECG features", "s")]:
        dd = d[d.features == fs].set_index("strategy").loc[order]
        axs[1].plot(range(4), dd["sens@0.5"], marker=mk, color=cols[fs], ls=":", label=f"{fs.split()[0]}, threshold 0.5", **sty)
        axs[1].plot(range(4), dd["sens@youden"], marker=mk, color=cols[fs], label=f"{fs.split()[0]}, in-fold threshold", **sty)
    axs[1].set_ylim(-0.03, 1.05); axs[1].set_ylabel("Sensitivity"); axs[1].set_title("Operating point (LR)")
    for fs, mk in [("JVP features", "o"), ("ECG features", "s")]:
        dd = d[d.features == fs].set_index("strategy").loc[order]
        axs[2].plot(range(4), dd["citl"], marker=mk, color=cols[fs], label=f"{fs.split()[0]} LR", **sty)
    axs[2].axhline(0, color=INK2, lw=0.6); axs[2].set_ylabel("Mean predicted $-$ prevalence"); axs[2].set_title("Calibration-in-the-large ")
    for ax in axs[:3]:
        ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=6, rotation=30, ha="right"); ax.legend(**leg)
    c = T[T.model == "1D-CNN"].set_index("strategy").loc[["none", "weighted", "focal", "oversample"]]
    x = np.arange(4)
    axs[3].bar(x - 0.27, c.auroc, 0.26, color=C_MOD["JVP"], label="AUROC")
    axs[3].bar(x, c["sens@0.5"], 0.26, color=C_ACC, label="sensitivity at 0.5")
    axs[3].bar(x + 0.27, c["citl"], 0.26, color=INK2, label="mean predicted $-$ prevalence")
    axs[3].set_xticks(x); axs[3].set_xticklabels(["none", "weighted BCE", "focal loss", "oversampling"], fontsize=6, rotation=30, ha="right"); axs[3].set_ylim(0, 1.05)
    axs[3].set_title("Sensor 1D-CNN"); axs[3].legend(**leg)
    grid = np.linspace(0.01, 0.3, 60)
    for m, col, ls in [("JVP 1D-CNN", C_MOD["JVP"], "-"), ("JVP L1-LR", C_MOD["JVP"], ":"), ("ECG LR", C_MOD["ECG"], "-")]:
        dd = PV[PV.model == m]; se, sp = dd.sens.iloc[0], dd.spec.iloc[0]
        axs[4].plot(grid * 100, [ppv_npv(se, sp, g)[0] for g in grid], color=col, ls=ls, lw=1.2, label=m)
    axs[4].axvline(11.5, color=GRID, lw=0.8); axs[4].text(12.2, 0.95, "this\ncohort", fontsize=5, color=INK2, va="top")
    axs[4].set_xlabel("Prevalence (%)"); axs[4].set_ylabel("PPV"); axs[4].set_ylim(0, 1.02); axs[4].set_title("PPV vs prevalence")
    axs[4].legend(**{**leg, "bbox_to_anchor": (0.5, -0.24)})
    for ax, L in zip(axs, "abcde"):
        panel(ax, L)
    save(fig, "figS3_imbalance")


if __name__ == "__main__":
    main()
