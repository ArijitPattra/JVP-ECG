"""Subject-level repeated stratified cross-validation, metrics, bootstrap, permutation, DeLong."""
import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold


def repeated_cv_oof(make_model, subjects, y, n_repeats=20, n_folds=3, seed0=0, verbose=False):
    """Returns OOF probabilities, shape (n_repeats, n_subjects). Model is re-created per fold."""
    subjects, y = np.asarray(subjects), np.asarray(y)
    P = np.full((n_repeats, len(y)), np.nan)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed0 + r)
        for tr, te in skf.split(subjects, y):
            m = make_model(seed0 + r).fit(subjects[tr], y[tr])
            P[r, te] = m.predict_proba(subjects[te])
        if verbose:
            print(f"  repeat {r + 1}/{n_repeats}: AUROC={roc_auc_score(y, P[r]):.3f}", flush=True)
    return P


def threshold_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return dict(sens=sens, spec=spec, ppv=ppv, npv=npv, f1=f1,
                bal_acc=balanced_accuracy_score(y, yhat), tp=tp, fn=fn, tn=tn, fp=fp)


def summarise(y, P, n_boot=2000, seed=0):
    """Per-repeat metrics (mean, 2.5-97.5 pct) + subject bootstrap CI on repeat-averaged probs."""
    y = np.asarray(y)
    per = []
    for p in P:
        d = dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                 brier=brier_score_loss(y, np.clip(p, 0, 1)))
        d.update(threshold_metrics(y, p))
        per.append(d)
    keys = ["auroc", "auprc", "brier", "sens", "spec", "ppv", "npv", "f1", "bal_acc"]
    out = {}
    for k in keys:
        v = np.array([d[k] for d in per])
        out[k] = v.mean(); out[k + "_rep_lo"] = np.percentile(v, 2.5); out[k + "_rep_hi"] = np.percentile(v, 97.5)
    pm = P.mean(0)
    rng = np.random.default_rng(seed)
    bs = {"auroc": [], "auprc": [], "sens": [], "spec": [], "bal_acc": []}
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n_boot):  # stratified bootstrap keeps both classes present
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yb, pb = y[idx], pm[idx]
        bs["auroc"].append(roc_auc_score(yb, pb)); bs["auprc"].append(average_precision_score(yb, pb))
        tm = threshold_metrics(yb, pb)
        for k in ("sens", "spec", "bal_acc"):
            bs[k].append(tm[k])
    for k, v in bs.items():
        out[k + "_boot_lo"], out[k + "_boot_hi"] = np.percentile(v, [2.5, 97.5])
    out["auroc_meanprob"] = roc_auc_score(y, pm)
    return out


def permutation_pvalue(make_model, subjects, y, observed_auc, n_perm=200, n_repeats=3, seed=0):
    """Ojala & Garriga (2010) test 1: re-run the full CV on label-permuted data."""
    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_perm):
        yp = rng.permutation(y)
        P = repeated_cv_oof(make_model, subjects, yp, n_repeats=n_repeats, seed0=1000 + i)
        null.append(np.mean([roc_auc_score(yp, p) for p in P]))
    null = np.array(null)
    return (1 + np.sum(null >= observed_auc)) / (1 + n_perm), null


# ---------------------------------------------------------------- DeLong (paired)
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_paired(y, p1, p2):
    """Two-sided DeLong test for two correlated AUCs (DeLong et al., Biometrics 1988)."""
    y = np.asarray(y); pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, V10, V01 = [], [], []
    for p in (p1, p2):
        p = np.asarray(p)
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
        auc = tz[:m].sum() / (m * n) - (m + 1) / (2 * n)
        aucs.append(auc)
        V10.append((tz[:m] - tx) / n); V01.append(1 - (tz[m:] - ty) / m)
    S10, S01 = np.cov(np.array(V10)), np.cov(np.array(V01))
    S = S10 / m + S01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return aucs[0], aucs[1], 2 * stats.norm.sf(abs(z))


def bootstrap_auc_ci(y, s, n_boot=2000, seed=0):
    y, s = np.asarray(y), np.asarray(s); rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = [roc_auc_score(y[i], s[i]) for i in
         (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n_boot))]
    return roc_auc_score(y, s), np.percentile(v, 2.5), np.percentile(v, 97.5)


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
