"""Temporal check of the frozen sensor model on a later recording batch, and batch-effect checks.

1. Frozen model: the sensor-only 1D-CNN trained by deploy.py on the first batch (and locked before the later batch
   was received) is applied unchanged, with its pre-specified thresholds, to the participants of the later batch.
   The later batch contains diseased participants only, so this estimates sensitivity, not specificity.
2. Per-batch discrimination in the pooled cross-validation: AUROC of repeat-averaged out-of-fold scores for the
   diseased participants of each batch against all non-diseased participants.
3. Batch signature among the diseased: can start-agnostic sensor features tell the diseased participants of the two
   batches apart (repeated stratified 3-fold CV)? An AUROC near 0.5 argues against a recording-batch artefact.
4. Acquisition differences: sampling rate and window length of the stored windows by batch and class, how well the
   sampling rate alone tracks the label (pooled and within the first batch), and whether the primary CNN's
   out-of-fold score follows the sampling rate among non-diseased participants.

Usage: python temporal_validation.py --frozen <deploy_model dir of the earlier fit> --input <later raw batch dir>
Output: results/temporal_validation.json (aggregate), results/temporal/frozen_model_predictions.csv (per participant)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

from deploy import predict
from evaluation import repeated_cv_oof
from models import TabularModel
from run_benchmark import FOURIER_FEATS, JVP_FEATS, load

HERE = Path(__file__).parent
MAIN = {"JVP 1D-CNN": "JVP | 1D-CNN waveform", "JVP L1-LR": "JVP | L1-LR physiology features (embedded selection)",
        "JVP Fourier LR": "JVP | LR Fourier descriptors (shift-invariant)", "ECG LR": "ECG | LR V1/V2 features"}


def clopper_pearson(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=str(HERE / "archive_2026-10-09_n52/deploy_model"))
    ap.add_argument("--input", default=str(HERE.parent / "Diseased_HS_Oct_25_2026"))
    a = ap.parse_args()
    (HERE / "results/temporal").mkdir(parents=True, exist_ok=True)
    out = {}

    # 1. frozen model, pre-specified thresholds
    pred_csv = HERE / "results/temporal/frozen_model_predictions.csv"
    predict(a.frozen, a.input, "force", pred_csv)
    P = pd.read_csv(pred_csv)
    meta = json.load(open(Path(a.frozen) / "model.json"))
    fz = dict(n_train=meta["n_participants"], n_train_diseased=meta["n_diseased"], n_new=len(P), thresholds={})
    for name, t in meta["thresholds"].items():
        k = int(P[f"screen_positive_{name}"].sum())
        fz["thresholds"][name] = dict(threshold=t["threshold"], cv_specificity_at_lock=t["cv_specificity"],
                                      cv_sensitivity_at_lock=t["cv_sensitivity"], detected=k,
                                      sensitivity=k / len(P), ci=clopper_pearson(k, len(P)))
    fz["probability_median"] = float(P.probability.median())
    out["frozen_model"] = fz

    # 2. per-batch AUROC in the pooled cross-validation
    sub, J, E, waves, _ = load()
    y = sub.y.values
    first = sub.batch.iloc[0]
    later_pos = ((sub.batch != first) & (sub.y == 1)).values
    early_pos = ((sub.batch == first) & (sub.y == 1)).values
    neg = y == 0
    oof = pd.read_pickle(HERE / "results/oof_all.pkl")
    pb = {}
    for short, name in MAIN.items():
        p = oof[name].mean(0)
        pb[short] = {lab: float(roc_auc_score(np.r_[np.ones(m.sum()), np.zeros(neg.sum())], np.r_[p[m], p[neg]]))
                     for lab, m in (("first_batch_diseased", early_pos), ("later_batch_diseased", later_pos))}
    out["per_batch_auroc"] = pb

    # 3. batch signature among the diseased participants
    d = sub[sub.y == 1].reset_index(drop=True)
    yb = (d.batch != first).astype(int).values
    Jd = J.loc[d.subject]
    sig = {}
    for tag, cols, kind in (("L1-LR all features", JVP_FEATS, "L1LR"), ("LR Fourier descriptors", FOURIER_FEATS, "LR")):
        Pb = repeated_cv_oof(lambda s, c=cols, k=kind: TabularModel(k, Jd[c], s), d.subject.values, yb, n_repeats=30)
        sig[tag] = float(np.mean([roc_auc_score(yb, p) for p in Pb]))
    cyc = pd.read_csv(HERE / "data/jvp_cycles.csv", dtype={"cycle": str})
    spc = cyc.groupby(["subject", "cycle"]).size().groupby("subject").median()
    s_early, s_late = spc.loc[d.subject[yb == 0]], spc.loc[d.subject[yb == 1]]
    sig["samples_per_window_median"] = {"first_batch_diseased": float(s_early.median()),
                                        "later_batch_diseased": float(s_late.median()),
                                        "mannwhitney_p": float(mannwhitneyu(s_early, s_late).pvalue)}
    for f in ("vc_ratio", "relphase3_sin", "y_descent"):
        sig[f"{f}_median"] = {"first_batch_diseased": float(Jd[f][yb == 0].median()),
                              "later_batch_diseased": float(Jd[f][yb == 1].median()),
                              "mannwhitney_p": float(mannwhitneyu(Jd[f][yb == 0], Jd[f][yb == 1]).pvalue)}
    out["batch_signature_among_diseased"] = sig

    # 4. acquisition differences between batches
    w = cyc.groupby(["subject", "cycle"]).agg(n=("t", "size"), dur=("t", "max"))
    w["fs"] = (w.n - 1) / w.dur
    ws = w.groupby("subject")[["fs", "dur"]].median().loc[sub.subject]
    b1 = (sub.batch == first).values
    acq = {}
    for k in ("fs", "dur"):
        acq[k] = {"first_batch_controls": float(ws[k][b1 & neg].median()),
                  "first_batch_diseased": float(ws[k][early_pos].median()),
                  "later_batch_diseased": float(ws[k][later_pos].median())}
    acq["fs_auroc_lower_is_disease_pooled"] = float(roc_auc_score(y, -ws.fs))
    acq["fs_auroc_lower_is_disease_first_batch"] = float(roc_auc_score(y[b1], -ws.fs[b1]))
    r = spearmanr(oof[MAIN["JVP 1D-CNN"]].mean(0)[neg], ws.fs.values[neg])
    acq["cnn_score_vs_fs_controls_spearman"] = {"rho": float(r.statistic), "p": float(r.pvalue)}
    out["acquisition"] = acq
    json.dump(out, open(HERE / "results/temporal_validation.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
