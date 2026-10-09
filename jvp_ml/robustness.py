"""Sensitivity, robustness and JVP-vs-ECG agreement analyses.

1. QC exclusion: drop subjects whose workbook was copied from another subject
   (duplicated ECG and/or header subject mismatch).
2. Sensor channel: raw current vs calibrated force.
3. Number of cycles: first cycle only vs all cycles.
4. Test-time noise (SNR 30-0 dB) and decimation (1/2, 1/3 of samples) for models trained on clean data.
5. JVP-ECG feature correlations; agreement of the v/c rule and of the ECG RVH criteria with the diagnosis.
6. Alignment test: a model that reads the raw window (not shift-invariant) versus shift-invariant models, on the
   windows as stored and after random circular shifts, to show how much an alignment-dependent analysis would
   depend on where the windows happen to start.
Usage: python robustness.py [--repeats 20]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from evaluation import repeated_cv_oof, summarise
from features import jvp_subject_features, minmax, resample_cycle
from models import CNNModel, RocketModel, RuleModel, TabularModel
from run_benchmark import HYBRID_FEATS, JVP_FEATS, load

HERE = Path(__file__).parent
QC_EXCLUDE = ["HS31", "HS42", "HS43", "HS48", "HS52"]


def core_models(J, E, waves, quick=False):
    ep = 40 if quick else 120
    return {
        "JVP rule v/c>1": lambda s: RuleModel(J["vc_ratio"]),
        "JVP LR physiology": lambda s: TabularModel("LR", J[JVP_FEATS], s),
        "JVP ROCKET": lambda s: RocketModel(waves, s),
        "JVP 1D-CNN": lambda s: CNNModel(waves, epochs=ep, seeds=(s, s + 100)),  # primary sensor model
        "JVP L1-LR": lambda s: TabularModel("L1LR", J[JVP_FEATS], s),
        "ECG LR V1/V2": lambda s: TabularModel("LR", E, s),
    }


def run_set(tag, sub, J, E, waves, R, quick):
    rows = []
    for name, mk in core_models(J, E, waves, quick).items():
        r = max(2, R // 3) if "CNN" in name else R
        P = repeated_cv_oof(mk, sub.subject.values, sub.y.values, n_repeats=r)
        sm = summarise(sub.y.values, P, n_boot=1000)
        rows.append(dict(analysis=tag, model=name, n=len(sub), n_pos=int(sub.y.sum()), **{
            k: sm[k] for k in ("auroc", "auroc_boot_lo", "auroc_boot_hi", "auprc", "sens", "spec", "bal_acc")}))
        print(tag, name, f"{sm['auroc']:.3f}", flush=True)
    return rows


def corrupt(cyc, snr_db=None, decimate=1, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for (s, c), g in cyc.groupby(["subject", "cycle"], sort=False):
        g = g.iloc[::decimate].copy() if decimate > 1 and len(g) // decimate >= 4 else g.copy()
        if snr_db is not None:
            for col in ("force_N", "current_A"):
                sd = g[col].std() / (10 ** (snr_db / 20))
                g[col] = g[col] + rng.normal(0, sd, len(g))
        out.append(g)
    return pd.concat(out)


def degradation(sub, cyc, J_clean, waves_clean, R=10):
    """Train on clean data, evaluate held-out folds on corrupted signals."""
    S, y = sub.subject.values, sub.y.values
    conds = [("clean", None, 1)] + [(f"SNR {d} dB", d, 1) for d in (30, 20, 15, 10, 5, 0)] + \
            [("1/2 samples", None, 2), ("1/3 samples", None, 3)]
    res = []
    for label, snr, dec in conds:
        aucs = {"rule": [], "LR": [], "CNN": []}
        for r in range(R):
            cc = corrupt(cyc, snr, dec, seed=r) if label != "clean" else cyc
            Jc, wc = jvp_subject_features(cc)
            Jc = Jc.set_index("subject").loc[S]
            Wc = np.array([w[2] for w in wc], dtype=np.float32)
            p = {k: np.zeros(len(y)) for k in aucs}
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=r).split(S, y):
                lr = TabularModel("L1LR", J_clean[JVP_FEATS], r).fit(S[tr], y[tr])
                lr.X = pd.concat([J_clean[JVP_FEATS].loc[S[tr]], Jc[JVP_FEATS].loc[S[te]]])
                p["LR"][te] = lr.predict_proba(S[te])
                p["rule"][te] = RuleModel(Jc["vc_ratio"]).predict_proba(S[te])
                cnn = CNNModel(waves_clean, epochs=80, seeds=(r,)).fit(S[tr], y[tr])
                p["CNN"][te] = cnn.predict_proba(S[te], W_override=Wc)
            for k in aucs:
                aucs[k].append(roc_auc_score(y, p[k]))
        for k, v in aucs.items():
            res.append(dict(condition=label, model=k, auroc=np.mean(v), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
        print("degradation", label, {k: round(np.mean(v), 3) for k, v in aucs.items()}, flush=True)
    return pd.DataFrame(res)


def agreement(sub, J, E):
    ok = ~sub.subject.isin(QC_EXCLUDE).values
    out = {"n": int(ok.sum())}
    jf = ["vc_ratio", "slope_a_to_c", "y_descent", "amp_c", "relphase3_sin", "phase_x_to_v", "beat_consistency"]
    ef = ["V1_R_mV", "V1_S_mV", "V1_RS_ratio", "V1_PR_ms", "V1_QRS_ms", "V2_R_mV", "V2_S_mV", "V2_T_mV", "V1_beat_ms"]
    C = pd.DataFrame(index=jf, columns=ef, dtype=float); Pv = C.copy()
    for a in jf:
        for b in ef:
            rr, pp = spearmanr(J[a].values[ok], E[b].values[ok]); C.loc[a, b] = rr; Pv.loc[a, b] = pp
    C.to_csv(HERE / "results/jvp_ecg_spearman_r.csv"); Pv.to_csv(HERE / "results/jvp_ecg_spearman_p.csv")
    rule = (J["vc_ratio"] > 1).astype(int).values
    rvh = ((E["V1_R_mV"] >= 0.7) | (E["V1_RS_ratio"] >= 1)).astype(int).values
    y = sub.y.values
    out["ecg_rvh_criteria_sens"] = float(rvh[y == 1].mean()); out["ecg_rvh_criteria_spec"] = float(1 - rvh[y == 0].mean())
    out["rule_sens"] = float(rule[y == 1].mean()); out["rule_spec"] = float(1 - rule[y == 0].mean())
    out["n_rvh_tp"] = int(rvh[y == 1].sum()); out["n_rule_tp"] = int(rule[y == 1].sum()); out["n_rule_tn"] = int((1 - rule)[y == 0].sum())
    return out


def alignment_test(sub, R=20):
    """LR on the raw 64-point window (alignment-dependent) vs shift-invariant Fourier descriptors,
    on stored windows and after a random circular shift of every window."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cyc = pd.read_csv(HERE / "data/jvp_cycles.csv", dtype={"cycle": str})
    S, y = sub.subject.values, sub.y.values
    W, subj = [], []
    for (s, c), g in cyc.groupby(["subject", "cycle"], sort=False):
        W.append(minmax(resample_cycle(g.force_N.values))); subj.append(s)
    W, subj = np.array(W), np.array(subj)

    def fourier(w):
        H = np.fft.rfft(w - w.mean()); m = np.abs(H); m = m / (m[1:7].sum() + 1e-9); ph = np.angle(H)
        return list(m[1:7]) + sum([[np.cos(ph[k] - k * ph[1]), np.sin(ph[k] - k * ph[1])] for k in (2, 3, 4)], [])

    def cv(X, C):
        a = []
        for r in range(R):
            P = np.zeros(len(y))
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=r).split(S, y):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
                P[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            a.append(roc_auc_score(y, P))
        return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    out = {}
    rng = np.random.default_rng(0)
    for label, Wx in [("stored windows", W), ("random circular shift", np.array([np.roll(w, rng.integers(64)) for w in W]))]:
        raw = np.array([Wx[subj == s].mean(0) for s in S])
        fou = np.array([np.mean([fourier(w) for w in Wx[subj == s]], 0) for s in S])
        out[label] = {"raw_window_LR": cv(raw, 0.01), "shift_invariant_fourier_LR": cv(fou, 0.1)}
        print("alignment test", label, out[label], flush=True)
    # where does the deepest trough sit in the stored windows?
    pos = np.array([np.argmin(w) / 64 for w in W]); lab = dict(zip(sub.subject, sub.y))
    yy = np.array([lab[s] for s in subj])
    out["trough_position_median"] = float(np.median(pos)); out["trough_position_iqr"] = [float(np.percentile(pos, 25)), float(np.percentile(pos, 75))]
    out["trough_position_median_by_class"] = {"control": float(np.median(pos[yy == 0])), "diseased": float(np.median(pos[yy == 1]))}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    R = 2 if a.quick else a.repeats
    rows = []
    sub, J, E, waves, _ = load()
    st = {"agreement": agreement(sub, J, E), "alignment_test": alignment_test(sub, R=4 if a.quick else 20)}
    print(json.dumps(st, indent=1, default=str), flush=True)
    rows += run_set("primary (all 52)", sub, J, E, waves, R, a.quick)
    s2, J2, E2, w2, _ = load(exclude=QC_EXCLUDE)
    rows += run_set("QC-flagged workbooks excluded", s2, J2, E2, w2, R, a.quick)
    s3, J3, E3, w3, _ = load(signal="current_A")
    rows += run_set("raw current instead of force", s3, J3, E3, w3, R, a.quick)
    cyc = pd.read_csv(HERE / "data/jvp_cycles.csv", dtype={"cycle": str})
    first = cyc.groupby("subject").cycle.transform(lambda c: c == c.iloc[0])
    J4, w4 = jvp_subject_features(cyc[first]); J4 = J4.set_index("subject").loc[sub.subject]
    J4["vc_ratio_sd"] = 0.0; J4["beat_consistency"] = 1.0  # undefined with a single window
    rows += run_set("single cycle per subject", sub, J4, E, w4, R, a.quick)
    pd.DataFrame(rows).to_csv(HERE / "results/sensitivity_analyses.csv", index=False)
    deg = degradation(sub, cyc, J, waves, R=3 if a.quick else 10)
    deg.to_csv(HERE / "results/degradation.csv", index=False)
    json.dump(st, open(HERE / "results/agreement.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
