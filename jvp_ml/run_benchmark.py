"""Main benchmark: sensor-only JVP models vs ECG (V1/V2) comparator models, subject-level CV.

Intended use: the sensor alone at deployment; ECG models serve only as a benchmark, so no model combines the two.

Usage: python run_benchmark.py [--repeats 30] [--cnn-repeats 10] [--perms 200] [--quick]
Outputs (results/): oof_<model>.npy, benchmark_summary.csv, stats.json, univariate.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import (bootstrap_auc_ci, cohen_kappa, delong_paired, permutation_pvalue,
                        repeated_cv_oof, summarise)
from features import ecg_subject_features, ecg_waveforms, jvp_subject_features
from models import CNNModel, RocketModel, RuleModel, TabularModel

HERE = Path(__file__).parent
JVP_FEATS = ["vc_ratio", "va_ratio", "ca_ratio", "amp_v", "amp_c", "amp_a", "y_descent", "amp_y",
             "phase_x_to_v", "phase_c_to_x", "phase_v_to_c", "slope_a_to_c", "slope_v_upstroke", "slope_x_descent",
             "n_peaks", "skewness", "kurtosis", "frac_above_half", "roughness",
             "harm1", "harm2", "harm3", "harm4", "harm5", "harm6",
             "relphase2_cos", "relphase2_sin", "relphase3_cos", "relphase3_sin", "relphase4_cos", "relphase4_sin",
             "vc_ratio_sd", "beat_consistency"]
FOURIER_FEATS = ["harm1", "harm2", "harm3", "harm4", "harm5", "harm6", "relphase2_cos", "relphase2_sin",
                 "relphase3_cos", "relphase3_sin", "relphase4_cos", "relphase4_sin"]
HYBRID_FEATS = ["vc_ratio", "slope_a_to_c"]  # the two a-priori clinical hypotheses


def load(signal="force_N", exclude=()):
    cyc = pd.read_csv(HERE / "data/jvp_cycles.csv", dtype={"cycle": str})
    ecg = pd.read_csv(HERE / "data/ecg_beats.csv")
    sub = pd.read_csv(HERE / "data/subjects.csv")
    sub = sub[~sub.subject.isin(exclude)].reset_index(drop=True)
    cyc, ecg = cyc[cyc.subject.isin(sub.subject)], ecg[ecg.subject.isin(sub.subject)]
    J, waves = jvp_subject_features(cyc, signal)
    E = ecg_subject_features(ecg)
    J, E = J.set_index("subject").loc[sub.subject], E.set_index("subject").loc[sub.subject]
    ecg_w = ecg_waveforms(ecg)
    ecg_cycles = [(s, "ecg", ecg_w[s]) for s in sub.subject]
    return sub, J, E, waves, ecg_cycles


def model_zoo(J, E, waves, ecg_cycles, cnn_epochs=120):
    JX, EX = J[JVP_FEATS], E
    zoo = {
        "JVP | rule v/c>1 (no training)": ("JVP", lambda s: RuleModel(J["vc_ratio"])),
        "JVP | LR log(v/c) only": ("JVP", lambda s: TabularModel("LR", np.log(J[["vc_ratio"]]), s)),
    }
    for name in ("LR", "SVM", "RF", "GBM"):
        zoo[f"JVP | {name} physiology features"] = ("JVP", lambda s, n=name: TabularModel(n, JX, s))
    zoo["JVP | L1-LR physiology features (embedded selection)"] = ("JVP", lambda s: TabularModel("L1LR", JX, s))
    zoo["JVP | LR Fourier descriptors (shift-invariant)"] = ("JVP", lambda s: TabularModel("LR", J[FOURIER_FEATS], s))
    zoo["JVP | ROCKET (random conv kernels)"] = ("JVP", lambda s: RocketModel(waves, s))
    zoo["JVP | 1D-CNN waveform"] = ("JVP", lambda s: CNNModel(waves, epochs=cnn_epochs, seeds=(s, s + 100, s + 200)))
    zoo["JVP | Physio-hybrid CNN (waveform + v/c, a-c slope)"] = (
        "JVP", lambda s: CNNModel(waves, extra=J[HYBRID_FEATS], epochs=cnn_epochs, seeds=(s, s + 100, s + 200)))
    for name in ("LR", "SVM", "RF", "GBM"):
        zoo[f"ECG | {name} V1/V2 features"] = ("ECG", lambda s, n=name: TabularModel(n, EX, s))
    zoo["ECG | 1D-CNN V1/V2 beat"] = ("ECG", lambda s: CNNModel(ecg_cycles, epochs=cnn_epochs, c_in=2,
                                                                 seeds=(s, s + 100, s + 200)))
    return zoo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--cnn-repeats", type=int, default=10)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(exist_ok=True)
    sub, J, E, waves, ecg_cycles = load()
    S, y = sub.subject.values, sub.y.values
    zoo = model_zoo(J, E, waves, ecg_cycles, cnn_epochs=40 if a.quick else 120)

    rows, P_all = [], {}
    for name, (mod, make) in zoo.items():
        heavy = "CNN" in name or "ROCKET" in name or "RF" in name or "GBM" in name
        R = (2 if a.quick else (a.cnn_repeats if heavy else a.repeats))
        t0 = time.time()
        P = repeated_cv_oof(make, S, y, n_repeats=R)
        P_all[name] = P
        np.save(out / f"oof_{name.replace(' ', '_').replace('|', '').replace('/', '-')}.npy", P)
        sm = summarise(y, P)
        sm.update(model=name, modality=mod, repeats=R, seconds=time.time() - t0)
        rows.append(sm)
        print(f"{name:60s} AUROC {sm['auroc']:.3f} [{sm['auroc_boot_lo']:.2f},{sm['auroc_boot_hi']:.2f}] "
              f"AUPRC {sm['auprc']:.3f} sens {sm['sens']:.2f} spec {sm['spec']:.2f} ({sm['seconds']:.0f}s)", flush=True)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "benchmark_summary.csv", index=False)
    pd.to_pickle({k: v for k, v in P_all.items()}, out / "oof_all.pkl")

    # ---- statistics
    st = {}
    vc_auc = bootstrap_auc_ci(y, J["vc_ratio"].values)
    st["univariate_vc_ratio_auc"] = vc_auc
    from scipy.stats import mannwhitneyu
    st["vc_ratio_mannwhitney_p"] = float(mannwhitneyu(J.vc_ratio[y == 1], J.vc_ratio[y == 0]).pvalue)
    pairs = [("JVP | LR physiology features", "ECG | LR V1/V2 features"),
             ("JVP | 1D-CNN waveform", "ECG | 1D-CNN V1/V2 beat"),
             ("JVP | Physio-hybrid CNN (waveform + v/c, a-c slope)", "ECG | LR V1/V2 features"),
             ("JVP | 1D-CNN waveform", "ECG | LR V1/V2 features"),
             ("JVP | L1-LR physiology features (embedded selection)", "ECG | LR V1/V2 features")]
    st["delong"] = {}
    for p1, p2 in pairs:
        a1, a2, p = delong_paired(y, P_all[p1].mean(0), P_all[p2].mean(0))
        st["delong"][f"{p1} vs {p2}"] = dict(auc1=a1, auc2=a2, p=p)
    kap = cohen_kappa(P_all["JVP | LR physiology features"].mean(0) >= 0.5,
                      P_all["ECG | LR V1/V2 features"].mean(0) >= 0.5)
    st["kappa_JVP_LR_vs_ECG_LR"] = kap
    if a.perms > 0:
        st["permutation"] = {}
        for name in ["JVP | LR physiology features", "ECG | LR V1/V2 features",
                     "JVP | LR log(v/c) only"]:
            obs = summ.set_index("model").loc[name, "auroc"]
            p, null = permutation_pvalue(zoo[name][1], S, y, obs, n_perm=(20 if a.quick else a.perms))
            st["permutation"][name] = dict(observed_auroc=obs, p=p, null_mean=float(null.mean()),
                                           null_95=float(np.percentile(null, 95)))
            print("perm", name, obs, p, flush=True)
    json.dump(st, open(out / "stats.json", "w"), indent=2, default=float)
    print(json.dumps(st, indent=2, default=float))


if __name__ == "__main__":
    main()
