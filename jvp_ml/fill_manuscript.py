"""Build the Overleaf-ready manuscript folder.

Fills the double-brace tokens in manuscript/manuscript_template.tex from results/ and writes
JVP_ML_manuscript.tex together with the bibliography, the naturemag.bst style and the figure PDFs.
Usage: python fill_manuscript.py [--out ../overleaf_JVP_manuscript]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from evaluation import bootstrap_auc_ci, cohen_kappa, delong_paired
from run_benchmark import load

HERE = Path(__file__).parent
RES = HERE / "results"
MS = HERE / "manuscript"
ALIAS = {
    "rule": "JVP | rule v/c>1 (no training)", "jvp_lrvc": "JVP | LR log(v/c) only",
    "jvp_lr": "JVP | LR physiology features", "jvp_svm": "JVP | SVM physiology features",
    "jvp_rf": "JVP | RF physiology features", "jvp_gbm": "JVP | GBM physiology features",
    "jvp_rocket": "JVP | ROCKET (random conv kernels)", "jvp_cnn": "JVP | 1D-CNN waveform",
    "jvp_hcnn": "JVP | Physio-hybrid CNN (waveform + v/c, a-c slope)",
    "ecg_lr": "ECG | LR V1/V2 features", "ecg_svm": "ECG | SVM V1/V2 features", "ecg_rf": "ECG | RF V1/V2 features",
    "ecg_gbm": "ECG | GBM V1/V2 features", "ecg_cnn": "ECG | 1D-CNN V1/V2 beat",
    "jvp_l1": "JVP | L1-LR physiology features (embedded selection)", "jvp_fourier": "JVP | LR Fourier descriptors (shift-invariant)",
}
DL = {"lr": "JVP | LR physiology features vs ECG | LR V1/V2 features",
      "cnn": "JVP | 1D-CNN waveform vs ECG | 1D-CNN V1/V2 beat"}


def fp(p):
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE.parent / "overleaf_JVP_manuscript"))
    OUTDIR = Path(ap.parse_args().out); OUTDIR.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(RES / "table_benchmark.csv").set_index("model")
    st = json.load(open(RES / "stats.json"))
    ag = json.load(open(RES / "agreement.json"))["agreement"]
    sens = pd.read_csv(RES / "sensitivity_analyses.csv")
    deg = pd.read_csv(RES / "degradation.csv")
    qc = pd.read_csv(HERE / "data/qc_issues.csv")
    cal = pd.read_csv(HERE / "data/force_current_calibration.csv")
    cyc = pd.read_csv(HERE / "data/jvp_cycles.csv", dtype={"cycle": str})
    sub, J, E, waves, _ = load()
    y = sub.y.values
    ecg_best = T[T.modality == "ECG"].auroc.idxmax()
    ALIAS["ecg_best"] = ecg_best
    spc = cyc.groupby(["subject", "cycle"]).size()
    vc = bootstrap_auc_ci(y, J.vc_ratio.values)
    rule = (J.vc_ratio > 1).values
    rvh = ((E["V1_R_mV"] >= 0.7) | (E["V1_RS_ratio"] >= 1)).values

    def auc_of(D, f):
        a = roc_auc_score(y, D[f]); return f"{max(a, 1 - a):.2f}"

    v = {
        "N": str(len(sub)), "NPOS": str(int(y.sum())), "NNEG": str(int((1 - y).sum())),
        "NCYC": str(len(spc)), "SPC": f"{spc.median():.0f}", "SPC_RANGE": f"{spc.min()}--{spc.max()}",
        "RIF": f"{cal.r_current_force.abs().median():.5f}", "PREV": f"{y.mean():.3f}",
        "VCAUC": f"{vc[0]:.3f}", "VCCI": f"{vc[1]:.2f}--{vc[2]:.2f}", "VCP": fp(st["vc_ratio_mannwhitney_p"]),
        "VC_MED_DIS": f"{J.vc_ratio[y == 1].median():.2f}", "VC_MED_CTL": f"{J.vc_ratio[y == 0].median():.2f}",
        "RULE_TP": str(int(rule[y == 1].sum())), "RULE_TN": str(int((~rule[y == 0]).sum())),
        "RULE_SENS": f"{100 * rule[y == 1].mean():.0f}\\%", "RULE_SPEC": f"{100 * (~rule[y == 0]).mean():.1f}\\%",
        "RVH_TP": str(int(rvh[y == 1].sum())), "RVH_SENS": f"{100 * rvh[y == 1].mean():.0f}\\%",
        "RVH_SPEC": f"{100 * (~rvh[y == 0]).mean():.0f}\\%",
        "AUC_LATE": "",
        "AUC_V1S": auc_of(E, "V1_S_mV"), "AUC_PR": auc_of(E, "V1_PR_ms"),
        "KAPPA": f"{st['kappa_JVP_LR_vs_ECG_LR']:.2f}",
        "NPERM": "200", "PMIN": f"{1 / 201:.3f}",
    }
    k = st["kappa_JVP_LR_vs_ECG_LR"]
    v["KAPPATXT"] = "slight" if k < 0.2 else "fair" if k < 0.4 else "moderate" if k < 0.6 else "substantial"
    for a, name in ALIAS.items():
        r = T.loc[name]
        v[f"A:{a}"] = f"{r.auroc:.3f}"
        v[f"ACI:{a}"] = f"{r.auroc:.3f} (95\\% CI {r.auroc_lo:.2f}--{r.auroc_hi:.2f})"
        v[f"SE:{a}"] = f"{100 * r.sens:.0f}\\%"; v[f"SP:{a}"] = f"{100 * r.spec:.0f}\\%"
        v[f"PR:{a}"] = f"{r.auprc:.3f}"
    for a, key in DL.items():
        d = st["delong"][key]
        v[f"DLTXT:{a}"] = f"AUROC {d['auc1']:.2f} vs {d['auc2']:.2f} on repeat-averaged scores, $P={fp(d['p'])}$"
        v[f"DLP:{a}"] = f"$P={fp(d['p'])}$"
    P_all = pd.read_pickle(RES / "oof_all.pkl")
    pc, pe = P_all[ALIAS["jvp_cnn"]].mean(0), P_all[ALIAS["ecg_lr"]].mean(0)
    a1, a2, pdl = delong_paired(y, pc, pe)
    v["DLTXT:cnn_ecglr"] = f"AUROC {a1:.2f} vs {a2:.2f} on repeat-averaged scores, $P={fp(pdl)}$"
    v["DLP:cnn_ecglr"] = f"$P={fp(pdl)}$"
    kc = cohen_kappa(pc >= 0.5, pe >= 0.5); v["KAPPACNN"] = f"{kc:.2f}"
    v["KAPPACNNTXT"] = "slightly" if kc < 0.2 else "fairly" if kc < 0.4 else "moderately" if kc < 0.6 else "substantially"
    cp = json.load(open(RES / "cnn_permutation.json"))
    lp = json.load(open(RES / "l1lr_permutation.json"))
    v["L1PERM_P"] = f"{lp['p']:.3f}"; v["L1PERM_N"] = str(lp["n_perm"])
    v["CNNPERM_P"] = f"{cp['p']:.3f}"; v["CNNPERM_N"] = str(cp["n_perm"]); v["CNNPERM_OBS"] = f"{cp['observed_auroc_reduced']:.3f}"
    al = json.load(open(RES / "agreement.json"))["alignment_test"]
    for k1, k2 in (("stored", "stored windows"), ("shift", "random circular shift")):
        v[f"AL:{k1}|raw"] = f"{al[k2]['raw_window_LR'][0]:.3f}"; v[f"AL:{k1}|fourier"] = f"{al[k2]['shift_invariant_fourier_LR'][0]:.3f}"
    v["TROUGHPOS"] = f"{al['trough_position_median']:.2f}"; v["TROUGHIQR"] = "--".join(f"{q:.2f}" for q in al["trough_position_iqr"])
    v["TROUGHDIS"] = f"{al['trough_position_median_by_class']['diseased']:.2f}"; v["TROUGHCTL"] = f"{al['trough_position_median_by_class']['control']:.2f}"
    for tok, feat in (("SLOPE", "slope_a_to_c"), ("REL3", "relphase3_sin"), ("YD", "y_descent")):
        aa = roc_auc_score(y, J[feat]); v[f"AUC_{tok}"] = f"{max(aa, 1 - aa):.2f}"
        v[f"P_{tok}"] = fp(mannwhitneyu(J[feat][y == 1], J[feat][y == 0]).pvalue)
    alias_perm = {"jvp_lr": "JVP | LR physiology features", "ecg_lr": "ECG | LR V1/V2 features",
                  "jvp_lrvc": "JVP | LR log(v/c) only"}
    for a, name in alias_perm.items():
        v[f"PERM:{a}"] = f"{st['permutation'][name]['p']:.3f}"
    for _, r in sens.iterrows():
        v[f"SENS:{r.analysis}|{r.model}"] = f"{r.auroc:.3f}"
    for _, r in deg.iterrows():
        v[f"DEG:{r.condition}|{r.model}"] = f"{r.auroc:.2f}"
    C = pd.read_csv(RES / "jvp_ecg_spearman_r.csv", index_col=0)
    for a_ in C.index:
        for b_ in C.columns:
            v[f"RHO:{a_}|{b_}"] = f"{C.loc[a_, b_]:.2f}"
    ei = json.load(open(RES / "ecg_investigation.json"))
    v.update({"EI:agree": str(ei["crude_vc_agree"]), "EI:r2": f"{ei['six_gaussian_r2_median']:.4f}",
              "EI:r2n": str(ei["n_leads_r2_gt_0999"]), "EI:nleads": str(ei["n_leads"]),
              "EI:flat": f"{100 * ei['flat_fraction_median']:.0f}", "EI:pr120": str(ei["n_pr_lt_120"]),
              "EI:pr200": str(ei["n_pr_gt_200"]), "EI:sens": f"{100 * ei['crude_vc_sens']:.0f}\\%",
              "EI:spec": f"{100 * ei['crude_vc_spec']:.1f}\\%", "EI:auc": f"{ei['crude_vc_auroc']:.3f}",
              "EI:dis": ", ".join(ei["crude_vc_disagreements"])})
    IM = pd.read_csv(RES / "imbalance_strategies.csv")
    OPT = pd.read_csv(RES / "imbalance_operating_points.csv").set_index("model")
    PVT = pd.read_csv(RES / "imbalance_ppv_npv.csv")
    ij = json.load(open(RES / "imbalance.json"))
    f2 = lambda x: "--" if pd.isna(x) else f"{x:.2f}"
    for _, r in IM.iterrows():
        for m in ("auroc", "auprc", "sens@0.5", "spec@0.5", "sens@youden", "spec@youden", "mcc@youden", "mcc@0.5", "citl"):
            v[f"IMB:{r.features}|{r.model}|{r.strategy}|{m}"] = (f"{r[m]:+.3f}" if m == "citl" else
                                                                 f"{r[m]:.3f}" if m in ("auroc", "auprc") else f2(r[m]))
    for (fs, mdl), g in IM.groupby(["features", "model"]):
        for m in ("auroc", "auprc"):
            v[f"IMBMIN:{fs}|{mdl}|{m}"] = f"{g[m].min():.3f}"; v[f"IMBMAX:{fs}|{mdl}|{m}"] = f"{g[m].max():.3f}"
    corr = IM[(IM.model == "LR") & (IM.strategy != "none")].citl
    v["IMBCITLMIN"] = f"{corr.min():.2f}"; v["IMBCITLMAX"] = f"{corr.max():.2f}"
    _s05 = IM[(IM.features == "JVP features") & (IM.model == "LR") & (IM.strategy != "none")]["sens@0.5"]
    v["IMBS05MIN"] = f2(_s05.min()); v["IMBS05MAX"] = f2(_s05.max())
    for mdl, r in OPT.iterrows():
        for m in OPT.columns:
            v[f"OP:{mdl}|{m}"] = f"{r[m]:.2f}"
    for _, r in PVT.iterrows():
        v[f"PV:{r.model}|{r.prevalence:g}|ppv"] = f"{r.ppv:.2f}"; v[f"PV:{r.model}|{r.prevalence:g}|npv"] = f"{r.npv:.3f}"
    for k, d in ij["sample_size_auc_halfwidth_0p05_prev0p115"].items():
        for kk, vv in d.items():
            v[f"SS:{k}|{kk}"] = str(vv)
    v["SSHW"] = f"{ij['current_halfwidth_auc0p90']:.2f}"; v["SSSENS"] = str(ij["n_pos_for_sens0p9_halfwidth0p1"])
    v["PREVPCT"] = f"{100 * y.mean():.1f}"; v["ACCALLNEG"] = f"{100 * (1 - y.mean()):.1f}"
    rows_t = []
    for _, r in IM.iterrows():
        rows_t.append(" & ".join([r.features.replace("JVP waveform", "JVP waveform"), r.model, r.strategy.replace("undersampling ensemble", "undersampling"),
                                  f"{r.auroc:.3f}", f"{r.auprc:.3f}", f2(r["sens@0.5"]), f2(r["spec@0.5"]), f2(r.get("sens@youden")),
                                  f2(r.get("spec@youden")), f2(r.get("mcc@youden")), f"{r.citl:+.2f}"]) + " \\\\")
    v["IMBTABLE"] = "\n".join(rows_t)
    v["BENCHTABLE"] = open(RES / "table_benchmark.tex").read().strip().replace("Physio-hybrid CNN (waveform + v/c, a-c slope)", "Physiology-hybrid CNN").replace("ROCKET (random conv kernels)", "ROCKET").replace("L1-LR physiology features (embedded selection)", "L1-LR, all features").replace("LR Fourier descriptors (shift-invariant)", "LR, Fourier descriptors").replace("1D-CNN waveform", "1D-CNN (primary)")
    handling = {
        "ecg-duplicate-of-other-subject": "Kept in primary analysis; excluded in sensitivity analysis and agreement analysis",
        "header-subject-mismatch": "Kept; excluded in sensitivity analysis",
        "ecg-time-repaired": "Time axis rescaled to seconds",
        "header-lead-mismatch": "Lead identity taken from file name",
        "duplicate-cycle-within-subject": "Counted once",
        "cycle-content-differs": "Both versions retained as separate cycles",
        "jvp-column-length-mismatch": "Truncated to common length",
    }
    rows = []
    for iss, g in qc.groupby("issue"):
        who = ", ".join(sorted(set(g.subject), key=lambda s: int(s[2:])))
        rows.append(f"{iss.replace('-', ' ')} & {who} ({len(g)} files) & {handling.get(iss, '')} \\\\")
    v["QCTABLE"] = "\n".join(rows)

    tpl = open(MS / "manuscript_template.tex").read()
    missing = []

    def sub_tok(m):
        key = m.group(1)
        if key in v:
            return v[key]
        missing.append(key); return m.group(0)
    out = re.sub(r"\{\{([^{}]+)\}\}", sub_tok, tpl)
    if missing:
        raise SystemExit(f"Unfilled tokens: {sorted(set(missing))}")
    (OUTDIR / "figures_jvp").mkdir(exist_ok=True)
    for f in ("jvp_references.bib", "naturemag.bst"):
        shutil.copy(MS / f, OUTDIR / f)
    for f in (HERE / "figures").glob("*.pdf"):
        shutil.copy(f, OUTDIR / "figures_jvp" / f.name)
    open(OUTDIR / "JVP_ML_manuscript.tex", "w").write(out)
    print("wrote", OUTDIR / "JVP_ML_manuscript.tex")


if __name__ == "__main__":
    main()
