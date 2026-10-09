"""Publication figures and tables from saved results (run after run_benchmark.py and robustness.py).

Outputs: figures/*.pdf|png, results/table_benchmark.csv, results/table_benchmark.tex
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from features import PHASE, jvp_cycle_features, jvp_fiducials  # noqa: E402
from models import CNNModel, TabularModel  # noqa: E402
from run_benchmark import HYBRID_FEATS, JVP_FEATS, load  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
RES = HERE / "results"
C_CTL, C_DIS = "#2a78d6", "#eb6834"
C_MOD = {"JVP": "#4a3aa7", "ECG": "#008300"}  # sensor (deployed) and ECG (comparator only)
C_ACC = "#eda100"  # accent for evaluation boxes and secondary series
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
    "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "lines.linewidth": 1.4, "savefig.dpi": 300, "pdf.fonttype": 42, "axes.titleweight": "bold",
    "axes.titlelocation": "left"})
MM = 1 / 25.4


def panel(ax, letter):
    ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", color=INK)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ tables with coherent CIs
def _auc_rows(y, P):
    """Rank-based AUROC for every row of P (ties get average ranks), identical to roc_auc_score."""
    from scipy.stats import rankdata
    R = rankdata(P, axis=1); n1 = y.sum(); n0 = len(y) - n1
    return (R[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def _ap(y, s):
    """Average precision, same step-wise definition as sklearn.average_precision_score."""
    o = np.argsort(-s, kind="mergesort"); ys, ss = y[o], s[o]
    idx = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tps = np.cumsum(ys)[idx]; fps = idx + 1 - tps
    prec, rec = tps / (tps + fps), tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def coherent_summary(y, P, n_boot=2000, seed=0):
    """Point estimate = mean over CV repeats; CI = stratified subject bootstrap of that same mean."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    pt = _auc_rows(y, P).mean(); pr = np.mean([_ap(y, p) for p in P])
    assert abs(pt - np.mean([roc_auc_score(y, p) for p in P])) < 1e-9
    assert abs(pr - np.mean([average_precision_score(y, p) for p in P])) < 1e-9
    bs_a, bs_p = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        bs_a.append(_auc_rows(y[idx], P[:, idx]).mean())
        bs_p.append(np.mean([_ap(y[idx], p[idx]) for p in P]))
    return pt, *np.percentile(bs_a, [2.5, 97.5]), pr, *np.percentile(bs_p, [2.5, 97.5])


def tables(sub):
    y = sub.y.values
    P_all = pd.read_pickle(RES / "oof_all.pkl")
    summ = pd.read_csv(RES / "benchmark_summary.csv").set_index("model")
    rows = []
    for name, P in P_all.items():
        a, alo, ahi, pr, plo, phi = coherent_summary(y, P)
        s = summ.loc[name]
        from sklearn.metrics import matthews_corrcoef
        mcc = np.mean([matthews_corrcoef(y, (p >= 0.5).astype(int)) for p in P])  # same 0.5 rule as sens/spec
        rows.append(dict(model=name, modality=s.modality, repeats=int(s.repeats), auroc=a, auroc_lo=alo, auroc_hi=ahi,
                         auprc=pr, auprc_lo=plo, auprc_hi=phi, sens=s.sens, spec=s.spec, bal_acc=s.bal_acc,
                         mcc=mcc, ppv=s.ppv, npv=s.npv, brier=s.brier))
    T = pd.DataFrame(rows); T.to_csv(RES / "table_benchmark.csv", index=False)
    with open(RES / "table_benchmark.tex", "w") as fh:
        for _, r in T.iterrows():
            m = r.model.split("|")[1].strip().replace("&", "\\&").replace("v/c>1", "$v/c>1$")
            fh.write(f"{r.modality} & {m} & {r.auroc:.3f} ({r.auroc_lo:.2f}--{r.auroc_hi:.2f}) & "
                     f"{r.auprc:.3f} ({r.auprc_lo:.2f}--{r.auprc_hi:.2f}) & {r.sens:.2f} & {r.spec:.2f} & "
                     f"{r.bal_acc:.2f} & {r.mcc:.2f} & {r.brier:.3f} \\\\\n")
    return T


# ------------------------------------------------------------------ Fig 1: pipeline schematic
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(180 * MM, 50 * MM)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    W, H, X = 17.6, 11.5, [0.5, 20.6, 40.7, 60.8, 81.2]
    top, bot, mid = 17, 1.5, 9.25
    boxes = [
        (X[0], top, "rGO/CuO piezoresistive\nsensor over the EJV\n(current, force)", C_MOD["JVP"]),
        (X[0], bot, "Reference ECG\nleads V1 and V2\n(one beat per subject)", C_MOD["ECG"]),
        (X[1], top, "Sensor windows: PCHIP\nto 64 points, min-max,\nanchored at x trough", C_MOD["JVP"]),
        (X[1], bot, "Beat resampling;\nR, S, R/S, P, T,\nPR, QRS, QT", C_MOD["ECG"]),
        (X[2], top, "Shift-invariant features:\nv/c, a$\\to$c slope, descents,\nFourier descriptors", C_MOD["JVP"]),
        (X[3], top, "Rule, LR, SVM, RF, GBM,\ncircular ROCKET,\ncircular 1D-CNN, hybrid", C_MOD["JVP"]),
        (X[3], bot, "Comparator only:\nLR, SVM, RF, GBM,\n1D-CNN", C_MOD["ECG"]),
        (X[4], mid, "Subject-level repeated\nstratified CV; bootstrap,\npermutation, DeLong", C_ACC),
    ]
    for x, y0, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y0), W, H, boxstyle="round,pad=0.25,rounding_size=1.0", fc="white", ec=c, lw=1.1))
        ax.text(x + W / 2, y0 + H / 2, t, ha="center", va="center", fontsize=5.8, color=INK, linespacing=1.25)
    arr = dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0)
    yt, yb = top + H / 2, bot + H / 2
    segs = [(X[0] + W + .4, yt, X[1] - .4, yt), (X[0] + W + .4, yb, X[1] - .4, yb), (X[1] + W + .4, yt, X[2] - .4, yt),
            (X[2] + W + .4, yt, X[3] - .4, yt), (X[1] + W + .4, yb, X[3] - .4, yb),
            (X[3] + W + .4, yt, X[4] - .4, mid + H * 0.75), (X[3] + W + .4, yb, X[4] - .4, mid + H * 0.25)]
    for x1, y1, x2, y2 in segs:
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=arr)
    save(fig, "fig1_pipeline")


# ------------------------------------------------------------------ Fig 2: waveforms
def fig_waveforms(sub, waves, ecg_df):
    lab = dict(zip(sub.subject, sub.y))
    fig, axs = plt.subplots(1, 4, figsize=(180 * MM, 48 * MM), gridspec_kw=dict(wspace=0.45))
    cyc = pd.read_csv(HERE / "data/jvp_cycles.csv", dtype={"cycle": str})
    for ax, s, col, ttl, L in [(axs[0], "HS13", C_CTL, "Control HS13", "a"), (axs[1], "HS5", C_DIS, "Diseased HS5", "b")]:
        for k, (c, g) in enumerate(cyc[cyc.subject == s].groupby("cycle", sort=False)):
            f, r, fid = jvp_cycle_features(g.force_N.values)
            ax.plot(PHASE, r, color=col, alpha=1 if k == 0 else 0.35, lw=1.4 if k == 0 else 1)
            if k == 0:
                ax.annotate("x", (0, 0), xytext=(4, -9), textcoords="offset points", fontsize=7, fontweight="bold", color=INK)
                for nm, i in zip("vyac", fid):
                    if nm == "a" and fid[2] == fid[3]:
                        continue
                    ax.annotate(nm, (PHASE[i], r[i]), xytext=(0, 5 if nm in "vac" else -9), textcoords="offset points",
                                ha="center", fontsize=7, fontweight="bold", color=INK)
                vc0 = f["vc_ratio"]
        ax.set_title(f"{ttl}, v/c = {vc0:.2f}"); ax.set_xlabel("Phase from x trough"); ax.set_ylabel("Normalised JVP (a.u.)"); ax.set_ylim(-0.15, 1.18)
        panel(ax, L)
    ax = axs[2]
    for yy, col, nm in [(0, C_CTL, "Control"), (1, C_DIS, "Diseased")]:
        W = np.array([w[2] for w in waves if lab[w[0]] == yy]); m, sd = W.mean(0), W.std(0)
        ax.fill_between(PHASE, m - sd, m + sd, color=col, alpha=0.15, lw=0); ax.plot(PHASE, m, color=col, label=f"{nm} (n={int((sub.y == yy).sum())})")
    ax.set_title("Class mean ± s.d."); ax.set_xlabel("Phase from x trough"); ax.set_ylabel("Normalised JVP (a.u.)")
    ax.legend(frameon=False, loc="lower left"); panel(ax, "c")
    ax = axs[3]
    for yy, col, nm in [(0, C_CTL, "Control"), (1, C_DIS, "Diseased")]:
        B = []
        for s in sub.subject[sub.y == yy]:
            g = ecg_df[(ecg_df.subject == s) & (ecg_df.lead == "V1")]
            tt = np.linspace(0, 0.6, 200); B.append(np.interp(tt, g.t - g.t.min(), g.mV))
        B = np.array(B); ax.fill_between(tt * 1000, B.mean(0) - B.std(0), B.mean(0) + B.std(0), color=col, alpha=0.15, lw=0)
        ax.plot(tt * 1000, B.mean(0), color=col, label=nm)
    ax.set_title("ECG lead V1"); ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude (mV)"); panel(ax, "d")
    save(fig, "fig2_waveforms")


# ------------------------------------------------------------------ Fig 3: hypothesis features
def fig_features(sub, J, E):
    from scipy.stats import mannwhitneyu
    feats = [("vc_ratio", J, "JVP v/c ratio", 1.0), ("slope_a_to_c", J, "JVP a→c slope (a.u./phase)", None),
             ("relphase3_sin", J, "JVP 3rd-harmonic phase (sin)", None), ("y_descent", J, "JVP y descent (a.u.)", None),
             ("V1_S_mV", E, "ECG S wave, V1 (mV)", None)]
    fig, axs = plt.subplots(1, 5, figsize=(180 * MM, 46 * MM), gridspec_kw=dict(wspace=0.6))
    rng = np.random.default_rng(1)
    for i, (ax, (f, D, lab, thr)) in enumerate(zip(axs, feats)):
        for yy, col in [(0, C_CTL), (1, C_DIS)]:
            v = D[f].values[sub.y.values == yy]
            ax.scatter(yy + rng.uniform(-0.17, 0.17, len(v)), v, s=11, color=col, edgecolor="white", lw=0.4, zorder=3)
            ax.hlines(np.median(v), yy - 0.28, yy + 0.28, color=INK, lw=1.2, zorder=4)
        yv = sub.y.values; p = mannwhitneyu(D[f].values[yv == 1], D[f].values[yv == 0]).pvalue
        auc = roc_auc_score(sub.y, D[f]); auc = max(auc, 1 - auc)
        ax.set_title(f"AUC {auc:.2f}\np = {p:.1e}" if p < 1e-3 else f"AUC {auc:.2f}\np = {p:.3f}", fontweight="normal", loc="center")
        if thr is not None:
            ax.axhline(thr, color=INK2, ls="--", lw=0.7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "Diseased"]); ax.set_xlim(-0.5, 1.5); ax.set_ylabel(lab)
        panel(ax, "abcde"[i])
    save(fig, "fig3_features")


# ------------------------------------------------------------------ Fig 4: benchmark
def fig_benchmark(sub, T):
    y = sub.y.values
    P_all = pd.read_pickle(RES / "oof_all.pkl")
    fig = plt.figure(figsize=(180 * MM, 82 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    T2 = T.copy(); T2["order"] = T2.modality.map({"JVP": 0, "ECG": 1})
    T2 = T2.sort_values(["order", "auroc"], ascending=[True, True])
    ypos = np.arange(len(T2))
    for yi, (_, r) in zip(ypos, T2.iterrows()):
        c = C_MOD[r.modality]
        ax.plot([r.auroc_lo, r.auroc_hi], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(r.auroc, yi, "o", color=c, ms=4.5, mec="white", mew=0.6)
        ax.text(1.065, yi, f"{r.auroc:.2f}", va="center", fontsize=6, color=INK)
    ax.set_yticks(ypos); ax.set_yticklabels([m.split("|")[1].strip() for m in T2.model], fontsize=6)
    for lbl, (_, r) in zip(ax.get_yticklabels(), T2.iterrows()):
        lbl.set_color(INK)
    ax.axvline(0.5, color=INK2, ls=":", lw=0.7); ax.set_xlim(0.3, 1.05); ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=C_MOD[k], marker="o", lw=1.6) for k in C_MOD], ["JVP sensor (deployed)", "ECG V1/V2 (comparator)"],
              frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=6)
    ax.set_title("Subject-level cross-validated AUROC", pad=14)
    ax.text(-0.85, 1.06, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
    sel = [("ECG | LR V1/V2 features", C_MOD["ECG"], "-"),
           ("JVP | rule v/c>1 (no training)", C_MOD["JVP"], ":"), ("JVP | L1-LR physiology features (embedded selection)", C_MOD["JVP"], "--"),
           ("JVP | 1D-CNN waveform", C_MOD["JVP"], "-")]
    names = {"JVP | rule v/c>1 (no training)": "JVP rule v/c>1", "JVP | L1-LR physiology features (embedded selection)": "JVP L1-LR",
             "JVP | 1D-CNN waveform": "JVP 1D-CNN (primary)", "ECG | LR V1/V2 features": "ECG LR"}
    a2, a3 = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for nm, c, ls in sel:
        p = P_all[nm].mean(0); fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        lw = 2.2 if nm == "JVP | 1D-CNN waveform" else 1.2
        a2.plot(fpr, tpr, color=c, ls=ls, lw=lw, label=f"{names[nm]} ({roc_auc_score(y, p):.2f})")
        a3.plot(rc, pr, color=c, ls=ls, lw=lw, label=f"{names[nm]} ({average_precision_score(y, p):.2f})")
    a2.plot([0, 1], [0, 1], color=GRID, lw=0.8); a3.axhline(y.mean(), color=GRID, lw=0.8)
    a2.set_xlabel("1 − specificity"); a2.set_ylabel("Sensitivity"); a2.set_title("ROC (repeat-averaged scores)")
    a3.set_xlabel("Recall (sensitivity)"); a3.set_ylabel("Precision (PPV)"); a3.set_title("Precision–recall")
    for a_ in (a2, a3):
        h, l = a_.get_legend_handles_labels()
        a_.legend(h[::-1], l[::-1], frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.27))
    for a_, L in [(a2, "b"), (a3, "c")]:
        a_.set_xlim(-0.02, 1.02); a_.set_ylim(-0.02, 1.04); a_.set_aspect("equal"); panel(a_, L)
    save(fig, "fig4_benchmark")


# ------------------------------------------------------------------ Fig 5: interpretability + robustness
def jvp_cycle_features_from_wave(r):
    iv, iy, ia, ic, _ = jvp_fiducials(r)
    return np.array([iv, iy, ia, ic]) / len(r)


def fig_interpret(sub, J, E, waves):
    S, y = sub.subject.values, sub.y.values
    fig, axs = plt.subplots(1, 4, figsize=(180 * MM, 50 * MM), gridspec_kw=dict(wspace=0.55, width_ratios=[1.2, 1, 1, 1.1]))
    # a: standardised LR coefficients fitted on all subjects
    m = TabularModel("L1LR", J[JVP_FEATS], 0).fit(S, y)
    lr = m.est.best_estimator_[-1]; coef = pd.Series(lr.coef_[0], index=JVP_FEATS)
    top = coef.reindex(coef.abs().sort_values().index[-10:])
    axs[0].barh(range(len(top)), top.values, color=[C_DIS if v > 0 else C_CTL for v in top.values], height=0.65)
    axs[0].set_yticks(range(len(top))); axs[0].set_yticklabels(top.index, fontsize=5.8); axs[0].axvline(0, color=INK2, lw=0.6)
    axs[0].set_xlabel("Standardised L1-LR coefficient"); axs[0].set_title("Sensor feature weights"); panel(axs[0], "a")
    # b: saliency of the hybrid CNN by phase
    cnn = CNNModel(waves, epochs=120, seeds=(0, 1, 2)).fit(S, y)  # primary sensor model
    lab = dict(zip(S, y)); idx = np.arange(len(waves)); sal = cnn.saliency(idx)
    cy = np.array([lab[w[0]] for w in waves])
    for yy, col, nm in [(0, C_CTL, "Control"), (1, C_DIS, "Diseased")]:
        axs[1].plot(PHASE, sal[cy == yy].mean(0), color=col, label=nm)
    axs[1].axhline(0, color=INK2, lw=0.5)
    fids = np.array([jvp_cycle_features_from_wave(w[2]) for w in waves])  # median fiducial phases
    for nm, ph in [("x", 0.0), ("v", np.median(fids[:, 0])), ("a", np.median(fids[:, 2])), ("c", np.median(fids[:, 3]))]:
        axs[1].axvline(ph, color=GRID, lw=0.6); axs[1].text(ph, axs[1].get_ylim()[1], nm, ha="center", va="bottom", fontsize=6.5, color=INK2)
    axs[1].set_xlabel("Phase from x trough"); axs[1].set_ylabel("Gradient × input"); axs[1].set_title("1D-CNN saliency", pad=10)
    axs[1].legend(frameon=False, fontsize=6); panel(axs[1], "b")
    # c: permutation null
    st = json.load(open(RES / "stats.json"))
    perm = dict(st.get("permutation", {}))
    if (RES / "cnn_permutation.json").exists():
        cp = json.load(open(RES / "cnn_permutation.json"))
        perm["JVP 1D-CNN"] = dict(observed_auroc=cp["observed_auroc_reduced"], p=cp["p"], null_95=cp["null_95"])
    if (RES / "l1lr_permutation.json").exists():
        lp = json.load(open(RES / "l1lr_permutation.json"))
        perm["JVP L1-LR"] = dict(observed_auroc=lp["observed_auroc"], p=lp["p"], null_95=lp["null_95"])
    for k, col, nm, ls in [("JVP 1D-CNN", C_MOD["JVP"], "JVP 1D-CNN*", "-"), ("JVP L1-LR", C_MOD["JVP"], "JVP L1-LR", "--"),
                           ("ECG | LR V1/V2 features", C_MOD["ECG"], "ECG LR", "-")]:
        if k in perm:
            axs[2].axvline(perm[k]["observed_auroc"], color=col, lw=1.4, ls=ls, label=f"{nm}: p={perm[k]['p']:.3f}")
            axs[2].axvspan(0.5, perm[k]["null_95"], color=col, alpha=0.08)
    axs[2].set_xlim(0.3, 1.02); axs[2].set_xlabel("AUROC"); axs[2].set_yticks([])
    axs[2].set_title("Permutation test\n(shaded: null 50th–95th pct)", fontsize=6.5); axs[2].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    panel(axs[2], "c")
    # d: degradation
    deg = pd.read_csv(RES / "degradation.csv")
    conds = list(dict.fromkeys(deg.condition))
    for mdl, col, mk, nm in [("CNN", C_MOD["JVP"], "o", "1D-CNN"), ("LR", C_ACC, "s", "L1-LR"), ("rule", C_MOD["ECG"], "^", "rule v/c>1")]:  # noqa: E501
        d = deg[deg.model == mdl].set_index("condition").loc[conds]
        axs[3].errorbar(range(len(conds)), d.auroc, yerr=[np.clip(d.auroc - d.lo, 0, None), np.clip(d.hi - d.auroc, 0, None)], color=col, marker=mk, ms=3.5,
                        lw=1.1, capsize=1.5, label=nm)
    axs[3].set_xticks(range(len(conds))); axs[3].set_xticklabels([c.replace("SNR ", "").replace(" samples", "") for c in conds],
                                                                  rotation=60, fontsize=5.8)
    axs[3].set_xlabel("Test-time noise (dB SNR) / decimation"); axs[3].set_ylabel("AUROC"); axs[3].set_ylim(0.5, 1.02)
    axs[3].set_title("Robustness (trained on clean)"); axs[3].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3); panel(axs[3], "d")
    save(fig, "fig5_interpretability_robustness")


# ------------------------------------------------------------------ Fig 6: agreement with ECG
def fig_agreement():
    al = json.load(open(RES / "agreement.json"))["alignment_test"]
    C = pd.read_csv(RES / "jvp_ecg_spearman_r.csv", index_col=0); Pv = pd.read_csv(RES / "jvp_ecg_spearman_p.csv", index_col=0)
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 56 * MM), gridspec_kw=dict(wspace=0.55, width_ratios=[1.1, 0.05, 1.6]))
    ax = axs[0]; axs[1].set_axis_off()
    conds = ["stored windows", "random circular shift"]
    for j, (key, col, nm) in enumerate([("raw_window_LR", C_MOD["ECG"], "raw window (alignment-dependent)"),
                                        ("shift_invariant_fourier_LR", C_MOD["JVP"], "shift-invariant descriptors")]):
        vals = [al[c][key] for c in conds]
        x = np.arange(2) + (j - 0.5) * 0.36
        ax.bar(x, [v[0] for v in vals], 0.34, color=col, label=nm)
        ax.errorbar(x, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt="none", ecolor=INK2, lw=0.8, capsize=2)
    ax.axhline(0.5, color=INK2, ls=":", lw=0.7); ax.set_xticks([0, 1]); ax.set_xticklabels(["stored\nwindows", "random circular\nshift"])
    ax.set_ylim(0.3, 1.02); ax.set_ylabel("AUROC (LR)"); ax.set_title("Dependence on window alignment", fontweight="normal")
    ax.legend(frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.25)); panel(ax, "a")
    ax = axs[2]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [C_CTL, "#f0efec", C_DIS])
    im = ax.imshow(C.values.astype(float), cmap=cmap, vmin=-0.7, vmax=0.7, aspect="auto")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if Pv.values[i, j] < 0.05:
                ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=5.2, color=INK)
    ax.set_xticks(range(C.shape[1])); ax.set_xticklabels(C.columns, rotation=55, ha="right", fontsize=5.8)
    ax.set_yticks(range(C.shape[0])); ax.set_yticklabels(C.index, fontsize=5.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Spearman ρ", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    ax.set_title("JVP–ECG feature coupling (values: p<0.05)", fontweight="normal"); panel(ax, "b")
    save(fig, "fig6_ecg_agreement")


def fig_sensitivity():
    S = pd.read_csv(RES / "sensitivity_analyses.csv")
    an = list(dict.fromkeys(S.analysis)); models = list(dict.fromkeys(S.model))
    fig, ax = plt.subplots(figsize=(120 * MM, 55 * MM))
    w = 0.8 / len(an); cols = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
    for k, a in enumerate(an):
        d = S[S.analysis == a].set_index("model").loc[models]
        ax.bar(np.arange(len(models)) + (k - (len(an) - 1) / 2) * w, d.auroc, width=w * 0.92, color=cols[k], label=a)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=6); ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUROC"); ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "figS1_sensitivity")


def main():
    sub, J, E, waves, _ = load()
    ecg_df = pd.read_csv(HERE / "data/ecg_beats.csv")
    T = tables(sub)
    print(T[["model", "auroc", "auroc_lo", "auroc_hi", "auprc", "sens", "spec"]].round(3).to_string())
    fig_pipeline(); fig_waveforms(sub, waves, ecg_df); fig_features(sub, J, E); fig_benchmark(sub, T)
    fig_agreement(); fig_sensitivity(); fig_interpret(sub, J, E, waves)


if __name__ == "__main__":
    main()
