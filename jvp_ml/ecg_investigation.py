"""How like-for-like is the 12-lead ECG benchmark? Four checks.

1. Fidelity of the ECG beats: are they recorded signals or parametric reconstructions?
   (fit of a six-Gaussian P-Q-R-S-S'-T model to every lead; fraction of exactly flat samples)
2. Internal consistency of the limb leads: simultaneously recorded limb leads obey Einthoven's law
   (III = II - I) and Goldberger's relations (aVR = -(I + II)/2, aVL = I - II/2, aVF = II - I/2).
3. Conduction information: PR intervals (lead II) against the physiological range (120-200 ms) and
   the first-degree AV-block threshold (>200 ms).
5. Between-participant similarity of the 12-lead beats (Pearson r of the concatenated, resampled 12-lead beat) within
   first-batch controls, first-batch diseased and later-batch diseased participants.
4. A crude, start-agnostic v/c > 1 rule on the raw sensor samples (no interpolation, no fiducial
   detection): how well does it agree with the physicians' verified diagnoses?
Outputs: results/ecg_investigation.json, figures/figS2_ecg_investigation.{pdf,png}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score

from features import ecg_subject_features
from make_figures import C_CTL, C_DIS, GRID, INK, INK2, MM, panel, plt, save

HERE = Path(__file__).parent
RES = HERE / "results"


def gauss_sum(t, *p):
    out = np.zeros_like(t)
    for i in range(0, len(p), 3):
        out += p[i] * np.exp(-0.5 * ((t - p[i + 1]) / p[i + 2]) ** 2)
    return out


def fit_six_gaussians(t, v):
    iR, iS = int(np.argmax(v)), int(np.argmin(v))
    p0 = [0.1 * v.max(), t[iR] - 0.08, 0.02, v.max(), t[iR], 0.01, v.min(), t[iS], 0.012,
          0.2 * max(v.max(), 0.1), t[iR] + 0.22, 0.04, -0.05, t[iR] - 0.015, 0.008, 0.05, t[iS] + 0.03, 0.01]
    try:
        p, _ = curve_fit(gauss_sum, t, v, p0=p0, maxfev=20000)
        fit = gauss_sum(t, *p)
        return 1 - np.var(v - fit) / np.var(v), fit
    except RuntimeError:
        return np.nan, None


def crude_vc(cyc):
    """Start-agnostic crude v/c on the raw samples: rotate each window so its deepest sample (x) comes first,
    then v = highest sample in the first half after x and c = highest sample in the second half."""
    rows = []
    for (s, c), d in cyc.groupby(["subject", "cycle"], sort=False):
        F = d.force_N.values; F = (F - F.min()) / (np.ptp(F) + 1e-12)
        F = np.roll(F, -int(np.argmin(F))); n = len(F)
        rows.append((s, F[1: n // 2 + 1].max() / max(F[n // 2:].max(), 1e-3)))
    return pd.DataFrame(rows, columns=["subject", "vc"]).groupby("subject", sort=False).vc.median()


def main():
    ecg = pd.read_csv(HERE / "data/ecg_beats.csv"); cyc = pd.read_csv(HERE / "data/jvp_cycles.csv", dtype={"cycle": str})
    sub = pd.read_csv(HERE / "data/subjects.csv"); lab = dict(zip(sub.subject, sub.y))
    E = ecg_subject_features(ecg).set_index("subject").loc[sub.subject]
    y = sub.y.values
    out = {}
    # 1. fidelity
    r2, flat, example = [], [], None
    for (s, l), d in ecg.groupby(["subject", "lead"], sort=False):
        t = d.t.values - d.t.values[0]; v = d.mV.values
        r, fit = fit_six_gaussians(t, v); r2.append(r); flat.append(np.mean(np.abs(v) < 1e-4))
        if s == "HS16" and l == "II":
            example = (t, v, fit, r)
    r2 = np.array(r2)
    out["six_gaussian_r2_median"] = float(np.nanmedian(r2))
    out["six_gaussian_r2_p10"] = float(np.nanpercentile(r2, 10))
    out["n_leads_r2_gt_0999"] = int(np.sum(r2 > 0.999)); out["n_leads"] = int(len(r2))
    out["flat_fraction_median"] = float(np.median(flat))
    out["six_gaussian_r2_median_by_lead"] = {}
    keys = [k for k, _ in ecg.groupby(["subject", "lead"], sort=False)]
    for L in ecg.lead.unique():
        out["six_gaussian_r2_median_by_lead"][L] = float(np.nanmedian([r for (s, l), r in zip(keys, r2) if l == L]))
    # 2. limb-lead consistency (relative error: max |residual| / peak-to-peak amplitude of lead II)
    dup = set(sub.subject[sub.qc_ecg_duplicate == 1])
    rel = {"Einthoven_III": [], "aVR": [], "aVL": [], "aVF": []}
    for s, d in ecg.groupby("subject", sort=False):
        if s in dup:
            continue
        g = {l: dd.mV.values for l, dd in d.groupby("lead")}
        I, II = g["I"], g["II"]; sc = np.ptp(II) + 1e-9
        for k, res in (("Einthoven_III", g["III"] - (II - I)), ("aVR", g["aVR"] + (I + II) / 2),
                       ("aVL", g["aVL"] - (I - II / 2)), ("aVF", g["aVF"] - (II - I / 2))):
            rel[k].append(np.max(np.abs(res)) / sc)
    out["limb_lead_relative_error_median"] = {k: float(np.median(v)) for k, v in rel.items()}
    out["limb_lead_n_einthoven_within_10pct"] = int(np.sum(np.array(rel["Einthoven_III"]) < 0.10))
    out["limb_lead_n"] = len(rel["Einthoven_III"])
    # 5. between-participant similarity of the 12-lead beat
    from features import ecg_waveforms
    W = ecg_waveforms(ecg, n=200)
    first = sub.batch == sub.batch.iloc[0]
    grp = {"first_batch_controls": sub[first & (sub.y == 0) & ~sub.subject.isin(dup)].subject,
           "first_batch_diseased": sub[first & (sub.y == 1)].subject, "later_batch_diseased": sub[~first].subject}
    cr = lambda a, b: float(np.corrcoef(W[a].ravel(), W[b].ravel())[0, 1])
    sim = {}
    for k, ss in grp.items():
        ss = list(ss); c = [cr(a, b) for i, a in enumerate(ss) for b in ss[i + 1:]]
        sim[k] = dict(n=len(ss), median_r=float(np.median(c)), n_pairs=len(c), n_pairs_r_gt_0p9=int(np.sum(np.array(c) > 0.9)))
    dis = list(sub.subject[sub.y == 1]); ctl = list(grp["first_batch_controls"])
    sim["diseased_vs_controls_median_r"] = float(np.median([cr(a, b) for a in dis for b in ctl]))
    cd = [cr(a, b) for i, a in enumerate(dis) for b in dis[i + 1:]]
    sim["all_diseased"] = dict(median_r=float(np.median(cd)), n_pairs=len(cd), n_pairs_r_gt_0p9=int(np.sum(np.array(cd) > 0.9)))
    out["ecg_similarity"] = sim
    # 3. conduction
    pr = E["PR_II_ms"].values
    out["pr_median_ms"] = float(np.median(pr)); out["n_pr_lt_120"] = int(np.sum(pr < 120))
    out["n_pr_gt_200"] = int(np.sum(pr > 200)); out["pr_diseased_ms"] = {s: float(E.PR_II_ms[s]) for s in sub.subject[y == 1]}
    out["pr_range_ms"] = [float(pr.min()), float(pr.max())]
    out["qrs_global_ge_120"] = int(np.sum(E["QRS_global_ms"].values >= 120))
    # 3. crude v/c vs physician diagnosis
    vc = crude_vc(cyc).loc[sub.subject].values
    pred = vc > 1
    out["crude_vc_auroc"] = float(roc_auc_score(y, vc))
    out["crude_vc_agree"] = int(np.sum(pred == (y == 1))); out["n"] = int(len(y))
    out["crude_vc_sens"] = float(pred[y == 1].mean()); out["crude_vc_spec"] = float((~pred[y == 0]).mean())
    out["crude_vc_disagreements"] = [s for s, p, t in zip(sub.subject, pred, y) if p != (t == 1)]
    json.dump(out, open(RES / "ecg_investigation.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 52 * MM), gridspec_kw=dict(wspace=0.45))
    t, v, fit, r = example
    axs[0].plot(t * 1000, v, color=INK, lw=1.6, label="ECG file (HS16, lead II)")
    axs[0].plot(t * 1000, fit, color=C_DIS, lw=1.0, ls="--", label=f"6-Gaussian model, $R^2$={r:.5f}")
    axs[0].set_xlabel("Time (ms)"); axs[0].set_ylabel("Amplitude (mV)"); axs[0].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    axs[0].set_title("ECG beats are parametric"); panel(axs[0], "a")
    rng = np.random.default_rng(0)
    for yy, col in [(0, C_CTL), (1, C_DIS)]:
        axs[1].scatter(yy + rng.uniform(-.17, .17, (y == yy).sum()), pr[y == yy], s=11, color=col, edgecolor="white", lw=.4, zorder=3)
    axs[1].axhspan(120, 200, color=GRID, zorder=0); axs[1].axhline(200, color=INK2, ls="--", lw=.7)
    axs[1].text(1.48, 160, "normal\nPR", fontsize=6, color=INK2, va="center", ha="right")
    axs[1].text(1.48, 207, "1st-degree\nAV block", fontsize=6, color=INK2, va="bottom", ha="right")
    axs[1].set_ylim(0, 260); axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["Control", "Diseased"]); axs[1].set_xlim(-.5, 1.5)
    axs[1].set_ylabel("PR interval, lead II (ms)"); axs[1].set_title("No conduction delay encoded"); panel(axs[1], "b")
    for yy, col in [(0, C_CTL), (1, C_DIS)]:
        axs[2].scatter(yy + rng.uniform(-.17, .17, (y == yy).sum()), vc[y == yy], s=11, color=col, edgecolor="white", lw=.4, zorder=3)
    axs[2].axhline(1, color=INK2, ls="--", lw=.7); axs[2].set_xticks([0, 1]); axs[2].set_xticklabels(["Control", "Diseased"])
    axs[2].set_xlim(-.5, 1.5); axs[2].set_ylabel("Crude v/c (raw samples)")
    axs[2].set_title(f"Crude v/c vs diagnosis ({out['crude_vc_agree']}/{out['n']})"); panel(axs[2], "c")
    save(fig, "figS2_ecg_investigation")


if __name__ == "__main__":
    main()
