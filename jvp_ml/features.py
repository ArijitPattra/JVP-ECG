"""Start-agnostic feature extraction for JVP sensor windows, and ECG beat features.

JVP: each workbook holds a short sensor window (0.6-1.1 s, about one cardiac period). Nothing is
assumed about where in the cardiac cycle the window starts, and its duration is not used. Each window
is resampled to 64 points (PCHIP), min-max normalised (contact pressure and gain differ between
participants) and treated as one period of a periodic signal. All features are invariant to a
circular shift of the window:
  * trough-anchored fiducials: the window is rotated so that its deepest trough (taken as the x
    descent) is at the origin; the prominent peaks then follow in physiological order
    x -> v -> (y) -> a -> c, so v is the first prominent peak after x and c the last one before it;
  * Fourier descriptors: normalised harmonic magnitudes and shift-invariant relative phases
    (phi_k - k * phi_1);
  * amplitude-distribution shape (skewness, kurtosis, time above half-range) and roughness;
  * beat-to-beat similarity (maximum circular cross-correlation between windows).
Only the sensor signal is used; ECG timing is never used to locate JVP features.

ECG: one reference beat per lead for all 12 standard leads (V1-V6, I, II, III, aVR, aVL, aVF). Per lead: R and
S amplitudes, R/S ratio, R-|S|, QRS duration, P and T amplitudes, PR and QT intervals. Global 12-lead measures:
median QRS duration over leads, PR and QT in lead II, Bazett-corrected QT (beat length as RR), frontal QRS axis
(net QRS deflection in I and aVF), Sokolow-Lyon RVH (R V1 + S V5/V6) and LVH (S V1 + R V5/V6) voltages, R/S in
V6, and the voltage and axis criteria for RVH and LVH (Hancock et al., Circulation 2009). The V1/V2 subset
(ECG_V1V2_FEATS) reproduces the earlier two-lead comparator.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

N_PHASE = 64
PHASE = np.linspace(0.0, 1.0, N_PHASE, endpoint=False)
PROMINENCE = 0.08  # minimum peak prominence (fraction of the window range)
K_RATIO = 0.05  # ratio regulariser (5 % of the range) keeps ratios bounded when a wave is absent


def resample_cycle(y, n=N_PHASE):
    y = np.asarray(y, float)
    x = np.linspace(0.0, 1.0, len(y))
    return PchipInterpolator(x, y)(np.linspace(0.0, 1.0, n))


def minmax(y):
    lo, hi = np.min(y), np.max(y)
    return (y - lo) / (hi - lo + 1e-12)


def circular_peaks(r, prominence=PROMINENCE):
    """Prominent peaks of a periodic signal r (indices in 1..len-1, sorted)."""
    L = len(r)
    ext = np.r_[r, r, r]
    pk, pr = find_peaks(ext, prominence=prominence)
    keep = (pk >= L) & (pk < 2 * L)
    return pk[keep] - L, pr["prominences"][keep]


def trough_anchor(w):
    """Rotate a normalised window so that its global minimum (x descent) is at index 0."""
    i0 = int(np.argmin(w))
    return np.roll(w, -i0), i0


def jvp_fiducials(r):
    """r: trough-anchored window (r[0] = x). Returns indices (v, y, a, c) and the number of peaks."""
    L = len(r)
    pk, _ = circular_peaks(r)
    pk = pk[pk > 0]
    if len(pk) >= 2:
        iv, ic = pk[0], pk[-1]
        ia = pk[-2] if len(pk) >= 3 else ic
    else:  # a single dominant hump: split the period after the x trough into halves
        iv = int(np.argmax(r[1: L // 2])) + 1
        ic = L // 2 + int(np.argmax(r[L // 2:]))
        ia = ic
    nxt = pk[pk > iv]
    stop = nxt[0] if len(nxt) else ic
    iy = iv + int(np.argmin(r[iv: stop + 1])) if stop > iv else iv
    return iv, iy, ia, ic, len(pk)


def jvp_cycle_features(raw):
    w = minmax(resample_cycle(raw))
    r, _ = trough_anchor(w)
    iv, iy, ia, ic, npk = jvp_fiducials(r)
    yv, yy, ya, yc = r[[iv, iy, ia, ic]]
    k, L = K_RATIO, len(r)
    dr = np.gradient(np.r_[r[-1], r, r[0]])[1:-1] * L
    f = {
        "vc_ratio": (yv + k) / (yc + k),  # hypothesis: v/c > 1 -> diseased
        "va_ratio": (yv + k) / (ya + k),
        "ca_ratio": (yc + k) / (ya + k),
        "amp_v": yv, "amp_c": yc, "amp_a": ya, "y_descent": yv - yy, "amp_y": yy,
        "phase_x_to_v": iv / L, "phase_c_to_x": (L - ic) / L, "phase_v_to_c": (ic - iv) / L,
        "slope_a_to_c": (yc - ya) / ((ic - ia) / L) if ic > ia else 0.0,  # hypothesis: flattened in heart block
        "slope_v_upstroke": float(np.max(dr[: iv + 1])) if iv > 0 else 0.0,
        "slope_x_descent": float(np.min(dr[ic:])) if ic < L else 0.0,
        "n_peaks": npk,
        "skewness": float(skew(w)), "kurtosis": float(kurtosis(w)), "frac_above_half": float(np.mean(w > 0.5)),
        "roughness": float(np.sum(np.abs(np.r_[w[1:], w[0]] - 2 * w + np.r_[w[-1], w[:-1]]))),
    }
    H = np.fft.rfft(w - w.mean())
    mag = np.abs(H); mag = mag / (mag[1:7].sum() + 1e-9); ph = np.angle(H)
    for h in range(1, 7):
        f[f"harm{h}"] = float(mag[h])
    for h in (2, 3, 4):  # relative phases are invariant to circular shifts of the window
        d = ph[h] - h * ph[1]
        f[f"relphase{h}_cos"], f[f"relphase{h}_sin"] = float(np.cos(d)), float(np.sin(d))
    return f, r, (iv, iy, ia, ic)


def max_circular_xcorr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.max(np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))) / len(a))


def jvp_subject_features(cyc_df, signal="force_N"):
    """Subject-level features (median over windows) and trough-anchored windows for the waveform models."""
    rows, waves = [], []
    for s, g in cyc_df.groupby("subject", sort=False):
        feats, rs = [], []
        for c, gc in g.groupby("cycle", sort=False):
            f, r, _ = jvp_cycle_features(gc[signal].values)
            feats.append(f); rs.append(r); waves.append((s, c, r))
        F = pd.DataFrame(feats)
        row = {"subject": s, **{k: F[k].median() for k in F.columns}}
        row["vc_ratio_sd"] = F["vc_ratio"].std(ddof=0)
        row["beat_consistency"] = (np.mean([max_circular_xcorr(rs[i], rs[j]) for i in range(len(rs))
                                            for j in range(i + 1, len(rs))]) if len(rs) > 1 else np.nan)
        row["n_cycles_used"] = len(rs)
        rows.append(row)
    return pd.DataFrame(rows), waves


def jvp_cycle_table(cyc_df, signal="force_N"):
    """Window-level feature table (for plots)."""
    rows = []
    for (s, c), g in cyc_df.groupby(["subject", "cycle"], sort=False):
        f, r, fid = jvp_cycle_features(g[signal].values)
        f.update(subject=s, cycle=c, n_samples=len(g))
        rows.append(f)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ECG
def ecg_lead_features(t, v):
    t, v = np.asarray(t, float), np.asarray(v, float)
    v = v - np.median(v[: max(5, len(v) // 20)])
    fs = (len(t) - 1) / (t[-1] - t[0])
    dv = np.gradient(v) * fs  # index-based: some exports contain repeated time stamps
    # QRS: steepest slope; R = max, S = min near it
    i_steep = int(np.argmax(np.abs(dv)))
    w = int(0.06 * fs)  # R and S lie within 60 ms of the steepest QRS slope (keeps tall T waves out)
    lo, hi = max(0, i_steep - w), min(len(v), i_steep + w)
    iR = lo + int(np.argmax(v[lo:hi]))
    iS = lo + int(np.argmin(v[lo:hi]))
    # QRS onset: PR-segment minimum in the 80 ms before the first QRS peak, then walk back
    # along the descending Q limb; offset: first return to baseline after the last QRS deflection
    i_first, i_last = min(iR, iS), max(iR, iS)
    amp = max(abs(v[iR]), abs(v[iS]))
    w0 = max(0, i_first - int(0.08 * fs))
    q_on = w0 + int(np.argmin(np.abs(v[w0: i_first + 1]) - 1e-9 * np.arange(i_first + 1 - w0)))
    while q_on > 0 and abs(v[q_on - 1]) > 0.05 * amp and abs(v[q_on - 1]) < abs(v[q_on]) + 1e-12:
        q_on -= 1
    w1 = min(len(v), i_last + int(0.12 * fs))
    back = np.where(np.abs(v[i_last:w1]) < 0.15 * amp)[0]
    q_off = i_last + (back[0] if len(back) else int(np.argmin(np.abs(dv[i_last:w1]))))
    pre = v[: max(1, q_on - int(0.02 * fs))]
    iP = int(np.argmax(pre)) if len(pre) else 0
    P = pre[iP] if len(pre) else 0.0
    p_on = np.where(pre[: iP + 1] < 0.1 * P)[0] if P > 0 else []
    p_on = p_on[-1] if len(p_on) else 0
    post_start = min(len(v) - 1, q_off + int(0.04 * fs))
    post = v[post_start:]
    iT = post_start + int(np.argmax(np.abs(post))) if len(post) else q_off
    T = v[iT]
    after = np.where(np.abs(v[iT:]) < 0.1 * abs(T))[0]
    t_end = iT + after[0] if len(after) else len(v) - 1
    R, S = max(v[iR], 0.0), min(v[iS], 0.0)
    return {
        "R_mV": R, "S_mV": S, "RS_ratio": R / (abs(S) + 1e-3), "R_minus_absS": R - abs(S),
        "QRS_ms": (t[q_off] - t[q_on]) * 1000, "P_mV": P, "T_mV": T,
        "PR_ms": (t[q_on] - t[p_on]) * 1000, "QT_ms": (t[t_end] - t[q_on]) * 1000,
        "beat_ms": (t[-1] - t[0]) * 1000,
    }


LEADS12 = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "II", "III", "aVR", "aVL", "aVF"]
LEAD_FEATS = ["R_mV", "S_mV", "RS_ratio", "R_minus_absS", "QRS_ms", "P_mV", "T_mV", "PR_ms", "QT_ms"]
RVH_V1V2 = ["RVH_crit_RV1_ge_0.7", "RVH_crit_RS_V1_ge_1"]
ECG_GLOBAL = ["beat_ms", "QRS_global_ms", "PR_II_ms", "QT_II_ms", "QTc_ms", "QRS_axis_deg", "SokolowLyon_RVH_mV",
              "SokolowLyon_LVH_mV", "RS_V6_ratio"]
ECG_CRITERIA = RVH_V1V2 + ["RVH_crit_SL_gt_1.05", "RVH_crit_S_V5V6_ge_0.7", "RVH_crit_RS_V6_le_1", "RVH_crit_RAD",
                           "RVH_any_criterion", "LVH_crit_SL_ge_3.5"]
ECG_V1V2_FEATS = [f"{L}_{k}" for L in ("V1", "V2") for k in LEAD_FEATS + ["beat_ms"]] + RVH_V1V2
ECG12_FEATS = [f"{L}_{k}" for L in LEADS12 for k in LEAD_FEATS] + ECG_GLOBAL + ECG_CRITERIA


def ecg_global_features(row):
    """Clinical 12-lead measures from the per-lead features of one participant."""
    g = {}
    qrs = [row[f"{L}_QRS_ms"] for L in LEADS12 if f"{L}_QRS_ms" in row]
    g["beat_ms"] = row.get("II_beat_ms", row.get("V1_beat_ms", np.nan))
    g["QRS_global_ms"] = float(np.median(qrs)) if qrs else np.nan
    g["PR_II_ms"], g["QT_II_ms"] = row.get("II_PR_ms", np.nan), row.get("II_QT_ms", np.nan)
    g["QTc_ms"] = g["QT_II_ms"] / np.sqrt(g["beat_ms"] / 1000.0)
    g["QRS_axis_deg"] = float(np.degrees(np.arctan2(row.get("aVF_R_minus_absS", np.nan), row.get("I_R_minus_absS", np.nan))))
    sv56 = max(abs(row.get("V5_S_mV", 0.0)), abs(row.get("V6_S_mV", 0.0)))
    rv56 = max(row.get("V5_R_mV", 0.0), row.get("V6_R_mV", 0.0))
    g["SokolowLyon_RVH_mV"] = row.get("V1_R_mV", np.nan) + sv56
    g["SokolowLyon_LVH_mV"] = abs(row.get("V1_S_mV", np.nan)) + rv56
    g["RS_V6_ratio"] = row.get("V6_RS_ratio", np.nan)
    c = {
        "RVH_crit_RV1_ge_0.7": row.get("V1_R_mV", np.nan) >= 0.7,
        "RVH_crit_RS_V1_ge_1": row.get("V1_RS_ratio", np.nan) >= 1.0,
        "RVH_crit_SL_gt_1.05": g["SokolowLyon_RVH_mV"] > 1.05,
        "RVH_crit_S_V5V6_ge_0.7": sv56 >= 0.7,
        "RVH_crit_RS_V6_le_1": g["RS_V6_ratio"] <= 1.0,
        "RVH_crit_RAD": g["QRS_axis_deg"] > 90.0,  # right-axis deviation (normal -30 to +90 degrees)
        "LVH_crit_SL_ge_3.5": g["SokolowLyon_LVH_mV"] >= 3.5,
    }
    c["RVH_any_criterion"] = any(c[k] for k in c if k.startswith("RVH"))
    g.update({k: float(v) for k, v in c.items()})
    return g


def ecg_subject_features(ecg_df):
    rows = []
    for s, g in ecg_df.groupby("subject", sort=False):
        row = {"subject": s}
        for lead in LEADS12:
            gl = g[g.lead == lead]
            if len(gl) < 10:
                continue
            for k, val in ecg_lead_features(gl.t.values, gl.mV.values).items():
                row[f"{lead}_{k}"] = val
        row.update(ecg_global_features(row))
        rows.append(row)
    return pd.DataFrame(rows)


def ecg_waveforms(ecg_df, n=200, leads=tuple(LEADS12)):
    """Resampled beat (n_leads x n) per subject; missing lead -> zeros."""
    out = {}
    for s, g in ecg_df.groupby("subject", sort=False):
        X = np.zeros((len(leads), n))
        for j, lead in enumerate(leads):
            gl = g[g.lead == lead]
            if len(gl) >= 10:
                tt = np.linspace(gl.t.min(), gl.t.max(), n)
                X[j] = np.interp(tt, gl.t.values, gl.mV.values)
        out[s] = X
    return out
