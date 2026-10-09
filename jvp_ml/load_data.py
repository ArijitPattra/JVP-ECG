"""Load the rGO/CuO JVP-sensor + ECG workbooks into tidy tables and a QC report.

Each subject folder (HS<n>) contains JVP_<lead>_Cycle<k>.xlsx workbooks, where <lead> is
V1, V2 or V1_V2. Every workbook holds:
  * one manually segmented JVP cycle (sensor time, current in A, calibrated force in N),
  * one ECG beat (time, amplitude in mV) for the lead(s) in the file name,
  * a subject-level label ("Diseased" / "Not Diseased").
The same JVP cycle is repeated across the V1, V2 and V1_V2 workbooks of a subject, so
cycles are de-duplicated by content. Column positions differ between workbooks, so
columns are located from the header text.

Usage: python load_data.py <raw_data_dir> <out_dir>
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd


def _num(col):
    return [float(v) for v in col if isinstance(v, (int, float)) and not isinstance(v, bool)]


def read_workbook(path):
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).active
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    h0, h1, body = rows[0], rows[1], rows[2:]
    ncol = max(len(r) for r in rows)
    pad = lambda r: list(r) + [None] * (ncol - len(r))
    h0, h1 = pad(h0), pad(h1)
    cols = list(zip(*[pad(r) for r in body])) if body else [[]] * ncol
    txt = lambda x: str(x).strip() if x is not None else ""
    tcols = [i for i, v in enumerate(h1) if txt(v).lower().startswith("time")]
    lab_col = [i for i, v in enumerate(h0) if "label" in txt(v).lower()]
    out = {
        "header_subject": txt(h0[0]),
        "label": txt(h1[lab_col[0]]) if lab_col else "",
        "jvp_t": _num(cols[tcols[0]]),
        "jvp_I": _num(cols[tcols[0] + 1]),
        "jvp_F": _num(cols[tcols[0] + 2]),
        "ecg_t": [],
        "ecg": {},
    }
    if len(tcols) > 1:
        out["ecg_t"] = _num(cols[tcols[1]])
        k = tcols[1] + 1
        amp_cols = []
        while k < ncol and "amplitude" in txt(h1[k]).lower():
            amp_cols.append(k)
            k += 1
        out["ecg_cols"] = [(txt(h0[c]) or txt(h0[c - 1])) for c in amp_cols]
        out["ecg_amp"] = [_num(cols[c]) for c in amp_cols]
    return out


def _hash(x):
    return hashlib.md5(np.round(np.asarray(x, float), 12).tobytes()).hexdigest()[:12]


def fix_ecg_time(t, n, ref_duration=None):
    """ECG time stamps are seconds in most files; repair unit errors (ms, x10, /1000)."""
    t = np.asarray(t, float)
    dur = t[-1] - t[0] if len(t) > 1 else np.nan
    note = ""
    if len(t) != n or not np.isfinite(dur) or dur <= 0:
        dur = ref_duration if ref_duration else 0.8
        note = "time missing/len mismatch -> uniform grid"
        t = np.linspace(0, dur, n)
    elif dur > 100:  # milliseconds
        t, note = t / 1000.0, "ms->s"
    elif dur > 3:  # factor-10 typo (e.g. 8.8 s for one beat)
        t, note = t / 10.0, "x10 typo -> /10"
    elif dur < 0.1:  # stored in ks
        t, note = (t / t[-1]) * (ref_duration or 0.8), "implausible duration -> rescaled to other lead"
    return t, note


def main(raw_dir, out_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subj_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("HS")],
                       key=lambda p: int(p.name[2:]))
    cyc_rows, ecg_rows, subj_rows, qc = [], [], [], []
    ecg_hash_owner = {}
    for sd in subj_dirs:
        s = sd.name
        labels, hdr_ids = set(), set()
        cycles = {}  # hash -> (cycle number, data)
        ecg = {}
        for f in sorted(sd.glob("JVP_*_Cycle*.xlsx")):
            m = re.match(r"JVP_(V1_V2|V1|V2)_Cycle(\d+)\.xlsx", f.name)
            leads_in_name = m.group(1).split("_")
            cyc = int(m.group(2))
            wb = read_workbook(f)
            labels.add(wb["label"])
            hdr_ids.add(wb["header_subject"])
            h = _hash(wb["jvp_I"])
            if h not in cycles:
                cycles[h] = (cyc, f.name, wb)
            elif cycles[h][0] != cyc:
                qc.append((s, f.name, "duplicate-cycle-within-subject",
                           f"Cycle{cyc} identical to Cycle{cycles[h][0]}; counted once"))
            # lead identity comes from the file name (single-lead headers are unreliable)
            for j, amp in enumerate(wb.get("ecg_amp", [])):
                lead = leads_in_name[j] if j < len(leads_in_name) else f"X{j}"
                hdr_lead = wb["ecg_cols"][j].replace("_", " ").replace("Lead", "").strip()
                if hdr_lead and hdr_lead != lead:
                    qc.append((s, f.name, "header-lead-mismatch", f"header says {hdr_lead}, file name says {lead}"))
                if lead not in ecg:
                    ecg[lead] = (wb["ecg_t"], amp, f.name)
        if len(labels) != 1:
            qc.append((s, "*", "label-inconsistent", str(labels)))
        for hid in hdr_ids - {s}:
            qc.append((s, "*", "header-subject-mismatch", f"workbook header names {hid} (copied template?)"))
        # cycles: keep content-unique cycles, renumber by order of appearance
        seen_cycle_numbers = {}
        for h, (cyc, fname, wb) in sorted(cycles.items(), key=lambda kv: (kv[1][0], kv[1][1])):
            if cyc in seen_cycle_numbers:
                qc.append((s, fname, "cycle-content-differs", f"Cycle{cyc} differs between workbooks; kept both"))
            seen_cycle_numbers.setdefault(cyc, 0)
            seen_cycle_numbers[cyc] += 1
            cid = f"{cyc}" if seen_cycle_numbers[cyc] == 1 else f"{cyc}b"
            t, I, F = map(np.asarray, (wb["jvp_t"], wb["jvp_I"], wb["jvp_F"]))
            n = min(len(t), len(I), len(F))
            if len({len(t), len(I), len(F)}) > 1:
                qc.append((s, fname, "jvp-column-length-mismatch",
                           f"t={len(t)}, I={len(I)}, F={len(F)}; truncated to {n}"))
            for i in range(n):
                cyc_rows.append((s, cid, i, t[i] - t[0], I[i], F[i]))
        # within-subject identical cycles
        arrs = {}
        for h, (cyc, fname, wb) in cycles.items():
            arrs.setdefault(_hash(wb["jvp_F"]), []).append(cyc)
        # ECG
        durations = {}
        for lead, (t, amp, fname) in ecg.items():
            if len(t) == len(amp) and len(t) > 1 and 0.3 < t[-1] - t[0] < 2.0:
                durations[lead] = t[-1] - t[0]
        for lead, (t, amp, fname) in ecg.items():
            ref = np.mean(list(durations.values())) if durations else None
            tt, note = fix_ecg_time(t[: len(amp)] if len(t) >= len(amp) else t, len(amp), ref)
            if note:
                qc.append((s, fname, "ecg-time-repaired", f"{lead}: {note}"))
            hh = _hash(amp)
            if hh in ecg_hash_owner and ecg_hash_owner[hh] != s:
                qc.append((s, fname, "ecg-duplicate-of-other-subject", f"{lead} identical to {ecg_hash_owner[hh]}"))
            ecg_hash_owner.setdefault(hh, s)
            for i in range(len(amp)):
                ecg_rows.append((s, lead, i, tt[i], amp[i]))
        lab = sorted(l for l in labels if l)[0] if any(labels) else ""
        subj_rows.append((s, lab, int(lab.lower() == "diseased"), len(cycles), ",".join(sorted(ecg))))

    cyc_df = pd.DataFrame(cyc_rows, columns=["subject", "cycle", "idx", "t", "current_A", "force_N"])
    ecg_df = pd.DataFrame(ecg_rows, columns=["subject", "lead", "idx", "t", "mV"])
    subj_df = pd.DataFrame(subj_rows, columns=["subject", "label", "y", "n_cycles", "ecg_leads"])
    qc_df = pd.DataFrame(qc, columns=["subject", "file", "issue", "detail"])

    # within-subject duplicated cycles (identical waveform under different cycle numbers)
    for s, g in cyc_df.groupby("subject"):
        hs = {c: _hash(gg.force_N.values) for c, gg in g.groupby("cycle")}
        inv = {}
        for c, h in hs.items():
            inv.setdefault(h, []).append(c)
        for h, cs in inv.items():
            if len(cs) > 1:
                qc_df.loc[len(qc_df)] = (s, "*", "duplicate-cycle-within-subject", f"cycles {cs} identical")

    # force vs current calibration check
    cal = []
    for (s, c), g in cyc_df.groupby(["subject", "cycle"]):
        r = np.corrcoef(g.current_A, g.force_N)[0, 1]
        cal.append((s, c, r))
    cal = pd.DataFrame(cal, columns=["subject", "cycle", "r_current_force"])

    ecg_dup_subjects = sorted(set(qc_df.loc[qc_df.issue == "ecg-duplicate-of-other-subject", "subject"]))
    subj_df["qc_ecg_duplicate"] = subj_df.subject.isin(ecg_dup_subjects).astype(int)
    subj_df["qc_header_mismatch"] = subj_df.subject.isin(
        set(qc_df.loc[qc_df.issue == "header-subject-mismatch", "subject"])).astype(int)

    cyc_df.to_csv(out_dir / "jvp_cycles.csv", index=False)
    ecg_df.to_csv(out_dir / "ecg_beats.csv", index=False)
    subj_df.to_csv(out_dir / "subjects.csv", index=False)
    qc_df.to_csv(out_dir / "qc_issues.csv", index=False)
    cal.to_csv(out_dir / "force_current_calibration.csv", index=False)

    with open(out_dir / "qc_report.md", "w") as fh:
        fh.write("# Data QC report\n\n")
        fh.write(f"* Subject folders: {len(subj_df)} (missing IDs: "
                 f"{sorted(set(range(1, max(int(s[2:]) for s in subj_df.subject) + 1)) - set(int(s[2:]) for s in subj_df.subject))})\n")
        fh.write(f"* Labels: {subj_df.label.value_counts().to_dict()}\n")
        fh.write(f"* JVP cycles (content-unique): {cyc_df.groupby(['subject','cycle']).ngroups}; "
                 f"samples/cycle median {cyc_df.groupby(['subject','cycle']).size().median():.0f} "
                 f"(range {cyc_df.groupby(['subject','cycle']).size().min()}-{cyc_df.groupby(['subject','cycle']).size().max()})\n")
        fh.write(f"* |r(current, force)| median {cal.r_current_force.abs().median():.6f}, min {cal.r_current_force.abs().min():.6f} "
                 f"-> force is a per-recording linear calibration of current\n")
        fh.write(f"* Subjects whose ECG is byte-identical to another subject: {ecg_dup_subjects}\n\n")
        fh.write("## Issues\n\n" + qc_df.to_markdown(index=False) + "\n")
    print(open(out_dir / "qc_report.md").read()[:3000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
