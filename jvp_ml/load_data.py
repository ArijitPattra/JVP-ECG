"""Load the rGO/CuO JVP-sensor + ECG workbooks into tidy tables and a QC report.

Each subject folder (HS<n>) contains JVP_<lead>_Cycle<k>.xlsx workbooks, where <lead> is
V1, V2 or V1_V2. Every workbook holds:
  * one manually segmented JVP cycle (sensor time, current in A, calibrated force in N),
  * one ECG beat (time, amplitude in mV) for the lead(s) in the file name,
  * a subject-level label ("Diseased" / "Not Diseased").
From the November 2026 delivery onwards each folder also holds "All leads.xlsx": one reference ECG beat for all
12 standard leads (V1-V6, I, II, III, aVR, aVL, aVF) on a common time axis, with the subject label. When this file
is present it is the ECG source; its V1/V2 traces are compared with those of the single-lead workbooks (QC).
The same JVP cycle is repeated across the V1, V2 and V1_V2 workbooks of a subject, so
cycles are de-duplicated by content. Column positions differ between workbooks, so
columns are located from the header text.

Several raw directories (recording batches) can be given; subject IDs must be unique across them, and the
batch (directory name) of each subject is recorded in subjects.csv.

With --map <csv> (from map_subjects.py) the recording batch and the earlier subject ID are taken from the map
instead of from the directory name.

Usage: python load_data.py [--map subject_map.csv] <raw_data_dir> [<raw_data_dir> ...] <out_dir>
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
        out["ecg_unlabelled_amp"] = 0
        while k < ncol:
            if "amplitude" in txt(h1[k]).lower():
                amp_cols.append(k)
            elif not txt(h1[k]) and "lead" in txt(h0[k]).lower() and _num(cols[k]):
                amp_cols.append(k)  # amplitude column under a lead header but without its unit header
                out["ecg_unlabelled_amp"] += 1
            else:
                break
            k += 1
        out["ecg_cols"] = [(txt(h0[c]) or txt(h0[c - 1])) for c in amp_cols]
        out["ecg_amp"] = [_num(cols[c]) for c in amp_cols]
    return out


LEADS12 = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "II", "III", "aVR", "aVL", "aVF"]


def read_all_leads(path):
    """12-lead workbook: row 0 = subject ID + 'Label', row 1 = 'Time (in s)', 'Lead V1' ... 'Lead aVF', label."""
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).active
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    txt = lambda x: str(x).strip() if x is not None else ""
    h0, h1 = rows[0], rows[1]
    tcol = [i for i, v in enumerate(h1) if txt(v).lower().startswith("time")][0]
    lab_col = [i for i, v in enumerate(h0) if "label" in txt(v).lower()]
    leads = {}
    for i, v in enumerate(h1):
        name = txt(v).replace("Lead", "").strip()
        if name in LEADS12:
            leads[name] = i
    body = rows[2:]
    t = [r[tcol] for r in body]
    keep = [k for k, x in enumerate(t) if isinstance(x, (int, float))]
    return {"header_subject": txt(h0[0]), "label": txt(h1[lab_col[-1]]) if lab_col else "",
            "t": np.array([float(t[k]) for k in keep]),
            "leads": {L: np.array([float(body[k][c]) if isinstance(body[k][c], (int, float)) else np.nan for k in keep])
                      for L, c in leads.items()}}


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


def main(raw_dirs, out_dir, subject_map=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subj_dirs = sorted([p for d in map(Path, raw_dirs) for p in d.iterdir() if p.is_dir() and p.name.startswith("HS")],
                       key=lambda p: int(p.name[2:]))
    dup_ids = sorted({p.name for p in subj_dirs if [q.name for q in subj_dirs].count(p.name) > 1})
    if dup_ids:
        raise SystemExit(f"subject IDs present in more than one raw directory: {dup_ids}")
    cyc_rows, ecg_rows, subj_rows, qc = [], [], [], []
    ecg_hash_owner = {}
    ecg_dup_pairs = set()
    for sd in subj_dirs:
        s = sd.name
        labels, hdr_ids = set(), set()
        cycles = {}  # hash -> (cycle number, data)
        ecg = {}
        ecg12 = read_all_leads(sd / "All leads.xlsx") if (sd / "All leads.xlsx").exists() else None
        if ecg12 is not None:
            labels.add(ecg12["label"]); hdr_ids.add(ecg12["header_subject"])
            missing = [L for L in LEADS12 if L not in ecg12["leads"]]
            if missing:
                qc.append((s, "All leads.xlsx", "ecg12-lead-missing", ",".join(missing)))
        single = {}
        for f in sorted(sd.glob("JVP_*_Cycle*.xlsx")):
            m = re.match(r"JVP_(V1_V2|V1|V2)_Cycle(\d+)\.xlsx", f.name)
            leads_in_name = m.group(1).split("_")
            cyc = int(m.group(2))
            wb = read_workbook(f)
            if wb.get("ecg_unlabelled_amp"):
                qc.append((s, f.name, "ecg-amplitude-header-missing",
                           f"{wb['ecg_unlabelled_amp']} amplitude column(s) without unit header; read by lead header"))
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
                if lead not in single:
                    single[lead] = (wb["ecg_t"], amp, f.name)
        if ecg12 is None:
            ecg = single
        else:
            for L, amp in ecg12["leads"].items():
                ecg[L] = (list(ecg12["t"]), list(amp), "All leads.xlsx")
            for L in ("V1", "V2"):  # agreement of the 12-lead file with the single-lead workbooks
                if L in single and L in ecg12["leads"]:
                    a, b = np.asarray(single[L][1], float), ecg12["leads"][L]
                    if len(a) != len(b) or np.nanmax(np.abs(a - b)) > 1e-6:
                        d = f"length {len(a)} vs {len(b)}" if len(a) != len(b) else f"max |diff| {np.nanmax(np.abs(a - b)):.3f} mV"
                        qc.append((s, "All leads.xlsx", "ecg12-differs-from-single-lead-workbook", f"{L}: {d}"))
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
            if note and not (fname == "All leads.xlsx" and any(q[:3] == (s, fname, "ecg-time-repaired") for q in qc)):
                qc.append((s, fname, "ecg-time-repaired", note if fname == "All leads.xlsx" else f"{lead}: {note}"))
            hh = _hash(amp)
            if hh in ecg_hash_owner and ecg_hash_owner[hh] != s:
                ecg_dup_pairs.add((ecg_hash_owner[hh], s))
            ecg_hash_owner.setdefault(hh, s)
            for i in range(len(amp)):
                ecg_rows.append((s, lead, i, tt[i], amp[i]))
        lab = sorted(l for l in labels if l)[0] if any(labels) else ""
        subj_rows.append((s, lab, int(lab.lower() == "diseased"), len(cycles), ",".join(sorted(ecg)), sd.parent.name))

    cyc_df = pd.DataFrame(cyc_rows, columns=["subject", "cycle", "idx", "t", "current_A", "force_N"])
    ecg_df = pd.DataFrame(ecg_rows, columns=["subject", "lead", "idx", "t", "mV"])
    subj_df = pd.DataFrame(subj_rows, columns=["subject", "label", "y", "n_cycles", "ecg_leads", "batch"])
    qc_df = pd.DataFrame(qc, columns=["subject", "file", "issue", "detail"])

    # ECG copied between participants: the copy is the member of the pair whose 12-lead file disagrees with its
    # own single-lead workbooks; if neither (or both) disagree, the later participant is flagged
    differs = set(qc_df.loc[qc_df.issue == "ecg12-differs-from-single-lead-workbook", "subject"])
    for a, b in sorted(ecg_dup_pairs, key=lambda p: (int(p[0][2:]), int(p[1][2:]))):
        copy, orig = (a, b) if (a in differs and b not in differs) else (b, a)
        qc_df.loc[len(qc_df)] = (copy, "All leads.xlsx" if a in differs or b in differs else "*",
                                 "ecg-duplicate-of-other-subject", f"ECG identical to {orig}")
    qc_df = qc_df.drop_duplicates().reset_index(drop=True)
    if subject_map is not None:
        m = pd.read_csv(subject_map).set_index("subject")
        subj_df["batch"] = subj_df.subject.map(m["batch"])
        subj_df["previous_id"] = subj_df.subject.map(m["previous_id"])

    # within-subject duplicated cycles (identical waveform under different cycle numbers)
    for s, g in cyc_df.groupby("subject"):
        hs = {c: _hash(gg.force_N.values) for c, gg in g.groupby("cycle")}
        inv = {}
        for c, h in hs.items():
            inv.setdefault(h, []).append(c)
        for h, cs in inv.items():
            if len(cs) > 1:
                qc_df.loc[len(qc_df)] = (s, "*", "duplicate-cycle-within-subject", f"cycles {cs} identical")

    # sensor windows identical to a window of another subject
    owner = {}
    for (s, c), g in cyc_df.groupby(["subject", "cycle"], sort=False):
        h = _hash(g.force_N.values)
        if h in owner and owner[h][0] != s:
            qc_df.loc[len(qc_df)] = (s, "*", "jvp-duplicate-of-other-subject", f"cycle {c} identical to {owner[h][0]} cycle {owner[h][1]}")
        owner.setdefault(h, (s, c))

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
        fh.write(f"* Recording batches (raw directories): "
                 f"{subj_df.groupby(['batch', 'label']).size().to_dict()}\n")
        fh.write(f"* JVP cycles (content-unique): {cyc_df.groupby(['subject','cycle']).ngroups}; "
                 f"samples/cycle median {cyc_df.groupby(['subject','cycle']).size().median():.0f} "
                 f"(range {cyc_df.groupby(['subject','cycle']).size().min()}-{cyc_df.groupby(['subject','cycle']).size().max()})\n")
        fh.write(f"* |r(current, force)| median {cal.r_current_force.abs().median():.6f}, min {cal.r_current_force.abs().min():.6f} "
                 f"-> force is a per-recording linear calibration of current\n")
        fh.write(f"* Subjects whose ECG is byte-identical to another subject: {ecg_dup_subjects}\n\n")
        fh.write("## Issues\n\n" + qc_df.to_markdown(index=False) + "\n")
    print(open(out_dir / "qc_report.md").read()[:3000])


if __name__ == "__main__":
    args = sys.argv[1:]
    smap = None
    if args[:1] == ["--map"]:
        smap, args = args[1], args[2:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    main(args[:-1], args[-1], smap)
