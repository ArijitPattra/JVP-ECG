"""Map the subject IDs of the current data delivery to earlier deliveries and assign recording batches.

The 12-lead delivery (`Arijit JVP Sensor1`, November 2026) renumbered the participants HS1-HS58 (earlier
deliveries: HS1-HS54 with HS10/HS14 absent, plus a later batch HS55-HS60). Participants are matched by the
content hash of their sensor windows, which are unchanged between deliveries. A participant inherits the
recording batch of the earlier directory in which their sensor windows were found.

Usage: python map_subjects.py <current_dir> <earlier_dir_batch1> [<earlier_dir_batch2> ...] <out_csv>
"""
import sys
from pathlib import Path

import pandas as pd

from load_data import _hash, read_workbook


def window_hashes(d):
    return {_hash(read_workbook(f)["jvp_I"]) for f in sorted(Path(d).glob("JVP_*_Cycle*.xlsx"))}


def main(cur, earlier, out_csv):
    old = {}
    for b, e in enumerate(earlier, 1):
        for d in Path(e).iterdir():
            if d.is_dir() and d.name.startswith("HS"):
                old[(b, d.name)] = window_hashes(d)
    rows = []
    for d in sorted([p for p in Path(cur).iterdir() if p.is_dir() and p.name.startswith("HS")], key=lambda p: int(p.name[2:])):
        h = window_hashes(d)
        best = max(old, key=lambda k: len(h & old[k]))
        n = len(h & old[best])
        rows.append(dict(subject=d.name, previous_id=best[1] if n else "", batch=f"batch {best[0]}" if n else "unmatched",
                         windows_matched=n, windows=len(h)))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    ren = df[df.subject != df.previous_id]
    print(f"{len(df)} subjects; batches {df.batch.value_counts().to_dict()}; renumbered:\n{ren.to_string(index=False)}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:-1], sys.argv[-1])
