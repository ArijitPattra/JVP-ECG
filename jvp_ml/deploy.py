"""Sensor-only deployment of the primary model (start-agnostic 1D-CNN).

The deployed system uses ONLY the rGO/CuO sensor readings; ECG is never required.

  python deploy.py train   [--out deploy_model]
      Trains the 1D-CNN (3 seeds, 120 epochs, positive-weighted loss) on the sensor windows of all participants
      and stores the weights together with operating thresholds derived from the cross-validated (out-of-fold)
      scores in results/oof_all.pkl. Thresholds from cross-validation are used because thresholds chosen on the
      training data itself would be optimistic.

  python deploy.py predict --model deploy_model --input <path> [--channel force|current] [--out predictions.csv]
      <path> is either
        * a CSV with columns participant, window, value (one row per sensor sample, in time order), or
        * a folder of participant sub-folders with JVP_*_Cycle*.xlsx workbooks (only the sensor columns are read).
      Prints and writes one row per participant: number of windows, probability, screen result at each threshold.

The model was developed on 52 participants (6 diseased) and has not been externally validated: outputs are for
research use only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features import jvp_cycle_features
from models import CNN1D, CNNModel, two_channel

HERE = Path(__file__).parent
SEEDS, EPOCHS = (0, 1, 2), 120
CV_KEY = "JVP | 1D-CNN waveform"


def sensor_windows_from_training_data(channel="force_N"):
    """Training windows: sensor readings only (no time stamps, no ECG)."""
    cyc = pd.read_csv(HERE / "data/jvp_cycles.csv", dtype={"cycle": str})
    sub = pd.read_csv(HERE / "data/subjects.csv")
    waves = []
    for (s, c), g in cyc.groupby(["subject", "cycle"], sort=False):
        _, r, _ = jvp_cycle_features(g[channel].values)
        waves.append((s, c, r))
    return sub, waves


def thresholds_from_cv(y):
    """Operating thresholds on repeat-averaged out-of-fold CNN scores."""
    p = pd.read_pickle(HERE / "results/oof_all.pkl")[CV_KEY].mean(0)
    cand = np.unique(p)
    youden = cand[int(np.argmax([(p[y == 1] >= t).mean() + (p[y == 0] < t).mean() for t in cand]))]
    neg = np.sort(p[y == 0])
    spec90 = neg[int(np.ceil(0.90 * len(neg))) - 1] + 1e-9
    out = {}
    for name, t in (("youden", youden), ("spec90", spec90)):
        out[name] = dict(threshold=float(t), cv_sensitivity=float((p[y == 1] >= t).mean()),
                         cv_specificity=float((p[y == 0] < t).mean()))
    return out


def train(out):
    out = Path(out); out.mkdir(exist_ok=True)
    sub, waves = sensor_windows_from_training_data()
    m = CNNModel(waves, epochs=EPOCHS, seeds=SEEDS).fit(sub.subject.values, sub.y.values)
    for i, net in enumerate(m.nets):
        torch.save(net.state_dict(), out / f"cnn_seed{i}.pt")
    meta = dict(model="start-agnostic 1D-CNN (circular padding, test-time shift averaging)", inputs="sensor readings only",
                n_participants=int(len(sub)), n_diseased=int(sub.y.sum()), n_windows=len(waves), epochs=EPOCHS,
                seeds=list(SEEDS), window_points=int(len(waves[0][2])), thresholds=thresholds_from_cv(sub.y.values),
                note="Research use only; developed on 52 participants without external validation.")
    json.dump(meta, open(out / "model.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


def read_input(path, channel):
    path = Path(path)
    col = {"force": "jvp_F", "current": "jvp_I"}[channel]
    windows = {}
    if path.is_file() and path.suffix.lower() == ".csv":
        d = pd.read_csv(path)
        for (p, w), g in d.groupby(["participant", "window"], sort=False):
            windows.setdefault(str(p), []).append(g["value"].to_numpy(float))
        return windows
    from load_data import _hash, read_workbook
    for sd in sorted(q for q in path.iterdir() if q.is_dir()):
        seen = set()
        for f in sorted(sd.glob("JVP_*_Cycle*.xlsx")):
            x = np.asarray(read_workbook(f)[col], float)  # sensor column only
            h = _hash(x)
            if h not in seen:
                seen.add(h); windows.setdefault(sd.name, []).append(x)
    return windows


@torch.no_grad()
def predict(model_dir, inp, channel, out):
    model_dir = Path(model_dir); meta = json.load(open(model_dir / "model.json"))
    nets = []
    for i in range(len(meta["seeds"])):
        net = CNN1D(2, 16, 0, circular=True); net.load_state_dict(torch.load(model_dir / f"cnn_seed{i}.pt")); net.eval()
        nets.append(net)
    rows = []
    for pid, ws in read_input(inp, channel).items():
        R = np.array([jvp_cycle_features(w)[1] for w in ws if len(w) >= 4], dtype=np.float32)
        L = R.shape[1]
        ps = [torch.sigmoid(n(torch.from_numpy(two_channel(np.roll(R, sh, axis=1))))).numpy()
              for sh in range(0, L, L // 8) for n in nets]
        prob = float(np.mean(ps))
        row = dict(participant=pid, n_windows=len(R), probability=round(prob, 4))
        for name, t in meta["thresholds"].items():
            row[f"screen_positive_{name}"] = bool(prob >= t["threshold"])
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(out, index=False)
    print(df.to_string(index=False)); print(f"\nwrote {out}  (research use only; see {model_dir / 'model.json'})")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--out", default=str(HERE / "deploy_model"))
    p = sp.add_parser("predict"); p.add_argument("--model", default=str(HERE / "deploy_model"))
    p.add_argument("--input", required=True); p.add_argument("--channel", default="force", choices=["force", "current"])
    p.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()
    train(a.out) if a.cmd == "train" else predict(a.model, a.input, a.channel, a.out)


if __name__ == "__main__":
    main()
