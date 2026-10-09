"""Permutation tests for the best sensor models.

--model cnn  : start-agnostic 1D-CNN (primary sensor model), reduced configuration (see below).
--model l1lr : sparse logistic regression, full configuration (3 CV repeats per permutation, as for the
               other tabular models in run_benchmark.py).


The full CNN configuration (10 CV repeats x 3 seeds) is too costly to re-run on hundreds of permuted label
sets, so the observed statistic and the null distribution are computed with the same reduced configuration:
one repeated-CV pass (stratified 3-fold), one seed, 120 epochs (Ojala & Garriga, 2010, test 1).
Output: results/cnn_permutation.json or results/l1lr_permutation.json
Usage: python cnn_permutation.py [--model cnn|l1lr] [--perms 100]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from evaluation import permutation_pvalue
from models import CNNModel, TabularModel
from run_benchmark import JVP_FEATS, load

HERE = Path(__file__).parent


def cv_auc(waves, S, y, seed, epochs):
    P = np.zeros(len(y))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(S, y):
        P[te] = CNNModel(waves, epochs=epochs, seeds=(seed,)).fit(S[tr], y[tr]).predict_proba(S[te])
    return roc_auc_score(y, P)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--model", default="cnn", choices=["cnn", "l1lr"])
    a = ap.parse_args()
    sub, J, E, waves, _ = load(); S, y = sub.subject.values, sub.y.values
    if a.model == "l1lr":
        import pandas as pd
        P = pd.read_pickle(HERE / "results/oof_all.pkl")["JVP | L1-LR physiology features (embedded selection)"]
        obs = float(np.mean([roc_auc_score(y, p) for p in P]))
        p, null = permutation_pvalue(lambda s: TabularModel("L1LR", J[JVP_FEATS], s), S, y, obs, n_perm=a.perms)
        out = dict(observed_auroc=obs, n_perm=a.perms, p=float(p), null_mean=float(null.mean()), null_95=float(np.percentile(null, 95)))
        json.dump(out, open(HERE / "results/l1lr_permutation.json", "w"), indent=2)
        return print(json.dumps(out, indent=2))
    obs = cv_auc(waves, S, y, 0, a.epochs)
    print(f"observed AUROC (reduced configuration) {obs:.3f}", flush=True)
    rng = np.random.default_rng(0); null = []
    for i in range(a.perms):
        null.append(cv_auc(waves, S, rng.permutation(y), 1000 + i, a.epochs))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{a.perms} permutations, null mean {np.mean(null):.3f}", flush=True)
    null = np.array(null)
    out = dict(observed_auroc_reduced=float(obs), n_perm=a.perms, p=float((1 + np.sum(null >= obs)) / (1 + a.perms)),
               null_mean=float(null.mean()), null_95=float(np.percentile(null, 95)))
    json.dump(out, open(HERE / "results/cnn_permutation.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
