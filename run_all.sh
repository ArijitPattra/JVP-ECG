#!/usr/bin/env bash
# Full pipeline. Usage: ./run_all.sh "/path/to/raw subject folders (HS1 ... HS54)"
# Runtime about 1-1.5 h on an 8-core laptop CPU (CNN models and permutation tests dominate).
set -euo pipefail
RAW="${1:?give the folder that contains the HS* subject folders}"
cd "$(dirname "$0")/jvp_ml"
python load_data.py "$RAW" data                                     # parse + QC   -> data/
python run_benchmark.py --repeats 30 --cnn-repeats 10 --perms 200    # main benchmark -> results/
python robustness.py --repeats 20                                    # sensitivity, noise, alignment test
python cnn_permutation.py --model cnn --perms 100                    # permutation test, primary 1D-CNN (reduced config)
python cnn_permutation.py --model l1lr --perms 200                   # permutation test, sparse logistic regression
python make_figures.py                                               # figures/ + tables
python ecg_investigation.py                                          # ECG fidelity checks
python imbalance.py --repeats 20 --cnn-repeats 6                     # class-imbalance analysis
python fill_manuscript.py --out ../overleaf_build                    # Overleaf-ready manuscript
python deploy.py train                                               # sensor-only deployment model (local only)
