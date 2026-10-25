#!/usr/bin/env bash
# Full pipeline. Usage: ./run_all.sh "/path/to/batch 1" ["/path/to/batch 2" ...]
# Each argument is a folder of HS* subject folders (one recording batch); subject IDs must be unique across batches.
# Runtime about 1-1.5 h on an 8-core laptop CPU (CNN models and permutation tests dominate).
set -euo pipefail
[ $# -ge 1 ] || { echo "give the folder(s) that contain the HS* subject folders"; exit 1; }
RAW=("$@")
for i in "${!RAW[@]}"; do RAW[$i]="$(cd "${RAW[$i]}" && pwd)"; done
cd "$(dirname "$0")/jvp_ml"
python load_data.py "${RAW[@]}" data                                 # parse + QC   -> data/
python run_benchmark.py --repeats 30 --cnn-repeats 10 --perms 200    # main benchmark -> results/
python robustness.py --repeats 20                                    # sensitivity, noise, alignment test
python cnn_permutation.py --model cnn --perms 100                    # permutation test, primary 1D-CNN (reduced config)
python cnn_permutation.py --model l1lr --perms 200                   # permutation test, sparse logistic regression
python make_figures.py                                               # figures/ + tables
python ecg_investigation.py                                          # ECG fidelity checks
python imbalance.py --repeats 20 --cnn-repeats 6                     # class-imbalance analysis
# Temporal check of a model frozen before the later batch arrived (set FROZEN to that earlier deploy_model/ and LATER to the batch):
if [ -n "${FROZEN:-}" ]; then python temporal_validation.py --frozen "$FROZEN" --input "$LATER"; fi
python fill_manuscript.py --out ../overleaf_build                    # Overleaf-ready manuscript
python deploy.py train                                               # sensor-only deployment model (local only)
