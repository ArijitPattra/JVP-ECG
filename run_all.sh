#!/usr/bin/env bash
# Full pipeline. Usage: ./run_all.sh "/path/to/current delivery" ["/path/to/earlier batch 1" "/path/to/earlier batch 2" ...]
# The first argument is the folder of HS* subject folders with sensor workbooks and the 12-lead "All leads.xlsx".
# Optional further arguments are earlier deliveries; participants are matched to them by sensor-window content to
# recover the recording batch (needed only for the batch analyses).
# Runtime about 1-1.5 h on an 8-core laptop CPU (CNN models and permutation tests dominate).
set -euo pipefail
[ $# -ge 1 ] || { echo "give the folder(s) that contain the HS* subject folders"; exit 1; }
RAW=("$@")
for i in "${!RAW[@]}"; do RAW[$i]="$(cd "${RAW[$i]}" && pwd)"; done
cd "$(dirname "$0")/jvp_ml"
mkdir -p data
if [ ${#RAW[@]} -gt 1 ]; then
  python map_subjects.py "${RAW[0]}" "${RAW[@]:1}" data/subject_map.csv  # subject IDs -> recording batch
  python load_data.py --map data/subject_map.csv "${RAW[0]}" data      # parse + QC   -> data/
else
  python load_data.py "${RAW[0]}" data
fi
python run_benchmark.py --repeats 30 --cnn-repeats 10 --perms 200    # main benchmark -> results/
python robustness.py --repeats 20                                    # sensitivity, noise, alignment test
python cnn_permutation.py --model cnn --perms 100                    # permutation test, primary 1D-CNN (reduced config)
python make_figures.py                                               # figures/ + tables
python ecg_investigation.py                                          # ECG fidelity checks
python imbalance.py --repeats 20 --cnn-repeats 6                     # class-imbalance analysis
# Temporal check of a model frozen before the later batch arrived (set FROZEN to that earlier deploy_model/):
if [ -n "${FROZEN:-}" ]; then python temporal_validation.py --frozen "$FROZEN" --input "${RAW[0]}"; fi
python fill_manuscript.py --out ../overleaf_build                    # Overleaf-ready manuscript
python deploy.py train                                               # sensor-only deployment model (local only)
