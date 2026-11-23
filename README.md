# Wearable rGO/CuO JVP sensor + machine learning, compared with the 12-lead ECG

Analysis code for the manuscript *"A wearable rGO/CuO piezoresistive jugular venous pulse sensor and start-agnostic machine learning for
heart-disease screening: a pilot study with a 12-lead electrocardiogram comparator"*
(Pattra, Chatterjee, Dey; School of Electrical and Computer Sciences, IIT Bhubaneswar).

The pipeline analyses jugular venous pulse (JVP) windows recorded by a neck-mounted piezoresistive sensor.

**Intended use: the sensor alone.** At deployment a person is screened from sensor readings only. The 12-lead ECG is
used solely to build comparator models that benchmark conventional electrical screening in the same participants.
No model combines the two modalities, and the ECG is never used to train, align or select the sensor models.

**All sensor analyses are start-agnostic.** The position of each recorded window within the cardiac cycle is
undocumented, so every sensor feature and model is invariant to a circular shift of the window:

- trough-anchored *x → v → y → a → c* fiducials;
- Fourier magnitudes and relative phases;
- circular ROCKET;
- a circular-padded 1D-CNN trained on random shifts and averaged over shifts at test time.

ECG timing is never used to locate JVP waves.

> **Status: feasibility study.** 58 participants, of whom 12 are diseased. Six of the diseased participants come from a later
> recording batch without controls, which is a potential confound. See [Interpreting the results](#interpreting-the-results)
> before quoting any number.

## Repository layout

```
run_all.sh                 one-command pipeline
requirements.txt
jvp_ml/
  load_data.py             parse the Excel workbooks, de-duplicate cycles, repair ECG time units, QC report
  features.py              start-agnostic JVP features (trough anchoring, Fourier descriptors; 33 features); 12-lead ECG features (axis, Sokolow-Lyon, RVH/LVH criteria)
  models.py                v/c rule, LR/L1-LR/SVM/RF/GBM (nested tuning, in-fold thresholds), circular ROCKET,
                           circular 1D-CNN (primary sensor model), physiology-hybrid CNN
  evaluation.py            repeated stratified subject-level CV, bootstrap CIs, permutation test, DeLong test
  run_benchmark.py         main benchmark: 11 sensor models, 6 12-lead ECG and 2 V1/V2 comparator models
  robustness.py            QC-exclusion, current-vs-force, recording-batch, single-window, noise and decimation, alignment test
  cnn_permutation.py       permutation test for the 1D-CNN (reduced configuration)
  ecg_investigation.py     ECG fidelity checks; crude raw-sample v/c vs diagnosis
  imbalance.py             class-imbalance strategies, operating points, PPV/NPV vs prevalence, sample size
  make_figures.py          figures and tables (coherent bootstrap CIs)
  deploy.py                sensor-only deployment: train the primary 1D-CNN on all participants; predict from sensor readings
  temporal_validation.py   frozen first-batch model applied to a later batch; per-batch AUROC; batch and sampling-rate checks
  fill_manuscript.py       fills every number into manuscript/manuscript_template.tex -> Overleaf folder
  manuscript/              LaTeX template, bibliography (verified DOIs), naturemag.bst
  figures/                 all figures (PDF + PNG)
  results/                 aggregate results only (no per-participant outputs)
```

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

XGBoost is deliberately not used. Next to PyTorch it segfaults or deadlocks on macOS because of an OpenMP clash, so
scikit-learn's `GradientBoostingClassifier` is the boosting baseline.

## Data

**Participant data are not included** in this repository. `load_data.py` expects one or more batch folders, each
holding one folder per participant (`HS1 … HS54` in the first batch, `HS55 … HS60` in the second; IDs must be unique
across batches). Each participant folder has workbooks named `JVP_<V1|V2|V1_V2>_Cycle<k>.xlsx`. A workbook holds:

- one manually segmented JVP cycle: time (s), current (A), force (N);
- one ECG beat: time (s), amplitude (mV) for the lead(s) in the file name;
- a subject-level label, `Diseased` or `Not Diseased`.

Column positions vary between workbooks and are located from the header text. Data availability:
*[to be completed by the authors]*.

## Running

```bash
./run_all.sh "/path/to/batch 1" "/path/to/batch 2"
```

To repeat the temporal check, also set `FROZEN` (a `deploy_model/` trained on the earlier batch before the later one
arrived) and `LATER` (the later batch folder). All random seeds are fixed. The run takes about 1–1.5 h on an 8-core CPU. Every number in the manuscript is written by
`fill_manuscript.py` from `results/`, so edit `manuscript/manuscript_template.tex`, never the generated `.tex`.

## Sensor-only deployment (research use)

```bash
cd jvp_ml
python deploy.py train                      # -> deploy_model/ (weights + model.json with operating thresholds)
python deploy.py predict --input sensor_windows.csv --out predictions.csv
```

`sensor_windows.csv` has the columns `participant, window, value`, with one row per sensor sample (force by default;
`--channel current` for raw current). A folder of the study workbooks also works, and only their sensor columns are
read. The output gives one probability per participant and a screen result at two operating thresholds taken from the
cross-validated scores:

- `youden`: cross-validated sensitivity 0.83, specificity 0.91;
- `spec90`: sensitivity 0.83, specificity 0.91.

The model was developed on 58 participants (12 diseased) without external validation, and is not a medical device.
Trained weights are not distributed, because they are derived from participant data; run `train` with your own data
access.

## Evaluation protocol

- **Subject-level splits only.** Cycles from one participant are never in both the training and the test fold.
- **Repeated stratified 3-fold CV.** With 12 positives this gives 4 per test fold. There are 30 repeats for fast models and 10 for CNN, ROCKET and tree models.
- **Nested tuning.** Hyper-parameters (inner grid search) and decision thresholds (Youden index) are chosen inside each training fold.
- **Statistics.** 95% CIs come from 2,000 stratified participant bootstraps of the repeat-averaged AUROC. The permutation test re-runs the whole CV on 200 label permutations. Paired DeLong tests compare models.

## Main results (`jvp_ml/results/table_benchmark.csv`)

| Model | AUROC (95% CI) |
|---|---|
| **JVP sensor: start-agnostic 1D-CNN** (pre-specified primary model; chosen on the first batch) | **0.908 (0.80–0.98)**, permutation P = 0.010 (reduced configuration) |
| JVP sensor: physiology-hybrid CNN (waveform + *v/c*, *a→c* slope) | 0.905 (0.78–0.99) |
| JVP sensor: sparse (L1) logistic regression, 33 features | 0.900 (0.82–0.96), permutation P = 0.005 |
| JVP sensor: circular ROCKET | 0.895 (0.78–0.98) |
| JVP sensor: rule *v/c* > 1, no training | 0.741 (0.58–0.88) |
| **ECG 12-lead: 1D-CNN on the beat** (best ECG model) | **0.953 (0.90–0.99)** |
| ECG 12-lead: random forest / logistic regression (reference) | 0.893 (0.74–0.98) / 0.872 (0.74–0.96), permutation P = 0.005 |
| ECG V1/V2 only: random forest / logistic regression | 0.902 (0.82–0.97) / 0.872 (0.78–0.94) |

- **Sensor vs ECG.** The sensor CNN does not differ significantly from the 12-lead ECG CNN (DeLong P = 0.25) or from
  12-lead ECG logistic regression (P = 0.88). On the first batch alone the sensor CNN gives 0.79 and 12-lead ECG
  logistic regression 0.79.
- **Frozen-model check.** The CNN trained on the first 52 participants, frozen before the later batch arrived, flagged
  6 of 6 later patients at its Youden threshold (95% CI 0.54–1.00) and 4 of 6 at its 90%-specificity threshold. The
  later batch has no controls, so this measures sensitivity only.
- **Hypotheses.** The hypothesised *v/c* ratio is modestly discriminative (AUROC 0.74, P = 0.011). An exploratory
  waveform-asymmetry descriptor (third-harmonic relative phase) is the strongest single sensor feature.

## Interpreting the results

Labels are physician diagnoses, independently verified.

1. **The recording batch is a potential confound.** All six later-batch participants (HS53–HS58 in the current
   numbering) are diseased, and their windows were sampled more sparsely (median 8.7 vs about 12 samples/s). Both the
   sensor and the 12-lead ECG models separate the later patients from controls much better than the first-batch
   patients (sensor CNN 0.99 vs 0.84; ECG logistic regression 0.99 vs 0.82). Treat the first-batch estimates as the
   conservative ones.
2. **Window alignment matters.** Logistic regression on the raw window reaches AUROC 0.990 on the stored windows but
   0.467 after random circular shifts. Shift-invariant Fourier descriptors are unaffected, so the primary analyses are
   start-agnostic.
3. **The 12-lead ECG beats are parametric reconstructions** (six-Gaussian fit, median R² = 0.9999), the limb leads do
   not obey Einthoven's law (median deviation 40% of lead II), PR is below 120 ms in every participant, and the
   beats of diseased participants are far more alike than those of controls (median between-participant r 0.74 vs
   0.32). The ECG comparison is therefore not a comparison with a recorded clinical 12-lead ECG and may be optimistic.
4. **The pilot is small.** There are 12 diseased participants and the primary sensor model was selected post hoc on the
   first batch, so all estimates are imprecise and optimistic. About 97 diseased participants are needed to estimate
   an AUROC of 0.85 within ±0.05.

## Class imbalance (12 diseased vs 46 non-diseased)

`imbalance.py` compares correction strategies, always applied inside the training folds: none, class weighting,
SMOTE, undersampling ensemble, and focal loss / oversampling for the CNN. It reports AUROC, AUPRC (random = 0.207),
MCC, sensitivity/specificity at 0.5 and at in-fold thresholds, and calibration-in-the-large. It also reports PPV/NPV
at screening prevalences and the sample size a validation study needs.

The corrections mainly change the operating point and calibration. They inflate predicted risk by 0.12–0.28, and
they neither consistently improve nor harm the sensor models' AUROC. Choosing the threshold inside the training folds
achieves a similar operating point without any correction.

## Data-quality issues handled by `load_data.py`

| Issue | Participants (current numbering) | Handling |
|---|---|---|
| 12-lead ECG identical to another participant's | HS1, HS35, HS42, HS43, HS48, HS52 | Flagged; excluded in sensitivity analysis |
| 12-lead V1/V2 differ from the single-lead workbooks | HS1, HS2, HS3, HS7, HS35, HS45 | 12-lead file used |
| 12-lead time stored in ms | HS41, HS42, HS50 | Converted to s |
| Same cycle saved under two cycle numbers | HS15, HS18, HS43, HS45 | Counted once |
| Participants renumbered between deliveries | HS9, HS10, HS14, HS31, HS32, HS53, HS54 | Matched by sensor-window content (`map_subjects.py`) |

Force is a per-recording linear calibration of current (|r| ≈ 0.99998), so the two channels are redundant.

## License

The code is released under the [MIT License](LICENSE). The license covers the code, figures and aggregate results
in this repository. It does not cover participant data, which are not distributed here.

## Citation

If you use this code, please cite the accompanying manuscript (citation to be added on publication).
