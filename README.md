# Wearable rGO/CuO JVP sensor + machine learning, benchmarked against ECG

Analysis code for the manuscript *"A wearable rGO/CuO piezoresistive jugular venous pulse sensor with
physiology-informed machine learning for screening heart disease, benchmarked against the electrocardiogram"*
(Pattra, Chatterjee, Dey; School of Electrical and Computer Sciences, IIT Bhubaneswar).

The pipeline analyses jugular venous pulse (JVP) windows recorded by a neck-mounted piezoresistive sensor.

**Intended use: the sensor alone.** At deployment a person is screened from sensor readings only. ECG leads V1/V2 are
used solely to build comparator models that benchmark conventional electrical screening in the same participants.
No model combines the two modalities, and the ECG is never used to train, align or select the sensor models.

**All sensor analyses are start-agnostic.** The position of each recorded window within the cardiac cycle is
undocumented, so every sensor feature and model is invariant to a circular shift of the window:

- trough-anchored *x → v → y → a → c* fiducials;
- Fourier magnitudes and relative phases;
- circular ROCKET;
- a circular-padded 1D-CNN trained on random shifts and averaged over shifts at test time.

ECG timing is never used to locate JVP waves.

> **Status: feasibility study.** 52 participants, of whom 6 are diseased. See [Interpreting the results](#interpreting-the-results)
> before quoting any number.

## Repository layout

```
run_all.sh                 one-command pipeline
requirements.txt
jvp_ml/
  load_data.py             parse the Excel workbooks, de-duplicate cycles, repair ECG time units, QC report
  features.py              start-agnostic JVP features (trough anchoring, Fourier descriptors; 33 features); ECG V1/V2
  models.py                v/c rule, LR/L1-LR/SVM/RF/GBM (nested tuning, in-fold thresholds), circular ROCKET,
                           circular 1D-CNN (primary sensor model), physiology-hybrid CNN
  evaluation.py            repeated stratified subject-level CV, bootstrap CIs, permutation test, DeLong test
  run_benchmark.py         main benchmark: 11 sensor models and 5 ECG comparator models
  robustness.py            QC-exclusion, current-vs-force, single-window, noise and decimation, alignment test
  cnn_permutation.py       permutation tests for the 1D-CNN (reduced configuration) and sparse logistic regression
  ecg_investigation.py     ECG fidelity checks; crude raw-sample v/c vs diagnosis
  imbalance.py             class-imbalance strategies, operating points, PPV/NPV vs prevalence, sample size
  make_figures.py          figures and tables (coherent bootstrap CIs)
  deploy.py                sensor-only deployment: train the primary 1D-CNN on all participants; predict from sensor readings
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

**Participant data are not included** in this repository. `load_data.py` expects one folder per participant
(`HS1 … HS54`), each with workbooks named `JVP_<V1|V2|V1_V2>_Cycle<k>.xlsx`. A workbook holds:

- one manually segmented JVP cycle: time (s), current (A), force (N);
- one ECG beat: time (s), amplitude (mV) for the lead(s) in the file name;
- a subject-level label, `Diseased` or `Not Diseased`.

Column positions vary between workbooks and are located from the header text. Data availability:
*[to be completed by the authors]*.

## Running

```bash
./run_all.sh "/path/to/Arijit JVP Sensor1_Unknown Subjects"
```

All random seeds are fixed. The run takes about 1–1.5 h on an 8-core CPU. Every number in the manuscript is written by
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

- `youden`: cross-validated sensitivity 1.00, specificity 0.65;
- `spec90`: sensitivity 0.50, specificity 0.91.

The model was developed on 52 participants (6 diseased) without external validation, and is not a medical device.
Trained weights are not distributed, because they are derived from participant data; run `train` with your own data
access.

## Evaluation protocol

- **Subject-level splits only.** Cycles from one participant are never in both the training and the test fold.
- **Repeated stratified 3-fold CV.** With 6 positives this gives 2 per test fold. There are 30 repeats for fast models and 10 for CNN, ROCKET and tree models.
- **Nested tuning.** Hyper-parameters (inner grid search) and decision thresholds (Youden index) are chosen inside each training fold.
- **Statistics.** 95% CIs come from 2,000 stratified participant bootstraps of the repeat-averaged AUROC. The permutation test re-runs the whole CV on 200 label permutations. Paired DeLong tests compare models.

## Main results (`jvp_ml/results/table_benchmark.csv`)

| Model | AUROC (95% CI) |
|---|---|
| **JVP sensor: start-agnostic 1D-CNN** (best sensor model, selected among 11) | **0.839 (0.71–0.95)** |
| JVP sensor: sparse (L1) logistic regression, 33 features | 0.824 (0.69–0.95), permutation P = 0.010 |
| JVP sensor: physiology-hybrid CNN (waveform + *v/c*, *a→c* slope) | 0.808 (0.61–0.96) |
| JVP sensor: rule *v/c* > 1, no training | 0.707 (0.47–0.91) |
| **ECG V1/V2: logistic regression** (best ECG model) | **0.870 (0.76–0.96)**, permutation P = 0.005 |

- **Sensor vs ECG.** The sensor CNN and the best ECG model do not differ significantly (DeLong P = 0.68).
- **CNN permutation test.** It had to use a reduced configuration and gave P = 0.099, not significant. The similarly performing sparse model is significant (P = 0.010).
- **Hypotheses.** The hypothesised *v/c* ratio is weakly discriminative (AUROC 0.71, P = 0.11). An exploratory waveform-asymmetry descriptor (third-harmonic relative phase) gives AUROC 0.95.

## Interpreting the results

Labels are physician diagnoses, independently verified.

1. **Window alignment matters.** Logistic regression on the raw window reaches AUROC 0.970 on the stored windows but
   0.561 after random circular shifts. Shift-invariant Fourier descriptors are unaffected. Positional information
   can only be used if windows are extracted by a documented, sensor-only rule, so the primary analyses are
   start-agnostic.
2. **The ECG beats are parametric reconstructions** (six-Gaussian fit, median R² = 0.9999) with no AV-conduction
   information (PR < 120 ms in 51 of 52 participants). There are two leads and one beat per participant, so the ECG
   benchmark is not a clinical 12-lead ECG.
3. **The pilot is small.** There are 6 diseased participants and the sensor model was selected post hoc, so all estimates are imprecise and
   optimistic. About 94 diseased participants are needed to estimate an AUROC of 0.85 within ±0.05.

## Class imbalance (6 diseased vs 46 non-diseased)

`imbalance.py` compares correction strategies, always applied inside the training folds: none, class weighting,
SMOTE, undersampling ensemble, and focal loss / oversampling for the CNN. It reports AUROC, AUPRC (random = 0.115),
MCC, sensitivity/specificity at 0.5 and at in-fold thresholds, and calibration-in-the-large. It also reports PPV/NPV
at screening prevalences and the sample size a validation study needs.

The corrections mainly change the operating point and calibration. They inflate predicted risk by 0.09–0.38, and
for the sensor feature models resampling slightly lowers AUROC. Choosing the threshold inside the training folds
achieves the same operating point without any correction.

## Data-quality issues handled by `load_data.py`

| Issue | Participants | Handling |
|---|---|---|
| ECG identical to another participant's | HS42, HS43, HS48, HS52 | Flagged; excluded in sensitivity analysis |
| Workbook header names another participant | HS31, HS43, HS48 | Flagged; excluded in sensitivity analysis |
| ECG time stored in the wrong unit | HS3, HS21, HS28 | Repaired |
| Single-lead header names the wrong lead | HS1, HS2, HS37 | Lead taken from the file name |
| Same cycle saved under two cycle numbers | HS15, HS18, HS43, HS45 | Counted once |

Force is a per-recording linear calibration of current (|r| ≈ 0.99998), so the two channels are redundant.

## License

The code is released under the [MIT License](LICENSE). The license covers the code, figures and aggregate results
in this repository. It does not cover participant data, which are not distributed here.

## Citation

If you use this code, please cite the accompanying manuscript (citation to be added on publication).
