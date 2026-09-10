# Race prediction from T-PPG features

Code accompanying the ML4H Findings submission. Reproduces the main SVM
race-classification model, the confounder-control analyses, ABI-matched analysis,
the forest plot, and the appendix figures (mean ROC curve and pooled confusion matrix).
Because the underlying clinical dataset and proprietary preprocessing/featurization
pipeline cannot be distributed, the script requires the analysis-ready variables described below.

## Contents
- `analysis_code.py` — analysis pipeline.

## Requirements
Python 3.9+ with `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

## Data
The dataset is not publicly available due to institutional and patient
privacy policies. Preprocessing (signal QC, featurization, cohort
filtering) is therefore omitted. The script assumes the following
analysis-ready variables are available:

- `X`      : (n_signals, 78) DataFrame of T-PPG features
- `y`      : (n_signals,) Series, 1 = Black, 0 = White
- `groups` : (n_signals,) Series of patient IDs
- `sites`  : (n_signals,) Series in {"Site 1", "Site 2"}
- `abis`   : (n_signals,) Series of ABI values

## Running
Load or define the variables above before executing `analysis_code.py`.
The script prints pooled and per-fold AUROCs, a summary table across the
confounder-control analyses, and displays the three figures.
