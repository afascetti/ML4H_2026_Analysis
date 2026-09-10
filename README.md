# Race prediction from T-PPG features

Code accompanying the ML4H Findings submission. Reproduces the main SVM
race-classification model, the confounder-control analyses, and the
appendix figures (mean ROC curve, pooled confusion matrix, forest plot).

## Contents
- `analysis_code.py` — full analysis pipeline.

## Requirements
Python 3.9+ with `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

## Data
The dataset is not publicly available due to institutional and patient
privacy policies. Preprocessing (signal QC, featurization, cohort
filtering) is therefore omitted. The script assumes the following
variables are already defined in scope:

- `X`      : (n_signals, 78) DataFrame of T-PPG features
- `y`      : (n_signals,) Series, 1 = Black, 0 = White
- `groups` : (n_signals,) Series of patient IDs
- `sites`  : (n_signals,) Series in {"Site 1", "Site 2"}
- `abis`   : (n_signals,) Series of ABI values

## Running
Provide the variables above, then run `python race_prediction_appendix.py`.
Prints pooled and per-fold AUCs, a summary table across all confounder
controls, and displays the three appendix figures.
