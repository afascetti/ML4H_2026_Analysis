# Race prediction from T-PPG features

Code accompanying the ML4H 2026 Findings paper "Assessing Race-Associated Information in Toe Photoplethysmography Features".

Reproduces the main SVM race-classification model, confounder controls, ABI matching, site adjusted analysis, nonlinear gradient boosting analysis, and feature level analyses with ABI and clinical covariate adjustment. The script also generates the forest plot and appendix figures (mean ROC curve and pooled confusion matrix).

Because the underlying clinical dataset and proprietary preprocessing and featurization
pipeline cannot be distributed, the script requires the analysis-ready variables described below.

## Contents
- `analysis_code.py` - analysis pipeline.

## Requirements
Python 3.9+ with `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`.

## Data
The dataset is not publicly available due to institutional and patient
privacy policies. Preprocessing (signal quality control, featurization, and cohort
filtering) is therefore omitted. The script assumes the following
analysis-ready variables are available:

- `X`      : (n_signals, 78) DataFrame of T-PPG features
- `y`      : (n_signals,) Series, 1 = Black, 0 = White
- `groups` : (n_signals,) Series of patient IDs
- `sites`  : (n_signals,) Series in {"Site 1", "Site 2"}
- `abis`   : (n_signals,) Series of ABI values
- `covariates`   : DataFrame indexed by patient ID with Age, Female, Diabetes, ESRD, EverSmoker, CurrentSmoker, SmokingUnknown (patient-level)

Race labels correspond to self-reported 'White, Not-Hispanic' and 'Black, Not-Hispanic' patients. 

## Running
Load or define the variables above before executing `analysis_code.py`.

The script prints pooled and per fold AUROCs, PR AUC, balanced accuracy, confounder analysis results, nonlinear model performance, patient characteristics, and feature level results. It also displays the forest plot, mean ROC curve, and pooled confusion matrix.
