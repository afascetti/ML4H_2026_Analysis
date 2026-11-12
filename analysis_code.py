
# Race prediction from T-PPG features: main model, confounder controls,
# site-adjusted and nonlinear models, and feature-level analyses with
# clinical covariate adjustment.
# Site names are anonymized (Site 1, Site 2).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, average_precision_score, balanced_accuracy_score
from sklearn.utils import shuffle
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import norm


# ============================================================
# Data contract
# ------------------------------------------------------------
# Preprocessing (signal QC, featurization, and cohort filtering
# to White, Not Hispanic and Black, Not Hispanic patients) is
# omitted here. After preprocessing, the following are defined:
#
#   X       : (n_signals, 78) DataFrame of T-PPG features
#   y       : (n_signals,)    Series, 1 = Black, 0 = White
#   groups  : (n_signals,)    Series of patient IDs
#   sites   : (n_signals,)    Series in {"Site 1", "Site 2"}
#   abis    : (n_signals,)    Series of ABI values
#   covariates : DataFrame indexed by patient ID with Age, Female,
#                Diabetes, ESRD, EverSmoker, CurrentSmoker,
#                SmokingUnknown (patient-level)
# ============================================================


# Shuffle all arrays together so fold assignment is deterministic.
X, y, groups, sites, abis = shuffle(X, y, groups, sites, abis, random_state=42)


# ============================================================
# Patient-level stratified bootstrap CI for pooled OOF AUC.
# Patients are the sampling unit (a patient may contribute multiple
# signals). Resampling is stratified by race. The point estimate
# is the AUC of the original OOF predictions; bootstrap only
# provides the CI. Defaults to AUROC; other metrics can be passed via 'metric'.
# ============================================================

def patient_stratified_bootstrap_auc(
    y_true, y_score, groups_in, n_boot=1000, seed=0, metric=roc_auc_score
):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    groups_in = np.asarray(groups_in)

    # Map each patient to a single class label.
    unique_groups = np.unique(groups_in)
    group_to_class = {}
    for g in unique_groups:
        labels = np.unique(y_true[groups_in == g])
        if len(labels) != 1:
            raise ValueError(f"Patient {g} has multiple labels: {labels}")
        group_to_class[g] = labels[0]

    groups_by_class = {
        c: np.array([g for g in unique_groups if group_to_class[g] == c])
        for c in np.unique(y_true)
    }

    point_auc = metric(y_true, y_score)

    rng = np.random.default_rng(seed)
    boot_aucs = []
    for _ in range(n_boot):
        replicate_indices = []
        # Resample patients within class; include all signals per draw.
        for c in sorted(groups_by_class.keys()):
            sampled = rng.choice(
                groups_by_class[c], size=len(groups_by_class[c]), replace=True,
            )
            for g in sampled:
                replicate_indices.append(np.where(groups_in == g)[0])
        idx = np.concatenate(replicate_indices)
        try:
            boot_aucs.append(metric(y_true[idx], y_score[idx]))
        except ValueError:
            pass

    boot_aucs = np.asarray(boot_aucs)
    return point_auc, np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5)


# ============================================================
# Main model: 10-fold CV SVM on the 78 T-PPG features
# ============================================================

kf = StratifiedGroupKFold(n_splits=10)

mean_fpr = np.linspace(0, 1, 100)
fold_tprs = []
fold_aucs = []
fold_coefs = []
y_true_pooled, y_score_pooled, y_pred_pooled, groups_pooled = [], [], [], []

for train_idx, test_idx in kf.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]

    # Fit scaler on training data only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(
        kernel='linear', C=0.05,
        class_weight='balanced',
        probability=True, random_state=42,
    )
    model.fit(X_train, y_train)
    fold_coefs.append(model.coef_[0])

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Per-fold ROC, interpolated onto a common FPR grid.
    fpr, tpr, _ = roc_curve(y_test, y_score)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    fold_tprs.append(interp_tpr)
    fold_aucs.append(auc(fpr, tpr))

    y_true_pooled.extend(y_test.values)
    y_score_pooled.extend(y_score)
    y_pred_pooled.extend(y_pred)
    groups_pooled.extend(groups_test.values)

y_true_pooled = np.asarray(y_true_pooled)
y_score_pooled = np.asarray(y_score_pooled)
y_pred_pooled = np.asarray(y_pred_pooled)
groups_pooled = np.asarray(groups_pooled)

pooled_auc, pooled_lo, pooled_hi = patient_stratified_bootstrap_auc(
    y_true_pooled, y_score_pooled, groups_pooled,
)
mean_fold_auc = np.mean(fold_aucs)
std_fold_auc = np.std(fold_aucs)

print(f"Pooled OOF AUC: {pooled_auc:.3f} "
      f"(95% CI [{pooled_lo:.3f}, {pooled_hi:.3f}])")
print(f"Mean-of-fold AUC: {mean_fold_auc:.3f} ± {std_fold_auc:.3f}")

# Secondary metrics for the class-imbalanced setting. PR-AUC is
# reported alongside its chance level (prevalence of the positive
# class); balanced accuracy uses the model's default threshold.
pr_auc, pr_lo, pr_hi = patient_stratified_bootstrap_auc(
    y_true_pooled, y_score_pooled, groups_pooled,
    metric=average_precision_score,
)
bal_acc, bal_lo, bal_hi = patient_stratified_bootstrap_auc(
    y_true_pooled, y_pred_pooled, groups_pooled,
    metric=balanced_accuracy_score,
)
prevalence = y_true_pooled.mean()

print(f"Pooled OOF PR-AUC: {pr_auc:.3f} "
      f"(95% CI [{pr_lo:.3f}, {pr_hi:.3f}]; chance = {prevalence:.3f})")
print(f"Pooled OOF balanced accuracy: {bal_acc:.3f} "
      f"(95% CI [{bal_lo:.3f}, {bal_hi:.3f}])")

# ============================================================
# Appendix figures: mean ROC + pooled confusion matrix
# ============================================================

mean_tpr = np.mean(fold_tprs, axis=0)
 
plt.figure(figsize=(6, 6))
plt.plot(mean_fpr, mean_tpr, label=f"Linear SVM (mean AUROC = {mean_fold_auc:.3f} ± {std_fold_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', label='Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC across 10 cross-validation folds")
plt.legend()
plt.tight_layout()
# plt.savefig("ML4H_Results/mean_roc_300dpi_cameraready.png", dpi=300, bbox_inches='tight')

# gray scale version
ax = plt.gca()
ax.get_lines()[0].set_color("black")
ax.get_lines()[1].set_color("gray")
ax.legend()
# plt.savefig("ML4H_Results/mean_roc_300dpi_cameraready-gray.png", dpi=300, bbox_inches="tight")

plt.show()

# Confusion matrix: raw count + row percentage per cell, with
# each row colored on its own scale so class imbalance doesn't
# wash out the minority row.
cm = confusion_matrix(y_true_pooled, y_pred_pooled, labels=[0, 1])
row_totals = cm.sum(axis=1, keepdims=True)
cm_row_norm = cm / row_totals
 
fig, ax = plt.subplots(figsize=(5, 5))
labels_display = ["White", "Black"]
for i in range(cm.shape[0]):
    # One imshow per row so each row gets its own color scale.
    ax.imshow(
        np.where(np.arange(cm.shape[0])[:, None] == i, cm_row_norm, np.nan),
        cmap='Blues', vmin=0, vmax=1, aspect='equal',
    )
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        pct = cm_row_norm[i, j] * 100
        color = 'white' if cm_row_norm[i, j] > 0.5 else 'black'
        ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                ha='center', va='center', color=color)
ax.set_xticks([0, 1]); ax.set_xticklabels(labels_display)
ax.set_yticks([0, 1]); ax.set_yticklabels(labels_display)
ax.set_xlabel("Predicted race label")
ax.set_ylabel("True race label")
ax.set_title("Pooled out-of-fold confusion matrix")
plt.tight_layout()
# plt.savefig("ML4H_Results/confusion_matrix_300dpi_cameraready.png", dpi=300, bbox_inches='tight')

# gray scale version
for im in ax.images:
    im.set_cmap("Greys")
# plt.savefig("ML4H_Results/confusion_matrix_300dpi_cameraready-gray.png", dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# Confounder controls
# ------------------------------------------------------------
# Model-based rows use grouped CV, pooled OOF predictions, and a
# patient-stratified bootstrap CI. Single-variable controls (site,
# ABI) are scored directly from the variable with the same bootstrap.
# ============================================================

def run_svm_cv_pooled(X_in, y_in, groups_in, n_splits=10, random_state=42):
    """Grouped CV; returns pooled OOF (y_true, y_score, groups)."""
    if hasattr(X_in, "reset_index"):
        X_in = X_in.reset_index(drop=True)
    y_in = pd.Series(y_in).reset_index(drop=True)
    groups_in = pd.Series(groups_in).reset_index(drop=True)

    # Cap folds by smaller class's unique-patient count.
    y_arr, g_arr = y_in.to_numpy(), groups_in.to_numpy()
    min_groups = min(np.unique(g_arr[y_arr == c]).size for c in np.unique(y_arr))
    n_splits = max(2, min(n_splits, min_groups))

    kf_local = StratifiedGroupKFold(n_splits=n_splits)
    y_true, y_score, groups_out = [], [], []
    for tr, te in kf_local.split(X_in, y_in, groups=groups_in):
        X_tr = X_in.iloc[tr] if hasattr(X_in, 'iloc') else X_in[tr]
        X_te = X_in.iloc[te] if hasattr(X_in, 'iloc') else X_in[te]
        y_tr, y_te = y_in.iloc[tr], y_in.iloc[te]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        m = SVC(kernel='linear', C=0.05, class_weight='balanced',
                probability=True, random_state=random_state)
        m.fit(X_tr, y_tr)
        y_true.extend(y_te.values)
        y_score.extend(m.predict_proba(X_te)[:, 1])
        groups_out.extend(groups_in.iloc[te].values)
    return np.asarray(y_true), np.asarray(y_score), np.asarray(groups_out)


def evaluate_pooled_model(X_in, y_in, groups_in, n_splits=10):
    yt, ys, gs = run_svm_cv_pooled(X_in, y_in, groups_in, n_splits=n_splits)
    return patient_stratified_bootstrap_auc(yt, ys, gs)


results = [("PPG features (all sites)", pooled_auc, pooled_lo, pooled_hi)]

# Single-variable controls: AUROC computed directly from the variable
# (no classifier), since a model on one variable reduces to a monotone
# transform of it. Site 1 and lower ABI are scored toward Black.
# CIs use the same patient-level stratified bootstrap.
site_score = (np.asarray(sites) == "Site 1").astype(float)
results.append(("Site only",
                *patient_stratified_bootstrap_auc(y, site_score, groups)))

abi_score = -np.asarray(abis)
results.append(("ABI only",
                *patient_stratified_bootstrap_auc(y, abi_score, groups)))

# Per-site PPG-feature models. Reduced subsets use 5 folds.
for site_name in ["Site 1", "Site 2"]:
    mask = np.asarray(sites) == site_name
    results.append((
        f"PPG features — {site_name}",
        *evaluate_pooled_model(
            X.iloc[mask].reset_index(drop=True),
            y.iloc[mask].reset_index(drop=True),
            groups.iloc[mask].reset_index(drop=True),
            n_splits=5,
        ),
    ))


# ============================================================
# Site 1 ABI-matched analysis: deterministic 1:1 nearest-ABI
# match of White to Black signals, without replacement, then
# evaluated with the standard pipeline.
# ============================================================

def nearest_abi_match_without_replacement(y_in, abi_in):
    y_arr = np.asarray(y_in)
    abi_arr = np.asarray(abi_in)
    black_idx = np.where(y_arr == 1)[0]
    white_idx = np.where(y_arr == 0)[0]

    if len(white_idx) < len(black_idx):
        raise ValueError("Not enough White signals for 1:1 matching.")

    # Process Black signals in ascending ABI order; sorted available
    # pool makes np.argmin resolve ties by lower index reproducibly.
    black_order = black_idx[np.argsort(abi_arr[black_idx], kind='stable')]
    available = set(white_idx.tolist())
    matched_black, matched_white = [], []
    for b in black_order:
        avail = np.asarray(sorted(available))
        w = avail[np.argmin(np.abs(abi_arr[avail] - abi_arr[b]))]
        matched_black.append(b)
        matched_white.append(w)
        available.remove(int(w))
    return np.asarray(matched_black), np.asarray(matched_white)


site1_mask = np.asarray(sites) == "Site 1"
X_s1 = X.iloc[site1_mask].reset_index(drop=True)
y_s1 = y.iloc[site1_mask].reset_index(drop=True)
g_s1 = groups.iloc[site1_mask].reset_index(drop=True)
abi_s1 = abis.iloc[site1_mask].reset_index(drop=True)

matched_black, matched_white = nearest_abi_match_without_replacement(y_s1, abi_s1)
matched_idx = np.concatenate([matched_black, matched_white])

X_s1_m = X_s1.iloc[matched_idx].reset_index(drop=True)
y_s1_m = y_s1.iloc[matched_idx].reset_index(drop=True)
g_s1_m = g_s1.iloc[matched_idx].reset_index(drop=True)

# Matching diagnostics.
pair_diffs = np.abs(
    abi_s1.iloc[matched_black].to_numpy()
    - abi_s1.iloc[matched_white].to_numpy()
)
print("\nSite 1 ABI-matched cohort")
print(f"  Black signals:   {len(matched_black)}")
print(f"  White signals:   {len(matched_white)}")
print(f"  Total signals:   {len(matched_idx)}")
print(f"  Unique patients: {g_s1_m.nunique()}")
print(f"  Median |ABI diff|: {np.median(pair_diffs):.4f}")
print(f"  Mean |ABI diff|:   {np.mean(pair_diffs):.4f}")
print(f"  Max |ABI diff|:    {np.max(pair_diffs):.4f}")
print(f"  Mean ABI, Black: {abi_s1.iloc[matched_black].mean():.4f}")
print(f"  Mean ABI, White: {abi_s1.iloc[matched_white].mean():.4f}")

results.append((
    "PPG features — Site 1, ABI-matched",
    *evaluate_pooled_model(X_s1_m, y_s1_m, g_s1_m, n_splits=5),
))



# ============================================================
# PPG + site vs. site only
# ------------------------------------------------------------
# Tests whether the PPG features add race-predictive information
# beyond acquisition site. A site indicator is appended to the
# 78 PPG features and evaluated with the standard pipeline. The
# difference vs. site only uses a paired patient-level bootstrap
# (both scores evaluated on the same resampled patients).
# ============================================================

def patient_stratified_bootstrap_auc_diff(
    y_true, score_a, score_b, groups_in, n_boot=1000, seed=0,
):
    """Paired bootstrap CI for AUC(score_a) - AUC(score_b)."""
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    groups_in = np.asarray(groups_in)

    unique_groups = np.unique(groups_in)
    group_to_class = {g: y_true[groups_in == g][0] for g in unique_groups}
    groups_by_class = {
        c: np.array([g for g in unique_groups if group_to_class[g] == c])
        for c in np.unique(y_true)
    }

    point_diff = roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b)

    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(n_boot):
        replicate_indices = []
        for c in sorted(groups_by_class.keys()):
            sampled = rng.choice(
                groups_by_class[c], size=len(groups_by_class[c]), replace=True,
            )
            for g in sampled:
                replicate_indices.append(np.where(groups_in == g)[0])
        idx = np.concatenate(replicate_indices)
        boot_diffs.append(
            roc_auc_score(y_true[idx], score_a[idx])
            - roc_auc_score(y_true[idx], score_b[idx])
        )

    boot_diffs = np.asarray(boot_diffs)
    return point_diff, np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5)


# PPG features + site indicator (Site 1 as reference).
X_ppg_site = X.reset_index(drop=True).copy()
X_ppg_site["Site 2"] = (np.asarray(sites) == "Site 2").astype(float)

yt_ps, ys_ps, gs_ps = run_svm_cv_pooled(X_ppg_site, y, groups)
ppg_site_auc, ppg_site_lo, ppg_site_hi = patient_stratified_bootstrap_auc(
    yt_ps, ys_ps, gs_ps,
)

# Align site-only scores to the pooled OOF order (same deterministic split).
oof_idx = np.concatenate([
    te for _, te in StratifiedGroupKFold(n_splits=10).split(
        X_ppg_site, y.reset_index(drop=True), groups=groups.reset_index(drop=True),
    )
])
assert np.array_equal(np.asarray(y)[oof_idx], yt_ps)
assert np.array_equal(np.asarray(groups)[oof_idx], gs_ps)
site_score_oof = site_score[oof_idx]

diff, diff_lo, diff_hi = patient_stratified_bootstrap_auc_diff(
    yt_ps, ys_ps, site_score_oof, gs_ps,
)

print("\nPPG + site vs. site only")
print(f"  PPG + site:  {ppg_site_auc:.3f}  [{ppg_site_lo:.3f}, {ppg_site_hi:.3f}]")
print(f"  Site only:   {roc_auc_score(yt_ps, site_score_oof):.3f}")
print(f"  Difference:  {diff:+.3f}  [{diff_lo:+.3f}, {diff_hi:+.3f}]")

results.insert(2, ("PPG features + site", ppg_site_auc, ppg_site_lo, ppg_site_hi))



# ============================================================
# Summary table
# ============================================================

print(f"\n{'Analysis':<38} {'AUC':>6}  {'95% CI':>16}")
print("-" * 62)
for label, m, lo, hi in results:
    print(f"{label:<38} {m:>6.3f}  [{lo:.3f}, {hi:.3f}]")


# ============================================================
# Forest plot
# ============================================================

labels = [r[0] for r in results]
means = np.asarray([r[1] for r in results])
lo_err = means - np.asarray([r[2] for r in results])
hi_err = np.asarray([r[3] for r in results]) - means
y_pos = np.arange(len(results))[::-1]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(means, y_pos, xerr=[lo_err, hi_err],
            fmt='o', color='black', ecolor='gray',
            capsize=3, markersize=6, linewidth=1.2)
ax.axvline(0.5, linestyle='--', color='red', alpha=0.6, label='Chance')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Race-prediction AUROC (95% CI)")
ax.set_xlim(0.3, 0.75)
ax.set_title("Race-prediction performance across analyses")
ax.legend(loc='lower right', frameon=False)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()

# plt.savefig("ML4H_Results/race_confounder_forest_300dpi_cameraready.png", dpi=300, bbox_inches='tight')

# gray scale version
for line in ax.get_lines():
    if line.get_color() == "red":
        line.set_color("black")
ax.legend(loc='lower right', frameon=False)
# plt.savefig("ML4H_Results/race_confounder_forest_300dpi_cameraready-gray.png", dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# Nonlinear probe: gradient-boosted trees
# ------------------------------------------------------------
# Tests whether a more flexible model extracts more race
# information from the same 78 features than the linear SVM.
# Same grouped 10-fold CV and patient-level bootstrap. Trees do
# not require feature scaling; class weighting mirrors the SVM.
# Default hyperparameters.
# ============================================================


def run_gbm_cv_pooled(X_in, y_in, groups_in, n_splits=10, random_state=42):
    """Grouped CV with gradient-boosted trees; returns pooled OOF (y_true, y_score, groups)."""
    X_in = X_in.reset_index(drop=True)
    y_in = pd.Series(y_in).reset_index(drop=True)
    groups_in = pd.Series(groups_in).reset_index(drop=True)

    kf_local = StratifiedGroupKFold(n_splits=n_splits)
    y_true, y_score, groups_out = [], [], []
    for tr, te in kf_local.split(X_in, y_in, groups=groups_in):
        m = HistGradientBoostingClassifier(
            class_weight='balanced', random_state=random_state,
        )
        m.fit(X_in.iloc[tr], y_in.iloc[tr])
        y_true.extend(y_in.iloc[te].values)
        y_score.extend(m.predict_proba(X_in.iloc[te])[:, 1])
        groups_out.extend(groups_in.iloc[te].values)
    return np.asarray(y_true), np.asarray(y_score), np.asarray(groups_out)


gbm_auc, gbm_lo, gbm_hi = patient_stratified_bootstrap_auc(
    *run_gbm_cv_pooled(X, y, groups),
)
print(f"\nGradient-boosted trees (all sites): "
      f"AUROC {gbm_auc:.3f} [{gbm_lo:.3f}, {gbm_hi:.3f}]")


# ============================================================
# Feature-level analysis
# ------------------------------------------------------------
# (1) Per-feature standardized mean difference (SMD, Black minus
#     White) on patient-level feature means, computed within each
#     site and combined across sites by inverse-variance weighting,
#     so that site imbalance is not mistaken for a race difference.
#     P-values are Benjamini-Hochberg FDR-corrected across the 78
#     features.
# (2) Main-model linear SVM coefficients averaged over the 10 folds
#     (features are standardized, so magnitudes are comparable).
# Positive values indicate higher in / weighted toward Black.
# ============================================================


def site_smd(feats_in, y_in, groups_in):
    """Patient-level SMD (Black - White) per feature, with its variance."""
    df = feats_in.reset_index(drop=True).copy()
    df["_y"] = np.asarray(y_in)
    df["_g"] = np.asarray(groups_in)

    # Average each patient's signals so patients are the unit of analysis.
    patient_means = df.groupby("_g").mean()
    black = patient_means[patient_means["_y"] == 1].drop(columns="_y")
    white = patient_means[patient_means["_y"] == 0].drop(columns="_y")
    n1, n0 = len(black), len(white)

    pooled_sd = np.sqrt(
        ((n1 - 1) * black.var() + (n0 - 1) * white.var()) / (n1 + n0 - 2)
    )
    d = (black.mean() - white.mean()) / pooled_sd
    var_d = (n1 + n0) / (n1 * n0) + d**2 / (2 * (n1 + n0))
    return d, var_d, n1, n0


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(p_values)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def residualize(feats_in, covariate_in):
    """Remove the linear effect of one or more covariates from each feature."""
    F = feats_in.to_numpy(dtype=float)
    C = np.asarray(covariate_in, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    Z = np.column_stack([np.ones(len(F)), C])
    beta, *_ = np.linalg.lstsq(Z, F, rcond=None)
    return pd.DataFrame(F - Z @ beta, columns=feats_in.columns)


def combined_site_smd(feats_in, y_in, groups_in, sites_in, adjust_for=None):
    """Within-site patient-level SMDs combined across sites by
    inverse-variance weighting, with BH-adjusted q-values. If
    adjust_for is given, features are first residualized on that
    covariate within each site."""
    sites_arr = np.asarray(sites_in)
    smd_by_site, var_by_site = {}, {}
    for site_name in ["Site 1", "Site 2"]:
        mask = sites_arr == site_name
        feats_site = feats_in.iloc[mask].reset_index(drop=True)
        if adjust_for is not None:
            feats_site = residualize(feats_site, np.asarray(adjust_for)[mask])
        d, var_d, _, _ = site_smd(
            feats_site, np.asarray(y_in)[mask], np.asarray(groups_in)[mask],
        )
        smd_by_site[site_name] = d.reindex(feats_in.columns)
        var_by_site[site_name] = var_d.reindex(feats_in.columns)

    w1 = 1 / var_by_site["Site 1"]
    w2 = 1 / var_by_site["Site 2"]
    smd_combined = (w1 * smd_by_site["Site 1"] + w2 * smd_by_site["Site 2"]) / (w1 + w2)
    se_combined = np.sqrt(1 / (w1 + w2))
    p_combined = 2 * norm.sf(np.abs(smd_combined / se_combined))
    return smd_by_site, smd_combined, benjamini_hochberg(p_combined)


feature_names = X.columns

# (1) Within-site SMDs, combined across sites.
smd_by_site, smd_combined, q_combined = combined_site_smd(X, y, groups, sites)

# (2) Fold-averaged SVM coefficients from the main model.
mean_coef = pd.Series(np.mean(fold_coefs, axis=0), index=feature_names)
coef_rank = mean_coef.abs().rank(ascending=False).astype(int)

feature_table = pd.DataFrame({
    "SMD Site 1": smd_by_site["Site 1"].values,
    "SMD Site 2": smd_by_site["Site 2"].values,
    "SMD combined": smd_combined.values,
    "q (FDR)": q_combined,
    "SVM coef": mean_coef.values,
    "Coef rank": coef_rank.values,
}, index=feature_names)
feature_table = feature_table.reindex(
    feature_table["SMD combined"].abs().sort_values(ascending=False).index
)

n_sig = (feature_table["q (FDR)"] < 0.05).sum()
top10_smd = set(feature_table.index[:10])
top10_coef = set(coef_rank[coef_rank <= 10].index)

print(f"\nFeatures with FDR q < 0.05: {n_sig} of {len(feature_names)}")
print(f"Overlap of top-10 by |SMD| and top-10 by |SVM coef|: "
      f"{len(top10_smd & top10_coef)} of 10")
print("\nTop 10 features by |combined SMD|")
print(feature_table.head(10).to_string(float_format=lambda v: f"{v:.3f}"))

# feature_table.to_csv("ML4H_Results/feature_level_analysis.csv")

# ------------------------------------------------------------
# ABI-adjusted SMDs: features are residualized on ABI within
# each site before computing patient-level SMDs, testing whether
# race differences in the features reflect PAD severity.
# ------------------------------------------------------------
_, smd_combined_abi, q_combined_abi = combined_site_smd(
    X, y, groups, sites, adjust_for=abis,
)
smd_combined_abi = smd_combined_abi.reindex(feature_table.index)
q_combined_abi = pd.Series(q_combined_abi, index=X.columns).reindex(feature_table.index)

feature_table["SMD combined (ABI-adj)"] = smd_combined_abi.values
feature_table["q (ABI-adj)"] = q_combined_abi.values

n_sig_abi = (feature_table["q (ABI-adj)"] < 0.05).sum()
shrinkage = (feature_table["SMD combined (ABI-adj)"].abs()
             / feature_table["SMD combined"].abs()).median()

print(f"\nABI-adjusted: features with FDR q < 0.05: {n_sig_abi} of {len(X.columns)}")
print(f"Median |SMD| retained after ABI adjustment: {shrinkage:.0%}")
print("\nTop 10 features (by unadjusted |SMD|): unadjusted vs. ABI-adjusted")
print(feature_table.head(10)[[
    "SMD combined", "q (FDR)", "SMD combined (ABI-adj)", "q (ABI-adj)",
]].to_string(float_format=lambda v: f"{v:.3f}"))


# ============================================================
# Clinical covariates
# ------------------------------------------------------------
# (1) Patient-level descriptive comparison by race, overall and
#     within each site.
# (2) Feature SMDs adjusted for ABI, age, sex, diabetes, ESRD, and smoking
#     within site (extends the ABI-adjusted analysis above).
# Covariates are patient-level (mean age across measurements;
# comorbidities ever recorded).
# ============================================================

sites_arr = np.asarray(sites)
y_arr = np.asarray(y)
g_arr = np.asarray(groups)


def describe_patients(patient_ids):
    c = covariates.loc[patient_ids]
    return pd.Series({
        "Patients": f"{len(c)}",
        "Age, mean (SD)": f"{c['Age'].mean():.1f} ({c['Age'].std():.1f})",
        "Female, %": f"{100 * c['Female'].mean():.1f}",
        "Diabetes, %": f"{100 * c['Diabetes'].mean():.1f}",
        "ESRD, %": f"{100 * c['ESRD'].mean():.1f}",
        "Ever smoker, %": f"{100 * c['EverSmoker'].mean():.1f}",
        "Current smoker, %": f"{100 * c['CurrentSmoker'].mean():.1f}",
        "Smoking unknown, %": f"{100 * c['SmokingUnknown'].mean():.1f}",
    })


# (1) Descriptive table.
desc = {}
for site_label, site_mask in [
    ("All sites", np.ones(len(y_arr), dtype=bool)),
    ("Site 1", sites_arr == "Site 1"),
    ("Site 2", sites_arr == "Site 2"),
]:
    for race_label, race_val in [("White", 0), ("Black", 1)]:
        ids = np.unique(g_arr[site_mask & (y_arr == race_val)])
        desc[(site_label, race_label)] = describe_patients(ids)

print("\nPatient characteristics by race and site")
print(pd.DataFrame(desc).to_string())

# Signal-level adjustment covariates (ABI + patient-level clinical).
adjusters = pd.concat([
    pd.Series(np.asarray(abis, dtype=float), name="ABI"),
    covariates.loc[g_arr].reset_index(drop=True),
], axis=1)

# (2) Covariate-adjusted SMDs.
_, smd_combined_cov, q_combined_cov = combined_site_smd(
    X, y, groups, sites, adjust_for=adjusters,
)
feature_table["SMD combined (ABI+clin-adj)"] = (
    smd_combined_cov.reindex(feature_table.index).values
)
feature_table["q (ABI+clin-adj)"] = (
    pd.Series(q_combined_cov, index=X.columns).reindex(feature_table.index).values
)

n_sig_cov = (feature_table["q (ABI+clin-adj)"] < 0.05).sum()
retained_cov = (feature_table["SMD combined (ABI+clin-adj)"].abs()
                / feature_table["SMD combined"].abs()).median()

print(f"\nABI + clinical adjusted: features with FDR q < 0.05: "
      f"{n_sig_cov} of {len(X.columns)}")
print(f"Median |SMD| retained after ABI + clinical adjustment: {retained_cov:.0%}")
print("\nTop 10 features (by unadjusted |SMD|)")
print(feature_table.head(10)[[
    "SMD combined", "SMD combined (ABI-adj)",
    "SMD combined (ABI+clin-adj)", "q (ABI+clin-adj)",
]].to_string(float_format=lambda v: f"{v:.3f}"))

