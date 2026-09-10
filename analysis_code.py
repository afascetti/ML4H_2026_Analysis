
# Race prediction from T-PPG features: main model and confounder controls.
# Site names are anonymized (Site 1, Site 2).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.utils import shuffle



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
# ============================================================


# Shuffle all arrays together so fold assignment is deterministic.
X, y, groups, sites, abis = shuffle(X, y, groups, sites, abis, random_state=42)


# ============================================================
# Patient-level stratified bootstrap CI for pooled OOF AUC.
# Patients are the sampling unit (a patient may contribute multiple
# signals). Resampling is stratified by race. The point estimate
# is the AUC of the original OOF predictions; bootstrap only
# provides the CI.
# ============================================================

def patient_stratified_bootstrap_auc(
    y_true, y_score, groups_in, n_boot=1000, seed=0,
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

    point_auc = roc_auc_score(y_true, y_score)

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
            boot_aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
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
# plt.savefig("ML4H_Results/mean_roc_300dpi.png", dpi=300, bbox_inches='tight')
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
# plt.savefig("ML4H_Results/confusion_matrix_300dpi.png", dpi=300, bbox_inches='tight')
plt.show()


# ============================================================
# Confounder controls
# ------------------------------------------------------------
# Every row of the forest plot does: grouped CV,
# pooled OOF predictions, patient-stratified bootstrap CI.
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

# Site alone (one-hot, no PPG features).
X_site = pd.get_dummies(
    pd.Series(sites).reset_index(drop=True), drop_first=True,
).astype(float)
results.append(("Site only", *evaluate_pooled_model(X_site, y, groups)))

# ABI alone.
X_abi = pd.DataFrame({'ABI': np.asarray(abis)})
results.append(("ABI only", *evaluate_pooled_model(X_abi, y, groups)))

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

# plt.savefig("ML4H_Results/race_confounder_forest_300dpi_final.png", dpi=300, bbox_inches='tight')
plt.show()
