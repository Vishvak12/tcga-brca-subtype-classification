"""
04_analyze_results.py
Day 2 — Aggregate all trial results, retrain the best model on train+val,
evaluate on held-out test set, extract biomarker gene list, and
generate publication-quality plots (UMAP + feature importance).

Run on the login node (or an interactive session) after all array tasks finish:
    python 04_analyze_results.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — no display needed on HPC
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xgboost as xgb
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# Optional — install with:  pip install umap-learn
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠  umap-learn not installed — UMAP plot will be skipped.")
    print("   pip install umap-learn")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH     = os.environ.get("SCRATCH", "./data")
PROC_DIR    = os.path.join(SCRATCH, "processed")
RESULTS_DIR = os.path.join(SCRATCH, "results")
PLOTS_DIR   = os.path.join(SCRATCH, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 1. Load & Aggregate Trial Results ─────────────────────────────────────────
print("── Loading trial results ──")
result_files = glob.glob(os.path.join(RESULTS_DIR, "result_*.json"))
assert result_files, "No result files found — did the array jobs finish?"

records = []
for fp in result_files:
    with open(fp) as f:
        records.append(json.load(f))

df = pd.DataFrame(records).sort_values("val_accuracy", ascending=False)
df.to_csv(os.path.join(PLOTS_DIR, "all_trials.csv"), index=False)

print(df[["trial_id","val_accuracy","val_f1_macro",
          "learning_rate","max_depth","n_estimators",
          "subsample","colsample_bytree"]].head(10).to_string(index=False))

best = df.iloc[0]
print(f"\n✓  Best trial {int(best.trial_id)}:  "
      f"val_acc={best.val_accuracy:.4f}  val_f1={best.val_f1_macro:.4f}")

# ── 2. Reload Data ────────────────────────────────────────────────────────────
X_train = np.load(os.path.join(PROC_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(PROC_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy"))
y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"))
y_val   = np.load(os.path.join(PROC_DIR, "y_val.npy"))
y_test  = np.load(os.path.join(PROC_DIR, "y_test.npy"))

le         = joblib.load(os.path.join(PROC_DIR, "label_encoder.pkl"))
gene_names = pd.read_csv(os.path.join(PROC_DIR, "gene_names.csv"))["0"].tolist()

# ── 3. Retrain Best Model on train+val ────────────────────────────────────────
print("\n── Retraining best model on train+val ──")
X_tv = np.vstack([X_train, X_val])
y_tv = np.concatenate([y_train, y_val])

n_classes = len(le.classes_)
best_params = {k: best[k] for k in [
    "n_estimators","max_depth","learning_rate",
    "subsample","colsample_bytree","min_child_weight",
    "reg_alpha","reg_lambda",
]}
# Cast ints back (JSON round-trips them as float sometimes)
for k in ["n_estimators","max_depth","min_child_weight"]:
    best_params[k] = int(best_params[k])

final_model = xgb.XGBClassifier(
    objective         = "multi:softprob",
    num_class         = n_classes,
    eval_metric       = "mlogloss",
    tree_method       = "hist",
    device            = "cpu",
    use_label_encoder = False,
    verbosity         = 0,
    n_jobs            = 8,
    random_state      = 42,
    **best_params,
)
final_model.fit(X_tv, y_tv)
final_model.save_model(os.path.join(RESULTS_DIR, "final_model.json"))

# ── 4. Test-Set Evaluation ────────────────────────────────────────────────────
print("\n── Test-set evaluation ──")
y_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1  = f1_score(y_test, y_pred, average="macro")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  F1 Macro : {test_f1:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save metrics
metrics = {"test_accuracy": test_acc, "test_f1_macro": test_f1}
with open(os.path.join(PLOTS_DIR, "test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ── 5. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, colorbar=True, xticks_rotation=45, cmap="Blues")
ax.set_title(f"Confusion Matrix — Test Set  (acc={test_acc:.3f})", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("✓  Saved confusion_matrix.png")

# ── 6. Feature Importance — Top 30 Biomarkers ────────────────────────────────
importances = final_model.feature_importances_
feat_df = pd.DataFrame({
    "gene"       : gene_names,
    "importance" : importances,
}).sort_values("importance", ascending=False)
feat_df.to_csv(os.path.join(PLOTS_DIR, "feature_importances.csv"), index=False)

top_n  = 30
top_df = feat_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.RdYlGn(np.linspace(0.85, 0.25, top_n))
bars = ax.barh(top_df["gene"][::-1], top_df["importance"][::-1],
               color=colors[::-1], edgecolor="none")
ax.set_xlabel("XGBoost F-score (importance)", fontsize=11)
ax.set_title(f"Top {top_n} Predictive Genes (Candidate Biomarkers)", fontsize=13)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("✓  Saved feature_importance.png")
print(f"\nTop 10 candidate biomarkers:\n{top_df[['gene','importance']].head(10).to_string(index=False)}")

# ── 7. UMAP Visualization ─────────────────────────────────────────────────────
if UMAP_AVAILABLE:
    print("\n── Generating UMAP ──  (this may take a few minutes)")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        verbose=True,
    )
    # Use the full train+val set for a richer embedding
    embedding = reducer.fit_transform(X_tv)

    cmap    = plt.cm.get_cmap("tab20", n_classes)
    fig, ax = plt.subplots(figsize=(14, 10))
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = y_tv == cls_idx
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[cmap(cls_idx)], label=cls_name,
            s=18, alpha=0.75, linewidths=0,
        )
    ax.set_title("UMAP — TCGA Pan-Cancer Gene Expression\n"
                 f"({n_classes} cancer types, {len(y_tv)} samples)", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper right", fontsize=7, ncol=2,
              framealpha=0.7, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "umap_cancertypes.png"), dpi=180)
    plt.close()
    print("✓  Saved umap_cancertypes.png")
else:
    print("⚠  Skipped UMAP (umap-learn not available).")

# ── 8. Hyperparameter Sweep Summary Plot ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sweep_params = ["learning_rate", "max_depth", "n_estimators"]
for ax, param in zip(axes, sweep_params):
    ax.scatter(df[param], df["val_accuracy"], alpha=0.6, s=25, c="steelblue")
    ax.set_xlabel(param)
    ax.set_ylabel("Val Accuracy")
    ax.set_title(f"Accuracy vs {param}")
    ax.grid(True, linestyle="--", alpha=0.4)
plt.suptitle("Hyperparameter Sweep Results", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sweep_summary.png"), dpi=150)
plt.close()
print("✓  Saved sweep_summary.png")

print(f"\n✓  All outputs written to {PLOTS_DIR}")
print("   Files: confusion_matrix.png  feature_importance.png  "
      "umap_cancertypes.png  sweep_summary.png  "
      "feature_importances.csv  all_trials.csv  test_metrics.json")
