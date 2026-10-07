"""
06_ensemble.py
Combine the best XGBoost model and best 1D-CNN via:
  1. Soft voting (weighted average of probability outputs)
  2. Stacking   (logistic regression meta-learner on val-set predictions)

Then produce a clean 3-way comparison table and final plots.

Run after both sweeps are complete:
    python 06_ensemble.py
"""

import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH      = os.environ.get("SCRATCH", "./data")
PROC_DIR     = os.path.join(SCRATCH, "processed")
XGB_RESULTS  = os.path.join(SCRATCH, "results")
CNN_RESULTS  = os.path.join(SCRATCH, "results_cnn")
PLOTS_DIR    = os.path.join(SCRATCH, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load Data ─────────────────────────────────────────────────────────────────
# XGBoost uses full feature set
X_train = np.load(os.path.join(PROC_DIR, "X_train.npy")).astype(np.float32)
X_val   = np.load(os.path.join(PROC_DIR, "X_val.npy")).astype(np.float32)
X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy")).astype(np.float32)

# CNN uses reordered+filtered set if available
_cnn_suffix = "_cnn" if os.path.exists(os.path.join(PROC_DIR, "X_train_cnn.npy")) else ""
X_train_cnn_in = np.load(os.path.join(PROC_DIR, f"X_train{_cnn_suffix}.npy")).astype(np.float32)
X_val_cnn_in   = np.load(os.path.join(PROC_DIR, f"X_val{_cnn_suffix}.npy")).astype(np.float32)
X_test_cnn_in  = np.load(os.path.join(PROC_DIR, f"X_test{_cnn_suffix}.npy")).astype(np.float32)
print(f"  CNN input shape: {X_train_cnn_in.shape}")
y_train = np.load(os.path.join(PROC_DIR, "y_train.npy")).astype(np.int64)
y_val   = np.load(os.path.join(PROC_DIR, "y_val.npy")).astype(np.int64)
y_test  = np.load(os.path.join(PROC_DIR, "y_test.npy")).astype(np.int64)
le      = joblib.load(os.path.join(PROC_DIR, "label_encoder.pkl"))
n_classes = len(le.classes_)

# ── Helper: get probability matrix from a DataLoader ─────────────────────────
def cnn_predict_proba(model, X):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X).unsqueeze(1)),
        batch_size=64, shuffle=False,
    )
    probs = []
    with torch.no_grad():
        for (Xb,) in loader:
            logits = model(Xb.to(device))
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)

# ── 1. Load Best XGBoost ──────────────────────────────────────────────────────
print("── Loading best XGBoost ──")
xgb_result_files = glob.glob(os.path.join(XGB_RESULTS, "result_*.json"))
xgb_results = pd.DataFrame([json.load(open(f)) for f in xgb_result_files])
best_xgb_id = int(xgb_results.sort_values("val_accuracy", ascending=False).iloc[0]["trial_id"])

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(XGB_RESULTS, f"model_trial_{best_xgb_id:04d}.json"))

xgb_val_probs  = xgb_model.predict_proba(X_val)
xgb_test_probs = xgb_model.predict_proba(X_test)

xgb_val_acc  = accuracy_score(y_val,  xgb_model.predict(X_val))
xgb_test_acc = accuracy_score(y_test, xgb_model.predict(X_test))
xgb_test_f1  = f1_score(y_test, xgb_model.predict(X_test), average="macro")
print(f"  XGBoost val_acc={xgb_val_acc:.4f}  test_acc={xgb_test_acc:.4f}  test_f1={xgb_test_f1:.4f}")

# ── 2. Load Best CNN ──────────────────────────────────────────────────────────
print("── Loading best CNN ──")
cnn_result_files = glob.glob(os.path.join(CNN_RESULTS, "cnn_result_*.json"))
cnn_results = pd.DataFrame([json.load(open(f)) for f in cnn_result_files])
best_cnn_row = cnn_results.sort_values("val_accuracy", ascending=False).iloc[0]
best_cnn_id  = int(best_cnn_row["trial_id"])

# Re-instantiate model with saved config
# ── GeneCNN class (copied from 05_train_1d_cnn.py) ────────────────────────────
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class GeneCNN(nn.Module):
    def __init__(self, n_genes, n_classes, n_filters, kernel_size, n_blocks, dropout):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            *[ResBlock1D(n_filters, kernel_size, dropout) for _ in range(n_blocks)]
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_filters, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x)
        return self.head(x)

cnn_model = GeneCNN(
    n_genes    = X_train_cnn_in.shape[1],
    n_classes  = n_classes,
    n_filters  = int(best_cnn_row["n_filters"]),
    kernel_size= int(best_cnn_row["kernel_size"]),
    n_blocks   = int(best_cnn_row["n_blocks"]),
    dropout    = float(best_cnn_row["dropout"]),
).to(device)
cnn_model.load_state_dict(
    torch.load(
        os.path.join(CNN_RESULTS, f"cnn_trial_{best_cnn_id:04d}_best.pt"),
        map_location=device,
    )
)

cnn_val_probs  = cnn_predict_proba(cnn_model, X_val_cnn_in)
cnn_test_probs = cnn_predict_proba(cnn_model, X_test_cnn_in)

cnn_val_acc  = accuracy_score(y_val,  cnn_val_probs.argmax(1))
cnn_test_acc = accuracy_score(y_test, cnn_test_probs.argmax(1))
cnn_test_f1  = f1_score(y_test, cnn_test_probs.argmax(1), average="macro")
print(f"  CNN    val_acc={cnn_val_acc:.4f}  test_acc={cnn_test_acc:.4f}  test_f1={cnn_test_f1:.4f}")

# ── 3. Soft Voting Ensemble ───────────────────────────────────────────────────
print("── Soft Voting Ensemble ──")

# Find optimal weight via val set sweep (α * XGB + (1-α) * CNN)
best_alpha, best_val = 0.5, 0.0
for alpha in np.linspace(0, 1, 21):
    blended   = alpha * xgb_val_probs + (1 - alpha) * cnn_val_probs
    val_acc_a = accuracy_score(y_val, blended.argmax(1))
    if val_acc_a > best_val:
        best_val, best_alpha = val_acc_a, alpha

print(f"  Best alpha (XGB weight): {best_alpha:.2f}  |  val_acc={best_val:.4f}")

soft_test_probs = best_alpha * xgb_test_probs + (1 - best_alpha) * cnn_test_probs
soft_test_preds = soft_test_probs.argmax(1)
soft_test_acc   = accuracy_score(y_test, soft_test_preds)
soft_test_f1    = f1_score(y_test, soft_test_preds, average="macro")
print(f"  Soft Vote test_acc={soft_test_acc:.4f}  test_f1={soft_test_f1:.4f}")

# ── 4. Stacking Meta-Learner ──────────────────────────────────────────────────
print("── Stacking Meta-Learner ──")

# Stack val predictions to train meta-learner
# Shape: (n_val, 2 * n_classes)
val_stack  = np.hstack([xgb_val_probs,  cnn_val_probs])
test_stack = np.hstack([xgb_test_probs, cnn_test_probs])

meta = LogisticRegression(
    C=1.0, max_iter=1000, solver="lbfgs",
    multi_class="multinomial", random_state=42
)
meta.fit(val_stack, y_val)
stack_test_preds = meta.predict(test_stack)
stack_test_acc   = accuracy_score(y_test, stack_test_preds)
stack_test_f1    = f1_score(y_test, stack_test_preds, average="macro")
print(f"  Stacking test_acc={stack_test_acc:.4f}  test_f1={stack_test_f1:.4f}")

# ── 5. Comparison Table ───────────────────────────────────────────────────────
comparison = pd.DataFrame([
    {"Model": "XGBoost (best trial)",    "Test Accuracy": xgb_test_acc,   "Test F1 (macro)": xgb_test_f1},
    {"Model": "1D-CNN (best trial)",     "Test Accuracy": cnn_test_acc,   "Test F1 (macro)": cnn_test_f1},
    {"Model": "Soft Voting Ensemble",    "Test Accuracy": soft_test_acc,  "Test F1 (macro)": soft_test_f1},
    {"Model": "Stacking Ensemble",       "Test Accuracy": stack_test_acc, "Test F1 (macro)": stack_test_f1},
])
comparison = comparison.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
print("\n── Model Comparison ──")
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(PLOTS_DIR, "model_comparison.csv"), index=False)

# ── 6. Plots ──────────────────────────────────────────────────────────────────

# 6a — Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(comparison))
w     = 0.35
colors_acc = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
colors_f1  = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]
bars1 = ax.bar(x - w/2, comparison["Test Accuracy"],  w, label="Accuracy",    color=colors_acc)
bars2 = ax.bar(x + w/2, comparison["Test F1 (macro)"], w, label="F1 Macro", color=colors_f1)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(comparison["Model"], rotation=15, ha="right")
ax.set_ylim(0.85, 1.01)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — TCGA Pan-Cancer Classification")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
plt.close()
print("✓  Saved model_comparison.png")

# 6b — Confusion matrix for best ensemble
best_preds = soft_test_preds if soft_test_acc >= stack_test_acc else stack_test_preds
best_label = "Soft Voting" if soft_test_acc >= stack_test_acc else "Stacking"
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(14, 12))
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
    ax=ax, colorbar=True, xticks_rotation=45, cmap="YlOrRd"
)
ax.set_title(f"Ensemble ({best_label}) Confusion Matrix — Test Set", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ensemble_confusion_matrix.png"), dpi=150)
plt.close()
print("✓  Saved ensemble_confusion_matrix.png")

# 6c — Per-class F1 comparison (XGB vs CNN vs Ensemble)
from sklearn.metrics import f1_score as f1
class_f1_xgb  = f1(y_test, xgb_model.predict(X_test),   average=None)
class_f1_cnn  = f1(y_test, cnn_test_probs.argmax(1),     average=None)
class_f1_ens  = f1(y_test, best_preds,                   average=None)

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(n_classes)
w = 0.28
ax.bar(x - w,   class_f1_xgb, w, label="XGBoost",  color="#2196F3", alpha=0.85)
ax.bar(x,       class_f1_cnn, w, label="1D-CNN",    color="#4CAF50", alpha=0.85)
ax.bar(x + w,   class_f1_ens, w, label="Ensemble",  color="#FF9800", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(le.classes_, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 Score: XGBoost vs 1D-CNN vs Ensemble")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "per_class_f1.png"), dpi=150)
plt.close()
print("✓  Saved per_class_f1.png")

print(f"\n✓  All ensemble outputs in {PLOTS_DIR}")
