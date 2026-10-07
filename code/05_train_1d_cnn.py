"""
05_train_1d_cnn.py
Train a 1D-CNN on TCGA gene expression data.

Key design choices for small sample size (800 samples):
  - Heavy dropout (0.4–0.6) at every layer
  - Early stopping on val loss
  - Gene ordering by chromosomal position (optional but recommended)
  - Label smoothing to prevent overconfident predictions
  - Cosine annealing LR schedule

Usage:
    python 05_train_1d_cnn.py [--config configs/cnn_config.json]

Or let the Slurm array call it with different configs:
    python 05_train_1d_cnn.py --trial_id $SLURM_ARRAY_TASK_ID
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import joblib

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--trial_id", type=int, default=0,
                    help="Index into CNN hyperparam grid JSON")
parser.add_argument("--config",   type=str, default=None,
                    help="Path to a single JSON config (overrides grid)")
args = parser.parse_args()

SCRATCH      = os.environ.get("SCRATCH", "./data")
PROC_DIR     = os.path.join(SCRATCH, "processed")
RESULTS_DIR  = os.path.join(SCRATCH, "results_cnn")
CNN_GRID     = os.path.join(SCRATCH, "cnn_grid.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
if args.config:
    with open(args.config) as f:
        cfg = json.load(f)
elif os.path.exists(CNN_GRID):
    with open(CNN_GRID) as f:
        grid = json.load(f)
    cfg = grid[args.trial_id]
else:
    # Sensible defaults if no grid found
    cfg = {
        "n_filters"     : 128,
        "kernel_size"   : 7,
        "n_blocks"      : 3,
        "dropout"       : 0.5,
        "learning_rate" : 1e-3,
        "batch_size"    : 32,
        "weight_decay"  : 1e-4,
        "label_smoothing": 0.1,
        "epochs"        : 150,
        "patience"      : 20,
    }

print(f"[Trial {args.trial_id}] Config: {cfg}")

# ── Load Data ─────────────────────────────────────────────────────────────────
# Load reordered+filtered arrays if available, else fall back
_suffix = "_cnn" if os.path.exists(os.path.join(PROC_DIR, "X_train_cnn.npy")) else ""
X_train = np.load(os.path.join(PROC_DIR, f"X_train{_suffix}.npy")).astype(np.float32)
X_val   = np.load(os.path.join(PROC_DIR, f"X_val{_suffix}.npy")).astype(np.float32)
X_test  = np.load(os.path.join(PROC_DIR, f"X_test{_suffix}.npy")).astype(np.float32)
print(f"  Using arrays with suffix: '{_suffix}'  (shape: {X_train.shape})")
y_train = np.load(os.path.join(PROC_DIR, "y_train.npy")).astype(np.int64)
y_val   = np.load(os.path.join(PROC_DIR, "y_val.npy")).astype(np.int64)
y_test  = np.load(os.path.join(PROC_DIR, "y_test.npy")).astype(np.int64)

n_genes   = X_train.shape[1]
n_classes = len(np.unique(y_train))
print(f"Genes: {n_genes}  |  Classes: {n_classes}  |  "
      f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

# ── Optional: sort genes by chromosomal position ──────────────────────────────
# If you have a gene annotation file (e.g. from Ensembl), sort columns here
# so the 1D-CNN sees genomically adjacent genes as spatially adjacent.
#
# gene_names = pd.read_csv(os.path.join(PROC_DIR, "gene_names.csv"))["0"].tolist()
# annotation = pd.read_csv("gene_positions.csv")   # chr, start, gene_name
# ordered_idx = annotation.set_index("gene_name").loc[gene_names, "start"].argsort()
# X_train = X_train[:, ordered_idx]
# X_val   = X_val[:,   ordered_idx]
# X_test  = X_test[:,  ordered_idx]
# print("✓  Genes sorted by chromosomal position")

# ── Dataset / DataLoader ──────────────────────────────────────────────────────
def make_loader(X, y, batch_size, shuffle):
    Xt = torch.from_numpy(X).unsqueeze(1)   # (N, 1, genes) — channel dim for Conv1d
    yt = torch.from_numpy(y)
    return DataLoader(TensorDataset(Xt, yt),
                      batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=(device.type == "cuda"))

train_loader = make_loader(X_train, y_train, cfg["batch_size"], shuffle=True)
val_loader   = make_loader(X_val,   y_val,   cfg["batch_size"], shuffle=False)
test_loader  = make_loader(X_test,  y_test,  cfg["batch_size"], shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    """
    Residual block: Conv → BN → ReLU → Dropout → Conv → BN → (+skip) → ReLU
    Keeps gradient flow healthy even with deep stacks.
    """
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        pad = kernel_size // 2
        self.conv1  = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn1    = nn.BatchNorm1d(channels)
        self.conv2  = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2    = nn.BatchNorm1d(channels)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class GeneCNN(nn.Module):
    """
    Architecture:
        Input  (N, 1, G)
        → Stem conv: 1 → n_filters, kernel 15, stride 2  (aggressive downsampling)
        → n_blocks residual blocks
        → Global average pool  (N, n_filters, 1) → (N, n_filters)
        → FC 512 → Dropout → FC n_classes
    """
    def __init__(self, n_genes, n_classes, n_filters, kernel_size, n_blocks, dropout):
        super().__init__()

        # Stem — wide kernel to capture broad co-expression patterns
        self.stem = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks
        self.blocks = nn.Sequential(
            *[ResBlock1D(n_filters, kernel_size, dropout) for _ in range(n_blocks)]
        )

        # Global average pool → classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_filters, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x)
        return self.head(x)

    def get_penultimate(self, x):
        """Return the 512-dim embedding before the final linear — used in ensemble."""
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x)
        x = nn.Flatten()(x)
        x = F.relu(self.head[1](x))   # first Linear + ReLU
        return x


model = GeneCNN(
    n_genes    = n_genes,
    n_classes  = n_classes,
    n_filters  = cfg["n_filters"],
    kernel_size= cfg["kernel_size"],
    n_blocks   = cfg["n_blocks"],
    dropout    = cfg["dropout"],
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")

# ── Loss, Optimizer, Scheduler ────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = cfg["learning_rate"],
    weight_decay = cfg["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg["epochs"], eta_min=1e-6
)

# ── Training Loop ─────────────────────────────────────────────────────────────
best_val_loss = float("inf")
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_model_path = os.path.join(RESULTS_DIR, f"cnn_trial_{args.trial_id:04d}_best.pt")

t0 = time.time()
for epoch in range(1, cfg["epochs"] + 1):

    # — Train —
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(yb)
    train_loss /= len(y_train)

    # — Validate —
    model.eval()
    val_loss = 0.0
    val_preds = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits  = model(Xb)
            val_loss += criterion(logits, yb).item() * len(yb)
            val_preds.extend(logits.argmax(1).cpu().numpy())
    val_loss /= len(y_val)
    val_acc   = accuracy_score(y_val, val_preds)

    scheduler.step()
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    # — Early stopping —
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

elapsed = time.time() - t0

# ── Test Evaluation ───────────────────────────────────────────────────────────
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
test_preds, test_probs = [], []
with torch.no_grad():
    for Xb, _ in test_loader:
        logits = model(Xb.to(device))
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        test_probs.extend(probs)
        test_preds.extend(logits.argmax(1).cpu().numpy())

test_acc = accuracy_score(y_test, test_preds)
test_f1  = f1_score(y_test, test_preds, average="macro")
print(f"\n── Test Results ──")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  F1 Macro : {test_f1:.4f}")
print(f"  Time     : {elapsed:.1f}s")

# ── Save Everything ───────────────────────────────────────────────────────────
# Probability outputs (needed by ensemble script)
np.save(os.path.join(RESULTS_DIR, f"cnn_test_probs_{args.trial_id:04d}.npy"),
        np.array(test_probs))

result = {
    "trial_id"      : args.trial_id,
    "val_accuracy"  : max(history["val_acc"]),
    "test_accuracy" : test_acc,
    "test_f1_macro" : test_f1,
    "train_time_s"  : elapsed,
    "epochs_trained": len(history["train_loss"]),
    **cfg,
}
with open(os.path.join(RESULTS_DIR, f"cnn_result_{args.trial_id:04d}.json"), "w") as f:
    json.dump(result, f, indent=2)

pd.DataFrame(history).to_csv(
    os.path.join(RESULTS_DIR, f"cnn_history_{args.trial_id:04d}.csv"), index=False
)

print(f"\n✓  Saved model → {best_model_path}")
