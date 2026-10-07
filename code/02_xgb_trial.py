"""
02_xgb_trial.py
A single XGBoost hyperparameter trial — launched by each Slurm array task.

Each job reads its own hyper-params from a pre-generated JSON grid file,
trains on the full training set, and writes its val-accuracy + params
to a results row in RESULTS_DIR.

Usage (called automatically by sbatch array script):
    python 02_xgb_trial.py --trial_id $SLURM_ARRAY_TASK_ID
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
import joblib

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--trial_id", type=int, required=True)
args = parser.parse_args()

SCRATCH      = os.environ.get("SCRATCH", "./data")
PROC_DIR     = os.path.join(SCRATCH, "processed")
RESULTS_DIR  = os.path.join(SCRATCH, "results")
GRID_FILE    = os.path.join(SCRATCH, "hyperparam_grid.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load Grid ─────────────────────────────────────────────────────────────────
with open(GRID_FILE) as f:
    grid = json.load(f)

params = grid[args.trial_id]
print(f"[Trial {args.trial_id}]  params → {params}")

# ── Load Data ─────────────────────────────────────────────────────────────────
X_train = np.load(os.path.join(PROC_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(PROC_DIR, "X_val.npy"))
y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"))
y_val   = np.load(os.path.join(PROC_DIR, "y_val.npy"))

n_classes = len(np.unique(y_train))

# ── Build & Train ─────────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    objective        = "multi:softprob",
    num_class        = n_classes,
    eval_metric      = "mlogloss",
    tree_method      = "hist",          # fast histogram method
    device           = "cuda",           # swap to "cuda" if GPU nodes available
    n_estimators     = params["n_estimators"],
    max_depth        = params["max_depth"],
    learning_rate    = params["learning_rate"],
    subsample        = params["subsample"],
    colsample_bytree = params["colsample_bytree"],
    min_child_weight = params["min_child_weight"],
    reg_alpha        = params["reg_alpha"],     # L1
    reg_lambda       = params["reg_lambda"],    # L2
    use_label_encoder= False,
    verbosity        = 0,
    n_jobs           = 1,
    random_state     = 42,
    early_stopping_rounds = 20,
)

t0 = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
elapsed = time.time() - t0

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds    = model.predict(X_val)
val_acc  = accuracy_score(y_val, preds)
val_f1   = f1_score(y_val, preds, average="macro")

print(f"[Trial {args.trial_id}]  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  time={elapsed:.1f}s")

# ── Save Model & Results ──────────────────────────────────────────────────────
model_path = os.path.join(RESULTS_DIR, f"model_trial_{args.trial_id:04d}.json")
model.save_model(model_path)

result = {
    "trial_id"        : args.trial_id,
    "val_accuracy"    : val_acc,
    "val_f1_macro"    : val_f1,
    "best_iteration"  : model.best_iteration,
    "train_time_s"    : elapsed,
    **params,
}
result_path = os.path.join(RESULTS_DIR, f"result_{args.trial_id:04d}.json")
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"[Trial {args.trial_id}]  Saved → {result_path}")
