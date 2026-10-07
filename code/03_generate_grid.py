"""
03_generate_grid.py
Builds the hyperparameter grid JSON that each Slurm array task reads.
Run ONCE on the login node before submitting the array job.

    python 03_generate_grid.py

Outputs:
    $SCRATCH/hyperparam_grid.json   — list of param dicts
    $SCRATCH/n_trials.txt           — integer, used in sbatch --array flag
"""

import itertools
import json
import os
import random

SCRATCH   = os.environ.get("SCRATCH", "./data")
os.makedirs(SCRATCH, exist_ok=True)

# ── Define Search Space ───────────────────────────────────────────────────────
# Each list entry is a candidate value.  Full grid = product of all lists.
search_space = {
    "n_estimators"     : [300, 600, 1000],
    "max_depth"        : [4, 6, 8],
    "learning_rate"    : [0.01, 0.05, 0.1, 0.2],
    "subsample"        : [0.7, 0.85, 1.0],
    "colsample_bytree" : [0.5, 0.7, 0.9],
    "min_child_weight" : [1, 5, 10],
    "reg_alpha"        : [0, 0.1, 1.0],
    "reg_lambda"       : [1.0, 5.0],
}

keys   = list(search_space.keys())
values = list(search_space.values())

all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

# ── Optionally sub-sample (full grid can be huge) ────────────────────────────
MAX_TRIALS = 60          # ← tune to your 48-hour budget / node count
random.seed(42)
if len(all_combos) > MAX_TRIALS:
    grid = random.sample(all_combos, MAX_TRIALS)
    print(f"Full grid has {len(all_combos)} combos — randomly sampling {MAX_TRIALS}.")
else:
    grid = all_combos
    print(f"Full grid: {len(grid)} trials.")

# ── Write ─────────────────────────────────────────────────────────────────────
grid_path   = os.path.join(SCRATCH, "hyperparam_grid.json")
ntrial_path = os.path.join(SCRATCH, "n_trials.txt")

with open(grid_path, "w") as f:
    json.dump(grid, f, indent=2)

with open(ntrial_path, "w") as f:
    f.write(str(len(grid)))

print(f"✓  Grid written  → {grid_path}")
print(f"✓  n_trials      → {len(grid)}")
print(f"\nNext step:  sbatch --array=0-{len(grid)-1} sweep_array.slurm")
