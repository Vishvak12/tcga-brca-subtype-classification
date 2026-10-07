"""
05b_generate_cnn_grid.py
Generate the CNN hyperparameter search grid.
Run once on the login node before submitting the CNN array job.

    python 05b_generate_cnn_grid.py
"""

import itertools, json, os, random

SCRATCH = os.environ.get("SCRATCH", "./data")
os.makedirs(SCRATCH, exist_ok=True)

search_space = {
    "n_filters"      : [64, 128, 256],
    "kernel_size"    : [5, 7, 11],
    "n_blocks"       : [2, 3, 4],
    "dropout"        : [0.3, 0.5, 0.6],
    "learning_rate"  : [5e-4, 1e-3, 3e-3],
    "batch_size"     : [16, 32],
    "weight_decay"   : [1e-4, 1e-3],
    "label_smoothing": [0.05, 0.1],
    "epochs"         : [150],       # fixed; early stopping handles actual duration
    "patience"       : [20],
}

keys   = list(search_space.keys())
values = list(search_space.values())
all_combos = [dict(zip(keys, c)) for c in itertools.product(*values)]

MAX_TRIALS = 30
random.seed(7)
grid = random.sample(all_combos, min(MAX_TRIALS, len(all_combos)))

grid_path = os.path.join(SCRATCH, "cnn_grid.json")
with open(grid_path, "w") as f:
    json.dump(grid, f, indent=2)

with open(os.path.join(SCRATCH, "n_cnn_trials.txt"), "w") as f:
    f.write(str(len(grid)))

print(f"✓  CNN grid: {len(grid)} trials  →  {grid_path}")
print(f"\nNext:  sbatch --array=0-{len(grid)-1} cnn_array.slurm")
