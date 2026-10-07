# TCGA Pan-Cancer — XGBoost + 1D-CNN Ensemble Pipeline
### 48-Hour HPC Sprint · Slurm · XGBoost · PyTorch · Soft Voting + Stacking

---

## File Map

```
tcga_pipeline/
├── 01_download_and_preprocess.py   # Data download, cleaning, train/val/test split
├── 02_xgb_trial.py                 # Single XGBoost trial (called by array job)
├── 03_generate_grid.py             # XGBoost hyperparameter grid
├── 04_analyze_results.py           # XGBoost results: metrics, UMAP, biomarkers
├── 05_train_1d_cnn.py              # Single CNN trial (called by array job)
├── 05b_generate_cnn_grid.py        # CNN hyperparameter grid
├── 06_ensemble.py                  # Soft voting + stacking + comparison plots
├── sweep_array.slurm               # Slurm array for XGBoost sweep
├── cnn_array.slurm                 # Slurm array for CNN sweep (GPU)
└── README.md
```

---

## Full Execution Order

### Day 1 — Setup & Launch

```bash
# 0. Environment (once)
module load python/3.11 cuda/12.1
python -m venv $HOME/envs/tcga && source $HOME/envs/tcga/bin/activate
pip install xgboost scikit-learn pandas numpy joblib matplotlib umap-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

export SCRATCH=/scratch/$USER/tcga

# 1. Download & preprocess
python 01_download_and_preprocess.py

# 2. Generate both grids
python 03_generate_grid.py          # XGBoost grid
python 05b_generate_cnn_grid.py     # CNN grid

# 3. Launch both sweeps simultaneously
mkdir -p logs
sbatch sweep_array.slurm            # XGBoost: 60 CPU trials
sbatch cnn_array.slurm              # CNN:     30 GPU trials

# Monitor
watch -n 30 squeue -u $USER
```

### Day 2 — Analysis & Ensemble

```bash
python 04_analyze_results.py        # UMAP, biomarkers, sweep plots
python 06_ensemble.py               # Combine models, comparison plots
```

---

## Architecture

```
Gene expression vector  (N x 20,000 genes)
           |
    +------+------+
    |             |
 XGBoost       1D-CNN
 (trees)       (ResBlocks)
    |             |
 P(class)      P(class)
    |             |
    +------+------+
           |
    Soft Voting  (alpha*XGB + (1-alpha)*CNN)
    OR
    Stacking     (Logistic Regression meta-learner)
           |
    Final prediction
```

Why this works:
- XGBoost finds the globally most discriminative genes
- 1D-CNN captures local co-expression patterns between adjacent genes
- Their errors are complementary — ensemble beats both individually

---

## CNN Regularization (for 800-sample dataset)

- Dropout 0.3-0.6 at every residual block and classifier head
- Label smoothing (0.05-0.1) prevents overconfident predictions
- Cosine annealing LR schedule
- Early stopping (patience=20 epochs)
- Gradient clipping (norm=1.0)

---

## Key Gotchas

"No module named train_1d_cnn_module" in 06_ensemble.py
  -> Copy the GeneCNN class from 05_train_1d_cnn.py into 06_ensemble.py directly

CNN underperforms XGBoost on UCI 5-class dataset
  -> Expected on 800 samples. Ensemble still gains.
  -> On the 12K HuggingFace dataset the gap closes significantly.

Gene ordering for CNN
  -> Uncomment the chromosomal-sort block in 05_train_1d_cnn.py if you have
     a gene annotation CSV. Without it the CNN still works but ignores genomic adjacency.
