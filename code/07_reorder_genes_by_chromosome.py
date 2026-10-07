"""
07_reorder_genes_by_chromosome.py
Reorders gene columns by chromosomal position so the 1D-CNN has
biological spatial structure to learn from. Also selects top-K variance
genes to reduce noise.

Saves CNN-specific files (X_train_cnn.npy etc.) leaving the XGBoost files
untouched.

    python 07_reorder_genes_by_chromosome.py
"""
import os
import urllib.request
import numpy as np
import pandas as pd

SCRATCH  = os.environ.get("SCRATCH", "/scratch/vp2781")
PROC_DIR = os.path.join(SCRATCH, "processed")
BRCA_DIR = os.path.join(SCRATCH, "brca")
os.makedirs(BRCA_DIR, exist_ok=True)

TOP_K = 2000   # number of genes to keep for CNN

# ── 1. Load existing processed data ──────────────────────────────────────────
print("Loading processed arrays …")
X_train = np.load(os.path.join(PROC_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(PROC_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy"))
gene_names = pd.read_csv(os.path.join(PROC_DIR, "gene_names.csv"))["0"].tolist()
print(f"  Shape: {X_train.shape}  (samples x genes)")

# ── 2. Get gene-to-chromosome positions (UCSC Xena probemap) ─────────────────
ANNOT_PATH = os.path.join(BRCA_DIR, "gene_positions.tsv")
if not os.path.exists(ANNOT_PATH):
    print("\nDownloading gene-position annotation from UCSC Xena …")
    url = "https://tcga-xena-hub.s3.dualstack.us-east-1.amazonaws.com/download/probeMap%2Fhugo_gencode_good_hg19_V24lift37_probemap"
    urllib.request.urlretrieve(url, ANNOT_PATH)
    print("  Done.")
annot = pd.read_csv(ANNOT_PATH, sep="\t")
print(f"  Annotation rows: {len(annot)}")
print(f"  Columns: {list(annot.columns)}")

# Standardize column names
col_gene  = "gene"       if "gene" in annot.columns else annot.columns[1]
col_chrom = "chrom"      if "chrom" in annot.columns else "chromosome"
col_start = "chromStart" if "chromStart" in annot.columns else "start"

def chrom_to_int(c):
    c = str(c).replace("chr", "")
    if c == "X":   return 23
    if c == "Y":   return 24
    if c in ("M", "MT"): return 25
    try:    return int(c)
    except: return 99

annot = annot.dropna(subset=[col_gene, col_chrom, col_start])
annot["chrom_int"] = annot[col_chrom].apply(chrom_to_int)
annot = annot.drop_duplicates(subset=[col_gene], keep="first")
pos_map = annot.set_index(col_gene)[["chrom_int", col_start]].to_dict("index")

matched   = [g for g in gene_names if g in pos_map]
unmatched = [g for g in gene_names if g not in pos_map]
print(f"\n  Matched genes  : {len(matched)} / {len(gene_names)}")
print(f"  Unmatched      : {len(unmatched)}  (will be placed at the end)")

# ── 3. Sort matched genes by (chromosome, start position) ────────────────────
matched_sorted = sorted(matched, key=lambda g: (pos_map[g]["chrom_int"], pos_map[g][col_start]))
new_order_names = matched_sorted + unmatched
new_order_idx   = [gene_names.index(g) for g in new_order_names]

# ── 4. Reorder & select top variance genes ───────────────────────────────────
print("\nReordering columns and selecting top variance genes …")
X_train_ord = X_train[:, new_order_idx]
X_val_ord   = X_val[:,   new_order_idx]
X_test_ord  = X_test[:,  new_order_idx]

variances = X_train_ord.var(axis=0)
top_idx   = np.sort(np.argsort(variances)[-TOP_K:])   # keep in genomic order!

X_train_cnn = X_train_ord[:, top_idx].astype(np.float32)
X_val_cnn   = X_val_ord[:,   top_idx].astype(np.float32)
X_test_cnn  = X_test_ord[:,  top_idx].astype(np.float32)

selected_names = [new_order_names[i] for i in top_idx]
print(f"  CNN input shape: {X_train_cnn.shape}")

# ── 5. Save CNN-specific arrays ──────────────────────────────────────────────
np.save(os.path.join(PROC_DIR, "X_train_cnn.npy"), X_train_cnn)
np.save(os.path.join(PROC_DIR, "X_val_cnn.npy"),   X_val_cnn)
np.save(os.path.join(PROC_DIR, "X_test_cnn.npy"),  X_test_cnn)
pd.Series(selected_names).to_csv(os.path.join(PROC_DIR, "cnn_gene_order.csv"), index=False)

print("\n[Done] Saved:")
print(f"  X_train_cnn.npy  ({X_train_cnn.shape})")
print(f"  X_val_cnn.npy    ({X_val_cnn.shape})")
print(f"  X_test_cnn.npy   ({X_test_cnn.shape})")
print(f"  cnn_gene_order.csv  ({len(selected_names)} gene names in genomic order)")
print(f"\nXGBoost files untouched - it can still use the full {X_train.shape[1]}-gene matrix.")
print("\nNext: patch 05_train_1d_cnn.py to use the *_cnn.npy files, then re-run sweep.")
