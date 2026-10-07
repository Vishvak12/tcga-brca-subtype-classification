"""
01_download_and_preprocess.py — BRCA PAM50 subtype classification
Loads TCGA BRCA expression + PAM50 calls from UCSC Xena, joins on barcode,
saves train/val/test splits as .npy files.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

SCRATCH  = os.environ.get("SCRATCH", "./data")
BRCA_DIR = os.path.join(SCRATCH, "brca")
PROC_DIR = os.path.join(SCRATCH, "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ── 1. Load expression matrix (genes × samples) ──────────────────────────────
print("Loading BRCA expression matrix …")
X_raw = pd.read_csv(os.path.join(BRCA_DIR, "brca_expression.tsv"),
                    sep="\t", index_col=0)
print(f"  Raw shape (genes × samples): {X_raw.shape}")

# Transpose so rows = samples, columns = genes
X_raw = X_raw.T
print(f"  Transposed shape (samples × genes): {X_raw.shape}")

# ── 2. Load phenotype file, extract PAM50 ────────────────────────────────────
print("Loading phenotype file …")
pheno = pd.read_csv(os.path.join(BRCA_DIR, "brca_phenotype.tsv"),
                    sep="\t", index_col=0, low_memory=False)
print(f"  Phenotype rows: {pheno.shape[0]}")

pam50 = pheno["PAM50Call_RNAseq"].dropna()
print(f"  Samples with PAM50 call: {len(pam50)}")
print(f"  Class distribution:\n{pam50.value_counts()}")

# ── 3. Inner-join on barcode ─────────────────────────────────────────────────
common = X_raw.index.intersection(pam50.index)
print(f"\n  Samples in both expression + PAM50: {len(common)}")

X_aligned = X_raw.loc[common]
y_aligned = pam50.loc[common]

# Drop any "Normal" if it's very small (often <10 samples, hard to learn)
class_counts = y_aligned.value_counts()
print(f"\n  Aligned class distribution:\n{class_counts}")
tiny = class_counts[class_counts < 10].index.tolist()
if tiny:
    print(f"  Dropping classes with <10 samples: {tiny}")
    keep_mask = ~y_aligned.isin(tiny)
    X_aligned = X_aligned[keep_mask]
    y_aligned = y_aligned[keep_mask]

# ── 4. Handle missing values & filter low-variance genes ─────────────────────
X_aligned = X_aligned.fillna(0)            # rare NaNs in expression
variance   = X_aligned.var(axis=0)
X_filtered = X_aligned.loc[:, variance > 0.01]
print(f"  After variance threshold: {X_filtered.shape[1]} genes remaining")

# ── 5. Scale & encode ────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)
le       = LabelEncoder()
y_enc    = le.fit_transform(y_aligned.values)
print(f"  Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 6. Train / Val / Test split ──────────────────────────────────────────────
X_tv, X_test, y_tv, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15, stratify=y_tv, random_state=42)
print(f"\nSplit sizes  →  train:{len(y_train)}  val:{len(y_val)}  test:{len(y_test)}")

# ── 7. Save ──────────────────────────────────────────────────────────────────
np.save(os.path.join(PROC_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROC_DIR, "X_val.npy"),   X_val)
np.save(os.path.join(PROC_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(PROC_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROC_DIR, "y_val.npy"),   y_val)
np.save(os.path.join(PROC_DIR, "y_test.npy"),  y_test)
joblib.dump(scaler, os.path.join(PROC_DIR, "scaler.pkl"))
joblib.dump(le,     os.path.join(PROC_DIR, "label_encoder.pkl"))
pd.Series(X_filtered.columns.tolist()).to_csv(os.path.join(PROC_DIR, "gene_names.csv"), index=False)
print("\n✓  All artefacts saved to", PROC_DIR)

# ── 8. Sanity baseline ───────────────────────────────────────────────────────
print("\n── Sanity baseline: Random Forest (100 trees) ──")
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics   import accuracy_score, classification_report
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
acc = accuracy_score(y_val, rf.predict(X_val))
print(f"  Validation accuracy: {acc:.4f}")
print("\n", classification_report(y_val, rf.predict(X_val), target_names=le.classes_))

if acc > 0.96:
    print("  ⚠  Still too easy — consider stratifying differently or adding noise.")
elif acc < 0.6:
    print("  ⚠  Very low — check class balance and label join.")
else:
    print("  ✓  Good headroom — XGBoost sweep should show meaningful gains.")
