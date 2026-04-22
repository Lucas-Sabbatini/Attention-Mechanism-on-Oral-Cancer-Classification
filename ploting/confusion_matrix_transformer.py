import sys
import warnings
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import StratifiedKFold

# ── project root on the path ────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

from preProcess.baseline_correction import BaselineCorrection
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization
from transformer.model import BioSpectralFormer

# ── 1. Load & preprocess (mirrors main.py exactly) ──────────────────────────
dataset = np.loadtxt(project_root / "dataset_cancboca.dat")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
y = np.where(y == -1, 0, 1)

X = X - BaselineCorrection().asls_baseline(X)
X = Normalization().peak_normalization(X, 1660.0, 1630.0)
X = WavenumberTruncator().trucate_range(X, 3050.0, 850.0)

# ── 2. 10-fold stratified cross-validation ──────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = BioSpectralFormer(num_spectral_points=X.shape[1])
model.reset_fold_counter()

for train_idx, test_idx in skf.split(X, y):
    model.evaluate(X[train_idx], X[test_idx], y[train_idx], y[test_idx])

# ── 3. Aggregate predictions from all folds ─────────────────────────────────
all_true = []
all_pred = []
for fold in BioSpectralFormer.get_fold_diagnostics():
    all_true.extend(fold["true_labels"])
    all_pred.extend(fold["predictions"])

all_true = np.array(all_true)
all_pred = np.array(all_pred)

# Build 2x2 confusion matrix:  rows = true, cols = predicted
# Labels: 0 = Healthy, 1 = Cancerous
labels = [0, 1]
cm = np.zeros((2, 2), dtype=int)
for t, p in zip(all_true, all_pred):
    cm[t, p] += 1

tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
total       = cm.sum()
accuracy    = (tp + tn) / total
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / SE
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # SP
precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
f1          = 2 * precision * sensitivity / (precision + sensitivity) \
              if (precision + sensitivity) > 0 else 0.0
mean_se_sp  = (sensitivity + specificity) / 2

# ── 4. Plot ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Healthy", "Cancerous"]
CMAP        = plt.cm.Blues

fig, ax = plt.subplots(figsize=(6.5, 5.8))

img = ax.imshow(cm, interpolation="nearest", cmap=CMAP, vmin=0, vmax=cm.max())
cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Count", fontsize=10)

# Cell annotations
thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        count = cm[i, j]
        pct   = count / total * 100
        color = "white" if count > thresh else "black"
        ax.text(j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=13,
                fontweight="bold", color=color)

# Axes decoration
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_yticklabels(CLASS_NAMES, fontsize=11)
ax.set_xlabel("Predicted label", fontsize=12, labelpad=8)
ax.set_ylabel("True label", fontsize=12, labelpad=8)
ax.set_title("BioSpectralFormer — Confusion Matrix\n10-fold Stratified Cross-Validation",
             fontsize=13, fontweight="bold", pad=12)

# ── 5. Metrics text box ──────────────────────────────────────────────────────
metrics_text = (
    f"Accuracy   : {accuracy:.3f}\n"
    f"Sensitivity: {sensitivity:.3f}\n"
    f"Specificity: {specificity:.3f}\n"
    f"Mean SE/SP : {mean_se_sp:.3f}\n"
    f"Precision  : {precision:.3f}\n"
    f"F1-score   : {f1:.3f}\n"
    f"n = {total} samples"
)
props = dict(boxstyle="round,pad=0.5", facecolor="#f0f4ff", edgecolor="#9bb0d4", alpha=0.9)
ax.text(1.38, 0.5, metrics_text,
        transform=ax.transAxes,
        fontsize=9.5, verticalalignment="center",
        fontfamily="monospace", bbox=props)

plt.tight_layout()

output_path = Path(__file__).parent / "img" / "confusion_matrix_transformer.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved to {output_path}")
