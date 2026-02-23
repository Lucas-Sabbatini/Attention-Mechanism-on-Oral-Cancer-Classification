"""
Statistical comparison of all classifiers via Friedman + Nemenyi + CD diagram.

Runs all 8 models under the same 10-fold stratified CV, then:
  1. Friedman test + Nemenyi post-hoc (metric: mean_se_sp)
  2. CD diagram saved to postProcess/img/cd_diagram.png
  3. Paired t-test: BioSpectralFormer vs LightGBM (detailed pairwise view)
"""

import warnings
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

from preProcess.baseline_correction import BaselineCorrection
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization

from models.model_xgb import XGBModel
from models.model_svm import SVMRBFModel
from models.model_tabpfn import TabPFNModel
from models.model_catboost import CatBoostModel
from models.model_realmlp import RealMLPModel
from models.model_tabm import TabMModel
from models.model_lightgbm import LightGBMModel

from transformer.model import BioSpectralFormer
from transformer.visualize import paired_t_test

from postProcess.friedman_nemenyi import friedman_nemenyi_test, plot_cd_diagram

# ---------------------------------------------------------------------------
# Suppress noisy output from third-party libraries
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------
dataset = np.loadtxt("dataset_cancboca.dat")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
y = np.where(y == -1, 0, 1)

baseline = BaselineCorrection().asls_baseline(X)
X = X - baseline

X = Normalization().peak_normalization(X, 1660.0, 1630.0)
X = WavenumberTruncator().trucate_range(X, 3050.0, 850.0)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
spectra_model   = BioSpectralFormer(num_spectral_points=X.shape[1])
xgb_model       = XGBModel()
svm_model       = SVMRBFModel()
tabpfn_model    = TabPFNModel()
catboost_model  = CatBoostModel()
realmlp_model   = RealMLPModel()
tabm_model      = TabMModel()
lightgbm_model  = LightGBMModel()

model_names = [
    "BioSpectralFormer", "XGBoost", "SVM-RBF", "TabPFN",
    "CatBoost", "RealMLP", "TabM", "LightGBM",
]
models = [
    spectra_model, xgb_model, svm_model, tabpfn_model,
    catboost_model, realmlp_model, tabm_model, lightgbm_model,
]

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
all_results: dict[str, list[tuple]] = {name: [] for name in model_names}

if hasattr(spectra_model, "reset_fold_counter"):
    spectra_model.reset_fold_counter()

print("Running 10-fold cross-validation for all models...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"  Fold {fold}/10", end="\r", flush=True)
    for name, model in zip(model_names, models):
        metrics = model.evaluate(X_train, X_test, y_train, y_test)
        all_results[name].append(metrics)

print("\nCross-validation complete.\n")

# ---------------------------------------------------------------------------
# Step 1: Friedman + Nemenyi
# ---------------------------------------------------------------------------
fn_results = friedman_nemenyi_test(
    all_results,
    metric="mean_se_sp",
    alpha=0.05,
    verbose=True,
)

# ---------------------------------------------------------------------------
# Step 2: CD diagram
# ---------------------------------------------------------------------------
plot_cd_diagram(
    avg_ranks=fn_results["avg_ranks"],
    cd=fn_results["cd"],
    title="Critical Difference Diagram — Mean(SE, SP)",
    alpha=0.05,
    save_path="postProcess/img/cd_diagram.png",
)

# ---------------------------------------------------------------------------
# Step 3: Detailed pairwise comparison — BioSpectralFormer vs LightGBM
# ---------------------------------------------------------------------------
paired_t_test(
    all_results["BioSpectralFormer"],
    all_results["LightGBM"],
    model1_name="BioSpectralFormer",
    model2_name="LightGBM",
)
