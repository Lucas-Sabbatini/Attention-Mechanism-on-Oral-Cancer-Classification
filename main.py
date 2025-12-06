from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
import os

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
from models.model import BaseClassifierModel

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress PyTorch Lightning verbose output
os.environ['PYTHONWARNINGS'] = 'ignore'

dataset_path = "dataset_cancboca.dat"

# Sample Data
dataset = np.loadtxt(dataset_path)
X = dataset[:,:-1]
y = dataset[:,-1].astype(int)
y = np.where(y == -1, 0, 1)

#Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
lst_accu_stratified = []

#Baseline correction
baseline = BaselineCorrection().asls_baseline(X)
X = X - baseline

# Normalize data
normalizer = Normalization()
X = normalizer.peak_normalization(X, 1660.0, 1630.0)
#X = normalizer.mean_normalization(X)

# Smooth data
#X = BaselineCorrection().savgol_filter(X)


# Trucate to biologically relevant range
truncator = WavenumberTruncator()
X = truncator.trucate_range(X, 3050.0, 850.0)

xgb_model = XGBModel()
svm_model = SVMRBFModel()
tabpfn_model = TabPFNModel()
catboost_model = CatBoostModel()
realmlp_model = RealMLPModel()
tabm_model = TabMModel()
lightgbm_model = LightGBMModel()

models_list: list[BaseClassifierModel] = [catboost_model]

for model in models_list:

    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        eval_metrics = model.evaluate(X_train_fold, X_test_fold, y_train_fold, y_test_fold)

        lst_accu_stratified.append(eval_metrics)

    avg_metrics = np.mean(lst_accu_stratified, axis=0)
    std_metrics = np.std(lst_accu_stratified, axis=0)

    print(f"\nModel: {model}")
    print(f"Accuracy: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"Precision: {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"Recall (Sensitivity): {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"Specificity: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
    print(f"Mean(SE,SP): {avg_metrics[4]:.4f} ± {std_metrics[4]:.4f}")

