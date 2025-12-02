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

models_list = ['RealMLP']
xgb_model = XGBModel()
svm_model = SVMRBFModel()
tabpfn_model = TabPFNModel()
catboost_model = CatBoostModel()
realmlp_model = RealMLPModel()
tabm_model = TabMModel()
lightgbm_model = LightGBMModel()

for model in models_list:

    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        if model == 'XGBoost':
            eval_metrics = xgb_model.xgb_model(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        elif model == 'Suport Vector Machine (RBF)':
            eval_metrics = svm_model.svm_rbf_model(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        elif model == 'TabPFN V2':
            eval_metrics = tabpfn_model.tabpfn_model(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        elif model == 'CatBoost':
            eval_metrics = catboost_model.catboost_model(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        elif model == 'RealMLP':
            eval_metrics = realmlp_model.realmlp_model(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        elif model == 'TabM':
            eval_metrics = tabm_model.tabm_model(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        elif model == 'LightGBM':
            eval_metrics = lightgbm_model.lightgbm_model(X_train_fold, X_test_fold, y_train_fold, y_test_fold)

        lst_accu_stratified.append(eval_metrics)

    avg_metrics = np.mean(lst_accu_stratified, axis=0)
    std_metrics = np.std(lst_accu_stratified, axis=0)

    print(f"\nModel: {model}")
    print(f"Accuracy: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"Precision: {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"Recall (Sensitivity): {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"Specificity: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
    print(f"Mean(SE,SP): {avg_metrics[4]:.4f} ± {std_metrics[4]:.4f}")




    #TODO:
    # 1. Acho que essa parte do filtro savgol pode ser uma coisa a questionar o Murilo sobre.