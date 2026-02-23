from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
import os

from transformer.visualize import paired_t_test

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

from transformer.model import BioSpectralFormer

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

#Baseline correction
baseline = BaselineCorrection().asls_baseline(X)
X = X - baseline

# Normalize data
normalizer = Normalization()
X = normalizer.peak_normalization(X, 1660.0, 1630.0)

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
spectra_model = BioSpectralFormer(num_spectral_points=X.shape[1])

models_list: list[BaseClassifierModel] = [ spectra_model]

transformer_results, lgbm_results = [], []
for train_idx, test_idx in skf.split(X, y):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    transformer_results.append(spectra_model.evaluate(X_train_fold, X_test_fold, y_train_fold, y_test_fold))
    lgbm_results.append(lightgbm_model.evaluate(X_train_fold, X_test_fold, y_train_fold, y_test_fold))

paired_t_test(transformer_results, lgbm_results)