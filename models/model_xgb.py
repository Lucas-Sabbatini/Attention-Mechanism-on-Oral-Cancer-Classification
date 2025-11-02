import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pathlib import Path

from preProcess.baseline_correction import SavitzkyFilter
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization

def xgb_model(X_train : np.array, X_test : np.array, y_train : np.array, y_test : np.array):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Specify our model hyperparameters
    param = {'max_depth': 4, 'eta': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = ['auc']
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = 10

    # Instantiate the model
    model = xgb.train(param, dtrain, num_round, evallist)

    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    #Eval metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    esp = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred)

    return (acc, prec, rec, esp, f1)

project_root = Path(__file__).parent.parent
dataset_path = project_root / "dataset_cancboca.dat"

# Sample Data
dataset = np.loadtxt(dataset_path)
X = dataset[:,:-1]
y = dataset[:,-1].astype(int)
y = np.where(y == -1, 0, 1)

#Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
lst_accu_stratified = []


#Preprocess data with Savitzky-Golay filter
X = SavitzkyFilter().buildFilter(X)

# Normalize data
normalizer = Normalization()
X = normalizer.min_max_scaling(X)

# Truncate wavenumber range [1800, 900]
truncator = WavenumberTruncator()
X = truncator.trucate_range(X, 3050.0, 850.0)


for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    eval_metrics = xgb_model(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
    lst_accu_stratified.append(eval_metrics)

avg_metrics = np.mean(lst_accu_stratified, axis=0)
std_metrics = np.std(lst_accu_stratified, axis=0)

print(f"Accuracy: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
print(f"Precision: {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
print(f"Recall (Sensitivity): {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
print(f"Specificity: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
print(f"F1 Score: {avg_metrics[4]:.4f} ± {std_metrics[4]:.4f}")