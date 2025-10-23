import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from pathlib import Path

from preProcess.savitzky_filter import SavitzkyFilter
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization

project_root = Path(__file__).parent.parent
dataset_path = project_root / "dataset_cancboca.dat"

# Sample Data
dataset = np.loadtxt(dataset_path)
X = dataset[:,:-1]
y = dataset[:,-1].astype(int)
y = np.where(y == -1, 0, 1)


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Preprocess data with Savitzky-Golay filter
X_train = SavitzkyFilter().buildFilter(X_train)
X_test = SavitzkyFilter().buildFilter(X_test)

# Truncate wavenumber range [1800, 900]
truncator = WavenumberTruncator()
X_train = truncator.trucate_range(3050.0, 850.0, X_train)
X_test = truncator.trucate_range(3050.0, 850.0, X_test)

# Normalize data
normalizer = Normalization()
#X_train = normalizer.normalize_data(X_train)
#X_test = normalizer.normalize_data(X_test)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Specify our model hyperparameters
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = ['auc']
evallist = [(dtrain, 'train'), (dtest, 'eval')]
num_round = 10

# Instantiate the model
model = xgb.train(param, dtrain, num_round, evallist)

y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
esp = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred)

print(f"Acurácia: {acc:.4f}")
print(f"Precisão: {prec:.4f}")
print(f"Sensibilidade (Recall): {rec:.4f}")
print(f"Especificidade: {esp:.4f}")
print(f"F1-score: {f1:.4f}")