from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
import numpy as np
from pathlib import Path

from preProcess.baseline_correction import SavitzkyFilter
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization

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
X = normalizer.peak_normalization(X, 1660.0, 1630.0)

# Trucate to biologically relevant range
truncator = WavenumberTruncator()
X = truncator.trucate_range(X, 3050.0, 850.0)

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    #Train SVM model
    clf = svm.SVC(kernel='rbf', C=1, gamma='scale')
    clf.fit(X_train_fold, y_train_fold)

    #Evaluate model
    y_pred = clf.predict(X_test_fold)

    acc = accuracy_score(y_test_fold, y_pred)
    prec = precision_score(y_test_fold, y_pred)
    rec = recall_score(y_test_fold, y_pred)
    esp = recall_score(y_test_fold, y_pred, pos_label=0)
    f1 = f1_score(y_test_fold, y_pred)

    lst_accu_stratified.append((acc, prec, rec, esp, f1))

avg_metrics = np.mean(lst_accu_stratified, axis=0)
std_metrics = np.std(lst_accu_stratified, axis=0)

print(f"Accuracy: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
print(f"Precision: {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
print(f"Recall (Sensitivity): {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
print(f"Specificity: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
print(f"F1 Score: {avg_metrics[4]:.4f} ± {std_metrics[4]:.4f}")