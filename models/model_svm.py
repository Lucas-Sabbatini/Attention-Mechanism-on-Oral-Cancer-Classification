from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
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

# Normalize data
normalizer = Normalization()
#X_train = normalizer.peak_normalization(X_train, 1660.0, 1630.0)
#X_test = normalizer.peak_normalization(X_test, 1660.0, 1630.0)

# Truncate wavenumber range [1800, 900]
truncator = WavenumberTruncator()
X_train = truncator.trucate_range( X_train, 3050.0, 850.0)
X_test = truncator.trucate_range(X_test, 3050.0, 850.0)



clf = svm.SVC(kernel='rbf', C=1, gamma='scale')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


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