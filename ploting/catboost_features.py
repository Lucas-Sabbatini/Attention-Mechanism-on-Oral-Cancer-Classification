import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import matplotlib.pyplot as plt

from preProcess.baseline_correction import BaselineCorrection
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization
from models.model_catboost import CatBoostModel


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

# Trucate to biologically relevant range
truncator = WavenumberTruncator()
X = truncator.trucate_range(X, 3050.0, 850.0)
features_wavenumbers = truncator.get_wavenumbers_in_range(3050.0, 850.0)

catboost_model = CatBoostModel()
lst_accu_stratified = []
features_importances = []

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    eval_metrics = catboost_model.catboost_model(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
    features_importance = catboost_model.get_feature_importances(X_train_fold, y_train_fold)

    lst_accu_stratified.append(eval_metrics)
    features_importances.append(features_importance)

avg_metrics = np.mean(lst_accu_stratified, axis=0)
std_metrics = np.std(lst_accu_stratified, axis=0)
features_mean_importances = np.mean(features_importances, axis=0)

print(f"\nModel: CatBoost")
print(f"Accuracy: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
print(f"Precision: {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
print(f"Recall (Sensitivity): {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
print(f"Specificity: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
print(f"Mean(SE,SP): {avg_metrics[4]:.4f} ± {std_metrics[4]:.4f}")


fi = np.asarray(features_mean_importances)
n_features = fi.size

highest_importance_indices = np.argsort(fi)[-10:][::-1]
print("\nTop 10 Mean Important Features (Wavenumber cm⁻¹ and Importance):")
for idx in highest_importance_indices:
    print(f"Wavenumber: {features_wavenumbers[idx]:.2f}, Importance: {fi[idx]:.4f}")

plt.figure(figsize=(20, 5))
plt.bar(features_wavenumbers, fi, color="blue")
plt.gca().invert_xaxis()
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Importance")
plt.title("CatBoost Mean Feature Importances (k=10)")

plt.tight_layout()
output_path = Path(__file__).parent / "img" / "catboost_features_importance.png"
plt.savefig(output_path, dpi=300)
plt.close()