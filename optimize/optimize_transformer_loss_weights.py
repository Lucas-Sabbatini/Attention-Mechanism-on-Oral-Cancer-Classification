import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
import warnings

from preProcess.baseline_correction import BaselineCorrection
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization
from transformer.model import BioSpectralFormer

warnings.filterwarnings('ignore')

# Preprocessing (run once, not per trial)
dataset_path = "dataset_cancboca.dat"
dataset = np.loadtxt(dataset_path)
X_raw = dataset[:, :-1]
y = dataset[:, -1].astype(int)
y = np.where(y == -1, 0, 1)

baseline = BaselineCorrection().asls_baseline(X_raw)
X = X_raw - baseline
X = Normalization().peak_normalization(X, 1660.0, 1630.0)
X = WavenumberTruncator().trucate_range(X, 3050.0, 850.0)

N_TRIALS = 50
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)


def objective(trial):
    center_loss_weight = trial.suggest_float('center_loss_weight', 0.0, 1.0)
    supcon_weight      = trial.suggest_float('supcon_weight', 0.0, 1.0 - center_loss_weight)
    bce_weight         = 1.0 - center_loss_weight - supcon_weight

    all_metrics = []
    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model = BioSpectralFormer(
            num_spectral_points=X.shape[1],
            center_loss_weight=center_loss_weight,
            supcon_weight=supcon_weight,
            bce_weight=bce_weight,
            verbose=False,
        )
        eval_metrics = model.evaluate(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        all_metrics.append(eval_metrics)

    avg_metrics = np.mean(all_metrics, axis=0)

    trial.report(avg_metrics[4], step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return avg_metrics[4]  # Mean(Sensitivity, Specificity)


study = optuna.create_study(
    direction='maximize',
    study_name='transformer_loss_weights_optimization',
    sampler=optuna.samplers.TPESampler(seed=42),
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n" + "=" * 60)
print(f"\nBest Score (Mean Sensitivity + Specificity): {study.best_value:.4f}")
print("\nBest loss weights found:")
clw = study.best_params['center_loss_weight']
sw  = study.best_params['supcon_weight']
bw  = 1.0 - clw - sw
print(f"  center_loss_weight : {clw:.4f}")
print(f"  supcon_weight      : {sw:.4f}")
print(f"  bce_weight         : {bw:.4f}")
print(f"  sum                : {clw + sw + bw:.4f}")
print("\n" + "=" * 60)

print("\nParameter importances:")
importance = optuna.importance.get_param_importances(study)
for param, value in importance.items():
    print(f"  - {param}: {value:.4f}")
