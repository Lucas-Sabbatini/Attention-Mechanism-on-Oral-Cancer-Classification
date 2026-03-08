import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
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

N_TRIALS = 100
N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=1)


def objective(trial):
    # Architecture parameters
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    nhead = trial.suggest_categorical('nhead', [1, 2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [32, 64, 128])
    patch_size = trial.suggest_categorical('patch_size', [8, 16, 32])

    # Regularization
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # Training parameters
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    patience = trial.suggest_int('patience', 30, 80, step=10)

    # Contrastive learning
    supcon_temperature = trial.suggest_float('supcon_temperature', 0.03, 0.2, log=True)

    # Validate nhead divides d_model
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    all_metrics = []
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model = BioSpectralFormer(
            num_spectral_points=X.shape[1],
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            patch_size=patch_size,
            lr=lr,
            weight_decay=weight_decay,
            n_epochs=200,
            batch_size=batch_size,
            patience=patience,
            supcon_temperature=supcon_temperature,
            verbose=False,
        )
        eval_metrics = model.evaluate(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        all_metrics.append(eval_metrics)

        # Prune after 3 folds if performance is poor
        if fold_idx == 2:
            interim_score = np.mean([m[4] for m in all_metrics])
            trial.report(interim_score, step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    avg_metrics = np.mean(all_metrics, axis=0)
    std_metrics = np.std(all_metrics, axis=0)

    # Store all metrics for the results file
    trial.set_user_attr('avg_accuracy', avg_metrics[0])
    trial.set_user_attr('avg_precision', avg_metrics[1])
    trial.set_user_attr('avg_sensitivity', avg_metrics[2])
    trial.set_user_attr('avg_specificity', avg_metrics[3])
    trial.set_user_attr('avg_mean_se_sp', avg_metrics[4])
    trial.set_user_attr('std_accuracy', std_metrics[0])
    trial.set_user_attr('std_mean_se_sp', std_metrics[4])

    return avg_metrics[4]  # Optimize Mean(Sensitivity, Specificity)


study = optuna.create_study(
    direction='maximize',
    study_name='transformer_params_optimization',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# Write results to file
output_path = "optimize/transformer_params_results.txt"
with open(output_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("BioSpectralFormer Hyperparameter Optimization Results\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Trials: {N_TRIALS} | Folds: {N_FOLDS}\n")
    f.write("=" * 70 + "\n\n")

    # Best trial
    best = study.best_trial
    f.write("BEST TRIAL\n")
    f.write("-" * 70 + "\n")
    f.write(f"Trial #: {best.number}\n")
    f.write(f"Mean(SE,SP): {best.value:.4f} ± {best.user_attrs.get('std_mean_se_sp', 0):.4f}\n")
    f.write(f"Accuracy:    {best.user_attrs.get('avg_accuracy', 0):.4f} ± {best.user_attrs.get('std_accuracy', 0):.4f}\n")
    f.write(f"Precision:   {best.user_attrs.get('avg_precision', 0):.4f}\n")
    f.write(f"Sensitivity: {best.user_attrs.get('avg_sensitivity', 0):.4f}\n")
    f.write(f"Specificity: {best.user_attrs.get('avg_specificity', 0):.4f}\n\n")

    f.write("BEST PARAMETERS\n")
    f.write("-" * 70 + "\n")

    # Group parameters by category
    arch_params = ['d_model', 'nhead', 'num_layers', 'dim_feedforward', 'patch_size']
    train_params = ['lr', 'weight_decay', 'batch_size', 'patience', 'dropout']
    loss_params = ['center_loss_weight', 'supcon_weight', 'supcon_temperature', 'mask_penalty']

    f.write("  Architecture:\n")
    for p in arch_params:
        f.write(f"    {p:25s}: {best.params[p]}\n")

    f.write("  Training:\n")
    for p in train_params:
        f.write(f"    {p:25s}: {best.params[p]}\n")

    bce_w = 1.0 - best.params['center_loss_weight'] - best.params['supcon_weight']
    f.write("  Loss weights:\n")
    for p in loss_params:
        f.write(f"    {p:25s}: {best.params[p]:.6f}\n")
    f.write(f"    {'bce_weight (derived)':25s}: {bce_w:.6f}\n\n")

    # Parameter importances
    f.write("PARAMETER IMPORTANCES\n")
    f.write("-" * 70 + "\n")
    importance = optuna.importance.get_param_importances(study)
    for param, value in importance.items():
        f.write(f"  {param:25s}: {value:.4f}\n")

    # Top 5 trials
    f.write(f"\nTOP 5 TRIALS\n")
    f.write("-" * 70 + "\n")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)
    for i, t in enumerate(sorted_trials[:5]):
        f.write(f"  #{t.number:3d} | Mean(SE,SP)={t.value:.4f} | "
                f"d={t.params.get('d_model','-')} h={t.params.get('nhead','-')} "
                f"L={t.params.get('num_layers','-')} ff={t.params.get('dim_feedforward','-')} "
                f"lr={t.params.get('lr', 0):.1e} bs={t.params.get('batch_size','-')}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("Copy-paste constructor:\n\n")
    f.write("BioSpectralFormer(\n")
    f.write(f"    num_spectral_points=X.shape[1],\n")
    for p, v in best.params.items():
        if isinstance(v, float):
            f.write(f"    {p}={v},\n")
        else:
            f.write(f"    {p}={v},\n")
    f.write(f"    bce_weight={bce_w},\n")
    f.write(")\n")
    f.write("=" * 70 + "\n")

print(f"\nResults written to {output_path}")
print(f"Best Mean(SE,SP): {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
