import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
import warnings

from preProcess.baseline_correction import BaselineCorrection
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization

from models.model_svm import SVMRBFModel

warnings.filterwarnings('ignore')

dataset_path = "dataset_cancboca.dat"
dataset = np.loadtxt(dataset_path)
X_original = dataset[:, :-1]
y = dataset[:, -1].astype(int)
y = np.where(y == -1, 0, 1)
N_TRIALS = 100


def objective(trial):
    """
    Função objetivo para otimização dos hiperparâmetros do SVM-RBF usando Optuna.
    
    Hiperparâmetros otimizados:
    - C: parâmetro de regularização (controla o trade-off entre margem e erros)
    - gamma: coeficiente do kernel RBF (controla a influência de um único exemplo de treino)
    """
    
    # Sugerir hiperparâmetros para otimização
    C = trial.suggest_float('C', 1e-2, 1e3, log=True)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('gamma_type', ['auto', 'manual']) == 'auto' else trial.suggest_float('gamma_value', 1e-5, 1e2, log=True)
    
    # Simplificar gamma para usar apenas o valor sugerido
    if isinstance(gamma, str):
        gamma_param = gamma
    else:
        gamma_param = gamma
    
    # Preprocessing pipeline
    baseline = BaselineCorrection().asls_baseline(X_original)
    X = X_original - baseline
    
    normalizer = Normalization()
    X = normalizer.peak_normalization(X, 1660.0, 1630.0)
    
    
    truncator = WavenumberTruncator()
    X = truncator.trucate_range(X, 3050.0, 850.0)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    all_metrics = []
    
    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        # Criar modelo com hiperparâmetros otimizados
        model = SVMRBFModel(C=C, gamma=gamma_param)
        eval_metrics = model.evaluate(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        all_metrics.append(eval_metrics)
    
    avg_metrics = np.mean(all_metrics, axis=0)

    trial.report(avg_metrics[4], step=0)
    if trial.should_prune():   
        raise optuna.exceptions.TrialPruned()
    
    # Mean(SE, SP)
    return avg_metrics[4]


study = optuna.create_study(
    direction='maximize',
    study_name='svm_rbf_optimization',
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n" + "=" * 60)
print(f"\nMelhor Score (Média Sensibilidade + Especificidade): {study.best_value:.4f}")
print("\nMelhores hiperparâmetros encontrados:")
print(f"  - C: {study.best_params['C']:.6f}")
if 'gamma_type' in study.best_params:
    if study.best_params['gamma_type'] == 'auto':
        print(f"  - gamma: {study.best_params['gamma']}")
    else:
        print(f"  - gamma: {study.best_params['gamma_value']:.6f}")
print("\n" + "=" * 60)

print("\nImportância dos hiperparâmetros:")
importance = optuna.importance.get_param_importances(study)
for param, value in importance.items():
    print(f"  - {param}: {value:.4f}")
