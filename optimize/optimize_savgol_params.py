import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
import warnings

from preProcess.baseline_correction import BaselineCorrection
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization

from models.model_xgb import XGBModel

warnings.filterwarnings('ignore')

dataset_path = "dataset_cancboca.dat"
dataset = np.loadtxt(dataset_path)
X_original = dataset[:, :-1]
y = dataset[:, -1].astype(int)
y = np.where(y == -1, 0, 1)
N_TRIALS = 50
MODEL = XGBModel()

def objective(trial):
    """
    Função objetivo para otimização dos parâmetros do Savitzky-Golay filter usando Optuna.
    
    Parâmetros otimizados:
    - window_length: tamanho da janela (par ou ímpar, >= 3)
    - poly_order: ordem do polinômio (deve ser < window_length)
    - deriv: ordem da derivada (0, 1 ou 2)
    """
    
    # Permite valores pares e ímpares para window_length
    window_length = trial.suggest_int('window_length', 3, 25)
    poly_order = trial.suggest_int('poly_order', 2, window_length - 1)
    deriv = trial.suggest_int('deriv', 0, 2)
    
    baseline = BaselineCorrection().asls_baseline(X_original)
    X = X_original - baseline
    
    normalizer = Normalization()
    X = normalizer.peak_normalization(X, 1660.0, 1630.0)
    
    # Aplicar Savitzky-Golay filter com parâmetros sendo otimizados
    baseline_corr = BaselineCorrection()
    X = baseline_corr.savgol_filter(X, window_length=window_length, 
                                     poly_order=poly_order, deriv=deriv)
    
    truncator = WavenumberTruncator()
    X = truncator.trucate_range(X, 3050.0, 850.0)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    all_metrics = []
    
    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        eval_metrics = MODEL.evaluate(X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        all_metrics.append(eval_metrics)
    
    avg_metrics = np.mean(all_metrics, axis=0)

    trial.report(avg_metrics[4], step=0)
    if trial.should_prune():   
        raise optuna.exceptions.TrialPruned()
    
    # Mean(SE, SP)
    return avg_metrics[4]


study = optuna.create_study(
    direction='maximize',
    study_name='savgol_filter_optimization',
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n" + "=" * 60)
print(f"\nMelhor Score (Média Sensibilidade + Especificidade): {study.best_value:.4f}")
print("\nMelhores parâmetros encontrados:")
print(f"  - window_length: {study.best_params['window_length']}")
print(f"  - poly_order: {study.best_params['poly_order']}")
print(f"  - deriv: {study.best_params['deriv']}")
print("\n" + "=" * 60)

print("\nImportância dos parâmetros:")
importance = optuna.importance.get_param_importances(study)
for param, value in importance.items():
    print(f"  - {param}: {value:.4f}")