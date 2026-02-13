"""
Augmentation Pipeline Evaluation

Compares model performance with and without data augmentation.
"""

from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preProcess.baseline_correction import BaselineCorrection
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization
from models.model_xgb import XGBModel
from augmentation.spectral_augmentation import augment_with_labels

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def load_and_preprocess_data(dataset_path: str = "dataset_cancboca.dat") -> tuple:
    """
    Load and preprocess spectral data following the main pipeline.
    
    Returns:
        Tuple of (X, y) preprocessed arrays
    """
    # Load data
    dataset = np.loadtxt(dataset_path)
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    y = np.where(y == -1, 0, 1)
    
    # Baseline correction
    baseline = BaselineCorrection().asls_baseline(X)
    X = X - baseline
    
    # Normalize data
    normalizer = Normalization()
    X = normalizer.peak_normalization(X, 1660.0, 1630.0)
    
    # Truncate to biologically relevant range
    truncator = WavenumberTruncator()
    X = truncator.trucate_range(X, 3050.0, 850.0)
    
    return X, y


def evaluate_with_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    n_augmented: int = 1,
    random_state: int = 1,
    augmentation_probabilities: dict = None,
    scaling_params: dict = None,
    noise_params: dict = None,
    shift_params: dict = None,
    baseline_params: dict = None,
    verbose: bool = False
) -> dict:
    """
    Evaluate model performance with and without augmentation using stratified K-Fold.
    
    Args:
        X: Preprocessed spectral data
        y: Labels
        n_splits: Number of cross-validation folds
        n_augmented: Number of augmented samples per original sample
        random_state: Random seed for reproducibility
        augmentation_probabilities: Custom augmentation probabilities
        
    Returns:
        Dictionary with evaluation results
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    results_no_aug = []
    results_with_aug = []
    model_no_aug = XGBModel()
    model_with_aug = XGBModel()

    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Model WITHOUT augmentation
        metrics_no_aug = model_no_aug.evaluate(X_train, X_test, y_train, y_test)
        results_no_aug.append(metrics_no_aug)
        
        # Augment ONLY the training data
        X_train_aug, y_train_aug = augment_with_labels(
            X_train,
            y_train,
            n_augmented=n_augmented,
            probabilities=augmentation_probabilities,
            include_original=True,
            random_seed=random_state + fold,
            scaling_params=scaling_params,
            noise_params=noise_params,
            shift_params=shift_params,
            baseline_params=baseline_params
        )
        
        if verbose and fold == 1:
            print(f"Train samples: {len(X_train)}, Augmented total: {len(X_train_aug)} "
                  f"(+{len(X_train_aug) - len(X_train)} synthetic)")
        
        # Model WITH augmentation
        metrics_with_aug = model_with_aug.evaluate(X_train_aug, X_test, y_train_aug, y_test)
        results_with_aug.append(metrics_with_aug)
        
        if verbose:
            print(f"Fold {fold}/{n_splits} - No Aug Acc: {metrics_no_aug[0]:.4f}, "
                  f"With Aug Acc: {metrics_with_aug[0]:.4f}")
    
    # Compute statistics
    results_no_aug = np.array(results_no_aug)
    results_with_aug = np.array(results_with_aug)
    
    return {
        'no_augmentation': {
            'mean': np.mean(results_no_aug, axis=0),
            'std': np.std(results_no_aug, axis=0),
            'raw': results_no_aug
        },
        'with_augmentation': {
            'mean': np.mean(results_with_aug, axis=0),
            'std': np.std(results_with_aug, axis=0),
            'raw': results_with_aug
        }
    }


def print_comparison(results: dict) -> None:
    """Print comparison of augmented vs non-augmented results."""
    metric_names = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'Mean(SE,SP)']
    
    print("\n" + "=" * 70)
    print("AUGMENTATION EVALUATION RESULTS")
    print("=" * 70)
    
    print("\n--- Without Augmentation ---")
    no_aug = results['no_augmentation']
    for i, name in enumerate(metric_names):
        print(f"{name}: {no_aug['mean'][i]:.4f} ± {no_aug['std'][i]:.4f}")
    
    print("\n--- With Augmentation ---")
    with_aug = results['with_augmentation']
    for i, name in enumerate(metric_names):
        print(f"{name}: {with_aug['mean'][i]:.4f} ± {with_aug['std'][i]:.4f}")
    
    print("\n--- Difference (Augmented - Original) ---")
    for i, name in enumerate(metric_names):
        diff = with_aug['mean'][i] - no_aug['mean'][i]
        sign = "+" if diff >= 0 else ""
        print(f"{name}: {sign}{diff:.4f}")
    
    print("=" * 70)


def print_summary(name: str, results: dict) -> None:
    """Print a compact summary of augmented results."""
    with_aug = results['with_augmentation']
    print(f"  Accuracy: {with_aug['mean'][0]:.4f} ± {with_aug['std'][0]:.4f}")
    print(f"  Mean(SE,SP): {with_aug['mean'][4]:.4f} ± {with_aug['std'][4]:.4f}")


if __name__ == "__main__":
    # Change to project root directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data()
    print(f"Data shape: {X.shape}, Labels: {np.bincount(y)}")
    
    # Test configurations - each augmentation separately
    test_configs = {
        # Ultra-gentle settings with 50% probability
        'SCALING (±2%, 50% prob)': {
            'probabilities': {'scaling': 0.5, 'noise': 0.0, 'shift': 0.0, 'baseline': 0.0, 'none': 0.5},
            'scaling_params': {'alpha_range': (0.98, 1.02)},
        },
        'NOISE (0.1%, 50% prob)': {
            'probabilities': {'scaling': 0.0, 'noise': 0.5, 'shift': 0.0, 'baseline': 0.0, 'none': 0.5},
            'noise_params': {'sigma_percent_range': (0.001, 0.001)},
        },
        'SHIFT (±1 cm⁻¹, 50% prob)': {
            'probabilities': {'scaling': 0.0, 'noise': 0.0, 'shift': 0.5, 'baseline': 0.0, 'none': 0.5},
            'shift_params': {'delta_range': (-1.0, 1.0)},
        },
        # Combined ultra-gentle
        'COMBINED ULTRA-GENTLE': {
            'probabilities': {'scaling': 0.3, 'noise': 0.3, 'shift': 0.2, 'baseline': 0.0, 'none': 0.2},
            'scaling_params': {'alpha_range': (0.98, 1.02)},
            'noise_params': {'sigma_percent_range': (0.001, 0.001)},
            'shift_params': {'delta_range': (-1.0, 1.0)},
        },
    }
    
    all_results = {}
    
    # First get baseline (no augmentation)
    print("\n" + "=" * 70)
    print("BASELINE (No Augmentation)")
    print("=" * 70)
    baseline_results = evaluate_with_augmentation(
        X, y, n_splits=10, n_augmented=1, random_state=1,
        augmentation_probabilities={'scaling': 0.0, 'noise': 0.0, 'shift': 0.0, 'baseline': 0.0, 'none': 1.0},
        verbose=True
    )
    baseline_acc = baseline_results['no_augmentation']['mean'][0]
    baseline_mean_se_sp = baseline_results['no_augmentation']['mean'][4]
    print(f"\nBaseline Accuracy: {baseline_acc:.4f}")
    print(f"Baseline Mean(SE,SP): {baseline_mean_se_sp:.4f}")
    
    # Test each augmentation separately
    for name, config in test_configs.items():
        print("\n" + "=" * 70)
        print(f"Testing: {name}")
        print("=" * 70)
        
        results = evaluate_with_augmentation(
            X, y,
            n_splits=10,
            n_augmented=1,
            random_state=1,
            augmentation_probabilities=config.get('probabilities'),
            scaling_params=config.get('scaling_params'),
            noise_params=config.get('noise_params'),
            shift_params=config.get('shift_params'),
        )
        all_results[name] = results
        print_summary(name, results)
    
    # Final comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'Augmentation':<25} {'Accuracy':<20} {'Mean(SE,SP)':<20} {'Δ Acc':<10}")
    print("-" * 75)
    print(f"{'Baseline':<25} {baseline_acc:.4f}               {baseline_mean_se_sp:.4f}               {'---':<10}")
    
    for name, results in all_results.items():
        acc = results['with_augmentation']['mean'][0]
        mean_se_sp = results['with_augmentation']['mean'][4]
        delta = acc - baseline_acc
        sign = "+" if delta >= 0 else ""
        print(f"{name:<25} {acc:.4f}               {mean_se_sp:.4f}               {sign}{delta:.4f}")
