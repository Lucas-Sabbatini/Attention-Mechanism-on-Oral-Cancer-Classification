"""
Training Loss Curve Visualization

For each cross-validation fold, trains the BioSpectralFormer and records
the composite training loss (BCE + Center + SupCon) per epoch.

Outputs:
  ploting/img/loss/training_loss_all_folds.png  – all folds overlaid
  ploting/img/loss/training_loss_fold_N.png     – individual fold curves
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from preProcess.baseline_correction import BaselineCorrection
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization
from transformer.model import BioSpectralFormer

OUTPUT_BASE = Path(__file__).parent / "img" / "loss"


def load_and_preprocess_data():
    dataset = np.loadtxt(ROOT / "dataset_cancboca.dat")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    y = np.where(y == -1, 0, 1)

    X = X - BaselineCorrection().asls_baseline(X)
    X = Normalization().peak_normalization(X, 1660.0, 1630.0)

    truncator = WavenumberTruncator(ROOT / "wavenumbers_cancboca.dat")
    X = truncator.trucate_range(X, 3050.0, 850.0)

    return X, y


def main(random_state: int = 1, n_splits: int = 10):
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data()
    num_spectral_points = X.shape[1]
    print(f"Data shape: {X.shape}  |  Classes: {np.bincount(y)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    test_fold_metrics = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        fold_num = fold_idx + 1
        print(f"\n{'='*60}")
        print(f"Fold {fold_num}/{n_splits}")
        print(f"{'='*60}")

        X_train_fold = X[train_index]
        y_train_fold = y[train_index]
        X_test_fold = X[test_index]
        y_test_fold = y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold,
            test_size=0.3,
            random_state=random_state,
            stratify=y_train_fold,
        )
        model = BioSpectralFormer(num_spectral_points=num_spectral_points)

        # Reset fold counter for transformer model diagnostics
        if hasattr(model, 'reset_fold_counter'):
            model.reset_fold_counter()

        model.train_model(X_train, y_train, X_val, y_val)
        model.calibrate_threshold(X_val, y_val)

        # Collect val metrics for summary
        y_pred = model.predict(X_val)
        fold_metrics.append({
            'acc': accuracy_score(y_val, y_pred),
            'prec': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'spec': recall_score(y_val, y_pred, pos_label=0, zero_division=0),
            'mean_se_sp': (recall_score(y_val, y_pred, zero_division=0)
                           + recall_score(y_val, y_pred, pos_label=0, zero_division=0)) / 2,
        })

        # Collect test metrics
        y_test_pred = model.predict(X_test_fold)
        se = recall_score(y_test_fold, y_test_pred, zero_division=0)
        sp = recall_score(y_test_fold, y_test_pred, pos_label=0, zero_division=0)
        test_fold_metrics.append({
            'acc': accuracy_score(y_test_fold, y_test_pred),
            'prec': precision_score(y_test_fold, y_test_pred, zero_division=0),
            'recall': se,
            'spec': sp,
            'mean_se_sp': (se + sp) / 2,
        })

        if hasattr(model, 'print_fold_summary'):
            model.print_fold_summary()

        # Individual fold plot
        epochs = range(1, len(model.loss_history) + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, model.loss_history, color="steelblue", linewidth=2)
        ax.set_title(f"Training Loss – Fold {fold_num}", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (BCE + Center + SupCon)")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(OUTPUT_BASE / f"training_loss_fold_{fold_num}.png", dpi=300)
        plt.close(fig)
        print(f"  Saved fold {fold_num} loss curve")

    # Print cross-fold summary
    print(f"\n{'='*60}")
    print(f"CROSS-FOLD VALIDATION SUMMARY ({n_splits} folds)")
    print(f"{'='*60}")
    for metric, label in [('acc', 'Accuracy'),
                          ('prec', 'Precision'),
                          ('recall', 'Recall (SE)'),
                          ('spec', 'Specificity (SP)'),
                          ('mean_se_sp', 'Mean(SE,SP)')]:
        values = [m[metric] for m in fold_metrics]
        print(f"  {label:<18s}: {np.mean(values):.1%} +/- {np.std(values):.1%}")
    print(f"{'='*60}")

    # Print per-fold test set performance
    print(f"\n{'='*60}")
    print(f"TEST SET PERFORMANCE PER FOLD")
    print(f"{'='*60}")
    for i, m in enumerate(test_fold_metrics, 1):
        print(f"  Fold {i:>2d}  |  Acc: {m['acc']:.1%}  Prec: {m['prec']:.1%}  "
              f"Recall(SE): {m['recall']:.1%}  Spec(SP): {m['spec']:.1%}  "
              f"Mean(SE,SP): {m['mean_se_sp']:.1%}")
    print(f"{'='*60}")
    for metric, label in [('acc', 'Accuracy'),
                          ('prec', 'Precision'),
                          ('recall', 'Recall (SE)'),
                          ('spec', 'Specificity (SP)'),
                          ('mean_se_sp', 'Mean(SE,SP)')]:
        values = [m[metric] for m in test_fold_metrics]
        print(f"  {label:<18s}: {np.mean(values):.1%} +/- {np.std(values):.1%}")
    print(f"{'='*60}")

    print(f"\nDone. All outputs saved under {OUTPUT_BASE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot BioSpectralFormer training loss curves across CV folds"
    )
    parser.add_argument("--seed",  type=int, default=1,  help="Random seed (default: 1)")
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds (default: 10)")
    args = parser.parse_args()

    main(random_state=args.seed, n_splits=args.folds)
