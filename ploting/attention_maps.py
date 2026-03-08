"""
Attention Map Visualization Script

For each cross-validation fold, trains the BioSpectralFormer on the real
dataset and:
  1. Saves per-sample attention heatmaps and layer-comparison plots.
  2. Accumulates the spectral attention profile (intra-sample attention
     projected back onto the wavenumber axis) for every test sample.

After all folds, cross-fold mean attention maps are computed per class and
plotted using the same format as the per-fold plots:
  ploting/img/attention/mean_attention_cancer.png
  ploting/img/attention/mean_attention_healthy.png

Outputs per fold (ploting/img/attention/fold_N/):
  attention_cancer.png  – last-layer heatmaps + spectrum overlay (one cancer sample)
  attention_healthy.png – same for one healthy sample
  layers_cancer.png     – attention evolution across all transformer layers
  layers_healthy.png    – same for a healthy sample
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from preProcess.baseline_correction import BaselineCorrection
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization
from transformer.model import BioSpectralFormer
from transformer.visualize import plot_attention_maps, plot_layer_comparison
from transformer.visualize.plot_attention import _patches_to_spectrum_weights

OUTPUT_BASE = Path(__file__).parent / "img" / "attention"


# ---------------------------------------------------------------------------
# Data loading  (mirrors main.py exactly)
# ---------------------------------------------------------------------------

def load_and_preprocess_data():
    dataset = np.loadtxt(ROOT / "dataset_cancboca.dat")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    y = np.where(y == -1, 0, 1)

    X = X - BaselineCorrection().asls_baseline(X)
    X = Normalization().peak_normalization(X, 1660.0, 1630.0)

    truncator = WavenumberTruncator(ROOT / "wavenumbers_cancboca.dat")
    X = truncator.trucate_range(X, 3050.0, 850.0)
    wavenumbers = truncator.get_wavenumbers_in_range(3050.0, 850.0)

    return X, y, wavenumbers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _average_attn_dicts(attn_dicts):
    """
    Average a list of attn_dict (as returned by model.get_attention_maps)
    into a single attn_dict with mean per-head maps and mean alpha.
    """
    n_heads = len(attn_dicts[0]['inter_attention'])

    mean_inter = []
    mean_intra = []
    for h in range(n_heads):
        mean_inter.append(
            np.mean([d['inter_attention'][h] for d in attn_dicts], axis=0)
        )
        mean_intra.append(
            np.mean([d['intra_attention'][h] for d in attn_dicts], axis=0)
        )

    mean_alpha = float(np.mean([d['alpha'] for d in attn_dicts]))
    return {
        'inter_attention': mean_inter,
        'intra_attention': mean_intra,
        'alpha': mean_alpha,
    }


def _find_top_regions(wavenumbers, spectral_weights, n_regions=5,
                      min_distance_cm=30):
    """
    Find the top-N wavenumber peaks in a 1-D spectral attention curve.

    Returns list of (wavenumber, attention_value) sorted by descending weight.
    """
    if spectral_weights.max() == 0:
        return []
    step_cm = abs(float(wavenumbers[1] - wavenumbers[0]))
    min_dist = max(1, int(min_distance_cm / step_cm))
    peaks, props = find_peaks(
        spectral_weights, distance=min_dist, height=0,
        prominence=0.05 * (spectral_weights.max() - spectral_weights.min()),
    )
    if len(peaks) == 0:
        peaks = np.array([np.argmax(spectral_weights)])
    order = np.argsort(spectral_weights[peaks])[::-1]
    top = peaks[order[:n_regions]]
    return [(float(wavenumbers[p]), float(spectral_weights[p])) for p in top]


def _top_channel_dims(channel_attn, n_top=5):
    """
    Given a (d_model, d_model) channel-axis attention matrix, return
    the top-N most-attended channel dimensions (by mean received attention).
    """
    importance = channel_attn.mean(axis=0)  # mean attention received per dim
    top_idx = np.argsort(importance)[::-1][:n_top]
    return [(int(i), float(importance[i])) for i in top_idx]


def print_token_axis_summary(label, attn_dict, wavenumbers, num_spectral_points,
                             patch_size, n_regions=5):
    """Print top spectral regions for token-axis attention (merged + per head)."""
    intra_maps = attn_dict['intra_attention']
    n_heads = len(intra_maps)

    # Merged across all heads
    merged = np.mean(intra_maps, axis=0)
    weights = _patches_to_spectrum_weights(merged, num_spectral_points, patch_size)
    regions = _find_top_regions(wavenumbers, weights, n_regions)

    print(f"\n{'='*62}")
    print(f"  TOKEN-AXIS ATTENTION — {label}")
    print(f"{'='*62}")
    print(f"  All heads merged:")
    if regions:
        print(f"    {'Rank':<6} {'Wavenumber (cm⁻¹)':<22} {'Attention'}")
        print(f"    {'-'*44}")
        for i, (wn, val) in enumerate(regions, 1):
            print(f"    {i:<6} {wn:<22.1f} {val:.4f}")
    else:
        print(f"    (uniform attention — no focused regions)")

    # Per head
    for h in range(n_heads):
        weights_h = _patches_to_spectrum_weights(
            intra_maps[h], num_spectral_points, patch_size
        )
        regions_h = _find_top_regions(wavenumbers, weights_h, n_regions)
        print(f"\n  Head {h + 1}:")
        if regions_h:
            print(f"    {'Rank':<6} {'Wavenumber (cm⁻¹)':<22} {'Attention'}")
            print(f"    {'-'*44}")
            for i, (wn, val) in enumerate(regions_h, 1):
                print(f"    {i:<6} {wn:<22.1f} {val:.4f}")
        else:
            print(f"    (uniform attention)")
    print(f"{'='*62}\n")


def print_channel_axis_summary(label, attn_dict, n_top=5):
    """Print top channel dimensions for channel-axis attention (merged + per head)."""
    inter_maps = attn_dict['inter_attention']
    n_heads = len(inter_maps)

    merged = np.mean(inter_maps, axis=0)
    top_dims = _top_channel_dims(merged, n_top)

    print(f"\n{'='*62}")
    print(f"  CHANNEL-AXIS ATTENTION — {label}")
    print(f"{'='*62}")
    print(f"  All heads merged:")
    print(f"    {'Rank':<6} {'Channel dim':<16} {'Mean attn received'}")
    print(f"    {'-'*44}")
    for i, (dim, val) in enumerate(top_dims, 1):
        print(f"    {i:<6} {dim:<16} {val:.4f}")

    for h in range(n_heads):
        top_h = _top_channel_dims(inter_maps[h], n_top)
        print(f"\n  Head {h + 1}:")
        print(f"    {'Rank':<6} {'Channel dim':<16} {'Mean attn received'}")
        print(f"    {'-'*44}")
        for i, (dim, val) in enumerate(top_h, 1):
            print(f"    {i:<6} {dim:<16} {val:.4f}")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(random_state: int = 1, n_splits: int = 10):
    print("Loading and preprocessing data...")
    X, y, wavenumbers = load_and_preprocess_data()
    num_spectral_points = X.shape[1]
    print(f"Data shape: {X.shape}  |  Classes: {np.bincount(y)}  |  "
          f"Wavenumber range: {wavenumbers[-1]:.0f}–{wavenumbers[0]:.0f} cm⁻¹")

    if n_splits == 1:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(y))
        test_size = max(1, int(0.3 * len(y)))
        folds = [(indices[test_size:], indices[:test_size])]
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(skf.split(X, y))

    # Accumulate full attention dicts across all folds and all test samples
    all_attn_dicts_cancer  = []
    all_attn_dicts_healthy = []
    all_spectra_cancer     = []
    all_spectra_healthy    = []

    for fold_idx, (train_index, test_index) in enumerate(folds):
        fold_num = fold_idx + 1
        print(f"\n{'='*60}")
        print(f"Fold {fold_num}/{len(folds)}")
        print(f"{'='*60}")

        X_train_fold = X[train_index]
        X_test_fold  = X[test_index]
        y_train_fold = y[train_index]
        y_test_fold  = y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold,
            test_size=0.3,
            random_state=random_state,
            stratify=y_train_fold,
        )

        model = BioSpectralFormer(num_spectral_points=num_spectral_points)
        model.train_model(X_train, y_train, X_val, y_val)
        model.calibrate_threshold(X_val, y_val)

        fold_dir = OUTPUT_BASE / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # --- Per-class indices in test fold ---
        cancer_indices  = np.where(y_test_fold == 1)[0]
        healthy_indices = np.where(y_test_fold == 0)[0]

        if len(cancer_indices) == 0 or len(healthy_indices) == 0:
            print(f"  Skipping fold {fold_num}: only one class in test set.")
            continue

        # --- Per-sample plots (first sample of each class only) ---
        for label_name, idx in [('cancer', cancer_indices[0]),
                                 ('healthy', healthy_indices[0])]:
            X_sample = X_test_fold[idx : idx + 1]
            spectrum  = X_test_fold[idx]

            print(f"  Plotting attention for {label_name} sample (test idx={idx})...")

            attn_last = model.get_attention_maps(X_sample, layer_idx=-1)
            plot_attention_maps(
                attn_dict=attn_last,
                spectra=spectrum,
                wavenumbers=wavenumbers,
                num_spectral_points=num_spectral_points,
                patch_size=model.patch_size,
                save_path=str(fold_dir / f"attention_{label_name}.png"),
                layer_idx=-1,
            )

            all_layer_maps = [
                model.get_attention_maps(X_sample, layer_idx=i)
                for i in range(model.num_layers)
            ]
            plot_layer_comparison(
                all_attn_maps=all_layer_maps,
                save_path=str(fold_dir / f"layers_{label_name}.png"),
            )

        print(f"  Saved per-sample figures → {fold_dir}")

        # --- Accumulate attention dicts for ALL test samples ---
        print(f"  Accumulating attention data for all {len(test_index)} test samples...")
        for idx in cancer_indices:
            attn = model.get_attention_maps(X_test_fold[idx : idx + 1], layer_idx=-1)
            all_attn_dicts_cancer.append(attn)
            all_spectra_cancer.append(X_test_fold[idx])

        for idx in healthy_indices:
            attn = model.get_attention_maps(X_test_fold[idx : idx + 1], layer_idx=-1)
            all_attn_dicts_healthy.append(attn)
            all_spectra_healthy.append(X_test_fold[idx])

    # -----------------------------------------------------------------------
    # Cross-fold summary: average attention and plot like the per-fold plots
    # -----------------------------------------------------------------------
    if len(all_attn_dicts_cancer) == 0 or len(all_attn_dicts_healthy) == 0:
        print("No data accumulated – exiting.")
        return

    print(f"\nAccumulated {len(all_attn_dicts_cancer)} cancer and "
          f"{len(all_attn_dicts_healthy)} healthy samples across all folds.")

    mean_attn_cancer  = _average_attn_dicts(all_attn_dicts_cancer)
    mean_attn_healthy = _average_attn_dicts(all_attn_dicts_healthy)
    mean_spectrum_cancer  = np.mean(all_spectra_cancer, axis=0)
    mean_spectrum_healthy = np.mean(all_spectra_healthy, axis=0)

    for label_name, mean_attn in [
        ('Cancer',  mean_attn_cancer),
        ('Healthy', mean_attn_healthy),
    ]:
        print_token_axis_summary(
            label_name, mean_attn, wavenumbers,
            num_spectral_points, model.patch_size,
        )
        print_channel_axis_summary(label_name, mean_attn)

    for label_name, mean_attn, mean_spectrum in [
        ('cancer',  mean_attn_cancer,  mean_spectrum_cancer),
        ('healthy', mean_attn_healthy, mean_spectrum_healthy),
    ]:
        save_path = str(OUTPUT_BASE / f"mean_attention_{label_name}.png")
        print(f"  Plotting mean attention for {label_name}...")
        plot_attention_maps(
            attn_dict=mean_attn,
            spectra=mean_spectrum,
            wavenumbers=wavenumbers,
            num_spectral_points=num_spectral_points,
            patch_size=model.patch_size,
            save_path=save_path,
            layer_idx=-1,
        )

    print(f"\nDone.  All outputs saved under {OUTPUT_BASE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualise BioSpectralFormer attention maps on the real dataset"
    )
    parser.add_argument("--seed",  type=int, default=1,  help="Random seed (default: 1)")
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds (default: 10)")
    args = parser.parse_args()

    main(random_state=args.seed, n_splits=args.folds)
