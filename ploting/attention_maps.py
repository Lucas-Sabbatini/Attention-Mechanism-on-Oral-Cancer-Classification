"""
Attention Map Visualization Script

For each cross-validation fold, trains the BioSpectralFormer on the real
dataset and:
  1. Saves per-sample attention heatmaps and layer-comparison plots.
  2. Accumulates the spectral attention profile (intra-sample attention
     projected back onto the wavenumber axis) for every test sample.

After all folds a cross-fold mean is computed and a summary figure is saved:
  ploting/img/attention/mean_attention_regions.png

The top attention regions are also printed to stdout as a ranked table.

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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
# Spectral attention weight extraction
# ---------------------------------------------------------------------------

def _head_entropy(attn_map):
    """Mean Shannon entropy of a (seq_len, seq_len) row-stochastic attention map."""
    return float(-np.sum(attn_map * np.log(attn_map + 1e-12), axis=-1).mean())


def _entropy_weighted_merge(intra_maps):
    """
    Merge per-head intra-sample attention maps with inverse-entropy weighting.

    A head with low entropy (sharp, focused) gets more weight than a diffuse
    head with high entropy. This prevents uninformative heads from washing out
    the signal from focused ones.

    Args:
        intra_maps: list of (seq_len, seq_len) arrays, one per head.

    Returns:
        (seq_len, seq_len) weighted mean, and list of per-head entropies.
    """
    entropies = [_head_entropy(m) for m in intra_maps]
    inv_ent   = np.array([1.0 / (e + 1e-8) for e in entropies])
    weights   = inv_ent / inv_ent.sum()
    merged    = sum(w * m for w, m in zip(weights, intra_maps))
    return merged, entropies


def extract_spectral_weights(model, X_sample, num_spectral_points, patch_size):
    """
    Return the spectral attention weight vector, the merged 2-D attention
    matrix, and per-head entropy values for one sample.

    Uses entropy-weighted head merging so focused heads are not washed out
    by diffuse ones. Then projects from patch space to spectral-point space.
    Result is normalised to [0, 1]; returns zeros when attention is uniform.

    Returns:
        weights    : (num_spectral_points,) array in [0, 1]
        attn_matrix: (seq_len, seq_len) entropy-weighted merged attention
        entropies  : list of per-head Shannon entropy values
    """
    attn = model.get_attention_maps(X_sample, layer_idx=-1)
    merged, entropies = _entropy_weighted_merge(attn['intra_attention'])
    weights = _patches_to_spectrum_weights(merged, num_spectral_points, patch_size)
    return weights, merged, entropies


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def plot_mean_attention_regions(
    wavenumbers,
    weights_cancer,
    weights_healthy,
    top_regions,
    save_path,
):
    """
    Plot cross-fold mean spectral attention curves and highlight the top
    attention bands identified from the combined (cancer + healthy) mean.

    Args:
        wavenumbers:    1-D array (num_spectral_points,) in cm⁻¹.
        weights_cancer: 2-D array (n_cancer_samples, num_spectral_points).
        weights_healthy:2-D array (n_healthy_samples, num_spectral_points).
        top_regions:    list of dicts from _find_top_regions().
        save_path:      output PNG path.
    """
    mean_cancer  = weights_cancer.mean(axis=0)
    mean_healthy = weights_healthy.mean(axis=0)
    mean_combined = np.vstack([weights_cancer, weights_healthy]).mean(axis=0)

    std_cancer  = weights_cancer.std(axis=0)
    std_healthy = weights_healthy.std(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(
        'BioSpectralFormer – Cross-fold Mean Spectral Attention',
        fontsize=13, fontweight='bold',
    )

    ax = axes[0]

    # Cancer curve
    ax.plot(wavenumbers, mean_cancer,  color='#E94F37', linewidth=1.5,
            label=f'Cancer  (n={len(weights_cancer)})')
    ax.fill_between(wavenumbers,
                    mean_cancer - std_cancer,
                    mean_cancer + std_cancer,
                    color='#E94F37', alpha=0.15)

    # Healthy curve
    ax.plot(wavenumbers, mean_healthy, color='#2E86AB', linewidth=1.5,
            label=f'Healthy (n={len(weights_healthy)})')
    ax.fill_between(wavenumbers,
                    mean_healthy - std_healthy,
                    mean_healthy + std_healthy,
                    color='#2E86AB', alpha=0.15)

    # Highlight top regions
    REGION_COLORS = ['#FF9F1C', '#8AC926', '#6A4C93', '#1982C4', '#FF595E']
    for i, reg in enumerate(top_regions):
        col = REGION_COLORS[i % len(REGION_COLORS)]
        ax.axvspan(reg['wn_min'], reg['wn_max'],
                   alpha=0.18, color=col,
                   label=f'R{i+1}: {reg["wn_center"]:.0f} cm⁻¹')

    ax.set_ylabel('Normalised attention weight', fontsize=10)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.35)

    # Invert x-axis (FTIR convention: high → low wavenumber)
    if wavenumbers[0] > wavenumbers[-1]:
        ax.invert_xaxis()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))

    # Bottom panel: difference (cancer - healthy)
    ax2 = axes[1]
    diff = mean_cancer - mean_healthy
    ax2.bar(wavenumbers, diff,
            color=np.where(diff >= 0, '#E94F37', '#2E86AB'),
            width=(wavenumbers[0] - wavenumbers[1]) if wavenumbers[0] > wavenumbers[-1]
                   else (wavenumbers[1] - wavenumbers[0]),
            align='center')
    ax2.axhline(0, color='k', linewidth=0.7)
    ax2.set_ylabel('Cancer − Healthy', fontsize=9)
    ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    if wavenumbers[0] > wavenumbers[-1]:
        ax2.invert_xaxis()
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(200))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved summary figure → {save_path}")


# ---------------------------------------------------------------------------
# Mean attention + spectra summary plot
# ---------------------------------------------------------------------------

def _patch_wn_ticks(wavenumbers, seq_len, patch_size, n_ticks=6):
    """
    Build tick positions (in patch-index space) and wavenumber labels for
    a (seq_len × seq_len) attention heatmap.
    """
    stride       = patch_size // 2
    step         = max(1, seq_len // (n_ticks - 1))
    patch_idx    = list(range(0, seq_len, step))
    if patch_idx[-1] != seq_len - 1:
        patch_idx.append(seq_len - 1)
    spectral_idx = [min(p * stride + patch_size // 2, len(wavenumbers) - 1)
                    for p in patch_idx]
    labels       = [f"{wavenumbers[si]:.0f}" for si in spectral_idx]
    return patch_idx, labels


def plot_mean_attention_and_spectra(
    wavenumbers,
    mean_attn_cancer,
    mean_attn_healthy,
    weights_cancer,
    weights_healthy,
    spectra_cancer,
    spectra_healthy,
    patch_size,
    save_path,
):
    """
    Publication-quality summary figure combining:
      Row 1 (3 panels): mean intra-sample attention heatmaps
                        cancer | healthy | difference (cancer − healthy)
      Row 2 (full width): mean ± std spectra for both classes
                          with cross-fold spectral attention overlaid

    Args:
        wavenumbers:       1-D array (num_spectral_points,) in cm⁻¹.
        mean_attn_cancer:  (seq_len, seq_len) mean attention matrix, cancer.
        mean_attn_healthy: (seq_len, seq_len) mean attention matrix, healthy.
        weights_cancer:    (N_cancer,  num_spectral_points) spectral weights.
        weights_healthy:   (N_healthy, num_spectral_points) spectral weights.
        spectra_cancer:    (N_cancer,  num_spectral_points) raw spectra.
        spectra_healthy:   (N_healthy, num_spectral_points) raw spectra.
        patch_size:        Convolutional patch size (for tick labelling).
        save_path:         Output PNG path.
    """
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    seq_len   = mean_attn_cancer.shape[0]
    diff_attn = mean_attn_cancer - mean_attn_healthy

    # Build wavenumber tick labels for the heatmap axes
    tick_pos, tick_labels = _patch_wn_ticks(wavenumbers, seq_len, patch_size)

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1.6, 1],
        hspace=0.42, wspace=0.38,
    )
    fig.suptitle(
        'BioSpectralFormer – Cross-fold Mean Attention & Spectra',
        fontsize=13, fontweight='bold',
    )

    # --- Shared colour scale for cancer / healthy heatmaps ---
    vmax = max(mean_attn_cancer.max(), mean_attn_healthy.max())

    def _draw_heatmap(ax, matrix, title, cmap, vmin=0, vmax=vmax, cbar=True):
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            square=True,
            cbar=cbar,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels, rotation=0, fontsize=7)
        ax.set_xlabel('Key patch (cm⁻¹)',   fontsize=8)
        ax.set_ylabel('Query patch (cm⁻¹)', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')

    ax_c = fig.add_subplot(gs[0, 0])
    _draw_heatmap(ax_c, mean_attn_cancer,
                  f'Mean attention – Cancer (n={len(spectra_cancer)})',
                  'Reds')

    ax_h = fig.add_subplot(gs[0, 1])
    _draw_heatmap(ax_h, mean_attn_healthy,
                  f'Mean attention – Healthy (n={len(spectra_healthy)})',
                  'Blues')

    # Difference heatmap: diverging colour scale centred at 0
    ax_d = fig.add_subplot(gs[0, 2])
    abs_max = np.abs(diff_attn).max()
    _draw_heatmap(ax_d, diff_attn,
                  'Difference (Cancer − Healthy)',
                  'RdBu_r', vmin=-abs_max, vmax=abs_max)

    # --- Bottom row: mean spectra + spectral attention overlay ---
    ax_s = fig.add_subplot(gs[1, :])

    mean_sp_c = spectra_cancer.mean(axis=0)
    std_sp_c  = spectra_cancer.std(axis=0)
    mean_sp_h = spectra_healthy.mean(axis=0)
    std_sp_h  = spectra_healthy.std(axis=0)

    mean_wt_c = weights_cancer.mean(axis=0)
    mean_wt_h = weights_healthy.mean(axis=0)

    # Spectra (primary axis)
    ax_s.plot(wavenumbers, mean_sp_c, color='#E94F37', linewidth=1.2,
              label=f'Cancer spectrum (n={len(spectra_cancer)})')
    ax_s.fill_between(wavenumbers,
                      mean_sp_c - std_sp_c,
                      mean_sp_c + std_sp_c,
                      color='#E94F37', alpha=0.12)

    ax_s.plot(wavenumbers, mean_sp_h, color='#2E86AB', linewidth=1.2,
              label=f'Healthy spectrum (n={len(spectra_healthy)})')
    ax_s.fill_between(wavenumbers,
                      mean_sp_h - std_sp_h,
                      mean_sp_h + std_sp_h,
                      color='#2E86AB', alpha=0.12)

    ax_s.set_ylabel('Absorbance (normalised)', fontsize=9)
    ax_s.set_xlabel('Wavenumber (cm⁻¹)',       fontsize=9)
    ax_s.grid(True, linestyle='--', alpha=0.3)

    # Attention weights (secondary axis)
    ax_w = ax_s.twinx()
    ax_w.fill_between(wavenumbers, mean_wt_c,
                      alpha=0.30, color='#E94F37',
                      label='Cancer attention')
    ax_w.fill_between(wavenumbers, mean_wt_h,
                      alpha=0.30, color='#2E86AB',
                      label='Healthy attention')
    ax_w.set_ylabel('Attention weight (mean)', fontsize=9, color='dimgray')
    ax_w.tick_params(axis='y', labelcolor='dimgray')
    ax_w.set_ylim(0, 1.4)   # leave headroom so fills don't crowd the spectra

    # FTIR convention: high → low wavenumber
    if wavenumbers[0] > wavenumbers[-1]:
        ax_s.invert_xaxis()
        ax_w.invert_xaxis()
    ax_s.xaxis.set_major_locator(ticker.MultipleLocator(200))

    # Merge legends from both axes
    lines1, labs1 = ax_s.get_legend_handles_labels()
    lines2, labs2 = ax_w.get_legend_handles_labels()
    ax_s.legend(lines1 + lines2, labs1 + labs2,
                fontsize=8, ncol=2, loc='upper right')
    ax_s.set_title('Mean spectra with cross-fold spectral attention overlay',
                   fontsize=9)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved mean attention + spectra figure → {save_path}")


# ---------------------------------------------------------------------------
# Region detection
# ---------------------------------------------------------------------------

def _find_top_regions(wavenumbers, combined_weights, n_regions=5,
                      min_distance=30, max_region_cm=300):
    """
    Identify the top-N distinct wavenumber regions by attention magnitude.

    Uses scipy find_peaks with a prominence threshold to reject trivial bumps,
    then expands each peak to its half-maximum width — capped at max_region_cm
    so a single region can never balloon to cover the full spectrum.

    Args:
        wavenumbers:      1-D array of wavenumber values (may be descending).
        combined_weights: 1-D mean spectral attention curve in [0, 1].
        n_regions:        Number of top regions to return.
        min_distance:     Minimum separation between peaks in cm⁻¹.
        max_region_cm:    Hard cap on how wide one region can be (cm⁻¹).

    Returns:
        List of dicts sorted by descending mean_attn:
            wn_center, wn_min, wn_max, mean_attn, peak_attn
        Empty list when the curve is all-zeros (uniform attention).
    """
    wn = wavenumbers
    w  = combined_weights

    # All-zero curve means no focused attention — nothing to report
    if w.max() == 0:
        return []

    step_cm          = abs(float(wn[1]) - float(wn[0]))
    min_idx_distance = max(1, int(min_distance / step_cm))
    max_half_idx     = max(1, int(max_region_cm / 2 / step_cm))

    # Require prominence ≥ 5 % of the curve range to filter noise bumps
    prominence_threshold = 0.05 * (w.max() - w.min())

    peaks, _ = find_peaks(
        w,
        distance=min_idx_distance,
        height=0,
        prominence=max(prominence_threshold, 1e-4),
    )

    if len(peaks) == 0:
        # Fallback: global maximum only
        peaks = np.array([np.argmax(w)])

    # Take the top-N peaks by height
    order     = np.argsort(w[peaks])[::-1]
    top_peaks = peaks[order[:n_regions]]

    regions = []
    for p in top_peaks:
        half  = w[p] / 2.0
        left  = p
        right = p
        # Expand left, but no further than max_half_idx indices
        while left > 0 and w[left - 1] >= half and (p - left) < max_half_idx:
            left -= 1
        # Expand right, same cap
        while right < len(w) - 1 and w[right + 1] >= half and (right - p) < max_half_idx:
            right += 1

        wn_lo = min(wn[left], wn[right])
        wn_hi = max(wn[left], wn[right])

        regions.append({
            'wn_center': float(wn[p]),
            'wn_min':    float(wn_lo),
            'wn_max':    float(wn_hi),
            'peak_attn': float(w[p]),
            'mean_attn': float(w[left:right + 1].mean()),
        })

    regions.sort(key=lambda r: r['mean_attn'], reverse=True)
    return regions


def print_top_regions(regions, class_label='combined'):
    """Print a ranked table of highest-attention wavenumber regions."""
    if not regions:
        print(f"\n  [{class_label}]  No focused attention regions found "
              "(all samples had near-uniform attention).\n")
        return
    print(f"\n{'='*62}")
    print(f"  TOP ATTENTION REGIONS  ({class_label})")
    print(f"{'='*62}")
    print(f"  {'Rank':<5} {'Center (cm⁻¹)':<16} {'Range (cm⁻¹)':<22} {'Mean attn':<12} {'Peak attn'}")
    print(f"  {'-'*57}")
    for i, r in enumerate(regions, 1):
        rng = f"{r['wn_min']:.0f} – {r['wn_max']:.0f}"
        print(f"  {i:<5} {r['wn_center']:<16.1f} {rng:<22} {r['mean_attn']:<12.4f} {r['peak_attn']:.4f}")
    print(f"{'='*62}\n")


def print_fold_entropy_summary(fold_entropies):
    """
    Print a per-fold table showing mean per-head attention entropy.

    Low entropy  → focused attention (model learned something specific).
    High entropy → diffuse attention (model spread attention uniformly).
    The maximum possible entropy for seq_len=141 is ln(141) ≈ 4.95 nats.
    """
    print(f"\n{'='*72}")
    print(f"  FOLD-LEVEL ATTENTION ENTROPY  (lower = more focused)")
    print(f"  Max possible entropy for 141 patches ≈ {np.log(141):.2f} nats")
    print(f"{'='*72}")
    header = f"  {'Fold':<6} {'Class':<10} {'Head 1':>8} {'Head 2':>8} {'Head 3':>8} {'Head 4':>8} {'Mean':>8}"
    print(header)
    print(f"  {'-'*66}")
    for entry in fold_entropies:
        per_head = "  ".join(f"{e:>6.3f}" for e in entry['entropies'])
        mean_e   = np.mean(entry['entropies'])
        focus    = "focused" if mean_e < np.log(141) * 0.5 else "diffuse"
        print(f"  {entry['fold']:<6} {entry['class']:<10} {per_head}  {mean_e:>6.3f}  [{focus}]")
    print(f"{'='*72}\n")


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

    # Accumulate data across all folds and all test samples
    all_weights_cancer   = []   # 1-D spectral weight per sample
    all_weights_healthy  = []
    all_attn_cancer      = []   # 2-D merged attention matrix per sample
    all_attn_healthy     = []
    all_spectra_cancer   = []   # raw spectrum per sample
    all_spectra_healthy  = []
    fold_entropy_log     = []   # per-sample entropy records for the summary table

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

        # --- Accumulate data for ALL test samples ---
        print(f"  Accumulating attention data for all {len(test_index)} test samples...")
        for idx in cancer_indices:
            w, mat, ent = extract_spectral_weights(
                model, X_test_fold[idx : idx + 1], num_spectral_points, model.patch_size
            )
            all_weights_cancer.append(w)
            all_attn_cancer.append(mat)
            all_spectra_cancer.append(X_test_fold[idx])
            fold_entropy_log.append({'fold': fold_num, 'class': 'cancer',  'entropies': ent})

        for idx in healthy_indices:
            w, mat, ent = extract_spectral_weights(
                model, X_test_fold[idx : idx + 1], num_spectral_points, model.patch_size
            )
            all_weights_healthy.append(w)
            all_attn_healthy.append(mat)
            all_spectra_healthy.append(X_test_fold[idx])
            fold_entropy_log.append({'fold': fold_num, 'class': 'healthy', 'entropies': ent})

    # -----------------------------------------------------------------------
    # Cross-fold summary
    # -----------------------------------------------------------------------
    if len(all_weights_cancer) == 0 or len(all_weights_healthy) == 0:
        print("No data accumulated – exiting.")
        return

    weights_cancer  = np.array(all_weights_cancer)    # (N_cancer,  num_pts)
    weights_healthy = np.array(all_weights_healthy)   # (N_healthy, num_pts)
    attn_cancer     = np.array(all_attn_cancer)        # (N_cancer,  seq, seq)
    attn_healthy    = np.array(all_attn_healthy)       # (N_healthy, seq, seq)
    spectra_cancer  = np.array(all_spectra_cancer)     # (N_cancer,  num_pts)
    spectra_healthy = np.array(all_spectra_healthy)    # (N_healthy, num_pts)

    print(f"\nAccumulated {len(weights_cancer)} cancer and "
          f"{len(weights_healthy)} healthy samples across all folds.")

    combined = np.vstack([weights_cancer, weights_healthy]).mean(axis=0)

    # Identify top regions from the combined mean
    top_regions = _find_top_regions(wavenumbers, combined, n_regions=5)

    # Also compute class-specific top regions for reporting
    top_cancer  = _find_top_regions(wavenumbers, weights_cancer.mean(axis=0),  n_regions=5)
    top_healthy = _find_top_regions(wavenumbers, weights_healthy.mean(axis=0), n_regions=5)

    print_fold_entropy_summary(fold_entropy_log)
    print_top_regions(top_regions, class_label='cancer + healthy combined')
    print_top_regions(top_cancer,  class_label='cancer only')
    print_top_regions(top_healthy, class_label='healthy only')

    plot_mean_attention_regions(
        wavenumbers=wavenumbers,
        weights_cancer=weights_cancer,
        weights_healthy=weights_healthy,
        top_regions=top_regions,
        save_path=str(OUTPUT_BASE / "mean_attention_regions.png"),
    )

    # SpectralTransformerModel stores patch_size; grab it from defaults if needed
    _patch_size = BioSpectralFormer().patch_size

    plot_mean_attention_and_spectra(
        wavenumbers=wavenumbers,
        mean_attn_cancer=attn_cancer.mean(axis=0),
        mean_attn_healthy=attn_healthy.mean(axis=0),
        weights_cancer=weights_cancer,
        weights_healthy=weights_healthy,
        spectra_cancer=spectra_cancer,
        spectra_healthy=spectra_healthy,
        patch_size=_patch_size,
        save_path=str(OUTPUT_BASE / "mean_attention_and_spectra.png"),
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
