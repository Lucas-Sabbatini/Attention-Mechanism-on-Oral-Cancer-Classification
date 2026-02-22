"""
Attention map visualization utilities for BioSpectralFormer.

Two public functions are provided:

    plot_attention_maps()  - per-sample attention heatmaps + spectrum overlay
    plot_layer_comparison() - compare averaged attention across all layers
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patches_to_spectrum_weights(
    intra_attn: np.ndarray,
    num_spectral_points: int,
    patch_size: int = 16,
) -> np.ndarray:
    """
    Collapse a (seq_len, seq_len) intra-sample attention map to a 1-D weight
    vector aligned with the original spectral axis.

    Strategy: average each row of the attention matrix to get one importance
    weight per patch, then map patches back to spectral points via a simple
    overlapping-patch reconstruction (each point receives the mean weight of
    all patches that cover it).

    Args:
        intra_attn: Attention weight matrix, shape (seq_len, seq_len).
        num_spectral_points: Length of the original spectrum.
        patch_size: Kernel / patch size used in the convolutional embedding.

    Returns:
        1-D numpy array of shape (num_spectral_points,) with values in [0, 1].
    """
    # Average over query dimension -> one weight per key patch
    patch_weights = intra_attn.mean(axis=0)  # (seq_len,)
    seq_len = len(patch_weights)

    # Guard: check whether the patch-level attention is essentially uniform
    # by comparing its entropy to the theoretical maximum (log(seq_len)).
    # A near-maximum entropy means no patch was preferred over any other →
    # returning zeros is more honest than amplifying float noise.
    pw_sum = patch_weights.sum()
    if pw_sum > 0:
        pw_norm     = patch_weights / pw_sum
        entropy     = float(-np.sum(pw_norm * np.log(pw_norm + 1e-12)))
        max_entropy = np.log(seq_len)
        if entropy >= 0.99 * max_entropy:
            return np.zeros(num_spectral_points)

    # Determine stride (50 % overlap matches SpectralTransformer default)
    stride = patch_size // 2

    # Accumulate weights onto the spectral axis
    spectral_weights = np.zeros(num_spectral_points)
    count = np.zeros(num_spectral_points)
    for p_idx in range(seq_len):
        start = p_idx * stride
        end = min(start + patch_size, num_spectral_points)
        spectral_weights[start:end] += patch_weights[p_idx]
        count[start:end] += 1

    count = np.maximum(count, 1)
    spectral_weights /= count

    # Normalise to [0, 1]
    w_min, w_max = spectral_weights.min(), spectral_weights.max()
    w_range = w_max - w_min
    if w_range == 0:
        return np.zeros_like(spectral_weights)
    spectral_weights = (spectral_weights - w_min) / w_range

    return spectral_weights


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_attention_maps(
    attn_dict: dict,
    spectra: Optional[np.ndarray] = None,
    wavenumbers: Optional[np.ndarray] = None,
    num_spectral_points: int = 1141,
    patch_size: int = 16,
    save_path: Optional[str] = None,
    layer_idx: int = -1,
) -> None:
    """
    Visualise inter-feature and intra-sample attention maps for one layer.

    Creates a 3-row figure:
        Row 1 – Heatmaps of inter-feature attention (one per head).
        Row 2 – Heatmaps of intra-sample attention (one per head).
        Row 3 – Spectrum with attention overlay (similar to ACT paper Fig. 3).

    Args:
        attn_dict: Dictionary returned by ``SpectralTransformerModel.get_attention_maps()``.
            Expected keys: ``'inter_attention'``, ``'intra_attention'``, ``'alpha'``.
        spectra: 1-D array of spectral intensities (num_spectral_points,).
            When provided, Row 3 shows the actual spectrum.
        wavenumbers: 1-D array of wavenumber values (num_spectral_points,).
            When provided, used as x-axis labels for Row 3.
        num_spectral_points: Number of spectral points in the original signal.
        patch_size: Convolutional patch size used during embedding.
        save_path: If given, save figure here at 300 DPI; otherwise display.
        layer_idx: Layer index shown in the figure title (informational only).
    """
    inter_maps = attn_dict['inter_attention']  # list of (d_model, d_model)
    intra_maps = attn_dict['intra_attention']  # list of (seq_len, seq_len)
    alpha = attn_dict['alpha']
    n_heads = len(inter_maps)

    fig = plt.figure(figsize=(4 * n_heads, 12))
    fig.suptitle(
        f'BioSpectralFormer Attention Maps  '
        f'(layer {layer_idx}, α={alpha:.3f})',
        fontsize=14, fontweight='bold', y=1.01,
    )

    gs = gridspec.GridSpec(3, n_heads, figure=fig, hspace=0.45, wspace=0.35)

    # --- Row 1: inter-feature attention ---
    for h in range(n_heads):
        ax = fig.add_subplot(gs[0, h])
        sns.heatmap(
            inter_maps[h],
            ax=ax,
            cmap='viridis',
            square=True,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(f'Inter-feature\nHead {h + 1}', fontsize=9)
        if h == 0:
            ax.set_ylabel('Feature dim', fontsize=8)

    # --- Row 2: intra-sample attention ---
    for h in range(n_heads):
        ax = fig.add_subplot(gs[1, h])
        sns.heatmap(
            intra_maps[h],
            ax=ax,
            cmap='magma',
            square=True,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(f'Intra-sample\nHead {h + 1}', fontsize=9)
        if h == 0:
            ax.set_ylabel('Patch (query)', fontsize=8)
            ax.set_xlabel('Patch (key)', fontsize=8)

    # --- Row 3: spectrum + attention overlay ---
    # Merge all heads by averaging
    merged_intra = np.stack(intra_maps, axis=0).mean(axis=0)  # (seq_len, seq_len)
    spectral_weights = _patches_to_spectrum_weights(
        merged_intra, num_spectral_points, patch_size
    )

    ax_overlay = fig.add_subplot(gs[2, :])

    x_axis = wavenumbers if wavenumbers is not None else np.arange(num_spectral_points)
    x_label = 'Wavenumber (cm⁻¹)' if wavenumbers is not None else 'Spectral point index'

    # Attention fill (normalised 0-1 already)
    ax_overlay.fill_between(
        x_axis,
        spectral_weights,
        alpha=0.45,
        color='red',
        label='Attention weight',
    )
    ax_overlay.set_ylabel('Attention weight', color='red', fontsize=9)
    ax_overlay.tick_params(axis='y', labelcolor='red')

    if spectra is not None:
        ax2 = ax_overlay.twinx()
        ax2.plot(x_axis, spectra, color='steelblue', linewidth=0.9, label='Spectrum')
        ax2.set_ylabel('Absorbance', color='steelblue', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='steelblue')

    ax_overlay.set_xlabel(x_label, fontsize=9)
    ax_overlay.set_title('Mean intra-sample attention projected onto spectral axis', fontsize=9)

    # Reverse x-axis if wavenumbers descend (typical FTIR convention)
    if wavenumbers is not None and wavenumbers[0] > wavenumbers[-1]:
        ax_overlay.invert_xaxis()

    lines1, labels1 = ax_overlay.get_legend_handles_labels()
    if spectra is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_overlay.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    else:
        ax_overlay.legend(fontsize=8, loc='upper right')

    _save_or_show(fig, save_path)


def plot_layer_comparison(
    all_attn_maps: list[dict],
    save_path: Optional[str] = None,
) -> None:
    """
    Compare averaged attention patterns across all transformer layers.

    Creates a (n_layers × 2) subplot grid:
        Left column  – mean inter-feature attention map per layer.
        Right column – mean intra-sample attention map per layer.

    Args:
        all_attn_maps: List of attention dictionaries, one per layer, as
            returned by ``SpectralTransformerModel.get_attention_maps()``
            when called with ``layer_idx`` for each layer — **or** directly
            the ``all_attn_maps`` list from ``SpectralTransformer.forward()``
            with ``return_attention=True`` (after converting tensors to numpy
            via ``.cpu().numpy().mean(0)``).

            Each dict must have keys:
                ``'inter_attention'``: list of (d_model, d_model) arrays.
                ``'intra_attention'``: list of (seq_len, seq_len) arrays.
                ``'alpha'``: float.
        save_path: If given, save figure here at 300 DPI; otherwise display.
    """
    n_layers = len(all_attn_maps)
    fig, axes = plt.subplots(n_layers, 2, figsize=(10, 4 * n_layers))

    # Ensure axes is always 2-D
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        'BioSpectralFormer – Attention Evolution Across Layers',
        fontsize=13, fontweight='bold',
    )

    for layer_idx, layer_maps in enumerate(all_attn_maps):
        alpha = layer_maps['alpha']
        inter_maps = layer_maps['inter_attention']
        intra_maps = layer_maps['intra_attention']

        # Average over heads
        mean_inter = np.stack(inter_maps, axis=0).mean(axis=0)
        mean_intra = np.stack(intra_maps, axis=0).mean(axis=0)

        # Left: inter-feature
        ax_inter = axes[layer_idx, 0]
        sns.heatmap(
            mean_inter,
            ax=ax_inter,
            cmap='viridis',
            square=True,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
        )
        ax_inter.set_title(f'Layer {layer_idx + 1} – Inter-feature (α={alpha:.3f})', fontsize=9)
        ax_inter.set_ylabel('Feature dim', fontsize=8)

        # Right: intra-sample
        ax_intra = axes[layer_idx, 1]
        sns.heatmap(
            mean_intra,
            ax=ax_intra,
            cmap='magma',
            square=True,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
        )
        ax_intra.set_title(f'Layer {layer_idx + 1} – Intra-sample (α={alpha:.3f})', fontsize=9)

    plt.tight_layout()
    _save_or_show(fig, save_path)
