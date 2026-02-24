import torch
import numpy as np
from preProcess.fingerprint_trucate import WavenumberTruncator


def _wavenumber_range_to_truncated_indices(
    truncator: WavenumberTruncator,
    wn_high: int,
    wn_low: int,
    trunc_start: int,
    trunc_end: int,
) -> tuple[int, int]:
    """
    Convert a wavenumber range to indices in the truncated spectral space.

    The wavenumber file is in DESCENDING order. get_range_indices returns
    (lower_line, upper_line) in full-file space where lower_line corresponds
    to the higher wavenumber. We offset by trunc_start to get truncated-space indices.

    Returns:
        (start_idx, end_idx) in truncated space, clamped to valid range.
    """
    lower_line, upper_line = truncator.get_range_indices(wn_high, wn_low)

    num_truncated = trunc_end - trunc_start + 1
    start_idx = max(0, lower_line - trunc_start)
    end_idx = min(num_truncated - 1, upper_line - trunc_start)

    return start_idx, end_idx


def _compute_patch_overlaps(
    region_start: int,
    region_end: int,
    patch_size: int,
    num_patches: int,
) -> np.ndarray:
    """
    Compute fractional overlap of each patch with a spectral region.

    Patch i covers truncated indices [stride*i, stride*i + patch_size - 1]
    where stride = patch_size // 2.

    Returns:
        Array of shape (num_patches,) with values in [0.0, 1.0].
    """
    stride = patch_size // 2
    overlaps = np.zeros(num_patches, dtype=np.float32)

    for i in range(num_patches):
        patch_start = stride * i
        patch_end = patch_start + patch_size - 1

        overlap_start = max(patch_start, region_start)
        overlap_end = min(patch_end, region_end)

        if overlap_start <= overlap_end:
            overlaps[i] = (overlap_end - overlap_start + 1) / patch_size

    return overlaps


def build_region_attention_mask(
    region_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    patch_size: int = 16,
    num_spectral_points: int = 1141,
    truncation_range: tuple[int, int] = (3050, 850),
    wavenumber_file: str = "wavenumbers_cancboca.dat",
    penalty: float = 10.0,
) -> torch.Tensor:
    """
    Build a soft attention mask based on biochemical region interactions.

    For each region pair (A, B), patches overlapping region A can attend to
    patches overlapping region B (and vice versa). Patches within the same
    region also attend to each other. The weight is proportional to how much
    each patch overlaps with its respective region.

    Args:
        region_pairs: List of ((wn_high_A, wn_low_A), (wn_high_B, wn_low_B)).
            Each tuple defines a pair of interacting biochemical regions.
            Example: [((1700, 1600), (1580, 1480))]  for Amide I <-> Amide II
        patch_size: Size of each spectral patch.
        num_spectral_points: Number of spectral points after truncation.
        truncation_range: (upper_wn, lower_wn) used during preprocessing.
        wavenumber_file: Path to the wavenumber reference file.
        penalty: Suppression strength for non-relevant interactions.
            Higher = harder masking. 0 disables masking entirely.

    Returns:
        Additive bias tensor of shape (num_patches, num_patches).
        Values in [-penalty, 0]: 0 for fully relevant, -penalty for irrelevant.
    """
    stride = patch_size // 2
    num_patches = (num_spectral_points - patch_size) // stride + 1

    truncator = WavenumberTruncator(wavenumber_file)
    trunc_start, trunc_end = truncator.get_range_indices(
        truncation_range[0], truncation_range[1]
    )

    W = np.zeros((num_patches, num_patches), dtype=np.float32)

    for region_a, region_b in region_pairs:
        start_a, end_a = _wavenumber_range_to_truncated_indices(
            truncator, region_a[0], region_a[1], trunc_start, trunc_end
        )
        start_b, end_b = _wavenumber_range_to_truncated_indices(
            truncator, region_b[0], region_b[1], trunc_start, trunc_end
        )

        overlap_a = _compute_patch_overlaps(start_a, end_a, patch_size, num_patches)
        overlap_b = _compute_patch_overlaps(start_b, end_b, patch_size, num_patches)

        # Cross-region: A <-> B (symmetric)
        W_cross = np.outer(overlap_a, overlap_b)
        W_cross = W_cross + W_cross.T
        W_cross = np.minimum(W_cross, 1.0)

        # Self-region: patches within same region attend to each other
        W_self_a = np.outer(overlap_a, overlap_a)
        W_self_b = np.outer(overlap_b, overlap_b)

        W = np.maximum(W, W_cross)
        W = np.maximum(W, W_self_a)
        W = np.maximum(W, W_self_b)

    bias = penalty * (torch.tensor(W, dtype=torch.float32) - 1.0)

    return bias
