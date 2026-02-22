"""
Paired t-test utilities for comparing cross-validation results between two models.

Public API:

    paired_t_test() - compare two models across all 5 evaluation metrics
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METRIC_NAMES: list[str] = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "mean_se_sp",
]

_METRIC_LABELS: dict[str, str] = {
    "accuracy":    "Accuracy",
    "precision":   "Precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "mean_se_sp":  "Mean(SE,SP)",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_metric(fold_results: list[tuple], metric_idx: int) -> np.ndarray:
    """Extract a single metric column from a list of fold result tuples."""
    return np.array([fold[metric_idx] for fold in fold_results], dtype=float)


def _cohens_d_paired(differences: np.ndarray) -> float:
    """
    Cohen's d for a paired design: mean(diff) / std(diff, ddof=1).

    Returns nan if std is zero (no variance in differences).
    """
    std = np.std(differences, ddof=1)
    if std == 0.0:
        return float("nan")
    return float(np.mean(differences) / std)


def _significance_label(p_value: float, alpha: float) -> str:
    if np.isnan(p_value):
        return "nan"
    if p_value < 0.01:
        return "**"
    if p_value < alpha:
        return "*"
    return "ns"


def _run_paired_tests(a: np.ndarray, b: np.ndarray, alpha: float) -> dict:
    """Run paired t-test and Wilcoxon signed-rank test on two matched arrays."""
    diff = a - b

    # Paired t-test
    try:
        t_stat, t_p = ttest_rel(a, b)
    except Exception:
        t_stat, t_p = float("nan"), float("nan")

    # Wilcoxon signed-rank (non-parametric alternative)
    try:
        if np.all(diff == 0.0):
            raise ValueError("All differences are zero.")
        w_stat, w_p = wilcoxon(diff, alternative="two-sided")
    except ValueError as exc:
        warnings.warn(
            f"Wilcoxon test could not be computed: {exc}. Returning nan.",
            RuntimeWarning,
            stacklevel=4,
        )
        w_stat, w_p = float("nan"), float("nan")

    d = _cohens_d_paired(diff)

    return {
        "mean_model1": float(np.mean(a)),
        "mean_model2": float(np.mean(b)),
        "mean_diff":   float(np.mean(diff)),
        "std_diff":    float(np.std(diff, ddof=1)),
        "t_stat":      float(t_stat),
        "t_p":         float(t_p),
        "t_sig":       _significance_label(t_p, alpha),
        "w_stat":      float(w_stat),
        "w_p":         float(w_p),
        "w_sig":       _significance_label(w_p, alpha),
        "cohens_d":    d,
    }


def _print_table(results: dict, model1_name: str, model2_name: str, alpha: float) -> None:
    """Print a formatted summary table to stdout."""
    c = 12  # numeric column width
    n1 = model1_name[:c]
    n2 = model2_name[:c]

    separator = "-" * (14 + c * 9 + 5 * 2)

    print(f"\nPaired t-test: {model1_name} vs {model2_name}  (alpha={alpha})")
    print(f"  Sig: ** p<0.01  * p<0.05  ns not significant  |  Positive diff = {model1_name} higher")
    print(separator)
    print(
        f"{'Metric':<14}"
        f"{n1:>{c}}{n2:>{c}}"
        f"{'Mean Diff':>{c}}{'t-stat':>{c}}{'p (t-test)':>{c}}{'Sig':>5}"
        f"{'Cohen\'s d':>{c}}{'p (Wilcox)':>{c}}{'WSig':>5}"
    )
    print(separator)

    for metric in _METRIC_NAMES:
        r = results[metric]
        label = _METRIC_LABELS[metric]
        cohens_d_str = f"{r['cohens_d']:>{c}.3f}" if not np.isnan(r['cohens_d']) else f"{'nan':>{c}}"
        print(
            f"{label:<14}"
            f"{r['mean_model1']:>{c}.4f}"
            f"{r['mean_model2']:>{c}.4f}"
            f"{r['mean_diff']:>{c}.4f}"
            f"{r['t_stat']:>{c}.3f}"
            f"{r['t_p']:>{c}.4f}"
            f"{r['t_sig']:>5}"
            f"{cohens_d_str}"
            f"{r['w_p']:>{c}.4f}"
            f"{r['w_sig']:>5}"
        )

    print(separator)
    print(f"  n_folds = {results['_meta']['n_folds']}\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def paired_t_test(
    model1_results: list[tuple],
    model2_results: list[tuple],
    alpha: float = 0.05,
    model1_name: str = "BioSpectralFormer",
    model2_name: str = "LightGBM",
    verbose: bool = True,
) -> dict:
    """
    Run paired t-tests comparing two models across all 5 evaluation metrics.

    Each model's results must be a list of fold tuples in the format returned
    by ``BaseClassifierModel.evaluate()``:
        (accuracy, precision, sensitivity, specificity, mean_se_sp)

    Both parametric (paired t-test) and non-parametric (Wilcoxon signed-rank)
    tests are computed. With only 10 folds the normality assumption of the
    t-test is fragile, so both results are reported.

    Args:
        model1_results: List of fold tuples from model 1's cross-validation.
        model2_results: List of fold tuples from model 2's cross-validation.
        alpha:          Significance threshold. Default 0.05.
        model1_name:    Display name for model 1 in the printed table.
        model2_name:    Display name for model 2 in the printed table.
        verbose:        Print a formatted summary table. Default True.

    Returns:
        Dict with keys "accuracy", "precision", "sensitivity", "specificity",
        "mean_se_sp", each containing:
            mean_model1, mean_model2, mean_diff, std_diff,
            t_stat, t_p, t_sig,
            w_stat, w_p, w_sig,
            cohens_d
        Plus a "_meta" key with n_folds, alpha, model1_name, model2_name.

    Raises:
        ValueError: If result lists differ in length, are empty, or any tuple
                    has fewer than 5 elements.
    """
    if len(model1_results) != len(model2_results):
        raise ValueError(
            f"Result lists must have the same length. "
            f"Got {len(model1_results)} and {len(model2_results)}."
        )
    if len(model1_results) == 0:
        raise ValueError("Result lists must not be empty.")
    all_tuples = model1_results + model2_results
    if any(len(t) < 5 for t in all_tuples):
        raise ValueError(
            "Each fold tuple must contain at least 5 elements: "
            "(accuracy, precision, sensitivity, specificity, mean_se_sp)."
        )

    output: dict = {}

    for idx, metric in enumerate(_METRIC_NAMES):
        a = _extract_metric(model1_results, idx)
        b = _extract_metric(model2_results, idx)
        output[metric] = _run_paired_tests(a, b, alpha)

    output["_meta"] = {
        "n_folds":     len(model1_results),
        "alpha":       alpha,
        "model1_name": model1_name,
        "model2_name": model2_name,
    }

    if verbose:
        _print_table(output, model1_name, model2_name, alpha)

    return output
