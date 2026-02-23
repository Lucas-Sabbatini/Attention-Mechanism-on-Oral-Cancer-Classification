"""
Friedman + Nemenyi post-hoc statistical comparison of multiple classifiers.

Implements the two-step procedure from Demsar (2006):
  1. Friedman test  — non-parametric omnibus test across k classifiers and N datasets/folds.
  2. Nemenyi test   — post-hoc pairwise comparisons when Friedman rejects H0.
  3. CD diagram     — Demsar-style Critical Difference visualization.

Public API:

    friedman_nemenyi_test() — run Friedman + Nemenyi, print results table
    plot_cd_diagram()       — draw and optionally save the CD diagram

Reference:
    Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
    Journal of Machine Learning Research, 7, 1-30.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import friedmanchisquare, norm, rankdata


# ---------------------------------------------------------------------------
# Constants (Demsar 2006, Table 5)
# ---------------------------------------------------------------------------

_METRIC_IDX: dict[str, int] = {
    "accuracy":    0,
    "precision":   1,
    "sensitivity": 2,
    "specificity": 3,
    "mean_se_sp":  4,
}

# Critical values q_alpha for the two-tailed Nemenyi test (k = number of classifiers)
_Q_ALPHA: dict[float, dict[int, float]] = {
    0.05: {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
           7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
           7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920},
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_q_alpha(k: int, alpha: float) -> float:
    """Return the Nemenyi critical value for k classifiers at the given alpha level."""
    table = _Q_ALPHA.get(alpha)
    if table is None:
        raise ValueError(f"alpha must be one of {list(_Q_ALPHA.keys())}. Got {alpha}.")
    if k not in table:
        raise ValueError(
            f"q_alpha table covers k=2..10. Got k={k}. "
            "Provide a custom q_alpha for larger comparisons."
        )
    return table[k]


def _build_score_matrix(
    results: dict[str, list[tuple]], metric_idx: int
) -> tuple[np.ndarray, list[str]]:
    """
    Build a (N_folds, k_models) score matrix from the results dict.

    Returns (score_matrix, ordered_model_names).
    """
    names = list(results.keys())
    k = len(names)
    n = len(results[names[0]])
    matrix = np.zeros((n, k), dtype=float)
    for j, name in enumerate(names):
        for i, fold in enumerate(results[name]):
            matrix[i, j] = fold[metric_idx]
    return matrix, names


def _compute_avg_ranks(score_matrix: np.ndarray) -> np.ndarray:
    """
    Rank classifiers within each fold (rank 1 = highest score) and return
    the average rank per classifier across all folds.

    shape: (k,)
    """
    n, k = score_matrix.shape
    ranks = np.array([rankdata(-score_matrix[i]) for i in range(n)])  # (N, k)
    return ranks.mean(axis=0)


def _critical_difference(k: int, n: int, q_alpha: float) -> float:
    """CD = q_alpha * sqrt(k*(k+1) / (6*N))  [Demsar 2006, eq. 3]"""
    return q_alpha * np.sqrt(k * (k + 1) / (6 * n))


def _nemenyi_pvalue(rank_diff: float, k: int, n: int) -> float:
    """
    Normal approximation for the Nemenyi pairwise p-value.

    z = |rank_diff| / sqrt(k*(k+1)/(6*N))
    p = 2 * (1 - Phi(z))
    """
    se = np.sqrt(k * (k + 1) / (6 * n))
    if se == 0:
        return float("nan")
    z = abs(rank_diff) / se
    return float(2 * (1 - norm.cdf(z)))


def _find_cliques(
    names: list[str], avg_ranks: np.ndarray, cd: float
) -> list[list[str]]:
    """
    Find maximal groups (cliques) of classifiers whose pairwise average-rank
    differences are all less than the critical difference.

    Returns a list of groups (each group is a list of model names sorted by rank).
    Only groups with ≥ 2 members that are not proper subsets of a larger group are kept.
    """
    order = np.argsort(avg_ranks)  # best (lowest rank) first
    sorted_names = [names[i] for i in order]
    sorted_ranks = avg_ranks[order]
    k = len(sorted_names)

    # Build all maximal contiguous groups
    # (contiguous in rank-sorted order; non-contiguous cliques are also checked)
    raw_groups: list[set[str]] = []

    for start in range(k):
        group = {sorted_names[start]}
        for end in range(start + 1, k):
            # A group is valid if ALL pairwise differences < cd
            candidate = group | {sorted_names[end]}
            members = [i for i, n in enumerate(sorted_names) if n in candidate]
            if sorted_ranks[members[-1]] - sorted_ranks[members[0]] < cd:
                group = candidate
            else:
                break
        if len(group) >= 2:
            raw_groups.append(group)

    # Remove subsets
    maximal: list[set[str]] = []
    for g in raw_groups:
        if not any(g < other for other in raw_groups):
            maximal.append(g)

    # Convert back to ordered lists
    result = []
    for g in maximal:
        ordered = [n for n in sorted_names if n in g]
        if ordered not in result:
            result.append(ordered)
    return result


def _print_friedman_result(stat: float, p: float, alpha: float, metric: str) -> None:
    decision = "REJECT H0 — significant differences exist" if p < alpha else "FAIL TO REJECT H0 — no significant differences"
    print(f"\n{'='*62}")
    print(f"  FRIEDMAN TEST  |  metric: {metric}")
    print(f"{'='*62}")
    print(f"  chi² = {stat:.4f},  p = {p:.6f},  alpha = {alpha}")
    print(f"  → {decision}")
    print(f"{'='*62}")


def _print_nemenyi_table(
    names: list[str],
    avg_ranks: np.ndarray,
    pairwise: dict,
    cd: float,
) -> None:
    print(f"\n  Critical Difference (CD) = {cd:.4f}")
    print(f"  Sig: * p<0.05  ** p<0.01  ns not significant\n")

    # Average ranks summary
    order = np.argsort(avg_ranks)
    print(f"  {'Model':<22} {'Avg Rank':>10}")
    print(f"  {'-'*34}")
    for i in order:
        print(f"  {names[i]:<22} {avg_ranks[i]:>10.3f}")

    # Pairwise table
    print(f"\n  {'Model 1':<22} {'Model 2':<22} {'Rank Diff':>10} {'p-value':>10} {'Sig':>5}")
    print(f"  {'-'*72}")
    for (m1, m2), info in pairwise.items():
        sig = "**" if info["p"] < 0.01 else ("*" if info["p"] < 0.05 else "ns")
        print(f"  {m1:<22} {m2:<22} {info['rank_diff']:>10.3f} {info['p']:>10.4f} {sig:>5}")
    print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def friedman_nemenyi_test(
    results: dict[str, list[tuple]],
    metric: str = "mean_se_sp",
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Run Friedman omnibus test + Nemenyi post-hoc pairwise comparisons.

    Args:
        results: {model_name: [fold_tuple, ...]}
                 fold_tuple = (accuracy, precision, sensitivity, specificity, mean_se_sp)
        metric:  which metric to compare. One of:
                 "accuracy", "precision", "sensitivity", "specificity", "mean_se_sp"
        alpha:   significance level, 0.05 or 0.10
        verbose: print Friedman result and pairwise Nemenyi table

    Returns:
        {
          "avg_ranks":     {model_name: float},
          "friedman_stat": float,
          "friedman_p":    float,
          "cd":            float,
          "significant":   bool,
          "pairwise":      {(m1, m2): {"rank_diff": float, "p": float, "significant": bool}},
        }

    Raises:
        ValueError: if metric is invalid, alpha is unsupported, or k is outside 2–10.
    """
    if metric not in _METRIC_IDX:
        raise ValueError(f"metric must be one of {list(_METRIC_IDX.keys())}. Got '{metric}'.")

    metric_idx = _METRIC_IDX[metric]
    score_matrix, names = _build_score_matrix(results, metric_idx)
    n, k = score_matrix.shape

    # Friedman test
    cols = [score_matrix[:, j] for j in range(k)]
    stat, p = friedmanchisquare(*cols)

    # Rankings
    avg_ranks_arr = _compute_avg_ranks(score_matrix)

    # Critical difference
    q_alpha = _get_q_alpha(k, alpha)
    cd = _critical_difference(k, n, q_alpha)

    # Pairwise Nemenyi comparisons
    pairwise: dict = {}
    for i in range(k):
        for j in range(i + 1, k):
            diff = float(abs(avg_ranks_arr[i] - avg_ranks_arr[j]))
            pval = _nemenyi_pvalue(diff, k, n)
            pairwise[(names[i], names[j])] = {
                "rank_diff": diff,
                "p":         pval,
                "significant": diff > cd,
            }

    avg_ranks_dict = {names[j]: float(avg_ranks_arr[j]) for j in range(k)}

    if verbose:
        _print_friedman_result(stat, p, alpha, metric)
        if p < alpha:
            _print_nemenyi_table(names, avg_ranks_arr, pairwise, cd)
        else:
            print("  Post-hoc Nemenyi test skipped (Friedman not significant).\n")

    return {
        "avg_ranks":     avg_ranks_dict,
        "friedman_stat": float(stat),
        "friedman_p":    float(p),
        "cd":            float(cd),
        "significant":   bool(p < alpha),
        "pairwise":      pairwise,
    }


def plot_cd_diagram(
    avg_ranks: dict[str, float],
    cd: float,
    title: str = "Critical Difference Diagram",
    alpha: float = 0.05,
    save_path: str | None = None,
) -> None:
    """
    Draw a Demsar-style Critical Difference diagram.

    Models are sorted by average rank (rank 1 = best, placed on the left).
    Thick horizontal bars connect classifiers that are NOT significantly different
    (i.e. their average rank difference is less than cd).

    Args:
        avg_ranks: {model_name: avg_rank_float}
        cd:        critical difference value from friedman_nemenyi_test()
        title:     plot title
        alpha:     used only for the axis label
        save_path: file path to save the figure (PNG/PDF). If None, plt.show() is called.
    """
    names = list(avg_ranks.keys())
    ranks = np.array([avg_ranks[n] for n in names])

    # Sort best → worst (ascending rank)
    order = np.argsort(ranks)
    sorted_names = [names[i] for i in order]
    sorted_ranks = ranks[order]
    k = len(sorted_names)

    cliques = _find_cliques(sorted_names, sorted_ranks, cd)

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    fig_width = max(10, k * 1.2)
    fig_height = 4.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(1 - 0.5, k + 0.5)
    ax.set_ylim(-0.6, 1.2)
    ax.axis("off")

    # Main horizontal axis line
    ax.plot([1, k], [0, 0], "k-", linewidth=2, zorder=2)

    # Tick marks on axis
    for r in range(1, k + 1):
        ax.plot([r, r], [-0.05, 0.05], "k-", linewidth=1.5)

    # Rank labels
    for r in range(1, k + 1):
        ax.text(r, -0.12, str(r), ha="center", va="top", fontsize=9)

    # -----------------------------------------------------------------------
    # Place model names: top-half above axis, bottom-half below
    # -----------------------------------------------------------------------
    split = k // 2

    for idx, (name, rank) in enumerate(zip(sorted_names, sorted_ranks)):
        if idx < split:
            # Above the axis
            y_text = 0.55 + 0.18 * (idx % 2)  # slight stagger to avoid overlap
            ax.plot([rank, rank], [0.05, y_text - 0.07], "k-", linewidth=1)
            ax.text(rank, y_text, name, ha="center", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        else:
            # Below the axis
            y_text = -0.35 - 0.18 * (idx % 2)
            ax.plot([rank, rank], [-0.05, y_text + 0.07], "k-", linewidth=1)
            ax.text(rank, y_text, name, ha="center", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # -----------------------------------------------------------------------
    # Draw clique bars (thick horizontal lines above the axis)
    # -----------------------------------------------------------------------
    bar_heights = [0.22, 0.32, 0.42]  # stacked if multiple cliques
    for bar_idx, clique in enumerate(cliques):
        clique_ranks = sorted([avg_ranks[n] for n in clique])
        y = bar_heights[bar_idx % len(bar_heights)]
        ax.plot(
            [clique_ranks[0], clique_ranks[-1]],
            [y, y],
            linewidth=5,
            color="steelblue",
            solid_capstyle="butt",
            zorder=3,
        )

    # -----------------------------------------------------------------------
    # CD reference bracket at top-right
    # -----------------------------------------------------------------------
    cd_x_start = k - cd
    cd_x_end = k
    cd_y = 1.05
    ax.annotate(
        "", xy=(cd_x_end, cd_y), xytext=(cd_x_start, cd_y),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
    )
    ax.text(
        (cd_x_start + cd_x_end) / 2, cd_y + 0.06,
        f"CD = {cd:.3f}",
        ha="center", va="bottom", fontsize=9, fontstyle="italic",
    )

    # -----------------------------------------------------------------------
    # Title and axis label
    # -----------------------------------------------------------------------
    ax.text(
        (1 + k) / 2, 1.18,
        title,
        ha="center", va="top", fontsize=11, fontweight="bold",
    )
    ax.text(1, -0.20, "← better", ha="left", va="top", fontsize=8, color="gray")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  CD diagram saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()
