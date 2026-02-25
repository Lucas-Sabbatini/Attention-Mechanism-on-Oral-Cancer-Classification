import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class FRDAQualityReport:
    """
    Diagnostic suite for assessing FRDA-generated spectra quality.

    Produces statistical tests, boundary discontinuity metrics, and
    visualization plots comparing original vs augmented data.
    """

    def __init__(
        self,
        X_original: np.ndarray,
        y_original: np.ndarray,
        X_augmented: np.ndarray,
        y_augmented: np.ndarray,
        class_boundaries: dict[int, tuple[int, int]],
        wavenumbers: np.ndarray = None,
        output_dir: str = None,
    ):
        """
        Args:
            X_original: Original training spectra (N_orig, num_spectral_points).
            y_original: Original labels (N_orig,).
            X_augmented: Full augmented array (original + synthetic stacked).
            y_augmented: Full augmented labels.
            class_boundaries: {class_id: (b1, b2)} from GMMSegmenter.
            wavenumbers: Optional wavenumber axis for x-axis labels.
            output_dir: Directory to save plots. If None, plots are shown.
        """
        self.X_orig = X_original
        self.y_orig = y_original
        self.X_aug = X_augmented
        self.y_aug = y_augmented
        self.boundaries = class_boundaries
        self.wavenumbers = wavenumbers
        self.n_orig = len(X_original)
        self.n_points = X_original.shape[1]

        # Synthetic samples are those appended after the originals
        self.X_synth = X_augmented[self.n_orig:]
        self.y_synth = y_augmented[self.n_orig:]

        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

    def _x_axis(self):
        """Return wavenumber array or index array for plotting."""
        if self.wavenumbers is not None:
            return self.wavenumbers
        return np.arange(self.n_points)

    def _x_label(self):
        if self.wavenumbers is not None:
            return "Wavenumber (cm$^{-1}$)"
        return "Spectral Index"

    def _save_or_show(self, fig, name):
        if self.output_dir is not None:
            fig.savefig(self.output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------
    # 1. Spectral overlay: original vs synthetic per class
    # ------------------------------------------------------------------
    def plot_spectral_overlay(self, n_samples: int = 10):
        """
        Plot original and synthetic spectra overlaid per class to visually
        check that augmented spectra look realistic.
        """
        classes = np.unique(self.y_orig)
        x = self._x_axis()
        fig, axes = plt.subplots(len(classes), 1, figsize=(12, 4 * len(classes)),
                                 squeeze=False)

        for idx, cls in enumerate(classes):
            ax = axes[idx, 0]
            orig_cls = self.X_orig[self.y_orig == cls]
            synth_cls = self.X_synth[self.y_synth == cls] if len(self.X_synth) > 0 else np.empty((0, self.n_points))

            # Plot originals
            n_plot = min(n_samples, len(orig_cls))
            for i in range(n_plot):
                ax.plot(x, orig_cls[i], color="steelblue", alpha=0.4, linewidth=0.8)
            ax.plot([], [], color="steelblue", label=f"Original (n={len(orig_cls)})")

            # Plot synthetics
            n_plot_s = min(n_samples, len(synth_cls))
            for i in range(n_plot_s):
                ax.plot(x, synth_cls[i], color="coral", alpha=0.4, linewidth=0.8)
            ax.plot([], [], color="coral", label=f"Synthetic (n={len(synth_cls)})")

            # Mark boundaries
            b1, b2 = self.boundaries.get(int(cls), (self.n_points // 3, 2 * self.n_points // 3))
            for b, lbl in [(b1, "b1"), (b2, "b2")]:
                bx = x[b] if b < len(x) else x[-1]
                ax.axvline(bx, color="black", linestyle="--", alpha=0.6, label=lbl)

            ax.set_title(f"Class {int(cls)}")
            ax.set_xlabel(self._x_label())
            ax.set_ylabel("Intensity")
            if self.wavenumbers is not None:
                ax.invert_xaxis()
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.3)

        fig.suptitle("Spectral Overlay: Original vs Synthetic", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save_or_show(fig, "spectral_overlay")

    # ------------------------------------------------------------------
    # 2. Mean/std comparison per class
    # ------------------------------------------------------------------
    def plot_mean_std_comparison(self):
        """
        Compare mean and std of original vs synthetic spectra per class.
        Good augmentation should preserve the distributional shape.
        """
        classes = np.unique(self.y_orig)
        x = self._x_axis()
        fig, axes = plt.subplots(len(classes), 1, figsize=(12, 4 * len(classes)),
                                 squeeze=False)

        for idx, cls in enumerate(classes):
            ax = axes[idx, 0]
            orig_cls = self.X_orig[self.y_orig == cls]
            synth_cls = self.X_synth[self.y_synth == cls] if len(self.X_synth) > 0 else np.empty((0, self.n_points))

            m_o, s_o = orig_cls.mean(axis=0), orig_cls.std(axis=0)
            ax.plot(x, m_o, color="steelblue", linewidth=2, label="Original mean")
            ax.fill_between(x, m_o - s_o, m_o + s_o, color="steelblue", alpha=0.15)

            if len(synth_cls) > 0:
                m_s, s_s = synth_cls.mean(axis=0), synth_cls.std(axis=0)
                ax.plot(x, m_s, color="coral", linewidth=2, linestyle="--", label="Synthetic mean")
                ax.fill_between(x, m_s - s_s, m_s + s_s, color="coral", alpha=0.15)

            b1, b2 = self.boundaries.get(int(cls), (self.n_points // 3, 2 * self.n_points // 3))
            for b in [b1, b2]:
                bx = x[b] if b < len(x) else x[-1]
                ax.axvline(bx, color="black", linestyle="--", alpha=0.6)

            ax.set_title(f"Class {int(cls)}")
            ax.set_xlabel(self._x_label())
            ax.set_ylabel("Intensity")
            if self.wavenumbers is not None:
                ax.invert_xaxis()
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.3)

        fig.suptitle("Mean ± Std: Original vs Synthetic", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save_or_show(fig, "mean_std_comparison")

    # ------------------------------------------------------------------
    # 3. Boundary discontinuity analysis
    # ------------------------------------------------------------------
    def compute_boundary_discontinuity(self) -> dict[int, dict]:
        """
        Measure mean absolute jump at splice points b1 and b2 in synthetic
        spectra compared to the same positions in original spectra.

        Returns:
            {class_id: {'b1_synth': float, 'b1_orig': float,
                        'b2_synth': float, 'b2_orig': float}}
        """
        results = {}
        for cls, (b1, b2) in self.boundaries.items():
            orig_cls = self.X_orig[self.y_orig == cls]
            synth_cls = self.X_synth[self.y_synth == cls] if len(self.X_synth) > 0 else None

            def _jump(data, idx):
                if data is None or len(data) == 0 or idx <= 0 or idx >= data.shape[1]:
                    return 0.0
                return float(np.mean(np.abs(data[:, idx] - data[:, idx - 1])))

            results[int(cls)] = {
                "b1_orig": _jump(orig_cls, b1),
                "b1_synth": _jump(synth_cls, b1),
                "b2_orig": _jump(orig_cls, b2),
                "b2_synth": _jump(synth_cls, b2),
            }
        return results

    def plot_boundary_discontinuity(self):
        """Bar plot comparing discontinuity at boundaries: original vs synthetic."""
        disc = self.compute_boundary_discontinuity()
        classes = sorted(disc.keys())

        labels = []
        orig_vals = []
        synth_vals = []
        for cls in classes:
            for bname in ["b1", "b2"]:
                labels.append(f"Class {cls} {bname}")
                orig_vals.append(disc[cls][f"{bname}_orig"])
                synth_vals.append(disc[cls][f"{bname}_synth"])

        x_pos = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
        ax.bar(x_pos - width / 2, orig_vals, width, label="Original", color="steelblue")
        ax.bar(x_pos + width / 2, synth_vals, width, label="Synthetic", color="coral")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Mean |jump| at boundary")
        ax.set_title("Boundary Discontinuity: Original vs Synthetic", fontweight="bold")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        self._save_or_show(fig, "boundary_discontinuity")

    # ------------------------------------------------------------------
    # 4. Per-band KS test (distribution similarity)
    # ------------------------------------------------------------------
    def compute_ks_tests(self, n_bands: int = 20) -> dict[int, list[dict]]:
        """
        Split the spectrum into `n_bands` equal bands and run a KS test
        (original vs synthetic) on the mean intensity per band per class.

        Returns:
            {class_id: [{'band': (start, end), 'ks_stat': float, 'p_value': float}, ...]}
        """
        band_edges = np.linspace(0, self.n_points, n_bands + 1, dtype=int)
        results = {}

        for cls in np.unique(self.y_orig):
            orig_cls = self.X_orig[self.y_orig == cls]
            synth_cls = self.X_synth[self.y_synth == cls] if len(self.X_synth) > 0 else None
            band_results = []

            for i in range(n_bands):
                s, e = band_edges[i], band_edges[i + 1]
                orig_band = orig_cls[:, s:e].mean(axis=1)
                if synth_cls is not None and len(synth_cls) > 0:
                    synth_band = synth_cls[:, s:e].mean(axis=1)
                    stat, pval = ks_2samp(orig_band, synth_band)
                else:
                    stat, pval = 0.0, 1.0
                band_results.append({"band": (int(s), int(e)), "ks_stat": stat, "p_value": pval})

            results[int(cls)] = band_results
        return results

    def plot_ks_tests(self, n_bands: int = 20):
        """Heatmap-style plot of KS p-values across spectral bands per class."""
        ks = self.compute_ks_tests(n_bands)
        classes = sorted(ks.keys())
        fig, ax = plt.subplots(figsize=(12, 2 + len(classes)))

        matrix = []
        y_labels = []
        for cls in classes:
            pvals = [b["p_value"] for b in ks[cls]]
            matrix.append(pvals)
            y_labels.append(f"Class {cls}")

        matrix = np.array(matrix)
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        band_labels = [f"{b['band'][0]}-{b['band'][1]}" for b in ks[classes[0]]]
        ax.set_xticks(range(len(band_labels)))
        ax.set_xticklabels(band_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Spectral Band (index range)")
        ax.set_title("KS Test p-values: Original vs Synthetic (green = similar)", fontweight="bold")
        fig.colorbar(im, ax=ax, label="p-value")
        fig.tight_layout()
        self._save_or_show(fig, "ks_test_bands")

    # ------------------------------------------------------------------
    # 5. PCA scatter: original vs synthetic, colored by class + origin
    # ------------------------------------------------------------------
    def plot_pca_scatter(self):
        """
        PCA projection of all spectra. Points colored by class, markers
        distinguish original (circle) vs synthetic (cross).
        """
        pca = PCA(n_components=2, random_state=42)
        X_all = np.vstack([self.X_orig, self.X_synth]) if len(self.X_synth) > 0 else self.X_orig
        coords = pca.fit_transform(X_all)

        n_o = len(self.X_orig)
        coords_orig = coords[:n_o]
        coords_synth = coords[n_o:]

        class_colors = {0: "green", 1: "red"}
        classes = np.unique(self.y_orig)

        fig, ax = plt.subplots(figsize=(8, 6))

        for cls in classes:
            color = class_colors.get(int(cls), f"C{int(cls)}")
            mask_o = self.y_orig == cls
            ax.scatter(coords_orig[mask_o, 0], coords_orig[mask_o, 1],
                       c=color, marker="o", alpha=0.6, s=40,
                       label=f"Orig class {int(cls)}")

            if len(self.X_synth) > 0:
                mask_s = self.y_synth == cls
                if mask_s.any():
                    ax.scatter(coords_synth[mask_s, 0], coords_synth[mask_s, 1],
                               c=color, marker="x", alpha=0.4, s=30,
                               label=f"Synth class {int(cls)}")

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title("PCA: Original vs Synthetic Spectra", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        self._save_or_show(fig, "pca_scatter")

    # ------------------------------------------------------------------
    # 6. Silhouette score comparison
    # ------------------------------------------------------------------
    def compute_silhouette_scores(self) -> dict[str, float]:
        """
        Compute silhouette scores for class separation:
        - original only
        - augmented (original + synthetic)
        Higher is better. Augmentation should preserve or improve separation.
        """
        results = {}
        if len(np.unique(self.y_orig)) < 2:
            return {"original": -1.0, "augmented": -1.0}

        try:
            results["original"] = float(silhouette_score(self.X_orig, self.y_orig))
        except Exception:
            results["original"] = -1.0

        try:
            results["augmented"] = float(silhouette_score(self.X_aug, self.y_aug))
        except Exception:
            results["augmented"] = -1.0

        return results

    # ------------------------------------------------------------------
    # 7. Full report
    # ------------------------------------------------------------------
    def generate_report(self, n_bands: int = 20, save_plots: bool = None):
        """
        Run all diagnostics and print a summary.

        Args:
            n_bands: Number of spectral bands for KS test.
            save_plots: Whether to generate and save plots. Defaults to True
                when output_dir is set, False otherwise.

        Returns:
            Dictionary with all computed metrics.
        """
        if save_plots is None:
            save_plots = self.output_dir is not None

        report = {}

        # Class counts
        orig_counts = {int(c): int(n) for c, n in zip(*np.unique(self.y_orig, return_counts=True))}
        aug_counts = {int(c): int(n) for c, n in zip(*np.unique(self.y_aug, return_counts=True))}
        synth_counts = {int(c): int(n) for c, n in zip(*np.unique(self.y_synth, return_counts=True))} if len(self.y_synth) > 0 else {}
        report["class_counts"] = {
            "original": orig_counts,
            "synthetic": synth_counts,
            "augmented_total": aug_counts,
        }

        # Boundaries
        report["boundaries"] = {int(k): v for k, v in self.boundaries.items()}

        # Boundary discontinuity
        disc = self.compute_boundary_discontinuity()
        report["boundary_discontinuity"] = disc

        # KS tests
        ks = self.compute_ks_tests(n_bands)
        # Summarize: fraction of bands with p > 0.05 (no significant difference)
        ks_summary = {}
        for cls, bands in ks.items():
            n_pass = sum(1 for b in bands if b["p_value"] > 0.05)
            ks_summary[cls] = {"bands_similar": n_pass, "bands_total": len(bands),
                               "fraction_similar": n_pass / len(bands)}
        report["ks_summary"] = ks_summary

        # Silhouette
        sil = self.compute_silhouette_scores()
        report["silhouette"] = sil

        # Generate plots only when we have a directory to save them
        if save_plots:
            self.plot_spectral_overlay()
            self.plot_mean_std_comparison()
            self.plot_boundary_discontinuity()
            self.plot_ks_tests(n_bands)
            self.plot_pca_scatter()

        # Print summary
        print("\n" + "=" * 70)
        print("FRDA QUALITY REPORT")
        print("=" * 70)

        print(f"\nSample counts:")
        for cls in sorted(orig_counts.keys()):
            n_o = orig_counts.get(cls, 0)
            n_s = synth_counts.get(cls, 0)
            print(f"  Class {cls}: {n_o} original + {n_s} synthetic = {aug_counts.get(cls, n_o)} total")

        print(f"\nBoundaries:")
        for cls, (b1, b2) in sorted(report["boundaries"].items()):
            pct1 = b1 / self.n_points * 100
            pct2 = b2 / self.n_points * 100
            print(f"  Class {cls}: b1={b1} ({pct1:.0f}%), b2={b2} ({pct2:.0f}%)")

        print(f"\nBoundary discontinuity (mean |jump|):")
        for cls in sorted(disc.keys()):
            d = disc[cls]
            b1_ratio = d["b1_synth"] / d["b1_orig"] if d["b1_orig"] > 1e-10 else float("inf")
            b2_ratio = d["b2_synth"] / d["b2_orig"] if d["b2_orig"] > 1e-10 else float("inf")
            flag_b1 = " << WARNING" if b1_ratio > 2.0 else ""
            flag_b2 = " << WARNING" if b2_ratio > 2.0 else ""
            print(f"  Class {cls} b1: orig={d['b1_orig']:.6f}, synth={d['b1_synth']:.6f} "
                  f"(ratio={b1_ratio:.2f}x){flag_b1}")
            print(f"  Class {cls} b2: orig={d['b2_orig']:.6f}, synth={d['b2_synth']:.6f} "
                  f"(ratio={b2_ratio:.2f}x){flag_b2}")

        print(f"\nDistribution similarity (KS test, {n_bands} bands):")
        for cls, s in sorted(ks_summary.items()):
            status = "PASS" if s["fraction_similar"] >= 0.8 else "WARN"
            print(f"  Class {cls}: {s['bands_similar']}/{s['bands_total']} bands similar "
                  f"(p>0.05) = {s['fraction_similar']:.0%} [{status}]")

        print(f"\nSilhouette score (class separability):")
        print(f"  Original:  {sil['original']:.4f}")
        print(f"  Augmented: {sil['augmented']:.4f}")
        delta = sil["augmented"] - sil["original"]
        if delta < -0.1:
            print(f"  WARNING: Augmentation degraded class separation by {abs(delta):.4f}")
        elif delta > 0.05:
            print(f"  Augmentation improved class separation by {delta:.4f}")
        else:
            print(f"  Class separation preserved (delta={delta:+.4f})")

        if save_plots and self.output_dir:
            print(f"\nPlots saved to: {self.output_dir}/")
        elif not save_plots:
            print(f"\nSet output_dir to save diagnostic plots.")
        print("=" * 70)

        return report
