import numpy as np
from sklearn.mixture import GaussianMixture


class GMMSegmenter:
    """
    Clusters 1D-CAM activation curves using Gaussian Mixture Models to find
    two spectral boundary indices per class, splitting each spectrum into
    3 segments.
    """

    def __init__(self, n_components: int = 3, n_init: int = 10, random_state: int = 42):
        self.n_components = n_components
        self.n_init = n_init
        self.random_state = random_state

    def fit_class(self, cam_curves: np.ndarray) -> tuple[int, int]:
        """
        Find two boundary indices for a single class from its CAM curves.

        Args:
            cam_curves: CAM activations of shape (N_class, num_spectral_points).

        Returns:
            (boundary_1, boundary_2) with boundary_1 < boundary_2.
        """
        n_samples, n_points = cam_curves.shape
        rng = np.random.RandomState(self.random_state)
        n_draw = max(100, n_points)

        all_positions = []
        for i in range(n_samples):
            curve = cam_curves[i].copy()
            # Normalize to a probability distribution
            curve = np.clip(curve, 0, None)
            total = curve.sum()
            if total < 1e-12:
                curve = np.ones(n_points) / n_points
            else:
                curve = curve / total
            # Sample wavenumber positions weighted by CAM activation
            sampled = rng.choice(n_points, size=n_draw, replace=True, p=curve)
            all_positions.append(sampled)

        all_positions = np.concatenate(all_positions).reshape(-1, 1)

        gmm = GaussianMixture(
            n_components=self.n_components,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        gmm.fit(all_positions)

        # Assign each wavenumber index to a cluster
        indices = np.arange(n_points).reshape(-1, 1)
        cluster_labels = gmm.predict(indices)

        # Find transition points where cluster label changes
        transitions = []
        for j in range(1, n_points):
            if cluster_labels[j] != cluster_labels[j - 1]:
                transitions.append(j)

        # Derive b1 and b2 from transitions
        if len(transitions) < 2:
            return n_points // 3, 2 * n_points // 3

        mid = len(transitions) // 2
        b1 = int(np.mean(transitions[:mid]))
        b2 = int(np.mean(transitions[mid:]))

        if b1 >= b2:
            return n_points // 3, 2 * n_points // 3

        return b1, b2

    def fit_all_classes(
        self,
        cam_curves: np.ndarray,
        labels: np.ndarray,
    ) -> dict[int, tuple[int, int]]:
        """
        Find per-class spectral boundaries.

        Args:
            cam_curves: CAM activations of shape (N, num_spectral_points).
            labels: Class labels of shape (N,).

        Returns:
            Dictionary mapping class_id -> (b1, b2).
        """
        boundaries = {}
        for cls in np.unique(labels):
            mask = labels == cls
            boundaries[int(cls)] = self.fit_class(cam_curves[mask])
        return boundaries
