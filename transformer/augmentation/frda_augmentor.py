import numpy as np


class FRDAAugmentor:
    """
    Generates new spectra by randomly recombining the 3 segments from different
    samples of the same class using FRDA (Fingerprint Region based Data Augmentation).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_boundaries: dict[int, tuple[int, int]],
        n_augmented_per_class: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate augmented spectra and stack with originals.

        Args:
            X: Original spectra of shape (N, num_spectral_points).
            y: Labels of shape (N,).
            class_boundaries: {class_id: (b1, b2)} boundary indices.
            n_augmented_per_class: Number of new samples to generate per class.

        Returns:
            (X_augmented, y_augmented) with original + new samples stacked.
        """
        rng = np.random.RandomState(self.random_state)
        new_X_parts = []
        new_y_parts = []

        for cls, (b1, b2) in class_boundaries.items():
            cls_mask = y == cls
            X_cls = X[cls_mask]
            n_cls = len(X_cls)
            if n_cls < 2:
                continue

            new_spectra = np.empty((n_augmented_per_class, X.shape[1]), dtype=X.dtype)
            donors_A = rng.randint(0, n_cls, size=n_augmented_per_class)
            donors_B = rng.randint(0, n_cls, size=n_augmented_per_class)
            donors_C = rng.randint(0, n_cls, size=n_augmented_per_class)

            new_spectra[:, :b1] = X_cls[donors_A, :b1]
            new_spectra[:, b1:b2] = X_cls[donors_B, b1:b2]
            new_spectra[:, b2:] = X_cls[donors_C, b2:]

            new_X_parts.append(new_spectra)
            new_y_parts.append(np.full(n_augmented_per_class, cls, dtype=y.dtype))

        if new_X_parts:
            X_aug = np.concatenate([X] + new_X_parts, axis=0)
            y_aug = np.concatenate([y] + new_y_parts, axis=0)
        else:
            X_aug, y_aug = X.copy(), y.copy()

        return X_aug, y_aug

    def augment_to_balance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_boundaries: dict[int, tuple[int, int]],
        target_per_class: int = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate exactly enough samples per class to reach target_per_class.

        Args:
            X: Original spectra of shape (N, num_spectral_points).
            y: Labels of shape (N,).
            class_boundaries: {class_id: (b1, b2)} boundary indices.
            target_per_class: Target count per class. Defaults to max class count.

        Returns:
            (X_augmented, y_augmented) with balanced class counts.
        """
        classes, counts = np.unique(y, return_counts=True)
        if target_per_class is None:
            target_per_class = int(counts.max())

        rng = np.random.RandomState(self.random_state)
        new_X_parts = []
        new_y_parts = []

        for cls in classes:
            cls_mask = y == cls
            X_cls = X[cls_mask]
            n_cls = len(X_cls)

            n_needed = target_per_class - n_cls
            if n_needed <= 0 or n_cls < 2:
                continue

            b1, b2 = class_boundaries.get(int(cls), (X.shape[1] // 3, 2 * X.shape[1] // 3))

            new_spectra = np.empty((n_needed, X.shape[1]), dtype=X.dtype)
            donors_A = rng.randint(0, n_cls, size=n_needed)
            donors_B = rng.randint(0, n_cls, size=n_needed)
            donors_C = rng.randint(0, n_cls, size=n_needed)

            new_spectra[:, :b1] = X_cls[donors_A, :b1]
            new_spectra[:, b1:b2] = X_cls[donors_B, b1:b2]
            new_spectra[:, b2:] = X_cls[donors_C, b2:]

            new_X_parts.append(new_spectra)
            new_y_parts.append(np.full(n_needed, cls, dtype=y.dtype))

        if new_X_parts:
            X_aug = np.concatenate([X] + new_X_parts, axis=0)
            y_aug = np.concatenate([y] + new_y_parts, axis=0)
        else:
            X_aug, y_aug = X.copy(), y.copy()

        return X_aug, y_aug
