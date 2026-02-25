import numpy as np
import torch
import torch.nn as nn

from transformer.architecture.main import SpectralTransformer
from transformer.augmentation.cam_extractor import CAMExtractor
from transformer.augmentation.gmm_segmenter import GMMSegmenter
from transformer.augmentation.frda_augmentor import FRDAAugmentor


class FRDAPipeline:
    """
    Orchestrates the full two-phase FRDA pipeline.

    Phase 1: Train a lightweight SpectralTransformer to extract CAM curves,
    segment the spectrum with GMM, and augment training data via region-wise
    recombination.

    Phase 2 (external): The caller uses the augmented data to train the
    actual model.
    """

    def __init__(
        self,
        device: torch.device,
        num_spectral_points: int,
        d_model: int = 32,
        n_gmm_components: int = 3,
        n_augmented_per_class: int = 200,
        balance_classes: bool = True,
        cam_epochs: int = 50,
        random_state: int = 42,
        verbose: bool = True,
        quality_report_dir: str = None,
        wavenumbers: np.ndarray = None,
    ):
        self.device = device
        self.num_spectral_points = num_spectral_points
        self.d_model = d_model
        self.n_gmm_components = n_gmm_components
        self.n_augmented_per_class = n_augmented_per_class
        self.balance_classes = balance_classes
        self.cam_epochs = cam_epochs
        self.random_state = random_state
        self.verbose = verbose
        self.quality_report_dir = quality_report_dir
        self.wavenumbers = wavenumbers

        self.class_boundaries_ = None

    def fit_transform(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the full FRDA pipeline on training data only.

        Args:
            X_train: Training spectra of shape (N, num_spectral_points).
            y_train: Training labels of shape (N,).

        Returns:
            (X_aug, y_aug) — augmented training data.
        """
        if self.verbose:
            print("[FRDA] Phase 1: Training lightweight CAM model...")

        # Step 1: Train lightweight model for CAM extraction
        cam_model = self._train_cam_model(X_train, y_train)

        # Step 2: Extract CAM curves
        if self.verbose:
            print("[FRDA] Extracting CAM activation curves...")
        extractor = CAMExtractor(cam_model, self.device)
        cam_curves = extractor.extract(X_train, y_train, self.num_spectral_points)

        # Step 3: Segment with GMM
        if self.verbose:
            print("[FRDA] Fitting GMM segmenter...")
        segmenter = GMMSegmenter(
            n_components=self.n_gmm_components,
            random_state=self.random_state,
        )
        self.class_boundaries_ = segmenter.fit_all_classes(cam_curves, y_train)
        if self.verbose:
            for cls, (b1, b2) in self.class_boundaries_.items():
                print(f"  Class {cls}: boundaries at [{b1}, {b2}] / {self.num_spectral_points}")

        # Step 4: Augment
        if self.verbose:
            print("[FRDA] Augmenting training data...")
        augmentor = FRDAAugmentor(random_state=self.random_state)

        if self.balance_classes:
            X_aug, y_aug = augmentor.augment_to_balance(
                X_train, y_train, self.class_boundaries_
            )
        else:
            X_aug, y_aug = augmentor.augment(
                X_train, y_train, self.class_boundaries_,
                n_augmented_per_class=self.n_augmented_per_class,
            )

        if self.verbose:
            classes, counts = np.unique(y_aug, return_counts=True)
            dist = ", ".join(f"class {c}: {n}" for c, n in zip(classes, counts))
            print(f"[FRDA] Augmented: {len(X_train)} -> {len(X_aug)} samples ({dist})")

        # Quality report (optional)
        if self.quality_report_dir is not None or self.verbose:
            from transformer.augmentation.quality_report import FRDAQualityReport
            qr = FRDAQualityReport(
                X_original=X_train,
                y_original=y_train,
                X_augmented=X_aug,
                y_augmented=y_aug,
                class_boundaries=self.class_boundaries_,
                wavenumbers=self.wavenumbers,
                output_dir=self.quality_report_dir,
            )
            # Save plots only when a directory is provided
            self.quality_report_ = qr.generate_report(
                save_plots=self.quality_report_dir is not None,
            )

        # CAM model is discarded (not stored on self)
        return X_aug, y_aug

    def _train_cam_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> SpectralTransformer:
        """
        Train a lightweight SpectralTransformer for CAM extraction only.

        This model is intentionally small (1 layer, 2 heads) and is discarded
        after CAM curves are extracted.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        model = SpectralTransformer(
            num_spectral_points=self.num_spectral_points,
            d_model=self.d_model,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.3,
            num_classes=1,
            attention_mask_bias= None
        ).to(self.device)

        # Prepare data
        X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(self.device)

        # Class-weighted BCE loss
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        if num_pos > 0 and num_neg > 0:
            pos_weight = torch.tensor([num_neg / num_pos], device=self.device)
        else:
            pos_weight = torch.tensor([1.0], device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

        import copy

        best_loss = float("inf")
        best_state = None

        model.train()
        for epoch in range(self.cam_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = model(X_batch, return_logits=True)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(model.state_dict())

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"  [CAM model] Epoch {epoch+1}/{self.cam_epochs} - Loss: {avg_loss:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
            if self.verbose:
                print(f"  [CAM model] Restored best epoch (loss: {best_loss:.4f})")

        return model
