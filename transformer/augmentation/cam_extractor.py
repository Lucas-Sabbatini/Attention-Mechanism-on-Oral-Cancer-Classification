import numpy as np
import torch
import torch.nn.functional as F

from transformer.architecture.main import SpectralTransformer


class CAMExtractor:
    """
    Extracts 1D Class Activation Map curves from a trained SpectralTransformer.

    Registers a forward hook on the last transformer block to capture patch
    activations, then computes weighted sums using the classifier weights
    and interpolates back to the original spectral resolution.
    """

    def __init__(self, model: SpectralTransformer, device: torch.device):
        self.model = model
        self.device = device

    def extract(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        num_spectral_points: int,
    ) -> np.ndarray:
        """
        Extract CAM curves for all samples.

        Args:
            X: Input spectra of shape (N, num_spectral_points).
            labels: Class labels of shape (N,).
            num_spectral_points: Original spectral resolution for interpolation.

        Returns:
            Normalized CAM curves of shape (N, num_spectral_points) in [0, 1].
        """
        self.model.eval()
        activations_store: list[torch.Tensor] = []

        # Register hook on the last transformer block
        hook = self.model.transformer_blocks[-1].register_forward_hook(
            lambda module, inp, out: activations_store.append(out)
        )

        try:
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)

            with torch.no_grad():
                _ = self.model(X_tensor)

            # activations_store[0] shape: (N, num_patches, d_model)
            patch_activations = activations_store[0]

            # Classifier weight: (1, d_model) for binary classification
            classifier_weight = self.model.classifier.weight.squeeze(0)  # (d_model,)

            # Weighted sum: (N, num_patches)
            cam_patches = torch.matmul(patch_activations, classifier_weight)

            # Interpolate from patch space to spectral resolution
            # F.interpolate expects (N, C, L) -> add channel dim
            cam_patches = cam_patches.unsqueeze(1)  # (N, 1, num_patches)
            cam_interp = F.interpolate(
                cam_patches, size=num_spectral_points, mode="linear", align_corners=False
            )
            cam_interp = cam_interp.squeeze(1)  # (N, num_spectral_points)

            # Normalize each curve to [0, 1]
            cam_np = cam_interp.detach().cpu().numpy()
            for i in range(cam_np.shape[0]):
                curve = cam_np[i]
                cmin, cmax = curve.min(), curve.max()
                if cmax - cmin > 1e-8:
                    cam_np[i] = (curve - cmin) / (cmax - cmin)
                else:
                    cam_np[i] = 0.0

            return cam_np

        finally:
            hook.remove()
