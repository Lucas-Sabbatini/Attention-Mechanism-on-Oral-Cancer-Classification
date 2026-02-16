import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple

from transformer.architecture.main import SpectralTransformer

class TrainUtils:
    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _init_model(self):
        """Initialize a fresh model instance"""
        self._set_seeds(self.random_state)
        self.model = SpectralTransformer(
            num_spectral_points=self.num_spectral_points,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            patch_size=self.patch_size,
            num_classes=1
        ).to(self.device)
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None):
        """
        Convert numpy arrays to PyTorch tensors with correct shape.
        
        Args:
            X: Input features (batch_size, num_features)
            y: Labels (batch_size,), optional
            
        Returns:
            X_tensor: Shape (batch_size, num_features, 1)
            y_tensor: Shape (batch_size, 1) if y provided, else None
        """
        # Convert to tensor and add feature dimension
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, seq, 1)
        
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
            return X_tensor, y_tensor
        return X_tensor

    def _compute_validation_metrics(
        self,
        X_val_tensor: torch.Tensor,
        y_val_tensor: torch.Tensor,
        bce_criterion: nn.Module
    ) -> Tuple[torch.Tensor, float, float, torch.Tensor, torch.Tensor]:
        
        self.model.eval()
        with torch.no_grad():
            val_logits, val_repr, val_proj = self.model(
                X_val_tensor, return_logits=True, return_embeddings=True
            )
            
            val_bce_loss = bce_criterion(val_logits, y_val_tensor).item()
            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs >= 0.5).float()
            val_acc = (val_preds == y_val_tensor).float().mean().item()
        
        return val_logits, val_bce_loss, val_acc, val_repr, val_proj

    def _compute_cluster_quality_metrics(
        self,
        val_proj: torch.Tensor,
        y_val_tensor: torch.Tensor,
        X_train_tensor: torch.Tensor,
        y_train_tensor: torch.Tensor,
        y_val: np.ndarray
    ) -> Tuple[float, float]:

        silhouette = -1.0
        knn_acc = -1.0
        
        if len(np.unique(y_val)) <= 1:
            return silhouette, knn_acc
        
        val_embeddings = val_proj.cpu().numpy()
        val_labels = y_val_tensor.squeeze(-1).cpu().numpy()
        
        # Silhouette score (-1 to 1, higher is better)
        try:
            silhouette = silhouette_score(val_embeddings, val_labels)
        except:
            silhouette = -1.0
        
        # KNN accuracy in latent space (use train embeddings)
        try:
            with torch.no_grad():
                _, _, train_proj = self.model(
                    X_train_tensor, return_logits=True, return_embeddings=True
                )
            train_embeddings = train_proj.cpu().numpy()
            train_labels = y_train_tensor.squeeze(-1).cpu().numpy()
            
            knn = KNeighborsClassifier(n_neighbors=min(3, len(train_labels)-1))
            knn.fit(train_embeddings, train_labels)
            knn_preds = knn.predict(val_embeddings)
            knn_acc = (knn_preds == val_labels).mean()
        except:
            knn_acc = -1.0
        
        return silhouette, knn_acc

    def _compute_validation_score(
        self,
        val_acc: float,
        silhouette: float,
        knn_acc: float
    ) -> float:
        if silhouette > -1:
            return 0.4 * val_acc + 0.5 * ((silhouette + 1) / 2) + 0 * knn_acc
        return val_acc

    def _log_training_header(
        self,
        num_train: int,
        num_val: int,
        num_pos: int,
        num_neg: int,
        pos_weight: float
    ):
        """Log training configuration at the start of training."""
        if not self.verbose:
            return
        print(f"Starting SupCon+BCE training on {self.device}")
        print(f"Epochs: {self.n_epochs} | Batch size: {self.batch_size}")
        print(f"Train samples: {num_train} | Val samples: {num_val}")
        print(f"Class distribution - Pos: {num_pos} | Neg: {num_neg} | pos_weight: {pos_weight:.2f}")
        print("-" * 70)

    def _log_epoch(
        self,
        epoch: int,
        avg_bce: float,
        avg_supcon: float,
        val_acc: float,
        silhouette: float,
        knn_acc: float,
        current_lr: float,
        improved: bool,
        remaining_patience: int
    ):
        """Log metrics for a single epoch."""
        if not self.verbose:
            return
        if not (epoch % self.log_interval == 0 or epoch == self.n_epochs - 1 or improved or remaining_patience <= 0):
            return
        
        status = "✓" if improved else f"p:{remaining_patience}"
        print(f"E{epoch+1:3d} | BCE:{avg_bce:.3f} SupCon:{avg_supcon:.3f} | "
              f"vAcc:{val_acc:.1%} Sil:{silhouette:.2f} KNN:{knn_acc:.1%} | "
              f"LR:{current_lr:.1e} | {status}")

    def _log_early_stopping(self, epoch: int):
        """Log early stopping message."""
        if self.verbose:
            print(f"Early stopping at epoch {epoch+1}")

    def _log_training_complete(self, best_val_score: float):
        """Log training completion message."""
        if self.verbose:
            print("-" * 70)
            print(f"Training complete | Best composite score: {best_val_score:.4f}")