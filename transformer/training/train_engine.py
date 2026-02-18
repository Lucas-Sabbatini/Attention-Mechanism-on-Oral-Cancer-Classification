import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from transformer.training.sup_con_loss import SupConLoss
from transformer.training.center_loss import CenterLoss
from transformer.training.class_balanced_sampler import ClassBalancedSampler

from transformer.training.train_utils import TrainUtils

class TrainEngine(TrainUtils):
    """
    Mixin class that provides training functionality for SpectralTransformer models.
    
    Expected attributes from the inheriting class:
        - num_spectral_points: int
        - d_model: int
        - nhead: int
        - num_layers: int
        - dim_feedforward: int
        - lr: float
        - weight_decay: float
        - n_epochs: int
        - batch_size: int
        - patience: int
        - random_state: int
        - verbose: bool
        - log_interval: int
        - device: torch.device
        - model: SpectralTransformer (will be initialized)
    """
    
    

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray):
        """
        Train the transformer model with Supervised Contrastive Learning + BCE.
        
        Uses Joint Training (SupCon + BCE) which works better for tiny datasets
        compared to two-stage training that could overfit.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        # Initialize fresh model
        self._init_model()
        
        # Prepare data
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
        
        # Move to device
        X_train_tensor = X_train_tensor.to(self.device)
        y_train_tensor = y_train_tensor.to(self.device)
        X_val_tensor = X_val_tensor.to(self.device)
        y_val_tensor = y_val_tensor.to(self.device)
        
        # Calculate class weights to handle imbalance
        num_pos = y_train_tensor.sum().item()
        num_neg = len(y_train_tensor) - num_pos
        if num_pos > 0 and num_neg > 0:
            pos_weight = torch.tensor([num_neg / num_pos], device=self.device)
        else:
            pos_weight = torch.tensor([1.0], device=self.device)
        
        # Create DataLoader with class-balanced sampling for contrastive learning
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        
        # Use class-balanced sampler to ensure contrastive pairs exist
        sampler = ClassBalancedSampler(
            labels=y_train,
            batch_size=self.batch_size,
            min_per_class=2  # At least 2 samples per class per batch
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=False  # Drop incomplete batches
        )
        
        # Loss functions
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        supcon_criterion = SupConLoss(temperature=self.supcon_temperature)
       
        # Optimizer for model parameters
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Center Loss - only instantiate if weight > 0 (avoids shifting torch RNG state)
        if self.center_loss_weight > 0:
            feat_dim = self.d_model // 2
            center_criterion = CenterLoss(num_classes=2, feat_dim=feat_dim, device=self.device)
            center_separation_weight = getattr(self, 'center_separation_weight', 0.5)
            center_optimizer = torch.optim.SGD(
                center_criterion.parameters(),
                lr=self.lr * 10
            )
        else:
            center_criterion = None
            center_optimizer = None
        
        # Cosine annealing scheduler (works better for small datasets)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # Early stopping variables - track composite metric
        best_val_score = -float('inf')
        best_model_state = None
        remaining_patience = self.patience
        
        self._log_training_header(
            len(X_train_tensor), len(X_val_tensor),
            int(num_pos), int(num_neg), pos_weight.item()
        )
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            train_bce_losses = []
            train_supcon_losses = []
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                if center_optimizer is not None:
                    center_optimizer.zero_grad()

                # Forward pass with embeddings for contrastive loss
                logits, encoder_repr, projected = self.model(
                    X_batch, return_logits=True, return_embeddings=True
                )

                # BCE loss
                bce_loss = bce_criterion(logits, y_batch)

                labels_flat = y_batch.squeeze(-1)

                # Center loss (only if weight > 0)
                if center_criterion is not None:
                    center_loss = center_criterion(projected, labels_flat, center_separation_weight=center_separation_weight)
                    train_supcon_losses.append(center_loss.item())
                else:
                    center_loss = torch.tensor(0.0)

                supcon_loss = supcon_criterion(projected, labels_flat)

                # Combined loss
                loss = self.bce_weight * bce_loss + self.center_loss_weight * center_loss + self.supcon_weight * supcon_loss

                train_bce_losses.append(bce_loss.item())

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                if center_optimizer is not None:
                    center_optimizer.step()
                scheduler.step()
            
            # Validation phase
            val_logits, val_bce_loss, val_acc, val_repr, val_proj = self._compute_validation_metrics(
                X_val_tensor, y_val_tensor, bce_criterion
            )
            
            # Compute cluster quality metrics
            silhouette, knn_acc = self._compute_cluster_quality_metrics(
                val_proj, y_val_tensor, X_train_tensor, y_train_tensor, y_val
            )
            
            # Calculate average training losses
            avg_bce = np.mean(train_bce_losses)
            avg_supcon = np.mean(train_supcon_losses) if train_supcon_losses else 0
            current_lr = optimizer.param_groups[0]['lr']
            
            # Composite validation score for early stopping
            val_score = self._compute_validation_score(val_acc, silhouette, knn_acc)
            
            # Early stopping check
            if val_score >= best_val_score:
                best_val_score = val_score
                best_model_state = deepcopy(self.model.state_dict())
                remaining_patience = self.patience
                improved = True
            else:
                remaining_patience -= 1
                improved = False
            
            # Logging
            self._log_epoch(
                epoch, avg_bce, avg_supcon, val_acc, silhouette, knn_acc,
                current_lr, improved, remaining_patience
            )
            
            if remaining_patience <= 0:
                self._log_early_stopping(epoch)
                break
        
        self._log_training_complete(best_val_score)
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)