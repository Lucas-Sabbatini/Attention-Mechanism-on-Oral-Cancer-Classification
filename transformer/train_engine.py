import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from transformer.main import SpectralTransformer


class TrainEngine:
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

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray):
        """
        Train the transformer model with early stopping.
        
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
        
        # Create DataLoader for batching
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state)
        )
        
        # Loss function with class weighting (use BCEWithLogitsLoss for numerical stability)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing scheduler (works better for small datasets)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # Early stopping variables - track best validation accuracy instead of loss
        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_model_state = None
        remaining_patience = self.patience
        
        if self.verbose:
            print(f"Starting training on {self.device} | Epochs: {self.n_epochs} | Batch size: {self.batch_size}")
            print(f"Most important parameters: N-Layers{self.num_layers}, N-Heads{self.nhead}, d_model{self.d_model}")
            print(f"Train samples: {len(X_train_tensor)} | Val samples: {len(X_val_tensor)}")
            print(f"Class distribution - Positive: {int(num_pos)} | Negative: {int(num_neg)} | pos_weight: {pos_weight.item():.2f}")
            print("-" * 70)
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass (get logits for BCEWithLogitsLoss)
                outputs = self.model(X_batch, return_logits=True)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()  # Step scheduler after each batch for CosineAnnealingWarmRestarts
                train_losses.append(loss.item())
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor, return_logits=True)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                # Calculate validation accuracy for early stopping
                val_probs = torch.sigmoid(val_outputs)
                val_preds = (val_probs >= 0.5).float()
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            # Calculate average training loss
            avg_train_loss = np.mean(train_losses)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Early stopping check based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_model_state = deepcopy(self.model.state_dict())
                remaining_patience = self.patience
                improved = True
            else:
                remaining_patience -= 1
                improved = False
            
            # Logging
            if self.verbose and (epoch % self.log_interval == 0 or epoch == self.n_epochs - 1 or improved or remaining_patience <= 0):
                status = "✓ improved" if improved else f"patience: {remaining_patience}/{self.patience}"
                print(f"Epoch {epoch+1:3d}/{self.n_epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2%} | "
                      f"LR: {current_lr:.2e} | {status}")
            
            if remaining_patience <= 0:
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if self.verbose:
            print("-" * 70)
            print(f"Training complete | Best Val Acc: {best_val_acc:.2%} | Best Val Loss: {best_val_loss:.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)