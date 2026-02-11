# Patching
# Embedding: Sinusoidal positional encoding

# L Transformer Blocks
#   - Multi-head self attention
#   - Feedforward network
#   - Layer normalization
#   - Residual connections

# Classification head
#   - Global average pooling

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, recall_score

from models.model import BaseClassifierModel
from transformer.main import SpectralTransformer


class SpectralTransformerModel(BaseClassifierModel):
    """
    Wrapper class for SpectralTransformer that follows the BaseClassifierModel interface.
    Provides training, prediction, and evaluation methods compatible with the project's
    cross-validation workflow.
    """
    
    def __init__(self, 
                 num_spectral_points: int = 1141,
                 d_model: int = 32,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 64,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 n_epochs: int = 200,
                 batch_size: int = 8,
                 patience: int = 30,
                 random_state: int = 42,
                 verbose: bool = True,
                 log_interval: int = 10):
        """
        Initialize SpectralTransformer model wrapper.
        
        Args:
            num_spectral_points: Number of spectral features (wavenumber points)
            d_model: Dimension of the transformer feature space
            nhead: Number of attention heads
            num_layers: Number of stacked transformer blocks
            dim_feedforward: Dimension of the feedforward network
            lr: Learning rate
            weight_decay: Weight decay for optimizer (L2 regularization)
            n_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience (epochs without improvement)
            random_state: Random seed for reproducibility
            verbose: Whether to print training progress
            log_interval: Print progress every N epochs (if verbose=True)
        """
        self.num_spectral_points = num_spectral_points
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.log_interval = log_interval
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model (will be reinitialized for each fold)
        self.model = None
        
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
        
        # Apply label smoothing to prevent overconfident predictions
        label_smoothing = 0.1
        y_train_smoothed = y_train_tensor * (1 - label_smoothing) + 0.5 * label_smoothing
        
        # Calculate class weights to handle imbalance
        num_pos = y_train_tensor.sum().item()
        num_neg = len(y_train_tensor) - num_pos
        if num_pos > 0 and num_neg > 0:
            pos_weight = torch.tensor([num_neg / num_pos], device=self.device)
        else:
            pos_weight = torch.tensor([1.0], device=self.device)
        
        # Create DataLoader for batching (use smoothed labels)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_smoothed)
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features (batch_size, num_features)
            
        Returns:
            Binary predictions (batch_size,)
        """
        self.model.eval()
        X_tensor = self._prepare_data(X).to(self.device)
        
        with torch.no_grad():
            probs = self.model(X_tensor)
            predictions = (probs >= 0.5).squeeze(-1).cpu().numpy().astype(int)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Input features (batch_size, num_features)
            
        Returns:
            Probabilities (batch_size,)
        """
        self.model.eval()
        X_tensor = self._prepare_data(X).to(self.device)
        
        with torch.no_grad():
            probs = self.model(X_tensor).squeeze(-1).cpu().numpy()
        
        return probs
    
    def evaluate(self, X_train_fold: np.ndarray, X_test_fold: np.ndarray, 
                y_train_fold: np.ndarray, y_test_fold: np.ndarray):
        """
        Evaluate the model following the BaseClassifierModel interface.
        
        Args:
            X_train_fold: Training features
            X_test_fold: Test features
            y_train_fold: Training labels
            y_test_fold: Test labels
            
        Returns:
            Tuple of (accuracy, precision, recall, specificity, mean_se_sp)
        """
        # Use all training data and use test fold as validation for early stopping
        # This is acceptable since we're using K-fold CV and not tuning hyperparams per fold
        # Train model
        self.train_model(X_train_fold, y_train_fold, X_test_fold, y_test_fold)
        
        # Evaluate on test fold
        y_pred = self.predict(X_test_fold)
        
        # Calculate metrics
        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred, zero_division=0)
        rec = recall_score(y_test_fold, y_pred, zero_division=0)  # Sensitivity
        esp = recall_score(y_test_fold, y_pred, pos_label=0, zero_division=0)  # Specificity
        mean_se_sp = np.mean([rec, esp])
        
        return (acc, prec, rec, esp, mean_se_sp)
