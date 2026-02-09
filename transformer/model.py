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
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 4,
                 dim_feedforward: int = 128,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 n_epochs: int = 100,
                 batch_size: int = 16,
                 patience: int = 15,
                 random_state: int = 42):
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
        
        # Create DataLoader for batching
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state)
        )
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        remaining_patience = self.patience
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(self.model.state_dict())
                remaining_patience = self.patience
            else:
                remaining_patience -= 1
            
            if remaining_patience <= 0:
                break
        
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
        # Split training data into train/val for early stopping (80/20)
        val_size = max(int(len(X_train_fold) * 0.2), 1)
        train_size = len(X_train_fold) - val_size
        
        X_train = X_train_fold[:train_size]
        y_train = y_train_fold[:train_size]
        X_val = X_train_fold[train_size:]
        y_val = y_train_fold[train_size:]
        
        # Train model
        self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate on test fold
        y_pred = self.predict(X_test_fold)
        
        # Calculate metrics
        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred, zero_division=0)
        rec = recall_score(y_test_fold, y_pred, zero_division=0)  # Sensitivity
        esp = recall_score(y_test_fold, y_pred, pos_label=0, zero_division=0)  # Specificity
        mean_se_sp = np.mean([rec, esp])
        
        return (acc, prec, rec, esp, mean_se_sp)
