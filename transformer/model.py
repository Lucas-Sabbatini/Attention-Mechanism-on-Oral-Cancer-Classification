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
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from models.model import BaseClassifierModel
from transformer.train_engine import TrainEngine


class SpectralTransformerModel(TrainEngine, BaseClassifierModel):
    """
    Wrapper class for SpectralTransformer that follows the BaseClassifierModel interface.
    Provides training, prediction, and evaluation methods compatible with the project's
    cross-validation workflow.
    """
    
    def __init__(self, 
                 num_spectral_points: int = 1141,
                 d_model: int = 32,
                 nhead: int = 4,
                 num_layers: int = 1,
                 dim_feedforward: int = 64,
                 lr: float = 5e-3,
                 weight_decay: float = 5e-5,
                 n_epochs: int = 200,
                 batch_size: int = 8,
                 patience: int = 50,
                 random_state: int = 1,
                 verbose: bool = True,
                 log_interval: int = 5):
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
        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold, 
            test_size=0.3, 
            random_state=self.random_state,
            stratify=y_train_fold
        )
        
        # Train model with separate validation set
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
