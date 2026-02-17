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
from scipy.stats import ks_2samp

from models.model import BaseClassifierModel
from transformer.training.train_engine import TrainEngine

class SpectralTransformerModel(TrainEngine, BaseClassifierModel):
    """
    Wrapper class for SpectralTransformer that follows the BaseClassifierModel interface.
    Provides training, prediction, and evaluation methods compatible with the project's
    cross-validation workflow.
    """
    
    # Class-level fold counter for tracking
    _fold_counter = 0
    _fold_diagnostics = []
    
    def __init__(self, 
                 num_spectral_points: int = 1141,
                 d_model: int = 32,
                 nhead: int = 4,
                 num_layers: int = 1,
                 dim_feedforward: int = 64,
                 dropout: float = 0.3,
                 patch_size: int = 16,
                 lr: float = 5e-3,
                 weight_decay: float = 5e-5,
                 n_epochs: int = 200,
                 batch_size: int = 8,
                 patience: int = 50,
                 supcon_weight: float = 0.5,
                 supcon_temperature: float = 0.07,
                 random_state: int = 1,
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
            dropout: Dropout rate for regularization
            patch_size: Size of spectral patches for conv embedding
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
        self.dropout = dropout
        self.patch_size = patch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.log_interval = log_interval
        self.supcon_weight = supcon_weight
        self.supcon_temperature = supcon_temperature
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model (will be reinitialized for each fold)
        self.model = None
        self.optimal_threshold = 0.5  # Will be calibrated during training
    
    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features (batch_size, num_features)
            threshold: Classification threshold (uses calibrated threshold if None)
            
        Returns:
            Binary predictions (batch_size,)
        """
        if threshold is None:
            threshold = self.optimal_threshold
            
        self.model.eval()
        X_tensor = self._prepare_data(X).to(self.device)
        
        with torch.no_grad():
            probs = self.model(X_tensor)
            predictions = (probs >= threshold).squeeze(-1).cpu().numpy().astype(int)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw probabilities for threshold analysis."""
        self.model.eval()
        X_tensor = self._prepare_data(X).to(self.device)
        
        with torch.no_grad():
            probs = self.model(X_tensor).squeeze(-1).cpu().numpy()
        
        return probs
    
    def get_embeddings(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract encoder and projection head embeddings for visualization.
        
        Args:
            X: Input features (batch_size, num_features)
            
        Returns:
            Tuple of (encoder_embeddings, projected_embeddings)
            - encoder_embeddings: Output from global average pooling (batch_size, d_model)
            - projected_embeddings: Output from projection head (batch_size, d_model//2)
        """
        self.model.eval()
        X_tensor = self._prepare_data(X).to(self.device)
        
        with torch.no_grad():
            _, encoder_repr, projected = self.model(X_tensor, return_embeddings=True)
            encoder_embeddings = encoder_repr.cpu().numpy()
            projected_embeddings = projected.cpu().numpy()
        
        return encoder_embeddings, projected_embeddings
    
    def calibrate_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Find optimal threshold that maximizes balanced accuracy on validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Optimal threshold value
        """
        probs = self.predict_proba(X_val)
        
        best_threshold = 0.5
        best_score = 0.0
        
        # Test thresholds from 0.1 to 0.9
        for threshold in np.arange(0.1, 0.91, 0.05):
            preds = (probs >= threshold).astype(int)
            
            # Calculate balanced accuracy (mean of sensitivity and specificity)
            sensitivity = recall_score(y_val, preds, zero_division=0)
            specificity = recall_score(y_val, preds, pos_label=0, zero_division=0)
            balanced_acc = (sensitivity + specificity) / 2
            
            if balanced_acc > best_score:
                best_score = balanced_acc
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        if self.verbose:
            print(f"Calibrated threshold: {best_threshold:.2f} (balanced acc: {best_score:.1%})")
        
        return best_threshold
    
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
        SpectralTransformerModel._fold_counter += 1
        fold_num = SpectralTransformerModel._fold_counter
        
        # Fold diagnostics
        diagnostics = self._compute_fold_diagnostics(
            X_train_fold, X_test_fold, y_train_fold, y_test_fold, fold_num
        )
        
        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold, 
            test_size=0.3, 
            random_state=self.random_state,
            stratify=y_train_fold
        )
        
        # Train model with separate validation set
        self.train_model(X_train, y_train, X_val, y_val)
        
        # Calibrate threshold on validation set
        self.calibrate_threshold(X_val, y_val)
        
        # Evaluate on test fold using calibrated threshold
        y_pred = self.predict(X_test_fold)
        
        # Calculate metrics
        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred, zero_division=0)
        rec = recall_score(y_test_fold, y_pred, zero_division=0)  # Sensitivity
        esp = recall_score(y_test_fold, y_pred, pos_label=0, zero_division=0)  # Specificity
        mean_se_sp = np.mean([rec, esp])
        
        # Store diagnostics with results
        diagnostics['accuracy'] = acc
        diagnostics['sensitivity'] = rec
        diagnostics['specificity'] = esp
        diagnostics['threshold'] = self.optimal_threshold
        diagnostics['predictions'] = y_pred.tolist()
        diagnostics['true_labels'] = y_test_fold.tolist()
        SpectralTransformerModel._fold_diagnostics.append(diagnostics)
        
        if self.verbose:
            self._print_fold_diagnostics(diagnostics)
        
        return (acc, prec, rec, esp, mean_se_sp)
    
    def _compute_fold_diagnostics(self, X_train, X_test, y_train, y_test, fold_num):
        """Compute diagnostic statistics for this fold to identify issues."""
        diagnostics = {'fold': fold_num}
        
        # 1. Class distribution
        train_pos = np.sum(y_train)
        train_neg = len(y_train) - train_pos
        test_pos = np.sum(y_test)
        test_neg = len(y_test) - test_pos
        
        diagnostics['train_class_ratio'] = train_pos / len(y_train) if len(y_train) > 0 else 0
        diagnostics['test_class_ratio'] = test_pos / len(y_test) if len(y_test) > 0 else 0
        diagnostics['train_samples'] = len(y_train)
        diagnostics['test_samples'] = len(y_test)
        diagnostics['train_pos'] = int(train_pos)
        diagnostics['test_pos'] = int(test_pos)
        
        # 2. Distribution shift detection (KS test on mean spectra per sample)
        train_means = X_train.mean(axis=1)
        test_means = X_test.mean(axis=1)
        ks_stat, ks_pval = ks_2samp(train_means, test_means)
        diagnostics['ks_statistic'] = ks_stat
        diagnostics['ks_pvalue'] = ks_pval
        diagnostics['distribution_shift'] = ks_pval < 0.05  # Significant shift
        
        # 3. Feature variance comparison
        train_var = X_train.var(axis=0).mean()
        test_var = X_test.var(axis=0).mean()
        diagnostics['train_variance'] = train_var
        diagnostics['test_variance'] = test_var
        diagnostics['variance_ratio'] = test_var / train_var if train_var > 0 else float('inf')
        
        # 4. Per-class separation in test set (simple metric)
        if test_pos > 0 and test_neg > 0:
            pos_mean = X_test[y_test == 1].mean(axis=1).mean()
            neg_mean = X_test[y_test == 0].mean(axis=1).mean()
            pos_std = X_test[y_test == 1].mean(axis=1).std()
            neg_std = X_test[y_test == 0].mean(axis=1).std()
            pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
            diagnostics['class_separation'] = abs(pos_mean - neg_mean) / pooled_std if pooled_std > 0 else 0
        else:
            diagnostics['class_separation'] = 0
            
        return diagnostics
    
    def _print_fold_diagnostics(self, d):
        """Print diagnostic summary for a fold."""
        shift_warning = "⚠️ SHIFT" if d['distribution_shift'] else "✓"
        print(f"\n{'='*60}")
        print(f"FOLD {d['fold']} DIAGNOSTICS")
        print(f"{'='*60}")
        print(f"Train: {d['train_samples']} samples ({d['train_pos']} pos, {d['train_samples']-d['train_pos']} neg)")
        print(f"Test:  {d['test_samples']} samples ({d['test_pos']} pos, {d['test_samples']-d['test_pos']} neg)")
        print(f"Class ratio - Train: {d['train_class_ratio']:.2%}, Test: {d['test_class_ratio']:.2%}")
        print(f"Distribution shift (KS): {d['ks_statistic']:.3f} (p={d['ks_pvalue']:.3f}) {shift_warning}")
        print(f"Variance ratio (test/train): {d['variance_ratio']:.2f}")
        print(f"Class separation (Cohen's d): {d['class_separation']:.2f}")
        print(f"Calibrated threshold: {d['threshold']:.2f}")
        print(f"RESULT: Acc={d['accuracy']:.1%}, Sens={d['sensitivity']:.1%}, Spec={d['specificity']:.1%}")
        print(f"Predictions: {d['predictions']}")
        print(f"True labels: {d['true_labels']}")
        print(f"{'='*60}\n")
    
    @classmethod
    def get_fold_diagnostics(cls):
        """Return all fold diagnostics for analysis."""
        return cls._fold_diagnostics
    
    @classmethod
    def reset_fold_counter(cls):
        """Reset fold counter and diagnostics (call before new CV run)."""
        cls._fold_counter = 0
        cls._fold_diagnostics = []
    
    @classmethod
    def print_fold_summary(cls):
        """Print summary comparing all folds to identify problematic ones."""
        if not cls._fold_diagnostics:
            print("No fold diagnostics available.")
            return
            
        print("\n" + "="*90)
        print("FOLD COMPARISON SUMMARY")
        print("="*90)
        print(f"{'Fold':<6} {'Acc':<8} {'Sens':<8} {'Spec':<8} {'Thresh':<8} {'KS-p':<8} {'Shift':<8} {'Sep':<8}")
        print("-"*90)
        
        for d in cls._fold_diagnostics:
            shift = "⚠️" if d['distribution_shift'] else "✓"
            print(f"{d['fold']:<6} {d['accuracy']:<8.1%} {d['sensitivity']:<8.1%} "
                  f"{d['specificity']:<8.1%} {d['threshold']:<8.2f} {d['ks_pvalue']:<8.3f} {shift:<8} {d['class_separation']:<8.2f}")
        
        # Identify problematic folds
        accs = [d['accuracy'] for d in cls._fold_diagnostics]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        
        print("-"*90)
        print(f"Mean Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
        
        low_folds = [d['fold'] for d in cls._fold_diagnostics if d['accuracy'] < mean_acc - std_acc]
        if low_folds:
            print(f"⚠️  LOW PERFORMING FOLDS: {low_folds}")
            for fold in low_folds:
                d = cls._fold_diagnostics[fold-1]
                issues = []
                if d['distribution_shift']:
                    issues.append("distribution shift")
                if d['class_separation'] < 0.5:
                    issues.append("poor class separation")
                if d['variance_ratio'] > 1.5 or d['variance_ratio'] < 0.67:
                    issues.append(f"variance mismatch ({d['variance_ratio']:.2f}x)")
                print(f"   Fold {fold} issues: {', '.join(issues) if issues else 'unknown'}")
        print("="*90)
