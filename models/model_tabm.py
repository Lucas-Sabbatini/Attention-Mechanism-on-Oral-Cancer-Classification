from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np
from tabm import TabM
from copy import deepcopy
from typing import Optional
from torch import nn, Tensor



class TabMModel():
    def __init__(self, share_training_batches: bool = True, lr: float = 2e-3, 
                 weight_decay: float = 3e-4, n_epochs: int = 100, 
                 batch_size: int = 32, patience: int = 16, random_state: int = 0):
        """
        Initialize TabM model for binary classification
        
        Args:
            share_training_batches: Whether ensemble members share training batches
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            n_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            random_state: Random seed
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else None
        )
        self.amp_enabled = False and self.amp_dtype is not None
        self.grad_scaler = torch.cuda.amp.GradScaler() if self.amp_dtype is torch.float16 else None

        # Binary classification requires 2 output classes
        self.d_out = 2
        self.num_features = 1141
        self.share_training_batches = share_training_batches
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = TabM.make(
            n_num_features=self.num_features,
            d_out=self.d_out,
        ).to(self.device)

        self.gradient_clipping_norm: Optional[float] = 1.0
        self.lr = lr
        self.weight_decay = weight_decay
        self.base_loss_fn = nn.functional.cross_entropy

    def forward(self, X: np.array) -> np.array:
        """
        Forward pass through the model
        
        Args:
            X: Input data (batch_size, num_features)
            
        Returns:
            Binary predictions (batch_size,)
        """
        self.model.eval()
        with torch.no_grad():
            # Convert to tensor and move to device
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            
            # Get predictions: shape (batch_size, k, d_out)
            y_pred = self.model(X_tensor)
            batch_size = X.shape[0]
            assert y_pred.shape == (batch_size, self.model.k, self.d_out), \
                f"Expected shape ({batch_size}, {self.model.k}, {self.d_out}), got {y_pred.shape}"
            
            # Convert logits to probabilities
            y_pred_probs = torch.softmax(y_pred, dim=-1)  # (batch_size, k, d_out)
            
            # Average across ensemble members
            y_mean = y_pred_probs.mean(dim=1)  # (batch_size, d_out)
            
            # Get class with highest probability
            y_pred_class = y_mean.argmax(dim=-1)  # (batch_size,)
            
            return y_pred_class.cpu().numpy()
    
    def loss_fn(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # TabM produces k predictions. Each of them must be trained separately.

        # Classification: (batch_size, k, n_classes) -> (batch_size * k, n_classes)
        y_pred = y_pred.flatten(0, 1)

        if self.share_training_batches:
            # (batch_size,) -> (batch_size * k,)
            y_true = y_true.repeat_interleave(self.model.backbone.k)
        else:
            # (batch_size, k) -> (batch_size * k,)
            y_true = y_true.flatten(0, 1)

        return self.base_loss_fn(y_pred, y_true)


    def train_model(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array):

        # Reinitialize model weights for each fold to avoid data leakage
        self.model = TabM.make(
            n_num_features=self.num_features,
            d_out=self.d_out,
        ).to(self.device)
        
        # Reinitialize optimizer for each fold
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        train_size = len(X_train)
        
        # Convert data to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=self.device)
        
        best_val_loss = float('inf')
        best_model_state = None
        remaining_patience = self.patience
        
        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_losses = []
            
            # Create batches
            indices = torch.randperm(train_size, device=self.device)
            batches = (
                indices.split(self.batch_size)
                if self.share_training_batches
                else (
                    torch.rand((train_size, self.model.k), device=self.device)
                    .argsort(dim=0)
                    .split(self.batch_size, dim=0)
                )
            )
            
            for batch_idx in batches:
                self.optimizer.zero_grad()
                
                # Forward pass
                with torch.autocast(self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype):
                    y_pred = self.model(X_train_tensor[batch_idx])
                    loss = self.loss_fn(y_pred, y_train_tensor[batch_idx])
                
                # Backward pass
                if self.grad_scaler is None:
                    loss.backward()
                else:
                    self.grad_scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clipping_norm is not None:
                    if self.grad_scaler is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping_norm
                    )
                
                # Optimizer step
                if self.grad_scaler is None:
                    self.optimizer.step()
                else:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                
                epoch_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                with torch.autocast(self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype):
                    y_val_pred = self.model(X_val_tensor)
                    val_loss = self.loss_fn(y_val_pred, y_val_tensor).item()
            
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



    def tabm_model(self, X_train_fold: np.array, X_test_fold: np.array, y_train_fold: np.array, y_test_fold: np.array):
        # Split training data into train/val (80/20)
        val_size = max(int(len(X_train_fold) * 0.2), 1)
        train_size = len(X_train_fold) - val_size
        
        # Use last samples as validation (already shuffled by StratifiedKFold)
        X_train = X_train_fold[:train_size]
        y_train = y_train_fold[:train_size]
        X_val = X_train_fold[train_size:]
        y_val = y_train_fold[train_size:]
        
        # Train model
        self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model on test fold
        y_pred = self.forward(X_test_fold)

        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred, zero_division=0)
        rec = recall_score(y_test_fold, y_pred, zero_division=0)
        esp = recall_score(y_test_fold, y_pred, pos_label=0, zero_division=0)
        f1 = f1_score(y_test_fold, y_pred, zero_division=0)

        return (acc, prec, rec, esp, f1)