import torch
import torch.nn as nn
import math

from transformer.posEncoding import PositionalEncoding
from transformer.transformer_block import CustomTransformerBlock

class SpectralTransformer(nn.Module):
    def __init__(self, 
                 num_spectral_points,       # "d" (~1k features)
                 d_model=64,                # Dimension of the feature space
                 nhead=4,                   # "h" attention heads
                 num_layers=4,              # "L" stacked blocks
                 dim_feedforward=128,       # "d_ff" intermediate dimension
                 num_classes=1,             # Binary classification (Cancer vs Healthy)
                 dropout=0.3,               # Dropout rate for regularization
                 patch_size=16):            # Size of spectral patches for conv embedding
        super().__init__()

        self.num_spectral_points = num_spectral_points
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.patch_size = patch_size

        # Calculate stride for overlapping patches (50% overlap by default)
        self.patch_stride = patch_size // 2
        
        # Calculate number of patches with overlap: (L - K) / S + 1
        self.num_patches = (num_spectral_points - patch_size) // self.patch_stride + 1

        # 1. CONVOLUTIONAL PATCH EMBEDDING
        # Projects a patch of spectral intensities to feature space (d_model)
        # Conv1d: (batch, 1, num_spectral_points) -> (batch, d_model, num_patches)
        self.patch_embedding = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=self.patch_stride  # Overlapping patches (50% overlap)
        )
        self.embed_dropout = nn.Dropout(dropout)

        # 2. POSITIONAL ENCODING
        # Now based on num_patches since each patch becomes a token
        self.positional_encoding = PositionalEncoding(self.d_model, self.num_patches)

        # 3. STACKED TRANSFORMER BLOCKS ("L blocks")
        # Contains: Self-Attention, Feed-forward, LayerNorm, Residuals
        # Using our custom implementation with the Attention class
        self.transformer_blocks = nn.ModuleList([
            CustomTransformerBlock(d_model, nhead, dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 4. CLASSIFICATION HEAD
        # Linear layer for final logits
        self.pre_classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, return_logits=False):
        
        # A. Reshape for Conv1d: (Batch_Size, 1, num_spectral_points)
        if x.dim() == 3:
            x = x.squeeze(-1)  # Remove last dim if present
        x = x.unsqueeze(1)  # Add channel dimension
        
        # B. Apply Convolutional Patch Embedding
        # Output: (Batch_Size, d_model, num_patches)
        x = self.patch_embedding(x)
        
        # C. Transpose for transformer: (Batch_Size, num_patches, d_model)
        x = x.transpose(1, 2)
        x = x * math.sqrt(self.d_model)
        x = self.embed_dropout(x)
        
        # D. Add Positional Encoding
        x = self.positional_encoding.forward(x)

        # E. Pass through L Transformer Blocks
        # Output shape: (Batch_Size, num_patches, d_model)
        for block in self.transformer_blocks:
            x = block(x)

        # F. Global Average Pooling
        # Average across the patch dimension (dim=1)
        # Output shape: (Batch_Size, d_model)
        x = x.mean(dim=1)

        # G. Classification
        x = self.pre_classifier_dropout(x)
        logits = self.classifier(x)
        
        if return_logits:
            return logits
        
        return torch.sigmoid(logits)