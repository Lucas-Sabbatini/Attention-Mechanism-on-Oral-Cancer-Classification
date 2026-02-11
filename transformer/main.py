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
                 dropout=0.3):              # Dropout rate for regularization
        super().__init__()

        self.num_spectral_points = num_spectral_points
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes

        # 1. EMBEDDING LAYER
        # Projects scalar spectral intensity (1) to feature space (d_model)
        self.embedding = nn.Linear(1, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # 2. POSITIONAL ENCODING
        # We pre-calculate this once since spectral positions are fixed
        self.positional_encoding = PositionalEncoding(self.d_model, self.num_spectral_points)

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
        # Input x shape: (Batch_Size, num_spectral_points, 1)
        
        # A. Apply Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.embed_dropout(x)
        
        # B. Add Positional Encoding
        x = self.positional_encoding.forward(x)

        # C. Pass through L Transformer Blocks
        # Output shape: (Batch_Size, num_spectral_points, d_model)
        for block in self.transformer_blocks:
            x = block(x)

        # D. Global Average Pooling
        # Average across the spectral dimension (dim=1)
        # Output shape: (Batch_Size, d_model)
        x = x.mean(dim=1)

        # E. Classification
        x = self.pre_classifier_dropout(x)
        logits = self.classifier(x)
        
        if return_logits:
            return logits
        
        return torch.sigmoid(logits)