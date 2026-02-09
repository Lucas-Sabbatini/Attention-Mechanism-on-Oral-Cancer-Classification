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
import math

class SpectralTransformer(nn.Module):
    def __init__(self, 
                 num_spectral_points,       # "d" (~1k features)
                 d_model=64,                # Dimension of the feature space
                 nhead=4,                   # "h" attention heads
                 num_layers=4,              # "L" stacked blocks
                 dim_feedforward=128,       # "d_ff" intermediate dimension
                 num_classes=1):            # Binary classification (Cancer vs Healthy)
        super().__init__()

        # 1. EMBEDDING LAYER
        # Projects scalar spectral intensity (1) to feature space (d_model)
        self.embedding = nn.Linear(1, d_model)
        self.d_model = d_model

        # 2. POSITIONAL ENCODING
        # We pre-calculate this once since spectral positions are fixed
        self.register_buffer('pe', self._generate_positional_encoding(num_spectral_points, d_model))

        # 3. STACKED TRANSFORMER BLOCKS ("L blocks")
        # Contains: Self-Attention, Feed-forward, LayerNorm, Residuals
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True  # Expected input: (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. CLASSIFICATION HEAD
        # Linear layer for final logits
        self.classifier = nn.Linear(d_model, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _generate_positional_encoding(self, length, d_model):
        """
        Implements: PE(pos, 2i) = sin(...), PE(pos, 2i+1) = cos(...)
        """
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # Shape: (1, 1000, d_model)

    def forward(self, x):
        # Input x shape: (Batch_Size, 1000, 1)
        
        # A. Apply Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # B. Add Positional Encoding
        x = x + self.pe[:, :x.size(1), :]

        # C. Pass through L Transformer Blocks
        # Output shape: (Batch_Size, 1000, d_model)
        x = self.transformer_encoder(x)

        # D. Global Average Pooling
        # Average across the spectral dimension (dim=1)
        # Output shape: (Batch_Size, d_model)
        x = x.mean(dim=1)

        # E. Classification
        logits = self.classifier(x)
        probs = self.sigmoid(logits)
        
        return probs
