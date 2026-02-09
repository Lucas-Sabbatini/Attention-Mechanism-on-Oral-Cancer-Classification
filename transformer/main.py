import torch
import torch.nn as nn
import math

from transformer.posEncoding import PositionalEncoding

class SpectralTransformer(nn.Module):
    def __init__(self, 
                 num_spectral_points,       # "d" (~1k features)
                 d_model=64,                # Dimension of the feature space
                 nhead=4,                   # "h" attention heads
                 num_layers=4,              # "L" stacked blocks
                 dim_feedforward=128,       # "d_ff" intermediate dimension
                 num_classes=1):            # Binary classification (Cancer vs Healthy)
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

        # 2. POSITIONAL ENCODING
        # We pre-calculate this once since spectral positions are fixed
        self.positional_encoding = PositionalEncoding(self.d_model, self.num_spectral_points)

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

    def forward(self, x):
        # Input x shape: (Batch_Size, num_spectral_points, 1)
        
        # A. Apply Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # B. Add Positional Encoding
        x = self.positional_encoding.forward(x)

        # C. Pass through L Transformer Blocks
        # Output shape: (Batch_Size, num_spectral_points, d_model)
        x = self.transformer_encoder(x)

        # D. Global Average Pooling
        # Average across the spectral dimension (dim=1)
        # Output shape: (Batch_Size, d_model)
        x = x.mean(dim=1)

        # E. Classification
        logits = self.classifier(x)
        probs = self.sigmoid(logits)
        
        return probs