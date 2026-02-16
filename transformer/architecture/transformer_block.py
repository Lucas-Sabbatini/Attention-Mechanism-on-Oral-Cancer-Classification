import torch.nn as nn
import torch.nn.functional as F
import torch

from transformer.architecture.attention import MultiHeadAttention


class CustomTransformerBlock(nn.Module):
    """Custom transformer block using our Attention implementation."""
    
    def __init__(self, d_model, nhead, dim_feedforward, seq_len, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead, seq_len)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm: Normaliza ANTES de entrar na atenção
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        x = residual + self.dropout(x)
        
        # Pre-Norm: Normaliza ANTES de entrar no FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x


class ProjectorHead(nn.Module):
    """
    MLP Projector Head for contrastive learning.
    Maps encoder representations to a lower-dimensional latent space
    where the contrastive loss is applied.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)  # L2 normalize for contrastive loss