import torch.nn as nn

from transformer.attention import MultiHeadAttention


class CustomTransformerBlock(nn.Module):
    """Custom transformer block using our Attention implementation."""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead)
        
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
