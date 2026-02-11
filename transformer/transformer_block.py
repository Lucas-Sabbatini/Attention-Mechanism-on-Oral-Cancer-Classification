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
        # Self-attention with residual connection and layer norm
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection and layer norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x
