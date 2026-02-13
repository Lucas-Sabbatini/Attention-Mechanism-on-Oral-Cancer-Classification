import torch.nn as nn
import torch

class Attention(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features= d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features= d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features= d_model, bias=False)

        self.d_model = d_model

    def forward(self, encodings_q, encodings_k, encodings_v, mask=None):

        q = self.W_q(encodings_q)
        k = self.W_k(encodings_k)
        v = self.W_v(encodings_v)

        # Transpose last two dims for batched matmul: (batch, seq, d) @ (batch, d, seq) -> (batch, seq, seq)
        sims = torch.matmul(q, k.transpose(-2, -1))

        scaled_sims = sims / (self.d_model ** 0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = torch.softmax(scaled_sims, dim=-1)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
    
class MultiHeadAttention(nn.Module):
    """Multi-head attention using the custom Attention class."""
    
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Create attention heads
        self.heads = nn.ModuleList([
            Attention(self.head_dim) for _ in range(nhead)
        ])
        
        # Projection layers to split and combine heads
        self.W_split = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape: (batch, seq, d_model) -> (batch, seq, nhead, head_dim)
        x_proj = self.W_split(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        
        # Apply each attention head
        head_outputs = []
        for i, head in enumerate(self.heads):
            head_input = x_proj[:, :, i, :]  # (batch, seq, head_dim)
            head_out = head(head_input, head_input, head_input, mask)
            head_outputs.append(head_out)
        
        # Concatenate heads: (batch, seq, d_model)
        concat = torch.cat(head_outputs, dim=-1)
        
        # Final projection
        output = self.W_out(concat)
        return output

