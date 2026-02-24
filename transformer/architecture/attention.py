import torch.nn as nn
import torch

class Attention(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features= d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features= d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features= d_model, bias=False)

        self.d_model = d_model

    def forward(self, encodings_q, encodings_k, encodings_v, mask=None, return_attention=False):

        q = self.W_q(encodings_q)
        k = self.W_k(encodings_k)
        v = self.W_v(encodings_v)

        # Transpose last two dims for batched matmul: (batch, seq, d) @ (batch, d, seq) -> (batch, seq, seq)
        sims = torch.matmul(q, k.transpose(-2, -1))

        scaled_sims = sims / (self.d_model ** 0.5)

        if mask is not None:
            scaled_sims = scaled_sims + mask

        attention_percents = torch.softmax(scaled_sims, dim=-1)

        attention_scores = torch.matmul(attention_percents, v)

        if return_attention:
            return attention_scores, attention_percents

        return attention_scores
    
class MultiHeadAttention(nn.Module):
    """Multi-head attention using the custom Attention class."""
    
    def __init__(self, d_model, nhead, seq_len):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.seq_len = seq_len
        
        # Create attention heads (both use head_dim = d_model // nhead)
        self.heads_inter = nn.ModuleList([
            Attention(self.head_dim) for _ in range(nhead)
        ])

        self.heads_intra = nn.ModuleList([
            Attention(self.head_dim) for _ in range(nhead)
        ])
        
        # Projection layers for inter-feature attention
        # Project seq_len -> d_model, apply multi-head attention, project back
        self.W_proj_inter_in = nn.Linear(seq_len, d_model , bias=False)
        self.W_proj_inter_out = nn.Linear(d_model, seq_len, bias=False)
        self.W_split_inter = nn.Linear(d_model, d_model, bias=False)
        self.W_out_inter = nn.Linear(d_model, d_model, bias=False)
        
        # Projection layers for intra-sample attention (operates on d_model dimension)
        self.W_split_intra = nn.Linear(d_model, d_model, bias=False)
        self.W_out_intra = nn.Linear(d_model, d_model, bias=False)
        
        # Learnable parameter to control inter/intra attention mixing (initialized to 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_len, _ = x.shape

        ## Inter-feature attention: attend across features (seq_len dimension)
        # Transpose to treat features as sequence: (batch, seq, d_model) -> (batch, d_model, seq_len)
        x_transposed = x.transpose(1, 2)  # (batch, d_model, seq_len)

        # Project seq_len -> d_model: (batch, d_model, seq_len) -> (batch, d_model, d_model)
        x_proj_inter = self.W_proj_inter_in(x_transposed)

        # Split into heads: (batch, d_model, d_model) -> (batch, d_model, nhead, head_dim)
        x_proj_inter = self.W_split_inter(x_proj_inter).view(batch_size, self.d_model, self.nhead, self.head_dim)

        head_outputs_inter = []
        inter_attn_weights = []  # (nhead,) each: (batch, d_model, d_model)
        for i, head in enumerate(self.heads_inter):
            head_input = x_proj_inter[:, :, i, :]  # (batch, d_model, head_dim)
            if return_attention:
                head_out, attn_w = head(head_input, head_input, head_input, return_attention=True)
                inter_attn_weights.append(attn_w)
            else:
                head_out = head(head_input, head_input, head_input)
            head_outputs_inter.append(head_out)

        # Concatenate heads: (batch, d_model, d_model)
        concat_inter = torch.cat(head_outputs_inter, dim=-1)

        # Output projection in d_model space
        output_inter = self.W_out_inter(concat_inter)  # (batch, d_model, d_model)

        # Project back d_model -> seq_len: (batch, d_model, d_model) -> (batch, d_model, seq_len)
        output_inter = self.W_proj_inter_out(output_inter)

        # Transpose back: (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        output_inter = output_inter.transpose(1, 2)

        # Residual Connection
        output_inter = output_inter + x

        ## Intra-sample attention: attend across sequence positions (standard attention)
        # Project and reshape: (batch, seq, d_model) -> (batch, seq, nhead, head_dim)
        x_proj_intra = self.W_split_intra(output_inter).view(batch_size, seq_len, self.nhead, self.head_dim)

        # Apply each attention head
        head_outputs_intra = []
        intra_attn_weights = []  # (nhead,) each: (batch, seq_len, seq_len)
        for i, head in enumerate(self.heads_intra):
            head_input = x_proj_intra[:, :, i, :]  # (batch, seq, head_dim)
            if return_attention:
                head_out, attn_w = head(head_input, head_input, head_input, mask, return_attention=True)
                intra_attn_weights.append(attn_w)
            else:
                head_out = head(head_input, head_input, head_input, mask)
            head_outputs_intra.append(head_out)

        # Concatenate heads: (batch, seq, d_model)
        concat_intra = torch.cat(head_outputs_intra, dim=-1)

        # Final projection
        output = self.W_out_intra(concat_intra)

        # Combine inter and intra attention with learnable alpha
        # alpha controls balance: 0 = pure intra, 1 = pure inter
        alpha = torch.sigmoid(self.alpha)  # Constrain to [0, 1]
        output = (1 - alpha) * output + alpha * output_inter

        if return_attention:
            attn_maps = {
                'inter_attention': inter_attn_weights,  # list of (batch, d_model, d_model) per head
                'intra_attention': intra_attn_weights,  # list of (batch, seq_len, seq_len) per head
                'alpha': alpha.item(),
            }
            return output, attn_maps

        return output

