import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Implements: 
    PE(pos, 2i) = sin(pos/10000**(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000**(2i/d_model))
    """

    def __init__(self, d_model=64, max_len=1141):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1 / torch.tensor(10000.0) ** (embedding_index / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register with shape (1, max_len, d_model) for easy broadcasting
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, spectra_embeddings):
        """
        Args:
            spectra_embeddings: Tensor of shape (Batch, Seq_len, d_model)
        Returns:
            Tensor of shape (Batch, Seq_len, d_model) with positional encoding added
        """
        # self.pe shape: (1, max_len, d_model)
        # Slice to match sequence length and broadcast across batch
        return spectra_embeddings + self.pe[:, :spectra_embeddings.size(1), :]