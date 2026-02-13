"""
Transformer Architecture Components

Contains the core building blocks for the SpectralTransformer:
- SpectralTransformer: Main transformer model for spectral classification
- CustomTransformerBlock: Transformer encoder block with attention + FFN
- MultiHeadAttention: Multi-head self-attention mechanism
- PositionalEncoding: Sinusoidal positional encoding
- ProjectorHead: MLP projector for contrastive learning
"""

from transformer.architecture.main import SpectralTransformer
from transformer.architecture.transformer_block import CustomTransformerBlock, ProjectorHead
from transformer.architecture.attention import MultiHeadAttention
from transformer.architecture.posEncoding import PositionalEncoding

__all__ = [
    'SpectralTransformer',
    'CustomTransformerBlock',
    'MultiHeadAttention',
    'PositionalEncoding',
    'ProjectorHead',
]
