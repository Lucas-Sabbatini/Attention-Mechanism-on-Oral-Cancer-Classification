"""
Transformer Module

SpectralTransformer for spectral data classification with:
- Patch-based convolutional embedding
- Positional encoding
- Multi-head self-attention transformer blocks
- Supervised contrastive learning support

Submodules:
- architecture/: Core model components (SpectralTransformer, attention, etc.)
- training/: Training utilities (TrainEngine, SupConLoss, etc.)
"""

from transformer.model import BioSpectralFormer
from transformer.architecture import SpectralTransformer
from transformer.training import TrainEngine

__all__ = [
    'BioSpectralFormer',
    'SpectralTransformer',
    'TrainEngine',
]
