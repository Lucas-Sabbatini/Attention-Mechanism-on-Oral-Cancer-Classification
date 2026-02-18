"""
Training Components

Contains training utilities for the SpectralTransformer:
- TrainEngine: Mixin class providing training functionality with Center Loss support
- CenterLoss: Center Loss for intra-class compactness
- SupConLoss: Supervised Contrastive Loss implementation (legacy)
- ClassBalancedSampler: Sampler for balanced batches in contrastive learning
"""

from transformer.training.train_engine import TrainEngine
from transformer.training.center_loss import CenterLoss
from transformer.training.sup_con_loss import SupConLoss
from transformer.training.class_balanced_sampler import ClassBalancedSampler
from transformer.training.train_utils import TrainUtils
__all__ = [
    'TrainEngine',
    'CenterLoss',
    'SupConLoss',
    'ClassBalancedSampler',
    'TrainUtils'
]
