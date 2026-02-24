import torch
import torch.nn as nn
import math

from transformer.architecture.posEncoding import PositionalEncoding
from transformer.architecture.transformer_block import CustomTransformerBlock, ProjectorHead

# Biochemical regions of interest (wavenumber ranges in cm⁻¹)
# Each region is defined as (upper_wn, lower_wn) in descending order
BIOCHEMICAL_REGIONS = {
    'amide_i':      (1700, 1600),   # C=O stretch, protein secondary structure
    'amide_ii':     (1580, 1480),   # N-H bend + C-N stretch, protein conformation
    'amide_iii':    (1350, 1200),   # C-N stretch + N-H bend, protein backbone
    'lipid_ch':     (3050, 2800),   # C-H stretching, lipid content
    'carbohydrate': (1170, 1000),   # C-O stretch, glycogen and carbohydrates
    'nucleic_acid': (1270, 1185),   # PO₂⁻ asymmetric stretch, DNA/RNA
    'phosphate':    (1130, 1000),   # PO₂⁻ symmetric stretch, phosphorylation
}

# Biologically meaningful interaction pairs
# Each pair represents coupled biochemical processes relevant to cancer detection
DEFAULT_REGION_PAIRS = [
    ('amide_i', 'amide_ii'),        # Protein secondary structure coupling
    ('amide_i', 'amide_iii'),       # Protein backbone conformation
    ('amide_ii', 'amide_iii'),      # Protein backbone coupling
    ('amide_i', 'lipid_ch'),        # Protein-lipid interaction
    ('amide_i', 'carbohydrate'),    # Protein-glycan interaction
    ('nucleic_acid', 'phosphate'),  # DNA/RNA structural coupling
]


def resolve_region_pairs(
    pair_names: list[tuple[str, str]],
    region_definitions: dict[str, tuple[int, int]] = None,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Resolve named region pairs to wavenumber range tuples.

    Args:
        pair_names: List of (region_name_A, region_name_B) pairs.
        region_definitions: Dict mapping region names to (upper_wn, lower_wn).
            Defaults to BIOCHEMICAL_REGIONS.

    Returns:
        List of ((upper_A, lower_A), (upper_B, lower_B)) wavenumber tuples.
    """
    if region_definitions is None:
        region_definitions = BIOCHEMICAL_REGIONS

    resolved = []
    for name_a, name_b in pair_names:
        if name_a not in region_definitions:
            raise ValueError(f"Unknown region '{name_a}'. Available: {list(region_definitions.keys())}")
        if name_b not in region_definitions:
            raise ValueError(f"Unknown region '{name_b}'. Available: {list(region_definitions.keys())}")
        resolved.append((region_definitions[name_a], region_definitions[name_b]))
    return resolved


class SpectralTransformer(nn.Module):
    def __init__(self,
                 num_spectral_points,       # "d" (~1k features)
                 d_model=64,                # Dimension of the feature space
                 nhead=4,                   # "h" attention heads
                 num_layers=4,              # "L" stacked blocks
                 dim_feedforward=128,       # "d_ff" intermediate dimension
                 num_classes=1,             # Binary classification (Cancer vs Healthy)
                 dropout=0.3,               # Dropout rate for regularization
                 patch_size=16,             # Size of spectral patches for conv embedding
                 attention_mask_bias=None): # Additive attention bias (num_patches, num_patches)
        super().__init__()

        self.num_spectral_points = num_spectral_points
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.patch_size = patch_size

        # Calculate stride for overlapping patches (50% overlap by default)
        self.patch_stride = patch_size // 2

        # Calculate number of patches with overlap: (L - K) / S + 1
        self.num_patches = (num_spectral_points - patch_size) // self.patch_stride + 1

        # 0. ATTENTION MASK BIAS (static, not a learnable parameter)
        if attention_mask_bias is not None:
            self.register_buffer('attention_mask_bias', attention_mask_bias)
        else:
            self.attention_mask_bias = None

        # 1. CONVOLUTIONAL PATCH EMBEDDING
        # Projects a patch of spectral intensities to feature space (d_model)
        # Conv1d: (batch, 1, num_spectral_points) -> (batch, d_model, num_patches)
        self.patch_embedding = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=self.patch_stride  # Overlapping patches (50% overlap)
        )
        self.embed_dropout = nn.Dropout(dropout)

        # 2. POSITIONAL ENCODING
        # Now based on num_patches since each patch becomes a token
        self.positional_encoding = PositionalEncoding(self.d_model, self.num_patches)

        # 3. STACKED TRANSFORMER BLOCKS ("L blocks")
        # Contains: Self-Attention, Feed-forward, LayerNorm, Residuals
        # Using our custom implementation with the Attention class
        self.transformer_blocks = nn.ModuleList([
            CustomTransformerBlock(d_model, nhead, dim_feedforward, seq_len=self.num_patches, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 4. CLASSIFICATION HEAD
        # Linear layer for final logits
        self.pre_classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
        # 5. PROJECTOR HEAD for Contrastive Learning
        # Maps d_model -> latent space for SupCon loss
        self.projector = ProjectorHead(
            in_dim=d_model,
            hidden_dim=d_model * 2,
            out_dim=d_model // 2
        )

    def forward(self, x, return_logits=False, return_embeddings=False, return_attention=False):
        """
        Forward pass with options for different outputs.

        Args:
            x: Input tensor (batch_size, num_spectral_points) or (batch_size, num_spectral_points, 1)
            return_logits: If True, return raw logits instead of probabilities
            return_embeddings: If True, return (output, encoder_repr, projected_embeddings) tuple
                              for contrastive learning
            return_attention: If True, also return list of attention maps from each block

        Returns:
            - Default: sigmoid probabilities
            - return_logits=True: raw logits
            - return_embeddings=True: (logits/probs, encoder_repr, projected_embeddings)
            - return_attention=True: appends attention_maps list to any of the above returns
              attention_maps[i] is the dict from block i with keys
              'inter_attention', 'intra_attention', 'alpha'
        """
        # A. Reshape for Conv1d: (Batch_Size, 1, num_spectral_points)
        if x.dim() == 3:
            x = x.squeeze(-1)  # Remove last dim if present
        x = x.unsqueeze(1)  # Add channel dimension

        # B. Apply Convolutional Patch Embedding
        # Output: (Batch_Size, d_model, num_patches)
        x = self.patch_embedding(x)

        # C. Transpose for transformer: (Batch_Size, num_patches, d_model)
        x = x.transpose(1, 2)
        x = x * math.sqrt(self.d_model)
        x = self.embed_dropout(x)

        # D. Add Positional Encoding
        x = self.positional_encoding.forward(x)

        # E. Pass through L Transformer Blocks
        # Output shape: (Batch_Size, num_patches, d_model)
        all_attn_maps = []
        for block in self.transformer_blocks:
            if return_attention:
                x, attn_maps = block(x, mask=self.attention_mask_bias, return_attention=True)
                all_attn_maps.append(attn_maps)
            else:
                x = block(x, mask=self.attention_mask_bias)

        # F. Global Average Pooling
        # Average across the patch dimension (dim=1)
        # Output shape: (Batch_Size, d_model)
        encoder_repr = x.mean(dim=1)

        # G. Classification
        x = self.pre_classifier_dropout(encoder_repr)
        logits = self.classifier(x)

        if return_embeddings:
            # H. Project to contrastive latent space
            projected = self.projector(encoder_repr)
            if return_logits:
                main_out = (logits, encoder_repr, projected)
            else:
                main_out = (torch.sigmoid(logits), encoder_repr, projected)
        elif return_logits:
            main_out = logits
        else:
            main_out = torch.sigmoid(logits)

        if return_attention:
            if isinstance(main_out, tuple):
                return (*main_out, all_attn_maps)
            return main_out, all_attn_maps

        return main_out