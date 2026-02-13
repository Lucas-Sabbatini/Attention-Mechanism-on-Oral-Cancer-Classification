import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    
    Reference: https://arxiv.org/abs/2004.11362
    
    Pulls embeddings of the same class together and pushes different classes apart.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: L2-normalized embeddings (batch_size, embed_dim)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        labels = labels.contiguous().view(-1, 1)
        
        # Mask: 1 where labels match (same class), 0 otherwise
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute similarity matrix (dot product since features are L2-normalized)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Remove self-contrast (diagonal)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix * logits_mask, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Mean of log-likelihood over positive pairs
        # Handle case where there are no positive pairs for a sample
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss
