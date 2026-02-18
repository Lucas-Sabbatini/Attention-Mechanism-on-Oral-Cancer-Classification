import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Center Loss for improving intra-class compactness.
    
    Learns a center (centroid) for each class and minimizes the L2 distance
    between embeddings and their corresponding class centers.
    
    This directly attacks intra-class variance, which is the key to improving
    Silhouette Score. Unlike SupConLoss, it doesn't require large batches or pairs.
    
    Reference: Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition"
    
    L_center = (1/2) * sum_i ||f_i - c_{y_i}||^2
    
    Where:
        - f_i is the feature embedding of sample i
        - c_{y_i} is the learned center for class y_i
    """
    
    def __init__(self, num_classes: int, feat_dim: int, device: torch.device = None):
        """
        Args:
            num_classes: Number of classes (2 for binary classification)
            feat_dim: Dimension of the feature embeddings
            device: Device to place the centers on
        """
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # Learnable centers (one per class)
        # Initialize with zeros - will be updated during training
        self.centers = nn.Parameter(torch.zeros(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)
        
        if device is not None:
            self.centers = nn.Parameter(self.centers.to(device))
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, center_separation_weight: float = 0.0) -> torch.Tensor:
        """
        Compute the center loss.
        
        Args:
            features: L2-normalized embeddings (batch_size, feat_dim)
            labels: Ground truth labels (batch_size,) with values in [0, num_classes-1]
            center_separation_weight: Weight for the center separation penalty (default 0.0)
            
        Returns:
            Scalar loss value
        """
        batch_size = features.size(0)
        
        if batch_size == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Get centers for each sample's class
        # labels: (batch_size,) -> centers_batch: (batch_size, feat_dim)
        labels_long = labels.long()
        centers_batch = self.centers[labels_long]
        
        # Compute squared L2 distance to centers
        # ||f_i - c_{y_i}||^2
        diff = features - centers_batch
        dist_sq = (diff ** 2).sum(dim=1)
        
        # Mean over batch
        loss = dist_sq.mean() / 2.0
        
        # Center separation loss: maximize distance between class centers
        center_separation_loss = 0.0
        if center_separation_weight > 0.0 and self.num_classes > 1:
            # Compute pairwise distances between all class centers
            centers = self.centers
            # (num_classes, num_classes)
            dist_matrix = torch.cdist(centers, centers, p=2)
            # Mask diagonal (distance to self = 0)
            mask = ~torch.eye(self.num_classes, dtype=torch.bool, device=centers.device)
            # Minimize negative mean distance (maximize separation)
            center_separation_loss = -dist_matrix[mask].mean()
            loss = loss + center_separation_weight * center_separation_loss
        return loss
