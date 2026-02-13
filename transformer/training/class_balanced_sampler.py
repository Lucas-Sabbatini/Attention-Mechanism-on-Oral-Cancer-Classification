import numpy as np
import torch


class ClassBalancedSampler(torch.utils.data.Sampler):
    """
    Sampler that ensures each batch contains at least `min_per_class` samples
    from each class. Essential for contrastive learning on imbalanced data.
    """
    def __init__(self, labels: np.ndarray, batch_size: int, min_per_class: int = 2):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.min_per_class = min_per_class
        
        # Get indices per class
        self.class_indices = {}
        for cls in np.unique(self.labels):
            self.class_indices[cls] = np.where(self.labels == cls)[0]
        
        self.num_classes = len(self.class_indices)
        self.num_batches = len(self.labels) // batch_size
        
    def __iter__(self):
        # Shuffle indices within each class
        shuffled_indices = {
            cls: np.random.permutation(indices) 
            for cls, indices in self.class_indices.items()
        }
        
        # Track position in each class
        class_pos = {cls: 0 for cls in self.class_indices}
        
        all_indices = []
        for _ in range(self.num_batches):
            batch_indices = []
            
            # First, ensure min_per_class from each class
            for cls in self.class_indices:
                for _ in range(self.min_per_class):
                    idx = shuffled_indices[cls][class_pos[cls] % len(shuffled_indices[cls])]
                    batch_indices.append(idx)
                    class_pos[cls] += 1
            
            # Fill remaining slots randomly
            remaining = self.batch_size - len(batch_indices)
            if remaining > 0:
                all_remaining = np.concatenate(list(shuffled_indices.values()))
                extra = np.random.choice(all_remaining, size=min(remaining, len(all_remaining)), replace=False)
                batch_indices.extend(extra.tolist())
            
            np.random.shuffle(batch_indices)
            all_indices.extend(batch_indices)
        
        return iter(all_indices)
    
    def __len__(self):
        return self.num_batches * self.batch_size
