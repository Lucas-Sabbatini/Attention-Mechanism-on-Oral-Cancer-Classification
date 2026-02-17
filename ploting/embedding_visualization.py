"""
Embedding Visualization Script

Visualizes the encoder's ability to separate classes using t-SNE plots of:
1. Encoder output (after global average pooling)
2. Projection head output (for contrastive learning)

This helps assess whether the transformer encoder learns linearly separable representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preProcess.baseline_correction import BaselineCorrection
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization
from transformer.model import SpectralTransformerModel


def load_and_preprocess_data(dataset_path: str = "dataset_cancboca.dat"):
    """
    Load and preprocess spectral data following the main.py pipeline.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        X: Preprocessed features
        y: Labels (0 = healthy, 1 = cancer)
    """
    # Load data
    dataset = np.loadtxt(dataset_path)
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    y = np.where(y == -1, 0, 1)
    
    # Baseline correction
    baseline = BaselineCorrection().asls_baseline(X)
    X = X - baseline
    
    # Peak normalization
    normalizer = Normalization()
    X = normalizer.peak_normalization(X, 1660.0, 1630.0)
    
    # Truncate to biologically relevant range
    truncator = WavenumberTruncator()
    X = truncator.trucate_range(X, 3050.0, 850.0)
    
    return X, y


def plot_embeddings_with_misclassified(encoder_emb: np.ndarray,
                                        projected_emb: np.ndarray,
                                        labels: np.ndarray,
                                        predictions: np.ndarray,
                                        fold_num: int,
                                        output_dir: str = "ploting/img/encodings",
                                        perplexity: int = 30,
                                        random_state: int = 42,
                                        suffix: str = ""):
    """
    Create t-SNE visualizations with misclassified samples marked.
    
    Computes 2D linear separability by fitting a logistic regression on the t-SNE
    embeddings (metric only, no boundary plotted). Misclassified points (by the 
    actual model) are marked with an X.
    
    Args:
        encoder_emb: Encoder embeddings (n_samples, d_model)
        projected_emb: Projection head embeddings (n_samples, d_model//2)
        labels: True class labels (0 or 1)
        predictions: Model predictions (0 or 1)
        fold_num: Fold number for the title
        output_dir: Base directory for encodings
        perplexity: t-SNE perplexity parameter
        random_state: Random state for reproducibility
        suffix: Suffix for filename (e.g., '_train')
    """
    # Create fold-specific directory
    fold_dir = os.path.join(output_dir, f"fold_{fold_num}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # Compute t-SNE for both embeddings
    print("Computing t-SNE for encoder embeddings...")
    tsne_encoder = TSNE(n_components=2, perplexity=min(perplexity, len(labels) - 1),
                        random_state=random_state, n_iter=1000)
    encoder_2d = tsne_encoder.fit_transform(encoder_emb)
    
    print("Computing t-SNE for projection embeddings...")
    tsne_proj = TSNE(n_components=2, perplexity=min(perplexity, len(labels) - 1),
                     random_state=random_state, n_iter=1000)
    proj_2d = tsne_proj.fit_transform(projected_emb)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color mapping
    colors = ['#2E86AB', '#E94F37']  # Blue for healthy, Red for cancer
    class_names = ['Healthy', 'Cancer']
    
    # Identify misclassified samples
    misclassified = predictions != labels
    
    for ax, emb_2d, title in [(ax1, encoder_2d, 'Encoder Output (Pooling)'),
                               (ax2, proj_2d, 'Projection Head (Contrastive)')]:
        # Fit logistic regression on 2D embeddings for separability metric
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        clf.fit(emb_2d, labels)
        accuracy_2d = clf.score(emb_2d, labels)
        
        # Plot correctly classified points
        for class_idx in [0, 1]:
            mask = (labels == class_idx) & ~misclassified
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                      c=colors[class_idx], label=f'{class_names[class_idx]} (correct)',
                      alpha=0.8, edgecolors='white', linewidth=0.5, s=70, marker='o')
        
        # Plot misclassified points with X marker
        for class_idx in [0, 1]:
            mask = (labels == class_idx) & misclassified
            if mask.sum() > 0:
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                          c=colors[class_idx], label=f'{class_names[class_idx]} (misclassified)',
                          alpha=0.9, edgecolors='black', linewidth=1.5, s=100, marker='X')
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(f'{title}\nFold {fold_num} | 2D Linear Sep: {accuracy_2d:.1%}',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'embeddings{suffix}.png'
    output_path = os.path.join(fold_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved visualization to {output_path}")
    
    # Print misclassification summary
    n_misclassified = misclassified.sum()
    print(f"Misclassified samples: {n_misclassified}/{len(labels)} ({n_misclassified/len(labels):.1%})")
    
    return encoder_2d, proj_2d


def compute_class_separation_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute metrics to quantify class separation in embedding space.
    
    Args:
        embeddings: Embedding vectors (n_samples, dim)
        labels: Class labels
        
    Returns:
        Dictionary with separation metrics
    """
    pos_emb = embeddings[labels == 1]
    neg_emb = embeddings[labels == 0]
    
    # Compute centroids
    pos_centroid = pos_emb.mean(axis=0)
    neg_centroid = neg_emb.mean(axis=0)
    
    # Inter-class distance (between centroids)
    inter_class_dist = np.linalg.norm(pos_centroid - neg_centroid)
    
    # Intra-class distances (average distance to centroid)
    pos_intra = np.mean([np.linalg.norm(x - pos_centroid) for x in pos_emb])
    neg_intra = np.mean([np.linalg.norm(x - neg_centroid) for x in neg_emb])
    avg_intra = (pos_intra + neg_intra) / 2
    
    # Separation ratio (higher is better)
    separation_ratio = inter_class_dist / avg_intra if avg_intra > 0 else 0
    
    return {
        'inter_class_distance': inter_class_dist,
        'intra_class_distance_pos': pos_intra,
        'intra_class_distance_neg': neg_intra,
        'separation_ratio': separation_ratio
    }


def main(random_state: int = 1):
    """
    Main function to train model and visualize embeddings for all folds.
    
    Args:
        random_state: Random state for reproducibility
    """
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data()
    print(f"Data shape: {X.shape}, Labels: {np.bincount(y)}")
    
    # Setup StratifiedKFold
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Process all folds
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        fold_num = fold_idx + 1
        
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_num}/{n_splits}")
        print(f"{'='*60}")
        
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        # Split train into train/val for model training
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold,
            test_size=0.3,
            random_state=random_state,
            stratify=y_train_fold
        )
        
        # Initialize model (fresh for each fold)
        model = SpectralTransformerModel(
            num_spectral_points=X.shape[1],
            verbose=False
        )
        
        # Train the model
        print("Training model...")
        model.train_model(X_train, y_train, X_val, y_val)
        
        # Calibrate threshold on validation set
        model.calibrate_threshold(X_val, y_val)
        
        # Get model predictions on test set
        predictions_test = model.predict(X_test_fold)
        
        # Get embeddings for test set
        print("Extracting test embeddings...")
        encoder_emb, projected_emb = model.get_embeddings(X_test_fold)
        
        print(f"Encoder embedding shape: {encoder_emb.shape}")
        print(f"Projection embedding shape: {projected_emb.shape}")
        
        # Compute separation metrics
        encoder_metrics = compute_class_separation_metrics(encoder_emb, y_test_fold)
        proj_metrics = compute_class_separation_metrics(projected_emb, y_test_fold)
        
        print("\nClass Separation Metrics (Test):")
        print(f"  Encoder - Separation Ratio: {encoder_metrics['separation_ratio']:.3f}")
        print(f"  Projection - Separation Ratio: {proj_metrics['separation_ratio']:.3f}")
        
        # Plot test embeddings
        plot_embeddings_with_misclassified(
            encoder_emb,
            projected_emb,
            y_test_fold,
            predictions_test,
            fold_num=fold_num,
            suffix="_test"
        )
        
        # Also plot train embeddings for comparison
        print("\nExtracting train embeddings...")
        encoder_train, projected_train = model.get_embeddings(X_train_fold)
        predictions_train = model.predict(X_train_fold)
        
        plot_embeddings_with_misclassified(
            encoder_train,
            projected_train,
            y_train_fold,
            predictions_train,
            fold_num=fold_num,
            suffix="_train"
        )
    
    print(f"\n{'='*60}")
    print(f"Completed all {n_splits} folds!")
    print(f"Plots saved to: ploting/img/encodings/fold_{{1-{n_splits}}}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize transformer embeddings for all folds')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    
    args = parser.parse_args()
    main(random_state=args.seed)
