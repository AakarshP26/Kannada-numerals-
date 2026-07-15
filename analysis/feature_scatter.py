"""
Top Features Scatter Plot - Class Separability Visualization
Creates scatter plots of top feature pairs with DB Index
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from knsd_features import FEATURE_NAMES, extract_knsd_features, IMAGE_SIZE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from train_evaluate import load_kannada_mnist, preprocess_mnist


def extract_features(images, labels):
    """Extract KNSD features from images."""
    print("Extracting KNSD features...")
    features = []
    
    for img in tqdm(images):
        processed = preprocess_mnist(img)
        feat = extract_knsd_features(processed)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(feat)
    
    return np.array(features), labels


def create_feature_scatter_plots(features, labels, output_path):
    """Create 2x2 scatter plots of top feature pairs."""
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Top feature pairs to visualize (based on importance analysis)
    # These are indices into FEATURE_NAMES
    feature_pairs = [
        (0, 8, "Loop vs Endpoint Features"),     # num_loops vs num_endpoints
        (0, 28, "Loop vs Circularity"),          # num_loops vs circularity
        (8, 13, "Endpoint vs Junction"),         # num_endpoints vs num_junctions
        (3, 28, "Loop Ratio vs Circularity"),    # loop_area_ratio vs circularity
    ]
    
    # Kannada digit labels for legend
    kannada = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # Colors for 10 classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    fig, axes = plt.subplots(2, 2, figsize=(28, 24))
    axes = axes.flatten()
    
    for idx, (feat1_idx, feat2_idx, title) in enumerate(feature_pairs):
        ax = axes[idx]
        
        feat1 = features_scaled[:, feat1_idx]
        feat2 = features_scaled[:, feat2_idx]
        
        # Plot each class
        for digit in range(10):
            mask = labels == digit
            ax.scatter(
                feat1[mask], feat2[mask],
                c=[colors[digit]], 
                label=f'{digit} ({kannada[digit]})',
                alpha=0.6, s=15, edgecolors='none'
            )
        
        # Calculate DB Index for this feature pair
        X_pair = np.column_stack([feat1, feat2])
        db_index = davies_bouldin_score(X_pair, labels)
        
        ax.set_xlabel(FEATURE_NAMES[feat1_idx], fontsize=80)
        ax.set_ylabel(FEATURE_NAMES[feat2_idx], fontsize=80)
        ax.set_title(title, fontsize=90, fontweight='bold', 
                     bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
        ax.grid(True, alpha=0.3)
        
        # Add DB Index annotation
        ax.text(0.98, 0.02, f'DB Index: {db_index:.2f}', 
                transform=ax.transAxes, fontsize=16, color='red',
                ha='right', va='bottom', fontweight='bold')
    
    # Create legend
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper center', ncol=10, 
               bbox_to_anchor=(0.5, 1.02), fontsize=14, frameon=True,
               title='Kannada Digit Classes')
    
    plt.suptitle('KNSD Feature Space: Class Separability Analysis', 
                 fontsize=120, fontweight='bold', y=1.06)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=2500, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_top2_features_plot(features, labels, output_path):
    """Create single scatter plot of THE top 2 most important features."""
    from sklearn.feature_selection import mutual_info_classif
    
    # Find top 2 features by mutual information
    mi_scores = mutual_info_classif(features, labels, random_state=42)
    top2_indices = np.argsort(mi_scores)[-2:][::-1]
    
    print(f"Top 2 features: {FEATURE_NAMES[top2_indices[0]]}, {FEATURE_NAMES[top2_indices[1]]}")
    
    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    feat1 = features_scaled[:, top2_indices[0]]
    feat2 = features_scaled[:, top2_indices[1]]
    
    # Calculate DB Index
    X_pair = np.column_stack([feat1, feat2])
    db_index = davies_bouldin_score(X_pair, labels)
    
    # Colors and markers
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    kannada = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    fig, ax = plt.subplots(figsize=(120, 100))
    
    for digit in range(10):
        mask = labels == digit
        ax.scatter(
            feat1[mask], feat2[mask],
            c=[colors[digit]], 
            label=f'{digit} ({kannada[digit]})',
            alpha=0.7, s=30, edgecolors='white', linewidths=0.5
        )
    
    ax.set_xlabel(FEATURE_NAMES[top2_indices[0]], fontsize=100)
    ax.set_ylabel(FEATURE_NAMES[top2_indices[1]], fontsize=100)
    ax.set_title('KNSD: Top 2 Features Class Separability', fontsize=120, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # DB Index annotation
    ax.text(0.98, 0.02, f'DB Index: {db_index:.2f}', 
            transform=ax.transAxes, fontsize=16, color='red',
            ha='right', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=14, frameon=True, title='Digit Class')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=2500, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs_high_res_10thmarch')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'kannada_mnist')
    train_path = os.path.join(data_dir, 'train.csv')
    
    images, labels = load_kannada_mnist(train_path, max_samples_per_class=None)
    features, labels = extract_features(images, labels)
    
    print("\n1. Creating 2x2 feature scatter plots...")
    create_feature_scatter_plots(
        features, labels, 
        os.path.join(output_dir, 'feature_scatter_2x2.png')
    )
    
    print("\n2. Creating top 2 features plot...")
    create_top2_features_plot(
        features, labels,
        os.path.join(output_dir, 'top2_features_scatter.png')
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
