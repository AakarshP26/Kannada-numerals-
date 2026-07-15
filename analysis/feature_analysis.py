"""
Feature Analysis - Tables, Correlations, and Distributions
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from knsd_features import FEATURE_NAMES, extract_knsd_features, IMAGE_SIZE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from train_evaluate import load_kannada_mnist, preprocess_mnist

# Feature metadata
FEATURE_INFO = {
    'num_loops': {'category': 'Loop', 'description': 'Number of closed loops in the character', 'range': '0-3'},
    'total_loop_area': {'category': 'Loop', 'description': 'Sum of all loop areas in pixels', 'range': '0-5000'},
    'avg_loop_area': {'category': 'Loop', 'description': 'Average area per loop', 'range': '0-3000'},
    'loop_area_ratio': {'category': 'Loop', 'description': 'Ratio of loop area to total stroke area', 'range': '0-1'},
    'loop_cx': {'category': 'Loop', 'description': 'Normalized X centroid of largest loop', 'range': '0-1'},
    'loop_cy': {'category': 'Loop', 'description': 'Normalized Y centroid of largest loop', 'range': '0-1'},
    'avg_loop_cy': {'category': 'Loop', 'description': 'Average Y centroid of all loops', 'range': '0-1'},
    'loop_variance': {'category': 'Loop', 'description': 'Variance in loop sizes (normalized std)', 'range': '0-2'},
    'num_endpoints': {'category': 'Endpoint', 'description': 'Number of stroke termination points', 'range': '0-10'},
    'endpoint_avg_x': {'category': 'Endpoint', 'description': 'Average X position of endpoints', 'range': '0-1'},
    'endpoint_avg_y': {'category': 'Endpoint', 'description': 'Average Y position of endpoints', 'range': '0-1'},
    'endpoint_x_spread': {'category': 'Endpoint', 'description': 'Horizontal spread of endpoints', 'range': '0-1'},
    'endpoint_y_spread': {'category': 'Endpoint', 'description': 'Vertical spread of endpoints', 'range': '0-1'},
    'num_junctions': {'category': 'Junction', 'description': 'Number of stroke intersection points', 'range': '0-50'},
    'junction_avg_x': {'category': 'Junction', 'description': 'Average X position of junctions', 'range': '0-1'},
    'junction_avg_y': {'category': 'Junction', 'description': 'Average Y position of junctions', 'range': '0-1'},
    'curv_0': {'category': 'Curvature', 'description': 'Curvature histogram bin 0 (−π to −3π/4)', 'range': '0-1'},
    'curv_1': {'category': 'Curvature', 'description': 'Curvature histogram bin 1 (−3π/4 to −π/2)', 'range': '0-1'},
    'curv_2': {'category': 'Curvature', 'description': 'Curvature histogram bin 2 (−π/2 to −π/4)', 'range': '0-1'},
    'curv_3': {'category': 'Curvature', 'description': 'Curvature histogram bin 3 (−π/4 to 0)', 'range': '0-1'},
    'curv_4': {'category': 'Curvature', 'description': 'Curvature histogram bin 4 (0 to π/4)', 'range': '0-1'},
    'curv_5': {'category': 'Curvature', 'description': 'Curvature histogram bin 5 (π/4 to π/2)', 'range': '0-1'},
    'curv_6': {'category': 'Curvature', 'description': 'Curvature histogram bin 6 (π/2 to 3π/4)', 'range': '0-1'},
    'curv_7': {'category': 'Curvature', 'description': 'Curvature histogram bin 7 (3π/4 to π)', 'range': '0-1'},
    'aspect_ratio': {'category': 'Stroke', 'description': 'Width to height ratio of bounding box', 'range': '0.5-2'},
    'solidity': {'category': 'Stroke', 'description': 'Ratio of contour area to convex hull area', 'range': '0-1'},
    'extent': {'category': 'Stroke', 'description': 'Ratio of contour area to bounding box area', 'range': '0-1'},
    'perimeter': {'category': 'Stroke', 'description': 'Normalized perimeter of main contour', 'range': '0-5'},
    'circularity': {'category': 'Stroke', 'description': '4π×Area/Perimeter² (1 = perfect circle)', 'range': '0-1'},
}


def create_feature_table():
    """Create a comprehensive feature table visualization."""
    fig, ax = plt.subplots(figsize=(24, 20))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.98, 'KNSD: Kannada Numeral Shape Descriptor Features', 
            ha='center', va='top', fontsize=80, fontweight='bold')
    
    # Category colors
    cat_colors = {
        'Loop': '#27ae60',
        'Endpoint': '#3498db',
        'Junction': '#9b59b6',
        'Curvature': '#e74c3c',
        'Stroke': '#f39c12'
    }
    
    # Table data
    columns = ['#', 'Feature Name', 'Category', 'Description', 'Range']
    cell_text = []
    cell_colors = []
    
    for i, feat_name in enumerate(FEATURE_NAMES):
        info = FEATURE_INFO.get(feat_name, {'category': 'Unknown', 'description': 'N/A', 'range': 'N/A'})
        row = [str(i+1), feat_name, info['category'], info['description'], info['range']]
        cell_text.append(row)
        
        cat = info['category']
        color = cat_colors.get(cat, '#bdc3c7')
        # Lighter version for table cells
        cell_colors.append(['#f8f9fa', '#f8f9fa', color + '30', '#f8f9fa', '#f8f9fa'])
    
    # Create table
    table = ax.table(cellText=cell_text, colLabels=columns,
                     cellLoc='left', loc='center',
                     colColours=['#34495e']*5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(80)
    table.scale(1.2, 5.0)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        table[(0, i)].set_facecolor('#2c3e50')
    
    # Style data cells
    for i, row_colors in enumerate(cell_colors):
        for j, color in enumerate(row_colors):
            table[(i+1, j)].set_facecolor(color)
    
    # Column widths
    col_widths = [0.04, 0.15, 0.1, 0.55, 0.1]
    for i, width in enumerate(col_widths):
        for j in range(len(cell_text) + 1):
            table[(j, i)].set_width(width)
    
    # Legend
    legend_y = 0.02
    legend_x = 0.15
    for i, (cat, color) in enumerate(cat_colors.items()):
        rect = mpatches.FancyBboxPatch((legend_x + i*0.15, legend_y), 0.02, 0.02,
                                        transform=ax.transAxes, facecolor=color, edgecolor='none')
        ax.add_patch(rect)
        ax.text(legend_x + i*0.15 + 0.025, legend_y + 0.01, cat, fontsize=100,
                va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig


def create_feature_correlation_matrix(features, labels):
    """Create feature correlation heatmap."""
    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    
    fig, ax = plt.subplots(figsize=(40, 32))
    
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'})
    
    ax.set_title('KNSD Feature Correlation Matrix', fontsize=160, fontweight='bold', pad=40)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=80)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=80)
    
    plt.tight_layout()
    return fig


def create_feature_distribution_by_digit(features, labels):
    """Create feature distribution boxplots for each digit class."""
    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    df['digit'] = labels
    
    # Select key features (one from each category)
    key_features = ['num_loops', 'num_endpoints', 'num_junctions', 
                    'curv_0', 'aspect_ratio', 'circularity']
    
    fig, axes = plt.subplots(2, 3, figsize=(32, 20))
    axes = axes.flatten()
    
    for i, feat in enumerate(key_features):
        ax = axes[i]
        sns.boxplot(x='digit', y=feat, data=df, ax=ax, palette='viridis')
        ax.set_title(f'{feat}', fontsize=120, fontweight='bold')
        ax.set_xlabel('Kannada Digit', fontsize=90)
        ax.set_ylabel('Feature Value', fontsize=90)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('KNSD Feature Distributions by Digit Class', fontsize=160, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_curvature_histogram_expanded(features, labels):
    """Create expanded curvature histogram visualization per digit."""
    curv_indices = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith('curv_')]
    curv_features = features[:, curv_indices]
    
    fig, axes = plt.subplots(2, 5, figsize=(32, 12))
    axes = axes.flatten()
    
    angle_labels = ['−π to −3π/4', '−3π/4 to −π/2', '−π/2 to −π/4', '−π/4 to 0',
                    '0 to π/4', 'π/4 to π/2', 'π/2 to 3π/4', '3π/4 to π']
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 8))
    
    for digit in range(10):
        ax = axes[digit]
        digit_curv = curv_features[labels == digit].mean(axis=0)
        
        bars = ax.bar(range(8), digit_curv, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(f'Digit {digit}', fontsize=60, fontweight='bold')
        ax.set_xticks(range(8))
        ax.set_xticklabels([f'B{i}' for i in range(8)], fontsize=40)
        ax.set_ylim(0, 0.3)
        ax.set_ylabel('Normalized Frequency' if digit % 5 == 0 else '', fontsize=40)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Curvature Histogram Distribution by Kannada Digit', fontsize=80, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02, 'Bin Ranges: B0=−π to −3π/4, B1=−3π/4 to −π/2, ... B7=3π/4 to π', 
             ha='center', fontsize=50, style='italic')
    plt.tight_layout()
    return fig


def create_digit_feature_profiles(features, labels):
    """Create radar/spider chart of feature profiles per digit."""
    # Normalize features
    features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    
    # Select representative features
    selected = [0, 8, 13, 24, 28]  # num_loops, num_endpoints, num_junctions, aspect_ratio, circularity
    selected_names = [FEATURE_NAMES[i] for i in selected]
    
    fig, axes = plt.subplots(2, 5, figsize=(24, 12), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    angles = np.linspace(0, 2 * np.pi, len(selected), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    for digit in range(10):
        ax = axes[digit]
        digit_feats = features_norm[labels == digit][:, selected].mean(axis=0).tolist()
        digit_feats += digit_feats[:1]  # Complete the loop
        
        ax.plot(angles, digit_feats, 'o-', linewidth=5, color=plt.cm.tab10(digit))
        ax.fill(angles, digit_feats, alpha=0.25, color=plt.cm.tab10(digit))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([n[:8] for n in selected_names], size=35)
        ax.set_title(f'Digit {digit}', fontsize=55, fontweight='bold', pad=10)
        ax.set_ylim(-2, 2)
    
    plt.suptitle('KNSD Feature Profiles by Digit (Normalized)', fontsize=80, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def load_and_extract_features():
    """Load dataset and extract features."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'kannada_mnist')
    train_path = os.path.join(data_dir, 'train.csv')
    
    images, labels = load_kannada_mnist(train_path, max_samples_per_class=200)
    
    print("Extracting features...")
    features = []
    valid_labels = []
    
    from tqdm import tqdm
    for img, label in tqdm(zip(images, labels), total=len(images)):
        processed = preprocess_mnist(img)
        feat = extract_knsd_features(processed)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(feat)
        valid_labels.append(label)
    
    return np.array(features), np.array(valid_labels)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs_high_res_10thmarch')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data and extracting features...")
    features, labels = load_and_extract_features()
    
    print("\n1. Creating feature table...")
    fig = create_feature_table()
    fig.savefig(os.path.join(output_dir, 'feature_table.png'), dpi=1200, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'feature_table.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: feature_table.png (+ pdf)")
    
    print("\n2. Creating feature correlation matrix...")
    fig = create_feature_correlation_matrix(features, labels)
    fig.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'feature_correlation.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: feature_correlation.png (+ pdf)")
    
    print("\n3. Creating feature distribution by digit...")
    fig = create_feature_distribution_by_digit(features, labels)
    fig.savefig(os.path.join(output_dir, 'feature_distribution_by_digit.png'), dpi=1200, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'feature_distribution_by_digit.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: feature_distribution_by_digit.png (+ pdf)")
    
    print("\n4. Creating curvature histogram expanded...")
    fig = create_curvature_histogram_expanded(features, labels)
    fig.savefig(os.path.join(output_dir, 'curvature_histogram_expanded.png'), dpi=1200, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'curvature_histogram_expanded.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: curvature_histogram_expanded.png (+ pdf)")
    
    print("\n5. Creating digit feature profiles...")
    fig = create_digit_feature_profiles(features, labels)
    fig.savefig(os.path.join(output_dir, 'digit_feature_profiles.png'), dpi=1200, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'digit_feature_profiles.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: digit_feature_profiles.png (+ pdf)")
    
    print("\nFeature analysis complete!")


if __name__ == '__main__':
    main()
