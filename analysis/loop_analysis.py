"""
Loop Analysis - With Loop Features vs Without Loop Features
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from knsd_features import FEATURE_NAMES, extract_knsd_features, IMAGE_SIZE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from train_evaluate import load_kannada_mnist, preprocess_mnist


# Define feature groups
LOOP_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7]  # First 8 features
NON_LOOP_FEATURES = list(range(8, 29))  # Remaining 21 features


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


def evaluate_accuracy(X, y, feature_name="Features"):
    """Evaluate accuracy with 5-fold CV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    
    scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"  {feature_name}: {scores.mean()*100:.2f}% (±{scores.std()*100:.2f}%)")
    return scores.mean() * 100, scores.std() * 100, scores * 100


def create_loop_comparison_plot(results):
    """Create bar chart comparing with/without loop features."""
    fig, ax = plt.subplots(figsize=(24, 14))
    
    categories = ['Without Loop\nFeatures (21)', 'With Loop\nFeatures Only (8)', 
                  'All KNSD\nFeatures (29)']
    accuracies = [results['no_loop'][0], results['loop_only'][0], results['all'][0]]
    stds = [results['no_loop'][1], results['loop_only'][1], results['all'][1]]
    colors = ['#e74c3c', '#3498db', '#27ae60']
    
    bars = ax.bar(categories, accuracies, yerr=stds, capsize=8, color=colors,
                  edgecolor='white', linewidth=2, alpha=0.85)
    
    # Add value labels
    for bar, acc, std in zip(bars, accuracies, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=48, fontweight='bold')
    
    # Styling
    ax.set_ylabel('5-Fold Cross-Validation Accuracy (%)', fontsize=44)
    ax.set_title('Impact of Loop-Based Features on Classification Accuracy', 
                 fontsize=64, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xticklabels(categories, fontsize=40)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=results['all'][0], color='#27ae60', linestyle='--', alpha=0.5, linewidth=2)
    
    # Improvement annotation
    improvement = results['all'][0] - results['no_loop'][0]
    ax.annotate(f'+{improvement:.1f}% improvement\nfrom loop features',
                xy=(2, results['all'][0]), xytext=(2.4, results['all'][0] - 10),
                fontsize=40, color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#27ae60'))
    
    plt.tight_layout()
    return fig


def create_loop_distribution_by_digit(features, labels):
    """Show loop feature distribution for each digit class."""
    fig, axes = plt.subplots(2, 5, figsize=(32, 16))
    axes = axes.flatten()
    
    loop_names = [FEATURE_NAMES[i] for i in LOOP_FEATURES]
    
    for digit in range(10):
        ax = axes[digit]
        digit_features = features[labels == digit][:, LOOP_FEATURES]
        
        # Create heatmap of mean loop features
        mean_values = digit_features.mean(axis=0)
        
        # Normalize for visualization
        mean_norm = (mean_values - mean_values.min()) / (mean_values.max() - mean_values.min() + 1e-6)
        
        colors = plt.cm.Greens(mean_norm)
        bars = ax.bar(range(8), mean_values, color=colors, edgecolor='white')
        
        ax.set_title(f'Digit {digit}', fontsize=44, fontweight='bold')
        ax.set_xticks(range(8))
        ax.set_xticklabels([f'L{i}' for i in range(8)], fontsize=32)
        ax.tick_params(axis='y', labelsize=28)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Loop Feature Distribution by Kannada Digit', fontsize=72, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02, 'L0=num_loops, L1=total_area, L2=avg_area, L3=ratio, L4=cx, L5=cy, L6=avg_cy, L7=variance',
             ha='center', fontsize=36, style='italic')
    plt.tight_layout()
    return fig


def create_loop_count_histogram(features, labels):
    """Create histogram of loop counts per digit class."""
    fig, ax = plt.subplots(figsize=(24, 12))
    
    loop_counts = features[:, 0]  # num_loops is first feature
    
    # Compute mean loop count per digit
    mean_loops = []
    for digit in range(10):
        mean_val = loop_counts[labels == digit].mean()
        mean_loops.append(mean_val)
    
    # Kannada numerals
    kannada_digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    colors = plt.cm.viridis(np.array(mean_loops) / max(mean_loops))
    bars = ax.bar(range(10), mean_loops, color=colors, edgecolor='white', linewidth=2)
    
    ax.set_xticks(range(10))
    ax.set_xticklabels([f'{d}\n({kd})' for d, kd in zip(range(10), kannada_digits)], fontsize=40)
    ax.set_xlabel('Digit', fontsize=44)
    ax.set_ylabel('Average Number of Loops', fontsize=44)
    ax.set_title('Average Loop Count per Kannada Digit Class', fontsize=64, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate high-loop digits
    for i, (bar, mean_val) in enumerate(zip(bars, mean_loops)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean_val:.2f}', ha='center', va='bottom', fontsize=36)
    
    plt.tight_layout()
    return fig


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs_high_res_10thmarch')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'kannada_mnist')
    train_path = os.path.join(data_dir, 'train.csv')
    
    images, labels = load_kannada_mnist(train_path, max_samples_per_class=None)
    features, labels = extract_features(images, labels)
    
    print("\n=== Loop Feature Analysis ===")
    
    # Evaluate different feature sets
    print("\nEvaluating accuracy with different feature sets...")
    results = {}
    
    # Without loop features
    X_no_loop = features[:, NON_LOOP_FEATURES]
    acc, std, _ = evaluate_accuracy(X_no_loop, labels, "Without Loops")
    results['no_loop'] = (acc, std)
    
    # Loop features only
    X_loop_only = features[:, LOOP_FEATURES]
    acc, std, _ = evaluate_accuracy(X_loop_only, labels, "Loop Only")
    results['loop_only'] = (acc, std)
    
    # All features
    acc, std, _ = evaluate_accuracy(features, labels, "All Features")
    results['all'] = (acc, std)
    
    print("\n1. Creating loop comparison plot...")
    fig = create_loop_comparison_plot(results)
    # Save PNG
    fig.savefig(os.path.join(output_dir, 'loop_vs_no_loop.png'), dpi=1200,
                bbox_inches='tight', facecolor='white')
    # Save PDF
    fig.savefig(os.path.join(output_dir, 'loop_vs_no_loop.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: loop_vs_no_loop.png (+ pdf)")
    
    print("\n2. Creating loop distribution by digit...")
    fig = create_loop_distribution_by_digit(features, labels)
    fig.savefig(os.path.join(output_dir, 'loop_distribution_by_digit.png'), dpi=1200,
                bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'loop_distribution_by_digit.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: loop_distribution_by_digit.png (+ pdf)")
    
    print("\n3. Creating loop count histogram...")
    fig = create_loop_count_histogram(features, labels)
    fig.savefig(os.path.join(output_dir, 'loop_count_histogram.png'), dpi=1200,
                bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'loop_count_histogram.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: loop_count_histogram.png (+ pdf)")
    
    print("\nLoop analysis complete!")


if __name__ == '__main__':
    main()
