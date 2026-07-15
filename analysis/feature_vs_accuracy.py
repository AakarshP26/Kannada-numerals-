"""
Feature vs Accuracy Analysis
Analyze how accuracy changes with number of features
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from tqdm import tqdm
import cv2

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


def create_feature_vs_accuracy_plot(features, labels):
    """Create plot showing accuracy vs number of features."""
    print("\nComputing feature importance...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Compute mutual information for feature ranking
    mi_scores = mutual_info_classif(X_scaled, labels, random_state=42)
    feature_order = np.argsort(mi_scores)[::-1]  # Best first
    
    print("\nEvaluating accuracy with increasing number of features...")
    num_features_range = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24, 27, 29]
    accuracies = []
    stds = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    
    for n_feat in tqdm(num_features_range):
        selected = feature_order[:n_feat]
        X_subset = X_scaled[:, selected]
        
        scores = cross_val_score(svm, X_subset, labels, cv=cv, scoring='accuracy')
        accuracies.append(scores.mean() * 100)
        stds.append(scores.std() * 100)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(24, 14))
    
    ax.plot(num_features_range, accuracies, 'o-', linewidth=3, markersize=12,
            color='#2ecc71', label='5-Fold CV Accuracy')
    ax.fill_between(num_features_range, 
                    np.array(accuracies) - np.array(stds),
                    np.array(accuracies) + np.array(stds),
                    alpha=0.2, color='#2ecc71')
    
    # Mark best accuracy
    best_idx = np.argmax(accuracies)
    ax.scatter([num_features_range[best_idx]], [accuracies[best_idx]], 
               s=300, c='red', marker='*', zorder=5, label=f'Best: {accuracies[best_idx]:.2f}%')
    
    # Styling
    ax.set_xlabel('Number of Features (ranked by Mutual Information)', fontsize=44)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=44)
    ax.set_title('KNSD Feature Count vs Classification Accuracy', fontsize=64, fontweight='bold')
    ax.set_xticks(num_features_range)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=36)
    
    # Annotate key points
    ax.annotate(f'All 29 features\n{accuracies[-1]:.1f}%', 
                xy=(29, accuracies[-1]), xytext=(24, accuracies[-1]-10),
                fontsize=36, arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    return fig, feature_order, mi_scores


def create_feature_ranking_plot(mi_scores):
    """Create bar chart of feature importance ranking."""
    fig, ax = plt.subplots(figsize=(28, 16))
    
    # Sort by importance
    sorted_idx = np.argsort(mi_scores)[::-1]
    sorted_scores = mi_scores[sorted_idx]
    sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
    
    # Color by category
    cat_colors = {
        'Loop': '#27ae60',
        'Endpoint': '#3498db',
        'Junction': '#9b59b6',
        'Curvature': '#e74c3c',
        'Stroke': '#f39c12'
    }
    
    def get_category(name):
        n = name.lower()
        if 'loop' in n: return 'Loop'
        elif 'endpoint' in n: return 'Endpoint'
        elif 'junction' in n: return 'Junction'
        elif 'curv' in n: return 'Curvature'
        else: return 'Stroke'
    
    colors = [cat_colors[get_category(name)] for name in sorted_names]
    
    bars = ax.barh(range(len(sorted_names)), sorted_scores, color=colors, edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=32)
    ax.invert_yaxis()
    
    ax.set_xlabel('Mutual Information Score', fontsize=44)
    ax.set_title('KNSD Feature Importance Ranking (Mutual Information)', fontsize=64, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add rank numbers
    for i, (score, bar) in enumerate(zip(sorted_scores, bars)):
        ax.text(score + 0.002, i, f'#{i+1}', va='center', fontsize=28, color='gray')
    
    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color, label=cat) for cat, color in cat_colors.items()]
    ax.legend(handles=legend_handles, loc='lower right', title='Feature Category', fontsize=32, title_fontsize=36)
    
    plt.tight_layout()
    return fig


def create_feature_weightage_plot(features, labels):
    """Create feature weightage visualization using Logistic Regression coefficients."""
    from sklearn.linear_model import LogisticRegression
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Use logistic regression to get approximate feature weights
    lr = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
    lr.fit(X_scaled, labels)
    
    # Aggregate absolute weights across all classes
    weights = np.abs(lr.coef_).mean(axis=0)
    weights = weights / weights.sum()  # Normalize to sum to 1
    
    fig, ax = plt.subplots(figsize=(24, 16))
    
    sorted_idx = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_idx]
    sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
    
    # Create gradient colors
    colors = plt.cm.RdYlGn(np.linspace(0.9, 0.2, len(sorted_names)))
    
    bars = ax.barh(range(len(sorted_names)), sorted_weights * 100, color=colors, edgecolor='white')
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=32)
    ax.invert_yaxis()
    
    ax.set_xlabel('Relative Weight (%)', fontsize=44)
    ax.set_title('KNSD Feature Weight Contribution (Logistic Regression)', fontsize=64, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (weight, bar) in enumerate(zip(sorted_weights, bars)):
        ax.text(weight * 100 + 0.1, i, f'{weight*100:.1f}%', va='center', fontsize=28)
    
    plt.tight_layout()
    return fig


def create_feature_group_accuracy_plot(features, labels):
    """Create feature group ablation study."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    
    # Define feature groups
    groups = {
        'All Features (29)': list(range(29)),
        'Topology Only\n(Loops, Ends, Juncs)': [i for i, n in enumerate(FEATURE_NAMES) 
            if 'loop' in n.lower() or 'endpoint' in n.lower() or 'junction' in n.lower()],
        'Geometry Only\n(Curv, Stroke)': [i for i, n in enumerate(FEATURE_NAMES) 
            if 'curv' in n.lower() or 'stroke' in n.lower() or 'extent' in n.lower() or 'solidity' in n.lower()],
        'Loops Only (8)': [i for i, n in enumerate(FEATURE_NAMES) if 'loop' in n.lower()],
        'Rotation Invariant (8)': [i for i, n in enumerate(FEATURE_NAMES) if 'curv' in n.lower()],
        'Ends + Junctions (4)': [i for i, n in enumerate(FEATURE_NAMES) 
            if 'endpoint' in n.lower() or 'junction' in n.lower()],
    }
    
    results = {}
    for name, idxs in groups.items():
        if not idxs: continue
        scores = cross_val_score(svm, X_scaled[:, idxs], labels, cv=cv, scoring='accuracy')
        results[name] = (scores.mean() * 100, scores.std() * 100)
    
    fig, ax = plt.subplots(figsize=(24, 14))
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    
    y_pos = np.arange(len(names))
    colors = ['#2ecc71', '#e67e22', '#3498db', '#9b59b6', '#e74c3c', '#1abc9c']
    
    bars = ax.barh(y_pos, means, xerr=stds, align='center', color=colors, capsize=8, edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=36)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy (%)', fontsize=44)
    ax.set_title('Feature Group Ablation Study', fontsize=64, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (v, std) in enumerate(zip(means, stds)):
        ax.text(v + 2, i, f"{v:.1f}±{std:.1f}%", va='center', fontweight='bold', fontsize=36)
        
    plt.tight_layout()
    return fig


def create_class_wise_accuracy_plot(features, labels):
    """Create per-class (per-digit) accuracy bar chart."""
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2, stratify=labels, random_state=42)
    
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    fig, ax = plt.subplots(figsize=(24, 12))
    classes = range(10)
    norm = plt.Normalize(90, 100)
    colors = plt.cm.RdYlGn(norm(class_acc))
    
    bars = ax.bar(classes, class_acc, color=colors, edgecolor='black', alpha=0.8, width=0.7)
    
    ax.set_xlabel('Kannada Digit Class', fontsize=44)
    ax.set_ylabel('Recall Accuracy (%)', fontsize=44)
    ax.set_title('Class-wise Classification Performance', fontsize=64, fontweight='bold')
    ax.set_xticks(classes)
    ax.set_xticklabels(classes, fontsize=36)
    ax.set_ylim(max(0, min(class_acc)-10), 105)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=36, fontweight='bold')
    
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
    
    print("\n1. Creating feature vs accuracy plot...")
    fig, feature_order, mi_scores = create_feature_vs_accuracy_plot(features, labels)
    fig.savefig(os.path.join(output_dir, 'feature_vs_accuracy.png'), dpi=1200, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'feature_vs_accuracy.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print("\n2. Creating feature ranking plot...")
    fig = create_feature_ranking_plot(mi_scores)
    fig.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=1200, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'feature_importance.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print("\n3. Creating feature weightage plot...")
    fig = create_feature_weightage_plot(features, labels)
    fig.savefig(os.path.join(output_dir, 'feature_weightage.png'), dpi=1200, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'feature_weightage.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print("\n4. Creating ablation study plot...")
    fig = create_feature_group_accuracy_plot(features, labels)
    fig.savefig(os.path.join(output_dir, 'feature_group_accuracy.png'), dpi=1200, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'feature_group_accuracy.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print("\n5. Creating class-wise accuracy plot...")
    fig = create_class_wise_accuracy_plot(features, labels)
    fig.savefig(os.path.join(output_dir, 'class_wise_accuracy.png'), dpi=1200, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'class_wise_accuracy.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print("\nFeature vs accuracy analysis complete!")


if __name__ == '__main__':
    main()
