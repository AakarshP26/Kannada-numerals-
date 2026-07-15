"""
Classifier Analysis - Cross-validation, confusion matrix, and learning curves
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix, classification_report
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


def create_cv_fold_performance(features, labels):
    """Create 5-fold CV performance visualization."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    
    scores = cross_val_score(svm, X_scaled, labels, cv=cv, scoring='accuracy')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Bar plot
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))
    bars = ax1.bar(range(1, 6), scores * 100, color=colors, edgecolor='white', linewidth=2)
    ax1.axhline(y=scores.mean() * 100, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {scores.mean()*100:.2f}%')
    
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{score*100:.2f}%', ha='center', va='bottom', fontsize=22, fontweight='bold')
    
    ax1.set_xlabel('Fold Number', fontsize=120)
    ax1.set_ylabel('Accuracy (%)', fontsize=120)
    ax1.set_title('5-Fold Cross-Validation Performance', fontsize=140, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower right', fontsize=55)
    ax1.grid(axis='y', alpha=0.3)
    
    # Box plot
    ax2.boxplot([scores * 100], vert=True, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    ax2.scatter([1] * 5, scores * 100, s=100, c=colors, zorder=5, edgecolors='white')
    ax2.set_ylabel('Accuracy (%)', fontsize=120)
    ax2.set_title('Accuracy Distribution', fontsize=140, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['5-Fold CV'])
    ax2.grid(axis='y', alpha=0.3)
    
    # Stats annotation
    stats_text = f'Mean: {scores.mean()*100:.2f}%\nStd: {scores.std()*100:.2f}%\nMin: {scores.min()*100:.2f}%\nMax: {scores.max()*100:.2f}%'
    ax2.text(1.3, 50, stats_text, fontsize=100, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('5-Fold Stratified Cross-Validation Analysis', fontsize=160, fontweight='bold', y=1.05)
    plt.tight_layout()
    return fig


def create_enhanced_confusion_matrix(features, labels):
    """Create enhanced confusion matrix with per-class accuracy."""
    from sklearn.model_selection import train_test_split
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 14))
    
    # Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=range(10), yticklabels=range(10), annot_kws={'size': 80})
    ax1.set_xlabel('Predicted Digit', fontsize=120)
    ax1.set_ylabel('True Digit', fontsize=120)
    ax1.set_title('Confusion Matrix', fontsize=140, fontweight='bold')
    
    # Per-class accuracy bar chart
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    colors = plt.cm.RdYlGn(per_class_acc / 100)
    
    bars = ax2.barh(range(10), per_class_acc, color=colors, edgecolor='white', linewidth=2)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(range(10))
    ax2.set_xlabel('Accuracy (%)', fontsize=120)
    ax2.set_ylabel('Digit', fontsize=120)
    ax2.set_title('Per-Class Accuracy', fontsize=140, fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=per_class_acc.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {per_class_acc.mean():.1f}%')
    ax2.legend(loc='lower right', fontsize=55)
    
    for i, (acc, bar) in enumerate(zip(per_class_acc, bars)):
        ax2.text(acc + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=100, fontweight='bold')
    
    plt.suptitle('KNSD Classification Performance Analysis', fontsize=160, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_learning_curve(features, labels):
    """Create learning curve showing performance vs training samples."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    
    train_sizes = np.linspace(0.2, 1.0, 5)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        svm, X_scaled, labels, train_sizes=train_sizes,
        cv=5, scoring='accuracy', n_jobs=-1, random_state=42
    )
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    train_mean = train_scores.mean(axis=1) * 100
    train_std = train_scores.std(axis=1) * 100
    test_mean = test_scores.mean(axis=1) * 100
    test_std = test_scores.std(axis=1) * 100
    
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#3498db', linewidth=2,
            label='Training Score', markersize=8)
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='#3498db')
    
    ax.plot(train_sizes_abs, test_mean, 'o-', color='#e74c3c', linewidth=2,
            label='Cross-validation Score', markersize=8)
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('Training Samples', fontsize=120)
    ax.set_ylabel('Accuracy (%)', fontsize=120)
    ax.set_title('KNSD Learning Curve', fontsize=160, fontweight='bold')
    ax.set_ylim(50, 105)
    ax.legend(loc='lower right', fontsize=110)
    ax.grid(True, alpha=0.3)
    
    # Add annotation about convergence
    ax.annotate(f'Final CV: {test_mean[-1]:.1f}%', xy=(train_sizes_abs[-1], test_mean[-1]),
                xytext=(train_sizes_abs[-1] * 0.85, test_mean[-1] - 8),
                fontsize=22, arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    return fig


def create_classifier_comparison(features, labels):
    """Compare different classifiers on KNSD features."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    classifiers = {
        'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'SVM (Linear)': SVC(kernel='linear', C=1, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Logistic Reg.': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nEvaluating classifiers...")
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, labels, cv=cv, scoring='accuracy')
        results[name] = (scores.mean() * 100, scores.std() * 100)
        print(f"  {name}: {scores.mean()*100:.2f}% (±{scores.std()*100:.2f}%)")
    
    fig, ax = plt.subplots(figsize=(24, 14))
    
    names = list(results.keys())
    accs = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    
    # Sort by accuracy
    sorted_idx = np.argsort(accs)[::-1]
    names = [names[i] for i in sorted_idx]
    accs = [accs[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(names)))
    
    bars = ax.barh(range(len(names)), accs, xerr=stds, capsize=5, color=colors,
                   edgecolor='white', linewidth=2)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('5-Fold Cross-Validation Accuracy (%)', fontsize=60)
    ax.set_title('Classifier Comparison on KNSD Features', fontsize=80, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)
    
    # Highlight SVM (RBF)
    for i, name in enumerate(names):
        if 'RBF' in name:
            bars[i].set_edgecolor('gold')
            bars[i].set_linewidth(4)
    
    for i, (acc, std) in enumerate(zip(accs, stds)):
        ax.text(acc + std + 1, i, f'{acc:.1f}%', va='center', fontsize=50, fontweight='bold')
    
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
    
    print("\n1. Creating 5-fold CV performance plot...")
    fig = create_cv_fold_performance(features, labels)
    fig.savefig(os.path.join(output_dir, 'cv_fold_performance.png'), dpi=1200, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'cv_fold_performance.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: cv_fold_performance.png (+ pdf)")
    
    print("\n2. Creating enhanced confusion matrix...")
    fig = create_enhanced_confusion_matrix(features, labels)
    fig.savefig(os.path.join(output_dir, 'confusion_matrix_enhanced.png'), dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'confusion_matrix_enhanced.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: confusion_matrix_enhanced.png (+ pdf)")
    
    print("\n3. Creating learning curve...")
    fig = create_learning_curve(features, labels)
    fig.savefig(os.path.join(output_dir, 'learning_curve.png'), dpi=1200, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'learning_curve.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: learning_curve.png (+ pdf)")
    
    print("\n4. Creating classifier comparison...")
    fig = create_classifier_comparison(features, labels)
    fig.savefig(os.path.join(output_dir, 'classifier_comparison.png'), dpi=1200, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'classifier_comparison.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("   Saved: classifier_comparison.png (+ pdf)")
    
    print("\nClassifier analysis complete!")


if __name__ == '__main__':
    main()
