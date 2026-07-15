"""
Schematic Diagram Generator for KNSD Pipeline
Generates high-quality pipeline visualization using matplotlib
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

def create_knsd_schematic():
    """Create a professional schematic diagram of the KNSD pipeline."""
    fig, ax = plt.subplots(1, 1, figsize=(32, 24))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#3498db',
        'preprocess': '#9b59b6',
        'feature': '#2ecc71',
        'classify': '#e74c3c',
        'output': '#f39c12',
        'arrow': '#34495e',
        'text': '#2c3e50',
        'highlight': '#1abc9c'
    }
    
    # Title
    ax.text(8, 11.5, 'KNSD: Kannada Numeral Shape Descriptor Pipeline',
            fontsize=72, fontweight='bold', ha='center', color=colors['text'])
    ax.text(8, 11, 'Novel Loop-Based Feature Extraction for Handwritten Numeral Recognition',
            fontsize=48, ha='center', color='gray', style='italic')
    
    # Main process boxes
    def draw_box(ax, x, y, w, h, label, sublabel, color, fontsize=56):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.3",
                             facecolor=color, edgecolor='white', linewidth=6, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + 0.15, label, fontsize=fontsize, fontweight='bold',
                ha='center', va='center', color='white')
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25, sublabel, fontsize=36,
                    ha='center', va='center', color='white', alpha=0.9)
    
    # Draw main pipeline
    y_main = 8.5
    box_h = 1.2
    
    # Input
    draw_box(ax, 0.5, y_main, 2, box_h, 'INPUT', '28×28 Grayscale', colors['input'])
    
    # Preprocessing
    draw_box(ax, 3.5, y_main, 2.5, box_h, 'PREPROCESSING', 'Binarization & Normalization', colors['preprocess'])
    
    # Feature Extraction (larger)
    draw_box(ax, 7, y_main, 3, box_h, 'FEATURE EXTRACTION', 'KNSD (29 Features)', colors['feature'])
    
    # Classification
    draw_box(ax, 11, y_main, 2.2, box_h, 'CLASSIFICATION', 'SVM (RBF Kernel)', colors['classify'])
    
    # Output
    draw_box(ax, 13.8, y_main, 1.7, box_h, 'OUTPUT', 'Digit (0-9)', colors['output'])
    
    # Arrows between main boxes
    arrow_y = y_main + box_h/2
    arrow_props = dict(arrowstyle='->', color=colors['arrow'], lw=2, mutation_scale=20)
    ax.annotate('', xy=(3.4, arrow_y), xytext=(2.6, arrow_y), arrowprops=arrow_props)
    ax.annotate('', xy=(6.9, arrow_y), xytext=(6.1, arrow_y), arrowprops=arrow_props)
    ax.annotate('', xy=(10.9, arrow_y), xytext=(10.1, arrow_y), arrowprops=arrow_props)
    ax.annotate('', xy=(13.7, arrow_y), xytext=(13.3, arrow_y), arrowprops=arrow_props)
    
    # Feature extraction breakdown
    feat_y = 5.5
    feat_h = 1.5
    feat_colors = ['#27ae60', '#16a085', '#1abc9c', '#2980b9', '#8e44ad']
    feat_labels = [
        ('LOOP\nFEATURES', '8 features', 'Loop count, area\nposition, variance'),
        ('ENDPOINT\nFEATURES', '5 features', 'Count, positions\nspread'),
        ('JUNCTION\nFEATURES', '3 features', 'Intersection\npoints'),
        ('CURVATURE\nHISTOGRAM', '8 features', 'Directional\ndistribution'),
        ('STROKE\nSTATISTICS', '5 features', 'Shape metrics\ncircularity')
    ]
    
    feat_w = 2.2
    feat_start = 1.5
    feat_gap = 2.6
    
    for i, (label, count, desc) in enumerate(feat_labels):
        x = feat_start + i * feat_gap
        box = FancyBboxPatch((x, feat_y), feat_w, feat_h, 
                             boxstyle="round,pad=0.02,rounding_size=0.2",
                             facecolor=feat_colors[i], edgecolor='white', linewidth=4, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + feat_w/2, feat_y + feat_h - 0.35, label, fontsize=40, fontweight='bold',
                ha='center', va='center', color='white')
        ax.text(x + feat_w/2, feat_y + 0.55, count, fontsize=36,
                ha='center', va='center', color='white', alpha=0.95,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.2, edgecolor='none'))
        ax.text(x + feat_w/2, feat_y + 0.15, desc, fontsize=28,
                ha='center', va='center', color='white', alpha=0.85)
    
    # Arrow from main feature box to breakdown
    ax.annotate('', xy=(8.5, feat_y + feat_h + 0.1), xytext=(8.5, y_main - 0.1),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2, mutation_scale=15))
    
    # Preprocessing details box
    prep_y = 2.5
    prep_box = FancyBboxPatch((0.5, prep_y), 4, 2.2, 
                               boxstyle="round,pad=0.03,rounding_size=0.2",
                               facecolor='#f8f9fa', edgecolor=colors['preprocess'], 
                               linewidth=4, alpha=0.9)
    ax.add_patch(prep_box)
    ax.text(2.5, prep_y + 1.9, 'Preprocessing Steps', fontsize=44, fontweight='bold',
            ha='center', color=colors['preprocess'])
    steps = ['1. Grayscale conversion', '2. Adaptive thresholding', 
             '3. Morphological noise removal', '4. Bounding box extraction',
             '5. Square padding & centering', '6. Resize to 100×100']
    for i, step in enumerate(steps):
        ax.text(0.8, prep_y + 1.5 - i*0.25, step, fontsize=32, ha='left', color=colors['text'])
    
    # Classification details box
    class_y = 2.5
    class_box = FancyBboxPatch((5.5, class_y), 4.5, 2.2,
                                boxstyle="round,pad=0.03,rounding_size=0.2",
                                facecolor='#f8f9fa', edgecolor=colors['classify'],
                                linewidth=4, alpha=0.9)
    ax.add_patch(class_box)
    ax.text(7.75, class_y + 1.9, 'SVM Configuration', fontsize=44, fontweight='bold',
            ha='center', color=colors['classify'])
    svm_details = ['• Kernel: RBF (Radial Basis Function)',
                   '• C parameter: 10 (regularization)',
                   '• Gamma: scale (auto-computed)',
                   '• Feature scaling: StandardScaler',
                   '• Validation: 5-Fold Stratified CV']
    for i, detail in enumerate(svm_details):
        ax.text(5.8, class_y + 1.5 - i*0.27, detail, fontsize=32, ha='left', color=colors['text'])
    
    # Novel contribution highlight
    novel_box = FancyBboxPatch((10.8, class_y), 4.7, 2.2,
                                boxstyle="round,pad=0.03,rounding_size=0.2",
                                facecolor='#fff3e0', edgecolor=colors['highlight'],
                                linewidth=4, alpha=0.9)
    ax.add_patch(novel_box)
    ax.text(13.15, class_y + 1.9, '★ Novel Contributions', fontsize=44, fontweight='bold',
            ha='center', color='#e65100')
    novel_points = ['• Loop-based features capture Kannada',
                    '  numeral topology (0, 6, 8, 9 have loops)',
                    '• Endpoint & junction features for',
                    '  stroke structure analysis',
                    '• Language-specific shape descriptors',
                    '• Robust to writing style variations']
    for i, point in enumerate(novel_points):
        ax.text(11.1, class_y + 1.5 - i*0.25, point, fontsize=32, ha='left', color=colors['text'])
    
    # Legend for feature count
    ax.text(8, 1.2, 'Total: 29 KNSD Features', fontsize=56, fontweight='bold',
            ha='center', color=colors['text'],
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    return fig


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs_high_res_10thmarch')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating KNSD pipeline schematic...")
    fig = create_knsd_schematic()
    
    output_path = os.path.join(output_dir, 'knsd_pipeline_schematic.png')
    fig.savefig(output_path, dpi=1200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    # Save PDF
    pdf_path = output_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close(fig)

    
if __name__ == '__main__':
    main()
