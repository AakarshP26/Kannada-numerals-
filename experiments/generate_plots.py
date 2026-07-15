import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from knsd_features import extract_knsd_features, IMAGE_SIZE

def load_kannada_mnist(csv_path, max_samples_per_class=None):
    df = pd.read_csv(csv_path)
    if max_samples_per_class:
        df = df.groupby('label').apply(
            lambda x: x.sample(n=min(max_samples_per_class, len(x)), random_state=42)
        ).reset_index(drop=True)
    labels = df['label'].values
    pixels = df.drop('label', axis=1).values
    images = pixels.reshape(-1, 28, 28).astype(np.uint8)
    return images, labels

def load_custom_images(folder_path):
    images = []
    labels = []
    for digit in range(10):
        digit_folder = os.path.join(folder_path, str(digit))
        if not os.path.exists(digit_folder):
            continue
        for img_file in os.listdir(digit_folder):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(digit_folder, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                    if np.mean(img_resized) > 127:
                        img_resized = 255 - img_resized
                    images.append(img_resized)
                    labels.append(digit)
    return np.array(images, dtype=np.uint8), np.array(labels)

def augment_image(img):
    augmented = [img.copy()]
    # Rotation
    for angle in [-10, 10]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)
        augmented.append(rotated)
    # Scale
    for scale in [0.85, 1.15]:
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if scale < 1:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            result = np.zeros((h, w), dtype=np.uint8)
            result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled
        else:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            result = scaled[start_h:start_h+h, start_w:start_w+w]
        augmented.append(result)
    # Translation
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=0)
        augmented.append(shifted)
    return augmented

def augment_dataset(images, labels, augment_factor=3):
    aug_images = []
    aug_labels = []
    for img, label in zip(images, labels):
        augmented = augment_image(img)
        for aug_img in augmented[:augment_factor]:
            aug_images.append(aug_img)
            aug_labels.append(label)
    return np.array(aug_images, dtype=np.uint8), np.array(aug_labels)

def preprocess_mnist(img):
    resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)
    return binary

def extract_all_features(images):
    features = []
    for img in images:
        processed = preprocess_mnist(img)
        feat = extract_knsd_features(processed)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(feat)
    return np.array(features)

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'knsd_model.pkl')
    data_dir = os.path.join(base_dir, 'data', 'kannada_mnist')
    custom_data_dir = os.path.join(base_dir, 'data', 'dataset_custom')
    output_dir = os.path.join(base_dir, 'analysis', 'outputs_high_res_10thmarch')
    
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    data = joblib.load(model_path)
    model = data['model']
    scaler = data['scaler']

    print("Preparing test data to reproduce results...")
    # Load raw
    mnist_images, mnist_labels = load_kannada_mnist(os.path.join(data_dir, 'train.csv'), max_samples_per_class=600)
    custom_images, custom_labels = load_custom_images(custom_data_dir)

    # Augment
    mnist_aug, mnist_labels_aug = augment_dataset(mnist_images, mnist_labels, augment_factor=2)
    custom_aug, custom_labels_aug = augment_dataset(custom_images, custom_labels, augment_factor=5)

    # Combine
    all_images = np.concatenate([mnist_aug, custom_aug], axis=0)
    all_labels = np.concatenate([mnist_labels_aug, custom_labels_aug], axis=0)

    # Split (Same seed as training)
    _, X_test_img, _, y_test = train_test_split(
        all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    print(f"Extracting features for {len(X_test_img)} test images...")
    X_test_feats = extract_all_features(X_test_img)
    X_test_scaled = scaler.transform(X_test_feats)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")

    # 1. High-Res Confusion Matrix
    print("Generating High-Res Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    # Confusion Matrix
    print("Generating Confusion Matrix...")
    fig_cm, ax_cm = plt.subplots(figsize=(24, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', square=True, annot_kws={'size': 20})
    ax_cm.set_title(f'KNSD Confusion Matrix (Acc: {acc*100:.2f}%)', fontsize=32, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=1200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.pdf'), bbox_inches='tight')
    plt.close(fig_cm)

    # 2. Per-Class Metrics Chart
    print("Generating Per-Class Accuracy Chart...")
    report = classification_report(y_test, y_pred, output_dict=True)
    classes = [str(i) for i in range(10)]
    f1_scores = [report[c]['f1-score'] for c in classes]
    
    plt.xlabel('Kannada Digit', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.ylim(0, 1.1)
    
    # Add value labels
    # Class-wise Accuracy
    print("Generating Class-wise Performance...")
    fig_class, ax_class = plt.subplots(figsize=(24, 12))
    bars = ax_class.bar(range(10), class_acc, color=plt.cm.viridis(np.linspace(0, 1, 10)))
    ax_class.set_title('KNSD Per-Class Performance', fontsize=32, fontweight='bold')
    ax_class.set_xlabel('Kannada Digit', fontsize=24)
    ax_class.set_ylabel('Accuracy (%)', fontsize=24)
    ax_class.set_xticks(range(10))
    ax_class.set_ylim(0, 105)
    plt.savefig(os.path.join(output_dir, 'class_performance.png'), dpi=1200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'class_performance.pdf'), bbox_inches='tight')
    plt.close(fig_class)

    print(f"High-res figures saved to {output_dir}")

if __name__ == "__main__":
    main()
