"""
Train and Evaluate KNSD Classifier on Kannada-MNIST
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from knsd_features import preprocess_image, extract_knsd_features, IMAGE_SIZE, FEATURE_NAMES


def load_kannada_mnist(csv_path, max_samples_per_class=None):
    """Load Kannada-MNIST from CSV."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if max_samples_per_class:
        df = df.groupby('label').apply(
            lambda x: x.sample(n=min(max_samples_per_class, len(x)), random_state=42)
        ).reset_index(drop=True)
    
    labels = df['label'].values
    pixels = df.drop('label', axis=1).values
    images = pixels.reshape(-1, 28, 28).astype(np.uint8)
    
    print(f"  Loaded {len(images)} images, {len(np.unique(labels))} classes")
    return images, labels


def load_custom_images(folder_path):
    """Load custom images from folder structure (digit folders 0-9)."""
    print(f"Loading custom images from {folder_path}...")
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
                    # Resize to 28x28 to match MNIST
                    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                    # Invert if background is white (MNIST has black background)
                    if np.mean(img_resized) > 127:
                        img_resized = 255 - img_resized
                    images.append(img_resized)
                    labels.append(digit)
    
    print(f"  Loaded {len(images)} custom images, {len(set(labels))} classes")
    return np.array(images, dtype=np.uint8), np.array(labels)


def augment_image(img):
    """Apply random augmentations to an image."""
    augmented = []
    
    # Original
    augmented.append(img.copy())
    
    # Rotation variations (-15 to +15 degrees)
    for angle in [-10, 10]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)
        augmented.append(rotated)
    
    # Scale variations
    for scale in [0.85, 1.15]:
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Pad or crop to original size
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
    
    # Translation (shift)
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=0)
        augmented.append(shifted)
    
    return augmented


def augment_dataset(images, labels, augment_factor=3):
    """Augment entire dataset."""
    print(f"Augmenting dataset (factor={augment_factor})...")
    aug_images = []
    aug_labels = []
    
    for img, label in zip(images, labels):
        augmented = augment_image(img)
        # Take up to augment_factor augmentations (including original)
        for aug_img in augmented[:augment_factor]:
            aug_images.append(aug_img)
            aug_labels.append(label)
    
    print(f"  Augmented: {len(images)} -> {len(aug_images)} images")
    return np.array(aug_images, dtype=np.uint8), np.array(aug_labels)


def preprocess_mnist(img):
    """Preprocess 28x28 MNIST image."""
    resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)
    return binary


def extract_all_features(images, labels):
    """Extract KNSD features from all images."""
    print("Extracting KNSD features...")
    features = []
    
    for img in tqdm(images):
        processed = preprocess_mnist(img)
        feat = extract_knsd_features(processed)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(feat)
    
    return np.array(features), labels


def train_evaluate(X_train, y_train, X_test, y_test):
    """Train SVM and evaluate."""
    print("\nTraining SVM classifier...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
    
    cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Train on full training set
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = svm.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    return svm, scaler, test_acc, confusion_matrix(y_test, y_pred)


def plot_results(cm, save_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('KNSD Classifier - Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.tight_layout()
    
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")


def main():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data', 'kannada_mnist')
    custom_data_dir = os.path.join(base_dir, 'data', 'dataset_custom')
    model_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load FULL Kannada-MNIST data (60k samples - no subsampling)
    train_path = os.path.join(data_dir, 'train.csv')
    mnist_images, mnist_labels = load_kannada_mnist(train_path, max_samples_per_class=None)
    
    # Load custom images
    custom_images, custom_labels = load_custom_images(custom_data_dir)
    
    # Combine RAW datasets FIRST
    print("\nCombining raw datasets...")
    all_images = np.concatenate([mnist_images, custom_images], axis=0)
    all_labels = np.concatenate([mnist_labels, custom_labels], axis=0)
    print(f"  Total raw: {len(all_images)} images")
    
    # SPLIT FIRST (before augmentation) - prevents data leakage!
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    print(f"Split: Train={len(train_imgs)}, Test={len(test_imgs)}")
    
    # Augment TRAINING SET ONLY (test set stays pristine)
    print("\n--- Data Augmentation (Training Only) ---")
    train_imgs_aug, train_labels_aug = augment_dataset(train_imgs, train_labels, augment_factor=2)
    print(f"Train after augmentation: {len(train_imgs_aug)}")
    
    # Use augmented training, raw testing
    train_imgs, train_labels = train_imgs_aug, train_labels_aug
    
    # Extract features
    X_train, y_train = extract_all_features(train_imgs, train_labels)
    X_test, y_test = extract_all_features(test_imgs, test_labels)
    
    # Train and evaluate
    model, scaler, accuracy, cm = train_evaluate(X_train, y_train, X_test, y_test)
    
    # Save model AND scaler
    model_path = os.path.join(model_dir, 'knsd_model.pkl')
    joblib.dump(model, model_path)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print(f"\nModel saved: {model_path}")
    print(f"Scaler saved: {os.path.join(model_dir, 'scaler.pkl')}")
    
    # Plot results
    plot_results(cm, results_dir)
    
    print("\n" + "="*50)
    print(f"KNSD Classifier: {accuracy*100:.2f}% accuracy")
    print(f"Features: {X_train.shape[1]}")
    print("="*50)


if __name__ == '__main__':
    main()
