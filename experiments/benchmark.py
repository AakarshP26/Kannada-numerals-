
"""
Run Official Benchmark and Inference Latency Testing
"""
import os
import sys
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from knsd_features import extract_knsd_features
from train_evaluate import preprocess_mnist

def load_dataset(csv_path):
    """Load dataset from CSV."""
    print(f"Loading {os.path.basename(csv_path)}...")
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return None, None
        
    df = pd.read_csv(csv_path)
    
    # Check if 'label' column exists (test.csv might not have it)
    if 'label' in df.columns:
        labels = df['label'].values
        pixels = df.drop('label', axis=1).values
    elif 'id' in df.columns: # official test.csv has id, pixel0...
        labels = None # Unlabeled
        pixels = df.drop('id', axis=1).values
    else:
        labels = None
        pixels = df.values
        
    images = pixels.reshape(-1, 28, 28).astype(np.uint8)
    print(f"  Loaded {len(images)} images")
    return images, labels

def evaluate_on_dataset(model, scaler, images, labels, name="Dataset"):
    """Evaluate model on a dataset."""
    print(f"\nEvaluating on {name} ({len(images)} samples)...")
    
    # Extract features
    features = []
    print("  Extracting features...")
    for img in tqdm(images):
        processed = preprocess_mnist(img)
        feat = extract_knsd_features(processed)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(feat)
    
    X = np.array(features)
    X_scaled = scaler.transform(X)
    
    # Predict
    y_pred = model.predict(X_scaled)
    
    # Report
    if labels is not None:
        acc = accuracy_score(labels, y_pred)
        print(f"  Accuracy: {acc*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(labels, y_pred, digits=4))
        return acc
    else:
        print("  (Unlabeled dataset - saved predictions)")
        return y_pred

def latency_test(model, scaler, images, n_samples=100):
    """Measure per-image inference latency on a random subsample.
    
    Reports feature extraction time and SVM inference time separately,
    as well as total end-to-end latency per image.
    """
    print(f"\n{'='*60}")
    print(f"INFERENCE LATENCY TEST ({n_samples} images)")
    print(f"{'='*60}")
    
    # Random subsample
    n_samples = min(n_samples, len(images))
    indices = np.random.choice(len(images), n_samples, replace=False)
    sample_images = images[indices]
    
    feat_times = []
    infer_times = []
    total_times = []
    
    # Warm-up run (exclude from timing)
    warmup_img = preprocess_mnist(sample_images[0])
    warmup_feat = extract_knsd_features(warmup_img)
    warmup_feat = np.nan_to_num(warmup_feat, nan=0.0, posinf=0.0, neginf=0.0)
    scaler.transform([warmup_feat])
    model.predict(scaler.transform([warmup_feat]))
    
    print(f"  Measuring {n_samples} images...")
    for img in tqdm(sample_images, desc="  Latency test"):
        # --- Feature extraction timing ---
        t0 = time.perf_counter()
        processed = preprocess_mnist(img)
        feat = extract_knsd_features(processed)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        t1 = time.perf_counter()
        feat_times.append(t1 - t0)
        
        # --- SVM inference timing ---
        t2 = time.perf_counter()
        X_scaled = scaler.transform([feat])
        _ = model.predict(X_scaled)
        t3 = time.perf_counter()
        infer_times.append(t3 - t2)
        
        total_times.append(t3 - t0)
    
    feat_times = np.array(feat_times) * 1000   # Convert to ms
    infer_times = np.array(infer_times) * 1000
    total_times = np.array(total_times) * 1000
    
    # Build report
    report_lines = [
        "KNSD Inference Latency Report",
        "=" * 40,
        f"Sample size: {n_samples} images",
        f"Platform: {sys.platform}",
        "",
        "Per-Image Latency (milliseconds):",
        "-" * 40,
        f"  Feature Extraction:  {feat_times.mean():.2f} ± {feat_times.std():.2f} ms",
        f"  SVM Inference:       {infer_times.mean():.2f} ± {infer_times.std():.2f} ms",
        f"  Total (end-to-end):  {total_times.mean():.2f} ± {total_times.std():.2f} ms",
        "",
        "Percentiles (Total, ms):",
        f"  P50 (median):  {np.percentile(total_times, 50):.2f} ms",
        f"  P95:           {np.percentile(total_times, 95):.2f} ms",
        f"  P99:           {np.percentile(total_times, 99):.2f} ms",
        f"  Max:           {total_times.max():.2f} ms",
        "",
        f"Throughput: ~{1000.0 / total_times.mean():.0f} images/sec",
    ]
    
    report = "\n".join(report_lines)
    print(f"\n{report}")
    
    return report


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'knsd_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    data_dir = os.path.join(base_dir, 'data', 'kannada_mnist')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model
    if not os.path.exists(model_path):
        print("Model not found! Please run train_evaluate.py first.")
        return
        
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model loaded successfully.")
    
    # 1. Evaluate on Dig-MNIST (Hard Test)
    dig_path = os.path.join(data_dir, 'Dig-MNIST.csv')
    if os.path.exists(dig_path):
        images, labels = load_dataset(dig_path)
        evaluate_on_dataset(model, scaler, images, labels, "Dig-MNIST (Hard Generalization)")
    
    # 2. Evaluate on test.csv (Kaggle Test)
    test_path = os.path.join(data_dir, 'test.csv')
    if os.path.exists(test_path):
        images, labels = load_dataset(test_path)
        if labels is not None:
            evaluate_on_dataset(model, scaler, images, labels, "Official Test Set")
        else:
            print(f"\nEvaluating on test.csv (Unlabeled)...")
    
    # 3. Inference Latency Test (on train.csv subsample)
    train_path = os.path.join(data_dir, 'train.csv')
    if os.path.exists(train_path):
        images, labels = load_dataset(train_path)
        if images is not None:
            report = latency_test(model, scaler, images, n_samples=100)
            
            # Save latency report
            report_path = os.path.join(results_dir, 'latency_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\nLatency report saved: {report_path}")
    else:
        print("\nWarning: train.csv not found, skipping latency test.")


if __name__ == '__main__':
    main()
