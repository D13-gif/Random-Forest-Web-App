import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import pickle


IMG_SIZE = 64
DATASET_PATH = Path("dataset/Garbage classification")
MODEL_DIR = Path("model")
STATIC_DIR = Path("static")


def extract_features(img_path, img_size=64):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img_resized = cv2.resize(img, (img_size, img_size))

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    color_hist = np.concatenate([hist_h, hist_s, hist_v])
    color_hist /= (color_hist.sum() + 1e-7)
    cv2.normalize(color_hist, color_hist, 1.0, 0.0, cv2.NORM_L1)

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    hog_features = hog.compute(gray).flatten()

    gray_32 = cv2.resize(gray, (32, 32)).flatten()

    features = np.concatenate([color_hist, hog_features, gray_32]).astype(np.float32)
    return features


def load_dataset(dataset_path):
    features = []
    labels = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        label = class_dir.name
        img_files = [f for f in class_dir.iterdir()
                     if f.suffix.lower() in valid_exts]
        print(f"  {label}: {len(img_files)} images")

        for img_path in img_files:
            try:
                feat = extract_features(img_path, IMG_SIZE)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"  Skipping {img_path.name}: {e}")

    return np.array(features), np.array(labels)


def main():
    print("=" * 60)
    print("GARBAGE CLASSIFICATION - RANDOM FOREST TRAINING")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH.resolve()}. "
            "Please ensure the dataset folder exists."
        )

    print("\n[1/5] Loading dataset and extracting features...")
    X, y = load_dataset(DATASET_PATH)
    print(f"Total samples: {len(y)}")
    print(f"Feature vector size: {X.shape[1]}")

    print("\n[2/5] Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {list(le.classes_)}")

    print("\n[3/5] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    print("\n[4/5] Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    print("Training complete.")

    print("\n[5/5] Evaluating model...")
    y_pred = rf.predict(X_test)

    # ── Accuracy Score — eksplisit ────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  HASIL EVALUASI MODEL RANDOM FOREST")
    print(f"{'='*50}")
    print(f"  Akurasi       : {acc:.4f}")
    print(f"  Akurasi (%)   : {acc * 100:.2f}%")
    print(f"  Jumlah pohon  : {rf.n_estimators}")
    print(f"  Kedalaman max : {rf.max_depth}")
    print(f"{'='*50}")

    # Classification Report
    print(f"\nLaporan Klasifikasi Per Kelas:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - Garbage Classification (Random Forest)")
    plt.tight_layout()

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = STATIC_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "rf_model.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    print(f"\nModel saved to: {model_path}")
    print(f"Label encoder saved to: {encoder_path}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
