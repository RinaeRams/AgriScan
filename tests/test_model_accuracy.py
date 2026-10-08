#!/usr/bin/env python3
"""
Test model accuracy on the held-out PROCESSED test split (not the raw
PlantVillage dataset) and compute comprehensive metrics including
per-class precision, recall, F1-score, macro-averaged F1, and the
confusion matrix using scikit-learn.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "data" / "processed" / "split" / "test"
MODEL_DIR = BASE_DIR / "app" / "models"

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG')

with open(MODEL_DIR / "class_names.json", 'r') as f:
    class_names = json.load(f)

model = keras.models.load_model(str(MODEL_DIR / "agriscan_model.h5"))

print(f"Evaluating on PROCESSED test split: {TEST_DIR}")
print(f"Number of classes: {len(class_names)}")
print("=" * 70)


def is_bcrypt_hash(value):
    """Detect whether a string looks like a bcrypt hash."""
    return bool(value) and (value.startswith('$2a$') or
                            value.startswith('$2b$') or
                            value.startswith('$2y$'))


def load_test_images(test_dir, class_names):
    """Walk the test split directory and collect (image_path, true_label) pairs."""
    image_paths = []
    true_labels = []
    class_counts = {}

    for class_name in class_names:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"  WARNING: class '{class_name}' missing from test split")
            class_counts[class_name] = 0
            continue
        images = sorted([f for f in class_dir.iterdir()
                         if f.suffix in IMAGE_EXTENSIONS and f.is_file()])
        class_counts[class_name] = len(images)
        for img_path in images:
            image_paths.append(str(img_path))
            true_labels.append(class_name)

    return image_paths, true_labels, class_counts


def preprocess_image(img_path):
    """Load and preprocess a single image to match training preprocessing [0, 1]."""
    image = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_batch(image_paths, batch_size=32):
    """Run predictions on images in batches to manage memory."""
    all_preds = []
    all_probs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch = np.concatenate([preprocess_image(p) for p in batch_paths], axis=0)
        preds = model.predict(batch, verbose=0)
        pred_indices = np.argmax(preds, axis=1)
        all_preds.extend(pred_indices)
        all_probs.extend(preds.max(axis=1))
        print(f"  Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
    return all_preds, all_probs


print("\nCollecting test images from processed split...")
image_paths, true_labels, class_counts = load_test_images(TEST_DIR, class_names)

print("\nPer-class test image counts:")
for cn, count in class_counts.items():
    print(f"  {cn:45s}: {count}")

total = len(image_paths)
print(f"\nTotal test images: {total}")

if total == 0:
    raise RuntimeError("No test images found in the processed split directory.")

print("\nRunning predictions...")
pred_indices, confidences = predict_batch(image_paths)
predicted_labels = [class_names[idx] for idx in pred_indices]


true_indices = [class_names.index(label) for label in true_labels]

accuracy = np.mean([t == p for t, p in zip(predicted_labels, true_labels)])
print(f"\n{'=' * 70}")
print(f"Overall Accuracy: {accuracy * 100:.2f}% ({np.sum([t == p for t, p in zip(predicted_labels, true_labels)])}/{total})")
print(f"{'=' * 70}")

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(true_indices, pred_indices, labels=range(len(class_names)))
print(f"Matrix shape: {cm.shape}")
print(f"Total samples: {cm.sum()}")

print("\n=== Per-class Confusion Matrix ===")
header = " " * 45 + "".join([f"{cn[:8]:>10s}" for cn in class_names])
print(header)
for i, cn in enumerate(class_names):
    row = f"{cn[:45]:45s}" + "".join([f"{cm[i, j]:>10d}" for j in range(len(class_names))])
    print(row)

print("\n=== Classification Report (per-class precision, recall, F1) ===")
report = classification_report(
    true_indices,
    pred_indices,
    labels=range(len(class_names)),
    target_names=class_names,
    zero_division=0,
    digits=4
)
print(report)

report_dict = classification_report(
    true_indices,
    pred_indices,
    labels=range(len(class_names)),
    target_names=class_names,
    zero_division=0,
    output_dict=True
)

macro_f1 = report_dict['macro avg']['f1-score']
macro_precision = report_dict['macro avg']['precision']
macro_recall = report_dict['macro avg']['recall']
weighted_f1 = report_dict['weighted avg']['f1-score']

print("\n=== Summary ===")
print(f"Macro-averaged Precision: {macro_precision:.4f}")
print(f"Macro-averaged Recall:    {macro_recall:.4f}")
print(f"Macro-averaged F1-score:  {macro_f1:.4f}")
print(f"Weighted-averaged F1:     {weighted_f1:.4f}")
print(f"Overall Accuracy:         {accuracy * 100:.2f}%")

# Detailed per-class metrics
print("\n=== Per-class Metrics Summary ===")
print(f"{'Class':45s} {'Precision':>10s} {'Recall':>10s} {'F1-score':>10s} {'Support':>10s}")
for cn in class_names:
    if cn in report_dict:
        metrics = report_dict[cn]
        print(f"{cn:45s} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1-score']:>10.4f} {int(metrics['support']):>10d}")

# Per-class accuracy from confusion matrix
print("\n=== Per-class Accuracy (from confusion matrix diagonal) ===")
for i, cn in enumerate(class_names):
    row_total = cm[i].sum()
    correct = cm[i, i]
    per_class_acc = correct / row_total if row_total > 0 else 0.0
    print(f"  {cn:45s}: {correct}/{row_total} = {per_class_acc * 100:.2f}%")

# Prediction distribution
print("\n=== Prediction Distribution ===")
pred_counts = {cn: 0 for cn in class_names}
for pred in predicted_labels:
    pred_counts[pred] += 1
for cn, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
    print(f"  {cn[:45]:45s}: {count}")

# Save metrics to JSON
metrics_output = {
    "accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "total_samples": total,
    "per_class": {cn: report_dict[cn] for cn in class_names if cn in report_dict},
    "confusion_matrix": cm.tolist(),
    "class_names": class_names,
}
metrics_path = BASE_DIR / "tests" / "test_results.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics_output, f, indent=2)
print(f"\nMetrics saved to {metrics_path}")
