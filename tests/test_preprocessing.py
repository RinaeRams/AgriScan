#!/usr/bin/env python3
"""
Compare preprocessing methods ([0,1] vs [-1,1]) on the PROCESSED test split
(not raw PlantVillage data) and compute comprehensive metrics including
per-class precision, recall, F1-score, macro-averaged F1, and the confusion
matrix using scikit-learn.
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

print(f"Evaluating preprocessing on PROCESSED test split: {TEST_DIR}")
print(f"Number of classes: {len(class_names)}")
print("=" * 70)


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


def preprocess_01(img_path):
    """Preprocess to [0, 1] range (matches training script)."""
    image = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_mm11(img_path):
    """Preprocess to [-1, 1] range (some ML services use this)."""
    image = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(image, dtype=np.float32)

    arr_01 = img_array / 255.0
    arr_mm = (arr_01 - 0.5) * 2.0
    return np.expand_dims(arr_mm, axis=0)


def run_evaluation(image_paths, true_labels, preprocess_fn, label):
    """Run predictions and compute metrics for a given preprocessing method."""
    print(f"\n--- Running evaluation with {label} preprocessing ---")
    true_indices = [class_names.index(lbl) for lbl in true_labels]
    pred_indices = []

    for i in range(0, len(image_paths), 32):
        batch = image_paths[i:i + 32]
        images = np.concatenate([preprocess_fn(p) for p in batch], axis=0)
        preds = model.predict(images, verbose=0)
        pred_indices.extend(np.argmax(preds, axis=1))
        print(f"  Processed {min(i + 32, len(image_paths))}/{len(image_paths)} images")

    predicted_labels = [class_names[idx] for idx in pred_indices]
    accuracy = np.mean([t == p for t, p in zip(true_labels, predicted_labels)])

    cm = confusion_matrix(true_indices, pred_indices, labels=range(len(class_names)))
    report = classification_report(
        true_indices,
        pred_indices,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
        digits=4
    )
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

    print(f"\n[{label}] Accuracy: {accuracy * 100:.2f}%  |  Macro F1: {macro_f1:.4f}")
    print(f"[{label}] Macro Precision: {macro_precision:.4f}  |  Macro Recall: {macro_recall:.4f}")
    print(f"[{label}] Weighted F1: {weighted_f1:.4f}")

    print(f"\n[{label}] Classification Report:")
    print(report)

    # Per-class accuracy from confusion matrix
    print(f"\n[{label}] Per-class Accuracy:")
    for i, cn in enumerate(class_names):
        row_total = cm[i].sum()
        correct = cm[i, i]
        per_class_acc = correct / row_total if row_total > 0 else 0.0
        print(f"  {cn:45s}: {correct}/{row_total} = {per_class_acc * 100:.2f}%")

    return {
        "label": label,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }


print("\nCollecting test images from processed split...")
image_paths, true_labels, class_counts = load_test_images(TEST_DIR, class_names)

print("\nPer-class test image counts:")
for cn, count in class_counts.items():
    print(f"  {cn:45s}: {count}")

total = len(image_paths)
print(f"\nTotal test images: {total}")

if total == 0:
    raise RuntimeError("No test images found in the processed split directory.")

# Run both preprocessing methods
results_01 = run_evaluation(image_paths, true_labels, preprocess_01, "[0,1]")
results_mm11 = run_evaluation(image_paths, true_labels, preprocess_mm11, "[-1,1]")

# Comparison summary
print("\n" + "=" * 70)
print("=== Preprocessing Method Comparison ===")
print(f"{'Metric':<25s} {'[0,1]':>12s} {'[-1,1]':>12s} {'Winner':>12s}")
print("-" * 70)

comparison_rows = [
    ("Accuracy", results_01["accuracy"], results_mm11["accuracy"]),
    ("Macro Precision", results_01["macro_precision"], results_mm11["macro_precision"]),
    ("Macro Recall", results_01["macro_recall"], results_mm11["macro_recall"]),
    ("Macro F1", results_01["macro_f1"], results_mm11["macro_f1"]),
    ("Weighted F1", results_01["weighted_f1"], results_mm11["weighted_f1"]),
]

for metric_name, m01, mmm11 in comparison_rows:
    winner = "[0,1]" if m01 > mmm11 else ("[-1,1]" if mmm11 > m01 else "Tie")
    print(f"{metric_name:<25s} {m01:>12.4f} {mmm11:>12.4f} {winner:>12s}")

# Per-class comparison
print("\n=== Per-class F1 Score Comparison ===")
print(f"{'Class':45s} {'[0,1] F1':>10s} {'[-1,1] F1':>11s}")
for cn in class_names:
    r01 = results_01["classification_report"].get(cn, {})
    rmm = results_mm11["classification_report"].get(cn, {})
    f1_01 = r01.get('f1-score', 0.0)
    f1_mm = rmm.get('f1-score', 0.0)
    print(f"{cn:45s} {f1_01:>10.4f} {f1_mm:>11.4f}")

# Save comparison results
metrics_output = {
    "01_scaled": {k: v for k, v in results_01.items() if k != "classification_report" and k != "confusion_matrix"},
    "minus1_plus1": {k: v for k, v in results_mm11.items() if k != "classification_report" and k != "confusion_matrix"},
    "total_samples": total,
    "class_names": class_names,
}
metrics_path = BASE_DIR / "tests" / "preprocessing_comparison.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics_output, f, indent=2)
print(f"\nComparison metrics saved to {metrics_path}")
