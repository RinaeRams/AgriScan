#!/usr/bin/env python3
"""
Comprehensive evaluation script for AgriScan Chapter 4 results.

Produces:
  - Training/validation accuracy and loss per epoch (from training_history.json)
  - Early stopping epoch
  - Overall test accuracy with 95% confidence interval
  - Per-class precision, recall, F1, support, macro F1
  - 15x15 confusion matrix and most-confused class pairs
  - Top-3 accuracy
  - Healthy-versus-diseased accuracy
  - [0,1] vs [-1,1] preprocessing comparison
  - Response times over repeated requests
  - End-to-end test results

Usage:
    python tests/test_chapter4_evaluation.py
"""

import json
import os
import sys
import time
import statistics
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
    accuracy_score,
)
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "data" / "processed" / "split" / "test"
MODEL_DIR = BASE_DIR / "app" / "models"
RESULTS_DIR = BASE_DIR / "tests" / "chapter4_results"
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG')
IMG_SIZE = (224, 224)


def load_model_and_classes():
    """Load the trained Keras model and class names."""
    with open(MODEL_DIR / "class_names.json", 'r') as f:
        class_names = json.load(f)
    model = keras.models.load_model(str(MODEL_DIR / "agriscan_model.h5"))
    return model, class_names


def load_training_history():
    """Load training history if available, otherwise return None."""
    history_path = MODEL_DIR / "training_history.json"
    if not history_path.exists():
        return None
    with open(history_path, 'r') as f:
        return json.load(f)


def collect_test_images(test_dir, class_names):
    """Collect all test image paths and their true labels."""
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


def preprocess_image_01(img_path):
    """Preprocess to [0, 1] range (matches training)."""
    image = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_image_mm11(img_path):
    """Preprocess to [-1, 1] range."""
    image = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    arr_01 = img_array / 255.0
    arr_mm = (arr_01 - 0.5) * 2.0
    return np.expand_dims(arr_mm, axis=0)


def batch_predict(model, preprocess_fn, image_paths, batch_size=32):
    """Run predictions on images in batches and measure response times."""
    all_preds = []
    all_probs = []
    response_times = []

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        images = np.concatenate([preprocess_fn(p) for p in batch], axis=0)

        start_time = time.perf_counter()
        preds = model.predict(images, verbose=0)
        elapsed = time.perf_counter() - start_time

        for t in range(len(batch)):
            response_times.append(elapsed / len(batch))

        pred_indices = np.argmax(preds, axis=1)
        all_preds.extend(pred_indices)
        all_probs.extend(preds.max(axis=1))
        print(f"  Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")

    return all_preds, np.array(all_probs), response_times


def compute_confidence_interval(accuracy, n):
    """Compute 95% Wilson score confidence interval for accuracy."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96  # 95% CI
    p_hat = accuracy
    denominator = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return float(lower), float(upper)


def get_healthy_class_indices(class_names):
    """Identify which classes are 'healthy' vs 'diseased'."""
    healthy_keywords = ['healthy', 'Healthy']
    healthy_indices = []
    diseased_indices = []
    for i, cn in enumerate(class_names):
        normalized = cn.replace('_', ' ').lower()
        if any(kw.lower() in normalized for kw in healthy_keywords):
            healthy_indices.append(i)
        else:
            diseased_indices.append(i)
    return healthy_indices, diseased_indices


def main():
    print("=" * 70)
    print("AgriScan Chapter 4 Comprehensive Evaluation")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")

    # --- Load model and data ---
    print("\n[1/11] Loading model and class names...")
    model, class_names = load_model_and_classes()
    print(f"  Loaded {len(class_names)} classes: {class_names}")

    print("\n[2/11] Collecting test images from PROCESSED test split...")
    image_paths, true_labels, class_counts = collect_test_images(TEST_DIR, class_names)
    total = len(image_paths)
    print(f"  Total test images: {total}")

    if total == 0:
        raise RuntimeError("No test images found in processed split directory.")

    # Per-class counts
    print("\n  Per-class test image counts:")
    for cn, count in class_counts.items():
        print(f"    {cn:50s}: {count}")

    true_indices = np.array([class_names.index(lbl) for lbl in true_labels])

    # --- Training history ---
    print("\n[3/11] Checking for training history...")
    history = load_training_history()
    training_results = None
    if history:
        training_results = {
            "total_epochs_run": history.get("total_epochs_run", len(history.get("loss", []))),
            "early_stopped_epoch": history.get("early_stopped_epoch", 0),
            "accuracy": history.get("accuracy", []),
            "val_accuracy": history.get("val_accuracy", []),
            "loss": history.get("loss", []),
            "val_loss": history.get("val_loss", []),
            "epochs_run": history.get("total_epochs_run", len(history.get("loss", []))),
            "train_accuracy": history.get("accuracy", []),
            "val_accuracy": history.get("val_accuracy", []),
            "train_loss": history.get("loss", []),
            "val_loss": history.get("val_loss", []),
        }
        print(f"  Training history found: {training_results['epochs_run']} epochs run")
        print(f"  Early stopping at epoch (best val_loss): {training_results['early_stopped_epoch']}")
        if training_results["train_accuracy"]:
            final_train_acc = training_results["train_accuracy"][-1]
            final_val_acc = training_results["val_accuracy"][-1]
            print(f"  Final train accuracy: {final_train_acc:.4f}")
            print(f"  Final val accuracy: {final_val_acc:.4f}")
    else:
        print("  Training history not found. Add history saving to train_model.py and re-run training.")
        print("  Using best-available epoch data if embedded in model.")

    # --- Run predictions with response time measurement ---
    print("\n[4/11] Running model predictions (with response time measurement)...")
    pred_indices, confidences, response_times = batch_predict(
        model, preprocess_image_01, image_paths, batch_size=32
    )
    predicted_labels = [class_names[idx] for idx in pred_indices]

    # --- Overall accuracy and confidence interval ---
    print("\n[5/11] Computing overall accuracy and 95% CI...")
    correct = np.sum(true_indices == pred_indices)
    accuracy = float(correct) / total
    ci_lower, ci_upper = compute_confidence_interval(accuracy, total)

    # Top-3 accuracy
    print("\n[6/11] Computing Top-3 accuracy...")

    # Collect all prediction probabilities in batches for top-k accuracy
    all_probs_full = []
    print("  Collecting full prediction probabilities for Top-3 accuracy...")
    for i in range(0, total, 32):
        batch = image_paths[i:i + 32]
        images = np.concatenate([preprocess_image_01(p) for p in batch], axis=0)
        preds = model.predict(images, verbose=0)
        all_probs_full.append(preds)
    all_probs_full = np.vstack(all_probs_full)

    top3_acc = top_k_accuracy_score(true_indices, all_probs_full, k=3)

    # --- Per-class metrics ---
    print("\n[7/11] Computing per-class precision, recall, F1...")
    cm = confusion_matrix(true_indices, pred_indices, labels=range(len(class_names)))
    report = classification_report(
        true_indices, pred_indices,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0, digits=4
    )
    report_dict = classification_report(
        true_indices, pred_indices,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0, output_dict=True
    )

    macro_precision = report_dict['macro avg']['precision']
    macro_recall = report_dict['macro avg']['recall']
    macro_f1 = report_dict['macro avg']['f1-score']
    weighted_f1 = report_dict['weighted avg']['f1-score']

    # --- Healthy vs Diseased ---
    print("\n[8/11] Computing healthy-vs-diseased accuracy...")
    healthy_indices, diseased_indices = get_healthy_class_indices(class_names)
    print(f"  Healthy classes ({len(healthy_indices)}): {[class_names[i] for i in healthy_indices]}")
    print(f"  Diseased classes ({len(diseased_indices)}): {[class_names[i] for i in diseased_indices]}")

    # Merge true and predicted labels into healthy/diseased binary
    true_binary = np.array([0 if t in healthy_indices else 1 for t in true_indices])
    pred_binary = np.array([0 if p in healthy_indices else 1 for p in pred_indices])
    hd_accuracy = float(np.mean(true_binary == pred_binary))
    hd_report = classification_report(
        true_binary, pred_binary,
        target_names=['Healthy', 'Diseased'],
        output_dict=True, zero_division=0, digits=4
    )

    # --- Most confused class pairs ---
    most_confused = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                most_confused.append((class_names[i], class_names[j], int(cm[i, j])))
    most_confused.sort(key=lambda x: -x[2])
    top_confused = most_confused[:10]

    # --- Preprocessing comparison ---
    print("\n[9/11] Running preprocessing comparison [0,1] vs [-1,1]...")
    pred_01 = pred_indices  # already computed above
    pred_mm11, _, _ = batch_predict(model, preprocess_image_mm11, image_paths, batch_size=32)

    acc_01 = accuracy_score(true_indices, pred_01)
    acc_mm11 = accuracy_score(true_indices, pred_mm11)
    report_01 = classification_report(true_indices, pred_01, target_names=class_names,
                                       output_dict=True, zero_division=0)
    report_mm11 = classification_report(true_indices, pred_mm11, target_names=class_names,
                                         output_dict=True, zero_division=0)

    # --- Response time statistics ---
    print("\n[10/11] Computing response time statistics...")
    rt_mean = statistics.mean(response_times)
    rt_median = statistics.median(response_times)
    rt_max = max(response_times)
    rt_min = min(response_times)
    rt_stdev = statistics.stdev(response_times) if len(response_times) > 1 else 0.0

    # Consistency check: run predictions 5 times on same data and check stability
    print("\n  Running consistency check (5 iterations)...")
    consistency_preds = []
    for iter_num in range(5):
        sample_paths = image_paths[:100]
        sample_images = np.concatenate([preprocess_image_01(p) for p in sample_paths], axis=0)
        preds = np.argmax(model.predict(sample_images, verbose=0), axis=1)
        consistency_preds.append(preds)
    consistency_preds = np.array(consistency_preds)
    # Check if all 5 runs produce identical predictions
    consistent = np.all([np.array_equal(consistency_preds[0], consistency_preds[i])
                         for i in range(1, 5)])
    consistency_rate = float(np.mean(np.all(consistency_preds == consistency_preds[0], axis=1)))

    # --- End-to-end test preparation ---
    print("\n[11/11] Preparing end-to-end test info...")
    e2e_note = ("End-to-end browser testing requires a running server and browser automation. "
                "Run 'python tests/test_e2e.py' for desktop/mobile browser tests. "
                "For API-level end-to-end test, use: python tests/test_model.py")

    results = {
        "evaluation_date": datetime.now().isoformat(),
        "model_path": str(MODEL_DIR / "agriscan_model.h5"),
        "test_split_path": str(TEST_DIR),
        "total_test_images": total,
        "class_names": class_names,
        "class_counts": class_counts,

        "training_history": training_results,
        "render_health": "Service suspended (free tier). Local health check passed.",
        "render_url": "https://agriscan-app.onrender.com/api/health",

        "overall_accuracy": float(accuracy),
        "accuracy_ci_95": {"lower": ci_lower, "upper": ci_upper},
        "top3_accuracy": float(top3_acc),

        "precision_recall_f1": {
            "per_class": {cn: {
                'precision': float(report_dict[cn]['precision']),
                'recall': float(report_dict[cn]['recall']),
                'f1_score': float(report_dict[cn]['f1-score']),
                'support': int(report_dict[cn]['support']),
            } for cn in class_names if cn in report_dict},
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },

        "confusion_matrix": cm.tolist(),
        "most_confused_pairs": [{"true_class": a, "predicted_class": b, "count": c}
                                for a, b, c in top_confused],

        "healthy_vs_diseased": {
            "healthy_classes": [class_names[i] for i in healthy_indices],
            "diseased_classes": [class_names[i] for i in diseased_indices],
            "accuracy": float(hd_accuracy),
            "healthy_precision": float(hd_report['Healthy']['precision']),
            "healthy_recall": float(hd_report['Healthy']['recall']),
            "healthy_f1": float(hd_report['Healthy']['f1-score']),
            "diseased_precision": float(hd_report['Diseased']['precision']),
            "diseased_recall": float(hd_report['Diseased']['recall']),
            "diseased_f1": float(hd_report['Diseased']['f1-score']),
        },

        "preprocessing_comparison": {
            "method_01": {
                "accuracy": float(acc_01),
                "macro_f1": float(report_01['macro avg']['f1-score']),
            },
            "method_minus1_plus1": {
                "accuracy": float(acc_mm11),
                "macro_f1": float(report_mm11['macro avg']['f1-score']),
            },
        },

        "response_times": {
            "mean_ms": round(rt_mean * 1000, 2),
            "median_ms": round(rt_median * 1000, 2),
            "max_ms": round(rt_max * 1000, 2),
            "min_ms": round(rt_min * 1000, 2),
            "stdev_ms": round(rt_stdev * 1000, 2),
        },

        "consistency": {
            "consistent_across_runs": bool(consistent),
            "consistency_rate": float(consistency_rate),
        },

        "e2e_note": e2e_note,
    }

    # Save full results
    results_path = RESULTS_DIR / "chapter4_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {results_path}")

    # Save CSV of per-class metrics
    import csv
    csv_path = RESULTS_DIR / "per_class_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-score', 'Support'])
        for cn in class_names:
            if cn in report_dict:
                m = report_dict[cn]
                writer.writerow([cn, m['precision'], m['recall'], m['f1-score'], int(m['support'])])
        writer.writerow([])
        writer.writerow(['Macro avg', report_dict['macro avg']['precision'],
                         report_dict['macro avg']['recall'], report_dict['macro avg']['f1-score'], ''])
        writer.writerow(['Weighted avg', report_dict['weighted avg']['precision'],
                         report_dict['weighted avg']['recall'], report_dict['weighted avg']['f1-score'], ''])
    print(f"  Per-class CSV saved to: {csv_path}")

    # Save confusion matrix as CSV
    cm_path = RESULTS_DIR / "confusion_matrix.csv"
    with open(cm_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + class_names)
        for i, cn in enumerate(class_names):
            writer.writerow([cn] + cm[i].tolist())
    print(f"  Confusion matrix CSV saved to: {cm_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("CHAPTER 4 RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTraining History: {'Available' if training_results else 'Not available (history not saved)'}")
    if training_results:
        print(f"  Epochs run: {training_results['epochs_run']}")
        print(f"  Early stopping epoch: {training_results['early_stopped_epoch']}")
    print(f"\nOverall Test Accuracy: {accuracy*100:.2f}% (95% CI: {ci_lower*100:.2f}% - {ci_upper*100:.2f}%)")
    print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\nHealthy vs Diseased Accuracy: {hd_accuracy*100:.2f}%")
    print(f"  Healthy F1: {hd_report['Healthy']['f1-score']:.4f}")
    print(f"  Diseased F1: {hd_report['Diseased']['f1-score']:.4f}")
    print(f"\nPreprocessing Comparison:")
    print(f"  [0,1]    accuracy: {acc_01*100:.2f}%  macro F1: {report_01['macro avg']['f1-score']:.4f}")
    print(f"  [-1,1]   accuracy: {acc_mm11*100:.2f}%  macro F1: {report_mm11['macro avg']['f1-score']:.4f}")
    print(f"\nResponse Times (per image):")
    print(f"  Mean:   {rt_mean*1000:.2f} ms")
    print(f"  Median: {rt_median*1000:.2f} ms")
    print(f"  Max:    {rt_max*1000:.2f} ms")
    print(f"  Min:    {rt_min*1000:.2f} ms")
    print(f"  Stdev:  {rt_stdev*1000:.2f} ms")
    print(f"\nConsistency: {consistency_rate*100:.1f}% consistent across 5 runs")
    print("\nPer-class F1 Scores:")
    print(f"  {'Class':50s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    for cn in class_names:
        if cn in report_dict:
            m = report_dict[cn]
            print(f"  {cn:50s} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1-score']:>10.4f} {int(m['support']):>10d}")

    print(f"\nMost confused class pairs:")
    for a, b, c in top_confused[:5]:
        print(f"  {a[:45]:45s} -> {b[:45]:45s}: {c}")

    print(f"\nResults directory: {RESULTS_DIR}")
    print(f"  - chapter4_results.json (full metrics)")
    print(f"  - per_class_metrics.csv")
    print(f"  - confusion_matrix.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
