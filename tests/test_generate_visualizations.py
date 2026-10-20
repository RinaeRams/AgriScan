#!/usr/bin/env python3
"""
Generate confusion matrix heatmap and training curve visualizations
for Chapter 4 from the evaluation results.
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "tests" / "chapter4_results"
MODEL_DIR = BASE_DIR / "app" / "models"


def plot_confusion_matrix(cm, class_names, output_path):
    """Generate and save confusion matrix heatmap."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('AgriScan Confusion Matrix (15 Classes)\nTest Split: 5,497 Images', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


EPOCHS = 20

def plot_training_curves(history, output_path):
    """Generate training/validation learning curves from history."""
    if history is None:
        print("No training history available. Skipping curve plot.")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['loss']) + 1)

    axes[0].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training & Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    early_epoch = history.get('early_stopped_epoch', 0)
    total_epochs = history.get('total_epochs_run', len(history.get('loss', [])))
    if early_epoch and total_epochs < EPOCHS:
        axes[0].axvline(x=early_epoch, color='green', linestyle='--', alpha=0.7,
                        label=f'Early Stop (epoch {early_epoch})')
        axes[0].legend()
    elif early_epoch:
        axes[0].axvline(x=early_epoch, color='orange', linestyle='--', alpha=0.7,
                        label=f'Final Epoch (epoch {early_epoch})')
        axes[0].legend()

    axes[1].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training & Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    if early_epoch and total_epochs < EPOCHS:
        axes[1].axvline(x=early_epoch, color='green', linestyle='--', alpha=0.7,
                        label=f'Early Stop (epoch {early_epoch})')
        axes[1].legend()
    elif early_epoch:
        axes[1].axvline(x=early_epoch, color='orange', linestyle='--', alpha=0.7,
                        label=f'Final Epoch (epoch {early_epoch})')
        axes[1].legend()

    plt.suptitle('AgriScan Model Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {output_path}")
    return True


def main():
    results_path = RESULTS_DIR / "chapter4_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)

    class_names = results['class_names']
    cm = results['confusion_matrix']
    history = results.get('training_history')

    # Plot confusion matrix
    cm_image_path = RESULTS_DIR / "confusion_matrix_heatmap.png"
    plot_confusion_matrix(cm, class_names, str(cm_image_path))

    # Plot training curves (if history available)
    curves_image_path = RESULTS_DIR / "training_curves.png"
    if history:
        plot_training_curves(history, str(curves_image_path))
    else:
        print(f"Training history not available (training_history.json not found).")
        print(f"train_model.py has been updated to save history on future training runs.")
        print(f"To generate training curves: re-run train_model.py, then re-run this script.")

    print(f"\nVisualization files in {RESULTS_DIR}:")
    for f in RESULTS_DIR.iterdir():
        if f.suffix in ('.png', '.jpg'):
            print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
