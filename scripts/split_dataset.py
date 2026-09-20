#!/usr/bin/env python3
"""
Split filtered dataset into train/validation/test (70/15/15).
"""

import os
import random
import shutil
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / "data" / "processed" / "plantvillage_filtered"
DEST_DIR = BASE_DIR / "data" / "processed" / "split"

TRAIN_DIR = DEST_DIR / "train"
VAL_DIR = DEST_DIR / "validation"
TEST_DIR = DEST_DIR / "test"

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}

def main():
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        print("Please run filter_dataset.py first (Task 5-6)")
        return 1

    for split in ["train", "validation", "test"]:
        (DEST_DIR / split).mkdir(parents=True, exist_ok=True)

    all_classes = sorted([d.name for d in SOURCE_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(all_classes)} classes to split")

    total_counts = {"train": 0, "validation": 0, "test": 0}

    for class_name in all_classes:
        src_class_dir = SOURCE_DIR / class_name
        images = list(src_class_dir.glob("*.jpg")) + list(src_class_dir.glob("*.JPG")) + \
                 list(src_class_dir.glob("*.png")) + list(src_class_dir.glob("*.PNG")) + \
                 list(src_class_dir.glob("*.jpeg")) + list(src_class_dir.glob("*.JPEG"))

        random.shuffle(images)
        n = len(images)

        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["validation"])
        n_test = n - n_train - n_val

        splits = {
            "train": images[:n_train],
            "validation": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, split_images in splits.items():
            dst_class_dir = DEST_DIR / split_name / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy2(img, dst_class_dir / img.name)

        total_counts["train"] += n_train
        total_counts["validation"] += n_val
        total_counts["test"] += n_test

        print(f"  {class_name}: train={n_train}, val={n_val}, test={n_test} (total={n})")

    print(f"\nTotal: train={total_counts['train']}, validation={total_counts['validation']}, test={total_counts['test']}")
    print(f"Grand total: {sum(total_counts.values())} images across {len(all_classes)} classes")
    return 0

if __name__ == "__main__":
    exit(main())