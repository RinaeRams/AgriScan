#!/usr/bin/env python3
"""
Split filtered dataset into train/validation/test (70/15/15) using stratified
per-class splitting that guarantees a minimum number of images in each split
for every class.
"""

import os
import random
import shutil
import stat
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / "data" / "processed" / "plantvillage_filtered"
DEST_DIR = BASE_DIR / "data" / "processed" / "split"

TRAIN_DIR = DEST_DIR / "train"
VAL_DIR = DEST_DIR / "validation"
TEST_DIR = DEST_DIR / "test"

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
MIN_VAL_PER_CLASS = 1
MIN_TEST_PER_CLASS = 1


def compute_stratified_split(n, ratios, min_val, min_test):
    """Compute train/val/test counts with guaranteed minimums.

    Guarantees that val and test each receive at least ``min_val`` / ``min_test``
    images when the class is large enough, preventing classes like
    ``Tomato_healthy`` from disappearing entirely from a split due to rounding.
    """
    n_val = max(int(n * ratios["validation"]), min_val)
    n_test = max(int(n * ratios["test"]), min_test)
    if n_val + n_test >= n:
        # Not enough images to satisfy minimums: fall back to at least 1 test,
        # then 1 val, then the rest to train.
        n_test = min(1, n)
        n_val = min(1, n - n_test)
        n_train = n - n_test - n_val
    else:
        n_train = n - n_val - n_test
    return n_train, n_val, n_test


def main():
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        print("Please run filter_dataset.py first (Task 5-6)")
        return 1

    # Clean previous split
    if DEST_DIR.exists():
        def _on_rm_error(func, path, exc_info):
            """Handle read-only files on Windows during rmtree."""
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass
        shutil.rmtree(DEST_DIR, onexc=_on_rm_error)

    for split in ["train", "validation", "test"]:
        (DEST_DIR / split).mkdir(parents=True, exist_ok=True)

    all_classes = sorted([d.name for d in SOURCE_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(all_classes)} classes to split")
    print(f"Split ratios: train={SPLIT_RATIOS['train']}, val={SPLIT_RATIOS['validation']}, test={SPLIT_RATIOS['test']}")
    print(f"Minimum per class: val={MIN_VAL_PER_CLASS}, test={MIN_TEST_PER_CLASS}")
    print()

    total_counts = {"train": 0, "validation": 0, "test": 0}

    for class_name in all_classes:
        src_class_dir = SOURCE_DIR / class_name
        images = list(src_class_dir.glob("*.jpg")) + list(src_class_dir.glob("*.JPG")) + \
                 list(src_class_dir.glob("*.png")) + list(src_class_dir.glob("*.PNG")) + \
                 list(src_class_dir.glob("*.jpeg")) + list(src_class_dir.glob("*.JPEG"))

        random.shuffle(images)
        n = len(images)

        n_train, n_val, n_test = compute_stratified_split(n, SPLIT_RATIOS, MIN_VAL_PER_CLASS, MIN_TEST_PER_CLASS)

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

    # Verify that every source class has representation in val and test
    missing = []
    for class_name in all_classes:
        val_dir = DEST_DIR / "validation" / class_name
        test_dir = DEST_DIR / "test" / class_name
        val_count = len(list(val_dir.glob("*"))) if val_dir.exists() else 0
        test_count = len(list(test_dir.glob("*"))) if test_dir.exists() else 0
        if val_count < MIN_VAL_PER_CLASS or test_count < MIN_TEST_PER_CLASS:
            missing.append(f"{class_name}: val={val_count}, test={test_count}")
    if missing:
        print("\nWARNING: Some classes do not meet minimum split requirements:")
        for m in missing:
            print(f"  {m}")
    else:
        print("\nAll classes have sufficient images in val and test splits.")

    return 0


if __name__ == "__main__":
    exit(main())