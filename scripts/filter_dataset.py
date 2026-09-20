#!/usr/bin/env python3
"""
Filter PlantVillage dataset to keep only Tomato, Corn, Potato, Pepper classes.
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / "data" / "raw" / "plantvillage" / "PlantVillage"
DEST_DIR = BASE_DIR / "data" / "processed" / "plantvillage_filtered"

TARGET_CROPS = ["Tomato", "Corn", "Potato", "Pepper"]

def main():
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        print("Please run the download step first (Task 3)")
        return 1

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    all_classes = sorted([d.name for d in SOURCE_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(all_classes)} total classes in PlantVillage")

    filtered_classes = []
    for class_name in all_classes:
        if any(crop.lower() in class_name.lower() for crop in TARGET_CROPS):
            filtered_classes.append(class_name)

    print(f"Filtering to keep {len(filtered_classes)} classes: {filtered_classes}")

    total_images = 0
    for class_name in filtered_classes:
        src_class_dir = SOURCE_DIR / class_name
        dst_class_dir = DEST_DIR / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        images = list(src_class_dir.glob("*.jpg")) + list(src_class_dir.glob("*.JPG")) + \
                 list(src_class_dir.glob("*.png")) + list(src_class_dir.glob("*.PNG")) + \
                 list(src_class_dir.glob("*.jpeg")) + list(src_class_dir.glob("*.JPEG"))

        for img in images:
            shutil.copy2(img, dst_class_dir / img.name)

        print(f"  {class_name}: {len(images)} images")
        total_images += len(images)

    print(f"\nTotal classes retained: {len(filtered_classes)}")
    print(f"Total images copied: {total_images}")
    return 0

if __name__ == "__main__":
    exit(main())