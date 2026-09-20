#!/usr/bin/env python3
"""
Generate dataset summary JSON from split directories.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SPLIT_DIR = BASE_DIR / "data" / "processed" / "split"
OUTPUT_FILE = BASE_DIR / "data" / "dataset_summary.json"

def count_images(dir_path):
    if not dir_path.exists():
        return 0
    return len(list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.JPG")) +
               list(dir_path.glob("*.png")) + list(dir_path.glob("*.PNG")) +
               list(dir_path.glob("*.jpeg")) + list(dir_path.glob("*.JPEG")))

def main():
    if not SPLIT_DIR.exists():
        print(f"ERROR: Split directory not found: {SPLIT_DIR}")
        return 1

    splits = ["train", "validation", "test"]
    class_dirs = [d for d in (SPLIT_DIR / "train").iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in class_dirs])

    summary = {
        "classes": class_names,
        "num_classes": len(class_names),
        "splits": {},
        "total_images": 0
    }

    for split in splits:
        split_dir = SPLIT_DIR / split
        split_data = {}
        split_total = 0

        for class_name in class_names:
            class_dir = split_dir / class_name
            count = count_images(class_dir)
            split_data[class_name] = count
            split_total += count

        summary["splits"][split] = {
            "class_counts": split_data,
            "total": split_total
        }
        summary["total_images"] += split_total

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset summary saved to {OUTPUT_FILE}")
    print(f"Total classes: {summary['num_classes']}")
    print(f"Total images: {summary['total_images']}")
    for split in splits:
        print(f"  {split}: {summary['splits'][split]['total']} images")

    return 0

if __name__ == "__main__":
    exit(main())