#!/usr/bin/env python3
"""
Clean corrupted images from dataset directories.
"""

import os
from pathlib import Path
from PIL import Image

def clean_corrupted_images(root_dir):
    """Remove corrupted images from all subdirectories."""
    root = Path(root_dir)
    if not root.exists():
        print(f"Directory not found: {root_dir}")
        return 0
    
    removed = 0
    for img_path in root.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Re-open since verify() closes the file
                with Image.open(img_path) as img:
                    img.load()
            except Exception as e:
                print(f"Removing corrupted: {img_path} ({e})")
                img_path.unlink()
                removed += 1
    
    return removed

def main():
    base_dir = Path(__file__).parent.parent
    dirs = [
        base_dir / "data" / "processed" / "split" / "train",
        base_dir / "data" / "processed" / "split" / "validation",
        base_dir / "data" / "processed" / "split" / "test",
    ]
    
    total_removed = 0
    for d in dirs:
        if d.exists():
            print(f"Cleaning {d}...")
            removed = clean_corrupted_images(d)
            print(f"  Removed {removed} corrupted images")
            total_removed += removed
    
    print(f"\nTotal corrupted images removed: {total_removed}")
    return 0

if __name__ == "__main__":
    exit(main())