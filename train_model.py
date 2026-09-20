#!/usr/bin/env python3
"""
Train AgriScan crop disease detection model using MobileNetV2.
Uses tf.data.Dataset for robust handling of corrupted images.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, applications
from pathlib import Path

BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / "data" / "processed" / "split" / "train"
VAL_DIR = BASE_DIR / "data" / "processed" / "split" / "validation"
TEST_DIR = BASE_DIR / "data" / "processed" / "split" / "test"
MODEL_DIR = BASE_DIR / "app" / "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

def create_dataset(directory, class_names=None, class_to_idx=None, shuffle=True, augment=False):
    """Create tf.data.Dataset from directory with error handling for corrupted images."""
    
    # Get class names from training set if not provided
    if class_names is None:
        class_dirs = sorted([d for d in Path(directory).iterdir() if d.is_dir()])
        class_names = [d.name for d in class_dirs]
    
    if class_to_idx is None:
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Collect all image paths
    image_paths = []
    labels = []
    for class_name in class_names:
        class_dir = Path(directory) / class_name
        if not class_dir.exists():
            print(f"  Warning: Class directory not found: {class_dir}")
            continue
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image_paths.append(str(img_path))
                labels.append(class_to_idx[class_name])
    
    print(f"  Found {len(image_paths)} images in {len(class_names)} classes")
    
    # Create dataset from paths
    def load_and_preprocess(path, label):
        # Read file
        img = tf.io.read_file(path)
        # Decode with error handling
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        # Resize
        img = tf.image.resize(img, IMG_SIZE)
        # Normalize
        img = tf.cast(img, tf.float32) / 255.0
        # One-hot encode label
        label = tf.one_hot(label, depth=len(class_names))
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=42, reshuffle_each_iteration=True)
    
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, class_names, len(image_paths)

def build_model(num_classes):
    base_model = applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def main():
    print("=" * 60)
    print("AgriScan Model Training")
    print("=" * 60)

    if not TRAIN_DIR.exists():
        print(f"ERROR: Training directory not found: {TRAIN_DIR}")
        print("Please run the dataset pipeline first (Tasks 3-8)")
        return 1

    print(f"\nLoading data from:")
    print(f"  Train: {TRAIN_DIR}")
    print(f"  Validation: {VAL_DIR}")
    print(f"  Test: {TEST_DIR}")

    train_ds, class_names, train_count = create_dataset(TRAIN_DIR, shuffle=True, augment=True)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    val_ds, _, val_count = create_dataset(VAL_DIR, class_names=class_names, class_to_idx=class_to_idx, shuffle=False, augment=False)
    test_ds, _, test_count = create_dataset(TEST_DIR, class_names=class_names, class_to_idx=class_to_idx, shuffle=False, augment=False)

    num_classes = len(class_names)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {class_names}")
    print(f"Train samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Test samples: {test_count}")

    print("\nBuilding model...")
    model = build_model(num_classes)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = callbacks.ModelCheckpoint(
        MODEL_DIR / "agriscan_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    print("\nStarting training...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stop, model_checkpoint],
        verbose=1
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Save class names
    class_names_path = MODEL_DIR / "class_names.json"
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"\nClass names saved to {class_names_path}")

    print(f"Model saved to {MODEL_DIR / 'agriscan_model.h5'}")
    print("Training complete!")
    return 0

if __name__ == "__main__":
    exit(main())