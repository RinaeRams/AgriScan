#!/usr/bin/env python3
"""
Convert the trained AgriScan Keras model to TensorFlow Lite (.tflite)
for on-device / offline inference.

Supports two quantization modes:
  - float32   (default): Full-precision model.
  - float16   : Half-precision quantization (~50% size reduction, minimal
                accuracy loss, good for mobile GPUs).
  - int8      : Full integer quantization (smallest, fastest, requires a
                representative dataset for calibration).

Usage:
    python scripts/convert_to_tflite.py
    python scripts/convert_to_tflite.py --mode float16
    python scripts/convert_to_tflite.py --mode int8
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "app" / "models"
KERAS_MODEL_PATH = MODEL_DIR / "agriscan_model.h5"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

QUANTIZATION_MODES = {"float32", "float16", "int8"}


def load_class_names():
    with open(CLASS_NAMES_PATH, 'r') as f:
        return json.load(f)


def representative_dataset(class_names, test_dir, num_samples=100):
    """Yield representative samples for int8 quantization calibration."""
    image_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG')
    samples = []

    for class_name in class_names:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue
        images = [f for f in class_dir.iterdir()
                  if f.suffix in image_exts and f.is_file()]
        samples.extend(images[:min(len(images), num_samples // len(class_names))])

    def gen():
        for img_path in samples:
            image = Image.open(img_path).convert('RGB').resize((224, 224))
            img_array = np.array(image, dtype=np.float32) / 255.0
            yield [np.expand_dims(img_array, axis=0)]

    return gen, len(samples)


def convert_to_tflite(model_path, mode="float32"):
    """Convert a Keras model to TFLite format with the specified quantization."""
    model = keras.models.load_model(str(model_path))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if mode == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        class_names = load_class_names()
        test_dir = BASE_DIR / "data" / "processed" / "split" / "test"
        if not test_dir.exists():
            raise RuntimeError(f"Test directory not found for int8 calibration: {test_dir}")
        rep_data, num_samples = representative_dataset(class_names, test_dir)
        print(f"Using {num_samples} representative samples for int8 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        converter.optimizations = []

    tflite_model = converter.convert()
    return tflite_model


def verify_tflite_model(tflite_model, keras_model, class_names):
    """Sanity-check that the TFLite model produces similar output to Keras."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Use a dummy input
    input_shape = input_details[0]['shape']
    dummy_input = (np.random.random(input_shape).astype(np.float32) - 0.5) * 2

    # Keras prediction
    keras_output = keras_model.predict(dummy_input, verbose=0)[0]

    # TFLite prediction
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])[0]

    keras_pred = np.argmax(keras_output)
    tflite_pred = np.argmax(tflite_output)
    max_diff = np.max(np.abs(keras_output - tflite_output))

    print(f"  Keras prediction:   {class_names[keras_pred]} ({keras_output[keras_pred]*100:.1f}%)")
    print(f"  TFLite prediction: {class_names[tflite_pred]} ({tflite_output[tflite_pred]*100:.1f}%)")
    print(f"  Max output difference: {max_diff:.6f}")

    if keras_pred == tflite_pred and max_diff < 0.01:
        print("  Verification PASSED: predictions match")
    else:
        print("  Verification WARNING: predictions differ slightly (check output)")

    return keras_pred == tflite_pred


def main():
    parser = argparse.ArgumentParser(description="Convert AgriScan model to TFLite for offline inference")
    parser.add_argument("--mode", choices=sorted(QUANTIZATION_MODES), default="float32",
                        help="Quantization mode (default: float32)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip model verification")
    args = parser.parse_args()

    if not KERAS_MODEL_PATH.exists():
        print(f"ERROR: Keras model not found at {KERAS_MODEL_PATH}")
        print("Please run train_model.py first.")
        return 1

    class_names = load_class_names()
    print(f"Loaded {len(class_names)} class names")
    print(f"Converting model with {args.mode} quantization...")

    tflite_model = convert_to_tflite(KERAS_MODEL_PATH, mode=args.mode)

    # Save the .tflite file
    tflite_filename = f"agriscan_model_{args.mode}.tflite"
    tflite_path = MODEL_DIR / tflite_filename
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    keras_size = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)

    print("\nConversion complete!")
    print(f"  Keras model:  {keras_size:.2f} MB ({KERAS_MODEL_PATH.name})")
    print(f"  TFLite model: {tflite_size:.2f} MB ({tflite_filename})")
    print(f"  Size reduction: {(1 - tflite_size / keras_size) * 100:.1f}%")

    if not args.no_verify:
        print("\nVerifying TFLite model...")
        keras_model = keras.models.load_model(str(KERAS_MODEL_PATH))
        verify_tflite_model(tflite_model, keras_model, class_names)

    # Save metadata
    metadata = {
        "model_file": tflite_filename,
        "quantization": args.mode,
        "input_size": [224, 224, 3],
        "num_classes": len(class_names),
        "class_names": class_names,
        "keras_model_size_mb": round(keras_size, 2),
        "tflite_model_size_mb": round(tflite_size, 2),
    }
    metadata_path = MODEL_DIR / f"agriscan_model_{args.mode}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_path}")

    return 0


if __name__ == "__main__":
    exit(main())
