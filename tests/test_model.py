import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "data" / "processed" / "split" / "test"
MODEL_DIR = BASE_DIR / "app" / "models"

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG')

with open(MODEL_DIR / "class_names.json", 'r') as f:
    class_names = json.load(f)
print("Class names:", class_names)
print("Number of classes:", len(class_names))

from tensorflow import keras
model = keras.models.load_model(str(MODEL_DIR / "agriscan_model.h5"))
print("\nModel loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# Test with images from the PROCESSED test split (not raw dataset)
print(f"\nTest images directory: {TEST_DIR}")
test_classes = ['Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight',
                'Pepper__bell___Bacterial_spot', 'Potato___Early_blight']

for cls in test_classes:
    class_dir = TEST_DIR / cls
    if not class_dir.exists():
        print(f"\n{cls}: Directory not found in test split!")
        continue
    images = sorted([f for f in class_dir.iterdir()
                     if f.suffix in IMAGE_EXTENSIONS and f.is_file()])
    if not images:
        print(f"\n{cls}: No images found!")
        continue
    img_path = images[0]
    image = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_class_idx] * 100)
    predicted_name = class_names[predicted_class_idx]

    print(f"\n=== {cls} ===")
    print(f"  True class: {cls}")
    print(f"  Predicted: {predicted_name}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"  Match: {predicted_name == cls}")

    # Show top 3 predictions
    top3 = np.argsort(predictions)[::-1][:3]
    for i, idx in enumerate(top3):
        print(f"  #{i+1}: {class_names[idx]} ({predictions[idx]*100:.2f}%)")

# Check knowledge base mapping
print("\n=== Knowledge Base Mapping Check ===")
from app.services.agriculture_ml_service import AgricultureMLService
kb = AgricultureMLService().knowledge_base
for cn in class_names:
    display_name = cn.replace('__', ' ').replace('_', ' ')
    in_kb_underscore = cn in kb
    in_kb_spaces = display_name in kb
    print(f"  {cn}: KB(underscores)={in_kb_underscore}, KB(spaces)={in_kb_spaces}")
