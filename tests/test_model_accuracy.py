import json
import os
from PIL import Image
import numpy as np
from tensorflow import keras

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(base_dir, 'data', 'raw', 'plantvillage', 'PlantVillage')

with open(os.path.join(base_dir, 'app', 'models', 'class_names.json'), 'r') as f:
    class_names = json.load(f)

model = keras.models.load_model(os.path.join(base_dir, 'app', 'models', 'agriscan_model.h5'))

# Test with more images per class
correct = 0
total = 0
for cls in class_names:
    class_dir = os.path.join(dataset_dir, cls.replace(' ', '_'))
    if not os.path.isdir(class_dir):
        continue
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) < 3:
        continue
    # Test with first 3 images
    for img_name in images[:3]:
        img_path = os.path.join(class_dir, img_name)
        image = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - 0.5) * 2.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_name = class_names[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx] * 100)
        
        match = predicted_name == cls
        total += 1
        if match:
            correct += 1
        
        status = "OK" if match else "WRONG"
        print(f"[{status}] {cls[:35]:35s} -> {predicted_name[:35]:35s} conf={confidence:.1f}%")

print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")

# Show prediction distribution
print("\n=== Prediction Distribution ===")
pred_counts = {cn: 0 for cn in class_names}
for cls in class_names:
    class_dir = os.path.join(dataset_dir, cls.replace(' ', '_'))
    if not os.path.isdir(class_dir):
        continue
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) < 3:
        continue
    for img_name in images[:3]:
        img_path = os.path.join(class_dir, img_name)
        image = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - 0.5) * 2.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_name = class_names[predicted_class_idx]
        pred_counts[predicted_name] += 1

for cn, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
    print(f"  {cn[:40]:40s}: {count}")
