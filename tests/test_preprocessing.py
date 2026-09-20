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

# Test with TWO preprocessing methods
test_classes = ['Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight',
                'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                'Potato___Late_blight', 'Tomato__Tomato_YellowLeaf__Curl_Virus']

print("=== Comparing [0,1] vs [-1,1] preprocessing ===")
for cls in test_classes:
    class_dir = os.path.join(dataset_dir, cls)
    if not os.path.isdir(class_dir):
        continue
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) < 2:
        continue
    img_path = os.path.join(class_dir, images[0])
    image = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array_raw = np.array(image, dtype=np.float32)

    # Method 1: [0, 1] (matches training script)
    arr_01 = img_array_raw / 255.0
    arr_01 = np.expand_dims(arr_01, axis=0)

    # Method 2: [-1, 1] (current ML service)
    arr_mm = (img_array_raw / 255.0 - 0.5) * 2.0
    arr_mm = np.expand_dims(arr_mm, axis=0)

    pred_01 = model.predict(arr_01, verbose=0)[0]
    pred_mm = model.predict(arr_mm, verbose=0)[0]

    idx_01 = np.argmax(pred_01)
    idx_mm = np.argmax(pred_mm)

    print(f"\n{cls}:")
    print(f"  [0,1]  -> {class_names[idx_01]} ({pred_01[idx_01]*100:.1f}%) {'CORRECT' if class_names[idx_01] == cls else 'wrong'}")
    print(f"  [-1,1] -> {class_names[idx_mm]} ({pred_mm[idx_mm]*100:.1f}%) {'CORRECT' if class_names[idx_mm] == cls else 'wrong'}")

# Full accuracy test with [0,1] preprocessing
print("\n\n=== Full accuracy test with [0,1] preprocessing ===")
correct = 0
total = 0
for cls in class_names:
    class_dir = os.path.join(dataset_dir, cls)
    if not os.path.isdir(class_dir):
        continue
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) < 3:
        continue
    for img_name in images[:3]:
        img_path = os.path.join(class_dir, img_name)
        image = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_name = class_names[predicted_idx]

        total += 1
        if predicted_name == cls:
            correct += 1
        print(f"[{'OK' if predicted_name == cls else 'WRONG'}] {cls[:30]:30s} -> {predicted_name[:30]:30s} {predictions[predicted_idx]*100:.1f}%")

print(f"\nAccuracy with [0,1]: {correct}/{total} = {correct/total*100:.1f}%")
