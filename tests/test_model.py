import json
import os
import sys
from PIL import Image
import numpy as np

# Add the AgriNathiApp-master directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(base_dir, 'data', 'raw', 'plantvillage', 'PlantVillage')

# Load class names
with open(os.path.join(base_dir, 'app', 'models', 'class_names.json'), 'r') as f:
    class_names = json.load(f)
print("Class names:", class_names)
print("Number of classes:", len(class_names))

# Load model
from tensorflow import keras
model = keras.models.load_model(os.path.join(base_dir, 'app', 'models', 'agriscan_model.h5'))
print("\nModel loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# Test with images from different classes
test_classes = ['Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight', 
                'Pepper__bell___Bacterial_spot', 'Potato___Early_blight']
for cls in test_classes:
    class_dir = os.path.join(dataset_dir, cls)
    if not os.path.isdir(class_dir):
        print(f"\n{cls}: Directory not found!")
        continue
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"\n{cls}: No images found!")
        continue
    img_path = os.path.join(class_dir, images[0])
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
    # Check if the class name exists in knowledge base (with spaces)
    display_name = cn.replace('__', ' ').replace('_', ' ')
    in_kb_underscore = cn in kb
    in_kb_spaces = display_name in kb
    print(f"  {cn}: KB(underscores)={in_kb_underscore}, KB(spaces)={in_kb_spaces}")
