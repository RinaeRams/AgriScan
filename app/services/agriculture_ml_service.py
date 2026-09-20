import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from PIL import Image

class AgricultureMLService:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.knowledge_base = self._load_knowledge_base()
        self._initialize_model()

    def _load_knowledge_base(self) -> Dict[str, Dict]:
        return {
            'Pepper__bell___Bacterial_spot': {
                'description': 'Bacterial spot causes small, dark, water-soaked spots on pepper leaves and fruit.',
                'recommendations': [
                    'Remove infected plant parts immediately',
                    'Avoid overhead irrigation and working with wet plants',
                    'Apply copper-based bactericide',
                    'Use disease-free seeds and transplants',
                    'Rotate crops and practice good sanitation'
                ]
            },
            'Pepper__bell___healthy': {
                'description': 'Your pepper plant appears healthy with no visible signs of disease.',
                'recommendations': [
                    'Continue regular watering and fertilization',
                    'Monitor for any changes in leaf color or texture',
                    'Maintain proper plant spacing for air circulation',
                    'Keep soil pH appropriate for pepper plants',
                    'Ensure adequate sunlight exposure'
                ]
            },
            'Potato___Early_blight': {
                'description': 'Early blight is a fungal disease causing dark brown spots with concentric rings on potato leaves.',
                'recommendations': [
                    'Remove affected leaves immediately',
                    'Apply fungicide preventively',
                    'Ensure good air circulation around plants',
                    'Avoid overhead watering',
                    'Mulch to prevent soil splash onto leaves'
                ]
            },
            'Potato___healthy': {
                'description': 'Your potato plant appears healthy with no visible signs of disease.',
                'recommendations': [
                    'Continue regular watering and fertilization',
                    'Monitor for any changes in leaf color or texture',
                    'Hill soil around plants to support tubers',
                    'Maintain proper plant spacing',
                    'Ensure adequate sunlight exposure'
                ]
            },
            'Potato___Late_blight': {
                'description': 'Late blight is a devastating disease causing water-soaked lesions that turn brown and necrotic.',
                'recommendations': [
                    'Remove infected plants immediately',
                    'Apply fungicide preventively every 7-10 days',
                    'Ensure good drainage and avoid overhead watering',
                    'Rotate crops and do not plant potatoes in same location',
                    'Use resistant varieties when available'
                ]
            },
            'Tomato__Target_Spot': {
                'description': 'Target spot causes brown, circular spots with concentric rings on leaves and fruit.',
                'recommendations': [
                    'Remove infected leaves and fruit',
                    'Apply fungicide preventively',
                    'Ensure proper plant spacing for air circulation',
                    'Avoid overhead watering',
                    'Rotate crops annually'
                ]
            },
            'Tomato__Tomato_mosaic_virus': {
                'description': 'Tomato mosaic virus causes mottled light and dark green areas on leaves, with possible fruit distortion.',
                'recommendations': [
                    'Remove and destroy infected plants immediately',
                    'Wash hands and tools after handling infected plants',
                    'Control aphid vectors with insecticidal soap',
                    'Use virus-free seeds and transplants',
                    'Avoid smoking near plants as tobacco can carry the virus'
                ]
            },
            'Tomato__Tomato_YellowLeaf__Curl_Virus': {
                'description': 'Yellow leaf curl virus causes upward curling of leaves with yellowing and stunted growth.',
                'recommendations': [
                    'Remove and destroy infected plants immediately',
                    'Control whitefly vectors with insecticidal soap or neem oil',
                    'Use reflective mulches to repel whiteflies',
                    'Plant resistant varieties when available',
                    'Use yellow sticky traps to monitor whitefly populations'
                ]
            },
            'Tomato_Bacterial_spot': {
                'description': 'Bacterial spot causes small, dark, water-soaked spots on tomato leaves and fruit.',
                'recommendations': [
                    'Remove infected plant parts immediately',
                    'Avoid overhead irrigation and working with wet plants',
                    'Apply copper-based bactericide',
                    'Use disease-free seeds and transplants',
                    'Rotate crops and practice good sanitation'
                ]
            },
            'Tomato_Early_blight': {
                'description': 'Early blight causes dark brown spots with concentric rings on lower leaves first.',
                'recommendations': [
                    'Remove affected lower leaves',
                    'Apply fungicide preventively',
                    'Ensure proper plant spacing for air circulation',
                    'Mulch to prevent soil splash',
                    'Rotate crops annually'
                ]
            },
            'Tomato_healthy': {
                'description': 'Your tomato plant appears healthy with no visible signs of disease.',
                'recommendations': [
                    'Continue regular watering and fertilization',
                    'Monitor for any changes in leaf color or texture',
                    'Maintain proper plant spacing for air circulation',
                    'Prune suckers for better airflow',
                    'Ensure adequate sunlight exposure'
                ]
            },
            'Tomato_Late_blight': {
                'description': 'Late blight is a devastating disease causing irregular water-soaked lesions that rapidly expand.',
                'recommendations': [
                    'Remove infected plants immediately',
                    'Apply fungicide preventively every 7-10 days',
                    'Ensure good drainage and avoid overhead watering',
                    'Rotate crops and do not plant tomatoes in same location',
                    'Use resistant varieties when available'
                ]
            },
            'Tomato_Leaf_Mold': {
                'description': 'Leaf mold causes pale green or yellowish spots on upper leaf surfaces with olive-green mold on undersides.',
                'recommendations': [
                    'Improve air circulation by proper spacing',
                    'Avoid overhead watering',
                    'Apply fungicide if infection is severe',
                    'Remove and destroy infected leaves',
                    'Grow resistant varieties when possible'
                ]
            },
            'Tomato_Septoria_leaf_spot': {
                'description': 'Septoria leaf spot causes small, circular spots with dark borders and gray centers on leaves.',
                'recommendations': [
                    'Remove infected leaves immediately',
                    'Apply fungicide preventively',
                    'Avoid overhead watering',
                    'Mulch to prevent soil splash',
                    'Rotate crops and practice good sanitation'
                ]
            },
            'Tomato_Spider_mites_Two_spotted_spider_mite': {
                'description': 'Spider mites cause tiny yellow or bronze speckling on leaves, with fine webbing on undersides.',
                'recommendations': [
                    'Spray plants with strong water stream to dislodge mites',
                    'Apply insecticidal soap or neem oil',
                    'Introduce predatory mites for biological control',
                    'Maintain adequate humidity around plants',
                    'Remove heavily infested leaves'
                ]
            }
        }

    def _initialize_model(self):
        """
        Initialize the model from the trained model file.
        """
        try:
            from tensorflow import keras

            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(model_dir, exist_ok=True)

            # Load class names from file
            class_names_path = os.path.join(model_dir, 'class_names.json')
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"Loaded {len(self.class_names)} class names from file")
            else:
                # Fallback to hardcoded list
                self.class_names = [
                    'Apple Black Rot', 'Apple Cedar Rust', 'Apple Healthy',
                    'Corn Common Rust', 'Corn Gray Leaf Spot', 'Corn Healthy',
                    'Corn Northern Leaf Blight',
                    'Potato Early Blight', 'Potato Healthy', 'Potato Late Blight',
                    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Healthy',
                    'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
                    'Tomato Spider Mites', 'Tomato Target Spot',
                    'Tomato Mosaic Virus', 'Tomato Yellow Leaf Curl Virus'
                ]
                print(f"Using default {len(self.class_names)} class names")

            model_path = os.path.join(model_dir, 'agriscan_model.h5')
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                self.model = keras.models.load_model(model_path)
                print("Loaded trained AgriScan model successfully")
            else:
                print(f"Model not found at {model_path}")
                print("Model will use fallback predictions until trained")
                self.model = None

        except Exception as e:
            print(f"Could not initialize model: {e}")
            self.model = None

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model input.
        Resizes to 224x224 and scales to [0, 1] range as used during training.
        """
        # Resize to 224x224
        image = image.resize((224, 224))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to array and scale to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def analyze_plant_image(self, image: Image.Image) -> Dict:
        """
        Analyze plant image for disease detection.
        Alias for predict() to maintain compatibility with existing code.
        """
        return self.predict(image)

    def predict(self, image: Image.Image) -> Dict:
        """
        Predict disease from plant image.
        Returns dict with disease name, confidence, all predictions, and timestamp.
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not trained yet. Please run train_model.py first.',
                'disease': None,
                'confidence': 0.0,
                'all_predictions': {},
                'timestamp': datetime.utcnow().isoformat()
            }

        try:
            # Preprocess image
            img_array = self.preprocess_image(image)

            # Get predictions
            predictions = self.model.predict(img_array, verbose=0)[0]

            # Get top prediction
            predicted_class_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_class_idx] * 100)
            disease_name = self.class_names[predicted_class_idx]

            # Get all predictions
            all_predictions = {
                self.class_names[i]: float(predictions[i] * 100)
                for i in range(len(self.class_names))
            }

            # Sort by confidence descending
            all_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))

            # Get disease info
            disease_info = self.knowledge_base.get(disease_name, {
                'description': 'Disease detected. Consult an agricultural expert for treatment options.',
                'recommendations': ['Consult a local agricultural extension officer for specific treatment recommendations']
            })

            return {
                'success': True,
                'disease': disease_name,
                'confidence': round(confidence, 2),
                'all_predictions': {k: round(v, 2) for k, v in all_predictions.items()},
                'description': disease_info['description'],
                'recommendations': disease_info['recommendations'],
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'disease': None,
                'confidence': 0.0,
                'all_predictions': {},
                'timestamp': datetime.utcnow().isoformat()
            }

    def generate_advice_text(self, query: str) -> str:
        query_lower = query.lower()
        for disease_name, info in self.knowledge_base.items():
            normalized_name = disease_name.lower().replace('__', ' ').replace('_', ' ')
            if normalized_name in query_lower:
                return f"{disease_name}: {info['description']} Recommendations: {', '.join(info['recommendations'][:3])}"

        return "Please upload a clear image of your plant for accurate disease diagnosis."