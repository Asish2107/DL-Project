import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"src/components/artifacts/SKIN Diseases.h5"))
        self.class_names = self.class_names = [
            'Atopic Dermatitis',
            'Eczema',
            'Melanocytic Nevi',
            'Psoriasis pictures',
            'Seborrheic Keratoses Benign Tumors',
            'Tinea Ringworm Candidiasis',
            'Warts Molluscum'
        ]

        # Verify model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        self.model = self.load_model()

    def load_model(self):
        """Load the trained model"""
        return tf.keras.models.load_model(self.model_path)
    
    def preprocess_image(self, image_path):
        """Process image for MobileNetV2"""
        img = load_img(image_path, target_size=(244, 244))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    
    def predict(self, image_path):
        """Make prediction on single image"""
        processed_img = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_img)
        return self.class_names[np.argmax(predictions)], np.max(predictions)