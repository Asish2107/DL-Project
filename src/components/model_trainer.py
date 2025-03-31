import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import tensorflow as tf  # TensorFlow backend
from tensorflow import keras  # TensorFlow's Keras API
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "SKIN Diseases.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,image_gen,train,test,val):
        try:
            logging.info("Load pre-trained MobileNetV2 (trained on 1.4M ImageNet images)")
            base_model = MobileNetV2(weights='imagenet', # Use pre-trained weights
                                     include_top=False, # Don't include final classification layer
                                     input_shape=(244, 244, 3)) # Expected input format
            
            # Freeze base layers (prevent weights from updating)
            for layer in base_model.layers:
                layer.trainable = False

            logging.info("Build a new model on top of the pre-trained base")
            classes=list(train.class_indices.keys())
            num_classes = len(classes)
            model = Sequential([
            base_model, # Pre-trained base
            GlobalAveragePooling2D(), # Reduce spatial dimensions
            Dense(512, activation='relu'), # New trainable layer
            Dropout(0.5), # Randomly disable 50% neurons to prevent overfitting
            Dense(num_classes, activation='softmax') # Final layer with num_classes i.e. 7 outputs (for 7 classes)
            ])

            # Compile model with settings
            model.compile(optimizer=Adam(learning_rate=0.001), # Optimization algorithm
            loss='categorical_crossentropy', # Loss function for multi-class
            metrics=['accuracy']) # Track accuracy during training

            transfer_learning_model = model
            # Learning rate schedule callback
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', # Watch validation loss
                                          factor=0.2, # Reduce LR(learning rate) by 80% when improvement stops
                                          patience=3, # Wait 3 epochs before reducing
                                          min_lr=1e-6)  # Minimum learning rate
            
            # Train model for 15 cycles (epochs)
            history_transfer_learning = transfer_learning_model.fit(train, # Training data
                                                                    epochs=15, # Full passes through dataset
                                                                    validation_data=val, # Validation data
                                                                    callbacks=[reduce_lr]) # Learning rate adjustment
            # Evaluation on the test set
            test_loss, test_accuracy = transfer_learning_model.evaluate(test)
            logging.info(f'Test Accuracy: {test_accuracy * 100:.2f}%')
            accuracy = test_accuracy * 100
            transfer_learning_model.save(self.model_trainer_config.trained_model_file_path,save_format='h5')
            # save_object(self.model_trainer_config.trained_model_file_path, transfer_learning_model)

            return {"accuracy": accuracy}

        except Exception as e:
            raise CustomException(e, sys)