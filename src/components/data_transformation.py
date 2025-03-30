import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
import tensorflow as tf  # TensorFlow backend
from tensorflow import keras  # TensorFlow's Keras API
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Image preprocessing utility

@dataclass
class DataTransformation:
    def initiate_data_transformation(self, train_images, test_images, train_set, val_set):
        try:
            # Image Generator to preprocess images and feed into model
            image_gen = ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet_v2.preprocess_input) # Specific normalization/rescales pixel value for MobileNetV2

            # Train set generator
            train = image_gen.flow_from_dataframe(dataframe= train_set,x_col="filepaths",# Column with image paths
                                                  y_col="labels",# Column with labels
                                                  target_size=(244,244), # Resize all images to 244x244 pixels
                                                  color_mode='rgb', # Use color images
                                                  class_mode="categorical", # For multi-class classification
                                                  batch_size=32, # Process 32 images at a time
                                                  shuffle=False            #do not shuffle data and Keep original order (for inspection)
                                                  )
            # Test set generator
            test = image_gen.flow_from_dataframe(dataframe= test_images,x_col="filepaths", y_col="labels",
                                                 target_size=(244,244),
                                                 color_mode='rgb',
                                                 class_mode="categorical",
                                                 batch_size=32,
                                                 shuffle= False
                                                 )
            # Validation set generator
            val = image_gen.flow_from_dataframe(dataframe= val_set,x_col="filepaths", y_col="labels",
                                                target_size=(244,244),
                                                color_mode= 'rgb',
                                                class_mode="categorical",
                                                batch_size=32,
                                                shuffle=False
                                                )

            logging.info("Data transformation complete.")
            return image_gen, train, test, val

        except Exception as e:
            raise CustomException(e, sys)