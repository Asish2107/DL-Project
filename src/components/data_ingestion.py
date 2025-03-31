import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    base_dir:str = "/Users/kasish/Desktop/JGASVEMLKNPR-PROJECT/DL-Project/notebook/"

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # Define paths to each skin disease class image folder
            Eczema_dir = os.path.join(self.ingestion_config.base_dir, "IMG_CLASSES", "1. Eczema 1677")
            Warts_Molluscum_dir = os.path.join(self.ingestion_config.base_dir, "IMG_CLASSES", "10. Warts Molluscum and other Viral Infections - 2103")
            Atopic_Dermatitis_dir = os.path.join(self.ingestion_config.base_dir, "IMG_CLASSES", "3. Atopic Dermatitis - 1.25k")
            Melanocytic_Nevi_dir = os.path.join(self.ingestion_config.base_dir, "IMG_CLASSES", "5. Melanocytic Nevi (NV) - 7970")
            Psoriasis_pictures_dir = os.path.join(self.ingestion_config.base_dir, "IMG_CLASSES", "7. Psoriasis pictures Lichen Planus and related diseases - 2k")
            Seborrheic_Keratoses_Benign_Tumors_dir = os.path.join(self.ingestion_config.base_dir, "IMG_CLASSES", "8. Seborrheic Keratoses and other Benign Tumors - 1.8k")
            Tinea_Ringworm_Candidiasis_dir = os.path.join(self.ingestion_config.base_dir, "IMG_CLASSES", "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k")

            # Initialize lists to store image filepaths and corresponding labels
            filepaths = []
            labels = []

            # List of all directory paths
            dict_list = [Eczema_dir, Warts_Molluscum_dir , Atopic_Dermatitis_dir, Melanocytic_Nevi_dir,Psoriasis_pictures_dir ,Seborrheic_Keratoses_Benign_Tumors_dir ,Tinea_Ringworm_Candidiasis_dir]

            # Class label names for each folder (used to tag each image correctly)
            class_labels = ['Eczema', 'Warts Molluscum', 'Atopic Dermatitis','Melanocytic Nevi', 'Psoriasis pictures', 'Seborrheic Keratoses Benign Tumors','Tinea Ringworm Candidiasis']

            # Loop through each directory, collect image file paths and attach respective labels
            for i, j in enumerate(dict_list):
                  flist = os.listdir(j) # List of all image files in that folder(Provided directory)
                  for f in flist:
                        fpath = os.path.join(j, f)
                        filepaths.append(fpath)
                        labels.append(class_labels[i])

            # Convert file paths and labels into pandas Series
            Fseries = pd.Series(filepaths, name="filepaths")
            Lseries = pd.Series(labels, name="labels")

            # Combine into a DataFrame (our master image-label mapping file)
            skin_data = pd.concat([Fseries, Lseries], axis=1)
            skin_df = pd.DataFrame(skin_data)

            # Count of images per class label
            logging.info(f"{skin_df['labels'].value_counts()}")
            logging.info('Dataset read as DataFrame')

            logging.info("Train test split initiated")
            # Split the dataset into train and test sets (70%-30%)
            train_images, test_images = train_test_split(skin_df, test_size=0.3, random_state=42)

            # Split again into validation (20% of total dataset)
            train_set, val_set = train_test_split(skin_df, test_size=0.2, random_state=42)
            logging.info("Ingestion completed")
            return train_images,test_images,train_set,val_set

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_images,test_images,train_set,val_set = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    image_gen, train, test, val = data_transformation.initiate_data_transformation(train_images, test_images, train_set, val_set)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(image_gen,train,test,val))