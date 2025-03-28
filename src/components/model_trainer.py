import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Best model with tuned hyperparameters + balanced weights
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight="balanced",  # Fix for bias
                random_state=42
            )

            model.fit(X_train, y_train)
            save_object(self.model_trainer_config.trained_model_file_path, model)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            f1 = f1_score(y_test, y_pred, average='weighted') * 100
            roc_auc = roc_auc_score(y_test, y_pred) * 100

            logging.info(f"Final Model Accuracy: {acc}%")
            logging.info(f"Final Model F1-score: {f1}%")
            logging.info(f"Final Model ROC-AUC: {roc_auc}%")

            return {"accuracy": acc, "f1_score": f1, "roc_auc": roc_auc}

        except Exception as e:
            raise CustomException(e, sys)