
import os
import pickle
import pandas as pd

from src.exceptions import CustomException
import sys


class PredictionPipeline:

    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"
        self.threshold_path = "artifacts/threshold.pkl"

    def predict(self, input_data: pd.DataFrame):
        try:
           
            with open(self.model_path, "rb") as file:
                saved_model = pickle.load(file)

            with open(self.preprocessor_path, "rb") as file:
                preprocessor = pickle.load(file)

            threshold = 0.25
            if os.path.exists(self.threshold_path):
                with open(self.threshold_path, "rb") as file:
                    threshold = float(pickle.load(file))

            input_data = input_data.copy()
            input_data["Temperature difference"] = (
                input_data["Process temperature [K]"]
                - input_data["Air temperature [K]"]
            )
            input_data["Mechanical power"] = (
                input_data["Torque [Nm]"]
                * input_data["Rotational speed [rpm]"]
            )

            feature_columns = [
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "Temperature difference",
                "Mechanical power"
            ]
            input_data = input_data[feature_columns]

            transformed_data = preprocessor.transform(input_data)

            model = (
                saved_model.named_steps["model"]
                if hasattr(saved_model, "named_steps")
                else saved_model
            )

            probability = float(
                model.predict_proba(transformed_data)[0, 1]
            )
            prediction = int(probability >= threshold)

            return {
                "failure_probability": probability,
                "threshold": threshold,
                "prediction": prediction,
                "result": (
                    "Machine Failure"
                    if prediction == 1
                    else "Normal Machine"
                )
            }

        except Exception as e:
            raise CustomException(e, sys)
