
import os
import sys
import pickle
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.exceptions import CustomException
from src.logger import logging


class ModelExplainer:

    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"
        self.output_dir = "artifacts"

    def initiate_model_explainer(self):

        try:
            logging.info("Loading trained model and transformed data")

            with open(self.model_path, "rb") as file:
                saved_model = pickle.load(file)

            with open(self.preprocessor_path, "rb") as file:
                preprocessor = pickle.load(file)

            X_train = np.load(
                "artifacts/X_train_transformed.npy"
            )
            X_test = np.load(
                "artifacts/X_test_transformed.npy"
            )
            y_test = np.load(
                "artifacts/y_test.npy"
            )

            model = (
                saved_model.named_steps["model"]
                if hasattr(saved_model, "named_steps")
                else saved_model
            )

            rng = np.random.default_rng(42)
            sample_size = min(500, len(X_test))

            indices = rng.choice(
                len(X_test),
                size=sample_size,
                replace=False
            )

            X_explain = X_test[indices]

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif getattr(shap_values, "ndim", 0) == 3:
                shap_values = shap_values[:, :, 1]

            features = preprocessor.get_feature_names_out()
            feature_names = [name.replace("num__", "").replace("cat__", "") for name in features]


            os.makedirs(self.output_dir, exist_ok=True)

            # 1. SHAP summary plot
            shap.summary_plot(
                shap_values,
                X_explain,
                feature_names=feature_names,
                show=False
            )
            plt.savefig(
                "artifacts/shap_summary.png",
                dpi=200,
                bbox_inches="tight"
            )
            plt.close()

            # 2. Global feature importance plot
            shap.summary_plot(
                shap_values,
                X_explain,
                feature_names=feature_names,
                plot_type="bar",
                show=False
            )
            plt.savefig(
                "artifacts/shap_importance.png",
                dpi=200,
                bbox_inches="tight"
            )
            plt.close()

            os.makedirs("artifacts/shap_plots", exist_ok=True)

            normal_indices = np.where(y_test == 0)[0]

            sample_index_0 = normal_indices[0]

            X_sample = X_test[[sample_index_0]]

            probability_0 = model.predict_proba(X_sample)[0, 1]
            prediction_0 = int(probability_0 >= 0.25)

            print("Actual:", y_test[sample_index_0])
            print("Failure probability:", probability_0)
            print("Prediction:", prediction_0)

            sample_shap_0 = explainer(X_sample)

            sample_shap_0.feature_names = feature_names

            plt.figure()
            shap.plots.waterfall(sample_shap_0[0], show=False)
            plt.savefig("artifacts/shap_plots/normal_machine_waterfall.png",bbox_inches="tight")
            plt.close()

            failure_indices = np.where(y_test == 1)[0]

            sample_index_1 = failure_indices[0]

            X_sample_1 = X_test[[sample_index_1]]

            probability_1 = model.predict_proba(X_sample_1)[0, 1]
            prediction_1 = int(probability_1 >= 0.25)

            print("Actual:", y_test[sample_index_1])
            print("Failure probability:", probability_1)
            print("Prediction:", prediction_1)

            sample_shap_1 = explainer(X_sample_1)

            sample_shap_1.feature_names = feature_names

            plt.figure()
            shap.plots.waterfall(sample_shap_1[0], show=False)
            plt.savefig("artifacts/shap_plots/failure_machine_waterfall.png",bbox_inches="tight")
            plt.close()

            # 3. Numeric feature importance
            importance = pd.DataFrame({
                "Feature": feature_names,
                "Mean |SHAP|": np.abs(shap_values).mean(axis=0)
            }).sort_values(
                "Mean |SHAP|",
                ascending=False
            )

            print("\nGlobal SHAP Feature Importance:")
            print(importance.to_string(index=False))

            logging.info("SHAP explanations generated successfully")

            return importance

        except Exception as e:
            raise CustomException(e, sys)
