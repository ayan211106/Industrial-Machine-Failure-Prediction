import os
import sys
import numpy as np

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_path, test_path):

        try:
            logging.info("Reading transformed train and test data")

            train_array = train_path
            test_array = test_path

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1].astype(int)

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1].astype(int)

            logging.info("Splitting features and target completed")

            models = {
                "Logistic Regression": (
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42
                    ),
                    {
                        "model__C": [0.01, 0.1, 1, 10],
                        "model__solver": ["liblinear", "lbfgs"]
                    }
                ),

                "Decision Tree": (
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=42
                    ),
                    {
                        "model__max_depth": [3, 5, 10, 15, None],
                        "model__min_samples_split": [2, 5, 10],
                        "model__min_samples_leaf": [1, 2, 5, 10],
                        "model__criterion": ["gini", "entropy"]
                    }
                ),

                "Random Forest": (
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1
                    ),
                    {
                        "model__n_estimators": [100, 200, 300],
                        "model__max_depth": [5, 10, 15, None],
                        "model__min_samples_split": [2, 5, 10],
                        "model__min_samples_leaf": [1, 2, 5],
                        "model__max_features": ["sqrt", "log2"]
                    }
                ),

                "Gradient Boosting": (
                    GradientBoostingClassifier(random_state=42),
                    {
                        "model__n_estimators": [100, 200],
                        "model__learning_rate": [0.01, 0.05, 0.1],
                        "model__max_depth": [2, 3, 5],
                        "model__min_samples_split": [2, 5, 10],
                        "model__min_samples_leaf": [1, 2, 5]
                    }
                ),

                "XGBoost": (
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1
                    ),
                    {
                        "model__n_estimators": [100, 200, 300],
                        "model__max_depth": [3, 5, 7],
                        "model__learning_rate": [0.01, 0.05, 0.1],
                        "model__subsample": [0.8, 1.0],
                        "model__colsample_bytree": [0.8, 1.0]
                    }
                ),

                "CatBoost": (
                    CatBoostClassifier(
                        verbose=False,
                        random_state=42,
                        thread_count=1,
                        allow_writing_files=False
                    ),
                    {
                        "model__iterations": [100, 200, 300],
                        "model__depth": [4, 6, 8],
                        "model__learning_rate": [0.01, 0.05, 0.1],
                        "model__l2_leaf_reg": [1, 3, 5]
                    }
                )
            }

            results:dict=evaluate_models(X_train=X_train,y_train=y_train,models=models)


            best_model_name = max( results,key=lambda name: results[name].best_score_)
            best_grid = results[best_model_name]
            final_model = results[best_model_name].best_estimator_

            logging.info(f"Best model selected: {best_model_name}")
            logging.info(f"Best CV PR-AUC: {results[best_model_name].best_score_:.4f}")

            threshold = 0.25

            y_prob = final_model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            test_pr_auc = average_precision_score(y_test, y_prob)
            test_roc_auc = roc_auc_score(y_test, y_prob)
            test_balanced_accuracy = balanced_accuracy_score(
                y_test, y_pred
            )

            logging.info("Final test evaluation completed")

            print("\nSelected Model:", best_model_name)
            print("Best CV PR-AUC:", best_grid.best_score_)
            print("Decision Threshold:", threshold)

            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))

            print("Balanced Accuracy:", test_balanced_accuracy)
            print("ROC-AUC:", test_roc_auc)
            print("PR-AUC:", test_pr_auc)
            print("Precision:", precision_score(
                y_test, y_pred, zero_division=0
            ))
            print("Recall:", recall_score(
                y_test, y_pred, zero_division=0
            ))
            print("F1 Score:", f1_score(
                y_test, y_pred, zero_division=0
            ))

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model
            )

            threshold_path = os.path.join(
                "artifacts", "threshold.pkl"
            )
            save_object(
                file_path=threshold_path,
                obj=threshold
            )

            logging.info("Winning model and threshold saved successfully")

            model_report = {
                name: {
                    "best_cv_pr_auc": grid.best_score_,
                    "best_params": grid.best_params_
                }
                for name, grid in results.items()
            }

            for name, report in model_report.items():
                print( f"{name:<22} | "
                f"CV PR-AUC: {report['best_cv_pr_auc']:.4f}")

            return {
                "best_model_name": best_model_name,
                "best_cv_pr_auc": best_grid.best_score_,
                "test_pr_auc": test_pr_auc,
                "test_roc_auc": test_roc_auc,
                "test_balanced_accuracy": test_balanced_accuracy,
                "threshold": threshold,
                "model_path": (
                    self.model_trainer_config.trained_model_file_path
                ),
                "model_report": model_report
            }

        except Exception as e:
            raise CustomException(e, sys)




            