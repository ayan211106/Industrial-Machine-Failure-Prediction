import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.exceptions import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,models):
    try:
        results = {}

        for name, (model, param_grid) in models.items():

                pipeline = Pipeline([
                    ("model", model)
                ])

                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=5,
                    scoring="average_precision",
                    n_jobs=-1,
                    verbose=1,
                    refit=True
                )

                grid.fit(X_train, y_train)
                results[name] = grid
        return results
    except Exception as e:
        raise CustomException(e, sys)
