
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_explainer import ModelExplainer


def run_training_pipeline():
    # 1. Data ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2. Data transformation
    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = (
        transformation.initiate_data_transformation(
            train_path, test_path
        )
    )

    # 3. Model training and evaluation
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)

    # 4. SHAP explanations
    explainer = ModelExplainer()
    explainer.initiate_model_explainer()


if __name__ == "__main__":
    run_training_pipeline()
