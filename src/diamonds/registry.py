
import os
import pickle
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_FOLDER
import mlflow
 

def save_model(estimator: BaseEstimator, name : str):
    """Save the model to the specified path."""
    # Implement the logic to save the model (e.g., using pickle, joblib, etc.)
    estimator_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
    with open(estimator_path, "wb") as f:
        pickle.dump(estimator, f)

def load_model(name: str) -> BaseEstimator:
    """Load the model from the specified path."""
    estimator_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
    with open(estimator_path, "rb") as f:
        estimator = pickle.load(f)
    return estimator

def register_model_mlflow(model, model_name: str, parameters: dict, evaluation_metrics: dict) -> None:
    """
    Save the trained model to the registry (MLflow).
    """
    # 1. Set the tracking URI to the MLflow server
    mlflow.set_tracking_uri("http://localhost:5001")
    # 2. Set the Experiment name
    mlflow.set_experiment("Diamond_Experiment")

    with mlflow.start_run(run_name=f"train_{model_name}") as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(evaluation_metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

def load_registered_model_mlflow(model_name: str):
    """
    Load a model from the registry (MLflow) by its name.
    """
    mlflow.set_tracking_uri("http://localhost:5001")
    model_uri = f"models:/{model_name}/latest"
    model = mlflow.sklearn.load_model(model_uri)
    return model
