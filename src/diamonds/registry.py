
import os
import pickle
from sklearn.base import BaseEstimator
<<<<<<< HEAD
from diamonds.params import MODEL_FOLDER
import mlflow
 
=======
from diamonds.params import MODEL_REGISTRY 
import pickle
import os
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6

def save_model(estimator: BaseEstimator, name : str):
    """Save the model to the specified path."""
<<<<<<< HEAD
    # Implement the logic to save the model (e.g., using pickle, joblib, etc.)
    estimator_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
    with open(estimator_path, "wb") as f:
        pickle.dump(estimator, f)
=======
    with open(path, "wb") as f:
        pickle.dump(model, f)
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6

def load_model(name: str) -> BaseEstimator:
    """Load the model from the specified path."""
<<<<<<< HEAD
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
=======
    # Implement the logic to load the model (e.g., using pickle, joblib, etc.)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6
