
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_REGISTRY 
import pickle
import os

def save_model(model, path):
    """Save the model to the specified path."""
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path) -> BaseEstimator:
    """Load the model from the specified path."""
    # Implement the logic to load the model (e.g., using pickle, joblib, etc.)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model