from sklearn.base import BaseEstimator, Pipeline, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, mean_absolute_percentage
from sklearn.metrics import mean_absolute_percentage_error as mape
from .params import model_name, model


def create_model(model_name: str) -> BaseEstimator:
    if model_name == "Linear_Regression":
        return LinearRegression() and model = lin
    elif model_name == "Random Forest Regressor":
        return RandomForestRegressor() and model = forest
    elif model_name == "KNeighborsRegressor":
        return KNeighborsRegressor() and model = knn
    elif model_name == "Support Vector Regressor":
        return SVR() and model = svr
    else:
        raise ValueError(f"Model name '{model_name}' is not recognized. Please choose from 'Linear_Regression', 'Random Forest Regressor', 'KNeighborsRegressor', or 'Support Vector Regressor'.")


def create_preproc() -> Pipeline:
    """
    Create a preprocessing pipeline.
    """
    pass

# compléter après avec le bloc-notes

def train_model(model, X_train_scaled, y_train):
    if model is None:
        raise ValueError("Model is None. Please create a model first.")
    for name, m in model.items():
        if m is None:
            raise ValueError(f"Model '{name}' is None. Please ensure all models are created properly.")
        elif not hasattr(m, "fit"):
            raise ValueError(f"The model '{name}' does not have a fit method. Please provide a valid model for training.")
        elif not isinstance(X_train_scaled, (pd.DataFrame, np.ndarray)) or not isinstance(y_train, (pd.Series, np.ndarray)):
            raise ValueError("X_train_scaled and y_train must be pandas DataFrame/Series or numpy arrays. Please provide valid training data.") 
        elif len(X_train_scaled) == 0 or len(y_train) == 0:
            raise ValueError("X_train_scaled and y_train cannot be empty. Please provide non-empty training data for training.")    
        elif len(X_train_scaled) != len(y_train):
            raise ValueError("X_train_scaled and y_train must have the same number of samples. Please ensure they are aligned.")
        elif hasattr(m, "fit") and (not hasattr(m, "predict") or not callable(getattr(m, "predict"))):
            raise ValueError(f"The model '{name}' does not have a valid predict method. Please provide a valid model for training.")
        elif hasattr(m, "fit") and (not hasattr(m, "score") or not callable(getattr(m, "score"))):
            raise ValueError(f"The model '{name}' does not have a valid score method. Please provide a valid model for training.")
        elif hasattr(m, "fit") and (not hasattr(m, "get_params") or not callable(getattr(m, "get_params"))):
            raise ValueError(f"The model '{name}' does not have a valid get_params method. Please provide a valid model for training.") 
        elif hasattr(m, "fit") and (not hasattr(m, "set_params") or not callable(getattr(m, "set_params"))):
            raise ValueError(f"The model '{name}' does not have a valid set_params method. Please provide a valid model for training.") 
        elif hasattr(m, "fit") and (not hasattr(m, "get_feature_names_out") or not callable(getattr(m, "get_feature_names_out"))):
            raise ValueError(f"The model '{name}' does not have a valid get_feature_names_out method. Please provide a valid model for training.")
        elif hasattr(m, "fit") and (not hasattr(m, "get_feature_names_in") or not callable(getattr(m, "get_feature_names_in"))):
            raise ValueError(f"The model '{name}' does not have a valid get_feature_names_in method. Please provide a valid model for training.")
        elif m.fit(X_train_scaled, y_train) is None:
            raise ValueError(f"Training the model '{name}' did not return a fitted model. Please ensure the fit method is implemented correctly.")
        elif 
    for name, m in model.items():
        m.fit(X_train_scaled, y_train)


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    # NB : mae, mse, r2_score, mape
    if mae is None or mse is None or r2_score is None or mape is None:
        raise ValueError("One or more evaluation metrics are None. Please ensure all metrics are calculated.")
    elif not isinstance(mae, (int, float)) or not isinstance(mse, (int, float)) or not isinstance(r2_score, (int, float)) or not isinstance(mape, (int, float)):
        raise ValueError("One or more evaluation metrics are not numeric. Please ensure all metrics are numeric values.")
    elif r2_score < 0 or r2_score > 1:
        raise ValueError("R2 score must be between 0 and 1. Please ensure the R2 score is calculated correctly.")
    elif mape < 0 or mape > 100:
        raise ValueError("MAPE must be between 0 and 100. Please ensure the MAPE is calculated correctly.")
    elif mae < 0 or mse < 0:
        raise ValueError("MAE and MSE cannot be negative. Please ensure they are calculated correctly.")
    return {"mae": mae, "mse": mse, "r2_score": r2_score, "mape": mape}


def predict(model, X):
    """
    Make predictions using the trained model.

    Parameters
    ----------
    model : any
        The trained model
    X : pd.DataFrame
        The raw data

    Returns
    -------
    pd.Series
        The predicted values
    """
    if model is None:
        raise ValueError("Model is None. Please create a model first.")
    elif X_test is None or y_test is None:
        raise ValueError("X_test and y_test cannot be None. Please provide test data for evaluation.")
    elif len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("X_test and y_test cannot be empty. Please provide non-empty test data for evaluation.")
    elif len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of samples. Please ensure they are aligned.")
    elif not hasattr(model, "predict"):
        raise ValueError("The provided model does not have a predict method. Please provide a valid model for evaluation.")
    y_pred = model.predict(X_test)
    pass

