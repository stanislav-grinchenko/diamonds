<<<<<<< HEAD
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR as SVRRegressor
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

# ---------------------------------------------------------------------------
# Supported models registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {
    "LinearRegression": LinearRegression,
    "baseline":         LinearRegression,
    "Ridge":            Ridge,
    "RandomForestRegressor":     RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "KNNRegressor": KNNRegressor,
    "SVRRegressor": SVRRegressor,
}


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(model_name: str, **kwargs) -> BaseEstimator:
    """Instantiate a regression model by name.

    Parameters
    ----------
    model_name : str
        One of: "LinearRegression", "baseline", "Ridge",
        "RandomForestRegressor", "GradientBoostingRegressor",
        "KNNRegressor", "SVRRegressor".
    **kwargs
        Optional hyperparameters forwarded to the model constructor.

    Returns
    -------
    BaseEstimator
        An unfitted scikit-learn estimator.

    Raises
    ------
    ValueError
        If *model_name* is not in the registry.
    """
    if model_name not in _MODEL_REGISTRY:
        supported = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{model_name}'. Supported models: {supported}"
        )
    return _MODEL_REGISTRY[model_name](**kwargs)
=======
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

>>>>>>> ca5cc9026c6b1d316e7d6ac0e8f0b2cbc133bed5


<<<<<<< HEAD
def list_models() -> list[str]:
    """Return the list of supported model names."""
    return sorted(_MODEL_REGISTRY.keys())

=======
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

>>>>>>> ca5cc9026c6b1d316e7d6ac0e8f0b2cbc133bed5

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def create_preproc() -> ColumnTransformer:
    """Build a preprocessing pipeline for the diamonds dataset.

    Numeric columns  → KNN imputation → StandardScaler
    Categorical cols → most-frequent imputation → OneHotEncoder (drop first)

    Returns
    -------
    ColumnTransformer
        An unfitted preprocessor.
    """
<<<<<<< HEAD
    num_pipe = Pipeline([
        ("knn_imp", KNNImputer(n_neighbors=5)),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("cat_imp", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(drop="first", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric",     num_pipe, make_column_selector(dtype_include="number")),
            ("categorical", cat_pipe, make_column_selector(dtype_exclude="number")),
        ],
        remainder="drop",
    ).set_output(transform="pandas")
    return preprocessor


def fit_preproc(preproc: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessor on the training features.

    Parameters
    ----------
    preproc : ColumnTransformer
        An unfitted preprocessor (from :func:`create_preproc`).
    X_train : pd.DataFrame
        Training feature matrix (without the target column).

    Returns
    -------
    ColumnTransformer
        The same preprocessor, now fitted.
    """
    return preproc.fit(X_train)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """Fit a regression model on the (already preprocessed) training data.

    Parameters
    ----------
    model : BaseEstimator
        An unfitted scikit-learn estimator.
    X_train : pd.DataFrame
        Preprocessed training feature matrix.
    y_train : pd.Series
        Training target vector.

    Returns
    -------
    BaseEstimator
        The fitted model.
    """
    return model.fit(X_train, y_train)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute regression metrics for a fitted model on the test set.

    Metrics returned
    ----------------
    MAE   : Mean Absolute Error
    RMSE  : Root Mean Squared Error  (more interpretable than MSE)
    MSE   : Mean Squared Error
    R2    : Coefficient of determination
    MAPE  : Mean Absolute Percentage Error

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn estimator.
    X_test : pd.DataFrame
        Preprocessed test feature matrix.
    y_test : pd.Series
        True target values.

    Returns
    -------
    dict[str, float]
        Dictionary of metric names to values.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "MSE":  mse,
        "R2":   r2_score(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
    }


def print_metrics(metrics: dict[str, float]) -> None:
    """Pretty-print a metrics dictionary returned by :func:`evaluate_model`.

    Parameters
    ----------
    metrics : dict[str, float]
        Metrics as returned by :func:`evaluate_model`.
    """
    width = max(len(k) for k in metrics)
    print("-" * (width + 14))
    for name, value in metrics.items():
        print(f"  {name:<{width}} : {value:.4f}")
    print("-" * (width + 14))


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a fitted model.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn estimator.
    X : pd.DataFrame
        Preprocessed feature matrix (output of the fitted preprocessor).

    Returns
    -------
    np.ndarray
        Array of predicted target values.
    """
    return model.predict(X)
=======
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

>>>>>>> ca5cc9026c6b1d316e7d6ac0e8f0b2cbc133bed5
