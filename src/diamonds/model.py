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


def list_models() -> list[str]:
    """Return the list of supported model names."""
    return sorted(_MODEL_REGISTRY.keys())


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
