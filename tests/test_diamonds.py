# tests/test_diamonds.py
import pandas as pd
import seaborn as sns
import pytest

from src.diamonds.data import load_data, clean_data, keep_not_null
from src.diamonds.model import (
    create_model, create_preproc, fit_preproc,
    train_model, evaluate_model, predict
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR as SNNRegressor
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def raw_df():
    """Raw diamonds dataset (uncleaned)."""
    return load_data()


@pytest.fixture(scope="session")
def cleaned_df(raw_df):
    """Cleaned diamonds dataset."""
    return clean_data(raw_df)


@pytest.fixture(scope="session")
def split_data(cleaned_df):
    """Train/test split of features and target."""
    X = cleaned_df.drop(columns=["price"])
    y = cleaned_df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def fitted_preproc(split_data):
    """Fitted preprocessing pipeline."""
    X_train, _, _, _ = split_data
    preproc = create_preproc()
    return fit_preproc(preproc, X_train)


@pytest.fixture(scope="session")
def trained_linear_model(split_data, fitted_preproc):
    """Trained LinearRegression model."""
    X_train, _, y_train, _ = split_data
    X_train_t = fitted_preproc.transform(X_train)
    model = create_model("LinearRegression")
    return train_model(model, X_train_t, y_train)


# ---------------------------------------------------------------------------
# Module existence
# ---------------------------------------------------------------------------

def test_module_exists():
    import src.diamonds as diamonds
    assert diamonds is not None


# ---------------------------------------------------------------------------
# data.py — load_data
# ---------------------------------------------------------------------------

def test_load_data_returns_dataframe(raw_df):
    assert isinstance(raw_df, pd.DataFrame)


def test_load_data_not_empty(raw_df):
    assert raw_df.shape[0] > 0


def test_load_data_expected_columns(raw_df):
    expected_cols = {"carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"}
    assert expected_cols.issubset(set(raw_df.columns))


def test_load_data_shape_matches_seaborn(raw_df):
    reference = sns.load_dataset("diamonds")
    assert raw_df.shape == reference.shape


# ---------------------------------------------------------------------------
# data.py — keep_not_null
# ---------------------------------------------------------------------------

def test_keep_not_null_returns_true_when_no_zeros():
    row = pd.Series({"A": 1, "B": 2, "C": 3})
    assert keep_not_null(row) is True


def test_keep_not_null_returns_false_when_zero_present():
    row = pd.Series({"A": 1, "B": 0, "C": 3})
    assert keep_not_null(row) is False


def test_keep_not_null_returns_false_when_all_zeros():
    row = pd.Series({"A": 0, "B": 0})
    assert keep_not_null(row) is False


def test_keep_not_null_single_nonzero_value():
    row = pd.Series({"A": 5})
    assert keep_not_null(row) is True


# ---------------------------------------------------------------------------
# data.py — clean_data
# ---------------------------------------------------------------------------

def test_clean_data_returns_dataframe(cleaned_df):
    assert isinstance(cleaned_df, pd.DataFrame)


def test_clean_data_removes_zero_rows(cleaned_df):
    """No row in the cleaned dataset should contain a 0 in any column."""
    has_zeros = (cleaned_df == 0).any(axis=1)
    assert not has_zeros.any()


def test_clean_data_smaller_than_raw(raw_df, cleaned_df):
    assert cleaned_df.shape[0] <= raw_df.shape[0]


def test_clean_data_preserves_columns(raw_df, cleaned_df):
    assert list(cleaned_df.columns) == list(raw_df.columns)


# ---------------------------------------------------------------------------
# model.py — create_preproc
# ---------------------------------------------------------------------------

def test_create_preproc_returns_column_transformer():
    preproc = create_preproc()
    assert isinstance(preproc, ColumnTransformer)


def test_create_preproc_has_numeric_and_categorical():
    preproc = create_preproc()
    transformer_names = [name for name, _, _ in preproc.transformers]
    assert "numeric" in transformer_names
    assert "categorical" in transformer_names


# ---------------------------------------------------------------------------
# model.py — fit_preproc
# ---------------------------------------------------------------------------

def test_fit_preproc_returns_fitted_transformer(fitted_preproc):
    assert fitted_preproc is not None


def test_fit_preproc_can_transform(fitted_preproc, split_data):
    X_train, _, _, _ = split_data
    transformed = fitted_preproc.transform(X_train)
    assert transformed is not None
    assert transformed.shape[0] == X_train.shape[0]


# ---------------------------------------------------------------------------
# model.py — create_model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name,expected_type", [
    ("LinearRegression",    LinearRegression),
    ("baseline",            LinearRegression),
    ("RandomForestRegressor", RandomForestRegressor),
    ("KNNRegressor",        KNNRegressor),
    ("SNNRegressor",        SNNRegressor),
])
def test_create_model_returns_correct_type(model_name, expected_type):
    model = create_model(model_name)
    assert isinstance(model, expected_type)


def test_create_model_returns_estimator():
    model = create_model("LinearRegression")
    assert isinstance(model, BaseEstimator)


# ---------------------------------------------------------------------------
# model.py — train_model
# ---------------------------------------------------------------------------

def test_train_model_returns_fitted_model(trained_linear_model):
    assert trained_linear_model is not None


def test_train_model_has_coef(trained_linear_model):
    """LinearRegression exposes coef_ after fitting."""
    assert hasattr(trained_linear_model, "coef_")


# ---------------------------------------------------------------------------
# model.py — evaluate_model
# ---------------------------------------------------------------------------

def test_evaluate_model_returns_dict(trained_linear_model, fitted_preproc, split_data):
    _, X_test, _, y_test = split_data
    X_test_t = fitted_preproc.transform(X_test)
    metrics = evaluate_model(trained_linear_model, X_test_t, y_test)
    assert isinstance(metrics, dict)


def test_evaluate_model_contains_expected_keys(trained_linear_model, fitted_preproc, split_data):
    _, X_test, _, y_test = split_data
    X_test_t = fitted_preproc.transform(X_test)
    metrics = evaluate_model(trained_linear_model, X_test_t, y_test)
    assert set(metrics.keys()) == {"MAE", "MSE", "R2", "MAPE"}


def test_evaluate_model_mae_non_negative(trained_linear_model, fitted_preproc, split_data):
    _, X_test, _, y_test = split_data
    X_test_t = fitted_preproc.transform(X_test)
    metrics = evaluate_model(trained_linear_model, X_test_t, y_test)
    assert metrics["MAE"] >= 0


def test_evaluate_model_mse_non_negative(trained_linear_model, fitted_preproc, split_data):
    _, X_test, _, y_test = split_data
    X_test_t = fitted_preproc.transform(X_test)
    metrics = evaluate_model(trained_linear_model, X_test_t, y_test)
    assert metrics["MSE"] >= 0


def test_evaluate_model_r2_at_most_one(trained_linear_model, fitted_preproc, split_data):
    _, X_test, _, y_test = split_data
    X_test_t = fitted_preproc.transform(X_test)
    metrics = evaluate_model(trained_linear_model, X_test_t, y_test)
    assert metrics["R2"] <= 1.0


def test_evaluate_model_mape_non_negative(trained_linear_model, fitted_preproc, split_data):
    _, X_test, _, y_test = split_data
    X_test_t = fitted_preproc.transform(X_test)
    metrics = evaluate_model(trained_linear_model, X_test_t, y_test)
    assert metrics["MAPE"] >= 0


# ---------------------------------------------------------------------------
# model.py — predict
# ---------------------------------------------------------------------------

def test_predict_returns_array(trained_linear_model, fitted_preproc, split_data):
    _, X_test, _, _ = split_data
    X_test_t = fitted_preproc.transform(X_test)
    predictions = predict(trained_linear_model, X_test_t)
    assert predictions is not None


def test_predict_shape(trained_linear_model, fitted_preproc, split_data):
    _, X_test, _, _ = split_data
    X_test_t = fitted_preproc.transform(X_test)
    predictions = predict(trained_linear_model, X_test_t)
    assert predictions.shape == (X_test.shape[0],)


def test_predict_values_are_numeric(trained_linear_model, fitted_preproc, split_data):
    _, X_test, _, _ = split_data
    X_test_t = fitted_preproc.transform(X_test)
    predictions = predict(trained_linear_model, X_test_t)
    assert pd.api.types.is_numeric_dtype(predictions)
