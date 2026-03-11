from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
<<<<<<< HEAD
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
=======
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR as SNNRegressor
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split   
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
 
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6

import loguru

from diamonds.registry import save_model, load_model

logger = loguru.logger

def create_model(model_name: str) -> BaseEstimator:
    """
    Create an untrained model with the best hyperparameters found during tuning.

    Parameters
    ----------
    model_name : str
        The name of the model (e.g. "ridge", "random_forest")

    Returns
    -------
    BaseEstimator
        The model ready to be fitted
    """
<<<<<<< HEAD
    model = RandomForestRegressor(n_estimators=50, max_depth=10)
    logger.info(f"Created the model: {model_name}")
    return model
=======
    if model_name == "LinearRegression" or model_name == "baseline":
        return LinearRegression()
    elif model_name == "RandomForestRegressor":
        return RandomForestRegressor()
    elif model_name == "KNNRegressor":
        return KNNRegressor()
    elif model_name == "SNNRegressor":
        return SNNRegressor()
        
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6

def create_preproc() -> Pipeline:
    """
    Create a preprocessing pipeline.
    """
    cat_pipe = Pipeline(
<<<<<<< HEAD
    [ ("cat_imp",SimpleImputer(strategy="most_frequent"))
      ,("ohe",OneHotEncoder(drop="first",sparse_output=False))
        ])
    num_pipe = Pipeline(
    [("knn_imp", KNNImputer(n_neighbors=5))
     ,("scaler", StandardScaler())
      ])
    preprocessor = ColumnTransformer(
    [("numeric",num_pipe, make_column_selector(dtype_include="number"))
    ,("categorical", cat_pipe, make_column_selector(dtype_exclude="number"))
      ]).set_output(transform="pandas")
    return preprocessor

def train_model(model, X_train, y_train , save: bool = True) -> None:
    """Train the model in place and save it."""
    model.fit(X_train, y_train)
    if save: save_model(model, "model")
=======
        [ ("cat_imp",SimpleImputer(strategy="most_frequent"))
        ,("ohe",OneHotEncoder(drop="first",sparse_output=False))
        ])
    num_pipe = Pipeline(
        [("knn_imp", KNNImputer(n_neighbors=5))
        ,("scaler", StandardScaler())
        ])
    preprocessor = ColumnTransformer(
        [("numeric",num_pipe, make_column_selector(dtype_include="number"))
        ,("categorical", cat_pipe, make_column_selector(dtype_exclude="number"))
        ]).set_output(transform="pandas")
    
    return preprocessor

def fit_preproc(preproc, X_train):
    preproc.fit(X_train)
    return preproc


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6

def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    # NB : mae, mse, r2_score, mape
    # Only print the metrics for now
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
<<<<<<< HEAD
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"Evaluation metrics: MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2f}, MAPE={mape:.2%}")
    return {"mae": mae, "mse": mse, "r2": r2, "mape": mape}

=======
    r2 = r2_score(y_test, y_pred)   
    mape = mean_absolute_percentage_error(y_test, y_pred)
    

    return {"MAE": mae, "MSE": mse, "R2": r2, "MAPE": mape}


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
    return model.predict(X)
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6
