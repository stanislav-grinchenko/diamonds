import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
# Import other necessary libraries here

def keep_not_null(row) :
    """keep_not_null is a function that checks if there are any null values in the row. 
    If there are, it returns False, otherwise it returns True.

    Args:
        row represents a row of the dataframe

    Returns:
        bool: True if there are no null values in the row, False otherwise
    """
    if 0 in row.values : return False
    return True

def load_data(cache = True) -> pd.DataFrame:
    """
    Load the diamonds dataset.

    Parameters
    ----------
    cache : bool, optional
        Whether to cache the dataset, by default True

    Returns
    -------
    pd.DataFrame
        The diamonds dataset
    """
    return sns.load_dataset('diamonds')

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The diamonds dataset

    Returns
    -------
    pd.DataFrame
        The cleaned diamonds dataset
    """
    return df[df.apply(keep_not_null,axis=1)]

def preprocess_data(df: pd.DataFrame, preprocsessor: Pipeline) -> pd.DataFrame:
    """
    Preprocess the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned diamonds dataset

    Returns
    -------
    pd.DataFrame
        The preprocessed diamonds dataset
    """
    df_cat = df.select_dtypes(include="category")
    cat_pipe = Pipeline(
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
    
    
    
    

def create_train_test_X_y(df: pd.DataFrame, test_size: float, random_state: int) ->tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Create the feature matrix X and target vector y from the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed diamonds dataset

    Returns
    -------
    (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series)
        The feature matrix X and target vector y
    """
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=test_size, random_state=random_state)
    return X_train, y_train, X_test, y_test



if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    X_train, y_train, X_test, y_test  = create_train_test_X_y(df_clean)
    # df_preprocessed = preprocess_data(df_clean)
    