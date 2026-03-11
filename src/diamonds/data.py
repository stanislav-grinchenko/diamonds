import pandas as pd
<<<<<<< HEAD
import loguru
import os

# For loading data
import seaborn as sns

from diamonds.params import DATA_PATH
from diamonds.model import create_preproc
from diamonds.registry import save_model, load_model

from sklearn.model_selection import train_test_split

logger = loguru.logger

=======
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6
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

def load_data() -> pd.DataFrame:
    """
    Load the diamonds dataset.

    Parameters
    ----------
    
    Returns
    -------
    pd.DataFrame
        The diamonds dataset
    """
<<<<<<< HEAD
    logger.info("Loading the diamonds dataset...")
    csv_path = os.path.join(DATA_PATH,"raw", "diamonds.csv")
    if not os.path.exists(csv_path):
        logger.info("Caching the diamonds dataset...")
        df = sns.load_dataset("diamonds")
        df.to_csv(csv_path, index=False)
        logger.info("✅ Diamonds dataset cached successfully.")
    else:
        logger.info("Loading the diamonds dataset from cache...")
        df = pd.read_csv(csv_path)
    return df
            
    
=======
    return sns.load_dataset('diamonds')

>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6
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
<<<<<<< HEAD
    rows = len(df)
    def keep_not_null(row) :
        if 0 in row.values : return False
        return True
    df_clean = df[df.apply(keep_not_null,axis=1)]
    logger.info(f"Cleaned the diamonds dataset: {rows} rows -> {len(df_clean)} rows")
    return df_clean


def preprocess_data( X: pd.DataFrame
                    , train: bool = True) -> pd.DataFrame:
=======
    return df[df.apply(keep_not_null,axis=1)]

def preprocess_data(df: pd.DataFrame, preprocsessor: Pipeline) -> pd.DataFrame:
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6
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
<<<<<<< HEAD
    # Instantier la pipeline 
    if train : 
        preprocessor = create_preproc()
        preprocessor.fit(X)
        save_model(preprocessor, "preprocessor")
    else :
        preprocessor = load_model("preprocessor")
    df_preprocessed = preprocessor.transform(X)
    logger.info(f"Preprocessed the diamonds dataset: {X.shape} -> {df_preprocessed.shape}") 
    return df_preprocessed

def create_X_y(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) ->tuple[pd.DataFrame, pd.Series]:
=======
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
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6
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
<<<<<<< HEAD
    
    X = df.drop(columns="price")
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)

    return X_train, X_test, y_train, y_test
=======
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=test_size, random_state=random_state)
    return X_train, y_train, X_test, y_test
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6



if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
<<<<<<< HEAD
    df_preprocessed = preprocess_data(df_clean)
    X, y = create_X_y(df_preprocessed)
=======
    X_train, y_train, X_test, y_test  = create_train_test_X_y(df_clean)
    # df_preprocessed = preprocess_data(df_clean)
    
>>>>>>> 7a7695159ff1ae0eca8583123e385a2942e0a0f6
