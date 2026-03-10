import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def keep_not_null(row: pd.Series) -> bool:
    """Return True if the row contains no zero or NaN values.

    In the diamonds dataset, physical dimensions (x, y, z) and other
    numeric columns should never legitimately be 0, so zeros are treated
    as missing/erroneous values alongside NaN.

    Parameters
    ----------
    row : pd.Series
        A single row of the DataFrame.

    Returns
    -------
    bool
        True when neither zeros nor NaN values are found in the row.
    """
    numeric_values = row.select_dtypes(include="number")
    if numeric_values.isna().any():
        return False
    if (numeric_values == 0).any():
        return False
    return True


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(cache: bool = True) -> pd.DataFrame:
    """Load the diamonds dataset from seaborn.

    Parameters
    ----------
    cache : bool, optional
        Whether seaborn should cache the downloaded dataset, by default True.

    Returns
    -------
    pd.DataFrame
        The raw diamonds dataset.
    """
    return sns.load_dataset("diamonds", cache=cache)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that contain zero or NaN values in any numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        The raw diamonds dataset.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame with a reset index.
    """
    mask = df.apply(keep_not_null, axis=1)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def create_train_test_X_y(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the cleaned dataset into train and test feature/target pairs.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned diamonds dataset.
    test_size : float, optional
        Proportion of the dataset to include in the test split, by default 0.2.
    random_state : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    if "price" not in df.columns:
        raise ValueError("DataFrame must contain a 'price' column.")

    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()
    print(f"Raw dataset shape     : {df.shape}")

    df_clean = clean_data(df)
    print(f"Cleaned dataset shape : {df_clean.shape}")
    print(f"Rows removed          : {df.shape[0] - df_clean.shape[0]}")

    X_train, X_test, y_train, y_test = create_train_test_X_y(df_clean)
    print(f"X_train shape         : {X_train.shape}")
    print(f"X_test  shape         : {X_test.shape}")
