from diamonds.data import (load_data, clean_data
                            , preprocess_data, create_X_y)
from diamonds.model import create_model, train_model, evaluate_model
from diamonds.registry import save_model, load_model, register_model_mlflow, load_registered_model_mlflow



def train(
    model_name: str = "baseline",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Simple end‑to‑end pipeline:

    - load and clean the raw data
    - preprocess it and build X, y
    - split into train / test
    - build the model and preprocessing
    - train, evaluate, and save the trained model
    """
    # 1) Data
    df = load_data()
    df_clean = clean_data(df)
    # 2) Model + preprocessing
    X_train, X_test, y_train, y_test = create_X_y(df_clean, test_size=test_size, random_state=random_state)
    X_train_preproc = preprocess_data(X_train, train=True)
    X_test_preproc = preprocess_data(X_test, train=False)
    
    model = create_model(model_name)
    train_model(model, X_train_preproc, y_train)
    # 3) Evaluation
    evaluation_metrics = evaluate_model(model, X_test_preproc, y_test)
    
    # 4) Register the model in MLflow
    register_model_mlflow(model, model_name, parameters={"test_size": test_size, "random_state": random_state}, evaluation_metrics=evaluation_metrics)
    loaded_model = load_registered_model_mlflow(model_name)
    print(loaded_model)

if __name__ == "__main__":
    train()

