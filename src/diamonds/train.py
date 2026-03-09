import data
import model
import registry
import params
import os
def train(
    model_name: str = "RandomForestRegressor",
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
    df = data.load_data()
    df_cleaned = data.clean_data(df)
    X_test, y_test, X_train, y_train = data.create_train_test_X_y(df_cleaned, test_size, random_state)
    # 2) Model + preprocessing
    preproc = model.create_preproc()
    preproc = model.fit_preproc(preproc, X_train)
    X_train_transformed = preproc.transform(X_train)
    X_test_transformed = preproc.transform(X_test)
    mdl = model.create_model(model_name)
    mdl = model.train_model(mdl, X_train_transformed, y_train)
    # 3) Evaluation
    metrics = model.evaluate_model(mdl, X_test_transformed, y_test)
    print(metrics)
    # 4) Persistence
    path =  os.path.join(params.MODEL_PATH, model_name + ".pkl")
    registry.save_model(mdl, path)


if __name__ == "__main__":
    train()

