import os

DATA_PATH= "data"
MODEL_PATH = "models"

MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY", "local")

# define model names as constants to avoid typos and ensure consistency across the codebase
model_name = "Linear_Regression", "Random Forest Regressor", "KNeighborsRegressor", "Support Vector Regressor"

model = {lin, forest, knn, svr}

metrics = {
    "MAE": {},
    "MAPE": {},
    "MSE": {},
    "R2": {}
}


