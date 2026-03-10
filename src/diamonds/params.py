"""
params.py
---------
Central configuration for the diamonds project.

All values can be overridden via environment variables so that the same
codebase works locally, in CI, and in production without code changes.

Usage
-----
    from diamonds.params import MODEL_PATH, MODEL_REGISTRY
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

<<<<<<< HEAD
# Root of the repository (two levels above this file: src/diamonds/params.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH: Path = Path(
    os.environ.get("DATA_PATH", str(_REPO_ROOT / "data"))
)

MODEL_PATH: Path = Path(
    os.environ.get("MODEL_PATH", str(_REPO_ROOT / "models"))
)

# Ensure the models directory exists at import time so callers never have
# to remember to create it manually.
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model registry backend
# ---------------------------------------------------------------------------

# Supported values: "local" | "gcs"
MODEL_REGISTRY: str = os.environ.get("MODEL_REGISTRY", "local").lower()

if MODEL_REGISTRY not in ("local", "gcs"):
    raise ValueError(
        f"Unsupported MODEL_REGISTRY value '{MODEL_REGISTRY}'. "
        "Accepted values: 'local', 'gcs'."
    )

# GCS bucket name — only required when MODEL_REGISTRY == "gcs"
GCS_BUCKET: str | None = os.environ.get("GCS_BUCKET", None)

if MODEL_REGISTRY == "gcs" and not GCS_BUCKET:
    raise EnvironmentError(
        "MODEL_REGISTRY is set to 'gcs' but GCS_BUCKET env var is not set."
    )

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME: str   = os.environ.get("DEFAULT_MODEL_NAME", "RandomForestRegressor")
DEFAULT_TEST_SIZE: float  = float(os.environ.get("DEFAULT_TEST_SIZE", "0.2"))
DEFAULT_RANDOM_STATE: int = int(os.environ.get("DEFAULT_RANDOM_STATE", "42"))
=======
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


>>>>>>> ca5cc9026c6b1d316e7d6ac0e8f0b2cbc133bed5
