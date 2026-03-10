"""
registry.py
-----------
Model persistence layer for the diamonds project.

Supports two backends controlled by the ``MODEL_REGISTRY`` param:

* ``"local"``  — saves/loads models as ``.joblib`` files on the local filesystem.
* ``"gcs"``    — saves/loads models to/from a Google Cloud Storage bucket.

joblib is preferred over pickle for scikit-learn objects because it is
faster on large numpy arrays and more robust across library versions.

Usage
-----
    from diamonds.registry import save_model, load_model
"""

import io
import logging
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

from diamonds.params import GCS_BUCKET, MODEL_PATH, MODEL_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_filepath(model_name: str) -> Path:
    """Return the local ``.joblib`` path for *model_name*."""
    return MODEL_PATH / f"{model_name}.joblib"


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------

def _save_local(model: BaseEstimator, model_name: str) -> Path:
    path = _model_filepath(model_name)
    joblib.dump(model, path)
    logger.info("Model saved locally → %s", path)
    return path


def _load_local(model_name: str) -> BaseEstimator:
    path = _model_filepath(model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model found at '{path}'. "
            "Train and save the model first."
        )
    model = joblib.load(path)
    logger.info("Model loaded from   → %s", path)
    return model


# ---------------------------------------------------------------------------
# GCS backend
# ---------------------------------------------------------------------------

def _save_gcs(model: BaseEstimator, model_name: str) -> str:
    """Upload a serialised model to GCS and return the blob URI."""
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required for GCS support. "
            "Install it with: pip install google-cloud-storage"
        ) from exc

    blob_name = f"models/{model_name}.joblib"
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(buffer, content_type="application/octet-stream")

    uri = f"gs://{GCS_BUCKET}/{blob_name}"
    logger.info("Model uploaded to   → %s", uri)
    return uri


def _load_gcs(model_name: str) -> BaseEstimator:
    """Download and deserialise a model from GCS."""
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required for GCS support. "
            "Install it with: pip install google-cloud-storage"
        ) from exc

    blob_name = f"models/{model_name}.joblib"
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(
            f"No model blob found at gs://{GCS_BUCKET}/{blob_name}."
        )

    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)

    model = joblib.load(buffer)
    logger.info("Model downloaded from gs://%s/%s", GCS_BUCKET, blob_name)
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_model(model: BaseEstimator, model_name: str) -> None:
    """Persist *model* using the configured backend.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn estimator.
    model_name : str
        Logical name used as the filename stem (e.g. ``"RandomForestRegressor"``).
    """
    if MODEL_REGISTRY == "gcs":
        _save_gcs(model, model_name)
    else:
        _save_local(model, model_name)


def load_model(model_name: str) -> BaseEstimator:
    """Load a previously saved model from the configured backend.

    Parameters
    ----------
    model_name : str
        Logical name used when the model was saved.

    Returns
    -------
    BaseEstimator
        The deserialised, fitted estimator.

    Raises
    ------
    FileNotFoundError
        If no saved model is found for *model_name*.
    """
    if MODEL_REGISTRY == "gcs":
        return _load_gcs(model_name)
    return _load_local(model_name)
