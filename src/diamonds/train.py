"""
train.py
--------
End-to-end training pipeline for the diamonds price regression project.

Steps
-----
1. Load and clean the raw dataset
2. Split into train / test sets
3. Fit the preprocessor on training data only (no data leakage)
4. Transform both splits
5. Train the chosen regression model
6. Evaluate and print metrics
7. Persist the fitted model via the configured registry backend

Usage
-----
Run with defaults:
    python -m diamonds.train

Override via CLI arguments:
    python -m diamonds.train --model RandomForestRegressor --test-size 0.2 --seed 42
"""

import argparse
import logging

import diamonds.data as data
import diamonds.model as model
import diamonds.registry as registry
from diamonds.params import (
    DEFAULT_MODEL_NAME,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def train(
    model_name: str = DEFAULT_MODEL_NAME,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, float]:
    """Run the full training pipeline and return the evaluation metrics.

    Parameters
    ----------
    model_name : str
        Name of the regression model to train (must be in model._MODEL_REGISTRY).
    test_size : float
        Fraction of data held out for evaluation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        Evaluation metrics: MAE, RMSE, MSE, R2, MAPE.
    """
    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    logger.info("Loading data …")
    df = data.load_data()
    logger.info("  Raw shape      : %s", df.shape)

    df_clean = data.clean_data(df)
    logger.info("  Cleaned shape  : %s  (%d rows removed)",
                df_clean.shape, df.shape[0] - df_clean.shape[0])

    X_train, X_test, y_train, y_test = data.create_train_test_X_y(
        df_clean, test_size=test_size, random_state=random_state
    )
    logger.info("  Train / test   : %d / %d rows", len(X_train), len(X_test))

    # ------------------------------------------------------------------
    # 2. Preprocessing  (fit on train only — no data leakage)
    # ------------------------------------------------------------------
    logger.info("Fitting preprocessor …")
    preproc = model.create_preproc()
    preproc = model.fit_preproc(preproc, X_train)

    X_train_t = preproc.transform(X_train)
    X_test_t  = preproc.transform(X_test)

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    logger.info("Training model : %s …", model_name)
    mdl = model.create_model(model_name)
    mdl = model.train_model(mdl, X_train_t, y_train)

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    logger.info("Evaluating …")
    metrics = model.evaluate_model(mdl, X_test_t, y_test)
    model.print_metrics(metrics)

    # ------------------------------------------------------------------
    # 5. Persistence
    # ------------------------------------------------------------------
    logger.info("Saving model …")
    registry.save_model(mdl, model_name)
    logger.info("Done.")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a diamonds price regression model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        default=DEFAULT_MODEL_NAME,
        choices=model.list_models(),
        help="Regression model to train.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of data reserved for testing.",
    )
    parser.add_argument(
        "--seed",
        dest="random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        model_name=args.model_name,
        test_size=args.test_size,
        random_state=args.random_state,
    )
