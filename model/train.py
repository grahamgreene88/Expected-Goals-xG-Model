"""
Trains two xG models and saves fitted sklearn Pipeline objects to artifacts/:
  - artifacts/lr_pipeline.pkl   (logistic regression baseline)
  - artifacts/xgb_pipeline.pkl  (XGBoost)

Each saved pipeline includes preprocessing — it accepts the raw feature
DataFrame directly and requires no external scaler or encoder at inference time.

Usage:
    python -m model.train
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
import numpy as np

from model.features import FEATURE_COLS, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

TRAIN_PATH = Path("data/processed/train.parquet")
TEST_PATH = Path("data/processed/test.parquet")
ARTIFACTS_DIR = Path("artifacts")

## Feature Column Groups

# Scaled for logistic regression, as-is for XGBoost
NUMERIC_COLS = [
    "shot_distance",
    "shot_angle",
    "x_coord",
    "y_coord",
    "shooting_team_strength_diff",
]

# One-hot encoded for both models
CATEGORICAL_COLS = ["period", "shot_type", "zone"]

# Already 0 or 1; passed through as-is for both model
BINARY_COLS = ["is_behind_net", "is_home_team"]

RANDOM_STATE = 42

## Pipeline builders


def _build_lr_pipeline() -> Pipeline:
    """
    Logistic regression pipeline.

    Preprocessing:
      - Numeric features: StandardScaler (required for correct regularization)
      - Categorical features: OneHotEncoder
      - Binary features: passed through unchanged

    max_iter=1000 avoids convergence warnings on scaled data.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_COLS,
            ),
            ("bin", "passthrough", BINARY_COLS),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _build_xgb_pipeline() -> Pipeline:
    """
    XGBoost pipeline.

    Preprocessing:
      - Numeric features: passed through (trees are scale-invariant)
      - Categorical features: OneHotEncoder
      - Binary features: passed through unchanged
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="passthrough",  # numeric and binary pass through untouched
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    verbosity=0,
                ),
            ),
        ]
    )


## Evaluation helper


def _log_metrics(label: str, y_true, y_pred_proba) -> None:
    brier = brier_score_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    log.info(f"  {label:<25} Brier: {brier:.4f}   AUC: {auc:.4f}")


## Main


def train() -> None:
    # Load data ---
    log.info("Loading parquet files...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    log.info(f"  Train: {len(train_df):,} rows")
    log.info(f"  Test:  {len(test_df):,} rows")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].astype(int)

    # Logistic regression
    log.info("\nTraining logistic regression...")
    lr_pipeline = _build_lr_pipeline()
    lr_pipeline.fit(X_train, y_train)

    log.info("  Metrics:")
    _log_metrics("LR train", y_train, lr_pipeline.predict_proba(X_train)[:, 1])
    _log_metrics("LR test", y_test, lr_pipeline.predict_proba(X_test)[:, 1])

    # XGBoost
    log.info("\nTraining XGBoost...")
    xgb_pipeline = _build_xgb_pipeline()
    xgb_pipeline.fit(X_train, y_train)

    log.info("  Metrics:")
    _log_metrics("XGB train", y_train, xgb_pipeline.predict_proba(X_train)[:, 1])
    _log_metrics("XGB test", y_test, xgb_pipeline.predict_proba(X_test)[:, 1])

    # Save artifacts
    log.info("\nSaving pipelines...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    lr_path = ARTIFACTS_DIR / "lr_pipeline.pkl"
    xgb_path = ARTIFACTS_DIR / "xgb_pipeline.pkl"

    with open(lr_path, "wb") as f:
        pickle.dump(lr_pipeline, f)
    with open(xgb_path, "wb") as f:
        pickle.dump(xgb_pipeline, f)

    log.info(f"  Saved {lr_path}")
    log.info(f"  Saved {xgb_path}")
    log.info("\nDone.")


if __name__ == "__main__":
    train()
