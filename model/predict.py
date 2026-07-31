"""
Inference interface for the xG model.

Entry point: score_shots(df) takes the raw shots DataFrame (queried directly
from the shots table), applies feature engineering, and returns a DataFrame
with an xg column added.

Filtered-out shots (shootouts, empty net, unnormalized coordinates) are
excluded from the output entirely.

Usage:
    from model.predict import score_shots
    scored_df = score_shots(raw_df)
"""

import pickle
from pathlib import Path

import pandas as pd

from model.features import FEATURE_COLS, build_features

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "xgb_pipeline.pkl"


def _load_pipeline():
    """Load the fitted XGBoost pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run python -m model.train to generate it."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def score_shots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of raw shots with xG predictions.

    Applies feature engineering and filtering (shootouts, empty net shots,
    and unnormalized coordinates are excluded). The returned DataFrame
    contains only scoreable shots with an xg column appended.

    Parameters
    ----------
    df : pd.DataFrame
        Raw shots DataFrame queried directly from the shots table.

    Returns
    -------
    pd.DataFrame
        Filtered shots with all original columns plus:
            xg : float, predicted goal probability (0–1)
        Index reset.
    """
    pipeline = _load_pipeline()

    # Build features — applies all filters, returns clean feature DataFrame
    features_df = build_features(df)

    # Score using feature columns only
    xg = pipeline.predict_proba(features_df[FEATURE_COLS])[:, 1]

    # Attach xg to the features DataFrame and return
    # game_id is already in features_df from build_features()
    result = features_df.copy()
    result["xg"] = xg

    return result.reset_index(drop=True)
