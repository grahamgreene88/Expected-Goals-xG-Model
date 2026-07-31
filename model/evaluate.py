"""
Reusable evaluation functions for xG model assessment.
Designed to be called from notebooks or scripts.

Functions:
    compute_metrics        → Brier score, AUC, log loss
    compute_calibration    → predicted vs actual goal rates by probability bin
    get_feature_importance → cleaned feature names with importance scores
"""

import re

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline


## Metrics
def compute_metrics(y_true: pd.Series, y_pred_proba: np.ndarray) -> dict:
    """
    Compute standard xG evaluation metrics.

    Parameters
    ----------
    y_true       : binary series or array (1 = goal, 0 = no goal)
    y_pred_proba : predicted goal probabilities, shape (n_samples,)

    Returns
    -------
    dict with keys: brier, auc, log_loss, null_brier
        null_brier is the Brier score of a model that always predicts
        the mean goal rate — the baseline any useful model must beat.
    """
    p_bar = np.mean(y_true)
    null_brier = float(np.mean((p_bar - np.array(y_true)) ** 2))

    return {
        "brier": round(brier_score_loss(y_true, y_pred_proba), 4),
        "auc": round(roc_auc_score(y_true, y_pred_proba), 4),
        "log_loss": round(log_loss(y_true, y_pred_proba), 4),
        "null_brier": round(null_brier, 4),
    }


## Calibration
def compute_calibration(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute calibration data by binning predicted probabilities.

    A well-calibrated model should have predicted_mean ≈ actual_rate
    in every bin. Large gaps indicate the model is over- or under-confident.

    Parameters
    ----------
    y_true       : binary series or array (1 = goal, 0 = no goal)
    y_pred_proba : predicted goal probabilities, shape (n_samples,)
    n_bins       : number of equal-width bins between 0 and 1

    Returns
    -------
    pd.DataFrame with columns:
        bin_start      : lower edge of probability bin
        bin_end        : upper edge of probability bin
        bin_mid        : midpoint of bin
        predicted_mean : mean predicted probability in bin
        actual_rate    : actual goal rate in bin
        count          : number of shots in bin
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Include upper edge in the final bin to capture predictions of exactly 1.0
        if i < n_bins - 1:
            mask = (y_pred_proba >= low) & (y_pred_proba < high)
        else:
            mask = (y_pred_proba >= low) & (y_pred_proba <= high)

        count = mask.sum()
        if count == 0:
            continue

        rows.append(
            {
                "bin_start": round(low, 4),
                "bin_end": round(high, 4),
                "bin_mid": round((low + high) / 2, 4),
                "predicted_mean": round(float(y_pred_proba[mask].mean()), 4),
                "actual_rate": round(float(y_true[mask].mean()), 4),
                "count": int(count),
            }
        )

    return pd.DataFrame(rows)


## Feature Importance


def _clean_feature_names(names: list[str]) -> list[str]:
    """
    Strip sklearn ColumnTransformer prefixes from feature names.

    Handles prefixes produced by both pipelines:
      LR:  num__, cat__, bin__
      XGB: cat__, remainder__
    """
    prefixes = r"^(num__|cat__|bin__|remainder__)"
    return [re.sub(prefixes, "", name) for name in names]


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract feature importances from a fitted sklearn Pipeline.

    For XGBoost: uses gain-based feature_importances_ from the classifier.
    For LogisticRegression: uses absolute coefficient values.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline with steps 'preprocessor' and 'classifier'

    Returns
    -------
    pd.DataFrame with columns:
        feature    : cleaned feature name
        importance : importance score (higher = more important)
    Sorted by importance descending.
    """
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    raw_names = list(preprocessor.get_feature_names_out())
    clean_names = _clean_feature_names(raw_names)

    classifier_type = type(classifier).__name__

    if hasattr(classifier, "feature_importances_"):
        # XGBoost, RandomForest, etc.
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        # LogisticRegression — use absolute coefficients
        importances = np.abs(classifier.coef_[0])
    else:
        raise ValueError(
            f"Cannot extract feature importance from {classifier_type}. "
            "Classifier must have feature_importances_ or coef_ attribute."
        )

    return (
        pd.DataFrame({"feature": clean_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
