"""
Tests for:
  - model/evaluate.py      : compute_metrics, compute_calibration, get_feature_importance
"""

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.evaluate import (
    _clean_feature_names,
    compute_calibration,
    compute_metrics,
    get_feature_importance,
)

## Helpers


def make_perfect_predictions(n=1000, goal_rate=0.1):
    """Perfect classifier: predicted probability equals actual outcome.
    Used to test Brier score and AUC to ensure they return 0 and 1 respectively."""
    y_true = np.array([1] * int(n * goal_rate) + [0] * int(n * (1 - goal_rate)))
    return y_true, y_true.astype(float)


def make_null_predictions(n=1000, goal_rate=0.1):
    """Null model: always predicts the mean goal rate."""
    y_true = np.array([1] * int(n * goal_rate) + [0] * int(n * (1 - goal_rate)))
    y_pred = np.full(n, goal_rate)
    return y_true, y_pred


def make_simple_lr_pipeline() -> Pipeline:
    """
    Minimal fitted sklearn Pipeline with named steps matching production structure.
    Uses a single numeric feature for simplicity.
    """
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 0, 1])

    pipeline = Pipeline(
        [
            ("preprocessor", StandardScaler()),
            ("classifier", LogisticRegression()),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


## compute_metrics tests


class TestComputeMetrics:

    def test_returns_expected_keys(self):
        y_true, y_pred = make_null_predictions()
        result = compute_metrics(y_true, y_pred)
        assert set(result.keys()) == {"brier", "auc", "log_loss", "null_brier"}

    def test_perfect_model_has_zero_brier(self):
        y_true, y_pred = make_perfect_predictions()
        result = compute_metrics(y_true, y_pred)
        assert result["brier"] == 0.0

    def test_null_model_brier_matches_null_brier(self):
        # A model that always predicts the mean should score exactly null_brier
        y_true, y_pred = make_null_predictions(goal_rate=0.1)
        result = compute_metrics(y_true, y_pred)
        assert abs(result["brier"] - result["null_brier"]) < 1e-4

    def test_better_model_beats_null_brier(self):
        # Any model with real signal should beat the null baseline
        y_true, y_pred = make_perfect_predictions()
        result = compute_metrics(y_true, y_pred)
        assert result["brier"] < result["null_brier"]

    def test_perfect_model_has_auc_of_one(self):
        y_true, y_pred = make_perfect_predictions()
        result = compute_metrics(y_true, y_pred)
        assert result["auc"] == 1.0


## compute_calibration tests


class TestComputeCalibration:

    def test_returns_expected_columns(self):
        y_true, y_pred = make_null_predictions()
        result = compute_calibration(y_true, y_pred)
        expected_cols = {
            "bin_start",
            "bin_end",
            "bin_mid",
            "predicted_mean",
            "actual_rate",
            "count",
        }
        assert set(result.columns) == expected_cols

    def test_skips_empty_bins(self):
        # All predictions in 0.0–0.1 range — higher bins should be skipped
        y_true = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0.05] * 10)
        result = compute_calibration(y_true, y_pred, n_bins=10)
        assert len(result) == 1

    def test_bin_count_sums_to_total_shots(self):
        y_true, y_pred = make_null_predictions(n=1000)
        result = compute_calibration(y_true, y_pred, n_bins=10)
        assert result["count"].sum() == 1000

    def test_bin_mid_is_midpoint_of_bin_edges(self):
        y_true, y_pred = make_null_predictions()
        result = compute_calibration(y_true, y_pred, n_bins=10)
        for _, row in result.iterrows():
            expected_mid = round((row["bin_start"] + row["bin_end"]) / 2, 4)
            assert abs(row["bin_mid"] - expected_mid) < 1e-4

    def test_actual_rate_within_zero_and_one(self):
        y_true, y_pred = make_null_predictions()
        result = compute_calibration(y_true, y_pred)
        assert (result["actual_rate"] >= 0).all()
        assert (result["actual_rate"] <= 1).all()

    def test_n_bins_controls_maximum_bins(self):
        y_true = np.zeros(100)
        y_pred = np.linspace(0, 0.99, 100)
        result = compute_calibration(y_true, y_pred, n_bins=5)
        assert len(result) <= 5


## Feature importance tests


class TestCleanFeatureNames:

    def test_strips_num_prefix(self):
        assert _clean_feature_names(["num__shot_distance"]) == ["shot_distance"]

    def test_strips_cat_prefix(self):
        assert _clean_feature_names(["cat__shot_type_wrist"]) == ["shot_type_wrist"]

    def test_strips_bin_prefix(self):
        assert _clean_feature_names(["bin__is_home_team"]) == ["is_home_team"]

    def test_strips_remainder_prefix(self):
        assert _clean_feature_names(["remainder__x_coord"]) == ["x_coord"]

    def test_handles_mixed_prefixes(self):
        names = ["num__shot_distance", "cat__shot_type_wrist", "remainder__x_coord"]
        assert _clean_feature_names(names) == [
            "shot_distance",
            "shot_type_wrist",
            "x_coord",
        ]

    def test_leaves_unprefixed_names_unchanged(self):
        assert _clean_feature_names(["shot_distance"]) == ["shot_distance"]


class TestGetFeatureImportance:

    def test_returns_expected_columns(self):
        pipeline = make_simple_lr_pipeline()
        # Patch get_feature_names_out since StandardScaler doesn't have it natively
        with patch.object(
            pipeline.named_steps["preprocessor"],
            "get_feature_names_out",
            return_value=["num__feature_0"],
        ):
            result = get_feature_importance(pipeline)
        assert set(result.columns) == {"feature", "importance"}

    def test_sorted_descending_by_importance(self):
        pipeline = make_simple_lr_pipeline()
        with patch.object(
            pipeline.named_steps["preprocessor"],
            "get_feature_names_out",
            return_value=["num__feature_0"],
        ):
            result = get_feature_importance(pipeline)
        assert result["importance"].is_monotonic_decreasing

    def test_raises_for_unsupported_classifier(self):
        from sklearn.base import BaseEstimator, ClassifierMixin

        class NoImportanceClassifier(BaseEstimator, ClassifierMixin):
            pass

        pipeline = Pipeline(
            [
                ("preprocessor", StandardScaler()),
                ("classifier", NoImportanceClassifier()),
            ]
        )
        with patch.object(
            pipeline.named_steps["preprocessor"],
            "get_feature_names_out",
            return_value=["feature_0"],
        ):
            with pytest.raises(ValueError, match="Cannot extract feature importance"):
                get_feature_importance(pipeline)
