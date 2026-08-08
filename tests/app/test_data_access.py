import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app import data_access


# applies to all tests in the file to clear cached results
@pytest.fixture(autouse=True)
def clear_cache():
    yield
    for fn in [
        data_access.get_available_seasons,
        data_access.get_available_teams,
        data_access.get_headline_stats,
        data_access.get_scored_shots,
        data_access.get_calibration_data,
        data_access.get_feature_importance_data,
    ]:
        fn.clear()


@contextmanager
def fake_get_conn(conn=None):
    yield conn or MagicMock()


## get_headline_stats tests


class TestGetHeadlineStats:

    def test_get_headline_stats(self):
        shot_count_df = pd.DataFrame({"n": [622481]})
        season_count_df = pd.DataFrame({"n": [7]})
        fake_metrics = {"metrics": {"xgboost": {"auc": 0.782, "brier": 0.0813}}}

        with patch.object(
            data_access, "get_conn", return_value=fake_get_conn()
        ), patch.object(
            data_access.pd, "read_sql", side_effect=[shot_count_df, season_count_df]
        ), patch.object(
            data_access.Path, "read_text", return_value=json.dumps(fake_metrics)
        ):
            result = data_access.get_headline_stats()

        assert result == {
            "total_shots": 622481,
            "season_count": 7,
            "test_auc": 0.782,
            "test_brier": 0.0813,
        }


## get_calibration_data test
class TestGetCalibrationData:

    def test_get_calibration_data(self):
        fake_metrics = {
            "calibration": {
                "xgboost": {
                    "bin_start": [0, 0.1, 0.2],
                    "bin_end": [0.1, 0.2, 0.3],
                    "bin_mid": [0.05, 0.15, 0.25],
                    "predicted_mean": [0.0431, 0.1529, 0.2185],
                    "actual_rate": [0.0407, 0.1696, 0.2098],
                    "count": [76830, 11560, 3209],
                }
            }
        }
        with patch.object(
            data_access.Path, "read_text", return_value=json.dumps(fake_metrics)
        ):
            result = data_access.get_calibration_data("xgboost")

        assert list(result.columns) == [
            "bin_start",
            "bin_end",
            "bin_mid",
            "predicted_mean",
            "actual_rate",
            "count",
        ]
        expected = pd.DataFrame(
            {
                "bin_start": [0, 0.1, 0.2],
                "bin_end": [0.1, 0.2, 0.3],
                "bin_mid": [0.05, 0.15, 0.25],
                "predicted_mean": [0.0431, 0.1529, 0.2185],
                "actual_rate": [0.0407, 0.1696, 0.2098],
                "count": [76830, 11560, 3209],
            }
        )
        pd.testing.assert_frame_equal(result, expected)


## get_feature_importance test
class TestGetFeatureImportance:

    def test_get_feature_importance_data_logistic_regression(self):
        fake_metrics = {
            "feature_importance": {
                "xgboost": {"feature": ["shot_distance"], "importance": [0.4]},
                "logistic_regression": {
                    "feature": ["shot_angle"],
                    "importance": [0.25],
                },
            }
        }
        with patch.object(
            data_access.Path, "read_text", return_value=json.dumps(fake_metrics)
        ):
            result = data_access.get_feature_importance_data("xgboost")

        expected = pd.DataFrame({"feature": ["shot_distance"], "importance": [0.4]})
        pd.testing.assert_frame_equal(result, expected)
