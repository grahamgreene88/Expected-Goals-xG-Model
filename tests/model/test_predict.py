"""
Tests for:
  - model/predict.py       : _load_pipeline (missing model error)
"""

from unittest.mock import patch

## _load_pipeline tests


class TestLoadPipeline:

    @patch("model.predict.mlflow.sklearn.load_model")
    def test_loads_production_model(self, mock_load_model):
        from model.predict import _load_pipeline

        _load_pipeline()

        mock_load_model.assert_called_once_with("models:/nhl_xg_xgboost@production")
