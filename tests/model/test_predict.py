"""
Tests for:
  - model/predict.py       : _load_pipeline (missing model error)
"""

import pytest

## _load_pipeline tests


class TestLoadPipeline:

    def test_raises_file_not_found_for_missing_model(self, tmp_path):
        from model.predict import _load_pipeline

        with pytest.raises(FileNotFoundError, match="Run python -m model.train"):
            _load_pipeline(path=tmp_path / "missing.pkl")
