from sklearn.pipeline import Pipeline

from model.predict import _load_pipeline, get_production_model_version


def test_exported_production_model_loads():
    assert isinstance(_load_pipeline(), Pipeline)
    assert get_production_model_version().startswith("nhl_xg_xgboost:v")
