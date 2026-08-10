from prefect import flow, get_run_logger, task

from model.batch_predict import (
    get_unscored_games,
    score_and_persist_games,
)
from model.mlflow_helpers import get_production_model_version
from pipeline.db import get_connection

# Obtain model version to put in scored_shots table
MODEL_NAME = "nhl_xg_xgboost"


@task
def find_unscored_games(model_version: str):
    with get_connection() as conn:
        return get_unscored_games(conn, model_version)


@task
def predict_shots_batch(game_ids, model_version: str) -> int:
    with get_connection() as conn:
        return score_and_persist_games(conn, game_ids, model_version)


@flow(name="batch-predict-shots")
def batch_predict_flow(model_version: str | None = None):
    logger = get_run_logger
    model_version = model_version or get_production_model_version(MODEL_NAME)

    game_ids = find_unscored_games(model_version)
    count = predict_shots_batch(game_ids, model_version)
    logger.info(f"Scored {count} shots under model_version={model_version}")
    return count


if __name__ == "__main__":
    batch_predict_flow()
