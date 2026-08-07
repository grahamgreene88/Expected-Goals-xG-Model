from prefect import flow, task

from model.batch_predict import (
    MODEL_VERSION,
    get_unscored_games,
    score_and_persist_games,
)
from pipeline.db import get_connection


@task
def find_unscored_games(model_version: str):
    conn = get_connection()
    try:
        return get_unscored_games(conn, model_version)
    finally:
        conn.close()


@task
def predict_shots_batch(game_ids, model_version: str) -> int:
    conn = get_connection()
    try:
        return score_and_persist_games(conn, game_ids, model_version)
    finally:
        conn.close()


@flow(name="batch-predict-shots")
def batch_predict_flow(model_version: str = MODEL_VERSION):
    game_ids = find_unscored_games(model_version)
    count = predict_shots_batch(game_ids, model_version)
    print(f"Scored {count} shots under model_version={model_version}")
    return count


if __name__ == "__main__":
    batch_predict_flow()
