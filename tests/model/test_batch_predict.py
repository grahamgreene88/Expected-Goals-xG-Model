from unittest.mock import MagicMock, patch

import pandas as pd

from model.batch_predict import get_unscored_games, score_and_persist_games


def test_score_and_persist_games_logs_zero_shot_games():
    conn = MagicMock()

    raw_shots = pd.DataFrame({"game_id": [1, 2], "event_id": [10, 20]})
    scored = pd.DataFrame(
        {"game_id": [1], "event_id": [10], "xg": [0.15]}
    )  # game 2 filtered out entirely

    with patch("model.batch_predict.pd.read_sql", return_value=raw_shots), patch(
        "model.batch_predict.score_shots", return_value=scored
    ), patch("model.batch_predict.execute_values") as mock_execute_values:

        result = score_and_persist_games(conn, game_ids=[1, 2], model_version="v3")

    assert result == 1  # only 1 row scored

    # second execute_values call is the scored_games_log insert
    log_call_args = mock_execute_values.call_args_list[1]
    log_rows = log_call_args.args[2]
    assert (2, "v3", 0) in log_rows  # EDGE CASE: game 2 logged with 0 shots
    assert (1, "v3", 1) in log_rows  # normal case

    conn.commit.assert_called_once()


def test_get_unscored_games_returns_game_ids():
    conn = MagicMock()
    fake_df = pd.DataFrame({"game_id": [2023020001, 2023020002]})
    with patch(
        "model.batch_predict.pd.read_sql", return_value=fake_df
    ) as mock_read_sql:
        result = get_unscored_games(conn, model_version="v3")

    assert result == [2023020001, 2023020002]
    # check the model_version param was passed through
    _, kwargs = mock_read_sql.call_args
    assert kwargs["params"]["model_version"] == "v3"


def test_get_unscored_games_empty():
    conn = MagicMock()
    with patch(
        "model.batch_predict.pd.read_sql", return_value=pd.DataFrame({"game_id": []})
    ):
        result = get_unscored_games(conn, model_version="v3")
    assert result == []
