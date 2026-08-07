import pandas as pd
from psycopg2.extras import execute_values

from model.predict import score_shots

MODEL_VERSION = "xgb_v0.1"  # bump whenever xgb_pipeline.pkl is retrained/replaced


def get_unscored_games(conn, model_version: str = MODEL_VERSION) -> list[int]:
    """Games that have at least one shot not yet scored under model_version."""

    query = """
        SELECT DISTINCT g.game_id
        FROM games g
        LEFT JOIN scored_games_log sgl
          ON g.game_id = sgl.game_id AND sgl.model_version = %(model_version)s
        WHERE sgl.game_id IS NULL
    """
    df = pd.read_sql(query, conn, params={"model_version": model_version})  # type: ignore[arg-type]
    return df["game_id"].tolist()


def score_and_persist_games(
    conn, game_ids: list[int], model_version: str = MODEL_VERSION
) -> int:
    if not game_ids:
        return 0

    raw = pd.read_sql(
        "SELECT * FROM shots WHERE game_id = ANY(%(game_ids)s)",
        conn,
        params={"game_ids": game_ids},
    )  # type: ignore[arg-type]

    scored = score_shots(raw)
    output = scored[["game_id", "event_id", "xg"]].copy()
    output["model_version"] = model_version

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO scored_shots (game_id, event_id, xg, model_version)
            VALUES %s
            ON CONFLICT (game_id, event_id, model_version) DO UPDATE
              SET xg = EXCLUDED.xg, scored_at = now()
            """,
            list(
                output[["game_id", "event_id", "xg", "model_version"]].itertuples(
                    index=False, name=None
                )
            ),
        )
        # Log every requested game as processed, even ones that scored 0 shots
        # (e.g. every shot in that game was filtered out)
        shots_per_game = output.groupby("game_id").size().to_dict()
        log_rows = [
            (gid, model_version, shots_per_game.get(gid, 0)) for gid in game_ids
        ]
        execute_values(
            cur,
            """
            INSERT INTO scored_games_log (game_id, model_version, shots_scored)
            VALUES %s
            ON CONFLICT (game_id, model_version) DO UPDATE
              SET shots_scored = EXCLUDED.shots_scored, scored_at = now()
            """,
            log_rows,
        )
    conn.commit()
    return len(output)
