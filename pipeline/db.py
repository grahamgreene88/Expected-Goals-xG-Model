import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pipeline.config import DB_CONFIG, VALID_PIPELINE_STATUSES


# Connection
def get_connection():
    """Create and return a new PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


# Pipeline Log
def is_game_processed(game_id: int) -> bool:
    """
    Check if a game has already been successfully processed.
    Returns True if a 'success' record exists in pipeline_log.
    """
    query = """
        SELECT 1 FROM pipeline_log
        WHERE game_id = %s AND status = 'success'
        LIMIT 1
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (game_id,))
            return cur.fetchone() is not None


def write_pipeline_log(
    game_id: int,
    status: str,
    shots_inserted: int | None = None,
    error_message: str | None = None,
) -> None:
    """
    Write a record to pipeline_log for a processed game.
    """
    if status not in VALID_PIPELINE_STATUSES:
        raise ValueError(f"Invalid pipeline status: {status!r}")

    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))

    query = """
        INSERT INTO pipeline_log
            (game_id, status, shots_inserted, error_message, created_at_utc, created_at_et)
        VALUES
            (%s, %s, %s, %s, %s, %s)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    game_id,
                    status,
                    shots_inserted,
                    error_message,
                    now_utc,
                    now_et,
                ),
            )
        conn.commit()


# Games
def upsert_game(game: dict) -> None:
    """
    Upsert a single game record into the games table.
    On conflict, updates all non-key fields.
    """
    query = """
        INSERT INTO games (
            game_id, season, game_type, game_date, game_state,
            game_schedule_state, away_team_id, away_team_city,
            away_team_name, away_team_abbrev, away_team_score,
            home_team_id, home_team_city, home_team_name,
            home_team_abbrev, home_team_score, last_period_type,
            created_at_utc, created_at_et
        )
        VALUES (
            %(game_id)s, %(season)s, %(game_type)s, %(game_date)s, %(game_state)s,
            %(game_schedule_state)s, %(away_team_id)s, %(away_team_city)s,
            %(away_team_name)s, %(away_team_abbrev)s, %(away_team_score)s,
            %(home_team_id)s, %(home_team_city)s, %(home_team_name)s,
            %(home_team_abbrev)s, %(home_team_score)s, %(last_period_type)s,
            %(created_at_utc)s, %(created_at_et)s
        )
        ON CONFLICT (game_id) DO UPDATE SET
            game_state           = EXCLUDED.game_state,
            game_schedule_state  = EXCLUDED.game_schedule_state,
            away_team_score      = EXCLUDED.away_team_score,
            home_team_score      = EXCLUDED.home_team_score,
            last_period_type     = EXCLUDED.last_period_type
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, game)
        conn.commit()


# Shots
def upsert_shots(shots: list[dict]) -> int:
    """
    Upsert a list of shot records into the shots table.
    Returns the number of shots upserted.
    """
    if not shots:
        return 0

    columns = [
        "game_id",
        "event_id",
        "sort_order",
        "period",
        "period_type",
        "time_in_period",
        "time_remaining",
        "situation_code",
        "strength_state",
        "away_goalie_pulled",
        "home_goalie_pulled",
        "shooting_team_strength_state",
        "shooting_team_strength_diff",
        "shooter_id",
        "goalie_id",
        "shooter_team_id",
        "shooter_team_abbrev",
        "opponent_team_abbrev",
        "is_home_team",
        "event_type",
        "x_coord_raw",
        "y_coord_raw",
        "x_coord",
        "y_coord",
        "coord_normalized",
        "zone",
        "shot_type",
        "is_goal",
        "miss_reason",
        "created_at_utc",
        "created_at_et",
    ]

    values = [tuple(shot[col] for col in columns) for shot in shots]

    query = f"""
        INSERT INTO shots ({", ".join(columns)})
        VALUES %s
        ON CONFLICT (game_id, event_id) DO UPDATE SET
            sort_order                   = EXCLUDED.sort_order,
            situation_code               = EXCLUDED.situation_code,
            strength_state               = EXCLUDED.strength_state,
            away_goalie_pulled           = EXCLUDED.away_goalie_pulled,
            home_goalie_pulled           = EXCLUDED.home_goalie_pulled,
            shooting_team_strength_state = EXCLUDED.shooting_team_strength_state,
            shooting_team_strength_diff  = EXCLUDED.shooting_team_strength_diff,
            x_coord_raw                  = EXCLUDED.x_coord_raw,
            y_coord_raw                  = EXCLUDED.y_coord_raw,
            x_coord                      = EXCLUDED.x_coord,
            y_coord                      = EXCLUDED.y_coord,
            coord_normalized             = EXCLUDED.coord_normalized,
            zone                         = EXCLUDED.zone,
            shot_type                    = EXCLUDED.shot_type,
            is_goal                      = EXCLUDED.is_goal,
            miss_reason                  = EXCLUDED.miss_reason
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, values)
        conn.commit()

    return len(shots)
