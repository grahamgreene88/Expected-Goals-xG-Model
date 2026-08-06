import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st

from model.predict import score_shots
from pipeline.db import get_connection_pool


@contextmanager
def get_conn():
    """Checks out a pooled connection and guarantees it's returned, even on error."""
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


@st.cache_data(ttl=86400)  # option lists change once per season at most
def get_available_seasons() -> list[str]:
    query = "SELECT DISTINCT season FROM games ORDER BY season"
    with get_conn() as conn:
        df = pd.read_sql(query, conn)
    return df["season"].tolist()


@st.cache_data(ttl=86400)
def get_available_teams() -> list[str]:
    query = """
        SELECT DISTINCT (home_team_city || ' ' || home_team_name) AS team
        FROM games
        UNION
        SELECT DISTINCT (away_team_city || ' ' || away_team_name) AS team
        FROM games
        ORDER BY team
    """
    with get_conn() as conn:
        df = pd.read_sql(query, conn)
    return df["team"].tolist()


@st.cache_data(ttl=86400)
def get_headline_stats() -> dict:
    with get_conn() as conn:
        shot_count = pd.read_sql("SELECT COUNT(*) AS n FROM shots", conn)["n"].iloc[0]
        season_count = pd.read_sql(
            "SELECT COUNT(DISTINCT season) AS n FROM games", conn
        )["n"].iloc[0]

    metrics_path = Path(__file__).parent.parent / "artifacts" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())

    return {
        "total_shots": int(shot_count),
        "season_count": int(season_count),
        "test_auc": metrics["metrics"]["xgboost"]["auc"],
        "test_brier": metrics["metrics"]["xgboost"]["brier"],
    }


@st.cache_data(ttl=3600)
def get_scored_shots(
    season: str | None = None, team: str | None = None
) -> pd.DataFrame:
    """
    Queries raw shots (optionally filtered by season/team), scores them via
    score_shots(), and returns the filtered, xg-annotated DataFrame.
    """
    query = """
        SELECT s.*
        FROM shots s
        JOIN games g ON s.game_id = g.game_id
        WHERE (%(season)s IS NULL OR g.season = %(season)s)
          AND (%(team)s IS NULL OR
               (g.home_team_city || ' ' || g.home_team_name) = %(team)s OR
               (g.away_team_city || ' ' || g.away_team_name) = %(team)s)
    """
    with get_conn() as conn:
        df = pd.read_sql(query, conn, params={"season": season, "team": team})  # type: ignore[arg-type]
    return score_shots(df)


@st.cache_data(ttl=86400)
def get_calibration_data(model: str = "xgboost") -> pd.DataFrame:
    metrics_path = Path(__file__).parent.parent / "artifacts" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    return pd.DataFrame(metrics["calibration"][model])


@st.cache_data(ttl=86400)
def get_feature_importance_data(model: str = "xgboost") -> pd.DataFrame:
    metrics_path = Path(__file__).parent.parent / "artifacts" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    return pd.DataFrame(metrics["feature_importance"][model])
