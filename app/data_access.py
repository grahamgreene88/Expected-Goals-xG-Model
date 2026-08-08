import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st

from model.features import build_features
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


@st.cache_data(ttl=43200)
def get_scored_shots(
    season: str | None = None, team: str | None = None
) -> pd.DataFrame:
    """
    Queries raw shots (optionally filtered by season/team), joined against
    scored_shots for the most recently scored xg value per shot.
    """
    query = """
        SELECT DISTINCT ON (ss.game_id, ss.event_id)
            s.*, ss.xg, ss.model_version, ss.scored_at
        FROM shots s
        JOIN games g ON s.game_id = g.game_id
        JOIN scored_shots ss
          ON s.game_id = ss.game_id AND s.event_id = ss.event_id
        WHERE (%(season)s IS NULL OR g.season = %(season)s)
          AND (%(team)s IS NULL OR
               (g.home_team_city || ' ' || g.home_team_name) = %(team)s OR
               (g.away_team_city || ' ' || g.away_team_name) = %(team)s)
        ORDER BY ss.game_id, ss.event_id, ss.scored_at DESC
    """
    with get_conn() as conn:
        df = pd.read_sql(query, conn, params={"season": season, "team": team})  # type: ignore[arg-type]

    xg_cols = df[["game_id", "event_id", "xg", "model_version", "scored_at"]]
    raw_shots = df.drop(columns=["xg", "model_version", "scored_at"])

    featured = build_features(raw_shots)
    return featured.merge(xg_cols, on=["game_id", "event_id"], how="inner")


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


@st.cache_data(ttl=3600)
def get_team_season_summary(season: str | None = None) -> pd.DataFrame:
    query = """
        WITH shot_team AS (
            SELECT
                ss.xg, s.is_goal, g.season,
                CASE WHEN s.is_home_team THEN g.home_team_city || ' ' || g.home_team_name
                     ELSE g.away_team_city || ' ' || g.away_team_name END AS shooting_team,
                CASE WHEN s.is_home_team THEN g.away_team_city || ' ' || g.away_team_name
                     ELSE g.home_team_city || ' ' || g.home_team_name END AS defending_team
            FROM scored_shots ss
            JOIN shots s ON ss.game_id = s.game_id AND ss.event_id = s.event_id
            JOIN games g ON s.game_id = g.game_id
            WHERE (%(season)s IS NULL OR g.season = %(season)s)
        )
        SELECT
            season, team,
            SUM(GF) AS "GF", SUM(xGF) AS "xGF",
            SUM(GA) AS "GA", SUM(xGA) AS "xGA"
        FROM (
            SELECT season, shooting_team AS team, is_goal::int AS GF, xg AS xGF, 0 AS GA, 0 AS xGA FROM shot_team
            UNION ALL
            SELECT season, defending_team AS team, 0 AS GF, 0 AS xGF, is_goal::int AS GA, xg AS xGA FROM shot_team
        ) combined
        GROUP BY season, team
    """
    with get_conn() as conn:
        df = pd.read_sql(query, conn, params={"season": season})  # type: ignore[arg-type]
    df["xGF_pct"] = df["xGF"] / (df["xGF"] + df["xGA"])
    df["xG_diff"] = df["xGF"] - df["xGA"]
    return df
