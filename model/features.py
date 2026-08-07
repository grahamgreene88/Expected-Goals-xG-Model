"""
Feature engineering for the xG model.

Entry point: build_features(df) takes the raw shots DataFrame
(queried directly from the shots table) and returns a DataFrame
of engineered features ready for model training or inference.
"""

import numpy as np
import pandas as pd

## Constants

# Net position in normalized coordinate space.
# Shooting team always attacks positive x, so the opponent net is at (89, 0).
NET_X = 89.0
NET_Y = 0.0

# Shot types representing >= 1% of total shots. All others collapse to "other".
VALID_SHOT_TYPES = {"wrist", "snap", "slap", "tip-in", "backhand", "deflected"}

FEATURE_COLS = [
    # Positional
    "shot_distance",
    "shot_angle",
    "x_coord",
    "y_coord",
    "is_behind_net",
    # Situational
    "period",
    "shooting_team_strength_diff",
    "is_home_team",
    # Categorical
    "shot_type",
    "zone",
]

TARGET_COL = "is_goal"


## Helper functions


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that should not be included in xG modeling:
      - Shots where coordinate normalization failed
      - Shootout events
      - Shots on an empty net (opponent's goalie is pulled)
    """
    # Only rows where coordinate normalization succeeded (df["coord_normalized"] == True)
    mask = df["coord_normalized"]

    # Exclude shootout shots
    mask &= df["period_type"] != "SO"

    # Exclude shots on an empty net
    # The opponent's net is empty when:
    # Shooting team is home AND away goalie is pulled
    # Shooting team is away AND home goalie is pulled
    opponent_goalie_pulled = (df["is_home_team"] & df["away_goalie_pulled"]) | (
        ~df["is_home_team"] & df["home_goalie_pulled"]
    )
    # exlude shots where oponent goalie is pulled
    mask &= ~opponent_goalie_pulled

    filtered = df[mask].copy()

    if len(filtered) == 0:
        raise ValueError("No rows remain after filtering — check input data.")

    return filtered


def _compute_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute shot geometry features.

    Assumes coordinates are normalized so the shooting team always attacks
    positive x, with the opponent net at (NET_X, NET_Y).

    Features added:
      shot_distance : Euclidean distance from shot location to net (feet)
      shot_angle    : Angle in degrees from the goal-line extended.
                      Shots in front of net have 0–90°
                      Shots behind the net have 90–180°
      is_behind_net : True when x_coord > NET_X
    """
    # x-coordinate distance
    dx = NET_X - df["x_coord"]
    # y-coordinate distance
    dy = NET_Y - df["y_coord"]

    # Euclidean distance calculated using pythagorean theorem
    df["shot_distance"] = np.sqrt(dx**2 + dy**2)

    # arctan2(abs(y), dx) gives the angle from positive x-axis
    # dx is negative behind the net, so arctan2 naturally extends beyond 90°
    # shots directly in front of net are 0°
    # shots at the side are 90°
    # shots behind the net are greater than 90°
    df["shot_angle"] = np.degrees(np.arctan2(np.abs(df["y_coord"]), dx))

    df["is_behind_net"] = (df["x_coord"] > NET_X).astype(bool)

    return df


def _clean_shot_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode rare shot types (< 1% of total shots) and nulls to 'other'.

    Retained categories: wrist, snap, slap, tip-in, backhand, deflected.
    All others (wrap-around, poke, bat, between-legs, cradle, null) → 'other'.
    """
    df["shot_type"] = (
        df["shot_type"]
        .fillna("other")
        .apply(lambda x: x if x in VALID_SHOT_TYPES else "other")
    )
    return df


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure downstream sklearn ColumnTransformer sees the right dtypes.
      - period cast to str so it is treated as categorical, not ordinal numeric.
      - is_home_team and is_behind_net cast to int (sklearn prefers 0/1 over bool).
    """
    df["period"] = df["period"].astype(str)
    df["is_home_team"] = df["is_home_team"].astype(int)
    df["is_behind_net"] = df["is_behind_net"].astype(int)
    return df


## Main function


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for feature engineering.

    Takes the raw shots DataFrame (queried directly from the shots table)
    and returns a clean DataFrame of engineered features ready for input
    into an sklearn Pipeline.

    game_id is retained in the output so callers can perform season-based
    train/test splits without needing to re-join.

    Parameters
    ----------
    df : pd.DataFrame
        Raw shots DataFrame. Must contain all columns present in the shots table.

    Returns
    -------
    pd.DataFrame
        Columns: game_id, FEATURE_COLS, TARGET_COL.
        Filtered (no shootouts, no empty-net shots, coord_normalized only).
        Index reset.
    """
    df = _apply_filters(df)
    df = _compute_geometry(df)
    df = _clean_shot_type(df)
    df = _cast_types(df)

    output_cols = ["game_id"] + ["event_id"] + FEATURE_COLS + [TARGET_COL]
    return df[output_cols].reset_index(drop=True)
