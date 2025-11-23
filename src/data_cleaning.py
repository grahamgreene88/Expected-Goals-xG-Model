import pandas as pd
import numpy as np
from pathlib import Path


def _clean_miss_reason(df: pd.DataFrame) -> pd.DataFrame:
    """Fill miss_reason='on net' only for SOG or goal-type events missing miss_reason."""
    mask = df["event_type"].isin(["shot-on-goal", "goal"])
    df.loc[mask & df["miss_reason"].isna(), "miss_reason"] = "on-net"
    return df


def _clean_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MM:SS to seconds and handle period-type length differences (REG/OT)."""

    def time_to_seconds(t):
        try:
            m, s = map(int, t.split(":"))
            return m * 60 + s
        except:
            return np.nan

    # Rename column for clarity
    df = df.rename(columns={"time_in_period": "time_into_period"})
    # Convert column to seconds
    df["time_into_period_sec"] = df["time_into_period"].apply(time_to_seconds)
    return df


def _remove_shootout_shots(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["period"] != 5]


def _drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        "x_coord_raw",
        "y_coord_raw",
        "strength_state",
        "situation_code",
        "time_remaining",
        "time_into_period",
        # Remove placeholder cols for now
        "score_state",
        "score_differential",
        "is_rebound",
        "is_rush_shot",
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


def _convert_bool_columns_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns into binary."""
    # Boolean columns to convert
    bool_cols = [
        "is_home_team",
    ]

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            # Optional: warn or ignore missing columns
            print(f"Warning: Column '{col}' not found — skipping.")

    return df


def clean_shot_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _clean_miss_reason(df)
    df = _clean_time_columns(df)
    df = _remove_shootout_shots(df)
    df = _drop_redundant_columns(df)
    df = _convert_bool_columns_to_binary(df)
    return df


if __name__ == "__main__":
    # Path to your raw CSV
    raw_path = Path("data/raw/nhl_shots_2019_2024.csv")
    processed_path = Path("data/processed/nhl_shots_cleaned.csv")

    # Load raw data
    df_raw = pd.read_csv(raw_path)

    # Clean the data
    df_clean = clean_shot_data(df_raw)

    # Save to processed folder
    processed_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    df_clean.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")
