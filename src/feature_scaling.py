import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List
from pathlib import Path


def _transform_skewed_features(
    df: pd.DataFrame, skewed_cols: List[str]
) -> pd.DataFrame:
    """
    Apply log1p or other transformations to reduce skewness in numeric features.
    Assumes the input df contains all skewed_cols and they are numeric.
    NOTE: this function only currently works for 'x_coord' which gets reflected.
    """
    df = df.copy()
    for col in skewed_cols:
        x = df[col].dropna()
        x_reflected = max(x) - x  # shift so min = 0
        df[col + "_log"] = np.log1p(x_reflected)
        # Drop original column
        df = df.drop(columns=[col])
    return df


def _scale_numeric_features(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Standardize numeric columns to have mean=0, std=1."""
    df = df.copy()
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def _encode_categorical_features(
    df: pd.DataFrame, categorical_cols: List[str]
) -> pd.DataFrame:
    """
    One-hot encode categorical features.
    """
    df = df.copy()
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded_array = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    return df


def scale_features(
    df: pd.DataFrame,
    skewed_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    """
    Master function for feature scaling and encoding.
    """
    df = df.copy()
    df = _transform_skewed_features(df, skewed_cols)
    df = _scale_numeric_features(df, numeric_cols)
    df = _encode_categorical_features(df, categorical_cols)
    return df


# Load raw data
df_in = pd.read_csv(
    "/Users/graham/Projects/xG_model/data/processed/nhl_shots_cleaned_features.csv"
)

# Define which columns to transform/scale/encode
skewed_cols = ["x_coord"]  # if already created in feature_engineering
numeric_cols = ["x_coord_log", "y_coord", "time_into_period_sec"]
categorical_cols = [
    "period",
    "period_type",
    "shooting_team_strength_state",
    "event_type",
    "zone",
    "shot_type",
    "miss_reason",
]
# cat_cols_low_card = ["period", "period_type", "shooting_team_strength_state", "event_type", "zone", "shot_type", "miss_reason"]
# cat_cols_high_card = ["shooter_id", "goalie_id", "team_id", "shooter_team_abbrev", "opponent_team_abrrev"]

# Clean the data
df_out = scale_features(df_in, skewed_cols, numeric_cols, categorical_cols)
df_out

if __name__ == "__main__":
    # Path to your raw CSV
    in_path = Path("data/processed/nhl_shots_cleaned_features.csv")
    out_path = Path("data/processed/nhl_shots_scaled.csv")

    # Load raw data
    df_in = pd.read_csv(in_path)

    # Define which columns to transform/scale/encode
    skewed_cols = ["x_coord"]  # if already created in feature_engineering
    numeric_cols = ["x_coord_log", "y_coord", "time_into_period_sec"]
    categorical_cols = ["empty_net_shot", "is_home_team"]

    # Clean the data
    df_out = scale_features(df_in)

    # Save to processed folder
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    df_out.to_csv(out_path, index=False)
    print(f"Scaled data saved to {out_path}")
