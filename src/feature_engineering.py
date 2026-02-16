import pandas as pd
from pathlib import Path


def _add_empty_net_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary empty_net_shot column and remove goalie pulled columns."""
    df["empty_net_shot"] = (
        (
            df["is_home_team"] & df["away_goalie_pulled"]
        )  # home shooter & away goalie pulled
        | (
            ~df["is_home_team"] & df["home_goalie_pulled"]
        )  # away shooter & home goalie pulled
    ).astype(int)

    # Drop the original goalie pulled columns
    df = df.drop(
        columns=[
            c for c in ["home_goalie_pulled", "away_goalie_pulled"] if c in df.columns
        ]
    )
    return df


# def _add_shot_distance(df: pd.DataFrame) -> pd.DataFrame:
#     """Computes shot distance."""
#     # Example: distance from center of net at x=89, y=0
#     if "x_coord" in df.columns and "y_coord" in df.columns:
#         df["shot_distance"] = ((df["x_coord"] - 89) ** 2 + df["y_coord"] ** 2) ** 0.5
#     return df

# def _add_shot_angle(df: pd.DataFrame) -> pd.DataFrame:
#     """Placeholder for computing shot angle."""
#     # TODO: implement actual shot angle calculation
#     df["shot_angle"] = np.nan
#     return df

# def _add_time_since_last_shot(df: pd.DataFrame) -> pd.DataFrame:
#     """Placeholder for computing time since last shot."""
#     # TODO: implement actual time delta
#     df["time_since_last_shot"] = np.nan
#     return df
#


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features to be used in modeling by calling helper functions.

    Args:
        df (pd.DataFrame): Cleaned df.

    Returns:
        pd.DataFrame: Df with features added.
    """
    df = df.copy()
    df = _add_empty_net_feature(df)
    # df = _add_shot_distance(df)
    # df = _add_shot_angle(df)
    # df = _add_time_since_last_shot(df)
    return df


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent  # src
    PROJECT_ROOT = SCRIPT_DIR.parent  # xG_model
    # Path to your raw CSV
    clean_path = PROJECT_ROOT / "data/processed/nhl_shots_cleaned.csv"
    clean_features_path = PROJECT_ROOT / "data/processed/nhl_shots_cleaned_features.csv"

    # Load clean data
    df_clean = pd.read_csv(clean_path)

    # Add features
    df_clean_features = add_features(df_clean)

    # Save to processed folder
    clean_features_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # ensure folder exists
    df_clean_features.to_csv(clean_features_path, index=False)
    print(f" Added features to clean data and saved to {clean_features_path}")

    # df = pd.read_csv("data/processed/nhl_shots_cleaned.csv")
    # df = add_features(df)
    # df.to_csv("data/processed/nhl_shots_cleaned_features.csv", index=False)
    # print("Feature engineering complete.")
