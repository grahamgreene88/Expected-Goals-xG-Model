import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List
from pathlib import Path
import joblib
import os

# Always resolve paths relative to project root, not cwd
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # xG_model
DATA_DIR = PROJECT_ROOT / "data"
TRANSFORMER_DIR = PROJECT_ROOT / "models/scalers"


# Utility functions for saving and loading transformers
def _save_transformer(obj, name: str):
    """Save a fitted scaler/encoder."""
    import os

    os.makedirs(TRANSFORMER_DIR, exist_ok=True)
    joblib.dump(obj, f"{TRANSFORMER_DIR}/{name}.pkl")


def _load_transformer(name: str):
    """Load a previously saved scaler/encoder."""
    return joblib.load(f"{TRANSFORMER_DIR}/{name}.pkl")


# Skew transformation
def _transform_skewed_features(
    df: pd.DataFrame, skewed_features: list, for_training: bool = True
) -> pd.DataFrame:
    """
    Apply log1p or other transformations to reduce skewness in numeric features.
    Assumes the input df contains all skewed_cols and they are numeric.
    For reflection, the function uses max(x). This must be saved during training
    so inference applies the same shift.
    NOTE: this function only currently works for 'x_coord' which gets reflected.
    """
    df = df.copy()
    # Ensure directory exists
    os.makedirs(TRANSFORMER_DIR, exist_ok=True)

    for col in skewed_features:

        transformer_path = os.path.join(TRANSFORMER_DIR, f"skew_{col}.joblib")

        if for_training:
            # Compute reflection max based only on training data
            x = df[col].dropna()
            reflect_max = float(x.max())
            # Save transformation parameter
            joblib.dump({"reflect_max": reflect_max}, transformer_path)

        else:
            # Load reflection parameter from training
            saved_params = joblib.load(transformer_path)
            reflect_max = saved_params["reflect_max"]

        # Perform reflected log1p transformation
        df[col + "_log"] = np.log1p(reflect_max - df[col])

        # Drop original column
        df.drop(columns=[col], inplace=True)

    return df


# Numeric scaling
def _scale_numeric_features(
    df: pd.DataFrame, numeric_cols: List[str], for_training: bool
) -> pd.DataFrame:
    """Standardize numeric columns to have mean=0, std=1.
    In training mode (for_training=True): fit StandardScaler on provided numeric_cols and save it.
    In inference mode (for_training=False): load the scaler and apply transform.
    """
    df = df.copy()
    if not numeric_cols:
        return df

    scaler_name = "numeric_scalar"

    if for_training:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        _save_transformer(scaler, scaler_name)
    else:
        scaler = _load_transformer(scaler_name)
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


# Categorical encoding for linear models
def _encode_categorical_features(
    df: pd.DataFrame, cat_cols: List[str], for_training: bool
) -> pd.DataFrame:
    """
    One-hot encode categorical columns for linear models.
    - If for_training=True: fit OneHotEncoder and save it.
    - If for_training=False: load saved encoder and transform.
    Returns DataFrame with encoded columns replacing original categorical_cols.
    High-cardinality features are intentionally excluded.
    """
    df = df.copy()
    if not cat_cols:
        return df

    encoder_name = "linear_ohe"

    if for_training:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(df[cat_cols])
        _save_transformer(encoder, encoder_name)
    else:
        encoder = _load_transformer(encoder_name)
        encoded = encoder.transform(df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index
    )

    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df


def scale_features(
    df: pd.DataFrame,
    skewed_cols: List[str],
    numeric_cols: List[str],
    cat_cols: List[str],
    for_training: bool = True,
    linear_model: bool = True,
) -> pd.DataFrame:
    """
    Orchestrate feature transformations:
     - skew transforms (creates <col>_log and drops original <col>)
     - scale numeric_cols ONLY if linear_model=True (trees keep them as-is)
     - encode categorical_cols ONLY if linear_model=True (trees keep them as-is)
    Parameters:
      df: input DataFrame (cleaned)
      skewed_cols: list of column names to log-transform (original column names)
      numeric_cols: list of numeric column names to scale (after skew transform these should include '*_log' names)
      categorical_cols: list of categorical columns to one-hot encode for linear models
      for_training: if True, fit & save transformers; if False, load existing transformers and apply
      linear_model: if True, scale numeric cols and encode categoricals; otherwise leave them intact (for tree models)
    Returns:
      DataFrame transformed according to the flags.
    """
    df = df.copy()

    # 1) Skew transforms (creates new *_log columns and drops originals)
    if skewed_cols:
        df = _transform_skewed_features(df, skewed_cols, for_training=for_training)

    # 2) Scale numeric features (numeric_cols should refer to final numeric names)
    if linear_model and numeric_cols:
        df = _scale_numeric_features(df, numeric_cols, for_training=for_training)

    # 3) Categorical encoding for linear models
    if linear_model and cat_cols:
        df = _encode_categorical_features(df, cat_cols, for_training=for_training)

    # 4) Convert categorical columns to "category" data type for tree models
    if not linear_model and cat_cols:
        for col in cat_cols:
            df[col] = df[col].astype("category")

    return df


# # Load raw data
# df_in = pd.read_csv(
#     "/Users/graham/Projects/xG_model/data/processed/nhl_shots_cleaned_features.csv"
# )

# # Define which columns to transform/scale/encode
# skewed_cols = ["x_coord"]  # if already created in feature_engineering
# numeric_cols = ["x_coord_log", "y_coord", "time_into_period_sec"]
# categorical_cols = [
#     "period",
#     "period_type",
#     "shooting_team_strength_state",
#     "event_type",
#     "zone",
#     "shot_type",
#     "miss_reason",
# ]
# # cat_cols_low_card = ["period", "period_type", "shooting_team_strength_state", "event_type", "zone", "shot_type", "miss_reason"]
# # cat_cols_high_card = ["shooter_id", "goalie_id", "team_id", "shooter_team_abbrev", "opponent_team_abrrev"]

# # Clean the data
# df_out = scale_features(df_in, skewed_cols, numeric_cols, categorical_cols)
# df_out
#


if __name__ == "__main__":

    # Path to your raw CSV
    in_path = DATA_DIR / "processed/nhl_shots_cleaned_features.csv"
    out_path_train_linear = DATA_DIR / "processed/nhl_shots_scaled_for_linear.csv"
    out_path_train_tree = DATA_DIR / "processed/nhl_shots_scaled_for_tree.csv"

    # Load raw data
    df_in = pd.read_csv(in_path)

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

    # Training run for linear model (fits & saves scalers/encoders)
    df_scaled_linear = scale_features(
        df_in,
        skewed_cols=skewed_cols,
        numeric_cols=numeric_cols,
        cat_cols=categorical_cols,
        for_training=True,
        linear_model=True,
    )
    df_scaled_linear.to_csv(out_path_train_linear, index=False)
    print(f"Saved scaled & encoded data for linear model to {out_path_train_linear}")

    # === Training run for tree model (transforms skewed features but does NOT scale numeric cols or encode categoricals) ===
    df_scaled_tree = scale_features(
        df_in,
        skewed_cols=skewed_cols,
        numeric_cols=[],  # leave numeric columns intact
        cat_cols=[],  # leave categorical columns intact
        for_training=True,
        linear_model=False,
    )
    df_scaled_tree.to_csv(out_path_train_tree, index=False)
    print(f"Saved scaled data for tree model to {out_path_train_tree}")

    # Note: For inference, call scale_features(..., for_training=False, linear_model=...) to load and apply saved transformers.
