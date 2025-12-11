"""
train_models.py

Handles:
- Loading processed feature data
- Train/test split
- Producing model-ready feature sets using feature_scaling.py
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from feature_scaling import scale_features

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # xG_model
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_FILE = DATA_DIR / "nhl_shots_cleaned_features.csv"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models/trained_models"

# These will be passed into scale_features()
SKEWED_COLS = ["x_coord"]  # example
NUMERIC_COLS = ["x_coord_log", "y_coord", "time_into_period_sec"]
CATEGORICAL_COLS = [
    "period",
    "period_type",
    "shooting_team_strength_state",
    "event_type",
    "zone",
    "shot_type",
    "miss_reason",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "is_goal"

# LINEAR_MODELS = [
#     "logistic_regression",
#     "ridge_classifier",
# ]

# TREE_MODELS = [
#     "random_forest",
#     "xgboost",
# ]

# CV_FOLDS = 5
# SCORING_METRIC = "roc_auc"

# --------------------------------------------------------------------------------------
# Load and Split Data
# --------------------------------------------------------------------------------------


def load_feature_data() -> pd.DataFrame:
    """Load the fully cleaned feature dataset."""
    print(f"Loading feature file: {FEATURE_FILE}")
    df = pd.read_csv(FEATURE_FILE)
    return df


def split_train_test(df: pd.DataFrame):
    """
    Split df into training and test sets, ensuring the target column is preserved.
    """
    print("Performing train/test split...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------------------
# Produce Linear + Tree Model Feature Sets
# --------------------------------------------------------------------------------------


def preprocess_features(X_train, X_test):
    """
    Apply your feature scaling module in both modes:
    - for_training=True to fit scalers/encoders on the training data
    - for_training=False to apply saved scalers/encoders on test data
    """

    print("\nPreparing LINEAR-model features...")
    X_train_linear = scale_features(
        df=X_train,
        skewed_cols=SKEWED_COLS,
        numeric_cols=NUMERIC_COLS,
        cat_cols=CATEGORICAL_COLS,
        for_training=True,
        linear_model=True,
    )

    X_test_linear = scale_features(
        df=X_test,
        skewed_cols=SKEWED_COLS,
        numeric_cols=NUMERIC_COLS,
        cat_cols=CATEGORICAL_COLS,
        for_training=True,
        linear_model=True,
    )

    print("\nPreparing TREE-model features...")
    X_train_tree = scale_features(
        df=X_train,
        skewed_cols=SKEWED_COLS,
        numeric_cols=NUMERIC_COLS,
        cat_cols=CATEGORICAL_COLS,
        for_training=True,
        linear_model=False,
    )

    X_test_tree = scale_features(
        df=X_test,
        skewed_cols=SKEWED_COLS,
        numeric_cols=NUMERIC_COLS,
        cat_cols=CATEGORICAL_COLS,
        for_training=True,
        linear_model=False,
    )

    return X_train_linear, X_test_linear, X_train_tree, X_test_tree


# --------------------------------------------------------------------------------------
# Main Entry
# --------------------------------------------------------------------------------------


def main():
    # Load data
    df = load_feature_data()

    # Split into train & test
    X_train, X_test, y_train, y_test = split_train_test(df)

    # Produce modeling-ready feature matrices
    (
        X_train_linear,
        X_test_linear,
        X_train_tree,
        X_test_tree,
    ) = preprocess_features(X_train, X_test)

    print("\nFeature preprocessing completed.")
    print("Linear model training set shape:", X_train_linear.shape)
    print("Tree model training set shape:", X_train_tree.shape)

    # TODO: Next steps (in later phases):
    # - Train logistic regression / linear model
    # - Train RandomForest / XGBoost / LightGBM
    # - Cross-validation pipelines
    # - Model comparison (ROC AUC, log loss, calibration)
    # - Save trained models

    return (
        X_train_linear,
        X_test_linear,
        X_train_tree,
        X_test_tree,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()
