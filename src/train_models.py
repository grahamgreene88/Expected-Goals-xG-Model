"""
train_models.py

Handles:
- Loading processed feature data
- Train/test split
- Producing model-ready feature sets using feature_scaling.py
"""

import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
import xgboost as xgb

from feature_scaling import scale_features

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # xG_model
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_FILE = DATA_DIR / "processed" / "nhl_shots_cleaned_features.csv"
MODEL_DIR = PROJECT_ROOT / "models/trained_models"

# These will be passed into scale_features()
SKEWED_COLS = ["x_coord"]  # example
NUMERIC_COLS = ["x_coord_log", "y_coord", "time_into_period_sec"]
CATEGORICAL_COLS = [
    "period",
    "period_type",
    "shooting_team_strength_state",
    "zone",
    "shot_type",
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


def preprocess_features(X_train, X_test, linear_model=True):
    """
    Apply your feature scaling module in both modes:
    - for_training=True to fit scalers/encoders on the training data
    - for_training=False to apply saved scalers/encoders on test data
    """
    if linear_model == True:
        print("\nPreparing LINEAR-model features...")
        X_train = scale_features(
            df=X_train,
            skewed_cols=SKEWED_COLS,
            numeric_cols=NUMERIC_COLS,
            cat_cols=CATEGORICAL_COLS,
            for_training=True,
            linear_model=True,
        )

        X_test = scale_features(
            df=X_test,
            skewed_cols=SKEWED_COLS,
            numeric_cols=NUMERIC_COLS,
            cat_cols=CATEGORICAL_COLS,
            for_training=False,
            linear_model=True,
        )
    else:
        print("\nPreparing TREE-model features...")
        X_train = scale_features(
            df=X_train,
            skewed_cols=SKEWED_COLS,
            numeric_cols=NUMERIC_COLS,
            cat_cols=CATEGORICAL_COLS,
            for_training=True,
            linear_model=False,
        )

        X_test = scale_features(
            df=X_test,
            skewed_cols=SKEWED_COLS,
            numeric_cols=NUMERIC_COLS,
            cat_cols=CATEGORICAL_COLS,
            for_training=False,
            linear_model=False,
        )

    return X_train, X_test


# --------------------------------------------------------------------------------------
# Train linear and tree functions
# --------------------------------------------------------------------------------------


def train_logistic_regression(X_train_linear, y_train):
    """Train Logistic Regression model."""
    print("\n" + "=" * 60)
    print("Training Logistic Regression")
    print("=" * 60)

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",  # Handle class imbalance
        solver="lbfgs",
    )

    model.fit(X_train_linear, y_train)
    print("Training complete!")

    return model


def train_xgboost(X_train_tree, y_train):
    """Train XGBoost model."""
    print("\n" + "=" * 60)
    print("Training XGBoost")
    print("=" * 60)

    # Calculate scale_pos_weight for imbalanced dataset
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        eval_metric="logloss",
        use_label_encoder=False,
        enable_categorical=True,
    )

    model.fit(X_train_tree, y_train)
    print("Training complete!")

    return model


# --------------------------------------------------------------------------------------
# Evaluate and save models
# --------------------------------------------------------------------------------------


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and return metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Goal", "Goal"]))

    # Calculate calibration (compare predicted vs actual goal rate in bins)
    print(f"\nCalibration Analysis:")
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 1.0]
    bin_labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-100%"]

    df_cal = pd.DataFrame({"predicted": y_pred_proba, "actual": y_test})
    df_cal["bin"] = pd.cut(df_cal["predicted"], bins=bins, labels=bin_labels)

    calibration = (
        df_cal.groupby("bin", observed=True)
        .agg({"predicted": ["mean", "count"], "actual": "mean"})
        .round(4)
    )
    calibration.columns = ["Avg Predicted", "Count", "Actual Rate"]
    print(calibration)

    metrics = {
        "roc_auc": float(roc_auc),
        "log_loss": float(logloss),
        "accuracy": float(accuracy),
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
    }

    return metrics, y_pred_proba


def save_model(model, model_name: str, metrics: dict, feature_names: list):
    """Save trained model and metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_filename = f"{model_name}_{timestamp}.pkl"
    model_path = MODEL_DIR / model_filename
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "metrics": metrics,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "model_file": model_filename,
    }

    metadata_filename = f"{model_name}_{timestamp}_metadata.json"
    metadata_path = MODEL_DIR / metadata_filename
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    return model_path, metadata_path


# --------------------------------------------------------------------------------------
# Main Entry
# --------------------------------------------------------------------------------------


def main():
    # Load data
    df = load_feature_data()

    # Split into train & test
    X_train, X_test, y_train, y_test = split_train_test(df)

    # -------------------------
    # Train Logistic Regression
    # -------------------------
    print("\n\n" + "#" * 60)
    print("# LOGISTIC REGRESSION PIPELINE")
    print("#" * 60)

    # Produce modeling-ready feature matrices
    X_train_linear, X_test_linear = preprocess_features(
        X_train, X_test, linear_model=True
    )

    print("\nFeature preprocessing completed.")
    print("Linear model training set shape:", X_train_linear.shape)
    # Train and evaluate
    lr_model = train_logistic_regression(X_train_linear, y_train)
    lr_metrics, lr_predictions = evaluate_model(
        lr_model, X_test_linear, y_test, "Logistic_Regression"
    )
    # Save model
    lr_model_path, lr_metadata_path = save_model(
        lr_model,
        "logistic_regression_model",
        lr_metrics,
        X_train.columns.tolist(),
    )

    # -------------------------
    # Train XGBoost
    # -------------------------
    print("\n\n" + "#" * 60)
    print("# XGBOOST PIPELINE")
    print("#" * 60)

    # Produce modeling-ready feature matrices
    X_train_tree, X_test_tree = preprocess_features(X_train, X_test, linear_model=False)

    print("\nFeature preprocessing completed.")
    print("Tree model training set shape:", X_train_tree.shape)

    # Train and evaluate
    xgb_model = train_xgboost(X_train_tree, y_train)
    xgb_metrics, xgb_predictions = evaluate_model(
        xgb_model, X_test_tree, y_test, "XGBoost"
    )
    # Save model
    xgb_model_path, xgb_metadata_path = save_model(
        xgb_model, "xgboost_model", xgb_metrics, X_train.columns.tolist()
    )


if __name__ == "__main__":
    main()
