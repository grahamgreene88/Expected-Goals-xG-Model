"""
Queries the shots table, applies feature engineering, and saves
train/test parquet files to data/processed/.

Train: 2019-20 through 2024-25 seasons
Test:  2025-26 season

Usage:
    python -m model.build_dataset
"""

import logging
from pathlib import Path

import pandas as pd

from model.features import build_features
from pipeline.db import get_connection

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"

QUERY = "SELECT * FROM shots"


def _split_by_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split features DataFrame into train and test by season.

    The first four digits of game_id are the season start year.
    2025xxxxxxxx -> 2025-26 season -> test set.
    All prior seasons -> train set.
    """
    is_test = df["game_id"].astype(str).str.startswith("2025")
    return df[~is_test].copy(), df[is_test].copy()


def build_dataset() -> None:
    log.info("Querying shots table...")
    with get_connection() as conn:
        raw_df = pd.read_sql(QUERY, conn)
    log.info(f"  {len(raw_df):,} rows fetched")

    log.info("Engineering features...")
    features_df = build_features(raw_df)
    log.info(f"  {len(features_df):,} rows after filtering")

    log.info("Splitting into train / test...")
    train_df, test_df = _split_by_season(features_df)
    log.info(f"  Train: {len(train_df):,} rows")
    log.info(f"  Test:  {len(test_df):,} rows")

    log.info("Saving parquet files...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)
    log.info(f"  Saved {TRAIN_PATH}")
    log.info(f"  Saved {TEST_PATH}")

    # Sanity check goal rates - should be very similar
    train_goal_rate = train_df["is_goal"].mean()
    test_goal_rate = test_df["is_goal"].mean()
    log.info(f"  Train goal rate: {train_goal_rate:.3f}")
    log.info(f"  Test goal rate:  {test_goal_rate:.3f}")


if __name__ == "__main__":
    build_dataset()
