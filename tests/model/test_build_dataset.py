"""
Tests for:
  - model/build_dataset.py : _split_by_season
"""

import pandas as pd

from model.build_dataset import _split_by_season

## Helpers


def make_season_df(*game_ids) -> pd.DataFrame:
    """Build a minimal DataFrame with game_id column."""
    return pd.DataFrame({"game_id": list(game_ids)})


## _split_by_season tests


class TestSplitBySeason:

    def test_2025_game_ids_go_to_test(self):
        df = make_season_df(2025020001, 2025020500)
        train, test = _split_by_season(df)
        assert len(test) == 2
        assert len(train) == 0

    def test_pre_2025_game_ids_go_to_train(self):
        df = make_season_df(2019020001, 2022020500, 2024021312)
        train, test = _split_by_season(df)
        assert len(train) == 3
        assert len(test) == 0

    def test_mixed_seasons_split_correctly(self):
        df = make_season_df(2023020001, 2024020001, 2025020001)
        train, test = _split_by_season(df)
        assert len(train) == 2
        assert len(test) == 1

    def test_train_contains_no_2025_game_ids(self):
        df = make_season_df(2023020001, 2024020001, 2025020001)
        train, _ = _split_by_season(df)
        assert not train["game_id"].astype(str).str.startswith("2025").any()

    def test_test_contains_only_2025_game_ids(self):
        df = make_season_df(2023020001, 2024020001, 2025020001)
        _, test = _split_by_season(df)
        assert test["game_id"].astype(str).str.startswith("2025").all()
