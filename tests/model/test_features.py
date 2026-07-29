"""
Tests for model/features.py

Covers:
  - _apply_filters: coord_normalized, shootout exclusion, empty-net exclusion
  - _compute_geometry: shot_distance, shot_angle, is_behind_net
  - _clean_shot_type: valid types retained, rare types and nulls set to 'other'
  - build_features: integration test (correct columns, shape, index reset)
"""

import math

import numpy as np
import pandas as pd
import pytest

from model.features import (
    FEATURE_COLS,
    TARGET_COL,
    VALID_SHOT_TYPES,
    _apply_filters,
    _clean_shot_type,
    _compute_geometry,
    build_features,
)

## Fixtures


def make_shot(**overrides) -> dict:
    """
    Return a dict representing one valid shot row with sensible defaults.
    Only includes columns that features.py actually reads.
    All overrides are applied on top of the defaults.
    """
    defaults = {
        "game_id": 2024020001,
        "period": 1,
        "period_type": "REG",
        "x_coord": 69.0,  # 20 feet in front of net, on centre line
        "y_coord": 0.0,
        "coord_normalized": True,
        "is_home_team": True,
        "away_goalie_pulled": False,
        "home_goalie_pulled": False,
        "shot_type": "wrist",
        "zone": "O",
        "shooting_team_strength_diff": 0,
        "is_goal": False,
    }
    return {**defaults, **overrides}


def make_df(*rows: dict) -> pd.DataFrame:
    """Build a DataFrame from one or more shot dicts."""
    return pd.DataFrame(list(rows))


## _compute_geometry tests


class TestComputeGeometry:
    """
    Net is at (89, 0). Shooting team always attacks positive x.

    Hand-calculated reference values:
      Straight on (69, 0)  -> distance=20, angle=0°
      Beside net  (89, 20) -> distance=20, angle=90°
      Behind net  (94, 0)  -> distance=5, angle=180°
      45-degree   (69, 20) -> distance≈28, angle=45°
    """

    def _geometry_for(self, x, y) -> pd.Series:
        df = make_df(make_shot(x_coord=x, y_coord=y))
        result = _compute_geometry(df)
        return result.iloc[0]

    def test_straight_on_distance(self):
        row = self._geometry_for(x=69, y=0)
        assert math.isclose(row["shot_distance"], 20.0, abs_tol=1e-6)

    def test_straight_on_angle_is_zero(self):
        row = self._geometry_for(x=69, y=0)
        assert math.isclose(row["shot_angle"], 0.0, abs_tol=1e-6)

    def test_beside_net_angle_is_90_degrees(self):
        # Directly beside the net at x=89 means angle is 90°
        row = self._geometry_for(x=89, y=20)
        assert math.isclose(row["shot_angle"], 90.0, abs_tol=1e-6)

    def test_behind_net_angle_is_180_degrees(self):
        row = self._geometry_for(x=94, y=0)
        assert math.isclose(row["shot_angle"], 180.0, abs_tol=1e-6)

    def test_45_degree_angle(self):
        row = self._geometry_for(x=69, y=20)
        assert math.isclose(row["shot_angle"], 45.0, abs_tol=1e-6)

    def test_45_degree_distance(self):
        row = self._geometry_for(x=69, y=20)
        assert math.isclose(row["shot_distance"], math.sqrt(800), abs_tol=1e-6)

    def test_angle_is_symmetric_for_positive_and_negative_y(self):
        # Shot from left side and right side at same distance should give same angle
        row_pos = self._geometry_for(x=69, y=20)
        row_neg = self._geometry_for(x=69, y=-20)
        assert math.isclose(row_pos["shot_angle"], row_neg["shot_angle"], abs_tol=1e-6)

    def test_is_behind_net_false_in_front(self):
        row = self._geometry_for(x=69, y=0)
        assert row["is_behind_net"] == False

    def test_is_behind_net_true_behind(self):
        row = self._geometry_for(x=94, y=0)
        assert row["is_behind_net"] == True

    def test_is_behind_net_false_at_goal_line(self):
        # x == NET_X is not behind the net
        row = self._geometry_for(x=89, y=0)
        assert row["is_behind_net"] == False

    def test_behind_net_angle_exceeds_90_degrees(self):
        row = self._geometry_for(x=94, y=10)
        assert row["shot_angle"] > 90.0


## _apply_filters tests


class TestApplyFilters:

    def test_excludes_shootout_shots(self):
        df = make_df(
            make_shot(period_type="REG"),
            make_shot(period_type="OT"),
            make_shot(period_type="SO"),
        )
        result = _apply_filters(df)
        assert len(result) == 2
        assert "SO" not in result["period_type"].values

    def test_excludes_unnormalized_coordinates(self):
        df = make_df(
            make_shot(coord_normalized=True),
            make_shot(coord_normalized=False),
        )
        result = _apply_filters(df)
        assert len(result) == 1
        assert result.iloc[0]["coord_normalized"] == True

    # Empty net exclusion

    def test_excludes_home_team_shooting_on_empty_away_net(self):
        # Home team shooting, away goalie pulled -> opponent net is empty -> exclude
        df = make_df(make_shot(is_home_team=True, away_goalie_pulled=True))
        with pytest.raises(
            ValueError, match="No rows remain after filtering — check input data."
        ):
            _apply_filters(df)

    def test_excludes_away_team_shooting_on_empty_home_net(self):
        # Away team shooting, home goalie pulled -> opponent net is empty -> exclude
        df = make_df(make_shot(is_home_team=False, home_goalie_pulled=True))
        with pytest.raises(
            ValueError, match="No rows remain after filtering — check input data."
        ):
            _apply_filters(df)

    def test_retains_home_team_shooting_with_own_net_empty(self):
        # Home team shooting, home goalie pulled -> own net is empty, not opponent's -> retain
        df = make_df(make_shot(is_home_team=True, home_goalie_pulled=True))
        result = _apply_filters(df)
        assert len(result) == 1

    def test_retains_away_team_shooting_with_own_net_empty(self):
        # Away team shooting, away goalie pulled -> own net is empty, not opponent's -> retain
        df = make_df(make_shot(is_home_team=False, away_goalie_pulled=True))
        result = _apply_filters(df)
        assert len(result) == 1

    def test_all_filters_applied_together(self):
        df = make_df(
            make_shot(),  # valid
            make_shot(coord_normalized=False),  # excluded: unnormalized
            make_shot(period_type="SO"),  # excluded: shootout
            make_shot(
                is_home_team=True, away_goalie_pulled=True
            ),  # excluded: empty net
        )
        result = _apply_filters(df)
        assert len(result) == 1

    def test_raises_when_no_rows_remain(self):
        # ValueError should surface from build_features, not _apply_filters directly
        df = make_df(make_shot(coord_normalized=False))
        with pytest.raises(ValueError, match="No rows remain after filtering"):
            build_features(df)


## _clean_shot_type tests


class TestCleanShotType:

    def _clean(self, shot_type) -> str:
        df = make_df(make_shot(shot_type=shot_type))
        result = _clean_shot_type(df)
        return result.iloc[0]["shot_type"]

    # Run each test for each rare shot type
    @pytest.mark.parametrize(
        "shot_type", ["wrap-around", "poke", "bat", "between-legs", "cradle"]
    )
    def test_rare_shot_types_become_other(self, shot_type):
        assert self._clean(shot_type) == "other"

    def test_null_shot_type_becomes_other(self):
        assert self._clean(None) == "other"

    def test_unknown_shot_type_becomes_other(self):
        assert self._clean("unknown-future-type") == "other"


## build_features (integration) tests


class TestBuildFeatures:

    def test_returns_expected_columns(self):
        df = make_df(make_shot())
        result = build_features(df)
        expected_cols = set(["game_id"] + FEATURE_COLS + [TARGET_COL])
        assert set(result.columns) == expected_cols

    def test_index_is_reset(self):
        # Even if input has non-contiguous index, output should be 0-based
        df = make_df(make_shot(), make_shot(period_type="SO"), make_shot())
        result = build_features(df)
        assert list(result.index) == list(range(len(result)))

    def test_filters_are_applied(self):
        df = make_df(
            make_shot(),
            make_shot(coord_normalized=False),
            make_shot(period_type="SO"),
        )
        result = build_features(df)
        assert len(result) == 1

    def test_period_is_string_in_output(self):
        df = make_df(make_shot(period=2))
        result = build_features(df)
        assert pd.api.types.is_string_dtype(result["period"])
        assert result.iloc[0]["period"] == "2"

    def test_is_home_team_is_int_in_output(self):
        df = make_df(make_shot(is_home_team=True))
        result = build_features(df)
        assert result["is_home_team"].dtype in (np.dtype("int32"), np.dtype("int64"))

    def test_is_behind_net_is_int_in_output(self):
        df = make_df(make_shot(x_coord=94.0))
        result = build_features(df)
        assert result["is_behind_net"].dtype in (np.dtype("int32"), np.dtype("int64"))

    def test_game_id_is_retained(self):
        df = make_df(make_shot(game_id=2024020999))
        result = build_features(df)
        assert result.iloc[0]["game_id"] == 2024020999
