import pytest

from pipeline.parse import normalize_coordinates, parse_shots


# Fixture
@pytest.fixture
def sample_api_response():
    """
    Minimal but realistic mock of a raw NHL play-by-play API response.
    Based on game 2024020954 (SJS @ OTT, 2025-03-01).
    Contains one shot-on-goal, one missed-shot, and one goal.
    Non-shot events are included to confirm they are correctly filtered out.
    """
    return {
        "id": 2024020954,
        "season": 20242025,
        "gameType": 2,
        "gameDate": "2025-03-01",
        "gameState": "OFF",
        "gameScheduleState": "OK",
        "periodDescriptor": {
            "number": 3,
            "periodType": "REG",
            "maxRegulationPeriods": 3,
        },
        "awayTeam": {
            "id": 28,
            "commonName": {"default": "Sharks"},
            "abbrev": "SJS",
            "score": 3,
            "placeName": {"default": "San Jose"},
        },
        "homeTeam": {
            "id": 9,
            "commonName": {"default": "Senators"},
            "abbrev": "OTT",
            "score": 5,
            "placeName": {"default": "Ottawa"},
        },
        "plays": [
            # Non-shot event — should be filtered out
            {
                "eventId": 51,
                "periodDescriptor": {
                    "number": 1,
                    "periodType": "REG",
                    "maxRegulationPeriods": 3,
                },
                "timeInPeriod": "00:00",
                "timeRemaining": "20:00",
                "situationCode": "1551",
                "homeTeamDefendingSide": "left",
                "typeDescKey": "faceoff",
                "sortOrder": 11,
                "details": {
                    "eventOwnerTeamId": 28,
                    "xCoord": 0,
                    "yCoord": 0,
                    "zoneCode": "N",
                },
            },
            # Shot-on-goal by away team (SJS)
            # homeTeamDefendingSide = left → away team attacks right → no flip
            {
                "eventId": 103,
                "periodDescriptor": {
                    "number": 1,
                    "periodType": "REG",
                    "maxRegulationPeriods": 3,
                },
                "timeInPeriod": "00:09",
                "timeRemaining": "19:51",
                "situationCode": "1551",
                "homeTeamDefendingSide": "left",
                "typeDescKey": "shot-on-goal",
                "sortOrder": 12,
                "details": {
                    "xCoord": -55,
                    "yCoord": 1,
                    "zoneCode": "O",
                    "shotType": "wrist",
                    "shootingPlayerId": 8477505,
                    "goalieInNetId": 8476999,
                    "eventOwnerTeamId": 28,
                },
            },
            # Missed-shot by home team (OTT)
            # homeTeamDefendingSide = left → home team attacks left → flip coordinates
            {
                "eventId": 115,
                "periodDescriptor": {
                    "number": 1,
                    "periodType": "REG",
                    "maxRegulationPeriods": 3,
                },
                "timeInPeriod": "01:09",
                "timeRemaining": "18:51",
                "situationCode": "1551",
                "homeTeamDefendingSide": "left",
                "typeDescKey": "missed-shot",
                "sortOrder": 26,
                "details": {
                    "xCoord": 41,
                    "yCoord": 37,
                    "zoneCode": "O",
                    "reason": "high-and-wide-left",
                    "shotType": "wrist",
                    "shootingPlayerId": 8480801,
                    "goalieInNetId": 8477970,
                    "eventOwnerTeamId": 9,
                },
            },
            # Goal by home team (OTT) on power play
            # situationCode 1541 → home team has 4 skaters, away has 5 skaters
            # homeTeamDefendingSide = left → home team attacks left → flip coordinates
            {
                "eventId": 264,
                "periodDescriptor": {
                    "number": 1,
                    "periodType": "REG",
                    "maxRegulationPeriods": 3,
                },
                "timeInPeriod": "11:05",
                "timeRemaining": "08:55",
                "situationCode": "1541",
                "homeTeamDefendingSide": "left",
                "typeDescKey": "goal",
                "sortOrder": 171,
                "details": {
                    "xCoord": 73,
                    "yCoord": 4,
                    "zoneCode": "O",
                    "shotType": "wrist",
                    "scoringPlayerId": 8481596,
                    "goalieInNetId": 8477970,
                    "eventOwnerTeamId": 9,
                },
            },
        ],
    }


# normalize_coordinates tests
class TestNormalizeCoordinates:

    def test_away_team_attacks_left_flip(self):
        """
        homeTeamDefendingSide = left → home team attacks right → away team attacks left.
        Away team shot should be flipped on x-axis.
        """
        x, y, normalized = normalize_coordinates(
            -55, 1, is_home_team=False, home_team_defending_side="left"
        )
        assert x == 55
        assert y == 1
        assert normalized is True

    def test_home_team_attacks_right_no_flip(self):
        """
        homeTeamDefendingSide = left → home team attacks right.
        Home team shot should not be flipped.
        """
        x, y, normalized = normalize_coordinates(
            41, 37, is_home_team=True, home_team_defending_side="left"
        )
        assert x == 41
        assert y == 37
        assert normalized is True

    def test_home_team_attacks_left_flip(self):
        """
        homeTeamDefendingSide = right → home team attacks left.
        Home team shot should be flipped on x-axis.
        """
        x, y, normalized = normalize_coordinates(
            55, 10, is_home_team=True, home_team_defending_side="right"
        )
        assert x == -55
        assert y == 10
        assert normalized is True

    def test_away_team_attacks_right_no_flip(self):
        """
        homeTeamDefendingSide = right → away team attacks right.
        Away team shot should not be flipped.
        """
        x, y, normalized = normalize_coordinates(
            -55, 10, is_home_team=False, home_team_defending_side="right"
        )
        assert x == -55
        assert y == 10
        assert normalized is True

    def test_none_x_coordinate_returns_false(self):
        """
        None x coordinate should return raw values with coord_normalized = False.
        """
        x, y, normalized = normalize_coordinates(
            None, 10, is_home_team=True, home_team_defending_side="left"
        )
        assert x is None
        assert y == 10
        assert normalized is False

    def test_none_y_coordinate_returns_false(self):
        """
        None y coordinate should return raw values with coord_normalized = False.
        """
        x, y, normalized = normalize_coordinates(
            55, None, is_home_team=True, home_team_defending_side="left"
        )
        assert x == 55
        assert y is None
        assert normalized is False

    def test_none_defending_side_returns_false(self):
        """
        None homeTeamDefendingSide should return raw coordinates with coord_normalized = False.
        """
        x, y, normalized = normalize_coordinates(
            55, 10, is_home_team=True, home_team_defending_side=None
        )
        assert x == 55
        assert y == 10
        assert normalized is False

    def test_unexpected_defending_side_returns_false(self):
        """
        Unexpected homeTeamDefendingSide value should return raw coordinates
        with coord_normalized = False.
        """
        x, y, normalized = normalize_coordinates(
            55, 10, is_home_team=True, home_team_defending_side="center"
        )
        assert x == 55
        assert y == 10
        assert normalized is False

    def test_y_axis_never_flipped(self):
        """
        Y coordinate should never be flipped regardless of attacking direction.
        """
        _, y_flipped, _ = normalize_coordinates(
            41, 37, is_home_team=False, home_team_defending_side="left"
        )
        _, y_no_flip, _ = normalize_coordinates(
            41, 37, is_home_team=True, home_team_defending_side="left"
        )
        assert y_flipped == 37
        assert y_no_flip == 37


# parse_shots tests
class TestParseShots:

    def test_returns_correct_shot_count(self, sample_api_response):
        """
        Should return exactly 3 shots — one shot-on-goal, one missed-shot, one goal.
        The faceoff event should be filtered out.
        """
        shots = parse_shots(sample_api_response, game_id=2024020954)
        assert len(shots) == 3

    def test_non_shot_events_filtered_out(self, sample_api_response):
        """
        Non-shot events like faceoffs should not appear in the results.
        """
        shots = parse_shots(sample_api_response, game_id=2024020954)
        event_types = {s["event_type"] for s in shots}
        assert "faceoff" not in event_types
        assert event_types == {"shot-on-goal", "missed-shot", "goal"}

    def test_shot_on_goal_fields(self, sample_api_response):
        """
        Shot-on-goal by away team (SJS) should have correct team,
        player, and coordinate fields.
        """
        shots = parse_shots(sample_api_response, game_id=2024020954)
        sog = next(s for s in shots if s["event_id"] == 103)

        assert sog["game_id"] == 2024020954
        assert sog["event_type"] == "shot-on-goal"
        assert sog["shooter_id"] == 8477505
        assert sog["goalie_id"] == 8476999
        assert sog["shooter_team_id"] == 28
        assert sog["shooter_team_abbrev"] == "SJS"
        assert sog["opponent_team_abbrev"] == "OTT"
        assert sog["is_home_team"] is False
        assert sog["is_goal"] is False
        assert sog["miss_reason"] is None

    def test_away_team_shot_coordinates_flipped(self, sample_api_response):
        """
        Away team (SJS) shot with homeTeamDefendingSide = left should be flipped —
        away team attacks left so x should be negated.
        """
        shots = parse_shots(sample_api_response, game_id=2024020954)
        sog = next(s for s in shots if s["event_id"] == 103)

        assert sog["x_coord_raw"] == -55
        assert sog["y_coord_raw"] == 1
        assert sog["x_coord"] == 55
        assert sog["y_coord"] == 1
        assert sog["coord_normalized"] is True

    def test_home_team_shot_coordinates_not_flipped(self, sample_api_response):
        """
        Home team (OTT) missed-shot with homeTeamDefendingSide = left should not
        be flipped — home team attacks right so coordinates stay as-is.
        """
        shots = parse_shots(sample_api_response, game_id=2024020954)
        missed = next(s for s in shots if s["event_id"] == 115)

        assert missed["x_coord_raw"] == 41
        assert missed["y_coord_raw"] == 37
        assert missed["x_coord"] == 41
        assert missed["y_coord"] == 37
        assert missed["coord_normalized"] is True

    def test_missed_shot_has_miss_reason(self, sample_api_response):
        """
        Missed-shot events should have miss_reason populated.
        """
        shots = parse_shots(sample_api_response, game_id=2024020954)
        missed = next(s for s in shots if s["event_id"] == 115)
        assert missed["miss_reason"] == "high-and-wide-left"

    def test_goal_fields(self, sample_api_response):
        """
        Goal by home team (OTT) should have is_goal = True,
        scorer ID from scoringPlayerId, and miss_reason = None.
        """
        shots = parse_shots(sample_api_response, game_id=2024020954)
        goal = next(s for s in shots if s["event_id"] == 264)

        assert goal["event_type"] == "goal"
        assert goal["is_goal"] is True
        assert goal["shooter_id"] == 8481596
        assert goal["miss_reason"] is None
        assert goal["is_home_team"] is True
        assert goal["shooter_team_abbrev"] == "OTT"
        assert goal["opponent_team_abbrev"] == "SJS"

    def test_5v5_situation_code_parsing(self, sample_api_response):
        """
        situationCode '1551' should parse to:
        - strength_state = '5v5'
        - away_goalie_pulled = False
        - home_goalie_pulled = False
        - shooting_team_strength_state and diff from shooter's perspective
        """
        shots = parse_shots(sample_api_response, game_id=2024020954)
        sog = next(s for s in shots if s["event_id"] == 103)

        assert sog["situation_code"] == "1551"
        assert sog["strength_state"] == "5v5"
        assert sog["away_goalie_pulled"] is False
        assert sog["home_goalie_pulled"] is False
        # Away team shooting 5v5
        assert sog["shooting_team_strength_state"] == "5v5"
        assert sog["shooting_team_strength_diff"] == 0

    def test_power_play_situation_code_parsing(self, sample_api_response):
        """
        situationCode '1541' — away team on power play (5 away skaters, 4 home skaters).
        Home team (OTT) shooting shorthanded should show 4v5 from their perspective.
        """
        shots = parse_shots(sample_api_response, game_id=2024020954)
        goal = next(s for s in shots if s["event_id"] == 264)

        assert goal["situation_code"] == "1541"
        assert goal["strength_state"] == "5v4"
        assert goal["away_goalie_pulled"] is False
        assert goal["home_goalie_pulled"] is False
        # Home team shooting shorthanded
        assert goal["shooting_team_strength_state"] == "4v5"
        assert goal["shooting_team_strength_diff"] == -1

    def test_empty_plays_returns_empty_list(self):
        """
        A response with no plays should return an empty list.
        """
        data = {
            "homeTeam": {"id": 9, "abbrev": "OTT"},
            "awayTeam": {"id": 28, "abbrev": "SJS"},
            "plays": [],
        }
        shots = parse_shots(data, game_id=2024020954)
        assert shots == []

    def test_all_shots_have_required_fields(self, sample_api_response):
        """
        Every shot record should contain all required fields
        with no missing keys.
        """
        required_fields = [
            "game_id",
            "event_id",
            "sort_order",
            "period",
            "period_type",
            "time_in_period",
            "time_remaining",
            "situation_code",
            "strength_state",
            "away_goalie_pulled",
            "home_goalie_pulled",
            "shooting_team_strength_state",
            "shooting_team_strength_diff",
            "shooter_id",
            "goalie_id",
            "shooter_team_id",
            "shooter_team_abbrev",
            "opponent_team_abbrev",
            "is_home_team",
            "event_type",
            "x_coord_raw",
            "y_coord_raw",
            "x_coord",
            "y_coord",
            "coord_normalized",
            "zone",
            "shot_type",
            "is_goal",
            "miss_reason",
            "created_at_utc",
            "created_at_et",
        ]
        shots = parse_shots(sample_api_response, game_id=2024020954)
        for shot in shots:
            for field in required_fields:
                assert (
                    field in shot
                ), f"Missing field '{field}' in shot {shot.get('event_id')}"
