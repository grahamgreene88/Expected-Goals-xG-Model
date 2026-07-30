import pytest

from pipeline.schedule import _parse_game_data


# Fixtures
@pytest.fixture
def sample_game_record():
    """
    A single game record from the club schedule endpoint.
    Based on game 2023020655 (SJS @ OTT, 2024-01-13).
    """
    return {
        "id": 2023020655,
        "season": 20232024,
        "gameType": 2,
        "gameDate": "2024-01-13",
        "gameState": "OFF",
        "gameScheduleState": "OK",
        "awayTeam": {
            "id": 28,
            "commonName": {"default": "Sharks"},
            "placeName": {"default": "San Jose"},
            "abbrev": "SJS",
            "score": 4,
        },
        "homeTeam": {
            "id": 9,
            "commonName": {"default": "Senators"},
            "placeName": {"default": "Ottawa"},
            "abbrev": "OTT",
            "score": 5,
        },
        "gameOutcome": {"lastPeriodType": "REG"},
    }


@pytest.fixture
def sample_game_record_no_score():
    """
    A game record missing score and outcome fields.
    Represents a postponed or future game where these fields
    are not yet available.
    """
    return {
        "id": 2023020655,
        "season": 20232024,
        "gameType": 2,
        "gameDate": "2024-01-13",
        "gameState": "FUT",
        "gameScheduleState": "OK",
        "awayTeam": {
            "id": 28,
            "commonName": {"default": "Sharks"},
            "placeName": {"default": "San Jose"},
            "abbrev": "SJS",
        },
        "homeTeam": {
            "id": 9,
            "commonName": {"default": "Senators"},
            "placeName": {"default": "Ottawa"},
            "abbrev": "OTT",
        },
    }


# _parse_game_data tests
class TestParseGameData:

    def test_all_fields_parse_correctly(self, sample_game_record):
        """
        A complete game record should parse all fields correctly.
        """
        result = _parse_game_data(sample_game_record)

        assert result["game_id"] == 2023020655
        assert result["season"] == 20232024
        assert result["game_type"] == 2
        assert result["game_date"] == "2024-01-13"
        assert result["game_state"] == "OFF"
        assert result["game_schedule_state"] == "OK"
        assert result["last_period_type"] == "REG"

    def test_away_team_fields(self, sample_game_record):
        """
        Away team fields should parse correctly from the awayTeam object.
        """
        result = _parse_game_data(sample_game_record)

        assert result["away_team_id"] == 28
        assert result["away_team_city"] == "San Jose"
        assert result["away_team_name"] == "Sharks"
        assert result["away_team_abbrev"] == "SJS"
        assert result["away_team_score"] == 4

    def test_home_team_fields(self, sample_game_record):
        """
        Home team fields should parse correctly from the homeTeam object.
        """
        result = _parse_game_data(sample_game_record)

        assert result["home_team_id"] == 9
        assert result["home_team_city"] == "Ottawa"
        assert result["home_team_name"] == "Senators"
        assert result["home_team_abbrev"] == "OTT"
        assert result["home_team_score"] == 5

    def test_game_date_fallback_to_game_record(self, sample_game_record):
        """
        When no game_date argument is passed, should fall back to
        gameDate field on the game record. Used by get_season_games
        which calls _parse_game_data without a date argument.
        """
        result = _parse_game_data(sample_game_record)
        assert result["game_date"] == "2024-01-13"

    def test_explicit_game_date_takes_precedence(self, sample_game_record):
        """
        When game_date is passed explicitly it should override the
        gameDate field on the game record. Used by get_games_for_date
        which passes the date from the outer gameWeek loop since the
        schedule endpoint does not include gameDate on the game object.
        """
        result = _parse_game_data(sample_game_record, game_date="2024-01-14")
        assert result["game_date"] == "2024-01-14"

    def test_missing_score_returns_none(self, sample_game_record_no_score):
        """
        Games without score fields (future or postponed games) should
        return None for scores rather than raising a KeyError.
        """
        result = _parse_game_data(sample_game_record_no_score)

        assert result["away_team_score"] is None
        assert result["home_team_score"] is None

    def test_missing_game_outcome_returns_none(self, sample_game_record_no_score):
        """
        Games without a gameOutcome field should return None for
        last_period_type rather than raising a KeyError.
        """
        result = _parse_game_data(sample_game_record_no_score)
        assert result["last_period_type"] is None

    def test_result_contains_all_required_fields(self, sample_game_record):
        """
        Parsed game record should contain all fields required by the
        games table schema with no missing keys.
        """
        required_fields = [
            "game_id",
            "season",
            "game_type",
            "game_date",
            "game_state",
            "game_schedule_state",
            "away_team_id",
            "away_team_city",
            "away_team_name",
            "away_team_abbrev",
            "away_team_score",
            "home_team_id",
            "home_team_city",
            "home_team_name",
            "home_team_abbrev",
            "home_team_score",
            "last_period_type",
            "created_at_utc",
            "created_at_et",
        ]
        result = _parse_game_data(sample_game_record)
        for field in required_fields:
            assert field in result, f"Missing field '{field}' in parsed game record"
