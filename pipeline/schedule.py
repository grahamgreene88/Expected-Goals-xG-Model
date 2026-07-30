import requests

from pipeline.config import API_TIMEOUT_SECONDS, NHL_API_BASE_URL

REGULAR_SEASON_GAME_TYPE = 2


# Helpers
def _get_team_abbreviations() -> list[str]:
    """
    Fetch all NHL team abbreviations from the standings endpoint.
    Used internally to drive season schedule fetches.
    """
    url = f"{NHL_API_BASE_URL}/v1/standings/now"
    response = requests.get(url, timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()

    abbrevs = []
    for team_record in data.get("standings", []):
        abbrev = team_record.get("teamAbbrev", {}).get("default")
        if abbrev:
            abbrevs.append(abbrev)

    return sorted(set(abbrevs))


def _parse_game_data(game: dict, game_date: str | None = None) -> dict:
    """
    Parse a single game's data from the schedule API response
    into a dictionary ready for DB upsert.

    Parameters:
    - game: raw game dict from schedule API response
    - game_date: optional date string in format 'YYYY-MM-DD'. If not provided,
      falls back to the 'gameDate' field on the game object (club schedule endpoint).
      Pass explicitly when using the schedule-by-date endpoint which does not
      include gameDate on the game object.

    Returns:
    - Dictionary of game fields matching the games table schema
    """
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))

    date = game_date or game.get("gameDate")

    return {
        "game_id": game["id"],
        "season": game["season"],
        "game_type": game["gameType"],
        "game_date": date,
        "game_state": game["gameState"],
        "game_schedule_state": game["gameScheduleState"],
        # Away team
        "away_team_id": game["awayTeam"]["id"],
        "away_team_city": game["awayTeam"]["placeName"]["default"],
        "away_team_name": game["awayTeam"]["commonName"]["default"],
        "away_team_abbrev": game["awayTeam"]["abbrev"],
        "away_team_score": game["awayTeam"].get("score"),
        # Home team
        "home_team_id": game["homeTeam"]["id"],
        "home_team_city": game["homeTeam"]["placeName"]["default"],
        "home_team_name": game["homeTeam"]["commonName"]["default"],
        "home_team_abbrev": game["homeTeam"]["abbrev"],
        "home_team_score": game["homeTeam"].get("score"),
        # Outcome
        "last_period_type": game.get("gameOutcome", {}).get("lastPeriodType"),
        # Timestamps
        "created_at_utc": now_utc,
        "created_at_et": now_et,
    }


# Backfill Schedule
def get_season_games(season: str) -> list[dict]:
    """
    Fetch all regular season game records for a given season.
    Iterates over all team schedules, deduplicates, and returns
    parsed game dictionaries ready for DB upsert.

    Parameters:
    - season: season string in format '20232024'

    Returns:
    - Sorted list of unique parsed game dictionaries
    """
    team_abbrevs = _get_team_abbreviations()
    seen_game_ids = set()
    games = []
    failed_teams = []

    for team in team_abbrevs:
        url = f"{NHL_API_BASE_URL}/v1/club-schedule-season/{team}/{season}"
        try:
            response = requests.get(url, timeout=API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()

            for game in data.get("games", []):
                # Only process regular season games
                if game.get("gameType") != REGULAR_SEASON_GAME_TYPE:
                    continue
                game_id = game["id"]
                if game_id in seen_game_ids:
                    continue
                seen_game_ids.add(game_id)

                try:
                    games.append(_parse_game_data(game))
                except (KeyError, TypeError) as e:
                    print(f"Failed to parse game {game_id} for team {team}: {e}")

        except requests.exceptions.Timeout:
            print(f"Timeout fetching schedule for {team} season {season}")
            failed_teams.append(team)
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error fetching schedule for {team} season {season}: {e}")
            failed_teams.append(team)
        except Exception as e:
            print(f"Unexpected error fetching schedule for {team} season {season}: {e}")
            failed_teams.append(team)

    if failed_teams:
        print(
            f"Warning: failed to fetch schedules for {len(failed_teams)} teams: {failed_teams}"
        )

    return sorted(games, key=lambda g: g["game_id"])


# Nightly Schedule
def get_games_for_date(date: str) -> list[dict]:
    """
    Fetch all game records scheduled for a specific date.
    Used by the nightly pipeline to pull previous day's games.

    Parameters:
    - date: date string in format 'YYYY-MM-DD'

    Returns:
    - List of parsed game dictionaries for that date
    """
    url = f"{NHL_API_BASE_URL}/v1/schedule/{date}"
    try:
        response = requests.get(url, timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()

        games = []
        for day in data.get("gameWeek", []):
            if day.get("date") == date:
                for game in day.get("games", []):
                    if game.get("gameType") != REGULAR_SEASON_GAME_TYPE:
                        continue
                    try:
                        games.append(_parse_game_data(game, date))
                    except (KeyError, TypeError) as e:
                        print(
                            f"Failed to parse game {game.get('id')} for date {date}: {e}"
                        )

        return games

    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout fetching schedule for date {date}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error fetching schedule for date {date}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching schedule for date {date}: {e}")
