import requests
from pipeline.config import NHL_API_BASE_URL, API_TIMEOUT_SECONDS


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


# Backfill Schedule
def get_season_game_ids(season: str) -> list[int]:
    """
    Fetch all regular season game IDs for a given season.
    Iterates over all team schedules and deduplicates.

    Parameters:
    - season: season string in format '20232024'

    Returns:
    - Sorted list of unique regular season game IDs
    """
    team_abbrevs = _get_team_abbreviations()
    game_ids = set()
    failed_teams = []

    for team in team_abbrevs:
        url = f"{NHL_API_BASE_URL}/v1/club-schedule-season/{team}/{season}"
        try:
            response = requests.get(url, timeout=API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()

            for game in data.get("games", []):
                if game.get("gameType") == 2:
                    game_ids.add(game["id"])

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

    return sorted(game_ids)


# Nightly Schedule
def get_game_ids_for_date(date: str) -> list[int]:
    """
    Fetch all game IDs scheduled for a specific date.
    Used by the nightly pipeline to pull previous day's games.

    Parameters:
    - date: date string in format 'YYYY-MM-DD'

    Returns:
    - List of game IDs for that date (all game types)
    """
    url = f"{NHL_API_BASE_URL}/v1/schedule/{date}"
    try:
        response = requests.get(url, timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()

        game_ids = []
        for day in data.get("gameWeek", []):
            if day.get("date") == date:
                for game in day.get("games", []):
                    game_ids.append(game["id"])

        return game_ids

    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout fetching schedule for date {date}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error fetching schedule for date {date}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching schedule for date {date}: {e}")
