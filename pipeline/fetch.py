import time

import requests
from prefect import get_run_logger

from pipeline.config import (
    API_MAX_RETRIES,
    API_RATE_LIMIT_SECONDS,
    API_TIMEOUT_SECONDS,
    NHL_API_BASE_URL,
)


def get_play_by_play(game_id: int) -> dict | None:
    """
    Fetch raw play-by-play data for a single game from the NHL API.
    Retries up to API_MAX_RETRIES times with exponential backoff.

    Parameters:
    - game_id: NHL game ID

    Returns:
    - Raw API response as a dictionary, or None if all retries failed
    """
    logger = get_run_logger()

    url = f"{NHL_API_BASE_URL}/v1/gamecenter/{game_id}/play-by-play"

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=API_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning(
                f"Timeout for game {game_id} (attempt {attempt}/{API_MAX_RETRIES})"
            )
        except requests.exceptions.HTTPError as e:
            status_code = (
                e.response.status_code if e.response is not None else "unknown"
            )
            logger.warning(
                f"HTTP {status_code} error for game {game_id} (attempt {attempt}/{API_MAX_RETRIES})"
            )
            # Don't retry client errors — they won't resolve on their own
            if e.response is not None and e.response.status_code < 500:
                logger.warning(f"Client error for game {game_id}, not retrying")
                return None
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Connection error for game {game_id} (attempt {attempt}/{API_MAX_RETRIES})"
            )
        except Exception as e:
            logger.warning(
                f"Unexpected error for game {game_id} (attempt {attempt}/{API_MAX_RETRIES}): {e}"
            )

        if attempt < API_MAX_RETRIES:
            wait = API_RATE_LIMIT_SECONDS * (2 ** (attempt - 1))
            logger.info(f"Retrying game {game_id} in {wait:.1f}s...")
            time.sleep(wait)

    logger.error(f"All retries exhausted for game {game_id}")
    return None


def get_game_state(game_id: int) -> str | None:
    """
    Fetch the current game state for a single game.
    Used by the nightly pipeline to check if a game is final before processing.

    Parameters:
    - game_id: NHL game ID

    Returns:
    - game state string (e.g. 'OFF', 'LIVE', 'FUT') or None if fetch failed
    """
    logger = get_run_logger()

    url = f"{NHL_API_BASE_URL}/v1/gamecenter/{game_id}/landing"

    try:
        response = requests.get(url, timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return data.get("gameState")

    except Exception as e:
        logger.error(f"Failed to fetch game state for game {game_id}: {e}")
        return None
