import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from prefect import flow, task, get_run_logger

from pipeline.config import (
    BACKFILL_SEASONS,
    API_RATE_LIMIT_SECONDS,
)
from pipeline.schedule import get_season_games, get_games_for_date
from pipeline.fetch import get_play_by_play
from pipeline.parse import parse_shots
from pipeline.db import (
    is_game_processed,
    upsert_game,
    upsert_shots,
    write_pipeline_log,
    get_connection,
)


# Shared Task
@task
def process_game(game: dict) -> None:
    """
    Process a single game through stages 2-5:
    - Dedup check
    - Play-by-play fetch
    - Shot parsing
    - DB upsert and pipeline log write
    """
    logger = get_run_logger()
    game_id = game["game_id"]

    # Stage 2: Dedup check
    if is_game_processed(game_id):
        logger.info(f"Game {game_id} already processed, skipping")
        write_pipeline_log(game_id, "skipped_duplicate")
        return

    # Nightly only; skip games not yet final
    if game.get("game_state") != "OFF":
        logger.info(f"Game {game_id} not final (state={game['game_state']}), skipping")
        write_pipeline_log(game_id, "skipped_in_progress")
        return

    # Stage 3: Play-by-play fetch
    data = get_play_by_play(game_id)
    if data is None:
        logger.error(f"Game {game_id} failed to fetch play-by-play")
        write_pipeline_log(game_id, "failed", error_message="Play-by-play fetch failed")
        return

    # Stage 4: Shot parsing
    try:
        shots = parse_shots(data, game_id)
    except Exception as e:
        logger.error(f"Game {game_id} failed during shot parsing: {e}")
        write_pipeline_log(game_id, "failed", error_message=f"Shot parsing failed: {e}")
        return

    # Stage 5: DB upsert and pipeline log
    try:
        upsert_game(game)
        shots_inserted = upsert_shots(shots)
        write_pipeline_log(game_id, "success", shots_inserted=shots_inserted)
        logger.info(
            f"Game {game_id} processed successfully — {shots_inserted} shots inserted"
        )
    except Exception as e:
        logger.error(f"Game {game_id} failed during DB write: {e}")
        write_pipeline_log(game_id, "failed", error_message=f"DB write failed: {e}")


# Backfill Flow
@flow(name="nhl-xg-backfill")
def backfill_flow() -> None:
    """
    One-time historical backfill covering all seasons in BACKFILL_SEASONS.
    Fetches game records, then processes each game sequentially with rate limiting.
    """
    logger = get_run_logger()

    for season in BACKFILL_SEASONS:
        logger.info(f"Starting season {season}")

        # Stage 1: Fetch game records
        games = get_season_games(season)
        logger.info(f"Season {season}: {len(games)} games found")

        for i, game in enumerate(games):
            process_game(game)
            time.sleep(API_RATE_LIMIT_SECONDS)

            if (i + 1) % 100 == 0:
                logger.info(f"Season {season}: processed {i + 1}/{len(games)} games")

        logger.info(f"Season {season} complete")

    logger.info("Backfill complete")


# Nightly Flow
@flow(name="nhl-xg-nightly")
def nightly_flow() -> None:
    """
    Nightly pipeline — runs at 4:00 AM ET, pulls games from the previous day.
    Only processes games where game_state == 'OFF' (final).
    Games not yet final are logged as skipped_in_progress and retried next run.
    """
    logger = get_run_logger()

    # Compute yesterday's date in ET
    yesterday = (
        datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    logger.info(f"Nightly run for date: {yesterday}")

    # Stage 1: Fetch game records
    try:
        games = get_games_for_date(yesterday)
    except RuntimeError as e:
        logger.error(f"Failed to fetch schedule for {yesterday}: {e}")
        return

    if not games:
        logger.info(f"No games found for {yesterday}")
        return

    logger.info(f"Found {len(games)} games for {yesterday}")

    for game in games:
        process_game(game)
        time.sleep(API_RATE_LIMIT_SECONDS)

    logger.info("Nightly run complete")


@flow(name="nhl-xg-catchup")
def catchup_flow() -> None:
    """
    Catchup pipeline — finds the most recent processed game date for the
    current season and processes all games from that date through yesterday.
    *Used to manually keep the 2025-26 season current until cloud deployment.
    """
    logger = get_run_logger()

    # Find the most recent processed game date for the current season
    query = """
        SELECT MAX(game_date) FROM games WHERE season = 20252026
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()

    if result is None or result[0] is None:
        logger.error("No 2025-26 games found in database, run backfill first")
        return

    last_processed_date = result[0]
    yesterday = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).date()

    if last_processed_date >= yesterday:
        logger.info("Already up to date, no catchup needed")
        return

    logger.info(f"Catching up from {last_processed_date} to {yesterday}")

    # Iterate through each date from the day after last processed through yesterday
    current_date = last_processed_date + timedelta(days=1)
    while current_date <= yesterday:
        date_str = current_date.strftime("%Y-%m-%d")
        logger.info(f"Processing date: {date_str}")

        try:
            games = get_games_for_date(date_str)
            if not games:
                logger.info(f"No games found for {date_str}")
            else:
                logger.info(f"Found {len(games)} games for {date_str}")
                for game in games:
                    process_game(game)
                    time.sleep(API_RATE_LIMIT_SECONDS)
        except RuntimeError as e:
            logger.error(f"Failed to fetch schedule for {date_str}: {e}")

        current_date += timedelta(days=1)

    logger.info("Catchup complete")


# Run the backfill only when this file is executed as a script
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "nightly":
        nightly_flow()
    elif len(sys.argv) > 1 and sys.argv[1] == "catchup":
        catchup_flow()
    else:
        backfill_flow()
