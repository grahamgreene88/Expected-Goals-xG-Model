from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pipeline.config import VALID_SHOT_TYPES


# Coordinate Normalization
def normalize_coordinates(
    x: float | None,
    y: float | None,
    is_home_team: bool,
    home_team_defending_side: str | None,
) -> tuple[float | None, float | None, bool]:
    """
    Normalize shot coordinates so the shooting team always attacks
    in the positive x direction.

    Returns:
        (x_coord, y_coord, coord_normalized)
        coord_normalized is True when normalization logic ran successfully.
        Fallback cases return raw coordinates with coord_normalized=False.
    """
    if x is None or y is None:
        return x, y, False

    if home_team_defending_side is None:
        # TODO: Implement fallback for seasons without 'homeTeamDefendingSide'
        return x, y, False

    if home_team_defending_side == "left":
        attacking_left = not is_home_team
    elif home_team_defending_side == "right":
        attacking_left = is_home_team
    else:
        # Unexpected value — return raw, flag as not normalized
        return x, y, False

    if attacking_left:
        return -x, y, True
    else:
        # Shooting team already attacks right — no flip needed, but
        # normalization logic ran successfully
        return x, y, True


# Shot Parsing
def parse_shots(data: dict, game_id: int) -> list[dict]:
    """
    Extract and parse shot events from raw play-by-play API response.

    Parameters:
    - data: raw play-by-play API response
    - game_id: NHL game ID

    Returns:
    - List of shot dictionaries ready for DB upsert
    """
    shot_events = []
    play_data = data.get("plays", [])

    # Team Context
    home_team = data.get("homeTeam", {})
    away_team = data.get("awayTeam", {})

    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))

    for event in play_data:
        event_type = event.get("typeDescKey")
        if event_type not in VALID_SHOT_TYPES:
            continue

        try:
            details = event.get("details", {})

            # Identify shooter ID and their team
            shooter_id = (
                details.get("scoringPlayerId")
                if event_type == "goal"
                else details.get("shootingPlayerId")
            )
            shooter_team_id = details.get("eventOwnerTeamId")
            # Determine if shooter is on home or away team
            is_home_team = shooter_team_id == home_team.get("id")
            shooter_team_abbrev = (
                home_team.get("abbrev") if is_home_team else away_team.get("abbrev")
            )
            opponent_team_abbrev = (
                away_team.get("abbrev") if is_home_team else home_team.get("abbrev")
            )

            # Situation code
            # Parse situation code and compute strength state
            # 4-digit situation code has this format: A-G-S-H  → Away goalie, Away skaters, Home skaters, Home goalie
            situation_code = event.get("situationCode")
            if isinstance(situation_code, int):
                situation_code = str(situation_code)

            if not (
                isinstance(situation_code, str)
                and len(situation_code) == 4
                and situation_code.isdigit()
            ):
                strength_state = None
                away_goalie_pulled = None
                home_goalie_pulled = None
                shooting_team_strength_state = None
                shooting_team_strength_diff = None
            else:
                away_goalie_pulled = situation_code[0] == "0"
                home_goalie_pulled = situation_code[3] == "0"
                # strength_state is 'Away skaters'v'Home skaters' e.g. '4v5' for Home team on powerplay
                strength_state = f"{situation_code[1]}v{situation_code[2]}"
                # Derive skater counts
                away_skaters = int(situation_code[1])
                home_skaters = int(situation_code[2])

                if is_home_team is True:
                    shooting_team_skaters = home_skaters
                    defending_team_skaters = away_skaters
                elif is_home_team is False:
                    shooting_team_skaters = away_skaters
                    defending_team_skaters = home_skaters
                else:
                    shooting_team_skaters = None
                    defending_team_skaters = None

                if (
                    shooting_team_skaters is not None
                    and defending_team_skaters is not None
                ):
                    shooting_team_strength_state = (
                        f"{shooting_team_skaters}v{defending_team_skaters}"
                    )
                    shooting_team_strength_diff = (
                        shooting_team_skaters - defending_team_skaters
                    )
                else:
                    shooting_team_strength_state = None
                    shooting_team_strength_diff = None

            # Shot Coordinates
            x_coord_raw = details.get("xCoord")
            y_coord_raw = details.get("yCoord")
            home_team_defending_side = event.get("homeTeamDefendingSide")
            x_coord, y_coord, coord_normalized = normalize_coordinates(
                x_coord_raw, y_coord_raw, is_home_team, home_team_defending_side
            )

            # Assemble shot record
            shot_events.append(
                {
                    # Game context
                    "game_id": game_id,
                    "event_id": event.get("eventId"),
                    "sort_order": event.get("sortOrder"),
                    "period": event.get("periodDescriptor", {}).get("number"),
                    "period_type": event.get("periodDescriptor", {}).get("periodType"),
                    "time_in_period": event.get("timeInPeriod"),
                    "time_remaining": event.get("timeRemaining"),
                    "situation_code": situation_code,
                    "strength_state": strength_state,
                    "away_goalie_pulled": away_goalie_pulled,
                    "home_goalie_pulled": home_goalie_pulled,
                    "shooting_team_strength_state": shooting_team_strength_state,
                    "shooting_team_strength_diff": shooting_team_strength_diff,
                    # Player and team info
                    "shooter_id": shooter_id,
                    "goalie_id": details.get("goalieInNetId"),
                    "shooter_team_id": shooter_team_id,
                    "shooter_team_abbrev": shooter_team_abbrev,
                    "opponent_team_abbrev": opponent_team_abbrev,
                    "is_home_team": is_home_team,
                    # Shot event info
                    "event_type": event_type,
                    "x_coord_raw": x_coord_raw,
                    "y_coord_raw": y_coord_raw,
                    "x_coord": x_coord,
                    "y_coord": y_coord,
                    "coord_normalized": coord_normalized,
                    "zone": details.get("zoneCode"),
                    "shot_type": details.get("shotType"),
                    "is_goal": event_type == "goal",
                    # Miss reason
                    "miss_reason": (
                        details.get("reason") if event_type == "missed-shot" else None
                    ),
                    # Created at
                    "created_at_utc": now_utc,
                    "created_at_et": now_et,
                }
            )

        except Exception as e:
            print(f"Skipping event {event.get('eventId')} in game {game_id}: {e}")
            continue

    return shot_events
