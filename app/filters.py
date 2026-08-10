import streamlit as st

from app.data_access import get_available_seasons, get_available_teams


def render_season_team_filters(key_prefix: str) -> tuple[str | None, str | None]:
    """
    Renders independent season and team selectboxes in the sidebar.
    key_prefix namespaces widget keys so the same filter pair can be
    reused on multiple pages without Streamlit key collisions.
    Returns (season, team), either of which may be None if 'All' is selected.
    """
    seasons = get_available_seasons()
    teams = get_available_teams()  # unfiltered — independent of season

    season = st.sidebar.selectbox(
        "Season",
        options=["All"] + seasons,
        key=f"{key_prefix}_season",
    )
    team = st.sidebar.selectbox(
        "Team",
        options=["All"] + teams,
        key=f"{key_prefix}_team",
    )

    return (
        None if season == "All" else season,
        None if team == "All" else team,
    )
