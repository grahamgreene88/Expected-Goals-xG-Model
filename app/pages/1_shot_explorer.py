import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.filters import render_season_team_filters
from app.data_access import get_scored_shots
from app.charts import make_rink_shot_chart

st.set_page_config(page_title="Shot Explorer", page_icon="🏒", layout="wide")

MAX_PLOT_POINTS = 5000

st.title("Shot Explorer")
st.markdown(
    "Explore individual shots colored by expected goal (xG) value. "
    "Use the sidebar to filter by season and team."
)

season, team = render_season_team_filters(key_prefix="shot_explorer")

if season is None:
    st.info("Select a season from the sidebar to view shots.")
else:
    df = get_scored_shots(season=season, team=team)
    if df.empty:
        st.warning("No shots found for the selected filters.")
    else:
        if len(df) > MAX_PLOT_POINTS:
            st.caption(
                f"Showing a random sample of {MAX_PLOT_POINTS:,} of {len(df):,} shots."
            )
            df = df.sample(MAX_PLOT_POINTS, random_state=42)
        else:
            st.caption(f"Showing {len(df):,} shots")

        fig = make_rink_shot_chart(df)
        st.plotly_chart(fig, use_container_width=True)
