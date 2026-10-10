import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.charts import make_shot_heatmap
from app.data_access import get_scored_shots
from app.filters import render_season_team_filters
from app.styling import SHOT_TYPES

st.set_page_config(page_title="Shot Explorer", page_icon="🏒", layout="wide")

MAX_PLOT_POINTS = 5000

st.title("Shot Explorer")
st.markdown(
    "Explore shot heatmaps colored by shot volume or average expected goal (xG) value. "
    "Use the sidebar to filter by season and team."
)

season, team = render_season_team_filters(key_prefix="shot_explorer")

metric_label = st.radio(
    "Heatmap shows:",
    options=["Shot Volume", "Average xG"],
    horizontal=True,
    key="shot_explorer_metric",
)
metric = "volume" if metric_label == "Shot Volume" else "avg_xg"

shot_types = st.multiselect(
    "Shot types",
    options=SHOT_TYPES,
    default=SHOT_TYPES,
    key="shot_explorer_shot_types",
)

if season is None:
    st.info("Select a season from the sidebar to view shots.")
else:
    df = get_scored_shots(season=season, team=team)
    df = df[df["x_coord"] >= 0]  # offensive half only, matches half-rink image

    if shot_types and set(shot_types) != set(SHOT_TYPES):
        df = df[df["shot_type"].isin(shot_types)]

    if df.empty:
        st.warning("No shots found for the selected filters.")
    else:
        st.caption(f"{len(df):,} shots")
        fig = make_shot_heatmap(df, metric=metric)
        st.plotly_chart(fig, use_container_width=True)
