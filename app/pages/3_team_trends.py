import streamlit as st

from app.charts import (
    make_team_trend_chart,
    make_team_xg_pct_trend_chart,
    make_xgf_pct_ranking_chart,
)
from app.data_access import (
    get_available_seasons,
    get_available_teams,
    get_team_season_summary,
)

st.title("Team xG Trends")

tab1, tab2 = st.tabs(["Team Over Time", "Compare Teams"])

with tab1:
    teams = get_available_teams()
    selected_team = st.selectbox("Team", teams, key="trend_team")

    summary = get_team_season_summary()  # all seasons, filtered client-side below
    team_df = summary[summary["team"] == selected_team]

    if team_df.empty:
        st.info("No data available for this team.")
    else:
        latest = team_df.sort_values("season").iloc[-1]
        st.metric("Current season xGF%:", f"{latest['xGF_pct']:.1%}")
        st.plotly_chart(make_team_xg_pct_trend_chart(team_df), use_container_width=True)
        st.plotly_chart(make_team_trend_chart(team_df), use_container_width=True)

with tab2:
    seasons = get_available_seasons()
    selected_season = st.selectbox("Season", seasons, key="compare_season")

    season_df = get_team_season_summary(season=selected_season)

    if season_df.empty:
        st.info("No data available for this season.")
    else:
        st.plotly_chart(make_xgf_pct_ranking_chart(season_df), use_container_width=True)
