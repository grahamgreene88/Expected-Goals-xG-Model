# app/Home.py
import sys
from pathlib import Path

# Ensure project root is importable, regardless of where `streamlit run` is invoked from
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.data_access import get_headline_stats

st.set_page_config(
    page_title="NHL xG Model",
    page_icon="🏒",
    layout="wide",
)

st.title("🏒 NHL Expected Goals (xG) Explorer")

st.markdown("""
    An expected-goals model trained on NHL shot data (seasons 2019-20 through 2024-25),
    using features like shot position, shot type, game state and others to estimate the probability
    that a given shot results in a goal.
    """)

st.divider()

stats = get_headline_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Shots Modeled", f"{stats['total_shots']:,}")
col2.metric("Seasons Covered", stats["season_count"])
col3.metric("Model AUC (2025-26 test)", f"{stats['test_auc']:.3f}")
col4.metric("Model Brier Score", f"{stats['test_brier']:.4f}")

st.divider()

st.markdown("""
    **Pages**
    - **Shot Explorer** — visualize shots based on volume or average xG on a rink diagram; filter by shot type
    - **Model Performance** — performance metrics, calibration curve and feature importance charts
    - **Team Trends** — xG vs. actual goals over time, by team
    - **COMING SOON: Player Lookup** — individual skater xG profile
    """)
