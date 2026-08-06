import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.charts import make_calibration_chart, make_feature_importance_chart
from app.data_access import (
    get_calibration_data,
    get_feature_importance_data,
    get_headline_stats,
)

st.set_page_config(page_title="Model Performance", page_icon="🏒", layout="wide")

st.title("Model Performance")
st.markdown(
    "Evaluation of the production XGBoost model on the 2025-26 held-out test season."
)

stats = get_headline_stats()

col1, col2 = st.columns(2)
col1.metric("AUC", f"{stats['test_auc']:.3f}")
col2.metric("Brier Score", f"{stats['test_brier']:.4f}")

st.divider()

st.subheader("Calibration")
st.markdown(
    "Each point represents a bin of predicted probabilities. Points near the "
    "dashed line indicate well-calibrated predictions; marker size reflects "
    "the number of shots in that bin."
)
calibration_df = get_calibration_data(model="xgboost")
st.plotly_chart(
    make_calibration_chart(calibration_df), use_container_width=True, theme=None
)

st.divider()

st.subheader("Feature Importance")
importance_df = get_feature_importance_data(model="xgboost")
st.plotly_chart(
    make_feature_importance_chart(importance_df), use_container_width=True, theme=None
)
