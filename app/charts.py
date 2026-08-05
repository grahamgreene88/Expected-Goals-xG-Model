from pathlib import Path

import plotly.graph_objects as go
import pandas as pd
from PIL import Image

RINK_IMAGE_PATH = Path(__file__).parent / "static" / "rink_full.png"
_rink_image = Image.open(RINK_IMAGE_PATH)  # loaded once at import time


def make_rink_shot_chart(df: pd.DataFrame) -> go.Figure:
    """
    Plots shots on a full-rink background, colored by xg, with hover
    tooltips showing shot detail.
    """
    df = df.copy()
    df["outcome_label"] = df["is_goal"].map({1: "Goal", 0: "No Goal"})

    # Force standard numpy float64 — pandas nullable Float64 extension
    # dtypes can silently fail to render in Plotly/Streamlit's Arrow pipeline
    for col in ["x_coord", "y_coord", "xg", "shot_distance", "shot_angle"]:
        df[col] = df[col].astype("float64")

    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=_rink_image,
            xref="x",
            yref="y",
            x=-100,
            y=42.5,  # top-left anchor, in data coords
            sizex=198,
            sizey=84,  # full rink: 198ft wide, 84ft tall
            sizing="stretch",
            layer="below",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["x_coord"],
            y=df["y_coord"],
            mode="markers",
            marker=dict(
                size=7,
                color=df["xg"],
                colorscale="YlOrRd",
                colorbar=dict(title="xG"),
                cmin=0,
                cmax=1,
                line=dict(width=0.5, color="black"),
                # size=10,
                # color="red",
                # line=dict(width=0.5, color="black"),
            ),
            customdata=df[
                ["xg", "shot_type", "shot_distance", "shot_angle", "is_goal", "period"]
            ],
            hovertemplate=(
                "<b>xG: %{customdata[0]:.3f}</b><br>"
                "Shot type: %{customdata[1]}<br>"
                "Distance: %{customdata[2]:.1f} ft<br>"
                "Angle: %{customdata[3]:.1f}°<br>"
                "Outcome: %{customdata[4]}<br>"
                "Period: %{customdata[5]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(range=[-100, 100], visible=False)
    fig.update_yaxes(range=[-42.5, 42.5], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
    )

    return fig
