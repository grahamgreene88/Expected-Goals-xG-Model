from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from PIL import Image

RINK_IMAGE_PATH = Path(__file__).parent / "static" / "rink_half.png"
_rink_image = Image.open(RINK_IMAGE_PATH)  # loaded once at import time


def make_rink_shot_chart(df: pd.DataFrame) -> go.Figure:
    """
    Plots shots on a full-rink background, colored by xg, with hover
    tooltips showing shot detail.
    """
    df = df[df["x_coord"] >= 0].copy()  # offensive half only

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
            x=0,
            y=42.5,  # top-left anchor, in data coords
            sizex=100,
            sizey=85,  # full rink: 198ft wide, 84ft tall
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

    fig.update_xaxes(range=[0, 100], visible=False)
    fig.update_yaxes(range=[-42.5, 42.5], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
    )

    return fig


def make_shot_heatmap(df: pd.DataFrame, metric: str = "volume") -> go.Figure:
    """
    Plots a binned shot density heatmap over the rink.
    metric: "volume" (shot count per bin) or "avg_xg" (average xG per bin).
    """
    df = df[df["x_coord"] >= 0].copy()  # offensive half only

    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=_rink_image,
            xref="x",
            yref="y",
            x=0,
            y=42.5,
            sizex=100,
            sizey=85,
            sizing="stretch",
            layer="below",
        )
    )

    heatmap_kwargs = dict(
        x=df["x_coord"],
        y=df["y_coord"],
        xbins=dict(start=0, end=100, size=2.5),
        ybins=dict(start=-42.5, end=42.5, size=5),
        opacity=0.75,
        colorscale="YlOrRd",
    )

    if metric == "volume":
        heatmap_kwargs.update(
            histfunc="count",
            colorbar=dict(title="Shots"),
            hovertemplate="Shots: %{z}<extra></extra>",
        )
    else:  # avg_xg
        heatmap_kwargs.update(
            z=df["xg"],
            histfunc="avg",
            zmin=0,
            zmax=0.3,
            colorbar=dict(title="Avg xG"),
            hovertemplate="Avg xG: %{z:.3f}<extra></extra>",
        )

    fig.add_trace(go.Histogram2d(**heatmap_kwargs))

    fig.update_xaxes(range=[0, 100], visible=False)
    fig.update_yaxes(range=[-42.5, 42.5], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)

    return fig


def make_calibration_chart(df: pd.DataFrame) -> go.Figure:
    """
    Plots predicted probability vs. actual goal rate per bin, with a
    perfect-calibration reference line. Marker size reflects bin count.
    """
    fig = go.Figure()

    # Perfect calibration reference line (y = x)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Perfect calibration",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["predicted_mean"],
            y=df["actual_rate"],
            mode="markers+lines",
            marker=dict(
                size=df["count"] / df["count"].max() * 30 + 5,
                color="crimson",
                line=dict(width=1, color="black"),
            ),
            customdata=df[["bin_start", "bin_end", "count"]],
            hovertemplate=(
                "Bin: %{customdata[0]:.2f}–%{customdata[1]:.2f}<br>"
                "Predicted: %{x:.3f}<br>"
                "Actual: %{y:.3f}<br>"
                "n = %{customdata[2]}"
                "<extra></extra>"
            ),
            name="Model calibration",
        )
    )

    fig.update_layout(
        xaxis_title="Predicted probability",
        yaxis_title="Actual goal rate",
        xaxis=dict(range=[0, max(df["predicted_mean"].max(), 0.1) * 1.1]),
        yaxis=dict(range=[0, max(df["actual_rate"].max(), 0.1) * 1.1]),
        height=450,
    )
    return fig


def make_feature_importance_chart(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of feature importances, most important on top."""
    df = df.head(top_n).iloc[::-1]  # reverse so most important renders at top

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker=dict(color="steelblue"),
        )
    )
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title=None,
        height=max(350, top_n * 28),
        margin=dict(l=175, r=15, t=15, b=15),
    )
    return fig
