import pandas as pd

from app.charts import (
    make_shot_heatmap,
)

## Fixtures


def sample_heatmap_df():
    return pd.DataFrame(
        {
            "x_coord": [10, 20, -5],
            "y_coord": [5, -10, 0],
            "xg": [0.1, 0.4, 0.9],
        }
    )


## make_shot_heatmap tests


class TestMakeShotHeatmap:

    def test_make_shot_heatmap_volume_branch(self):
        fig = make_shot_heatmap(sample_heatmap_df(), metric="volume")
        trace = fig.data[0]
        assert trace.histfunc == "count"
        assert trace.z is None  # volume doesn't bind a z column

    def test_make_shot_heatmap_avg_xg_branch(self):
        fig = make_shot_heatmap(sample_heatmap_df(), metric="avg_xg")
        trace = fig.data[0]
        assert trace.histfunc == "avg"
        assert trace.z is not None

    def test_make_shot_heatmap_filters_defensive_half(self):
        fig = make_shot_heatmap(sample_heatmap_df(), metric="volume")
        trace = fig.data[0]
        assert list(trace.x) == [10, 20]
