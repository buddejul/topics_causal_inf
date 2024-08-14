from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from pytask import Product, task

from topics_causal_inf.config import BLD, DGPS_TO_RUN
from topics_causal_inf.generic_ml.task_generic_ml_sim import ID_TO_KWARGS

PATHS_TO_RESULTS = [args.path_to_res for _, args in ID_TO_KWARGS.items()]


ID_TO_KWARGS_PLOTS = {
    dgp.name: {
        "path_to_plot": BLD
        / "generic_ml"
        / "plots"
        / f"generic_ml_gates_{dgp.name}.png",
    }
    for dgp in DGPS_TO_RUN
}

for _id, kwargs in ID_TO_KWARGS_PLOTS.items():

    @task(id=_id, kwargs=kwargs)
    def task_generic_ml_plot_gates(
        path_to_plot: Annotated[Path, Product],
        path_to_results: list[Path] = PATHS_TO_RESULTS,
    ):
        """Task for generic_ml plot gates."""
        # Load results
        res = pd.concat([pd.read_pickle(path) for path in path_to_results])

        # Plot Figure
        fig = go.Figure()

        colors = px.colors.qualitative.Plotly

        for i in range(5):
            color = colors[i % len(colors)]  # Cycle through the color palett
            fig.add_trace(
                go.Histogram(
                    x=res[f"gate_{i}"],
                    histnorm="probability",
                    name=f"gate_{i}",
                    marker_color=color,
                ),
            )
            fig.add_vline(
                x=np.mean(res[f"gate_{i}"]),
                line_dash="dash",
                line_color=color,
            )

        fig.update_layout(barmode="overlay")
        fig.update_traces(opacity=0.75)

        fig.write_image(path_to_plot)
