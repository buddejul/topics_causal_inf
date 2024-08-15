from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
from pytask import Product, task

from topics_causal_inf.classes import DGP
from topics_causal_inf.config import BLD, DGPS_TO_RUN
from topics_causal_inf.generic_ml.task_generic_ml_sim import ID_TO_KWARGS

PATHS_TO_RESULTS = [args.path_to_res for _, args in ID_TO_KWARGS.items()]


ID_TO_KWARGS_GATES = {
    dgp.name: {
        "dgp": dgp,
        "path_to_plot": BLD
        / "generic_ml"
        / "plots"
        / f"generic_ml_gates_{dgp.name}.png",
    }
    for dgp in DGPS_TO_RUN
}

for _id, kwargs in ID_TO_KWARGS_GATES.items():

    @task(id=_id, kwargs=kwargs)
    def task_generic_ml_plot_gates(
        dgp: DGP,
        path_to_plot: Annotated[Path, Product],
        path_to_results: list[Path] = PATHS_TO_RESULTS,
    ):
        """Task for generic_ml plot gates."""
        # Load results
        res = pd.concat([pd.read_pickle(path) for path in path_to_results])

        res = res[res["dgp"] == dgp.name]

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
                x=np.mean(res[f"true_gate_{i}"]),
                line_dash="dash",
                line_color=color,
            )

        fig.update_layout(barmode="overlay")
        fig.update_traces(opacity=0.75)

        fig.write_image(path_to_plot)


ID_TO_KWARGS_CLANS = {
    dgp.name: {
        "dgp": dgp,
        "path_to_plot": BLD
        / "generic_ml"
        / "plots"
        / f"generic_ml_clans_{dgp.name}.png",
    }
    for dgp in DGPS_TO_RUN
}

for _id, kwargs in ID_TO_KWARGS_CLANS.items():

    @task(id=_id, kwargs=kwargs)
    def task_generic_ml_plot_clans(
        dgp: DGP,
        path_to_plot: Annotated[Path, Product],
        path_to_results: list[Path] = PATHS_TO_RESULTS,
    ):
        """Task for generic_ml plot clans."""
        # Load results
        res = pd.concat([pd.read_pickle(path) for path in path_to_results])

        res = res[res["dgp"] == dgp.name]

        colors = px.colors.qualitative.Plotly

        fig = make_subplots(rows=2, cols=2)

        for i in range(4):
            row = i // 2 + 1
            col = i % 2 + 1

            color = colors[i % len(colors)]  # Cycle through the color palett
            fig.add_trace(
                go.Histogram(
                    x=res[f"clan_0_{i}"],
                    histnorm="probability",
                    nbinsx=100,
                    autobinx=False,
                    name=f"X{i}",
                    marker_color=color,
                    legendgroup="group1",
                    legendgrouptitle_text="Least Affected Group",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Histogram(
                    x=res[f"clan_1_{i}"],
                    histnorm="probability",
                    nbinsx=100,
                    autobinx=False,
                    marker={"color": color, "pattern": {"shape": "/"}},
                    name=f"X{i}",
                    marker_color=color,
                    legendgroup="group2",
                    legendgrouptitle_text="Most Affected Group",
                ),
                row=row,
                col=col,
            )
            fig.add_vline(
                x=np.mean(res[f"true_clan_0_{i}"]),
                line_dash="dash",
                line_color=color,
                row=row,
                col=col,
            )
            fig.add_vline(
                x=np.mean(res[f"true_clan_1_{i}"]),
                line_dash="dash",
                line_color=color,
                row=row,
                col=col,
            )

            fig.update_xaxes(range=[0, 1], row=row, col=col)

        fig.update_traces(opacity=0.75)

        fig.write_image(path_to_plot)
