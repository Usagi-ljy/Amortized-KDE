"""Reusable result tables, benchmark figures and result displays."""

from __future__ import annotations

from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .config import (
    FIGURE_DIR,
    GAUSSIAN_BENCHMARK_FIGURE,
    GMM32_BENCHMARK_FIGURE,
    MULTIFAMILY_BENCHMARK_FIGURES,
)
from .core import (
    curves_to_frame,
    dataframe_to_csv_bytes,
    distribution_request_url,
    format_number,
)
from .kde_estimators import KDEResult
from .kde_plotting import plot_kde_comparison
from .neural_bandwidth_selectors import NeuralBandwidthResult
from .simulation_tasks import SimulationTask

def bandwidth_table(
    bandwidths: Mapping[str, float],
    *,
    bounded_results: Optional[Mapping[str, KDEResult]] = None,
    unbounded_results: Optional[Mapping[str, KDEResult]] = None,
    scores: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, bandwidth in bandwidths.items():
        row: dict[str, object] = {
            "Method": method,
            "Bandwidth": bandwidth,
        }
        if bounded_results is not None:
            row["Bounded integral"] = bounded_results[method].numerical_integral
        if unbounded_results is not None:
            row["Displayed unbounded integral"] = unbounded_results[
                method
            ].numerical_integral
        if scores is not None:
            row["Empirical log score (bits)"] = scores[method]
        rows.append(row)
    return pd.DataFrame(rows)


def render_benchmark_figure(
    filename: str,
    *,
    title: str,
    caption: str,
) -> None:
    """Display a browser-friendly copy of one existing paper figure."""

    st.subheader(title)
    image_path = FIGURE_DIR / filename
    if image_path.is_file():
        st.image(
            str(image_path),
            caption=caption,
            width="stretch",
        )
        return

    eps_path = image_path.with_suffix(".eps")
    if eps_path.is_file():
        st.warning(
            f"Found `{eps_path.name}`, but browsers do not reliably render EPS. "
            f"Export the same figure as `{filename}` and place it in `figures/`."
        )
    else:
        st.info(
            f"Add `{filename}` to the repository's `figures/` directory. "
            "The matching EPS file may be stored beside it using the same stem."
        )


def render_benchmarks() -> None:
    """Show compact benchmark tabs before the interactive simulator."""

    st.header("Benchmark results")
    st.write(
        "The aggregate figures reproduce the repeated experiments reported "
        "in the paper. Family-specific figures extend that evaluation using "
        "the same Multi-family selector. Shaded bands are paired 90% bootstrap "
        "intervals; lower logarithmic score is better."
    )

    gaussian_tab, family_tab, gmm_tab = st.tabs(
        ("Gaussian", "Distribution families", "GMM K=32")
    )

    with gaussian_tab:
        render_benchmark_figure(
            GAUSSIAN_BENCHMARK_FIGURE,
            title="Gaussian benchmark",
            caption=(
                "Empirical logarithmic score versus sample size under Gaussian "
                "sampling, averaged over 30,000 independent tasks at each n."
            ),
        )

    with family_tab:
        available_figures = {
            label: filename
            for label, filename in MULTIFAMILY_BENCHMARK_FIGURES.items()
            if (FIGURE_DIR / filename).is_file()
        }
        if available_figures:
            selected_benchmark_family = st.selectbox(
                "Underlying distribution",
                tuple(available_figures),
                help=(
                    "Only distributions with an available PNG are listed. "
                    "New choices appear automatically when their figures are "
                    "added to the figures directory."
                ),
            )
            selected_family_figure = available_figures[
                selected_benchmark_family
            ]
            if selected_benchmark_family == "Multi-family":
                family_caption = (
                    "Equal-weight aggregate across the ten bounded families "
                    "on [-1, 1], using 30,000 tasks at each n. Multi-family "
                    "means one family is drawn for each task; the ten densities "
                    "are not averaged into a new distribution."
                )
            else:
                family_caption = (
                    "Multi-family selector evaluated on "
                    f"{selected_benchmark_family} tasks, using 10,000 tasks "
                    "at each n and paired bootstrap intervals."
                )
            st.image(
                str(FIGURE_DIR / selected_family_figure),
                caption=family_caption,
                width="stretch",
            )
        else:
            st.info(
                "Add the Multi-family aggregate or family-specific PNG files "
                "to the repository's `figures/` directory."
            )

        st.caption(
            "Observed data from another one-dimensional continuous "
            "distribution can already be analysed in the Data workflow. A "
            "request asks for that family to be added to Simulation with a "
            "sampler, true density and log-score evaluation."
        )
        st.link_button(
            "Request another distribution",
            distribution_request_url(),
        )

    with gmm_tab:
        render_benchmark_figure(
            GMM32_BENCHMARK_FIGURE,
            title="Bounded GMM K=32 benchmark",
            caption=(
                "Empirical logarithmic score versus sample size under bounded "
                "K=32 GMM sampling on [-1, 1], averaged over 30,000 independent "
                "tasks at each n."
            ),
        )


def render_neural_details(neural_result: Optional[NeuralBandwidthResult]) -> None:
    if neural_result is None:
        return
    feature_frame = pd.DataFrame(
        {
            "Feature": neural_result.feature_names,
            "Reference-scale value": neural_result.features,
        }
    )
    st.markdown(f"**Loaded selector:** `{neural_result.selector}`")
    st.markdown(
        f"**Reference-scale bandwidth:** "
        f"`{format_number(neural_result.reference_bandwidth)}`"
    )
    if neural_result.bandwidth_ratio is not None:
        st.markdown(
            f"**Predicted bandwidth ratio:** "
            f"`{format_number(neural_result.bandwidth_ratio)}`"
        )
    st.dataframe(feature_frame, hide_index=True, width="stretch")


def render_user_result(result: Mapping[str, object]) -> None:
    sample = np.asarray(result["sample"], dtype=np.float64)
    support = result["support"]
    bandwidths = result["bandwidths"]
    bounded_results = result["bounded_results"]
    unbounded_results = result["unbounded_results"]

    st.divider()
    st.subheader("Comparison result")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Sample size", int(sample.size))
    metric_columns[1].metric("Selected methods", len(bandwidths))
    metric_columns[2].metric(
        "Working interval",
        f"[{format_number(support[0])}, {format_number(support[1])}]",
    )

    bounded_tab, unbounded_tab = st.tabs(("Bounded KDE", "Unbounded KDE"))
    with bounded_tab:
        bounded_figure = plot_kde_comparison(
            bounded_results,
            samples=sample,
            show_histogram=False,
            show_rug=True,
            title="Bounded KDE: truncated and renormalized",
            figure_size=(10.5, 5.4),
        )
        st.pyplot(bounded_figure, clear_figure=True, width="stretch")
        plt.close(bounded_figure)
    with unbounded_tab:
        unbounded_figure = plot_kde_comparison(
            unbounded_results,
            samples=sample,
            show_histogram=False,
            show_rug=True,
            title="Unbounded KDE: ordinary Gaussian kernels",
            figure_size=(10.5, 5.4),
        )
        st.pyplot(unbounded_figure, clear_figure=True, width="stretch")
        plt.close(unbounded_figure)

    st.caption(
        "The bounded and unbounded views use the same observations and the "
        "same method-specific bandwidths. Only the density normalization and "
        "display support differ."
    )
    table = bandwidth_table(
        bandwidths,
        bounded_results=bounded_results,
        unbounded_results=unbounded_results,
    )
    st.dataframe(
        table.style.format(
            {
                "Bandwidth": "{:.6g}",
                "Bounded integral": "{:.6f}",
                "Displayed unbounded integral": "{:.6f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    download_left, download_right = st.columns(2)
    bounded_frame = curves_to_frame(bounded_results)
    unbounded_frame = curves_to_frame(unbounded_results)
    download_left.download_button(
        "Download bounded density CSV",
        dataframe_to_csv_bytes(bounded_frame),
        file_name="bounded_kde_comparison.csv",
        mime="text/csv",
        width="stretch",
    )
    download_right.download_button(
        "Download unbounded density CSV",
        dataframe_to_csv_bytes(unbounded_frame),
        file_name="unbounded_kde_comparison.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Technical details"):
        st.markdown(
            "Every method first sees the sample after the working interval is "
            "mapped to `[-1, 1]`. Its bandwidth is then multiplied by "
            "`(B - A) / 2` to return to the original units. For the three "
            "classical affine-equivariant rules, this is theoretically the "
            "same bandwidth as direct computation on the original scale."
        )
        render_neural_details(result["neural_result"])


def render_simulation_result(result: Mapping[str, object]) -> None:
    task: SimulationTask = result["task"]
    bandwidths = result["bandwidths"]
    curves = result["curves"]
    scores = result["scores"]
    true_density = np.asarray(result["true_density"], dtype=np.float64)

    st.divider()
    st.subheader(task.title)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Observed sample", int(task.observed.size))
    metric_columns[1].metric("Independent test sample", int(task.test.size))
    metric_columns[2].metric("Selected methods", len(bandwidths))

    density_figure = plot_kde_comparison(
        curves,
        samples=task.observed,
        true_density=true_density,
        true_density_label="True underlying density",
        show_histogram=False,
        show_rug=True,
        title="Density estimates for this generated task",
        figure_size=(10.5, 5.4),
    )
    st.pyplot(density_figure, clear_figure=True, width="stretch")
    plt.close(density_figure)

    if task.kde_mode == "unbounded":
        st.caption(
            "The horizontal range is only a finite display window for an "
            "unbounded density; it is not a support boundary and the KDE is "
            "not truncated there."
        )
    else:
        st.caption(
            "The true distribution and all bounded KDEs are supported on "
            f"[{format_number(task.working_support[0])}, "
            f"{format_number(task.working_support[1])}] and are normalized on "
            "that interval."
        )

    st.markdown("#### Numerical results for this generated task")
    st.write(
        "The bandwidth is selected from the observed sample. The empirical "
        "logarithmic score is then evaluated on the independent test sample "
        "and is not used to fit the bandwidth. Lower values are better."
    )
    table = bandwidth_table(bandwidths, scores=scores)
    st.dataframe(
        table.style.format(
            {
                "Bandwidth": "{:.6g}",
                "Empirical log score (bits)": "{:.6f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    curve_frame = curves_to_frame(curves, true_density=true_density)
    score_frame = table[["Method", "Bandwidth", "Empirical log score (bits)"]]
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download simulation curves",
        dataframe_to_csv_bytes(curve_frame),
        file_name="simulation_kde_curves.csv",
        mime="text/csv",
        width="stretch",
    )
    download_right.download_button(
        "Download bandwidths and scores",
        dataframe_to_csv_bytes(score_frame),
        file_name="simulation_log_scores.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Generated-task and selector details"):
        metadata_frame = pd.DataFrame(
            {
                "Setting": list(task.metadata),
                "Value": [str(value) for value in task.metadata.values()],
            }
        )
        st.dataframe(metadata_frame, hide_index=True, width="stretch")
        st.markdown(
            f"**Working interval used for all bandwidth selectors:** "
            f"`[{format_number(task.working_support[0])}, "
            f"{format_number(task.working_support[1])}]`"
        )
        render_neural_details(result["neural_result"])
