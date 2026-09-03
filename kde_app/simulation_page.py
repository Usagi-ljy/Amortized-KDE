"""Benchmark display and interactive Simulation workflow."""

from __future__ import annotations

import numpy as np
import streamlit as st

from .analytics import track_event
from .classical_bandwidth_selectors import sheather_jones_is_available
from .config import (
    DEFAULT_TEST_SIZE,
    MAX_SAMPLE_SIZE,
    METHOD_OPTIONS,
    MIN_SAMPLE_SIZE,
    MULTIFAMILY_OPTIONS,
)
from .core import default_method_selection
from .simulation_tasks import (
    generate_gaussian_task,
    generate_gmm32_task,
    generate_multifamily_task,
    run_simulation,
)
from .ui_components import render_benchmarks, render_simulation_result


def render_simulation_page() -> None:
    render_benchmarks()

    st.divider()
    st.header("Interactive single-task simulation")
    st.write(
        "Generate one new random task from a known underlying distribution, "
        "then compare the selected KDEs with its true density. This section "
        "illustrates one concrete sample and should not be interpreted as the "
        "aggregate benchmark trend reported above."
    )
    st.write(
        "The observed sample is used to select each bandwidth and construct "
        "the KDE. A separate independent test sample is used only to report "
        "the empirical logarithmic score in the result table."
    )

    simulation_kind = st.radio(
        "Simulation setup",
        ("Gaussian", "Distribution family", "GMM K=32"),
        horizontal=True,
    )

    parameter_left, parameter_middle, parameter_right = st.columns(3)
    with parameter_left:
        simulation_n = st.slider(
            "Observed sample size",
            MIN_SAMPLE_SIZE,
            MAX_SAMPLE_SIZE,
            64,
            help="Number of observations used to select bandwidths and fit KDEs.",
        )
    with parameter_middle:
        simulation_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=2026,
            step=1,
            help="Use the same seed to reproduce exactly the same generated task.",
        )
    with parameter_right:
        test_size = st.select_slider(
            "Independent test size",
            options=(512, 1024, 2048, 4096),
            value=DEFAULT_TEST_SIZE,
            help=(
                "Independent observations used only for empirical log-score "
                "evaluation; they are not used to choose bandwidths."
            ),
        )

    gaussian_mean = 0.0
    gaussian_standard_deviation = 1.0
    selected_family = "Multi-family (random family)"
    if simulation_kind == "Gaussian":
        gaussian_left, gaussian_right = st.columns(2)
        with gaussian_left:
            gaussian_mean = st.number_input(
                "Gaussian mean",
                value=0.0,
                format="%.6g",
            )
        with gaussian_right:
            gaussian_standard_deviation = st.number_input(
                "Gaussian standard deviation",
                min_value=0.000001,
                value=1.0,
                format="%.6g",
            )
        st.caption(
            "The default is the paper's standard-normal benchmark. Gaussian "
            "KDEs are evaluated on the real line."
        )
    elif simulation_kind == "Distribution family":
        selected_family = st.selectbox(
            "Underlying distribution",
            MULTIFAMILY_OPTIONS,
        )
        st.caption(
            "“Multi-family” draws one of the ten training families uniformly "
            "for this task; it does not average ten densities into a new "
            "mixture. Use the request link in the benchmark section to suggest "
            "another simulation family."
        )
    else:
        st.caption(
            "A fresh 32-component Gaussian mixture is drawn and conditioned "
            "exactly on [-1, 1], following the GMM K=32 task generator."
        )

    simulation_methods = st.multiselect(
        "Bandwidth methods",
        METHOD_OPTIONS,
        default=default_method_selection(),
        key="simulation_methods",
    )
    if (
        "Sheather–Jones" in simulation_methods
        and not sheather_jones_is_available()
    ):
        st.warning(
            "Exact Sheather–Jones requires Rscript. Add `r-base-core` to "
            "`packages.txt` for Streamlit Community Cloud."
        )

    simulation_signature = (
        simulation_kind,
        int(simulation_n),
        int(simulation_seed),
        int(test_size),
        (
            float(gaussian_mean),
            float(gaussian_standard_deviation),
        )
        if simulation_kind == "Gaussian"
        else None,
        selected_family if simulation_kind == "Distribution family" else None,
        tuple(simulation_methods),
    )

    generate_simulation = st.button(
        "Generate and evaluate simulation",
        type="primary",
        width="stretch",
        disabled=not simulation_methods,
    )
    if generate_simulation:
        try:
            with st.spinner("Generating the task and evaluating all methods..."):
                rng = np.random.default_rng(int(simulation_seed))
                if simulation_kind == "Gaussian":
                    task = generate_gaussian_task(
                        rng,
                        simulation_n,
                        int(test_size),
                        float(gaussian_mean),
                        float(gaussian_standard_deviation),
                    )
                elif simulation_kind == "Distribution family":
                    task = generate_multifamily_task(
                        rng,
                        simulation_n,
                        int(test_size),
                        selected_family,
                    )
                else:
                    task = generate_gmm32_task(
                        rng,
                        simulation_n,
                        int(test_size),
                    )
                simulation_result = run_simulation(
                    task, simulation_methods
                )
                simulation_result["config_signature"] = simulation_signature
                st.session_state["simulation_result"] = simulation_result
                if simulation_kind == "Distribution family":
                    tracked_distribution = str(
                        task.metadata.get("Realized family", selected_family)
                    )
                else:
                    tracked_distribution = simulation_kind
                track_event(
                    "simulation_generated",
                    workflow="simulation",
                    distribution=tracked_distribution,
                    sample_size=int(simulation_n),
                    methods=simulation_methods,
                )
        except Exception as error:
            st.error(f"Could not run the simulation: {error}")

    saved_simulation_result = st.session_state.get("simulation_result")
    if saved_simulation_result is not None:
        if (
            saved_simulation_result.get("config_signature")
            == simulation_signature
        ):
            render_simulation_result(saved_simulation_result)
        else:
            st.info(
                "The simulation settings changed. Generate the task again to "
                "update the result."
            )
