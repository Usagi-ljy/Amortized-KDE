"""Streamlit Data workflow."""

from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st

from .analytics import track_event
from .classical_bandwidth_selectors import sheather_jones_is_available
from .config import (
    DEFAULT_GRID_SIZE,
    FAMILY_LABELS,
    METHOD_OPTIONS,
    NEURAL_LABELS,
)
from .core import (
    compute_selected_bandwidths,
    default_method_selection,
    distribution_request_url,
    format_number,
    parse_manual_samples,
    read_uploaded_samples,
    sample_adaptive_interval,
    selector_for_known_family,
    validate_sample,
    validate_support,
)
from .kde_estimators import estimate_multiple_kdes
from .ui_components import render_user_result


def render_data_page() -> None:
    st.header("Compare methods on your data")
    st.write(
        "Paste values or upload a CSV/TXT file. If both are supplied, the "
        "uploaded file is used. Numeric CSV cells are flattened row by row."
    )
    with st.form("sample_confirmation_form"):
        input_left, input_right = st.columns(2)
        with input_left:
            manual_text = st.text_area(
                "Paste sample values",
                height=155,
                placeholder="0.12, -0.35, 0.48, 0.22, ...",
            )
        with input_right:
            uploaded_file = st.file_uploader(
                "Upload CSV or TXT",
                type=("csv", "txt"),
            )
            st.caption(
                "Headers and nonnumeric text are ignored. Numeric metadata "
                "will be treated as sample observations."
            )
        confirm_sample = st.form_submit_button(
            "Confirm sample",
            type="primary",
            width="stretch",
        )

    if confirm_sample:
        try:
            if uploaded_file is not None:
                parsed_sample = read_uploaded_samples(uploaded_file)
            else:
                parsed_sample = parse_manual_samples(manual_text)
            if not parsed_sample.size:
                raise ValueError("Paste sample values or upload a CSV/TXT file.")
            parsed_sample = validate_sample(parsed_sample)
            st.session_state["confirmed_sample"] = parsed_sample.copy()
            st.session_state["confirmed_sample_revision"] = int(
                st.session_state.get("confirmed_sample_revision", 0)
            ) + 1
            st.session_state["user_left_endpoint"] = float(parsed_sample.min())
            st.session_state["user_right_endpoint"] = float(parsed_sample.max())
            st.session_state.pop("user_result", None)
        except ValueError as error:
            st.error(str(error))

    confirmed_sample = st.session_state.get("confirmed_sample")
    if confirmed_sample is not None:
        current_sample = np.asarray(confirmed_sample, dtype=np.float64)
        if "user_left_endpoint" not in st.session_state:
            st.session_state["user_left_endpoint"] = float(current_sample.min())
        if "user_right_endpoint" not in st.session_state:
            st.session_state["user_right_endpoint"] = float(current_sample.max())
        st.success(
            f"Confirmed {current_sample.size} observations from "
            f"{format_number(current_sample.min())} to "
            f"{format_number(current_sample.max())}."
        )

        st.subheader("Analysis settings")
        distribution_status = st.radio(
            "Is the underlying distribution family known?",
            ("Unknown", "Known"),
            horizontal=True,
            key="user_distribution_status",
        )

        selected_family_label = "Unknown"
        neural_selector_name = "gmm32"
        if distribution_status == "Known":
            known_family_options = (
                *FAMILY_LABELS.values(),
                "Other / request another distribution",
            )
            selected_family_label = st.selectbox(
                "Known distribution family",
                known_family_options,
                key="user_known_family",
            )
            neural_selector_name = selector_for_known_family(
                selected_family_label
            )

            if selected_family_label == "Other / request another distribution":
                requested_distribution = st.text_input(
                    "Distribution name and parameters",
                    placeholder="Weibull(shape=2)",
                    key="user_requested_distribution",
                )
                st.caption(
                    "Your current sample can be analysed immediately with the "
                    "general-purpose GMM K=32 selector. Submitting a request "
                    "only asks for this family to be added to Simulation with "
                    "reproducible sampling, a true density and log-score "
                    "evaluation. No sample values are included in the request."
                )
                st.link_button(
                    "Request this distribution",
                    distribution_request_url(requested_distribution),
                )

        st.caption(
            "Amortized checkpoint used when that method is selected: "
            f"{NEURAL_LABELS[neural_selector_name]}."
        )

        option_left, option_right = st.columns(2)
        with option_left:
            selected_methods = st.multiselect(
                "Bandwidth methods",
                METHOD_OPTIONS,
                default=default_method_selection(),
                key="user_methods",
            )
            if (
                "Sheather–Jones" in selected_methods
                and not sheather_jones_is_available()
            ):
                st.warning(
                    "Exact Sheather–Jones requires Rscript. Add `r-base-core` "
                    "to `packages.txt` for Streamlit Community Cloud."
                )
        with option_right:
            interval_choice = st.radio(
                "Working interval",
                (
                    "Automatic sample-adaptive interval",
                    "Known finite support [A, B]",
                ),
                help=(
                    "This interval is used for reference rescaling and for the "
                    "bounded KDE."
                ),
            )

        support_error: Optional[str] = None
        support: Optional[tuple[float, float]] = None
        if interval_choice == "Known finite support [A, B]":
            bound_left, bound_right = st.columns(2)
            with bound_left:
                left_endpoint = st.number_input(
                    "A (left endpoint)",
                    key="user_left_endpoint",
                    format="%.8g",
                )
            with bound_right:
                right_endpoint = st.number_input(
                    "B (right endpoint)",
                    key="user_right_endpoint",
                    format="%.8g",
                )
            try:
                support = validate_support(
                    (left_endpoint, right_endpoint), current_sample
                )
            except ValueError as error:
                support_error = str(error)
        else:
            try:
                support = sample_adaptive_interval(current_sample)
                st.caption(
                    f"Automatic interval: [{format_number(support[0])}, "
                    f"{format_number(support[1])}]. It follows "
                    "A = x_min - R/(n-1) and B = x_max + R/(n-1). This is a "
                    "working interval, not a confidence interval or a general "
                    "support estimator."
                )
            except ValueError as error:
                support_error = str(error)

        if support_error:
            st.error(support_error)

        support_signature = (
            None
            if support is None
            else (float(support[0]), float(support[1]))
        )
        current_user_signature = (
            int(st.session_state.get("confirmed_sample_revision", 0)),
            distribution_status,
            selected_family_label,
            neural_selector_name,
            tuple(selected_methods),
            support_signature,
        )

        generate_user_result = st.button(
            "Generate KDE comparison",
            type="primary",
            width="stretch",
            disabled=(
                support is None
                or bool(support_error)
                or not selected_methods
            ),
        )
        if generate_user_result:
            try:
                with st.spinner(
                    "Selecting bandwidths and evaluating KDE curves..."
                ):
                    bandwidths, neural_result = compute_selected_bandwidths(
                        current_sample,
                        selected_methods,
                        neural_selector_name=neural_selector_name,
                        working_support=support,
                    )
                    bounded_results = estimate_multiple_kdes(
                        current_sample,
                        bandwidths,
                        mode="bounded",
                        support=support,
                        grid_size=DEFAULT_GRID_SIZE,
                    )
                    unbounded_results = estimate_multiple_kdes(
                        current_sample,
                        bandwidths,
                        mode="unbounded",
                        grid_size=DEFAULT_GRID_SIZE,
                    )
                st.session_state["user_result"] = {
                    "config_signature": current_user_signature,
                    "sample": current_sample.copy(),
                    "support": support,
                    "bandwidths": bandwidths,
                    "neural_result": neural_result,
                    "bounded_results": bounded_results,
                    "unbounded_results": unbounded_results,
                }
                tracked_distribution = (
                    "unknown"
                    if distribution_status == "Unknown"
                    else (
                        "other"
                        if selected_family_label
                        == "Other / request another distribution"
                        else selected_family_label
                    )
                )
                track_event(
                    "data_kde_generated",
                    workflow="data",
                    distribution=tracked_distribution,
                    sample_size=int(current_sample.size),
                    methods=selected_methods,
                )
            except Exception as error:
                st.error(f"Could not generate the comparison: {error}")

        saved_user_result = st.session_state.get("user_result")
        if saved_user_result is not None:
            if saved_user_result.get("config_signature") == current_user_signature:
                render_user_result(saved_user_result)
            else:
                st.info(
                    "The analysis settings changed. Generate the comparison "
                    "again to update the result."
                )
