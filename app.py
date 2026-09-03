"""Entry point for the Amortized KDE Streamlit application."""

from __future__ import annotations

import streamlit as st

from kde_app.config import APP_REVISION, APP_TITLE, GITHUB_URL, PAPER_URL

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")

from kde_app.analytics import track_event
from kde_app.data_page import render_data_page
from kde_app.simulation_page import render_simulation_page

if "_page_view_tracking_attempted" not in st.session_state:
    st.session_state["_page_view_tracking_attempted"] = True
    track_event("page_view", workflow="app")


st.title(APP_TITLE)
st.markdown(
    f"""
[Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic
Score]({PAPER_URL}) · [arXiv:2608.20445]({PAPER_URL})
"""
)
st.write(
    "Compare an amortized KDE bandwidth selector with Silverman, "
    "Sheather–Jones and LSCV on your own sample or on a generated task with "
    "known truth."
)

with st.expander("About the method and working interval"):
    st.markdown(
        """
This web application demonstrates the amortized bandwidth-selection framework
proposed in the paper. The framework learns the mapping from a finite sample
to a KDE bandwidth across a distribution of density-estimation tasks by
optimizing the logarithmic score. Once trained, it predicts a bandwidth
directly, without requiring a new optimization or bandwidth search for each
sample.

The available amortized selectors operate on one-dimensional samples of size
5–256 and use five features—sample size, mean, sample standard deviation,
skewness, and kurtosis—to predict the bandwidth. The predicted bandwidth is
transferred to the selected bounded interval and used to construct a
truncated-and-renormalized Gaussian KDE. For convenience, the web interface
allows the interval to be either specified by the user or generated using a
sample-adaptive rule.

The extended comparison keeps this original workflow and adds three classical
bandwidth selectors—Silverman, Sheather–Jones, and least-squares
cross-validation (LSCV)—together with an ordinary unbounded KDE view. The
appropriate amortized checkpoint is selected automatically from the available
information about the underlying family. The GMM K=32 selector is used when
the family is unknown or outside the ten-family collection.
"""
    )

link_columns = st.columns([1, 1, 4])
link_columns[0].link_button("Read the paper", PAPER_URL, width="stretch")
link_columns[1].link_button("View code on GitHub", GITHUB_URL, width="stretch")

with st.sidebar:
    st.header("KDE workflows")
    app_mode = st.radio(
        "Workflow",
        ("Data", "Simulation"),
        help="Use your own observations, or generate a task with known truth.",
    )
    st.divider()
    st.caption(
        "Samples are processed only during the current app session. The app "
        "does not intentionally save uploaded observations."
    )
    st.caption(APP_REVISION)


if app_mode == "Data":
    render_data_page()
else:
    render_simulation_page()
