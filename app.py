from __future__ import annotations

import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import streamlit as st

from inference import AmortizedKDE


DENSITY_GRID_SIZE = 512

APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = APP_DIR / "gmm32_selector.pt"

st.set_page_config(
    page_title="Amortized KDE",
    layout="centered",
)


@st.cache_resource
def load_estimator() -> AmortizedKDE:
    return AmortizedKDE(
        CHECKPOINT_PATH,
        device="cpu",
        strict_sample_size=True,
    )


def parse_manual_samples(text: str) -> np.ndarray:
    text = text.strip()

    if not text:
        return np.array([], dtype=np.float64)

    tokens = re.split(r"[\s,;]+", text)
    values = []

    for token in tokens:
        if not token:
            continue

        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(
                f"Cannot interpret '{token}' as a number."
            ) from exc

    return np.asarray(
        values,
        dtype=np.float64,
    )


def read_uploaded_samples(uploaded_file) -> np.ndarray:
    """
    Read one-dimensional samples from CSV or TXT.

    CSV behavior:
      - one column: supported;
      - one row: supported;
      - multiple rows and columns: supported;
      - every numeric cell is flattened row-by-row into one sample;
      - headers, labels, and empty cells are ignored.

    TXT behavior:
      - numeric values may be separated by commas, spaces,
        semicolons, tabs, or new lines.
    """
    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    raw_bytes = uploaded_file.getvalue()

    if suffix == ".txt":
        return parse_manual_samples(
            raw_bytes.decode("utf-8-sig")
        )

    if suffix == ".csv":
        try:
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                header=None,
            )

            # Convert every cell independently. Headers, labels,
            # and empty cells become NaN and are ignored.
            numeric_df = df.apply(
                pd.to_numeric,
                errors="coerce",
            )

            # Flatten all numeric cells in row-major order:
            # row 1 left-to-right, then row 2, etc.
            flat = numeric_df.to_numpy(
                dtype=np.float64
            ).ravel(
                order="C"
            )

            values = flat[
                np.isfinite(flat)
            ]

            if values.size == 0:
                raise ValueError(
                    "No numeric sample values were found in the CSV file."
                )

            return values

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Could not read the CSV file: {exc}"
            ) from exc

    raise ValueError(
        "Please upload a .csv or .txt file."
    )


def fmt(x: float) -> str:
    return f"{x:.6g}"


def sample_adaptive_working_interval(
    samples: np.ndarray,
) -> tuple[float, float]:
    """
    Automatic working interval with c=1:

        R = x_max - x_min
        A = x_min - R/(N-1)
        B = x_max + R/(N-1)

    This is a uniform-reference, sample-size-adaptive working
    interval. It is not a confidence interval or a general
    estimator of mathematical support.
    """
    x = np.asarray(
        samples,
        dtype=np.float64,
    )

    if x.ndim != 1:
        raise ValueError(
            "samples must be one-dimensional."
        )

    n = int(x.size)

    if n < 2:
        raise ValueError(
            "At least two observations are required."
        )

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    observed_range = x_max - x_min

    if observed_range <= 0.0:
        raise ValueError(
            "Cannot construct an automatic interval "
            "from a zero-width sample range."
        )

    margin = observed_range / float(n - 1)

    return (
        x_min - margin,
        x_max + margin,
    )



def trapezoidal_integral(
    y: np.ndarray,
    x: np.ndarray,
) -> float:
    """
    Version-independent trapezoidal integration.

    This intentionally avoids np.trapz and np.trapezoid so that the
    app works across both older and newer NumPy versions.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")
    if y.size != x.size:
        raise ValueError("x and y must have the same length.")
    if y.size < 2:
        raise ValueError("At least two grid points are required.")

    dx = np.diff(x)

    return float(
        np.sum(
            0.5
            * (y[:-1] + y[1:])
            * dx
        )
    )


st.title(
    "Amortized Bandwidth Learning for Kernel Density Estimation "
    "under Logarithmic Score Loss"
)

st.markdown(
    """
This interactive tool implements the amortized bandwidth selector developed in the accompanying paper.
Given a one-dimensional sample, the method summarizes the sample using five features—sample size,
mean, standard deviation, skewness, and kurtosis—and uses a neural selector trained across Gaussian-mixture
density-estimation tasks to predict the KDE bandwidth directly, without performing a new bandwidth search
for each dataset. The predicted bandwidth is then used to construct a truncated Gaussian kernel density
estimate on the selected finite interval.
"""
)


# ============================================================
# 1. Sample
# ============================================================

st.subheader(
    "1. Input sample"
)

input_method = st.radio(
    "Input method",
    [
        "Paste values",
        "Upload CSV or TXT",
    ],
    horizontal=True,
)

samples = np.array(
    [],
    dtype=np.float64,
)
input_error = None

if input_method == "Paste values":
    sample_text = st.text_area(
        "Sample values",
        height=140,
        placeholder=(
            "-0.72, -0.51, -0.35, -0.20, -0.12, "
            "0.03, 0.15, 0.27, 0.41, 0.58"
        ),
    )

    if sample_text.strip():
        try:
            samples = parse_manual_samples(
                sample_text
            )
        except Exception as exc:
            input_error = str(exc)

else:
    uploaded_file = st.file_uploader(
        "Upload sample file",
        type=[
            "csv",
            "txt",
        ],
        help=(
            "CSV may contain samples in one row, one column, or "
            "multiple rows and columns. All numeric cells are "
            "flattened into one one-dimensional sample."
        ),
    )

    if uploaded_file is not None:
        try:
            samples = read_uploaded_samples(
                uploaded_file
            )
        except Exception as exc:
            input_error = str(exc)


if input_method == "Upload CSV or TXT":
    st.caption(
        "For CSV files, all numeric cells are read as sample values "
        "and flattened row-by-row. Column names, text labels, and "
        "empty cells are ignored."
    )

if input_error:
    st.error(
        input_error
    )

if (
    samples.size > 0
    and
    np.all(np.isfinite(samples))
):
    st.caption(
        f"n = {samples.size}; observed range = "
        f"[{fmt(float(samples.min()))}, "
        f"{fmt(float(samples.max()))}]"
    )


# ============================================================
# 2. Interval
# ============================================================

st.subheader(
    "2. Specify interval"
)

interval_method = st.radio(
    "Choose an interval method",
    [
        "Known finite support [A, B]",
        "Automatic sample-adaptive interval",
    ],
    index=1,
)

support_left = None
support_right = None
support_note = ""

if interval_method == "Known finite support [A, B]":
    st.caption(
        "Use this option when the finite support is known."
    )

    c1, c2 = st.columns(2)

    with c1:
        support_left = st.number_input(
            "Lower bound A",
            value=-1.0,
            format="%.6f",
        )

    with c2:
        support_right = st.number_input(
            "Upper bound B",
            value=1.0,
            format="%.6f",
        )

    support_note = "User-specified known finite support."

else:
    st.caption(
        "The automatic rule constructs a sample-size-adaptive "
        "working interval from the observed minimum, maximum, "
        "and range. It is a uniform-reference rule, not a "
        "confidence interval or a general estimator of the "
        "mathematical support."
    )

    if (
        samples.size > 0
        and
        np.all(np.isfinite(samples))
    ):
        try:
            support_left, support_right = (
                sample_adaptive_working_interval(samples)
            )

            n_current = int(samples.size)
            observed_range = float(
                samples.max() - samples.min()
            )
            margin = (
                observed_range
                /
                float(n_current - 1)
            )

            st.latex(
                r"A=x_{\min}-\frac{R}{N-1},"
                r"\qquad "
                r"B=x_{\max}+\frac{R}{N-1}"
            )

            st.info(
                "Automatic working interval: "
                f"[{fmt(support_left)}, "
                f"{fmt(support_right)}]"
            )

            st.caption(
                f"Observed range R = {fmt(observed_range)}; "
                f"extension on each side = {fmt(margin)}."
            )

            support_note = (
                "Automatic sample-adaptive working interval: "
                "A = x_min - R/(N-1), "
                "B = x_max + R/(N-1)."
            )

        except Exception as exc:
            st.warning(
                f"Automatic interval cannot be constructed: {exc}"
            )


# ============================================================
# 3. Estimate
# ============================================================

st.subheader(
    "3. Estimate density"
)

if st.button(
    "Estimate Density",
    type="primary",
    use_container_width=True,
):
    if input_error:
        st.error(
            "Please correct the sample input first."
        )
        st.stop()

    if samples.size == 0:
        st.error(
            "Please enter or upload a sample."
        )
        st.stop()

    if not np.all(np.isfinite(samples)):
        st.error(
            "The sample contains NaN or infinite values."
        )
        st.stop()

    if not (5 <= samples.size <= 256):
        st.error(
            "The deployed selector supports 5 ≤ n ≤ 256."
        )
        st.stop()

    if support_left is None or support_right is None:
        st.error(
            "The interval could not be determined."
        )
        st.stop()

    if not float(support_left) < float(support_right):
        st.error(
            "The interval must satisfy A < B."
        )
        st.stop()

    if (
        float(samples.min()) < float(support_left)
        or
        float(samples.max()) > float(support_right)
    ):
        st.error(
            "At least one observation lies outside the selected interval."
        )
        st.stop()

    if not CHECKPOINT_PATH.is_file():
        st.error(
            "Checkpoint not found. Put this app file, its inference "
            "file, and the .pt checkpoint in the same folder."
        )
        st.stop()

    try:
        estimator = load_estimator()

        result = estimator.predict(
            samples,
            support=(
                float(support_left),
                float(support_right),
            ),
            grid_size=DENSITY_GRID_SIZE,
        )

    except Exception as exc:
        st.error(
            f"Inference failed: {exc}"
        )
        st.stop()

    c1, c2 = st.columns(
        2
    )

    with c1:
        st.metric(
            "Sample size",
            result.n,
        )

    with c2:
        st.metric(
            "Predicted bandwidth",
            f"{result.bandwidth:.6g}",
        )

    st.caption(
        "Effective interval: "
        f"[{result.support[0]:.6g}, "
        f"{result.support[1]:.6g}]"
    )

    st.subheader(
        "Estimated KDE"
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.1)
    )

    ax.plot(
        result.x_grid,
        result.density,
        linewidth=2.0,
    )

    ax.set_xlim(
        result.support
    )
    ax.set_ylim(
        bottom=0.0
    )

    ax.set_xlabel(
        "x"
    )
    ax.set_ylabel(
        "Density"
    )

    # Exactly 5 x-axis ticks; no user plotting controls.
    ax.set_xticks(
        np.linspace(
            result.support[0],
            result.support[1],
            5,
        )
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(
            nbins=5
        )
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )
    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.margins(
        x=0.0
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    density_df = pd.DataFrame(
        {
            "x": result.x_grid,
            "density": result.density,
        }
    )

    st.download_button(
        "Download density as CSV",
        data=density_df.to_csv(
            index=False
        ).encode(
            "utf-8"
        ),
        file_name="amortized_kde_density.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander(
        "Technical details"
    ):
        feature_df = pd.DataFrame(
            {
                "feature": [
                    "n",
                    "mean",
                    "sample std",
                    "skewness",
                    "kurtosis",
                ],
                "value": result.features,
            }
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True,
        )

        # Version-independent trapezoidal integration.
        integral = trapezoidal_integral(
            result.density,
            result.x_grid,
        )

        st.write(
            "Numerical density integral over interval: "
            f"**{integral:.10f}**"
        )
        st.write(
            f"Interval rule: **{support_note}**"
        )
        st.write(
            "Model: **GMM K=32 EMA checkpoint**"
        )
