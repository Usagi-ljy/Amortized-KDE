"""Input validation, bandwidth selection and shared KDE evaluation helpers."""

from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import logsumexp, ndtr

from .classical_bandwidth_selectors import (
    compute_classical_bandwidths,
    sheather_jones_is_available,
)
from .config import (
    CLASSICAL_METHOD_KEYS,
    FAMILY_FROM_LABEL,
    ISSUE_URL,
    LOG_2,
    MAX_SAMPLE_SIZE,
    MIN_SAMPLE_SIZE,
    MODEL_DIR,
    NEURAL_LABELS,
    SQRT_2PI,
)
from .kde_estimators import KDEResult
from .neural_bandwidth_selectors import NeuralBandwidthResult, NeuralBandwidthSelector

def format_number(value: float) -> str:
    """Format a scalar compactly for the interface."""

    return f"{float(value):.6g}"


def parse_manual_samples(text: str) -> np.ndarray:
    """Parse numbers separated by spaces, commas, semicolons, or new lines."""

    stripped = text.strip()
    if not stripped:
        return np.asarray([], dtype=np.float64)

    values: list[float] = []
    for token in re.split(r"[\s,;]+", stripped):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as error:
            raise ValueError(f"Cannot interpret {token!r} as a number.") from error
    return np.asarray(values, dtype=np.float64)


def read_uploaded_samples(uploaded_file) -> np.ndarray:
    """Read and flatten all numeric cells in a CSV, or parse a TXT file."""

    suffix = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.getvalue()
    if suffix == ".txt":
        try:
            return parse_manual_samples(raw_bytes.decode("utf-8-sig"))
        except UnicodeDecodeError as error:
            raise ValueError("The TXT file must use UTF-8 encoding.") from error

    if suffix == ".csv":
        try:
            frame = pd.read_csv(io.BytesIO(raw_bytes), header=None)
            numeric = frame.apply(pd.to_numeric, errors="coerce")
            flattened = numeric.to_numpy(dtype=np.float64).ravel(order="C")
            values = flattened[np.isfinite(flattened)]
        except Exception as error:
            raise ValueError(f"Could not read the CSV file: {error}") from error
        if values.size == 0:
            raise ValueError("No numeric sample values were found in the CSV file.")
        return values

    raise ValueError("Please upload a .csv or .txt file.")


def validate_sample(samples: np.ndarray) -> np.ndarray:
    """Validate a sample against the selectors' public operating range."""

    sample = np.asarray(samples, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError("The sample must be one-dimensional.")
    if not np.all(np.isfinite(sample)):
        raise ValueError("The sample contains NaN or infinite values.")
    if not (MIN_SAMPLE_SIZE <= sample.size <= MAX_SAMPLE_SIZE):
        raise ValueError(
            f"The published selectors support {MIN_SAMPLE_SIZE} ≤ n ≤ "
            f"{MAX_SAMPLE_SIZE}; received n={sample.size}."
        )
    if float(np.ptp(sample)) <= 0.0:
        raise ValueError("Bandwidth selection requires non-zero sample spread.")
    return sample


def sample_adaptive_interval(samples: np.ndarray) -> tuple[float, float]:
    """Return the uniform-reference, sample-adaptive working interval."""

    sample = np.asarray(samples, dtype=np.float64)
    if sample.ndim != 1 or sample.size < 2:
        raise ValueError("At least two one-dimensional observations are required.")
    observed_range = float(sample.max() - sample.min())
    if observed_range <= 0.0:
        raise ValueError("Cannot form an interval from a zero-width sample range.")
    margin = observed_range / float(sample.size - 1)
    return float(sample.min() - margin), float(sample.max() + margin)


def validate_support(
    support: tuple[float, float],
    samples: np.ndarray,
) -> tuple[float, float]:
    """Validate finite ordered endpoints and containment of the sample."""

    left, right = float(support[0]), float(support[1])
    if not (math.isfinite(left) and math.isfinite(right) and left < right):
        raise ValueError("The interval must have finite endpoints with A < B.")
    tolerance = 1e-12 * max(1.0, abs(left), abs(right))
    if samples.min() < left - tolerance or samples.max() > right + tolerance:
        raise ValueError("Every observation must lie inside the interval [A, B].")
    return left, right


def default_method_selection() -> list[str]:
    """Select every method by default when the exact SJ dependency is present."""

    defaults = ["Amortized selector", "Silverman", "LSCV"]
    if sheather_jones_is_available():
        defaults.insert(2, "Sheather–Jones")
    return defaults


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def distribution_request_url(distribution_name: str = "") -> str:
    """Build a prefilled GitHub issue for a requested simulation family."""

    cleaned_name = distribution_name.strip()
    requested_name = cleaned_name or "[distribution name and parameters]"
    query = urlencode(
        {
            "title": f"Distribution request: {requested_name}",
            "body": (
                "Please add the following distribution to the interactive "
                "simulation workflow.\n\n"
                f"Distribution: {requested_name}\n\n"
                "Suggested parameter range or reference:\n"
                "Additional notes:\n\n"
                "This request does not include uploaded sample data."
            ),
        }
    )
    return f"{ISSUE_URL}?{query}"


def selector_for_known_family(family_label: str) -> str:
    """Choose the matching public checkpoint for a known family label."""

    if family_label == "Gaussian":
        return "gaussian"
    if family_label in FAMILY_FROM_LABEL:
        return "multifamily"
    return "gmm32"


# ---------------------------------------------------------------------------
# Bandwidth selection and KDE evaluation
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_neural_selector(selector_name: str) -> NeuralBandwidthSelector:
    """Load each checkpoint at most once per Streamlit worker."""

    return NeuralBandwidthSelector(
        selector_name,
        checkpoint_directory=MODEL_DIR,
        device="cpu",
        strict_sample_size=True,
    )


def compute_selected_bandwidths(
    samples: np.ndarray,
    selected_methods: list[str],
    *,
    neural_selector_name: str,
    working_support: tuple[float, float],
) -> tuple[dict[str, float], Optional[NeuralBandwidthResult]]:
    """Compute requested bandwidths on one common affine reference scale."""

    if not selected_methods:
        raise ValueError("Select at least one bandwidth method.")

    bandwidths: dict[str, float] = {}
    neural_result: Optional[NeuralBandwidthResult] = None
    if "Amortized selector" in selected_methods:
        selector = load_neural_selector(neural_selector_name)
        neural_result = selector.predict_bandwidth(samples, support=working_support)
        bandwidths[NEURAL_LABELS[neural_selector_name]] = neural_result.bandwidth

    requested_classical = [
        CLASSICAL_METHOD_KEYS[method]
        for method in selected_methods
        if method in CLASSICAL_METHOD_KEYS
    ]
    if requested_classical:
        classical = compute_classical_bandwidths(
            samples,
            methods=requested_classical,
            support=working_support,
            rescale_to_reference=True,
        )
        bandwidths.update(classical)

    return bandwidths, neural_result


def kde_log_density_at_points(
    samples: np.ndarray,
    points: np.ndarray,
    bandwidth: float,
    *,
    mode: str,
    support: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Evaluate Gaussian KDE log density robustly at arbitrary test points."""

    sample = np.asarray(samples, dtype=np.float64)
    test = np.asarray(points, dtype=np.float64)
    bandwidth = float(bandwidth)
    standardised = (test[:, None] - sample[None, :]) / bandwidth
    log_kernel = -0.5 * standardised**2 - math.log(SQRT_2PI * bandwidth)
    log_density = logsumexp(log_kernel, axis=1) - math.log(sample.size)

    if mode == "bounded":
        if support is None:
            raise ValueError("support is required for a bounded log score.")
        left, right = support
        if np.any((test < left) | (test > right)):
            raise ValueError("A test observation lies outside the bounded support.")
        component_mass = ndtr((right - sample) / bandwidth) - ndtr(
            (left - sample) / bandwidth
        )
        normalising_constant = float(np.mean(np.maximum(component_mass, 0.0)))
        if not math.isfinite(normalising_constant) or normalising_constant <= 0.0:
            raise RuntimeError("The bounded KDE normalising constant is invalid.")
        log_density -= math.log(normalising_constant)
    elif mode != "unbounded":
        raise ValueError("mode must be 'bounded' or 'unbounded'.")
    return log_density


def empirical_log_scores(
    samples: np.ndarray,
    test_samples: np.ndarray,
    bandwidths: Mapping[str, float],
    *,
    mode: str,
    support: Optional[tuple[float, float]],
) -> dict[str, float]:
    """Return empirical negative log2 scores; lower values are better."""

    scores: dict[str, float] = {}
    for method, bandwidth in bandwidths.items():
        log_density = kde_log_density_at_points(
            samples,
            test_samples,
            bandwidth,
            mode=mode,
            support=support,
        )
        scores[method] = float(-np.mean(log_density) / LOG_2)
    return scores


def curves_to_frame(
    results: Mapping[str, KDEResult],
    *,
    true_density: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build one download table from curves that share an x-grid."""

    first = next(iter(results.values()))
    data: dict[str, np.ndarray] = {"x": first.x_grid}
    if true_density is not None:
        data["true_density"] = np.asarray(true_density, dtype=np.float64)
    for method, result in results.items():
        column = re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")
        data[f"density_{column}"] = result.density
    return pd.DataFrame(data)
