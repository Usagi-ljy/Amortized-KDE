"""Interactive comparison app for amortized and classical KDE selectors.

The application has two workflows:

1. ``Data`` compares selected bandwidth methods on an uploaded or pasted
   one-dimensional sample.  It displays both bounded and ordinary unbounded
   Gaussian KDEs on shared grids.
2. ``Simulation`` generates a fresh task from the Gaussian, multi-family, or
   bounded GMM K=32 setup used by the project.  It overlays the true density,
   estimates the KDEs, and evaluates empirical negative log2 score on an
   independent test sample.

Every classical selector follows the same explicit affine comparison route as
the neural selectors: map the working support to ``[-1, 1]``, select a
bandwidth there, and rescale it to the original data units.
"""

from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional
from urllib.parse import urlencode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize, stats
from scipy.special import logsumexp, ndtr, ndtri

from classical_bandwidth_selectors import (
    compute_classical_bandwidths,
    sheather_jones_is_available,
)
from kde_estimators import KDEResult, estimate_multiple_kdes
from kde_plotting import plot_kde_comparison
from neural_bandwidth_selectors import (
    NeuralBandwidthResult,
    NeuralBandwidthSelector,
)


# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------

APP_TITLE = (
    "Amortized Bandwidth Learning for Kernel Density Estimation "
    "under Logarithmic Score"
)
APP_REVISION = "2026-09-01 · clean-workflow-v6"
APP_DIR = Path(__file__).resolve().parent
FIGURE_DIR = APP_DIR / "figures"

PAPER_URL = "https://arxiv.org/abs/2608.20445"
GITHUB_URL = "https://github.com/Usagi-ljy/Amortized-KDE"
ISSUE_URL = f"{GITHUB_URL}/issues/new"

MIN_SAMPLE_SIZE = 5
MAX_SAMPLE_SIZE = 256
DEFAULT_GRID_SIZE = 512
DEFAULT_TEST_SIZE = 2048
REFERENCE_SUPPORT = (-1.0, 1.0)
LOG_2 = math.log(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)

METHOD_OPTIONS = (
    "Amortized selector",
    "Silverman",
    "Sheather–Jones",
    "LSCV",
)

CLASSICAL_METHOD_KEYS = {
    "Silverman": "silverman",
    "Sheather–Jones": "sheather_jones",
    "LSCV": "lscv",
}

NEURAL_LABELS = {
    "gaussian": "Amortized (Gaussian)",
    "multifamily": "Amortized (Multi-family)",
    "gmm32": "Amortized (GMM K=32)",
}

FAMILY_LABELS = {
    "gaussian": "Gaussian",
    "laplace": "Laplace",
    "student_t": "Student-t",
    "gamma": "Gamma",
    "beta": "Beta",
    "logistic": "Logistic",
    "lognormal": "Lognormal",
    "bimodal": "Bimodal Gaussian mixture",
    "trimodal": "Trimodal Gaussian mixture",
    "spike_slab": "Spike-and-slab",
}

MULTIFAMILY_OPTIONS = (
    "Multi-family (random family)",
    *FAMILY_LABELS.values(),
)
FAMILY_FROM_LABEL = {label: key for key, label in FAMILY_LABELS.items()}

# Web-display copies of the paper figures.  Keep the corresponding EPS files
# under the same stems for archival/download purposes; Streamlit displays PNG.
GAUSSIAN_BENCHMARK_FIGURE = "gaussian_benchmark.png"
MULTIFAMILY_BENCHMARK_FIGURES = {
    "Multi-family": "multifamily_benchmark.png",
    "Gaussian": "family_gaussian_benchmark.png",
    "Laplace": "family_laplace_benchmark.png",
    "Student-t": "family_student_t_benchmark.png",
    "Gamma": "family_gamma_benchmark.png",
    "Beta": "family_beta_benchmark.png",
    "Logistic": "family_logistic_benchmark.png",
    "Lognormal": "family_lognormal_benchmark.png",
    "Bimodal": "family_bimodal_benchmark.png",
    "Trimodal": "family_trimodal_benchmark.png",
    "Spike-and-slab": "family_spike_slab_benchmark.png",
}
GMM32_BENCHMARK_FIGURE = "gmm32_benchmark.png"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# General input and formatting helpers
# ---------------------------------------------------------------------------


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
        checkpoint_directory=APP_DIR,
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


# ---------------------------------------------------------------------------
# Simulation distributions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationTask:
    observed: np.ndarray
    test: np.ndarray
    kde_mode: str
    working_support: tuple[float, float]
    selector_name: str
    true_density: Callable[[np.ndarray], np.ndarray]
    title: str
    metadata: Mapping[str, object]


def log_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: Optional[int] = None,
) -> np.ndarray | float:
    values = np.exp(rng.uniform(math.log(low), math.log(high), size=size))
    return float(values) if size is None else values


def generate_gaussian_task(
    rng: np.random.Generator,
    n: int,
    test_size: int,
    mean: float,
    standard_deviation: float,
) -> SimulationTask:
    """Generate an unbounded Gaussian benchmark task."""

    observed = rng.normal(mean, standard_deviation, size=n)
    test = rng.normal(mean, standard_deviation, size=test_size)
    working_support = sample_adaptive_interval(observed)

    def density(x: np.ndarray) -> np.ndarray:
        return stats.norm.pdf(x, loc=mean, scale=standard_deviation)

    return SimulationTask(
        observed=observed,
        test=test,
        kde_mode="unbounded",
        working_support=working_support,
        selector_name="gaussian",
        true_density=density,
        title="Gaussian simulation",
        metadata={
            "Distribution": "Gaussian",
            "Mean": mean,
            "Standard deviation": standard_deviation,
            "KDE support": "Unbounded",
        },
    )


def sample_family_shape(
    family: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Draw one shape configuration from the multi-family training setup."""

    if family in {"gaussian", "laplace", "logistic"}:
        return {}
    if family == "student_t":
        return {"nu": log_uniform(rng, 3.0, 30.0)}
    if family == "gamma":
        return {
            "shape": log_uniform(rng, 0.5, 10.0),
            "sign": 1.0 if rng.random() < 0.5 else -1.0,
        }
    if family == "beta":
        regime = int(rng.integers(0, 4))
        if regime == 0:
            a, b = rng.uniform(0.4, 0.9, size=2)
            regime_name = "u_shaped"
        elif regime == 1:
            small = float(rng.uniform(0.4, 0.9))
            large = float(rng.uniform(1.2, 5.0))
            a, b = (small, large) if rng.random() < 0.5 else (large, small)
            regime_name = "one_sided"
        elif regime == 2:
            a, b = rng.uniform(1.2, 5.0, size=2)
            regime_name = "interior"
        else:
            a, b = rng.uniform(0.8, 1.2, size=2)
            regime_name = "near_uniform"
        return {"a": float(a), "b": float(b), "regime": regime_name}
    if family == "lognormal":
        return {
            "tau": float(rng.uniform(0.25, 1.0)),
            "sign": 1.0 if rng.random() < 0.5 else -1.0,
        }
    if family == "bimodal":
        return {
            "weight": float(rng.uniform(0.2, 0.8)),
            "rho_left": log_uniform(rng, 0.15, 0.45),
            "rho_right": log_uniform(rng, 0.15, 0.45),
        }
    if family == "trimodal":
        weights = rng.dirichlet(np.full(3, 2.0))
        while float(weights.min()) < 0.1:
            weights = rng.dirichlet(np.full(3, 2.0))
        return {
            "distance_left": float(rng.uniform(0.8, 1.2)),
            "distance_right": float(rng.uniform(0.8, 1.2)),
            "rhos": np.asarray(log_uniform(rng, 0.1, 0.3, size=3)),
            "weights": weights,
        }
    if family == "spike_slab":
        return {
            "spike_weight": float(rng.uniform(0.2, 0.8)),
            "spike_scale": log_uniform(rng, 0.05, 0.2),
        }
    raise ValueError(f"Unknown family: {family!r}.")


def base_family_cdf(
    family: str,
    z: np.ndarray | float,
    shape: Mapping[str, object],
) -> np.ndarray:
    """CDF of a centred, unit-reference member of a distribution family."""

    values = np.asarray(z, dtype=np.float64)
    if family == "gaussian":
        return stats.norm.cdf(values)
    if family == "laplace":
        return stats.laplace.cdf(values)
    if family == "student_t":
        return stats.t.cdf(values, df=float(shape["nu"]))
    if family == "gamma":
        k = float(shape["shape"])
        sign = float(shape["sign"])
        if sign > 0.0:
            return stats.gamma.cdf(values + k, a=k)
        return stats.gamma.sf(k - values, a=k)
    if family == "beta":
        a, b = float(shape["a"]), float(shape["b"])
        centre = a / (a + b)
        return stats.beta.cdf(values + centre, a=a, b=b)
    if family == "logistic":
        return stats.logistic.cdf(values)
    if family == "lognormal":
        tau = float(shape["tau"])
        sign = float(shape["sign"])
        centre = math.exp(0.5 * tau**2)
        if sign > 0.0:
            return stats.lognorm.cdf(values + centre, s=tau)
        return stats.lognorm.sf(centre - values, s=tau)
    if family == "bimodal":
        weight = float(shape["weight"])
        centre = weight * -1.0 + (1.0 - weight) * 1.0
        left_mean, right_mean = -1.0 - centre, 1.0 - centre
        return weight * stats.norm.cdf(
            values, loc=left_mean, scale=float(shape["rho_left"])
        ) + (1.0 - weight) * stats.norm.cdf(
            values, loc=right_mean, scale=float(shape["rho_right"])
        )
    if family == "trimodal":
        weights = np.asarray(shape["weights"], dtype=np.float64)
        raw_means = np.asarray(
            [-float(shape["distance_left"]), 0.0, float(shape["distance_right"])],
            dtype=np.float64,
        )
        means = raw_means - float(np.dot(weights, raw_means))
        rhos = np.asarray(shape["rhos"], dtype=np.float64)
        result = np.zeros_like(values, dtype=np.float64)
        for weight, mean, rho in zip(weights, means, rhos):
            result += weight * stats.norm.cdf(values, loc=mean, scale=rho)
        return result
    if family == "spike_slab":
        weight = float(shape["spike_weight"])
        return weight * stats.norm.cdf(
            values, scale=float(shape["spike_scale"])
        ) + (1.0 - weight) * stats.norm.cdf(values)
    raise ValueError(f"Unknown family: {family!r}.")


def base_family_pdf(
    family: str,
    z: np.ndarray,
    shape: Mapping[str, object],
) -> np.ndarray:
    """PDF matching :func:`base_family_cdf`."""

    values = np.asarray(z, dtype=np.float64)
    if family == "gaussian":
        return stats.norm.pdf(values)
    if family == "laplace":
        return stats.laplace.pdf(values)
    if family == "student_t":
        return stats.t.pdf(values, df=float(shape["nu"]))
    if family == "gamma":
        k = float(shape["shape"])
        sign = float(shape["sign"])
        argument = values + k if sign > 0.0 else k - values
        return stats.gamma.pdf(argument, a=k)
    if family == "beta":
        a, b = float(shape["a"]), float(shape["b"])
        centre = a / (a + b)
        return stats.beta.pdf(values + centre, a=a, b=b)
    if family == "logistic":
        return stats.logistic.pdf(values)
    if family == "lognormal":
        tau = float(shape["tau"])
        sign = float(shape["sign"])
        centre = math.exp(0.5 * tau**2)
        argument = values + centre if sign > 0.0 else centre - values
        return stats.lognorm.pdf(argument, s=tau)
    if family == "bimodal":
        weight = float(shape["weight"])
        centre = weight * -1.0 + (1.0 - weight) * 1.0
        left_mean, right_mean = -1.0 - centre, 1.0 - centre
        return weight * stats.norm.pdf(
            values, loc=left_mean, scale=float(shape["rho_left"])
        ) + (1.0 - weight) * stats.norm.pdf(
            values, loc=right_mean, scale=float(shape["rho_right"])
        )
    if family == "trimodal":
        weights = np.asarray(shape["weights"], dtype=np.float64)
        raw_means = np.asarray(
            [-float(shape["distance_left"]), 0.0, float(shape["distance_right"])],
            dtype=np.float64,
        )
        means = raw_means - float(np.dot(weights, raw_means))
        rhos = np.asarray(shape["rhos"], dtype=np.float64)
        result = np.zeros_like(values, dtype=np.float64)
        for weight, mean, rho in zip(weights, means, rhos):
            result += weight * stats.norm.pdf(values, loc=mean, scale=rho)
        return result
    if family == "spike_slab":
        weight = float(shape["spike_weight"])
        return weight * stats.norm.pdf(
            values, scale=float(shape["spike_scale"])
        ) + (1.0 - weight) * stats.norm.pdf(values)
    raise ValueError(f"Unknown family: {family!r}.")


def sample_base_family(
    family: str,
    shape: Mapping[str, object],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw from the centred base distribution used above."""

    if family == "gaussian":
        return rng.normal(size=size)
    if family == "laplace":
        return rng.laplace(size=size)
    if family == "student_t":
        return rng.standard_t(float(shape["nu"]), size=size)
    if family == "gamma":
        k = float(shape["shape"])
        return float(shape["sign"]) * (rng.gamma(k, size=size) - k)
    if family == "beta":
        a, b = float(shape["a"]), float(shape["b"])
        return rng.beta(a, b, size=size) - a / (a + b)
    if family == "logistic":
        return rng.logistic(size=size)
    if family == "lognormal":
        tau = float(shape["tau"])
        centre = math.exp(0.5 * tau**2)
        return float(shape["sign"]) * (rng.lognormal(0.0, tau, size=size) - centre)
    if family == "bimodal":
        weight = float(shape["weight"])
        component = rng.random(size) >= weight
        raw = np.empty(size, dtype=np.float64)
        raw[~component] = rng.normal(
            -1.0, float(shape["rho_left"]), size=int((~component).sum())
        )
        raw[component] = rng.normal(
            1.0, float(shape["rho_right"]), size=int(component.sum())
        )
        centre = weight * -1.0 + (1.0 - weight) * 1.0
        return raw - centre
    if family == "trimodal":
        weights = np.asarray(shape["weights"], dtype=np.float64)
        raw_means = np.asarray(
            [-float(shape["distance_left"]), 0.0, float(shape["distance_right"])],
            dtype=np.float64,
        )
        means = raw_means - float(np.dot(weights, raw_means))
        rhos = np.asarray(shape["rhos"], dtype=np.float64)
        component = rng.choice(3, size=size, p=weights)
        return rng.normal(means[component], rhos[component])
    if family == "spike_slab":
        weight = float(shape["spike_weight"])
        scale = np.where(
            rng.random(size) < weight,
            float(shape["spike_scale"]),
            1.0,
        )
        return rng.normal(size=size) * scale
    raise ValueError(f"Unknown family: {family!r}.")


def family_interval_mass(
    family: str,
    shape: Mapping[str, object],
    location: float,
    scale: float,
    support: tuple[float, float] = REFERENCE_SUPPORT,
) -> float:
    left, right = support
    upper = (right - location) / scale
    lower = (left - location) / scale
    return float(base_family_cdf(family, upper, shape) - base_family_cdf(family, lower, shape))


def solve_family_scale(
    family: str,
    shape: Mapping[str, object],
    location: float,
    target_mass: float,
) -> float:
    """Choose scale so the raw distribution has the requested mass in [-1,1]."""

    def objective(log_scale: float) -> float:
        return family_interval_mass(
            family,
            shape,
            location,
            math.exp(log_scale),
        ) - target_mass

    lower, upper = -20.0, 20.0
    while objective(lower) < 0.0:
        lower -= 4.0
    while objective(upper) > 0.0:
        upper += 4.0
    return float(math.exp(optimize.brentq(objective, lower, upper)))


def sample_conditioned_family(
    family: str,
    shape: Mapping[str, object],
    location: float,
    scale: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw exactly from the family conditional on the interval [-1,1]."""

    accepted: list[np.ndarray] = []
    total = 0
    while total < size:
        needed = size - total
        batch_size = max(64, int(math.ceil(needed / 0.85)))
        proposed = location + scale * sample_base_family(
            family, shape, batch_size, rng
        )
        kept = proposed[(proposed >= -1.0) & (proposed <= 1.0)]
        if kept.size:
            accepted.append(kept[:needed])
            total += min(needed, int(kept.size))
    return np.concatenate(accepted)[:size]


def serialisable_shape(shape: Mapping[str, object]) -> str:
    """Return compact JSON for NumPy-containing distribution parameters."""

    clean: dict[str, object] = {}
    for key, value in shape.items():
        if isinstance(value, np.ndarray):
            clean[key] = [float(item) for item in value]
        elif isinstance(value, (np.floating, np.integer)):
            clean[key] = float(value)
        else:
            clean[key] = value
    return json.dumps(clean, ensure_ascii=False, sort_keys=True)


def generate_multifamily_task(
    rng: np.random.Generator,
    n: int,
    test_size: int,
    selected_family_label: str,
) -> SimulationTask:
    """Generate one task from the paper's ten-family bounded setup."""

    if selected_family_label == "Multi-family (random family)":
        family = str(rng.choice(list(FAMILY_LABELS)))
    else:
        family = FAMILY_FROM_LABEL[selected_family_label]

    shape = sample_family_shape(family, rng)
    location = float(rng.uniform(-0.5, 0.5))
    target_mass = float(rng.uniform(0.9, 1.0))
    scale = solve_family_scale(family, shape, location, target_mass)
    observed = sample_conditioned_family(family, shape, location, scale, n, rng)
    test = sample_conditioned_family(family, shape, location, scale, test_size, rng)
    actual_mass = family_interval_mass(family, shape, location, scale)

    def density(x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        raw = base_family_pdf(family, (values - location) / scale, shape) / scale
        return np.where(
            (values >= -1.0) & (values <= 1.0),
            raw / actual_mass,
            0.0,
        )

    return SimulationTask(
        observed=observed,
        test=test,
        kde_mode="bounded",
        working_support=REFERENCE_SUPPORT,
        selector_name="multifamily",
        true_density=density,
        title=f"Multi-family simulation: {FAMILY_LABELS[family]}",
        metadata={
            "Requested option": selected_family_label,
            "Realized family": FAMILY_LABELS[family],
            "Location": location,
            "Scale": scale,
            "Raw mass in [-1, 1]": actual_mass,
            "Shape parameters": serialisable_shape(shape),
            "KDE support": "[-1, 1]",
        },
    )


def sample_gmm_component_scales(
    rng: np.random.Generator,
    component_count: int,
) -> tuple[np.ndarray, float, float]:
    """Sample the heterogeneous component scales used by the GMM K=32 task."""

    global_scale = float(log_uniform(rng, 0.025, 0.22))
    log_scale_sd = float(rng.uniform(0.10, 1.00))
    lower, upper = 0.015, 0.45
    lower_z = (math.log(lower) - math.log(global_scale)) / log_scale_sd
    upper_z = (math.log(upper) - math.log(global_scale)) / log_scale_sd
    cdf_low, cdf_high = ndtr(lower_z), ndtr(upper_z)
    uniforms = rng.uniform(cdf_low, cdf_high, size=component_count)
    scales = np.exp(math.log(global_scale) + log_scale_sd * ndtri(uniforms))
    return scales, global_scale, log_scale_sd


def sample_bounded_gmm(
    rng: np.random.Generator,
    size: int,
    means: np.ndarray,
    scales: np.ndarray,
    bounded_weights: np.ndarray,
) -> np.ndarray:
    """Draw exactly from a Gaussian mixture conditional on [-1,1]."""

    component = rng.choice(means.size, size=size, p=bounded_weights)
    selected_mean = means[component]
    selected_scale = scales[component]
    cdf_left = ndtr((-1.0 - selected_mean) / selected_scale)
    cdf_right = ndtr((1.0 - selected_mean) / selected_scale)
    if np.any(cdf_right <= cdf_left):
        raise RuntimeError("Degenerate truncated-normal sampling interval.")
    uniforms = cdf_left + rng.uniform(size=size) * (cdf_right - cdf_left)
    epsilon = np.finfo(np.float64).eps
    uniforms = np.clip(
        uniforms,
        epsilon,
        1.0 - epsilon,
    )
    samples = selected_mean + selected_scale * ndtri(uniforms)
    if np.any(samples < -1.0 - 1e-12) or np.any(samples > 1.0 + 1e-12):
        raise RuntimeError("The bounded sampler generated an out-of-range value.")
    return samples.astype(np.float64, copy=False)


def generate_gmm32_task(
    rng: np.random.Generator,
    n: int,
    test_size: int,
) -> SimulationTask:
    """Generate one bounded GMM K=32 task from the paper's setup."""

    component_count = 32
    alpha = float(log_uniform(rng, 0.5, 4.0))
    raw_weights = rng.dirichlet(np.full(component_count, alpha / component_count))
    means = rng.uniform(-1.0, 1.0, size=component_count)
    scales, global_scale, log_scale_sd = sample_gmm_component_scales(
        rng, component_count
    )
    component_mass = ndtr((1.0 - means) / scales) - ndtr((-1.0 - means) / scales)
    raw_mass = float(np.dot(raw_weights, component_mass))
    bounded_weights = raw_weights * component_mass / raw_mass

    observed = sample_bounded_gmm(
        rng, n, means, scales, bounded_weights
    )
    test = sample_bounded_gmm(
        rng, test_size, means, scales, bounded_weights
    )

    def density(x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        standardised = (values[:, None] - means[None, :]) / scales[None, :]
        raw = np.sum(
            raw_weights[None, :]
            * np.exp(-0.5 * standardised**2)
            / (SQRT_2PI * scales[None, :]),
            axis=1,
        )
        return np.where(
            (values >= -1.0) & (values <= 1.0),
            raw / raw_mass,
            0.0,
        )

    return SimulationTask(
        observed=observed,
        test=test,
        kde_mode="bounded",
        working_support=REFERENCE_SUPPORT,
        selector_name="gmm32",
        true_density=density,
        title="Bounded GMM simulation (K=32)",
        metadata={
            "Components": component_count,
            "Dirichlet alpha": alpha,
            "Global component scale": global_scale,
            "Log-scale standard deviation": log_scale_sd,
            "Raw GMM mass in [-1, 1]": raw_mass,
            "KDE support": "[-1, 1]",
        },
    )


def make_finite_density_curve(
    density_function: Callable[[np.ndarray], np.ndarray],
    x_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate a true density and guard only the plotted curve against infinities."""

    density = np.asarray(density_function(x_grid), dtype=np.float64)
    if density.shape != x_grid.shape:
        raise RuntimeError("The true density returned an unexpected shape.")
    density = np.maximum(density, 0.0)
    finite = np.isfinite(density)
    if not np.any(finite):
        raise RuntimeError("The true density is not finite on the plotting grid.")
    if not np.all(finite):
        cap = max(float(np.quantile(density[finite], 0.995)) * 1.2, 1.0)
        density = np.nan_to_num(density, nan=0.0, posinf=cap, neginf=0.0)
    return density


def run_simulation(
    task: SimulationTask,
    selected_methods: list[str],
) -> dict[str, object]:
    """Select bandwidths, draw curves, and score one generated task."""

    bandwidths, neural_result = compute_selected_bandwidths(
        task.observed,
        selected_methods,
        neural_selector_name=task.selector_name,
        working_support=task.working_support,
    )
    plotting_support = task.working_support if task.kde_mode == "bounded" else None
    curves = estimate_multiple_kdes(
        task.observed,
        bandwidths,
        mode=task.kde_mode,
        support=plotting_support,
        grid_size=DEFAULT_GRID_SIZE,
    )
    grid = next(iter(curves.values())).x_grid
    true_density = make_finite_density_curve(task.true_density, grid)
    scores = empirical_log_scores(
        task.observed,
        task.test,
        bandwidths,
        mode=task.kde_mode,
        support=plotting_support,
    )
    return {
        "task": task,
        "bandwidths": bandwidths,
        "neural_result": neural_result,
        "curves": curves,
        "true_density": true_density,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Result tables and rendering
# ---------------------------------------------------------------------------


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
            use_container_width=True,
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
                use_container_width=True,
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
    st.dataframe(feature_frame, hide_index=True, use_container_width=True)


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
        st.pyplot(bounded_figure, clear_figure=True, use_container_width=True)
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
        st.pyplot(unbounded_figure, clear_figure=True, use_container_width=True)
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
        use_container_width=True,
    )

    download_left, download_right = st.columns(2)
    bounded_frame = curves_to_frame(bounded_results)
    unbounded_frame = curves_to_frame(unbounded_results)
    download_left.download_button(
        "Download bounded density CSV",
        dataframe_to_csv_bytes(bounded_frame),
        file_name="bounded_kde_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_right.download_button(
        "Download unbounded density CSV",
        dataframe_to_csv_bytes(unbounded_frame),
        file_name="unbounded_kde_comparison.csv",
        mime="text/csv",
        use_container_width=True,
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
    st.pyplot(density_figure, clear_figure=True, use_container_width=True)
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
        use_container_width=True,
    )

    curve_frame = curves_to_frame(curves, true_density=true_density)
    score_frame = table[["Method", "Bandwidth", "Empirical log score (bits)"]]
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download simulation curves",
        dataframe_to_csv_bytes(curve_frame),
        file_name="simulation_kde_curves.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_right.download_button(
        "Download bandwidths and scores",
        dataframe_to_csv_bytes(score_frame),
        file_name="simulation_log_scores.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Generated-task and selector details"):
        metadata_frame = pd.DataFrame(
            {
                "Setting": list(task.metadata),
                "Value": [str(value) for value in task.metadata.values()],
            }
        )
        st.dataframe(metadata_frame, hide_index=True, use_container_width=True)
        st.markdown(
            f"**Working interval used for all bandwidth selectors:** "
            f"`[{format_number(task.working_support[0])}, "
            f"{format_number(task.working_support[1])}]`"
        )
        render_neural_details(result["neural_result"])


# ---------------------------------------------------------------------------
# Streamlit interface
# ---------------------------------------------------------------------------


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
link_columns[0].link_button("Read the paper", PAPER_URL, use_container_width=True)
link_columns[1].link_button("View code on GitHub", GITHUB_URL, use_container_width=True)

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
            use_container_width=True,
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
            use_container_width=True,
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


else:
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
        use_container_width=True,
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
