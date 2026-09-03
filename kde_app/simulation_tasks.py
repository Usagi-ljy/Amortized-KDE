"""Known-distribution task generators and single-task evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np
from scipy import optimize, stats
from scipy.special import logsumexp, ndtr, ndtri

from .config import (
    DEFAULT_GRID_SIZE,
    FAMILY_FROM_LABEL,
    FAMILY_LABELS,
    REFERENCE_SUPPORT,
    SQRT_2PI,
)
from .core import (
    compute_selected_bandwidths,
    empirical_log_scores,
    sample_adaptive_interval,
)
from .kde_estimators import estimate_multiple_kdes

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
