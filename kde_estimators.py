"""Shared bounded and unbounded one-dimensional Gaussian KDE calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Optional

import numpy as np
from scipy.special import ndtr


KDEMode = Literal["bounded", "unbounded"]
SQRT_2PI = math.sqrt(2.0 * math.pi)
MIN_PROBABILITY = np.finfo(np.float64).tiny


@dataclass(frozen=True)
class KDEResult:
    """One KDE curve evaluated on a shared plotting grid."""

    method: str
    bandwidth: float
    mode: str
    x_grid: np.ndarray
    density: np.ndarray
    numerical_integral: float
    support: Optional[tuple[float, float]] = None


def _as_sample(samples: Iterable[float]) -> np.ndarray:
    sample = np.asarray(samples, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    if sample.size == 0:
        raise ValueError("samples must not be empty.")
    if not np.all(np.isfinite(sample)):
        raise ValueError("samples contain NaN or infinite values.")
    return sample


def _as_grid(x_grid: Iterable[float]) -> np.ndarray:
    grid = np.asarray(x_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("x_grid must be one-dimensional with at least two points.")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("x_grid must be finite and strictly increasing.")
    return grid


def _validate_bandwidth(bandwidth: float) -> float:
    value = float(bandwidth)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("bandwidth must be finite and strictly positive.")
    return value


def _validate_support(support: tuple[float, float]) -> tuple[float, float]:
    if len(support) != 2:
        raise ValueError("support must be a pair (left, right).")
    left, right = float(support[0]), float(support[1])
    if not (math.isfinite(left) and math.isfinite(right) and left < right):
        raise ValueError("support must contain finite endpoints with left < right.")
    return left, right


def _trapezoidal_integral(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def unbounded_gaussian_kde(
    samples: Iterable[float],
    x_grid: Iterable[float],
    bandwidth: float,
) -> np.ndarray:
    """Evaluate the ordinary Gaussian KDE on the real line."""

    sample = _as_sample(samples)
    grid = _as_grid(x_grid)
    bandwidth = _validate_bandwidth(bandwidth)
    standardised = (grid[:, None] - sample[None, :]) / bandwidth
    return np.exp(-0.5 * standardised**2).mean(axis=1) / (
        SQRT_2PI * bandwidth
    )


def bounded_gaussian_kde(
    samples: Iterable[float],
    x_grid: Iterable[float],
    bandwidth: float,
    support: tuple[float, float],
) -> np.ndarray:
    """Evaluate the globally truncated-and-renormalised Gaussian KDE.

    The normalising constant is the mass of the entire raw KDE over the
    declared interval, exactly matching the neural-selector training loss.
    """

    sample = _as_sample(samples)
    grid = _as_grid(x_grid)
    bandwidth = _validate_bandwidth(bandwidth)
    left, right = _validate_support(support)
    tolerance = 1e-12 * max(1.0, abs(left), abs(right))
    if sample.min() < left - tolerance or sample.max() > right + tolerance:
        raise ValueError("At least one observation lies outside the support.")

    raw_density = unbounded_gaussian_kde(sample, grid, bandwidth)
    component_mass = ndtr((right - sample) / bandwidth) - ndtr(
        (left - sample) / bandwidth
    )
    normalising_constant = float(np.mean(np.maximum(component_mass, 0.0)))
    if not math.isfinite(normalising_constant) or normalising_constant <= MIN_PROBABILITY:
        raise RuntimeError("The bounded KDE normalising constant is invalid.")

    density = raw_density / normalising_constant
    inside = (grid >= left) & (grid <= right)
    return np.where(inside, density, 0.0)


def make_shared_grid(
    samples: Iterable[float],
    bandwidths: Iterable[float],
    *,
    mode: KDEMode,
    support: Optional[tuple[float, float]] = None,
    grid_size: int = 512,
    unbounded_tail_bandwidths: float = 4.0,
) -> np.ndarray:
    """Build one common x-grid for every selected method."""

    sample = _as_sample(samples)
    values = np.asarray(list(bandwidths), dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one bandwidth is required.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("All bandwidths must be finite and positive.")
    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")

    if mode == "bounded":
        if support is None:
            raise ValueError("support is required for bounded KDE.")
        left, right = _validate_support(support)
    elif mode == "unbounded":
        margin = float(unbounded_tail_bandwidths) * float(values.max())
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError("unbounded_tail_bandwidths must be positive.")
        left, right = float(sample.min() - margin), float(sample.max() + margin)
    else:
        raise ValueError("mode must be 'bounded' or 'unbounded'.")
    return np.linspace(left, right, grid_size, dtype=np.float64)


def estimate_multiple_kdes(
    samples: Iterable[float],
    bandwidths: Mapping[str, float],
    *,
    mode: KDEMode,
    support: Optional[tuple[float, float]] = None,
    x_grid: Optional[Iterable[float]] = None,
    grid_size: int = 512,
    unbounded_tail_bandwidths: float = 4.0,
) -> dict[str, KDEResult]:
    """Evaluate all methods on the same sample and exactly the same grid."""

    sample = _as_sample(samples)
    if not bandwidths:
        raise ValueError("At least one method must be selected.")
    checked_bandwidths = {
        str(method): _validate_bandwidth(value)
        for method, value in bandwidths.items()
    }
    grid = (
        make_shared_grid(
            sample,
            checked_bandwidths.values(),
            mode=mode,
            support=support,
            grid_size=grid_size,
            unbounded_tail_bandwidths=unbounded_tail_bandwidths,
        )
        if x_grid is None
        else _as_grid(x_grid)
    )

    checked_support = _validate_support(support) if support is not None else None
    results: dict[str, KDEResult] = {}
    for method, bandwidth in checked_bandwidths.items():
        if mode == "bounded":
            if checked_support is None:
                raise ValueError("support is required for bounded KDE.")
            density = bounded_gaussian_kde(
                sample, grid, bandwidth, checked_support
            )
            result_support: Optional[tuple[float, float]] = checked_support
        elif mode == "unbounded":
            density = unbounded_gaussian_kde(sample, grid, bandwidth)
            result_support = None
        else:
            raise ValueError("mode must be 'bounded' or 'unbounded'.")

        results[method] = KDEResult(
            method=method,
            bandwidth=bandwidth,
            mode=mode,
            x_grid=grid.copy(),
            density=density,
            numerical_integral=_trapezoidal_integral(density, grid),
            support=result_support,
        )
    return results

