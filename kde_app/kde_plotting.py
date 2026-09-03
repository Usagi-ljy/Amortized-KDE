"""Consistent plotting for KDE curves and empirical log-score comparisons."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .kde_estimators import KDEResult


TrueDensity = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]

METHOD_STYLES: Mapping[str, dict[str, object]] = {
    "amortized": {"color": "#0072B2", "linestyle": "-", "linewidth": 2.5},
    "gaussian": {"color": "#0072B2", "linestyle": "-", "linewidth": 2.5},
    "multifamily": {"color": "#009E73", "linestyle": "-", "linewidth": 2.5},
    "gmm32": {"color": "#D55E00", "linestyle": "-", "linewidth": 2.5},
    "silverman": {"color": "#E69F00", "linestyle": "--", "linewidth": 2.1},
    "sheather-jones": {"color": "#CC79A7", "linestyle": "-.", "linewidth": 2.1},
    "lscv": {"color": "#56B4E9", "linestyle": ":", "linewidth": 2.3},
}


def _normalise_method_name(method: str) -> str:
    key = method.lower().replace("–", "-").replace("—", "-").strip()
    if "silverman" in key:
        return "silverman"
    if "sheather" in key or key == "sj":
        return "sheather-jones"
    if "lscv" in key:
        return "lscv"
    if "multi" in key:
        return "multifamily"
    if "gmm" in key:
        return "gmm32"
    if "gaussian" in key or "normal" in key:
        return "gaussian"
    return "amortized"


def _validate_results(results: Mapping[str, KDEResult]) -> tuple[np.ndarray, str]:
    if not results:
        raise ValueError("At least one KDE result is required.")
    first = next(iter(results.values()))
    grid = np.asarray(first.x_grid, dtype=np.float64)
    mode = first.mode
    for result in results.values():
        if result.mode != mode:
            raise ValueError("All curves must use the same bounded/unbounded mode.")
        if result.x_grid.shape != grid.shape or not np.allclose(
            result.x_grid, grid, rtol=0.0, atol=0.0
        ):
            raise ValueError("All curves must use the same x-grid.")
    return grid, mode


def plot_kde_comparison(
    results: Mapping[str, KDEResult],
    *,
    samples: Optional[Iterable[float]] = None,
    true_density: Optional[TrueDensity] = None,
    true_density_label: str = "True density",
    show_histogram: bool = False,
    show_rug: bool = False,
    show_support: bool = True,
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = "Density",
    figure_size: tuple[float, float] = (8.0, 4.8),
) -> Figure:
    """Plot selected KDE estimates together, with bandwidths in the legend."""

    grid, mode = _validate_results(results)
    fig, axis = plt.subplots(figsize=figure_size)

    sample_array: Optional[np.ndarray] = None
    if samples is not None:
        sample_array = np.asarray(list(samples), dtype=np.float64)
        if sample_array.ndim != 1 or not np.all(np.isfinite(sample_array)):
            raise ValueError("samples must be a finite one-dimensional sequence.")
        if show_histogram:
            axis.hist(
                sample_array,
                bins="auto",
                density=True,
                color="#B8B8B8",
                alpha=0.28,
                edgecolor="none",
                label="Sample histogram",
            )

    if true_density is not None:
        density = (
            np.asarray(true_density(grid), dtype=np.float64)
            if callable(true_density)
            else np.asarray(true_density, dtype=np.float64)
        )
        if density.shape != grid.shape or not np.all(np.isfinite(density)):
            raise ValueError("true_density must return one finite value per grid point.")
        axis.plot(
            grid,
            density,
            color="#222222",
            linestyle="-",
            linewidth=2.7,
            label=true_density_label,
            zorder=4,
        )

    for method, result in results.items():
        style = dict(METHOD_STYLES[_normalise_method_name(method)])
        axis.plot(
            result.x_grid,
            result.density,
            label=f"{method} (h = {result.bandwidth:.4g})",
            zorder=3,
            **style,
        )

    if sample_array is not None and show_rug:
        axis.plot(
            sample_array,
            np.zeros_like(sample_array),
            "|",
            color="#444444",
            markersize=7,
            markeredgewidth=0.8,
            alpha=0.55,
            label="Observations",
        )

    if mode == "bounded" and show_support:
        first = next(iter(results.values()))
        if first.support is not None:
            left, right = first.support
            axis.axvline(left, color="#777777", linewidth=1.0, linestyle="--")
            axis.axvline(right, color="#777777", linewidth=1.0, linestyle="--")

    axis.set_xlim(float(grid[0]), float(grid[-1]))
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(
        title
        if title is not None
        else ("Bounded KDE comparison" if mode == "bounded" else "Unbounded KDE comparison")
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_log_score_comparison(
    log_scores: Mapping[str, float],
    *,
    title: str = "Empirical logarithmic score",
    ylabel: str = "Empirical logarithmic score (bits)",
    figure_size: tuple[float, float] = (7.0, 4.2),
) -> Figure:
    """Draw a method comparison for empirical negative log-score values."""

    if not log_scores:
        raise ValueError("At least one log score is required.")
    methods = list(log_scores)
    values = np.asarray([log_scores[name] for name in methods], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("All log scores must be finite.")
    colours = [METHOD_STYLES[_normalise_method_name(name)]["color"] for name in methods]

    fig, axis = plt.subplots(figsize=figure_size)
    bars = axis.bar(methods, values, color=colours, width=0.66)
    for bar, value in zip(bars, values):
        vertical_offset = 3 if value >= 0.0 else -13
        axis.annotate(
            f"{value:.4f}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=9,
        )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)
    fig.tight_layout()
    return fig
