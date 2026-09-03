"""Paper benchmark figures shown in the Simulation workflow."""

from __future__ import annotations

from .ui_components import render_benchmarks


def render_simulation_page() -> None:
    """Display repeated-experiment results without a single-task demo."""

    render_benchmarks()
