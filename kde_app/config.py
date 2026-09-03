"""Shared configuration for the Amortized KDE web application."""

from __future__ import annotations

from pathlib import Path

APP_TITLE = (
    "Amortized Bandwidth Learning for Kernel Density Estimation "
    "under Logarithmic Score"
)
APP_REVISION = "2026-09-03 · benchmarks-only-v8.1"
APP_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = APP_DIR / "figures"
MODEL_DIR = APP_DIR / "models"

PAPER_URL = "https://arxiv.org/abs/2608.20445"
GITHUB_URL = "https://github.com/Usagi-ljy/Amortized-KDE"
ISSUE_URL = f"{GITHUB_URL}/issues/new"

MIN_SAMPLE_SIZE = 5
MAX_SAMPLE_SIZE = 256
DEFAULT_GRID_SIZE = 512

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
FAMILY_FROM_LABEL = {label: key for key, label in FAMILY_LABELS.items()}

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
