"""Classical one-dimensional KDE bandwidth selectors used in the paper.

The definitions intentionally match the benchmark implementation:

* Silverman: ``0.9 * min(sample_std, IQR / 1.34) * n**(-1/5)``;
* Sheather--Jones: R ``stats::bw.SJ(method="ste")`` with ``"dpi"`` fallback;
* LSCV: Gaussian least-squares cross-validation over 60 logarithmically
  spaced candidates from ``0.2`` to ``5`` times the Silverman bandwidth.

The exact R call is retained for Sheather--Jones because commonly available
Python packages implement different selectors under similar names.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from typing import Iterable, Optional

import numpy as np


MIN_BANDWIDTH = 1e-10
LSCV_GRID_SIZE = 60
LSCV_FACTOR_LOW = 0.2
LSCV_FACTOR_HIGH = 5.0


def _as_sample(samples: Iterable[float]) -> np.ndarray:
    sample = np.asarray(samples, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    if sample.size < 2:
        raise ValueError("At least two observations are required.")
    if not np.all(np.isfinite(sample)):
        raise ValueError("samples contain NaN or infinite values.")
    if float(np.ptp(sample)) <= 0.0:
        raise ValueError("Bandwidth selection requires non-zero sample spread.")
    return sample


def silverman_bandwidth(samples: Iterable[float]) -> float:
    """Return the robust Silverman/R ``bw.nrd0`` bandwidth."""

    sample = _as_sample(samples)
    n = int(sample.size)
    sample_std = float(np.std(sample, ddof=1))
    q25, q75 = np.quantile(sample, [0.25, 0.75], method="linear")
    robust_scale = min(sample_std, float(q75 - q25) / 1.34)
    if not math.isfinite(robust_scale) or robust_scale <= 0.0:
        robust_scale = sample_std
    bandwidth = 0.9 * robust_scale * n ** (-0.2)
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise RuntimeError("Silverman's rule returned an invalid bandwidth.")
    return float(max(bandwidth, MIN_BANDWIDTH))


def sheather_jones_is_available(rscript_executable: Optional[str] = None) -> bool:
    """Return whether an Rscript executable is available."""

    executable = rscript_executable or "Rscript"
    return shutil.which(executable) is not None


def sheather_jones_bandwidth(
    samples: Iterable[float],
    *,
    rscript_executable: Optional[str] = None,
    timeout_seconds: float = 30.0,
) -> float:
    """Return R's Sheather--Jones solve-the-equation bandwidth.

    This calls ``stats::bw.SJ(x, method="ste")`` exactly as in the paper's
    experiments and falls back to ``method="dpi"`` only if STE fails.
    Values are sent through standard input; no user data are written to disk.
    """

    sample = _as_sample(samples)
    executable = rscript_executable or "Rscript"
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(
            "Sheather--Jones requires Rscript because the paper uses "
            "R stats::bw.SJ. Install R locally; for Streamlit deployment, "
            "add r-base-core to packages.txt."
        )

    r_code = r'''
x <- scan(file("stdin"), what=double(), quiet=TRUE)
h <- suppressWarnings(tryCatch(
    stats::bw.SJ(x, method="ste"),
    error=function(e) NA_real_
))
if (!is.finite(h) || h <= 0) {
    h <- suppressWarnings(tryCatch(
        stats::bw.SJ(x, method="dpi"),
        error=function(e) NA_real_
    ))
}
if (!is.finite(h) || h <= 0) {
    stop("stats::bw.SJ failed for this sample")
}
cat(format(h, digits=17, scientific=TRUE))
'''
    payload = "\n".join(format(value, ".17g") for value in sample) + "\n"
    try:
        completed = subprocess.run(
            [resolved, "--vanilla", "-e", r_code],
            input=payload,
            text=True,
            capture_output=True,
            check=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("R stats::bw.SJ timed out.") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or "R stats::bw.SJ failed."
        raise RuntimeError(message) from error

    try:
        bandwidth = float(completed.stdout.strip())
    except ValueError as error:
        raise RuntimeError("R stats::bw.SJ returned an unreadable value.") from error
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise RuntimeError("R stats::bw.SJ returned an invalid bandwidth.")
    return bandwidth


def lscv_bandwidth(
    samples: Iterable[float],
    *,
    reference_bandwidth: Optional[float] = None,
    grid_size: int = LSCV_GRID_SIZE,
    factor_low: float = LSCV_FACTOR_LOW,
    factor_high: float = LSCV_FACTOR_HIGH,
) -> float:
    """Return the Gaussian least-squares cross-validation bandwidth."""

    sample = _as_sample(samples)
    n = int(sample.size)
    reference = (
        silverman_bandwidth(sample)
        if reference_bandwidth is None
        else float(reference_bandwidth)
    )
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_bandwidth must be finite and positive.")
    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if not (0.0 < factor_low < factor_high):
        raise ValueError("Require 0 < factor_low < factor_high.")

    candidates = reference * np.logspace(
        math.log10(factor_low), math.log10(factor_high), grid_size
    )
    squared_distances = (sample[:, None] - sample[None, :]) ** 2
    off_diagonal = ~np.eye(n, dtype=bool)
    best_value = math.inf
    best_bandwidth = reference

    for bandwidth in candidates:
        first_kernel = np.exp(
            -squared_distances / (4.0 * bandwidth**2)
        ) / (bandwidth * math.sqrt(4.0 * math.pi))
        first_term = float(first_kernel.mean())

        second_kernel = np.exp(
            -squared_distances / (2.0 * bandwidth**2)
        ) / (bandwidth * math.sqrt(2.0 * math.pi))
        second_term = 2.0 * float(second_kernel[off_diagonal].sum()) / (
            n * (n - 1)
        )
        criterion = first_term - second_term
        if criterion < best_value:
            best_value = criterion
            best_bandwidth = float(bandwidth)

    if not math.isfinite(best_bandwidth) or best_bandwidth <= 0.0:
        raise RuntimeError("LSCV returned an invalid bandwidth.")
    return max(best_bandwidth, MIN_BANDWIDTH)


def compute_classical_bandwidths(
    samples: Iterable[float],
    methods: Iterable[str] = ("silverman", "sheather_jones", "lscv"),
    *,
    rscript_executable: Optional[str] = None,
) -> dict[str, float]:
    """Compute selected classical bandwidths in the requested order."""

    sample = _as_sample(samples)
    aliases = {
        "silverman": "silverman",
        "sj": "sheather_jones",
        "sheather-jones": "sheather_jones",
        "sheather_jones": "sheather_jones",
        "lscv": "lscv",
    }
    result: dict[str, float] = {}
    silverman_value: Optional[float] = None
    for requested in methods:
        key = str(requested).strip().lower()
        if key not in aliases:
            raise ValueError(f"Unknown classical selector: {requested!r}.")
        method = aliases[key]
        if method == "silverman":
            silverman_value = silverman_bandwidth(sample)
            result["Silverman"] = silverman_value
        elif method == "sheather_jones":
            result["Sheather–Jones"] = sheather_jones_bandwidth(
                sample, rscript_executable=rscript_executable
            )
        else:
            if silverman_value is None:
                silverman_value = silverman_bandwidth(sample)
            result["LSCV"] = lscv_bandwidth(
                sample, reference_bandwidth=silverman_value
            )
    return result

