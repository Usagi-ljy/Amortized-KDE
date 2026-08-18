from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INFERENCE_VERSION = "V3-20260818"

DTYPE = torch.float64
REF_LEFT = -1.0
REF_RIGHT = 1.0
TRAIN_N_MIN = 5
TRAIN_N_MAX = 256

H_MIN = 1e-10
EPS_MOM = 1e-12
EPS_PROB = 1e-300
FEAT_CLIP = 1e6

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


class BandwidthNet(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def positive_bandwidth(raw: torch.Tensor) -> torch.Tensor:
    h = F.softplus(raw) + H_MIN
    if h.ndim > 0 and h.shape[-1] == 1:
        h = h.squeeze(-1)
    return h


@dataclass(frozen=True)
class InferenceResult:
    n: int
    support: Tuple[float, float]
    bandwidth: float
    reference_bandwidth: float
    features: np.ndarray
    x_grid: np.ndarray
    density: np.ndarray


def _torch_load_full(path: Union[str, Path]):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / SQRT_2))


def _as_1d_numpy(samples: Iterable[float]) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    if x.size == 0:
        raise ValueError("samples must not be empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples contain NaN or infinite values.")

    return x


def _validate_support(
    support: Tuple[float, float],
) -> Tuple[float, float]:
    if len(support) != 2:
        raise ValueError("support must be a pair (left, right).")

    left = float(support[0])
    right = float(support[1])

    if not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError("support endpoints must be finite.")
    if not left < right:
        raise ValueError("support must satisfy left < right.")

    return left, right


def estimate_support_from_sample_range(
    samples: Iterable[float],
    coverage: float,
) -> Tuple[float, float]:
    """
    Treat the observed sample range as `coverage` of the TOTAL
    finite-support width, centered within that support.

    Example: coverage=0.95 means
        support_width = observed_range / 0.95.

    This is a deterministic support-extension rule, not a
    confidence interval and not a probability-content statement.
    """
    x = _as_1d_numpy(samples)
    coverage = float(coverage)

    if not math.isfinite(coverage):
        raise ValueError("coverage must be finite.")
    if not (0.0 < coverage <= 1.0):
        raise ValueError("coverage must satisfy 0 < coverage <= 1.")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    width = x_max - x_min

    if width <= 0.0:
        raise ValueError(
            "Cannot estimate support from a zero-width sample range."
        )

    center = 0.5 * (x_min + x_max)
    half_support_width = 0.5 * width / coverage

    return _validate_support(
        (
            center - half_support_width,
            center + half_support_width,
        )
    )


def to_reference_scale(
    samples: np.ndarray,
    support: Tuple[float, float],
) -> Tuple[np.ndarray, float, float]:
    left, right = _validate_support(support)

    center = 0.5 * (left + right)
    scale = 0.5 * (right - left)

    z = (samples - center) / scale
    return z, center, scale


def compute_features_rawn(
    sample_reference: torch.Tensor,
) -> torch.Tensor:
    if sample_reference.ndim != 1:
        raise ValueError("sample_reference must have shape [n].")

    n = int(sample_reference.numel())

    if n < 2:
        raise ValueError("At least two observations are required.")

    mean = sample_reference.mean()
    centered = sample_reference - mean
    sample_sd = sample_reference.std(unbiased=True)

    m2 = centered.square().mean()
    m3 = (centered ** 3).mean()
    m4 = (centered ** 4).mean()

    skewness = m3 / (torch.clamp(m2, min=EPS_MOM) ** 1.5)
    kurtosis = m4 / (torch.clamp(m2, min=EPS_MOM) ** 2.0)

    feature = torch.stack(
        [
            torch.tensor(
                float(n),
                dtype=sample_reference.dtype,
                device=sample_reference.device,
            ),
            mean,
            sample_sd,
            skewness,
            kurtosis,
        ]
    )

    feature = torch.clamp(
        feature,
        -FEAT_CLIP,
        FEAT_CLIP,
    )

    return feature.unsqueeze(0)


def truncated_gaussian_kde(
    samples: torch.Tensor,
    x_grid: torch.Tensor,
    bandwidth: torch.Tensor,
    support: Tuple[float, float],
) -> torch.Tensor:
    left, right = _validate_support(support)

    h = bandwidth.reshape(())

    if not torch.isfinite(h) or h <= 0.0:
        raise ValueError(
            "bandwidth must be finite and strictly positive."
        )

    z = (x_grid[:, None] - samples[None, :]) / h

    raw_density = (
        torch.exp(-0.5 * z.square())
        / (SQRT_2PI * h)
    ).mean(dim=1)

    z_right = (right - samples) / h
    z_left = (left - samples) / h

    component_mass = (
        _normal_cdf(z_right)
        - _normal_cdf(z_left)
    )

    component_mass = torch.clamp(
        component_mass,
        min=EPS_PROB,
    )

    truncation_constant = torch.clamp(
        component_mass.mean(),
        min=EPS_PROB,
    )

    density = raw_density / truncation_constant

    inside = (
        (x_grid >= left)
        &
        (x_grid <= right)
    )

    return torch.where(
        inside,
        density,
        torch.zeros_like(density),
    )


class AmortizedKDE:
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        strict_sample_size: bool = True,
    ) -> None:
        self.checkpoint_path = Path(
            checkpoint_path
        ).expanduser().resolve()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {self.checkpoint_path}"
            )

        self.device = torch.device(device)
        self.strict_sample_size = bool(strict_sample_size)

        checkpoint = _torch_load_full(
            self.checkpoint_path
        )

        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                "Expected a dictionary checkpoint."
            )

        if "ema_model" not in checkpoint:
            raise RuntimeError(
                "Checkpoint does not contain 'ema_model'."
            )

        self.checkpoint = checkpoint
        self.cfg = checkpoint.get("cfg", {})

        self._validate_checkpoint_metadata()

        self.model = BandwidthNet(
            in_dim=5,
            hidden=128,
        ).to(
            device=self.device,
            dtype=DTYPE,
        )

        state = checkpoint["ema_model"]

        if (
            state
            and all(isinstance(k, str) for k in state)
            and all(k.startswith("module.") for k in state)
        ):
            state = {
                k[len("module."):]: v
                for k, v in state.items()
            }

        self.model.load_state_dict(
            state,
            strict=True,
        )
        self.model.eval()

    def _validate_checkpoint_metadata(self) -> None:
        cfg = self.cfg if isinstance(self.cfg, dict) else {}
        generator = cfg.get("generator", {})

        observed_k = generator.get("K", None)
        if observed_k is not None and int(observed_k) != 32:
            raise RuntimeError(
                f"Expected GMM K=32 checkpoint, got K={observed_k}."
            )

        interval = cfg.get("interval", None)
        if interval is not None:
            if (
                len(interval) != 2
                or abs(float(interval[0]) + 1.0) > 1e-12
                or abs(float(interval[1]) - 1.0) > 1e-12
            ):
                raise RuntimeError(
                    f"Expected reference interval [-1,1], got {interval}."
                )

    def _validate_samples(
        self,
        samples: np.ndarray,
        support: Tuple[float, float],
    ) -> None:
        n = int(samples.size)
        left, right = support

        if self.strict_sample_size:
            if not (TRAIN_N_MIN <= n <= TRAIN_N_MAX):
                raise ValueError(
                    f"Selector was trained for "
                    f"{TRAIN_N_MIN} <= n <= {TRAIN_N_MAX}; got n={n}."
                )

        tol = 1e-12 * max(
            1.0,
            abs(left),
            abs(right),
        )

        if samples.min() < left - tol:
            raise ValueError(
                "A sample lies below the support."
            )

        if samples.max() > right + tol:
            raise ValueError(
                "A sample lies above the support."
            )

    @torch.no_grad()
    def predict_bandwidth(
        self,
        samples: Iterable[float],
        support: Tuple[float, float] = (-1.0, 1.0),
    ) -> Tuple[float, float, np.ndarray]:
        sample_np = _as_1d_numpy(samples)
        support = _validate_support(support)

        self._validate_samples(
            sample_np,
            support,
        )

        sample_ref_np, _, scale = to_reference_scale(
            sample_np,
            support,
        )

        sample_ref = torch.as_tensor(
            sample_ref_np,
            dtype=DTYPE,
            device=self.device,
        )

        features = compute_features_rawn(
            sample_ref
        )

        raw = self.model(features)
        h_ref = positive_bandwidth(raw).reshape(())

        h_ref_value = float(h_ref.item())
        h_original = float(scale) * h_ref_value

        return (
            h_original,
            h_ref_value,
            features.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64),
        )

    @torch.no_grad()
    def predict(
        self,
        samples: Iterable[float],
        support: Tuple[float, float] = (-1.0, 1.0),
        grid_size: int = 512,
        x_grid: Optional[Iterable[float]] = None,
    ) -> InferenceResult:
        sample_np = _as_1d_numpy(samples)
        support = _validate_support(support)

        self._validate_samples(
            sample_np,
            support,
        )

        bandwidth, reference_bandwidth, features = (
            self.predict_bandwidth(
                sample_np,
                support=support,
            )
        )

        left, right = support

        if x_grid is None:
            grid_size = int(grid_size)
            if grid_size < 2:
                raise ValueError(
                    "grid_size must be at least 2."
                )

            grid_np = np.linspace(
                left,
                right,
                grid_size,
                dtype=np.float64,
            )
        else:
            grid_np = _as_1d_numpy(x_grid)

        sample_t = torch.as_tensor(
            sample_np,
            dtype=DTYPE,
            device=self.device,
        )
        grid_t = torch.as_tensor(
            grid_np,
            dtype=DTYPE,
            device=self.device,
        )
        h_t = torch.tensor(
            bandwidth,
            dtype=DTYPE,
            device=self.device,
        )

        density_t = truncated_gaussian_kde(
            samples=sample_t,
            x_grid=grid_t,
            bandwidth=h_t,
            support=support,
        )

        density_np = (
            density_t.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        return InferenceResult(
            n=int(sample_np.size),
            support=support,
            bandwidth=float(bandwidth),
            reference_bandwidth=float(reference_bandwidth),
            features=features,
            x_grid=grid_np,
            density=density_np,
        )
