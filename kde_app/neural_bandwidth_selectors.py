"""Load the three published neural bandwidth selectors.

The public checkpoints use two different neural architectures:

* ``gaussian_selector.pt`` learns a dimensionless ratio from the sample size
  and returns ``sample_std * ratio``;
* ``multifamily_selector.pt`` and ``gmm32_selector.pt`` use the five raw
  features ``[n, mean, sample std, skewness, kurtosis]`` after the sample is
  affinely mapped to the reference interval ``[-1, 1]``.

This module only predicts bandwidths.  KDE evaluation is deliberately kept in
``kde_estimators.py`` so every selector uses the same density formula.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SelectorName = Literal["gaussian", "multifamily", "gmm32"]
PathLike = Union[str, Path]

DTYPE = torch.float64
REFERENCE_SUPPORT = (-1.0, 1.0)
TRAIN_N_MIN = 5
TRAIN_N_MAX = 256
MIN_BANDWIDTH = 1e-10
MOMENT_EPS = 1e-12
FEATURE_CLIP = 1e6

DEFAULT_CHECKPOINTS: Mapping[str, str] = {
    "gaussian": "gaussian_selector.pt",
    "multifamily": "multifamily_selector.pt",
    "gmm32": "gmm32_selector.pt",
}

_SELECTOR_ALIASES = {
    "gaussian": "gaussian",
    "normal": "gaussian",
    "multifamily": "multifamily",
    "multi-family": "multifamily",
    "multi_family": "multifamily",
    "gmm32": "gmm32",
    "gmm": "gmm32",
    "gmm-k32": "gmm32",
    "gmm_k32": "gmm32",
}


class GaussianBandwidthNetwork(nn.Module):
    """Gaussian selector architecture: ``1 -> 4 -> 4 -> 1``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, sample_size: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(sample_size))
        hidden = F.relu(self.fc2(hidden))
        return F.softplus(self.fc3(hidden).squeeze(-1))


class FiveFeatureBandwidthNetwork(nn.Module):
    """Multi-family/GMM32 architecture: ``5 -> 128 -> 128 -> 1``."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class _LegacyGaussianConfig:
    """Compatibility target for ``__main__.Config`` in the old checkpoint."""


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: ANN201
        if module == "__main__" and name == "Config":
            return _LegacyGaussianConfig
        return super().find_class(module, name)


class _CheckpointPickleModule:
    """Small module-like object accepted by ``torch.load``."""

    __name__ = "checkpoint_compat_pickle"
    Unpickler = _CheckpointUnpickler


@dataclass(frozen=True)
class NeuralBandwidthResult:
    """One neural-selector bandwidth prediction on the original data scale."""

    selector: str
    n: int
    support: tuple[float, float]
    bandwidth: float
    reference_bandwidth: float
    feature_names: tuple[str, ...]
    features: np.ndarray
    bandwidth_ratio: Optional[float] = None


def _normalise_selector_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _SELECTOR_ALIASES:
        allowed = ", ".join(DEFAULT_CHECKPOINTS)
        raise ValueError(f"Unknown selector {name!r}. Choose one of: {allowed}.")
    return _SELECTOR_ALIASES[key]


def _as_sample(samples: Iterable[float]) -> np.ndarray:
    sample = np.asarray(samples, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    if sample.size < 2:
        raise ValueError("At least two observations are required.")
    if not np.all(np.isfinite(sample)):
        raise ValueError("samples contain NaN or infinite values.")
    return sample


def _validate_support(support: tuple[float, float]) -> tuple[float, float]:
    if len(support) != 2:
        raise ValueError("support must be a pair (left, right).")
    left, right = float(support[0]), float(support[1])
    if not (math.isfinite(left) and math.isfinite(right) and left < right):
        raise ValueError("support must contain finite endpoints with left < right.")
    return left, right


def _validate_sample_in_support(
    sample: np.ndarray,
    support: tuple[float, float],
) -> None:
    left, right = support
    tolerance = 1e-12 * max(1.0, abs(left), abs(right))
    if sample.min() < left - tolerance or sample.max() > right + tolerance:
        raise ValueError("At least one observation lies outside the declared support.")


def to_reference_scale(
    samples: Iterable[float],
    support: tuple[float, float],
) -> tuple[np.ndarray, float, float]:
    """Map ``support`` to ``[-1, 1]`` and return sample, centre and scale."""

    sample = _as_sample(samples)
    left, right = _validate_support(support)
    centre = 0.5 * (left + right)
    scale = 0.5 * (right - left)
    return (sample - centre) / scale, centre, scale


def compute_five_features(sample_reference: Iterable[float]) -> np.ndarray:
    """Return the exact five raw-n features used for Multi-family/GMM32."""

    sample = _as_sample(sample_reference)
    mean = float(sample.mean())
    centred = sample - mean
    m2 = float(np.mean(centred**2))
    m3 = float(np.mean(centred**3))
    m4 = float(np.mean(centred**4))
    sample_std = float(np.std(sample, ddof=1))
    skewness = m3 / max(m2, MOMENT_EPS) ** 1.5
    kurtosis = m4 / max(m2, MOMENT_EPS) ** 2.0
    features = np.asarray(
        [sample.size, mean, sample_std, skewness, kurtosis],
        dtype=np.float64,
    )
    return np.clip(features, -FEATURE_CLIP, FEATURE_CLIP)


def _torch_load(path: Path, legacy_gaussian: bool) -> object:
    kwargs: dict[str, object] = {"map_location": "cpu"}
    if legacy_gaussian:
        kwargs["pickle_module"] = _CheckpointPickleModule
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _extract_state_dict(checkpoint: object, selector: str) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Expected a dictionary checkpoint.")

    candidate_keys = (
        ("model_state_dict", "model", "ema_model")
        if selector == "gaussian"
        else ("ema_model", "model", "model_state_dict")
    )
    state: object = None
    for key in candidate_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            state = value
            break

    if state is None and checkpoint and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in checkpoint.items()
    ):
        state = checkpoint

    if not isinstance(state, dict):
        raise RuntimeError(f"No model weights were found for selector {selector!r}.")

    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


class NeuralBandwidthSelector:
    """Load and reuse one published neural bandwidth selector."""

    def __init__(
        self,
        selector: str,
        checkpoint_path: Optional[PathLike] = None,
        *,
        checkpoint_directory: Optional[PathLike] = None,
        device: Union[str, torch.device] = "cpu",
        strict_sample_size: bool = True,
    ) -> None:
        self.selector = _normalise_selector_name(selector)
        self.device = torch.device(device)
        self.strict_sample_size = bool(strict_sample_size)

        if checkpoint_path is None:
            base = (
                Path(checkpoint_directory)
                if checkpoint_directory is not None
                else Path(__file__).resolve().parents[1] / "models"
            )
            checkpoint_path = base / DEFAULT_CHECKPOINTS[self.selector]
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = _torch_load(
            self.checkpoint_path,
            legacy_gaussian=self.selector == "gaussian",
        )
        state = _extract_state_dict(checkpoint, self.selector)

        if self.selector == "gaussian":
            model: nn.Module = GaussianBandwidthNetwork()
        else:
            model = FiveFeatureBandwidthNetwork(hidden=128)
        self.model = model.to(device=self.device, dtype=DTYPE)
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                f"Checkpoint architecture does not match selector {self.selector!r}: "
                f"{self.checkpoint_path.name}"
            ) from error
        self.model.eval()

    def _validate_sample_size(self, n: int) -> None:
        if self.strict_sample_size and not (TRAIN_N_MIN <= n <= TRAIN_N_MAX):
            raise ValueError(
                f"The selectors were trained for {TRAIN_N_MIN} <= n <= "
                f"{TRAIN_N_MAX}; received n={n}."
            )

    @torch.no_grad()
    def predict_bandwidth(
        self,
        samples: Iterable[float],
        support: tuple[float, float] = REFERENCE_SUPPORT,
    ) -> NeuralBandwidthResult:
        """Predict a bandwidth and return it on the original sample scale."""

        sample = _as_sample(samples)
        support = _validate_support(support)
        _validate_sample_in_support(sample, support)
        self._validate_sample_size(int(sample.size))

        sample_reference, _, scale = to_reference_scale(sample, support)
        if sample_reference.min() < -1.0 - 1e-10 or sample_reference.max() > 1.0 + 1e-10:
            raise RuntimeError("The affine reference transformation failed.")

        if self.selector == "gaussian":
            sample_std = float(np.std(sample, ddof=1))
            reference_std = float(np.std(sample_reference, ddof=1))
            if not math.isfinite(sample_std) or sample_std <= 0.0:
                raise ValueError("The Gaussian selector requires non-zero sample spread.")
            feature_tensor = torch.tensor(
                [[float(sample.size)]], dtype=DTYPE, device=self.device
            )
            ratio = float(self.model(feature_tensor).reshape(()).item())
            bandwidth = sample_std * ratio
            reference_bandwidth = reference_std * ratio
            feature_names = ("n",)
            features = np.asarray([sample.size], dtype=np.float64)
            bandwidth_ratio: Optional[float] = ratio
        else:
            features = compute_five_features(sample_reference)
            feature_tensor = torch.as_tensor(
                features[None, :], dtype=DTYPE, device=self.device
            )
            raw_output = self.model(feature_tensor)
            reference_bandwidth = float(
                (F.softplus(raw_output) + MIN_BANDWIDTH).reshape(()).item()
            )
            bandwidth = scale * reference_bandwidth
            feature_names = ("n", "mean", "sample_std", "skewness", "kurtosis")
            bandwidth_ratio = None

        if not math.isfinite(bandwidth) or bandwidth <= 0.0:
            raise RuntimeError("The neural selector returned an invalid bandwidth.")

        return NeuralBandwidthResult(
            selector=self.selector,
            n=int(sample.size),
            support=support,
            bandwidth=float(bandwidth),
            reference_bandwidth=float(reference_bandwidth),
            feature_names=feature_names,
            features=features.copy(),
            bandwidth_ratio=bandwidth_ratio,
        )


def load_neural_selectors(
    checkpoint_directory: Optional[PathLike] = None,
    selectors: Iterable[str] = ("gaussian", "multifamily", "gmm32"),
    *,
    device: Union[str, torch.device] = "cpu",
    strict_sample_size: bool = True,
) -> dict[str, NeuralBandwidthSelector]:
    """Load several selectors once for reuse by the web application."""

    loaded: dict[str, NeuralBandwidthSelector] = {}
    for name in selectors:
        canonical_name = _normalise_selector_name(name)
        loaded[canonical_name] = NeuralBandwidthSelector(
            canonical_name,
            checkpoint_directory=checkpoint_directory,
            device=device,
            strict_sample_size=strict_sample_size,
        )
    return loaded
