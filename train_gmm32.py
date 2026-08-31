"""Train the fresh bounded-GMM K=32 amortized KDE bandwidth selector.

This standalone script combines the final GMM generator, the neural/KDE
infrastructure and the training loop that were separate notebook cells in the
paper experiments.  Every training task receives a newly generated target GMM,
an observed sample and an independent Monte Carlo scoring sample.

The default settings reproduce the reported training protocol.  They are
computationally expensive; ``--quick`` is only a smoke test.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtr, ndtri


torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")

LEFT = -1.0
RIGHT = 1.0
N_COMPONENTS = 32
ALPHA_LOW = 0.50
ALPHA_HIGH = 4.00
MU_LOW = -1.0
MU_HIGH = 1.0
SIGMA_MIN = 0.015
SIGMA_MAX = 0.45
GLOBAL_SIGMA_LOW = 0.025
GLOBAL_SIGMA_HIGH = 0.22
LOG_SIGMA_SD_LOW = 0.10
LOG_SIGMA_SD_HIGH = 1.00

LOG_2PI = math.log(2.0 * math.pi)
LOG2 = math.log(2.0)
SQRT2 = math.sqrt(2.0)
EPS_BANDWIDTH = 1e-10
EPS_MOMENT = 1e-12
EPS_PROBABILITY = 1e-300
FEATURE_CLIP = 1e6


@dataclass
class Config:
    seed: int = 2026
    validation_seed: int = 777
    n_min: int = 5
    n_max: int = 256
    task_batch: int = 256
    mc_samples: int = 1024
    learning_rate: float = 5e-4
    max_steps: int = 40_000
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    validation_repeats_per_n: int = 20
    validation_mc_samples: int = 1024
    validation_every: int = 1000
    minimum_steps: int = 12_000
    early_stop_patience: int = 5
    early_stop_min_delta: float = 5e-5
    output_dir: str = "training_outputs/gmm32"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def sample_component_sigmas(
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Sample component scales from the exact truncated log-normal rule."""
    global_sigma = float(
        log_uniform(rng, GLOBAL_SIGMA_LOW, GLOBAL_SIGMA_HIGH)
    )
    log_sigma_sd = float(
        rng.uniform(LOG_SIGMA_SD_LOW, LOG_SIGMA_SD_HIGH)
    )
    log_center = math.log(global_sigma)
    lower = (math.log(SIGMA_MIN) - log_center) / log_sigma_sd
    upper = (math.log(SIGMA_MAX) - log_center) / log_sigma_sd
    cdf_lower = float(ndtr(lower))
    cdf_upper = float(ndtr(upper))
    if cdf_upper <= cdf_lower:
        raise RuntimeError("Invalid truncated log-sigma interval.")

    u = cdf_lower + rng.uniform(size=N_COMPONENTS) * (cdf_upper - cdf_lower)
    eps = np.finfo(np.float64).eps
    u = np.clip(u, eps, 1.0 - eps)
    sigmas = np.exp(log_center + log_sigma_sd * ndtri(u)).astype(np.float64)
    if np.any(sigmas < SIGMA_MIN) or np.any(sigmas > SIGMA_MAX):
        raise RuntimeError("A component scale lies outside the frozen range.")
    return sigmas, global_sigma, log_sigma_sd


def generate_bounded_gmm(rng: np.random.Generator) -> dict[str, object]:
    """Generate a raw GMM and condition it exactly on [-1, 1]."""
    alpha = float(log_uniform(rng, ALPHA_LOW, ALPHA_HIGH))
    concentration = np.full(N_COMPONENTS, alpha / N_COMPONENTS)
    raw_weights = rng.dirichlet(concentration).astype(np.float64)
    means = rng.uniform(MU_LOW, MU_HIGH, size=N_COMPONENTS).astype(np.float64)
    sigmas, global_sigma, log_sigma_sd = sample_component_sigmas(rng)

    component_mass = ndtr((RIGHT - means) / sigmas) - ndtr(
        (LEFT - means) / sigmas
    )
    truncation_mass = float(np.sum(raw_weights * component_mass))
    if not np.isfinite(truncation_mass) or truncation_mass <= 0.0:
        raise RuntimeError("Invalid GMM truncation mass.")
    bounded_weights = raw_weights * component_mass / truncation_mass

    return {
        "alpha": alpha,
        "raw_weights": raw_weights,
        "bounded_weights": bounded_weights,
        "means": means,
        "sigmas": sigmas,
        "truncation_mass": truncation_mass,
        "global_sigma": global_sigma,
        "log_sigma_sd": log_sigma_sd,
    }


def sample_from_bounded_gmm(
    rng: np.random.Generator,
    gmm: dict[str, object],
    n: int,
) -> np.ndarray:
    """Sample exactly from the conditioned GMM by inverse-CDF sampling."""
    component = rng.choice(
        N_COMPONENTS,
        size=int(n),
        p=np.asarray(gmm["bounded_weights"]),
    )
    means = np.asarray(gmm["means"])[component]
    sigmas = np.asarray(gmm["sigmas"])[component]
    cdf_left = ndtr((LEFT - means) / sigmas)
    cdf_right = ndtr((RIGHT - means) / sigmas)
    if np.any(cdf_right <= cdf_left):
        raise RuntimeError("Degenerate truncated-normal sampling interval.")

    u = cdf_left + rng.uniform(size=int(n)) * (cdf_right - cdf_left)
    eps = np.finfo(np.float64).eps
    x = means + sigmas * ndtri(np.clip(u, eps, 1.0 - eps))
    if np.any(x < LEFT - 1e-12) or np.any(x > RIGHT + 1e-12):
        raise RuntimeError("The bounded sampler generated an out-of-range value.")
    return x.astype(np.float64, copy=False)


def generator_metadata() -> dict[str, object]:
    return {
        "generator_name": "simplified_bounded_GMM_fresh_generation",
        "generator_version": 1,
        "K": N_COMPONENTS,
        "support": [LEFT, RIGHT],
        "alpha_distribution": "LogUniform(0.5,4.0)",
        "weight_distribution": "Dirichlet(alpha/K,...,alpha/K)",
        "mu_distribution": "iid Uniform(-1,1)",
        "sigma_min": SIGMA_MIN,
        "sigma_max": SIGMA_MAX,
        "global_sigma_distribution": "LogUniform(0.025,0.22)",
        "log_sigma_sd_distribution": "Uniform(0.10,1.00)",
        "conditioning": "exact conditioning on [-1,1]; no clipping",
    }


class BandwidthNet(nn.Module):
    """Five raw sample features -> positive bandwidth."""

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


def positive_bandwidth(raw_output: torch.Tensor) -> torch.Tensor:
    return (F.softplus(raw_output) + EPS_BANDWIDTH).squeeze(-1)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(
        ema_model.parameters(), model.parameters()
    ):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def sample_features(samples: list[torch.Tensor]) -> torch.Tensor:
    """Return [n, mean, unbiased sd, moment skewness, moment kurtosis]."""
    features = []
    for sample in samples:
        mean = sample.mean()
        centered = sample - mean
        m2 = centered.square().mean()
        m3 = centered.pow(3).mean()
        m4 = centered.pow(4).mean()
        feature = torch.stack(
            (
                torch.tensor(float(sample.numel()), device=DEVICE),
                mean,
                sample.std(unbiased=True),
                m3 / torch.clamp(m2, min=EPS_MOMENT).pow(1.5),
                m4 / torch.clamp(m2, min=EPS_MOMENT).pow(2.0),
            )
        )
        features.append(torch.clamp(feature, -FEATURE_CLIP, FEATURE_CLIP))
    return torch.stack(features)


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


def truncated_kde_log_score_bits(
    sample: torch.Tensor,
    bandwidth: torch.Tensor,
    evaluation_sample: torch.Tensor,
) -> torch.Tensor:
    """Monte Carlo -mean(log2 q_h^T(X)) on [-1, 1]."""
    bandwidth = torch.clamp(bandwidth.reshape(()), min=EPS_BANDWIDTH)
    z = (evaluation_sample[:, None] - sample[None, :]) / bandwidth
    log_kernel = -0.5 * z.square() - torch.log(bandwidth) - 0.5 * LOG_2PI
    log_raw_kde = torch.logsumexp(log_kernel, dim=1) - math.log(sample.numel())

    component_mass = normal_cdf((RIGHT - sample) / bandwidth) - normal_cdf(
        (LEFT - sample) / bandwidth
    )
    truncation_constant = torch.clamp(
        torch.clamp(component_mass, min=EPS_PROBABILITY).mean(),
        min=EPS_PROBABILITY,
    )
    log_bounded_kde = log_raw_kde - torch.log(truncation_constant)
    return -log_bounded_kde.mean() / LOG2


def generate_task(
    rng: np.random.Generator,
    n: int,
    scoring_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gmm = generate_bounded_gmm(rng)
    observed = sample_from_bounded_gmm(rng, gmm, n)
    scoring = sample_from_bounded_gmm(rng, gmm, scoring_size)
    return (
        torch.as_tensor(observed, dtype=torch.float64, device=DEVICE),
        torch.as_tensor(scoring, dtype=torch.float64, device=DEVICE),
    )


def validation_metadata(config: Config) -> dict[str, object]:
    return {
        "generator": generator_metadata(),
        "repeats_per_n": config.validation_repeats_per_n,
        "mc_samples": config.validation_mc_samples,
        "n_min": config.n_min,
        "n_max": config.n_max,
        "seed": config.validation_seed,
    }


@torch.no_grad()
def load_or_build_validation(
    path: Path,
    config: Config,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    expected = validation_metadata(config)
    if path.exists():
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(path, map_location="cpu")
        if saved.get("metadata") == expected:
            print(f"Loaded fixed validation pool: {path}")
            return saved["samples"], saved["scoring_samples"]

    rng = np.random.default_rng(config.validation_seed)
    samples: list[torch.Tensor] = []
    scoring_samples: list[torch.Tensor] = []
    total = (config.n_max - config.n_min + 1) * config.validation_repeats_per_n
    print(f"Building fixed validation pool with {total} fresh GMM tasks...")
    for n in range(config.n_min, config.n_max + 1):
        for _ in range(config.validation_repeats_per_n):
            sample, scoring = generate_task(
                rng,
                n=n,
                scoring_size=config.validation_mc_samples,
            )
            samples.append(sample.cpu())
            scoring_samples.append(scoring.cpu())

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": expected,
            "samples": samples,
            "scoring_samples": scoring_samples,
        },
        path,
    )
    print(f"Saved fixed validation pool: {path}")
    return samples, scoring_samples


@torch.no_grad()
def validate(
    model: BandwidthNet,
    samples: list[torch.Tensor],
    scoring_samples: list[torch.Tensor],
    features: torch.Tensor,
) -> float:
    model.eval()
    bandwidths = positive_bandwidth(model(features))
    total = torch.zeros((), dtype=torch.float64, device=DEVICE)
    for index, (sample, scoring) in enumerate(zip(samples, scoring_samples)):
        total += truncated_kde_log_score_bits(
            sample,
            bandwidths[index],
            scoring,
        )
    model.train()
    return float((total / len(samples)).item())


def save_checkpoint(
    path: Path,
    model: BandwidthNet,
    ema_model: BandwidthNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_step: int,
    best_validation: float,
    config: Config,
    history: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "best_step": int(best_step),
            "best_val_CE_bits": float(best_validation),
            "cfg": {
                **asdict(config),
                "generator": generator_metadata(),
                "features": "n, mean, sample std, skewness, kurtosis",
                "model": "BandwidthNet(in_dim=5, hidden=128)",
                "kde": "Gaussian KDE with exact truncation renormalization",
                "log_score_unit": "bits",
            },
            "history": history,
        },
        path,
    )


def save_history(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(config: Config) -> Path:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    best_path = output_dir / "gmm32_selector_best.pt"
    last_path = output_dir / "gmm32_selector_last.pt"
    validation_path = output_dir / "fixed_validation_pool.pt"
    history_path = output_dir / "validation_history.csv"

    training_rng = np.random.default_rng(config.seed)
    model = BandwidthNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ema_model = deepcopy(model).to(DEVICE)
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    val_samples_cpu, val_scoring_cpu = load_or_build_validation(
        validation_path,
        config,
    )
    val_samples = [sample.to(DEVICE) for sample in val_samples_cpu]
    val_scoring = [sample.to(DEVICE) for sample in val_scoring_cpu]
    with torch.no_grad():
        val_features = sample_features(val_samples)

    print("Fresh bounded-GMM K=32 selector training")
    print(f"device={DEVICE}, dtype={torch.get_default_dtype()}")
    print(
        f"n={config.n_min}..{config.n_max}, tasks/step={config.task_batch}, "
        f"MC/task={config.mc_samples}, max_steps={config.max_steps}"
    )

    best_validation = float("inf")
    best_step = 0
    bad_count = 0
    history: list[dict[str, object]] = []
    stopped_early = False
    final_step = 0

    for step in range(1, config.max_steps + 1):
        observed: list[torch.Tensor] = []
        scoring: list[torch.Tensor] = []
        for _ in range(config.task_batch):
            n = random.randint(config.n_min, config.n_max)
            sample, score_sample = generate_task(
                training_rng,
                n=n,
                scoring_size=config.mc_samples,
            )
            observed.append(sample)
            scoring.append(score_sample)

        features = sample_features(observed)
        bandwidths = positive_bandwidth(model(features))
        task_losses = torch.stack(
            [
                truncated_kde_log_score_bits(
                    observed[index],
                    bandwidths[index],
                    scoring[index],
                )
                for index in range(config.task_batch)
            ]
        )
        loss = task_losses.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        update_ema(ema_model, model, config.ema_decay)
        final_step = step

        if step != 1 and step % config.validation_every != 0:
            continue

        validation = validate(
            ema_model,
            val_samples,
            val_scoring,
            val_features,
        )
        improved = validation < best_validation - config.early_stop_min_delta
        if improved:
            best_validation = validation
            best_step = step
            bad_count = 0
        elif step >= config.minimum_steps:
            bad_count += 1

        history.append(
            {
                "step": step,
                "train_CE_bits": float(loss.item()),
                "val_CE_nn_bits": validation,
                "improved": improved,
                "bad_count": bad_count,
                "best_step": best_step,
                "best_val_CE_bits": best_validation,
            }
        )
        save_history(history_path, history)

        if improved:
            save_checkpoint(
                best_path,
                model,
                ema_model,
                optimizer,
                step,
                best_step,
                best_validation,
                config,
                history,
            )

        print(
            f"step={step:6d} fresh_tasks={step * config.task_batch:,} "
            f"train_bits={loss.item():.6f} val_bits={validation:.6f} "
            f"best={best_validation:.6f}@{best_step} "
            f"bad={bad_count}/{config.early_stop_patience}"
        )

        if (
            step >= config.minimum_steps
            and bad_count >= config.early_stop_patience
        ):
            stopped_early = True
            print(f"Early stopping at step {step}.")
            break

    save_checkpoint(
        last_path,
        model,
        ema_model,
        optimizer,
        final_step,
        best_step,
        best_validation,
        config,
        history,
    )
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Stopped early: {stopped_early}")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="training_outputs/gmm32")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(output_dir=args.output_dir)
    if args.quick:
        if args.output_dir == "training_outputs/gmm32":
            config.output_dir = "training_outputs/gmm32_quick"
        config.n_max = 10
        config.task_batch = 4
        config.mc_samples = 64
        config.max_steps = 2
        config.validation_repeats_per_n = 1
        config.validation_mc_samples = 64
        config.validation_every = 1
        config.minimum_steps = 2
        print("QUICK MODE: this checks execution only; it does not reproduce the model.")
    train(config)


if __name__ == "__main__":
    main()
