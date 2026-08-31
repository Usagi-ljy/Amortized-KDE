"""Train the Gaussian amortized KDE bandwidth selector.

This is the compact, standalone version of the Gaussian training notebook used
for the paper.  The target is N(0, 1), the network sees only the sample size n,
and its positive output is multiplied by the unbiased sample standard
deviation.  Expected logarithmic score is evaluated by Gauss--Hermite
quadrature in base-2 units.

The default settings reproduce the reported training protocol.  They are
computationally expensive; ``--quick`` is only a smoke test.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
LOG2 = math.log(2.0)


@dataclass
class Config:
    seed: int = 1234
    n_min: int = 5
    n_max: int = 256
    task_batch: int = 256
    learning_rate: float = 1e-3
    max_steps: int = 10_000
    grad_clip: float = 1.0
    quadrature_points: int = 256
    validation_tasks_per_n: int = 512
    validation_n: tuple[int, ...] = (
        5,
        10,
        15,
        20,
        30,
        50,
        75,
        100,
        150,
        200,
        256,
    )
    validation_every: int = 500
    minimum_steps: int = 3_000
    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4
    print_every: int = 100
    output_dir: str = "training_outputs/gaussian"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GaussianBandwidthNet(nn.Module):
    """Dimensionless bandwidth-ratio network: 1 -> 4 -> 4 -> 1."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.05)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, n_feature: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(n_feature))
        x = F.relu(self.fc2(x))
        return F.softplus(self.fc3(x).squeeze(-1))


def sample_std(samples: torch.Tensor) -> torch.Tensor:
    """Unbiased sample standard deviation, using denominator n - 1."""
    mean = samples.mean(dim=1, keepdim=True)
    variance = (samples - mean).square().sum(dim=1) / (samples.shape[1] - 1)
    return torch.sqrt(variance)


def predict_bandwidth(
    model: GaussianBandwidthNet,
    samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = samples.shape[1]
    n_feature = torch.full(
        (samples.shape[0], 1),
        float(n),
        dtype=samples.dtype,
        device=samples.device,
    )
    ratio = model(n_feature)
    return sample_std(samples) * ratio, ratio


def gauss_hermite_rule(points: int) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    nodes_t = torch.as_tensor(math.sqrt(2.0) * nodes, device=DEVICE)
    weights_t = torch.as_tensor(weights / math.sqrt(math.pi), device=DEVICE)
    return nodes_t, weights_t


def kde_log_density(
    evaluation_points: torch.Tensor,
    samples: torch.Tensor,
    bandwidths: torch.Tensor,
) -> torch.Tensor:
    """Natural-log Gaussian KDE density for a batch of equal-size tasks."""
    n = samples.shape[1]
    z = (
        evaluation_points[:, :, None] - samples[:, None, :]
    ) / bandwidths[:, None, None]
    log_kernel = (
        -0.5 * z.square()
        - torch.log(bandwidths[:, None, None])
        - LOG_SQRT_2PI
    )
    return torch.logsumexp(log_kernel, dim=2) - math.log(float(n))


def gaussian_log_score_bits(
    samples: torch.Tensor,
    bandwidths: torch.Tensor,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Per-task E[-log2 KDE(X)] for X ~ N(0, 1)."""
    evaluation_points = nodes.expand(samples.shape[0], -1)
    log_density = kde_log_density(evaluation_points, samples, bandwidths)
    return -(log_density * weights[None, :]).sum(dim=1) / LOG2


def generate_tasks(batch_size: int, n: int) -> torch.Tensor:
    return torch.randn(batch_size, n, device=DEVICE)


@torch.no_grad()
def build_fixed_validation(config: Config) -> list[tuple[int, torch.Tensor]]:
    return [
        (
            n,
            generate_tasks(config.validation_tasks_per_n, n).detach(),
        )
        for n in config.validation_n
    ]


@torch.no_grad()
def validate(
    model: GaussianBandwidthNet,
    validation_tasks: list[tuple[int, torch.Tensor]],
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    model.eval()
    losses = []
    for _, samples in validation_tasks:
        bandwidths, _ = predict_bandwidth(model, samples)
        losses.append(
            gaussian_log_score_bits(samples, bandwidths, nodes, weights).mean()
        )
    model.train()
    return float(torch.stack(losses).mean().item())


def save_checkpoint(
    path: Path,
    model: GaussianBandwidthNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Config,
    best_validation: float,
    history: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": asdict(config),
            "best_val": float(best_validation),
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
    best_path = output_dir / "gaussian_selector_best.pt"
    last_path = output_dir / "gaussian_selector_last.pt"
    history_path = output_dir / "validation_history.csv"

    model = GaussianBandwidthNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    nodes, weights = gauss_hermite_rule(config.quadrature_points)

    # Built once and then reused at every validation check.
    validation_tasks = build_fixed_validation(config)

    best_validation = float("inf")
    bad_count = 0
    history: list[dict[str, object]] = []
    final_step = 0

    print("Gaussian selector training")
    print(f"device={DEVICE}, dtype={torch.get_default_dtype()}")
    print(
        f"n={config.n_min}..{config.n_max}, tasks/step={config.task_batch}, "
        f"quadrature={config.quadrature_points}, max_steps={config.max_steps}"
    )

    model.train()
    for step in range(1, config.max_steps + 1):
        n = int(np.random.choice(np.arange(config.n_min, config.n_max + 1)))
        samples = generate_tasks(config.task_batch, n)
        bandwidths, ratios = predict_bandwidth(model, samples)
        loss = gaussian_log_score_bits(
            samples,
            bandwidths,
            nodes,
            weights,
        ).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        final_step = step

        if step % config.print_every == 0 or step == 1:
            print(
                f"step={step:6d} n={n:3d} loss_bits={loss.item():.6f} "
                f"mean_ratio={ratios.mean().item():.6f} "
                f"mean_h={bandwidths.mean().item():.6f}"
            )

        if step % config.validation_every != 0:
            continue

        validation = validate(model, validation_tasks, nodes, weights)
        improved = validation < best_validation - config.early_stop_min_delta
        if improved:
            best_validation = validation
            bad_count = 0
        else:
            bad_count += 1

        history.append(
            {
                "step": step,
                "train_log_score_bits": float(loss.item()),
                "validation_log_score_bits": validation,
                "best_validation_log_score_bits": best_validation,
                "improved": improved,
                "bad_count": bad_count,
            }
        )
        save_history(history_path, history)

        if improved:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                step,
                config,
                best_validation,
                history,
            )

        print(
            f"validation step={step:6d} value={validation:.6f} "
            f"best={best_validation:.6f} bad={bad_count}/{config.early_stop_patience}"
        )

        if (
            step >= config.minimum_steps
            and bad_count >= config.early_stop_patience
        ):
            print(f"Early stopping at step {step}.")
            break

    save_checkpoint(
        last_path,
        model,
        optimizer,
        final_step,
        config,
        best_validation,
        history,
    )
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="training_outputs/gaussian")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(output_dir=args.output_dir)
    if args.quick:
        if args.output_dir == "training_outputs/gaussian":
            config.output_dir = "training_outputs/gaussian_quick"
        config.task_batch = 8
        config.max_steps = 2
        config.quadrature_points = 32
        config.validation_tasks_per_n = 4
        config.validation_n = (5, 10)
        config.validation_every = 1
        config.minimum_steps = 2
        config.print_every = 1
        print("QUICK MODE: this checks execution only; it does not reproduce the model.")
    train(config)


if __name__ == "__main__":
    main()
