from __future__ import annotations

import math
from typing import Dict, List

from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from nti.config import TrainingConfig


def parameter_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, object]]:
    decay: List[Tensor] = []
    no_decay: List[Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            parameter.ndim < 2
            or name.endswith("bias")
            or "embedding" in name
            or name.endswith("raw_transport")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    return AdamW(
        parameter_groups(model, config.weight_decay),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )


def _learning_rate_multiplier(config: TrainingConfig, step: int) -> float:
    if step < 0:
        raise ValueError("scheduler step must be nonnegative")
    end_ratio = config.end_learning_rate / config.learning_rate
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return max(step, 1) / config.warmup_steps
    decay_steps = max(config.max_steps - config.warmup_steps, 1)
    progress = min(max((step - config.warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end_ratio + (1.0 - end_ratio) * cosine


def learning_rate_at_step(config: TrainingConfig, step: int) -> float:
    return config.learning_rate * _learning_rate_multiplier(config, step)


def build_lr_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
    *,
    last_epoch: int = -1,
) -> LambdaLR:
    def multiplier(step: int) -> float:
        return _learning_rate_multiplier(config, step)

    return LambdaLR(optimizer, multiplier, last_epoch=last_epoch)


def build_continuation_lr_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
    *,
    resume_step: int,
    expected_resume_lr: float,
) -> LambdaLR:
    """Re-anchor a migrated cosine schedule without loading its old state."""

    if resume_step <= 0:
        raise ValueError("continuation resume_step must be positive")
    if config.warmup_steps != resume_step:
        raise ValueError(
            "continuation config warmup_steps must equal resume_step: "
            f"warmup_steps={config.warmup_steps}, resume_step={resume_step}"
        )
    if not math.isclose(
        config.learning_rate,
        expected_resume_lr,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError(
            "continuation config learning_rate must equal the resume LR: "
            f"learning_rate={config.learning_rate}, expected={expected_resume_lr}"
        )

    for index, group in enumerate(optimizer.param_groups):
        current_lr = float(group["lr"])
        if not math.isfinite(current_lr) or not math.isclose(
            current_lr,
            expected_resume_lr,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "checkpoint optimizer LR is not continuous at the continuation "
                f"boundary for parameter group {index}: "
                f"checkpoint={current_lr}, expected={expected_resume_lr}"
            )
        group["initial_lr"] = config.learning_rate

    scheduler = build_lr_scheduler(
        optimizer,
        config,
        last_epoch=resume_step - 1,
    )
    expected_lrs = [config.learning_rate] * len(optimizer.param_groups)
    if scheduler.last_epoch != resume_step:
        raise RuntimeError(
            "continuation scheduler did not initialize at the resume step: "
            f"last_epoch={scheduler.last_epoch}, expected={resume_step}"
        )
    for label, values in (
        ("base_lrs", scheduler.base_lrs),
        ("last_lrs", scheduler.get_last_lr()),
        ("optimizer_lrs", [group["lr"] for group in optimizer.param_groups]),
    ):
        if len(values) != len(expected_lrs) or any(
            not math.isfinite(float(value))
            or not math.isclose(
                float(value),
                expected,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            )
            for value, expected in zip(values, expected_lrs)
        ):
            raise RuntimeError(
                f"continuation scheduler {label} are not anchored at "
                f"{config.learning_rate}: {list(values)}"
            )
    return scheduler
