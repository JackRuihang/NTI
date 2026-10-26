"""Low-frequency, training-safe gradient monitoring helpers.

The helpers in this module are deliberately independent of the optimizer and
gradient clipping implementation.  A trainer can collect the pre-clip metrics
after AMP unscaling, call its existing clipping function, and then collect the
small set of reAR-style post-clip metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Optional, Sequence

import torch
from torch import nn


_WRAPPER_PREFIXES = ("module.", "_orig_mod.")
_BLOCK_PATTERN = re.compile(r"^blocks\.(\d+)\.")
_POSITION_PARAMETERS = {
    "position_embedding",
    "target_aware_position_embedding",
    "timestep_embedding",
}


def canonical_parameter_name(name: str) -> str:
    """Strip repeated DDP/``torch.compile`` prefixes from a parameter name."""

    if not isinstance(name, str) or not name:
        raise ValueError("parameter name must be a non-empty string")
    canonical = name
    stripped = True
    while stripped:
        stripped = False
        for prefix in _WRAPPER_PREFIXES:
            if canonical.startswith(prefix):
                canonical = canonical[len(prefix) :]
                stripped = True
                break
    if not canonical:
        raise ValueError("parameter name contains only wrapper prefixes")
    return canonical


def _materialize_named_parameters(
    model: nn.Module,
) -> tuple[tuple[str, str, nn.Parameter], ...]:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    return tuple(
        (name, canonical_parameter_name(name), parameter)
        for name, parameter in model.named_parameters()
    )


def critical_block_indices(model: nn.Module) -> tuple[int, ...]:
    """Return the first, middle, and last block indices exposed by ``model``."""

    indices = sorted(
        {
            int(match.group(1))
            for _, canonical, _ in _materialize_named_parameters(model)
            if (match := _BLOCK_PATTERN.match(canonical)) is not None
        }
    )
    if not indices:
        return ()
    return tuple(dict.fromkeys((indices[0], indices[len(indices) // 2], indices[-1])))


def _validate_block_indices(block_indices: Sequence[int]) -> tuple[int, ...]:
    checked: list[int] = []
    for index in block_indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("block indices must be non-negative integers")
        if index not in checked:
            checked.append(index)
    return tuple(checked)


def critical_parameter_group(
    name: str,
    *,
    block_indices: Sequence[int],
) -> Optional[str]:
    """Map a canonical or wrapped parameter name to a compact monitoring group."""

    canonical = canonical_parameter_name(name)
    selected_blocks = set(_validate_block_indices(block_indices))
    if canonical.startswith("token_embedding."):
        return "input/token_embedding"
    if canonical == "cls_token" or canonical.startswith("class_embedding."):
        return "input/conditioning"
    if canonical in _POSITION_PARAMETERS:
        return "input/positioning"
    if canonical.startswith("icb."):
        return "interface/icb"

    block_match = _BLOCK_PATTERN.match(canonical)
    if block_match is not None:
        block_index = int(block_match.group(1))
        if block_index in selected_blocks:
            return f"backbone/block_{block_index:02d}"
        return None
    if canonical.startswith("final."):
        return "output/final"
    if canonical.startswith("lm_head."):
        return "output/lm_head"
    return None


@dataclass
class _GradientAccumulator:
    gradient_square_sum: float = 0.0
    parameter_square_sum: float = 0.0
    element_count: int = 0
    parameter_count: int = 0
    nonfinite_parameter_count: int = 0

    def add(self, parameter: nn.Parameter) -> None:
        gradient = parameter.grad
        if gradient is None:
            return
        gradient_values = (
            gradient.detach().coalesce().values()
            if gradient.is_sparse
            else gradient.detach()
        )
        gradient_norm = float(torch.linalg.vector_norm(gradient_values.float()).item())
        parameter_norm = float(
            torch.linalg.vector_norm(parameter.detach().float()).item()
        )
        self.element_count += parameter.numel()
        self.parameter_count += 1
        if not math.isfinite(gradient_norm) or not math.isfinite(parameter_norm):
            self.nonfinite_parameter_count += 1
            return
        self.gradient_square_sum += gradient_norm * gradient_norm
        self.parameter_square_sum += parameter_norm * parameter_norm

    @property
    def has_gradient(self) -> bool:
        return self.parameter_count > 0

    @property
    def is_finite(self) -> bool:
        return self.nonfinite_parameter_count == 0

    @property
    def gradient_l2(self) -> float:
        return math.sqrt(self.gradient_square_sum)

    @property
    def parameter_l2(self) -> float:
        return math.sqrt(self.parameter_square_sum)


def compute_clip_coefficient(
    global_gradient_l2: float,
    max_gradient_l2: float,
    *,
    eps: float = 1.0e-6,
) -> float:
    """Return PyTorch-style global norm clipping coefficient in ``[0, 1]``."""

    values = (global_gradient_l2, max_gradient_l2, eps)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("clip coefficient inputs must be finite")
    if global_gradient_l2 < 0.0 or max_gradient_l2 <= 0.0 or eps <= 0.0:
        raise ValueError(
            "global norm must be non-negative and max norm/eps must be positive"
        )
    return min(1.0, float(max_gradient_l2) / (float(global_gradient_l2) + eps))


def _accumulator_metrics(
    accumulator: _GradientAccumulator,
    *,
    group_name: str,
    ratio_eps: float,
) -> dict[str, float]:
    if not accumulator.has_gradient:
        return {}
    metrics = {
        f"grad/preclip_finite/{group_name}": float(accumulator.is_finite),
        f"grad/nonfinite_parameter_count/{group_name}": float(
            accumulator.nonfinite_parameter_count
        ),
    }
    if not accumulator.is_finite:
        return metrics
    gradient_l2 = accumulator.gradient_l2
    metrics.update(
        {
            f"grad/preclip_l2/{group_name}": gradient_l2,
            f"grad/preclip_rms/{group_name}": gradient_l2
            / math.sqrt(float(accumulator.element_count)),
            f"grad/grad_to_param/{group_name}": gradient_l2
            / (accumulator.parameter_l2 + ratio_eps),
        }
    )
    return metrics


def collect_preclip_gradient_metrics(
    model: nn.Module,
    *,
    max_grad_norm: float,
    block_indices: Optional[Sequence[int]] = None,
    ratio_eps: float = 1.0e-12,
    clip_eps: float = 1.0e-6,
) -> dict[str, float]:
    r"""Collect global and critical-group statistics before gradient clipping.

    For a parameter group :math:`S`, this reports
    :math:`\|g_S\|_2`, :math:`\|g_S\|_2/\sqrt{|S|}`, and
    :math:`\|g_S\|_2/(\|\theta_S\|_2+\epsilon)`.  Parameters without a gradient
    are omitted from both the numerator and element count.
    """

    if not math.isfinite(float(max_grad_norm)) or max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be finite and positive")
    if not math.isfinite(float(ratio_eps)) or ratio_eps <= 0.0:
        raise ValueError("ratio_eps must be finite and positive")
    if not math.isfinite(float(clip_eps)) or clip_eps <= 0.0:
        raise ValueError("clip_eps must be finite and positive")

    named = _materialize_named_parameters(model)
    selected_blocks = (
        critical_block_indices(model)
        if block_indices is None
        else _validate_block_indices(block_indices)
    )
    global_accumulator = _GradientAccumulator()
    groups: dict[str, _GradientAccumulator] = {}
    for original_name, _, parameter in named:
        if parameter.grad is None:
            continue
        global_accumulator.add(parameter)
        group_name = critical_parameter_group(
            original_name, block_indices=selected_blocks
        )
        if group_name is not None:
            groups.setdefault(group_name, _GradientAccumulator()).add(parameter)

    if not global_accumulator.has_gradient:
        return {}
    metrics = _accumulator_metrics(
        global_accumulator,
        group_name="global",
        ratio_eps=ratio_eps,
    )
    for group_name, accumulator in sorted(groups.items()):
        metrics.update(
            _accumulator_metrics(
                accumulator,
                group_name=group_name,
                ratio_eps=ratio_eps,
            )
        )
    if global_accumulator.is_finite:
        metrics["grad/clip_coefficient"] = compute_clip_coefficient(
            global_accumulator.gradient_l2,
            max_grad_norm,
            eps=clip_eps,
        )
    return metrics


def selected_rear_style_parameter_names(
    model: nn.Module,
    *,
    block_indices: Optional[Sequence[int]] = None,
) -> tuple[str, ...]:
    """Select a compact, stable subset of parameters for reAR-style logging."""

    canonical_names = {
        canonical for _, canonical, _ in _materialize_named_parameters(model)
    }
    selected_blocks = (
        critical_block_indices(model)
        if block_indices is None
        else _validate_block_indices(block_indices)
    )
    candidates = [
        "token_embedding.weight",
        "class_embedding.weight",
        "cls_token",
        "position_embedding",
        "target_aware_position_embedding",
        "timestep_embedding",
    ]
    for index in selected_blocks:
        candidates.extend(
            (
                f"blocks.{index}.attention.qkv.weight",
                f"blocks.{index}.mlp.0.weight",
                f"blocks.{index}.modulation.1.weight",
            )
        )
    candidates.extend(("final.modulation.1.weight", "lm_head.weight"))
    return tuple(name for name in candidates if name in canonical_names)


def collect_rear_style_gradient_metrics(
    model: nn.Module,
    *,
    parameter_names: Optional[Iterable[str]] = None,
    block_indices: Optional[Sequence[int]] = None,
) -> dict[str, float]:
    r"""Collect post-clip reAR-style :math:`\|g\|_2/\mathrm{numel}` metrics.

    This normalization is included only for qualitative comparison with the
    official reAR TensorBoard.  It is parameter-size dependent and is not a
    replacement for the pre-clip group RMS metrics.
    """

    named = _materialize_named_parameters(model)
    selected = (
        set(selected_rear_style_parameter_names(model, block_indices=block_indices))
        if parameter_names is None
        else {canonical_parameter_name(name) for name in parameter_names}
    )
    metrics: dict[str, float] = {}
    nonfinite_count = 0
    for _, canonical, parameter in named:
        if canonical not in selected or parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        gradient_values = (
            gradient.coalesce().values() if gradient.is_sparse else gradient
        )
        gradient_norm = float(torch.linalg.vector_norm(gradient_values.float()).item())
        if not math.isfinite(gradient_norm):
            nonfinite_count += 1
            continue
        metrics[f"grad/postclip_rear_style/{canonical}"] = gradient_norm / float(
            parameter.numel()
        )
    if nonfinite_count:
        metrics["grad/postclip_rear_style/nonfinite_parameter_count"] = float(
            nonfinite_count
        )
    return metrics


__all__ = [
    "canonical_parameter_name",
    "collect_preclip_gradient_metrics",
    "collect_rear_style_gradient_metrics",
    "compute_clip_coefficient",
    "critical_block_indices",
    "critical_parameter_group",
    "selected_rear_style_parameter_names",
]
