"""Frozen-snapshot context and target construction for NTI replay."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from .targets import (
    TVHistoryTargetResult,
    tv_history_conditioned_target,
    hard_token_cross_entropy,
    soft_token_cross_entropy,
    masked_token_mean,
)


@dataclass(frozen=True)
class LinearReplacementSchedule:
    """Linear LARP-style replacement schedule expressed in optimizer steps."""

    start_ratio: float = 0.0
    max_ratio: float = 0.5
    peak_fraction: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_ratio <= self.max_ratio <= 1.0:
            raise ValueError("ratios must satisfy 0 <= start_ratio <= max_ratio <= 1")
        if not 0.0 < self.peak_fraction <= 1.0:
            raise ValueError("peak_fraction must be in (0, 1]")

    def __call__(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
        ramp = min(progress / self.peak_fraction, 1.0)
        return self.start_ratio + (self.max_ratio - self.start_ratio) * ramp


@dataclass(frozen=True)
class RampCosineReplacementSchedule:
    """Linearly ramp to a peak, then cosine-anneal to zero."""

    start_ratio: float = 0.0
    max_ratio: float = 0.5
    peak_fraction: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_ratio <= self.max_ratio <= 1.0:
            raise ValueError("ratios must satisfy 0 <= start_ratio <= max_ratio <= 1")
        if not 0.0 < self.peak_fraction < 1.0:
            raise ValueError("peak_fraction must be in (0, 1) for cosine annealing")

    def __call__(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
        if progress <= self.peak_fraction:
            ramp = progress / self.peak_fraction
            return self.start_ratio + (self.max_ratio - self.start_ratio) * ramp

        decay_progress = (progress - self.peak_fraction) / (1.0 - self.peak_fraction)
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.max_ratio * cosine_weight


@dataclass(frozen=True)
class BellCosineReplacementSchedule:
    """Cosine ramp, hold, and cosine decay with clean boundary phases."""

    max_ratio: float = 0.5
    start_fraction: float = 0.1
    peak_fraction: float = 0.3
    decay_start_fraction: float = 0.5
    end_fraction: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_ratio <= 1.0:
            raise ValueError("max_ratio must be in [0, 1]")
        if not (
            0.0
            <= self.start_fraction
            < self.peak_fraction
            <= self.decay_start_fraction
            < self.end_fraction
            <= 1.0
        ):
            raise ValueError(
                "fractions must satisfy 0 <= start_fraction < peak_fraction "
                "<= decay_start_fraction < end_fraction <= 1"
            )

    def __call__(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
        if progress <= self.start_fraction:
            return 0.0
        if progress < self.peak_fraction:
            ramp_progress = (progress - self.start_fraction) / (
                self.peak_fraction - self.start_fraction
            )
            cosine_weight = 0.5 * (1.0 - math.cos(math.pi * ramp_progress))
            return self.max_ratio * cosine_weight
        if progress <= self.decay_start_fraction:
            return self.max_ratio
        if progress < self.end_fraction:
            decay_progress = (progress - self.decay_start_fraction) / (
                self.end_fraction - self.decay_start_fraction
            )
            cosine_weight = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return self.max_ratio * cosine_weight
        return 0.0


@dataclass(frozen=True)
class TwoPassConfig:
    """Configuration of the clean and altered-history objectives.

    ``clean_loss_weight=branch_loss_weight=1`` intentionally sums the objectives.
    Set both to ``0.5`` for the exact equal-round averaging used by LARP.  Summing
    is the safer default for active-prefix training because an empty active mask
    then leaves the clean objective unchanged instead of halving it.
    """

    target_mode: str = "tv_history_conditioned"
    soft_alpha: float = 0.5
    clean_target_temperature: float = 1.0
    sampling_temperature: float = 1.0
    sampling_top_k: int = 0
    sampling_top_p: float = 1.0
    replacement_ratio_sampling: str = "fixed"
    branch_image_fraction: float = 1.0
    branch_loss_scope: str = "active_prefix"
    clean_loss_weight: float = 1.0
    branch_loss_weight: float = 1.0
    schedule: (
        LinearReplacementSchedule
        | RampCosineReplacementSchedule
        | BellCosineReplacementSchedule
    ) = field(default_factory=LinearReplacementSchedule)

    def __post_init__(self) -> None:
        if self.target_mode != "tv_history_conditioned":
            raise ValueError("NTI requires tv_history_conditioned targets")
        if not 0.0 <= self.soft_alpha <= 1.0:
            raise ValueError("soft_alpha must be in [0, 1]")
        if self.clean_target_temperature <= 0.0:
            raise ValueError("clean_target_temperature must be positive")
        if self.sampling_temperature <= 0.0:
            raise ValueError("sampling_temperature must be positive")
        if self.sampling_top_k < 0:
            raise ValueError("sampling_top_k must be non-negative")
        if not 0.0 < self.sampling_top_p <= 1.0:
            raise ValueError("sampling_top_p must be in (0, 1]")
        if self.replacement_ratio_sampling not in {"fixed", "per_image_uniform"}:
            raise ValueError(
                "replacement_ratio_sampling must be 'fixed' or 'per_image_uniform'"
            )
        if not 0.0 <= self.branch_image_fraction <= 1.0:
            raise ValueError("branch_image_fraction must be in [0, 1]")
        if self.branch_loss_scope not in {"all", "active_prefix"}:
            raise ValueError("branch_loss_scope must be 'all' or 'active_prefix'")
        if (
            self.target_mode == "tv_history_conditioned"
            and self.branch_loss_scope != "active_prefix"
        ):
            raise ValueError(
                "history-gated targets require branch_loss_scope='active_prefix'"
            )
        if self.clean_loss_weight < 0.0 or self.branch_loss_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.clean_loss_weight == 0.0 and self.branch_loss_weight == 0.0:
            raise ValueError("at least one loss weight must be positive")
        if self.clean_loss_weight == 0.0 and self.branch_image_fraction == 0.0:
            raise ValueError(
                "branch_image_fraction=0 requires a positive clean_loss_weight"
            )


@dataclass
class TwoPassOutput:
    """Losses, branch construction, and detached monitoring values."""

    loss: torch.Tensor
    clean_loss: torch.Tensor
    branch_loss: torch.Tensor
    clean_logits: torch.Tensor
    branch_logits: torch.Tensor
    branch_target: Optional[torch.Tensor]
    branch_indices: torch.Tensor
    sampled_tokens: torch.Tensor
    mixed_tokens: torch.Tensor
    replacement_mask: torch.Tensor
    changed_mask: torch.Tensor
    active_prefix_mask: torch.Tensor
    branch_loss_mask: Optional[torch.Tensor]
    replacement_ratio: float
    branch_image_fraction: float
    metrics: Dict[str, torch.Tensor]


def exclusive_active_prefix_mask(changed_mask: torch.Tensor) -> torch.Tensor:
    """Mark target ``t`` active iff some mixed-history change occurred at ``s<t``."""

    if changed_mask.ndim != 2:
        raise ValueError("changed_mask must have shape [batch, tokens]")
    changed = changed_mask.bool()
    active = torch.zeros_like(changed)
    if changed.shape[1] > 1:
        active[:, 1:] = changed[:, :-1].long().cumsum(dim=1) > 0
    return active


def sample_tokens_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample detached clean logits with inference-style top-k/top-p filtering."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocabulary]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    with torch.no_grad():
        filtered_logits = logits.detach().float() / temperature
        vocabulary = filtered_logits.shape[-1]
        if top_k > 0:
            retained = min(int(top_k), vocabulary)
            threshold = torch.topk(filtered_logits, retained, dim=-1).values[
                ..., -1, None
            ]
            filtered_logits = filtered_logits.masked_fill(
                filtered_logits < threshold, -torch.inf
            )
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                filtered_logits, descending=True, dim=-1
            )
            cumulative_probability = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove_sorted = cumulative_probability > top_p
            remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
            remove_sorted[..., 0] = False
            remove = torch.zeros_like(remove_sorted).scatter(
                dim=-1, index=sorted_indices, src=remove_sorted
            )
            filtered_logits = filtered_logits.masked_fill(remove, -torch.inf)

        probability = torch.softmax(filtered_logits, dim=-1)
        sampled = torch.multinomial(
            probability.reshape(-1, probability.shape[-1]),
            num_samples=1,
            replacement=True,
            generator=generator,
        )
    return sampled.reshape(logits.shape[:2]).detach()


def _extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        logits = model_output
    elif isinstance(model_output, Mapping) and "logits" in model_output:
        logits = model_output["logits"]
    elif hasattr(model_output, "logits"):
        logits = model_output.logits
    elif isinstance(model_output, (tuple, list)) and model_output:
        logits = model_output[0]
    else:
        raise TypeError("model output must be logits or contain a 'logits' field")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("model logits must have shape [batch, tokens, vocabulary]")
    return logits


def _resolve_replacement_ratio(
    config: TwoPassConfig,
    replacement_ratio: Optional[float],
    step: Optional[int],
    total_steps: Optional[int],
) -> float:
    if replacement_ratio is None:
        if step is None or total_steps is None:
            raise ValueError("provide replacement_ratio or both step and total_steps")
        ratio = config.schedule(step, total_steps)
    else:
        ratio = float(replacement_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("replacement_ratio must be in [0, 1]")
    return ratio


def _branch_image_count(batch_size: int, fraction: float) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("branch_image_fraction must be in [0, 1]")
    if fraction == 0.0:
        return 0
    if fraction == 1.0:
        return batch_size
    return min(batch_size, max(1, math.floor(batch_size * fraction + 0.5)))


def _sample_branch_indices(
    batch_size: int,
    fraction: float,
    *,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    count = _branch_image_count(batch_size, fraction)
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == batch_size:
        # Preserve the legacy full-branch RNG stream when fraction is one.
        return torch.arange(batch_size, device=device, dtype=torch.long)
    return (
        torch.randperm(batch_size, device=device, generator=generator)[:count]
        .sort()
        .values
    )


def _subset_batch_kwargs(
    kwargs: Mapping[str, Any],
    indices: torch.Tensor,
    batch_size: int,
) -> Dict[str, Any]:
    subset: Dict[str, Any] = {}
    for name, value in kwargs.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == batch_size
        ):
            subset[name] = value.index_select(0, indices)
        else:
            subset[name] = value
    return subset


def _sample_replacement_mask(
    image_tokens: torch.Tensor,
    ratio: float,
    eligible_mask: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
    ratio_sampling: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eligible_mask is None:
        eligible = torch.ones_like(image_tokens, dtype=torch.bool)
    else:
        if eligible_mask.shape != image_tokens.shape:
            raise ValueError("eligible_mask must match image_tokens")
        eligible = eligible_mask.detach().to(
            device=image_tokens.device, dtype=torch.bool
        )
    batch_size = image_tokens.shape[0]
    if ratio_sampling == "fixed":
        probabilities = torch.full(
            (batch_size,),
            ratio,
            device=image_tokens.device,
            dtype=torch.float32,
        )
        # Preserve the legacy fixed-ratio RNG stream exactly.
        if ratio == 0.0:
            return torch.zeros_like(eligible), probabilities
        if ratio == 1.0:
            return eligible, probabilities
        random_value = torch.rand(
            image_tokens.shape, device=image_tokens.device, generator=generator
        )
        return (random_value < ratio) & eligible, probabilities
    if ratio_sampling != "per_image_uniform":
        raise ValueError("ratio_sampling must be 'fixed' or 'per_image_uniform'")
    if ratio == 0.0:
        probabilities = torch.zeros(
            batch_size, device=image_tokens.device, dtype=torch.float32
        )
        return torch.zeros_like(eligible), probabilities
    probabilities = (
        torch.rand((batch_size,), device=image_tokens.device, generator=generator)
        * ratio
    )
    random_value = torch.rand(
        image_tokens.shape, device=image_tokens.device, generator=generator
    )
    return (random_value < probabilities[:, None]) & eligible, probabilities


def _masked_accuracy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if target_ids.numel() == 0:
        return logits.detach().new_zeros(())
    correct = logits.detach().argmax(dim=-1).eq(target_ids).float()
    return masked_token_mean(correct, mask).detach()


def _masked_quantile(
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    quantile: float,
) -> torch.Tensor:
    """Return a detached monitoring quantile, or zero for an empty mask."""

    if value.ndim != 2:
        raise ValueError("value must have shape [batch, tokens]")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if mask is None:
        selected = value.reshape(-1)
    else:
        if mask.shape != value.shape:
            raise ValueError("mask must match value [batch, tokens]")
        selected = value[mask.detach().bool()]
    if selected.numel() == 0:
        return value.detach().sum() * 0.0
    return torch.quantile(selected.detach().float(), quantile)


def _tv_history_target_metrics(
    result: TVHistoryTargetResult,
    loss_mask: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        "target_tv_distance": masked_token_mean(result.tv_distance, loss_mask),
        "target_tv_overlap": masked_token_mean(result.overlap, loss_mask),
        "target_tv_overlap_q10": _masked_quantile(result.overlap, loss_mask, 0.10),
        "target_tv_overlap_q50": _masked_quantile(result.overlap, loss_mask, 0.50),
        "target_tv_overlap_q90": _masked_quantile(result.overlap, loss_mask, 0.90),
        "target_clean_confidence": masked_token_mean(
            result.clean_confidence, loss_mask
        ),
        "target_clean_gt_probability": masked_token_mean(
            result.clean_gt_probability, loss_mask
        ),
        "target_non_gt_dose": masked_token_mean(result.non_gt_dose, loss_mask),
        "target_non_gt_dose_q10": _masked_quantile(result.non_gt_dose, loss_mask, 0.10),
        "target_non_gt_dose_q50": _masked_quantile(result.non_gt_dose, loss_mask, 0.50),
        "target_non_gt_dose_q90": _masked_quantile(result.non_gt_dose, loss_mask, 0.90),
    }


def two_pass_forward(
    model: nn.Module,
    image_tokens: torch.Tensor,
    class_labels: torch.Tensor,
    *,
    config: Optional[TwoPassConfig] = None,
    codebook_kernel: Optional[torch.Tensor] = None,
    codebook_affinity: Optional[torch.Tensor] = None,
    replacement_ratio: Optional[float] = None,
    step: Optional[int] = None,
    total_steps: Optional[int] = None,
    eligible_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    model_kwargs: Optional[Mapping[str, Any]] = None,
) -> TwoPassOutput:
    """Run two complete, parallel causal forwards through one shared AR model.

    Contract: ``model(image_tokens, class_labels=...)`` receives complete image
    tokens ``[B,N]``.  The model itself constructs ``[class/BOS,z0,...,zN-2]``
    and returns ``[B,N,V]`` logits where position ``t`` predicts ``z_t``.  This
    function never externally shifts the image sequence and never performs a
    recursive rollout.

    The clean pass always uses hard GT cross entropy over the full batch.  A
    fixed-size, uniformly sampled image subset enters the altered-history pass;
    its detached clean logits are sampled and mixed into complete histories.
    The branch loss is averaged within that subset, without fraction scaling.
    Calling code owns the single backward/optimizer step on ``output.loss``.
    """

    config = config or TwoPassConfig()
    if image_tokens.ndim != 2:
        raise ValueError("image_tokens must have shape [batch, tokens]")
    if image_tokens.is_floating_point():
        raise TypeError("image_tokens must contain integer token IDs")
    if class_labels.ndim != 1 or class_labels.shape[0] != image_tokens.shape[0]:
        raise ValueError("class_labels must have shape [batch]")

    kwargs = dict(model_kwargs or {})
    if "class_labels" in kwargs:
        raise ValueError("class_labels must be passed through the named argument")
    if "class_drop_mask" not in kwargs:
        candidate = model.module if hasattr(model, "module") else model
        candidate = getattr(candidate, "_orig_mod", candidate)
        model_config = getattr(candidate, "config", None)
        class_dropout = float(getattr(model_config, "class_dropout", 0.0))
        if model.training and class_dropout > 0.0:
            kwargs["class_drop_mask"] = (
                torch.rand(class_labels.shape, device=class_labels.device)
                < class_dropout
            ).detach()
    ratio = _resolve_replacement_ratio(config, replacement_ratio, step, total_steps)

    clean_logits = _extract_logits(
        model(image_tokens, class_labels=class_labels, **kwargs)
    )
    if clean_logits.shape[:2] != image_tokens.shape:
        raise ValueError("clean logits must align with every complete image token")
    clean_loss = hard_token_cross_entropy(clean_logits, image_tokens)

    batch_size = image_tokens.shape[0]
    branch_indices = _sample_branch_indices(
        batch_size,
        config.branch_image_fraction,
        device=image_tokens.device,
        generator=generator,
    ).detach()
    branch_image_tokens = image_tokens.index_select(0, branch_indices)
    branch_class_labels = class_labels.index_select(0, branch_indices)
    branch_clean_logits = clean_logits.index_select(0, branch_indices)
    branch_kwargs = _subset_batch_kwargs(kwargs, branch_indices, batch_size)
    branch_eligible_mask: Optional[torch.Tensor]
    if eligible_mask is None:
        branch_eligible_mask = None
    else:
        if eligible_mask.shape != image_tokens.shape:
            raise ValueError("eligible_mask must match image_tokens")
        branch_eligible_mask = eligible_mask.detach().index_select(0, branch_indices)

    if branch_indices.numel() == 0:
        sampled_tokens = torch.empty_like(branch_image_tokens)
        replacement_mask = torch.zeros_like(branch_image_tokens, dtype=torch.bool)
        replacement_probabilities = torch.empty(
            0, device=image_tokens.device, dtype=torch.float32
        )
        mixed_tokens = branch_image_tokens.detach().clone()
    else:
        sampled_tokens = sample_tokens_from_logits(
            branch_clean_logits,
            temperature=config.sampling_temperature,
            top_k=config.sampling_top_k,
            top_p=config.sampling_top_p,
            generator=generator,
        )
        replacement_mask, replacement_probabilities = _sample_replacement_mask(
            branch_image_tokens,
            ratio,
            branch_eligible_mask,
            generator,
            config.replacement_ratio_sampling,
        )
        replacement_mask = replacement_mask.detach()
        replacement_probabilities = replacement_probabilities.detach()
        mixed_tokens = torch.where(
            replacement_mask, sampled_tokens, branch_image_tokens.detach()
        ).detach()
    changed_mask = (replacement_mask & mixed_tokens.ne(branch_image_tokens)).detach()
    active_prefix_mask = exclusive_active_prefix_mask(changed_mask).detach()

    if branch_indices.numel() == 0:
        branch_logits = branch_clean_logits
    else:
        branch_logits = _extract_logits(
            model(
                mixed_tokens,
                class_labels=branch_class_labels,
                **branch_kwargs,
            )
        )
    if branch_logits.shape != branch_clean_logits.shape:
        raise ValueError("clean and branch logits must have the same shape")

    branch_loss_mask: Optional[torch.Tensor]
    if config.branch_loss_scope == "active_prefix":
        branch_loss_mask = active_prefix_mask
    else:
        branch_loss_mask = None

    branch_target: Optional[torch.Tensor]
    target_metrics: Dict[str, torch.Tensor] = {}
    if config.target_mode == "tv_history_conditioned":
        if codebook_kernel is None:
            raise ValueError(
                "tv_history_conditioned target_mode requires codebook_kernel"
            )
        result = tv_history_conditioned_target(
            branch_clean_logits,
            branch_logits,
            branch_image_tokens,
            codebook_kernel,
            clean_temperature=config.clean_target_temperature,
        )
        branch_target = result.target
        if branch_indices.numel() == 0:
            branch_loss = clean_logits.sum() * 0.0
        else:
            branch_loss = soft_token_cross_entropy(
                branch_logits, branch_target, loss_mask=branch_loss_mask
            )
        target_metrics = _tv_history_target_metrics(result, branch_loss_mask)
    else:  # TwoPassConfig validates target_mode.
        raise AssertionError(f"unsupported target_mode {config.target_mode!r}")

    loss = (
        config.clean_loss_weight * clean_loss + config.branch_loss_weight * branch_loss
    )

    active_count = (
        active_prefix_mask.sum()
        if branch_loss_mask is not None
        else image_tokens.new_tensor(branch_image_tokens.numel())
    )
    actual_branch_fraction = branch_indices.numel() / batch_size
    with torch.no_grad():
        metrics = {
            "loss": loss.detach(),
            "clean_loss": clean_loss.detach(),
            "branch_loss": branch_loss.detach(),
            "replacement_ratio": torch.tensor(
                ratio, device=clean_logits.device, dtype=torch.float32
            ),
            "replacement_sampled_probability_mean": (
                replacement_probabilities.mean()
                if replacement_probabilities.numel()
                else clean_logits.new_zeros((), dtype=torch.float32)
            ),
            "branch_image_fraction_requested": torch.tensor(
                config.branch_image_fraction,
                device=clean_logits.device,
                dtype=torch.float32,
            ),
            "branch_image_fraction": torch.tensor(
                actual_branch_fraction,
                device=clean_logits.device,
                dtype=torch.float32,
            ),
            "branch_image_count": image_tokens.new_tensor(branch_indices.numel()),
            "replacement_selected_fraction": masked_token_mean(
                replacement_mask.float(), None
            ),
            "history_changed_fraction": masked_token_mean(changed_mask.float(), None),
            "active_prefix_fraction": masked_token_mean(
                active_prefix_mask.float(), None
            ),
            "replacement_selected_count": replacement_mask.sum().detach(),
            "history_changed_count": changed_mask.sum().detach(),
            "branch_active_count": active_count.detach(),
            "sampled_gt_fraction": masked_token_mean(
                sampled_tokens.eq(branch_image_tokens).float(), None
            ),
            "clean_top1": _masked_accuracy(clean_logits, image_tokens, None),
            "branch_top1": _masked_accuracy(
                branch_logits, branch_image_tokens, branch_loss_mask
            ),
        }
        if branch_target is not None:
            clean_probability = torch.softmax(
                branch_clean_logits.detach().float(), dim=-1
            )
            target_entropy = -(
                branch_target * branch_target.clamp_min(1e-12).log()
            ).sum(dim=-1)
            metrics["branch_target_entropy"] = masked_token_mean(
                target_entropy, branch_loss_mask
            ).detach()
            target_clean_kl = (
                branch_target
                * (
                    branch_target.clamp_min(1e-12).log()
                    - clean_probability.clamp_min(1e-12).log()
                )
            ).sum(dim=-1)
            metrics["branch_target_clean_kl"] = masked_token_mean(
                target_clean_kl, branch_loss_mask
            ).detach()
            metrics["branch_target_clean_l1"] = masked_token_mean(
                (branch_target - clean_probability).abs().sum(dim=-1),
                branch_loss_mask,
            ).detach()
            gt_mass = branch_target.gather(
                dim=-1, index=branch_image_tokens[..., None]
            ).squeeze(-1)
            metrics["branch_target_gt_mass"] = masked_token_mean(
                gt_mass, branch_loss_mask
            ).detach()
            metrics["branch_target_non_gt_mass"] = masked_token_mean(
                1.0 - gt_mass, branch_loss_mask
            ).detach()
        metrics.update({name: value.detach() for name, value in target_metrics.items()})

    return TwoPassOutput(
        loss=loss,
        clean_loss=clean_loss,
        branch_loss=branch_loss,
        clean_logits=clean_logits,
        branch_logits=branch_logits,
        branch_target=branch_target,
        branch_indices=branch_indices,
        sampled_tokens=sampled_tokens,
        mixed_tokens=mixed_tokens,
        replacement_mask=replacement_mask,
        changed_mask=changed_mask,
        active_prefix_mask=active_prefix_mask,
        branch_loss_mask=branch_loss_mask,
        replacement_ratio=ratio,
        branch_image_fraction=actual_branch_fraction,
        metrics=metrics,
    )
