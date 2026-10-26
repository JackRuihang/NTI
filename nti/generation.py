"""Autoregressive sampling for the NTI incremental backbone contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class IncrementalARModel(Protocol):
    """The only inference API required by NTI generation."""

    def prefill(self, class_labels: Tensor) -> tuple[Tensor, Any]:
        """Return first-token logits `[B,K]` and an initialized cache."""
        ...

    def decode_step(self, last_token: Tensor, cache: Any) -> tuple[Tensor, Any]:
        """Consume `[B]` token IDs and return next-token logits `[B,K]`."""
        ...


@dataclass(frozen=True)
class SamplingConfig:
    num_tokens: int = 256
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = True
    guidance_scale: float = 1.0
    guidance_schedule: Literal["constant", "cosine"] = "cosine"
    guidance_scale_pow: float = 2.75
    unconditional_class_id: int | None = None
    finite_check_interval: int = 32

    def validate(self) -> None:
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0,1]")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be nonnegative")
        if self.guidance_scale_pow <= 0:
            raise ValueError("guidance_scale_pow must be positive")
        if self.finite_check_interval < 0:
            raise ValueError("finite_check_interval must be nonnegative")
        if self.guidance_scale != 1.0 and self.unconditional_class_id is None:
            raise ValueError(
                "CFG requires unconditional_class_id when guidance_scale != 1"
            )


@dataclass(frozen=True)
class GenerationOutput:
    tokens: Tensor
    token_logprobs: Tensor
    guidance_scales: Tensor


def guidance_scale_schedule(
    config: SamplingConfig, device: torch.device | str
) -> Tensor:
    """Return REAR's float32 cosine CFG schedule on the sampling device."""

    if config.num_tokens <= 0 or config.guidance_scale_pow <= 0:
        raise ValueError("guidance schedule length and power must be positive")
    if config.guidance_schedule == "constant":
        return torch.full(
            (config.num_tokens,),
            config.guidance_scale,
            dtype=torch.float32,
            device=device,
        )
    if config.guidance_schedule != "cosine":
        raise ValueError(f"Unknown guidance schedule: {config.guidance_schedule}")
    positions = torch.arange(config.num_tokens, dtype=torch.float32, device=device)
    scale_pow = (
        torch.ones(1, dtype=torch.float32, device=device) * config.guidance_scale_pow
    )
    progress = (
        1.0 - torch.cos(((positions / config.num_tokens) ** scale_pow) * torch.pi)
    ) * 0.5
    return 1.0 + (config.guidance_scale - 1.0) * progress


def guidance_scale_at(position: int, config: SamplingConfig) -> float:
    if not 0 <= position < config.num_tokens:
        raise ValueError("guidance position is outside the token sequence")
    return float(guidance_scale_schedule(config, "cpu")[position].item())


def _validate_logits(
    logits: Tensor,
    expected_batch: int,
    location: str,
    *,
    check_finite: bool,
) -> None:
    if not torch.is_tensor(logits) or logits.ndim != 2:
        shape = (
            tuple(logits.shape) if torch.is_tensor(logits) else type(logits).__name__
        )
        raise RuntimeError(f"{location} must return logits [B,K], got {shape}")
    if logits.shape[0] != expected_batch:
        raise RuntimeError(
            f"{location} returned batch {logits.shape[0]}, expected {expected_batch}"
        )
    if logits.shape[1] <= 1:
        raise RuntimeError(
            f"{location} returned invalid vocabulary size {logits.shape[1]}"
        )
    if check_finite and not torch.isfinite(logits).all():
        raise RuntimeError(f"{location} returned NaN or Inf logits")


def _check_finite_at(position: int, config: SamplingConfig) -> bool:
    interval = config.finite_check_interval
    return interval > 0 and position % interval == 0


def _guided_logits(
    logits: Tensor, batch_size: int, scale: float | Tensor, cfg_enabled: bool
) -> Tensor:
    if not cfg_enabled:
        return logits
    conditional, unconditional = logits[:batch_size], logits[batch_size:]
    return unconditional + scale * (conditional - unconditional)


def filter_logits(logits: Tensor, *, top_k: int | None, top_p: float | None) -> Tensor:
    """Apply top-k and nucleus filtering without mutating the input."""

    filtered = logits.clone()
    vocabulary_size = filtered.shape[-1]
    if top_k is not None and top_k < vocabulary_size:
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, -torch.inf)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits.masked_fill_(remove, -torch.inf)
        filtered = torch.full_like(filtered, -torch.inf).scatter(
            -1, sorted_indices, sorted_logits
        )
    return filtered


def _choose_token(
    logits: Tensor,
    config: SamplingConfig,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    filtered = filter_logits(
        logits / config.temperature, top_k=config.top_k, top_p=config.top_p
    )
    log_probs = torch.log_softmax(filtered, dim=-1)
    if config.do_sample:
        token = torch.multinomial(
            log_probs.exp(), num_samples=1, generator=generator
        ).squeeze(1)
    else:
        token = filtered.argmax(dim=-1)
    selected_logprob = log_probs.gather(1, token[:, None]).squeeze(1)
    return token, selected_logprob


@torch.no_grad()
def autoregressive_sample(
    model: IncrementalARModel,
    class_labels: Tensor,
    *,
    config: SamplingConfig = SamplingConfig(),
    prefix_tokens: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> GenerationOutput:
    """Generate raster-order image tokens with optional classifier-free guidance.

    CFG shares the generated image history between conditional and unconditional
    branches.  Their class labels differ, and both branches are advanced in one
    concatenated `prefill`/`decode_step` call.
    """

    config.validate()
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    if class_labels.ndim != 1:
        raise ValueError(
            f"class_labels must have shape [B], got {tuple(class_labels.shape)}"
        )
    class_labels = class_labels.to(dtype=torch.long)
    batch_size = class_labels.shape[0]
    if batch_size == 0:
        raise ValueError("Cannot generate an empty batch")

    cfg_enabled = config.guidance_scale != 1.0
    if cfg_enabled:
        unconditional = torch.full_like(
            class_labels, int(config.unconditional_class_id)
        )
        model_labels = torch.cat([class_labels, unconditional], dim=0)
    else:
        model_labels = class_labels
    model_batch = model_labels.shape[0]

    logits, cache = model.prefill(model_labels)
    _validate_logits(logits, model_batch, "prefill", check_finite=True)
    vocabulary_size = logits.shape[1]

    if prefix_tokens is None:
        tokens = torch.empty(
            (batch_size, 0), dtype=torch.long, device=class_labels.device
        )
    else:
        if prefix_tokens.ndim != 2 or prefix_tokens.shape[0] != batch_size:
            raise ValueError(
                f"prefix_tokens must have shape [B,P] with B={batch_size}, got {tuple(prefix_tokens.shape)}"
            )
        if prefix_tokens.shape[1] > config.num_tokens:
            raise ValueError("prefix length exceeds num_tokens")
        tokens = prefix_tokens.to(device=class_labels.device, dtype=torch.long)
        if tokens.numel() and (
            int(tokens.min()) < 0 or int(tokens.max()) >= vocabulary_size
        ):
            raise ValueError(f"Prefix token outside [0,{vocabulary_size})")
        # A full-length prefix is already a complete sample. Consuming its last
        # token would ask the cache for position ``num_tokens``, which is one
        # step beyond the model's configured sequence length.
        if tokens.shape[1] < config.num_tokens:
            for position in range(tokens.shape[1]):
                last = tokens[:, position]
                if cfg_enabled:
                    last = torch.cat([last, last], dim=0)
                logits, cache = model.decode_step(last, cache)
                _validate_logits(
                    logits,
                    model_batch,
                    "decode_step",
                    check_finite=_check_finite_at(position + 1, config),
                )

    logprob_columns = [
        torch.full((batch_size,), torch.nan, device=class_labels.device)
        for _ in range(tokens.shape[1])
    ]
    scale_schedule = guidance_scale_schedule(config, class_labels.device)
    if tokens.shape[1]:
        scale_schedule = scale_schedule.clone()
        scale_schedule[: tokens.shape[1]] = torch.nan

    for position in range(tokens.shape[1], config.num_tokens):
        scale = scale_schedule[position]
        sample_logits = _guided_logits(logits, batch_size, scale, cfg_enabled)
        next_token, logprob = _choose_token(sample_logits, config, generator)
        tokens = torch.cat([tokens, next_token[:, None]], dim=1)
        logprob_columns.append(logprob)

        if position + 1 < config.num_tokens:
            consumed = (
                torch.cat([next_token, next_token], dim=0)
                if cfg_enabled
                else next_token
            )
            logits, cache = model.decode_step(consumed, cache)
            _validate_logits(
                logits,
                model_batch,
                "decode_step",
                check_finite=_check_finite_at(position + 1, config),
            )

    token_logprobs = torch.stack(logprob_columns, dim=1)
    result = GenerationOutput(
        tokens=tokens,
        token_logprobs=token_logprobs,
        guidance_scales=scale_schedule,
    )
    if was_training and hasattr(model, "train"):
        model.train()
    return result


@torch.no_grad()
def decode_generation(tokenizer: Any, output: GenerationOutput) -> Tensor:
    """Decode a generation through the frozen tokenizer adapter."""

    if not hasattr(tokenizer, "decode"):
        raise TypeError("tokenizer must expose decode(tokens)")
    return tokenizer.decode(output.tokens)
