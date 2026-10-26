"""Strict, frozen adapter for the MaskGIT VQGAN used by reAR.

This module deliberately does not contain a replacement VQGAN.  In production it
imports the exact Encoder, Decoder, and VectorQuantizer classes from a reAR source
checkout, builds that architecture, and then loads the supplied checkpoint.
Factory injection exists only to make the adapter contract testable without a
multi-hundred-megabyte checkpoint.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MaskGITVQGANSpec:
    """Architecture contract fixed by the ImageNet-f16-256 checkpoint."""

    codebook_size: int = 1024
    codebook_dim: int = 256
    num_tokens: int = 256
    image_size: int = 256
    image_channels: int = 3

    @property
    def latent_side(self) -> int:
        side = int(self.num_tokens**0.5)
        if side * side != self.num_tokens:
            raise ValueError(
                f"num_tokens must be square for MaskGIT VQGAN, got {self.num_tokens}"
            )
        return side


@dataclass(frozen=True)
class CheckpointLoadReport:
    checkpoint_path: str
    payload_path: str
    loaded_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _is_tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and any(
        torch.is_tensor(item) for item in value.values()
    )


def _state_dict_candidates(
    payload: Any, path: str = "root"
) -> Iterator[tuple[str, Mapping[str, Tensor]]]:
    if _is_tensor_mapping(payload):
        yield (
            path,
            {
                str(key): value
                for key, value in payload.items()
                if torch.is_tensor(value)
            },
        )
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, Mapping):
                yield from _state_dict_candidates(value, f"{path}.{key}")


def _read_checkpoint(path: Path) -> Any:
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - depends on training environment
            raise RuntimeError(
                "Loading .safetensors requires the safetensors package"
            ) from exc
        return load_file(str(path), device="cpu")

    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def _map_checkpoint_keys(
    checkpoint_state: Mapping[str, Tensor],
    model_state: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], tuple[str, ...]]:
    """Map arbitrary wrapper prefixes without renaming internal architecture keys."""

    mapped: dict[str, Tensor] = {}
    unexpected: list[str] = []
    model_keys = tuple(model_state)

    for checkpoint_key, tensor in checkpoint_state.items():
        if checkpoint_key in model_state:
            candidates = [checkpoint_key]
        else:
            candidates = [
                key for key in model_keys if checkpoint_key.endswith("." + key)
            ]

        shape_matches = [
            key for key in candidates if model_state[key].shape == tensor.shape
        ]
        if len(shape_matches) == 1:
            target_key = shape_matches[0]
            if target_key in mapped:
                raise RuntimeError(
                    f"Checkpoint keys collide after prefix removal: {checkpoint_key!r} -> {target_key!r}"
                )
            mapped[target_key] = tensor
        elif len(shape_matches) > 1:
            raise RuntimeError(
                f"Ambiguous checkpoint suffix {checkpoint_key!r}; matches {shape_matches}"
            )
        else:
            unexpected.append(checkpoint_key)

    return mapped, tuple(sorted(unexpected))


def _select_checkpoint_state(
    payload: Any,
    model_state: Mapping[str, Tensor],
) -> tuple[str, dict[str, Tensor], tuple[str, ...]]:
    scored: list[tuple[int, str, dict[str, Tensor], tuple[str, ...]]] = []
    for payload_path, candidate in _state_dict_candidates(payload):
        mapped, unexpected = _map_checkpoint_keys(candidate, model_state)
        scored.append((len(mapped), payload_path, mapped, unexpected))
    if not scored:
        raise RuntimeError("Checkpoint contains no tensor state_dict")

    scored.sort(key=lambda item: (item[0], -len(item[3])), reverse=True)
    best = scored[0]
    if best[0] == 0:
        paths = ", ".join(item[1] for item in scored)
        raise RuntimeError(
            f"No checkpoint tensors match the MaskGIT VQGAN; inspected {paths}"
        )
    if len(scored) > 1 and scored[1][0] == best[0] and scored[1][1] != best[1]:
        raise RuntimeError(
            f"Ambiguous checkpoint payloads {best[1]!r} and {scored[1][1]!r} "
            f"both match {best[0]} parameters"
        )
    return best[1], best[2], best[3]


def load_maskgit_codebook(checkpoint_path: str | Path) -> Tensor:
    """Load only `quantize.embedding.weight` without constructing the VQGAN.

    This is the training-time path for pretokenized AR runs.  It accepts wrapper
    prefixes such as `module.tokenizer.` but refuses ambiguous model/EMA payloads
    instead of silently choosing one.
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"MaskGIT VQGAN checkpoint not found: {checkpoint_path}"
        )
    payload = _read_checkpoint(checkpoint_path)
    matches: list[tuple[str, str, Tensor]] = []
    target_key = "quantize.embedding.weight"
    for payload_path, state in _state_dict_candidates(payload):
        for key, tensor in state.items():
            if key == target_key or key.endswith("." + target_key):
                if tuple(tensor.shape) == (1024, 256):
                    matches.append((payload_path, key, tensor))
                else:
                    raise ValueError(
                        f"MaskGIT codebook {payload_path}.{key} must have shape (1024, 256), "
                        f"got {tuple(tensor.shape)}"
                    )
    if not matches:
        raise RuntimeError(f"Checkpoint does not contain {target_key!r}")
    unique_locations = {(payload_path, key) for payload_path, key, _ in matches}
    if len(unique_locations) != 1:
        locations = [f"{payload_path}.{key}" for payload_path, key, _ in matches]
        raise RuntimeError(f"Ambiguous MaskGIT codebook tensors: {locations}")
    return matches[0][2].detach().to(device="cpu", dtype=torch.float32).contiguous()


@contextlib.contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    value = str(path.resolve())
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:  # pragma: no cover - defensive
            pass


class _ExactMaskGITBackend(nn.Module):
    """Thin composition of the official reAR MaskGIT building blocks."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, quantizer: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantize = quantizer

    def encode(self, images: Tensor) -> tuple[Tensor, dict[str, Tensor | None]]:
        hidden = self.encoder(images)
        quantized, indices, loss = self.quantize(hidden)
        return quantized, {"min_encoding_indices": indices, "codebook_loss": loss}

    def decode_tokens(self, tokens: Tensor) -> Tensor:
        quantized = self.quantize.get_codebook_entry(tokens)
        return self.decoder(quantized).clamp(0.0, 1.0)


def _build_official_backend(rear_repo: Path | None) -> nn.Module:
    if rear_repo is None:
        module_name = "modeling.modules.maskgit_vqgan"
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                "Official reAR modules are not importable. Pass rear_repo=... pointing to the reAR checkout."
            ) from exc
    else:
        rear_repo = rear_repo.resolve()
        expected = rear_repo / "modeling" / "modules" / "maskgit_vqgan.py"
        if not expected.is_file():
            raise FileNotFoundError(f"Official MaskGIT module not found: {expected}")
        with _temporary_sys_path(rear_repo):
            module = importlib.import_module("modeling.modules.maskgit_vqgan")
        module_file = Path(module.__file__).resolve()
        if rear_repo not in module_file.parents:
            raise RuntimeError(
                f"Imported modeling package from {module_file}, not requested reAR checkout {rear_repo}. "
                "Start a fresh process or remove the conflicting modeling package."
            )

    config = SimpleNamespace(
        channel_mult=[1, 1, 2, 2, 4],
        num_resolutions=5,
        dropout=0.0,
        hidden_channels=128,
        num_channels=3,
        num_res_blocks=2,
        resolution=256,
        z_channels=256,
    )
    encoder = module.Encoder(config)
    decoder = module.Decoder(config)
    quantizer = module.VectorQuantizer(
        num_embeddings=1024, embedding_dim=256, commitment_cost=0.25
    )
    return _ExactMaskGITBackend(encoder, decoder, quantizer)


class MaskGITVQGANAdapter(nn.Module):
    """Frozen `[0,1] image <-> 256 integer tokens` tokenizer interface."""

    def __init__(
        self,
        backend: nn.Module,
        *,
        spec: MaskGITVQGANSpec = MaskGITVQGANSpec(),
        checkpoint_report: CheckpointLoadReport | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.spec = spec
        self.checkpoint_report = checkpoint_report
        self._assert_codebook_contract()
        self._freeze()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        rear_repo: str | Path | None = None,
        backend_factory: Callable[[], nn.Module] | None = None,
        spec: MaskGITVQGANSpec = MaskGITVQGANSpec(),
        strict: bool = True,
    ) -> "MaskGITVQGANAdapter":
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MaskGIT VQGAN checkpoint not found: {checkpoint_path}"
            )
        backend = (
            backend_factory()
            if backend_factory is not None
            else _build_official_backend(
                Path(rear_repo) if rear_repo is not None else None
            )
        )

        payload = _read_checkpoint(checkpoint_path)
        model_state = backend.state_dict()
        payload_path, mapped, unexpected = _select_checkpoint_state(
            payload, model_state
        )
        missing = tuple(sorted(set(model_state) - set(mapped)))
        if strict and (missing or unexpected):
            raise RuntimeError(
                "MaskGIT VQGAN checkpoint mismatch: "
                f"missing={list(missing)[:20]}, unexpected={list(unexpected)[:20]}"
            )
        incompat = backend.load_state_dict(mapped, strict=False)
        missing = tuple(sorted(set(missing) | set(incompat.missing_keys)))
        unexpected = tuple(sorted(set(unexpected) | set(incompat.unexpected_keys)))
        if missing:
            raise RuntimeError(
                f"MaskGIT VQGAN is missing required parameters: {list(missing)[:20]}"
            )

        report = CheckpointLoadReport(
            checkpoint_path=str(checkpoint_path),
            payload_path=payload_path,
            loaded_keys=len(mapped),
            missing_keys=missing,
            unexpected_keys=unexpected,
        )
        return cls(backend, spec=spec, checkpoint_report=report)

    def _freeze(self) -> None:
        self.backend.eval()
        self.backend.requires_grad_(False)
        super().train(False)
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError(
                "Tokenizer freeze failed: at least one parameter still requires gradients"
            )

    def train(self, mode: bool = True) -> "MaskGITVQGANAdapter":
        # A parent model calling train() must never unfreeze dropout/statistics here.
        super().train(False)
        self.backend.eval()
        return self

    @property
    def codebook_weight(self) -> Tensor:
        quantizer = getattr(self.backend, "quantize", None)
        embedding = getattr(quantizer, "embedding", None)
        weight = getattr(embedding, "weight", None)
        if not torch.is_tensor(weight):
            raise RuntimeError("Backend does not expose quantize.embedding.weight")
        return weight

    def _assert_codebook_contract(self) -> None:
        expected = (self.spec.codebook_size, self.spec.codebook_dim)
        actual = tuple(self.codebook_weight.shape)
        if actual != expected:
            raise ValueError(
                f"MaskGIT codebook must have shape {expected}, got {actual}"
            )
        _ = self.spec.latent_side

    def _validate_images(self, images: Tensor) -> None:
        expected_tail = (
            self.spec.image_channels,
            self.spec.image_size,
            self.spec.image_size,
        )
        if images.ndim != 4 or tuple(images.shape[1:]) != expected_tail:
            raise ValueError(
                f"Expected images [B,{','.join(map(str, expected_tail))}], got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise TypeError(
                f"Images must be floating point in [0,1], got {images.dtype}"
            )
        if not torch.isfinite(images).all():
            raise ValueError("Images contain NaN or Inf")
        min_value = float(images.detach().amin())
        max_value = float(images.detach().amax())
        if min_value < -1e-6 or max_value > 1.0 + 1e-6:
            raise ValueError(
                f"MaskGIT VQGAN expects images in [0,1], got [{min_value}, {max_value}]"
            )

    def _validate_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.ndim == 3 and tuple(tokens.shape[1:]) == (
            self.spec.latent_side,
            self.spec.latent_side,
        ):
            tokens = tokens.reshape(tokens.shape[0], -1)
        if tokens.ndim != 2 or tokens.shape[1] != self.spec.num_tokens:
            raise ValueError(
                f"Expected tokens [B,{self.spec.num_tokens}], got {tuple(tokens.shape)}"
            )
        tokens = tokens.to(dtype=torch.long)
        if tokens.numel():
            low, high = int(tokens.min()), int(tokens.max())
            if low < 0 or high >= self.spec.codebook_size:
                raise ValueError(
                    f"Token IDs must be in [0,{self.spec.codebook_size}), got [{low},{high}]"
                )
        return tokens

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        self._validate_images(images)
        output = self.backend.encode(images)
        indices: Tensor | None = None
        if isinstance(output, Mapping):
            indices = output.get("min_encoding_indices")
        elif isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
            for item in output:
                if isinstance(item, Mapping) and torch.is_tensor(
                    item.get("min_encoding_indices")
                ):
                    indices = item["min_encoding_indices"]
                    break
            if indices is None:
                integer_tensors = [
                    item
                    for item in output
                    if torch.is_tensor(item) and not item.is_floating_point()
                ]
                if len(integer_tensors) == 1:
                    indices = integer_tensors[0]
        elif torch.is_tensor(output) and not output.is_floating_point():
            indices = output
        if indices is None:
            raise RuntimeError("Tokenizer encode() did not expose min_encoding_indices")
        return self._validate_tokens(indices).detach()

    @torch.no_grad()
    def decode(self, tokens: Tensor) -> Tensor:
        tokens = self._validate_tokens(tokens)
        if hasattr(self.backend, "decode_tokens"):
            images = self.backend.decode_tokens(tokens)
        elif hasattr(self.backend, "decode"):
            images = self.backend.decode(tokens)
        else:
            raise RuntimeError(
                "Tokenizer backend exposes neither decode_tokens() nor decode()"
            )
        expected_tail = (
            self.spec.image_channels,
            self.spec.image_size,
            self.spec.image_size,
        )
        if (
            not torch.is_tensor(images)
            or images.ndim != 4
            or tuple(images.shape[1:]) != expected_tail
        ):
            shape = (
                tuple(images.shape)
                if torch.is_tensor(images)
                else type(images).__name__
            )
            raise RuntimeError(f"Tokenizer decoded an invalid image shape: {shape}")
        return images.clamp(0.0, 1.0).detach()

    @torch.no_grad()
    def token_embeddings(self, tokens: Tensor) -> Tensor:
        tokens = self._validate_tokens(tokens)
        return torch.nn.functional.embedding(tokens, self.codebook_weight).detach()


def load_frozen_maskgit_vqgan(
    checkpoint_path: str | Path,
    *,
    rear_repo: str | Path | None = None,
    strict: bool = True,
) -> MaskGITVQGANAdapter:
    return MaskGITVQGANAdapter.from_checkpoint(
        checkpoint_path,
        rear_repo=rear_repo,
        strict=strict,
    )
