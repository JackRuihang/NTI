"""RNG-isolated qualitative previews for long-running NTI training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from nti.generation import (
    GenerationOutput,
    SamplingConfig,
    autoregressive_sample,
    decode_generation,
)


PREVIEW_MANIFEST_SCHEMA = "nti.training_preview.v1"
DEFAULT_PREVIEW_TAG = "Train Generated"


@dataclass(frozen=True)
class RNGSnapshot:
    """All process RNG state that an inline preview could perturb."""

    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: tuple[Tensor, ...]
    torch_cuda_device: int | None


def capture_all_rng_states() -> RNGSnapshot:
    """Capture Python, NumPy, CPU Torch, and every visible CUDA generator."""

    cuda_states: tuple[Tensor, ...] = ()
    cuda_device: int | None = None
    # A DDP rank owns only its current device. Reading all visible generators
    # can initialize CUDA contexts on peer GPUs and consume their memory.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_device = torch.cuda.current_device()
        cuda_states = (torch.cuda.get_rng_state(cuda_device).clone(),)
    return RNGSnapshot(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=cuda_states,
        torch_cuda_device=cuda_device,
    )


def restore_all_rng_states(snapshot: RNGSnapshot) -> None:
    """Restore a snapshot created by :func:`capture_all_rng_states`."""

    random.setstate(snapshot.python)
    np.random.set_state(snapshot.numpy)
    torch.set_rng_state(snapshot.torch_cpu)
    if snapshot.torch_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA became unavailable while restoring preview RNG state"
            )
        if snapshot.torch_cuda_device is None or len(snapshot.torch_cuda) != 1:
            raise RuntimeError("preview CUDA RNG snapshot is internally inconsistent")
        torch.cuda.set_rng_state(
            snapshot.torch_cuda[0], device=snapshot.torch_cuda_device
        )


@contextmanager
def preserve_all_rng_states() -> Iterator[RNGSnapshot]:
    """Restore all process RNG streams after success or an exception."""

    snapshot = capture_all_rng_states()
    try:
        yield snapshot
    finally:
        restore_all_rng_states(snapshot)


def make_fixed_generator(device: torch.device | str, seed: int) -> torch.Generator:
    """Construct a private sampler generator without touching global RNG state."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("preview seed must be an integer")
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    return generator


def make_preview_grid(images: Tensor, *, rows: int = 2) -> Tensor:
    """Return a row-major uint8 CHW grid with zero-filled padding cells."""

    if images.ndim != 4:
        raise ValueError(
            f"preview images must have shape [B,C,H,W], got {tuple(images.shape)}"
        )
    batch, channels, height, width = images.shape
    if batch <= 0 or channels not in {1, 3} or height <= 0 or width <= 0:
        raise ValueError(f"invalid preview image shape {tuple(images.shape)}")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("preview grid rows must be a positive integer")
    if not images.is_floating_point() and images.dtype != torch.uint8:
        raise TypeError("preview images must be floating point or uint8")
    if images.is_floating_point() and not torch.isfinite(images).all():
        raise ValueError("preview images contain NaN or Inf")

    columns = math.ceil(batch / rows)
    if images.is_floating_point():
        pixels = images.detach().float().clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    else:
        pixels = images.detach()
    pixels = pixels.cpu()
    grid = torch.zeros((channels, rows * height, columns * width), dtype=torch.uint8)
    for index, image in enumerate(pixels):
        row, column = divmod(index, columns)
        grid[
            :,
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = image
    return grid


def save_preview_grid_atomic(
    images: Tensor, path: str | Path, *, rows: int = 2
) -> Tensor:
    """Atomically save a preview grid as PNG and return its uint8 CHW tensor."""

    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("preview output path must end in .png")
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_preview_grid(images, rows=rows)
    if grid.shape[0] == 1:
        array = grid[0].numpy()
    else:
        array = grid.permute(1, 2, 0).numpy()
    image = Image.fromarray(np.asarray(array))

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            image.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return grid


@dataclass(frozen=True)
class PreviewManifestRecord:
    """Versioned sidecar record written only after the PNG is complete."""

    step: int
    png_path: str
    seed: int
    class_ids: tuple[int, ...]
    sampling: Mapping[str, Any]
    image_count: int
    grid_rows: int
    grid_columns: int
    grid_height: int
    grid_width: int
    png_sha256: str
    mean_token_logprob: float | None
    tensorboard_tag: str = DEFAULT_PREVIEW_TAG
    schema: str = PREVIEW_MANIFEST_SCHEMA
    event: str = "training_preview"

    def __post_init__(self) -> None:
        if self.schema != PREVIEW_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported preview manifest schema {self.schema!r}")
        if self.event != "training_preview":
            raise ValueError(f"unsupported preview event {self.event!r}")
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 0
        ):
            raise ValueError("preview step must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("preview seed must be an integer")
        if not self.tensorboard_tag:
            raise ValueError("preview TensorBoard tag must be nonempty")
        relative = PurePosixPath(self.png_path)
        if (
            relative.is_absolute()
            or "\\" in self.png_path
            or ".." in relative.parts
            or relative.suffix.lower() != ".png"
        ):
            raise ValueError(
                "preview png_path must be a traversal-free relative PNG path"
            )
        if not self.class_ids or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.class_ids
        ):
            raise ValueError("preview class_ids must be non-negative integers")
        if (
            isinstance(self.image_count, bool)
            or not isinstance(self.image_count, int)
            or self.image_count != len(self.class_ids)
        ):
            raise ValueError("preview image_count must equal the number of class_ids")
        dimensions = (
            self.grid_rows,
            self.grid_columns,
            self.grid_height,
            self.grid_width,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in dimensions
        ):
            raise ValueError("preview grid dimensions must be integers")
        if min(dimensions) <= 0:
            raise ValueError("preview grid dimensions must be positive")
        if self.grid_rows * self.grid_columns < self.image_count:
            raise ValueError("preview grid does not have enough cells")
        if len(self.png_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.png_sha256
        ):
            raise ValueError("preview png_sha256 must be a lowercase SHA256 digest")
        if self.mean_token_logprob is not None and not math.isfinite(
            self.mean_token_logprob
        ):
            raise ValueError("preview mean_token_logprob must be finite or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "event": self.event,
            "step": self.step,
            "tensorboard_tag": self.tensorboard_tag,
            "png_path": self.png_path,
            "png_sha256": self.png_sha256,
            "seed": self.seed,
            "class_ids": list(self.class_ids),
            "sampling": dict(self.sampling),
            "image_count": self.image_count,
            "grid": {
                "rows": self.grid_rows,
                "columns": self.grid_columns,
                "height": self.grid_height,
                "width": self.grid_width,
            },
            "mean_token_logprob": self.mean_token_logprob,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreviewManifestRecord":
        grid = payload.get("grid")
        if not isinstance(grid, Mapping):
            raise ValueError("preview manifest record has no grid mapping")
        sampling = payload.get("sampling")
        if not isinstance(sampling, Mapping):
            raise ValueError("preview manifest record has no sampling mapping")
        class_ids = payload.get("class_ids")
        if not isinstance(class_ids, Sequence) or isinstance(class_ids, (str, bytes)):
            raise ValueError("preview manifest record has no class_ids sequence")
        try:
            return cls(
                schema=str(payload["schema"]),
                event=str(payload["event"]),
                step=payload["step"],
                tensorboard_tag=str(payload["tensorboard_tag"]),
                png_path=str(payload["png_path"]),
                png_sha256=str(payload["png_sha256"]),
                seed=payload["seed"],
                class_ids=tuple(class_ids),
                sampling=dict(sampling),
                image_count=payload["image_count"],
                grid_rows=grid["rows"],
                grid_columns=grid["columns"],
                grid_height=grid["height"],
                grid_width=grid["width"],
                mean_token_logprob=payload.get("mean_token_logprob"),
            )
        except KeyError as error:
            raise ValueError(
                f"preview manifest record is missing {error.args[0]!r}"
            ) from error


def append_preview_manifest(path: str | Path, record: PreviewManifestRecord) -> None:
    """Append one complete, fsynced JSON object to a rank-zero manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        json.dumps(
            record.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def iter_preview_manifest(
    path: str | Path,
) -> Iterator[PreviewManifestRecord]:
    """Yield complete records, ignoring only a torn final JSONL write."""

    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                break
            try:
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("record is not a JSON object")
                yield PreviewManifestRecord.from_dict(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid preview manifest record at line {line_number}: {error}"
                ) from error


def resolve_preview_png(run_dir: str | Path, record: PreviewManifestRecord) -> Path:
    """Resolve a manifest PNG while refusing paths outside the training run."""

    root = Path(run_dir).resolve()
    resolved = (root / Path(record.png_path)).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("preview PNG resolves outside the training run")
    return resolved


def _unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "module"):
            current = current.module
        elif hasattr(current, "_orig_mod"):
            current = current._orig_mod
        else:
            break
    return current


def _module_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def generate_training_preview(
    model: nn.Module,
    tokenizer: nn.Module,
    *,
    step: int,
    class_ids: Sequence[int],
    seed: int,
    sampling: SamplingConfig,
    run_dir: str | Path,
    rows: int = 2,
    decode_batch_size: int = 1,
    tensorboard_tag: str = DEFAULT_PREVIEW_TAG,
    manifest_path: str | Path | None = None,
    device: torch.device | str | None = None,
) -> PreviewManifestRecord:
    """Generate one fixed qualitative grid without changing training RNG or mode."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("preview step must be a non-negative integer")
    normalized_classes = tuple(class_ids)
    if not normalized_classes or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in normalized_classes
    ):
        raise ValueError("preview class_ids must be non-negative integers")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("preview grid rows must be a positive integer")
    if (
        isinstance(decode_batch_size, bool)
        or not isinstance(decode_batch_size, int)
        or decode_batch_size <= 0
    ):
        raise ValueError("preview decode_batch_size must be a positive integer")
    sampling.validate()
    raw_model = _unwrap_model(model)
    preview_device = (
        torch.device(device) if device is not None else _module_device(raw_model)
    )
    run_dir = Path(run_dir)
    png_path = run_dir / "train_generated_images" / f"{step:08d}_s-generated.png"
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else run_dir / "preview_manifest.jsonl"
    )
    model_was_training = raw_model.training
    tokenizer_was_training = tokenizer.training

    with preserve_all_rng_states():
        try:
            raw_model.eval()
            tokenizer.eval()
            labels = torch.tensor(
                normalized_classes, dtype=torch.long, device=preview_device
            )
            generator = make_fixed_generator(preview_device, seed)
            output = autoregressive_sample(
                raw_model,
                labels,
                config=sampling,
                generator=generator,
            )
            decoded = []
            for start in range(0, output.tokens.shape[0], decode_batch_size):
                stop = start + decode_batch_size
                chunk = GenerationOutput(
                    tokens=output.tokens[start:stop],
                    token_logprobs=output.token_logprobs[start:stop],
                    guidance_scales=output.guidance_scales,
                )
                decoded.append(decode_generation(tokenizer, chunk).cpu())
            images = torch.cat(decoded, dim=0)
            grid = save_preview_grid_atomic(images, png_path, rows=rows)
            token_logprobs = output.token_logprobs.detach().float()
            finite_logprobs = token_logprobs[torch.isfinite(token_logprobs)]
            mean_logprob = (
                float(finite_logprobs.mean().item())
                if finite_logprobs.numel()
                else None
            )
            relative_png = png_path.resolve().relative_to(run_dir.resolve()).as_posix()
            record = PreviewManifestRecord(
                step=step,
                tensorboard_tag=tensorboard_tag,
                png_path=relative_png,
                png_sha256=_sha256_file(png_path),
                seed=seed,
                class_ids=normalized_classes,
                sampling={
                    **asdict(sampling),
                    "decode_batch_size": decode_batch_size,
                },
                image_count=len(normalized_classes),
                grid_rows=rows,
                grid_columns=math.ceil(len(normalized_classes) / rows),
                grid_height=int(grid.shape[1]),
                grid_width=int(grid.shape[2]),
                mean_token_logprob=mean_logprob,
            )
            append_preview_manifest(manifest_path, record)
            return record
        finally:
            raw_model.train(model_was_training)
            tokenizer.train(tokenizer_was_training)


__all__ = [
    "DEFAULT_PREVIEW_TAG",
    "PREVIEW_MANIFEST_SCHEMA",
    "PreviewManifestRecord",
    "RNGSnapshot",
    "append_preview_manifest",
    "capture_all_rng_states",
    "generate_training_preview",
    "iter_preview_manifest",
    "make_fixed_generator",
    "make_preview_grid",
    "preserve_all_rng_states",
    "resolve_preview_png",
    "restore_all_rng_states",
    "save_preview_grid_atomic",
]
