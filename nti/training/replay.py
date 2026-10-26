"""Sparse lagged replay artifacts for one-forward altered-history training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .targets import masked_token_mean


REPLAY_SCHEMA = "nti.lagged_sparse_replay.v1"
SHARDED_REPLAY_SCHEMA = "nti.lagged_sparse_replay.sharded.v1"
REPLAY_POINTER_SCHEMA = "nti.lagged_sparse_replay.pointer.v1"
ARRAY_NAMES = (
    "record_indices",
    "class_labels",
    "altered_tokens",
    "active_mask",
    "effective_non_gt_mass",
    "top_indices",
    "top_probabilities",
)


@dataclass(frozen=True)
class SparseReplayTarget:
    """Conservative Top-K representation of a dense detached target."""

    effective_non_gt_mass: torch.Tensor
    top_indices: torch.Tensor
    top_probabilities: torch.Tensor
    captured_non_gt_fraction: torch.Tensor


@dataclass(frozen=True)
class ReplayBatch:
    """One random-access batch loaded from a frozen replay bank."""

    record_indices: torch.Tensor
    class_labels: torch.Tensor
    class_drop_mask: torch.Tensor
    altered_tokens: torch.Tensor
    active_mask: torch.Tensor
    effective_non_gt_mass: torch.Tensor
    top_indices: torch.Tensor
    top_probabilities: torch.Tensor


def deterministic_replay_class_drop_mask(
    record_indices: np.ndarray,
    *,
    probability: float,
    seed: int,
) -> np.ndarray:
    """Derive an order-independent class-drop bit from each record index."""

    indices = np.asarray(record_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("record_indices must be a one-dimensional integer array")
    if not 0.0 <= probability < 1.0:
        raise ValueError("class-drop probability must be in [0, 1)")
    if probability == 0.0:
        return np.zeros(indices.shape, dtype=np.bool_)
    values = indices.astype(np.uint64, copy=False) ^ np.uint64(seed)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    uniform = (values >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    return (uniform < probability).astype(np.bool_, copy=False)


@dataclass(frozen=True)
class OnePassReplayOutput:
    """Losses and logits from one mixed clean/replay student forward."""

    loss: torch.Tensor
    clean_loss: torch.Tensor
    replay_loss: torch.Tensor
    logits: torch.Tensor
    input_tokens: torch.Tensor
    clean_loss_mask: torch.Tensor
    replay_loss_mask: torch.Tensor
    clean_loss_sum: torch.Tensor
    replay_loss_sum: torch.Tensor
    clean_token_count: torch.Tensor
    replay_token_count: torch.Tensor


def balanced_replay_image_mask(
    batch_size: int,
    *,
    clean_loss_weight: float,
    replay_loss_weight: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Allocate a mixed batch in proportion to objective weights.

    The allocation probability is derived rather than tuned:
    ``pi = replay_weight / (clean_weight + replay_weight)``.  A deterministic
    count avoids empty clean/replay subsets on individual DDP ranks.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if clean_loss_weight < 0.0 or replay_loss_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    total = clean_loss_weight + replay_loss_weight
    if total <= 0.0:
        raise ValueError("at least one loss weight must be positive")
    if replay_loss_weight == 0.0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if clean_loss_weight == 0.0:
        return torch.ones(batch_size, dtype=torch.bool, device=device)

    replay_count = int(round(batch_size * replay_loss_weight / total))
    replay_count = min(max(replay_count, 1), batch_size - 1)
    order = torch.randperm(batch_size, device=device, generator=generator)
    mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    mask[order[:replay_count]] = True
    return mask


def sparsify_detached_target(
    target_probability: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    top_k: int = 32,
    eps: float = 1e-8,
) -> SparseReplayTarget:
    """Move omitted non-GT tail mass back to GT and retain Top-K direction.

    If the dense target is ``q=(1-g)e_y+g r``, the sparse target is
    ``q_K=(1-g m_K)e_y+g m_K r_K``. Thus truncation is conservative: it never
    increases the non-GT dose and introduces no extra learned or tuned gate.
    """

    if target_probability.ndim != 3:
        raise ValueError("target_probability must be [batch, tokens, vocabulary]")
    if target_ids.shape != target_probability.shape[:2]:
        raise ValueError("target_ids must match target token dimensions")
    vocabulary = target_probability.shape[-1]
    if not 0 < top_k < vocabulary:
        raise ValueError("top_k must be in [1, vocabulary-1]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    with torch.no_grad():
        target = target_probability.detach().float().clamp_min(0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        non_gt = target.clone()
        non_gt.scatter_(-1, target_ids.detach().long()[..., None], 0.0)
        total_non_gt_mass = non_gt.sum(dim=-1)
        top_values, top_indices = non_gt.topk(top_k, dim=-1)
        captured_mass = top_values.sum(dim=-1)
        has_mass = captured_mass > 0.0
        top_probabilities = torch.where(
            has_mass[..., None],
            top_values / captured_mass[..., None].clamp_min(float(eps)),
            torch.zeros_like(top_values),
        )
        captured_fraction = torch.where(
            total_non_gt_mass > 0.0,
            captured_mass / total_non_gt_mass.clamp_min(float(eps)),
            torch.ones_like(total_non_gt_mass),
        ).clamp(0.0, 1.0)

    return SparseReplayTarget(
        effective_non_gt_mass=captured_mass.detach(),
        top_indices=top_indices.detach(),
        top_probabilities=top_probabilities.detach(),
        captured_non_gt_fraction=captured_fraction.detach(),
    )


def categorical_sample_detached_target(
    target_probability: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    sample_count: int = 32,
    generator: torch.Generator | None = None,
    eps: float = 1e-8,
) -> SparseReplayTarget:
    r"""Monte Carlo encode dense soft CE as an average of hard targets.

    For samples ``z_m ~ q``, ``-M^-1 sum_m log p(z_m)`` is an unbiased
    estimator of the dense soft cross entropy. The existing sparse replay
    representation stores non-GT samples explicitly and their realized fraction
    as the non-GT mass, so no vocabulary tail is reassigned to GT.
    """
    if target_probability.ndim != 3:
        raise ValueError("target_probability must be [batch, tokens, vocabulary]")
    if target_ids.shape != target_probability.shape[:2]:
        raise ValueError("target_ids must match target token dimensions")
    vocabulary = target_probability.shape[-1]
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    with torch.no_grad():
        target = target_probability.detach().float().clamp_min(0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        samples = torch.multinomial(
            target.reshape(-1, vocabulary),
            sample_count,
            replacement=True,
            generator=generator,
        ).reshape(*target_ids.shape, sample_count)
        non_gt = samples.ne(target_ids.detach().long()[..., None])
        non_gt_count = non_gt.sum(dim=-1)
        effective_non_gt_mass = non_gt_count.float() / float(sample_count)
        top_indices = torch.where(
            non_gt,
            samples,
            target_ids.detach().long()[..., None],
        )
        top_probabilities = torch.where(
            non_gt,
            1.0 / non_gt_count.clamp_min(1).float()[..., None],
            torch.zeros_like(samples, dtype=torch.float32),
        )

    return SparseReplayTarget(
        effective_non_gt_mass=effective_non_gt_mass.detach(),
        top_indices=top_indices.detach(),
        top_probabilities=top_probabilities.detach(),
        captured_non_gt_fraction=torch.ones_like(effective_non_gt_mass).detach(),
    )


def conditional_non_gt_sample_detached_target(
    target_probability: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    sample_count: int = 8,
    generator: torch.Generator | None = None,
    eps: float = 1e-8,
) -> SparseReplayTarget:
    """Encode exact non-GT dose with samples from its conditional direction."""

    if target_probability.ndim != 3:
        raise ValueError("target_probability must be [batch, tokens, vocabulary]")
    if target_ids.shape != target_probability.shape[:2]:
        raise ValueError("target_ids must match target token dimensions")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    vocabulary = target_probability.shape[-1]
    with torch.no_grad():
        target = target_probability.detach().float().clamp_min(0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        gt = target_ids.detach().long()
        non_gt = target.clone()
        non_gt.scatter_(-1, gt[..., None], 0.0)
        dose = non_gt.sum(dim=-1).clamp(0.0, 1.0)
        direction = non_gt / dose[..., None].clamp_min(float(eps))
        zero_dose = dose <= float(eps)
        if zero_dose.any():
            direction = direction.clone()
            direction[zero_dose] = F.one_hot(
                gt[zero_dose], num_classes=vocabulary
            ).float()
        samples = torch.multinomial(
            direction.reshape(-1, vocabulary),
            sample_count,
            replacement=True,
            generator=generator,
        ).reshape(*target_ids.shape, sample_count)
        weights = torch.ones_like(samples, dtype=torch.float32)

    return SparseReplayTarget(
        effective_non_gt_mass=dose.detach(),
        top_indices=samples.detach(),
        top_probabilities=weights.detach(),
        captured_non_gt_fraction=torch.ones_like(dose).detach(),
    )


def sparse_replay_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    sparse_target: SparseReplayTarget,
    *,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute CE without reconstructing a dense replay target."""

    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("logits and target_ids have incompatible shapes")
    expected = (*logits.shape[:2], sparse_target.top_indices.shape[-1])
    if tuple(sparse_target.top_indices.shape) != expected:
        raise ValueError("top_indices have incompatible shape")
    if sparse_target.top_probabilities.shape != sparse_target.top_indices.shape:
        raise ValueError("top probabilities must match top indices")
    if sparse_target.effective_non_gt_mass.shape != logits.shape[:2]:
        raise ValueError("effective_non_gt_mass must match token dimensions")

    per_token = _sparse_replay_per_token(logits, target_ids, sparse_target)
    return masked_token_mean(per_token, loss_mask)


def _sparse_replay_per_token(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    sparse_target: SparseReplayTarget,
) -> torch.Tensor:
    log_probability = F.log_softmax(logits.float(), dim=-1)
    gt_log_probability = log_probability.gather(
        -1, target_ids.long()[..., None]
    ).squeeze(-1)
    top_log_probability = log_probability.gather(-1, sparse_target.top_indices.long())
    mass = sparse_target.effective_non_gt_mass.detach().float().clamp(0.0, 1.0)
    direction = sparse_target.top_probabilities.detach().float().clamp_min(0.0)
    direction = direction / direction.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(
        (1.0 - mass) * gt_log_probability
        + mass * (direction * top_log_probability).sum(dim=-1)
    )


def _masked_sum_and_count(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_float = mask.detach().to(device=values.device, dtype=values.dtype)
    return (values * mask_float).sum(), mask_float.sum()


def one_pass_replay_forward(
    model: torch.nn.Module,
    image_tokens: torch.Tensor,
    class_labels: torch.Tensor,
    replay_batch: ReplayBatch,
    replay_image_mask: torch.Tensor,
    *,
    clean_loss_weight: float = 1.0,
    replay_loss_weight: float = 0.5,
    replay_all_tokens: bool = False,
    model_kwargs: Mapping[str, Any] | None = None,
) -> OnePassReplayOutput:
    """Run one student forward over disjoint clean and frozen-replay examples.

    The clean and replay losses are means over their own randomly assigned
    subsets. Each is therefore an unbiased population-risk estimate; summing
    them preserves the configured objective without forwarding any image twice.
    """

    if image_tokens.ndim != 2:
        raise ValueError("image_tokens must be [batch, tokens]")
    if class_labels.shape != image_tokens.shape[:1]:
        raise ValueError("class_labels must be [batch]")
    if replay_image_mask.shape != image_tokens.shape[:1]:
        raise ValueError("replay_image_mask must be [batch]")
    if replay_batch.altered_tokens.shape != image_tokens.shape:
        raise ValueError("replay altered tokens must match image_tokens")
    if replay_batch.active_mask.shape != image_tokens.shape:
        raise ValueError("replay active mask must match image_tokens")
    if replay_batch.class_drop_mask.shape != class_labels.shape:
        raise ValueError("replay class-drop mask must match class labels")
    if not torch.equal(replay_batch.class_labels.to(class_labels), class_labels):
        raise ValueError("replay class labels do not match the current batch")
    if clean_loss_weight < 0.0 or replay_loss_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if clean_loss_weight == 0.0 and replay_loss_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")

    replay_image_mask = replay_image_mask.detach().bool()
    input_tokens = torch.where(
        replay_image_mask[:, None], replay_batch.altered_tokens, image_tokens
    ).detach()
    kwargs = dict(model_kwargs or {})
    if "class_labels" in kwargs:
        raise ValueError("class_labels must use the named argument")
    raw_output = model(input_tokens, class_labels=class_labels, **kwargs)
    logits = raw_output["logits"] if isinstance(raw_output, Mapping) else raw_output
    if not isinstance(logits, torch.Tensor) or logits.shape[:2] != image_tokens.shape:
        raise ValueError("model logits must align with image_tokens")

    clean_loss_mask = (~replay_image_mask)[:, None].expand_as(image_tokens)
    replay_loss_mask = replay_image_mask[:, None].expand_as(image_tokens)
    if not replay_all_tokens:
        replay_loss_mask = replay_loss_mask & replay_batch.active_mask.detach().bool()
    clean_per_token = F.cross_entropy(
        logits.transpose(1, 2), image_tokens.long(), reduction="none"
    )
    clean_loss = masked_token_mean(clean_per_token, clean_loss_mask)
    sparse_target = SparseReplayTarget(
        effective_non_gt_mass=replay_batch.effective_non_gt_mass,
        top_indices=replay_batch.top_indices,
        top_probabilities=replay_batch.top_probabilities,
        captured_non_gt_fraction=torch.ones_like(
            replay_batch.effective_non_gt_mass, dtype=torch.float32
        ),
    )
    replay_per_token = _sparse_replay_per_token(logits, image_tokens, sparse_target)
    replay_loss = masked_token_mean(replay_per_token, replay_loss_mask)
    clean_loss_sum, clean_token_count = _masked_sum_and_count(
        clean_per_token, clean_loss_mask
    )
    replay_loss_sum, replay_token_count = _masked_sum_and_count(
        replay_per_token, replay_loss_mask
    )
    loss = clean_loss_weight * clean_loss + replay_loss_weight * replay_loss
    return OnePassReplayOutput(
        loss=loss,
        clean_loss=clean_loss,
        replay_loss=replay_loss,
        logits=logits,
        input_tokens=input_tokens,
        clean_loss_mask=clean_loss_mask,
        replay_loss_mask=replay_loss_mask,
        clean_loss_sum=clean_loss_sum,
        replay_loss_sum=replay_loss_sum,
        clean_token_count=clean_token_count,
        replay_token_count=replay_token_count,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_replay_bank(
    destination: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    provenance: Mapping[str, Any],
) -> Path:
    """Atomically publish immutable NumPy arrays plus strict provenance."""

    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    missing = set(ARRAY_NAMES) - set(arrays)
    extra = set(arrays) - set(ARRAY_NAMES)
    if missing or extra:
        raise ValueError(f"replay arrays mismatch: missing={missing}, extra={extra}")

    record_indices = np.asarray(arrays["record_indices"])
    if record_indices.ndim != 1 or record_indices.size == 0:
        raise ValueError("record_indices must be a non-empty vector")
    if not np.issubdtype(record_indices.dtype, np.integer):
        raise TypeError("record_indices must be integer")
    if np.any(record_indices[1:] <= record_indices[:-1]):
        raise ValueError("record_indices must be strictly increasing")
    count = record_indices.size
    for name in (
        "class_labels",
        "altered_tokens",
        "active_mask",
        "effective_non_gt_mass",
        "top_indices",
        "top_probabilities",
    ):
        if np.asarray(arrays[name]).shape[0] != count:
            raise ValueError(f"{name} must align with record_indices")

    partial = destination.with_name(destination.name + f".partial-{os.getpid()}")
    if partial.exists():
        raise FileExistsError(partial)
    partial.mkdir(parents=True)
    try:
        manifest_arrays: dict[str, dict[str, Any]] = {}
        for name in ARRAY_NAMES:
            value = np.asarray(arrays[name])
            path = partial / f"{name}.npy"
            np.save(path, value, allow_pickle=False)
            manifest_arrays[name] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _sha256(path),
            }
        manifest = {
            "schema": REPLAY_SCHEMA,
            "record_count": int(count),
            "arrays": manifest_arrays,
            "provenance": dict(provenance),
        }
        (partial / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (partial / "COMPLETE.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "manifest_sha256": _sha256(partial / "manifest.json"),
                    "schema": REPLAY_SCHEMA,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(partial, destination)
    finally:
        if partial.exists():
            for path in partial.iterdir():
                path.unlink()
            partial.rmdir()
    return destination


class LaggedReplayBank:
    """Read-only mmap view of an immutable replay-bank directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        complete_path = self.root / "COMPLETE.json"
        manifest_path = self.root / "manifest.json"
        if not complete_path.is_file() or not manifest_path.is_file():
            raise ValueError("replay bank is incomplete")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("schema") != REPLAY_SCHEMA or not complete.get("complete"):
            raise ValueError("replay completion marker has the wrong schema")
        if complete.get("manifest_sha256") != _sha256(manifest_path):
            raise ValueError("replay manifest checksum mismatch")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != REPLAY_SCHEMA:
            raise ValueError("replay manifest has the wrong schema")
        self.arrays = {
            name: np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ARRAY_NAMES
        }
        for name, array in self.arrays.items():
            expected = self.manifest["arrays"][name]
            if (
                list(array.shape) != expected["shape"]
                or str(array.dtype) != expected["dtype"]
            ):
                raise ValueError(f"replay array metadata mismatch: {name}")
        self.record_indices = self.arrays["record_indices"]

    def lookup(self, record_indices: torch.Tensor) -> ReplayBatch:
        if record_indices.ndim != 1:
            raise ValueError("record_indices must be a vector")
        query = record_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        locations = np.searchsorted(self.record_indices, query)
        valid = locations < self.record_indices.size
        valid &= (
            self.record_indices[np.minimum(locations, self.record_indices.size - 1)]
            == query
        )
        if not bool(valid.all()):
            missing = query[~valid][:8].tolist()
            raise KeyError(f"record indices absent from replay bank: {missing}")

        device = record_indices.device

        def tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(
                np.asarray(self.arrays[name][locations]), device=device
            )

        return ReplayBatch(
            record_indices=record_indices.detach().clone(),
            class_labels=tensor("class_labels").long(),
            class_drop_mask=torch.as_tensor(
                deterministic_replay_class_drop_mask(
                    query,
                    probability=float(
                        self.manifest.get("provenance", {}).get(
                            "class_drop_probability", 0.0
                        )
                    ),
                    seed=int(
                        self.manifest.get("provenance", {}).get("class_drop_seed", 0)
                    ),
                ),
                device=device,
            ),
            altered_tokens=tensor("altered_tokens").long(),
            active_mask=tensor("active_mask").bool(),
            effective_non_gt_mass=tensor("effective_non_gt_mass").float(),
            top_indices=tensor("top_indices").long(),
            top_probabilities=tensor("top_probabilities").float(),
        )

    def contains(self, record_indices: torch.Tensor) -> torch.Tensor:
        if record_indices.ndim != 1:
            raise ValueError("record_indices must be a vector")
        query = record_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        locations = np.searchsorted(self.record_indices, query)
        valid = locations < self.record_indices.size
        valid &= (
            self.record_indices[np.minimum(locations, self.record_indices.size - 1)]
            == query
        )
        return torch.as_tensor(valid, device=record_indices.device)


class ShardedLaggedReplayBank:
    """Read-only collection of non-overlapping contiguous replay shards."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        complete_path = self.root / "COMPLETE.json"
        manifest_path = self.root / "manifest.json"
        if not complete_path.is_file() or not manifest_path.is_file():
            raise ValueError("sharded replay bank is incomplete")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("schema") != SHARDED_REPLAY_SCHEMA or not complete.get(
            "complete"
        ):
            raise ValueError("sharded replay completion marker has the wrong schema")
        if complete.get("manifest_sha256") != _sha256(manifest_path):
            raise ValueError("sharded replay manifest checksum mismatch")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != SHARDED_REPLAY_SCHEMA:
            raise ValueError("sharded replay manifest has the wrong schema")

        shard_entries = self.manifest.get("shards")
        if not isinstance(shard_entries, list) or not shard_entries:
            raise ValueError("sharded replay manifest has no shards")
        self.shards: list[LaggedReplayBank] = []
        self.starts: list[int] = []
        self.ends: list[int] = []
        previous_end: int | None = None
        for entry in shard_entries:
            start = int(entry["start_record_index"])
            end = int(entry["end_record_index_exclusive"])
            if (
                start < 0
                or end <= start
                or (previous_end is not None and start < previous_end)
            ):
                raise ValueError(
                    "replay shard ranges must be positive and non-overlapping"
                )
            bank = LaggedReplayBank(self.root / str(entry["path"]))
            if (
                int(bank.record_indices[0]) != start
                or int(bank.record_indices[-1]) != end - 1
            ):
                raise ValueError("replay shard range does not match its records")
            bank_count = int(bank.manifest.get("record_count", -1))
            if int(entry.get("record_count", bank_count)) != bank_count:
                raise ValueError(
                    "replay shard record count does not match its manifest"
                )
            self.shards.append(bank)
            self.starts.append(start)
            self.ends.append(end)
            previous_end = end
        self.record_count = sum(
            int(bank.manifest.get("record_count", -1)) for bank in self.shards
        )
        if self.record_count != int(self.manifest.get("record_count", -1)):
            raise ValueError("sharded replay record count mismatch")
        provenance = self.manifest.get("provenance", {})
        if provenance.get("replay_stream_mode") == "all_records":
            dataset_count = int(provenance.get("dataset_record_count", -1))
            contiguous = self.starts[0] == 0 and all(
                start == end for start, end in zip(self.starts[1:], self.ends[:-1])
            )
            if (
                dataset_count <= 0
                or not contiguous
                or self.ends[-1] != dataset_count
                or self.record_count != dataset_count
                or any(
                    int(bank.manifest.get("record_count", -1)) != end - start
                    for bank, start, end in zip(self.shards, self.starts, self.ends)
                )
            ):
                raise ValueError(
                    "all-record replay bank does not exactly cover the dataset"
                )

    def lookup(self, record_indices: torch.Tensor) -> ReplayBatch:
        if record_indices.ndim != 1:
            raise ValueError("record_indices must be a vector")
        query = record_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if query.size == 0:
            raise ValueError("record_indices must be non-empty")
        shard_ids = np.searchsorted(np.asarray(self.ends), query, side="right")
        valid = shard_ids < len(self.shards)
        valid &= (
            query
            >= np.asarray(self.starts)[np.minimum(shard_ids, len(self.shards) - 1)]
        )
        if not bool(valid.all()):
            raise KeyError(
                f"record indices absent from replay bank: {query[~valid][:8].tolist()}"
            )

        pieces: dict[str, torch.Tensor] = {}
        for shard_id in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_id)
            position_tensor = torch.as_tensor(positions, device=record_indices.device)
            batch = self.shards[int(shard_id)].lookup(
                record_indices.index_select(0, position_tensor)
            )
            for name in (
                "record_indices",
                "class_labels",
                "class_drop_mask",
                "altered_tokens",
                "active_mask",
                "effective_non_gt_mass",
                "top_indices",
                "top_probabilities",
            ):
                value = getattr(batch, name)
                if name not in pieces:
                    pieces[name] = torch.empty(
                        (record_indices.shape[0], *value.shape[1:]),
                        dtype=value.dtype,
                        device=value.device,
                    )
                pieces[name].index_copy_(0, position_tensor, value)
        return ReplayBatch(**pieces)

    def contains(self, record_indices: torch.Tensor) -> torch.Tensor:
        if record_indices.ndim != 1:
            raise ValueError("record_indices must be a vector")
        query = record_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        shard_ids = np.searchsorted(np.asarray(self.ends), query, side="right")
        result = np.zeros(query.shape, dtype=np.bool_)
        for shard_id in np.unique(shard_ids[shard_ids < len(self.shards)]):
            positions = np.flatnonzero(shard_ids == shard_id)
            bank_records = self.shards[int(shard_id)].record_indices
            locations = np.searchsorted(bank_records, query[positions])
            valid = locations < bank_records.size
            valid &= (
                bank_records[np.minimum(locations, bank_records.size - 1)]
                == query[positions]
            )
            result[positions] = valid
        return torch.as_tensor(result, device=record_indices.device)


def open_replay_bank(root: str | Path) -> LaggedReplayBank | ShardedLaggedReplayBank:
    """Open either a legacy single-shard bank or a sharded bank root."""

    root = Path(root).expanduser().resolve()
    complete_path = root / "COMPLETE.json"
    if not complete_path.is_file():
        raise ValueError("replay bank is incomplete")
    schema = json.loads(complete_path.read_text(encoding="utf-8")).get("schema")
    if schema == REPLAY_SCHEMA:
        return LaggedReplayBank(root)
    if schema == SHARDED_REPLAY_SCHEMA:
        return ShardedLaggedReplayBank(root)
    raise ValueError(f"unsupported replay bank schema: {schema!r}")


def align_available_replay_batch(
    bank: LaggedReplayBank | ShardedLaggedReplayBank,
    record_indices: torch.Tensor,
    image_tokens: torch.Tensor,
    class_labels: torch.Tensor,
) -> tuple[ReplayBatch, torch.Tensor]:
    """Materialize sparse bank records into the current batch order."""

    available = bank.contains(record_indices).detach().bool()
    top_k = int(bank.manifest.get("provenance", {}).get("top_k", -1))
    if top_k <= 0:
        raise ValueError("replay bank provenance has no valid top_k")
    altered = image_tokens.detach().clone()
    active = torch.zeros_like(image_tokens, dtype=torch.bool)
    mass = torch.zeros_like(image_tokens, dtype=torch.float32)
    class_drop = torch.zeros_like(class_labels, dtype=torch.bool)
    top_indices = image_tokens[..., None].expand(*image_tokens.shape, top_k).clone()
    top_probabilities = torch.zeros(
        (*image_tokens.shape, top_k), dtype=torch.float32, device=image_tokens.device
    )
    top_probabilities[..., 0] = 1.0
    if available.any():
        positions = available.nonzero(as_tuple=False).flatten()
        selected = bank.lookup(record_indices.index_select(0, positions))
        if not torch.equal(
            selected.class_labels.to(class_labels),
            class_labels.index_select(0, positions),
        ):
            raise ValueError("replay class labels do not match selected examples")
        altered.index_copy_(0, positions, selected.altered_tokens)
        active.index_copy_(0, positions, selected.active_mask)
        mass.index_copy_(0, positions, selected.effective_non_gt_mass)
        class_drop.index_copy_(0, positions, selected.class_drop_mask)
        top_indices.index_copy_(0, positions, selected.top_indices)
        top_probabilities.index_copy_(0, positions, selected.top_probabilities)
    return (
        ReplayBatch(
            record_indices=record_indices.detach().clone(),
            class_labels=class_labels.detach().clone(),
            class_drop_mask=class_drop,
            altered_tokens=altered,
            active_mask=active,
            effective_non_gt_mass=mass,
            top_indices=top_indices,
            top_probabilities=top_probabilities,
        ),
        available,
    )


def resolve_replay_bank_root(root: str | Path) -> Path:
    """Resolve an immutable bank directly or through an atomic CURRENT pointer."""

    root = Path(root).expanduser().resolve()
    if (root / "COMPLETE.json").is_file():
        return root
    pointer_path = root / "CURRENT.json"
    if not pointer_path.is_file():
        raise ValueError("replay bank root has neither COMPLETE.json nor CURRENT.json")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema") != REPLAY_POINTER_SCHEMA:
        raise ValueError("replay pointer has the wrong schema")
    relative = Path(str(pointer.get("bank", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("replay pointer bank path must be a safe relative path")
    resolved = (root / relative).resolve()
    if root not in resolved.parents:
        raise ValueError("replay pointer escapes its root")
    manifest_path = resolved / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("replay pointer target is incomplete")
    expected_sha = str(pointer.get("manifest_sha256", ""))
    if not expected_sha or _sha256(manifest_path) != expected_sha:
        raise ValueError("replay pointer manifest checksum mismatch")
    return resolved


def write_sharded_replay_index(
    destination: str | Path,
    *,
    shards: list[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> Path:
    """Publish the top-level index after all immutable shards are complete."""

    destination = Path(destination).expanduser().resolve()
    if not destination.is_dir():
        raise FileNotFoundError(destination)
    if (destination / "COMPLETE.json").exists() or (
        destination / "manifest.json"
    ).exists():
        raise FileExistsError("replay index is already published")
    normalized = [dict(item) for item in shards]
    if not normalized:
        raise ValueError("at least one replay shard is required")
    record_count = 0
    previous_end: int | None = None
    for entry in normalized:
        start = int(entry["start_record_index"])
        end = int(entry["end_record_index_exclusive"])
        if (
            start < 0
            or end <= start
            or (previous_end is not None and start < previous_end)
        ):
            raise ValueError("replay shard ranges must be positive and non-overlapping")
        bank = LaggedReplayBank(destination / str(entry["path"]))
        if (
            int(bank.record_indices[0]) != start
            or int(bank.record_indices[-1]) != end - 1
        ):
            raise ValueError("replay shard metadata does not match its records")
        bank_count = int(bank.manifest.get("record_count", -1))
        if int(entry.get("record_count", bank_count)) != bank_count:
            raise ValueError("replay shard record count does not match its manifest")
        record_count += bank_count
        previous_end = end
    if provenance.get("replay_stream_mode") == "all_records":
        dataset_count = int(provenance.get("dataset_record_count", -1))
        contiguous = int(normalized[0]["start_record_index"]) == 0 and all(
            int(current["start_record_index"])
            == int(previous["end_record_index_exclusive"])
            for previous, current in zip(normalized, normalized[1:])
        )
        if (
            dataset_count <= 0
            or not contiguous
            or int(normalized[-1]["end_record_index_exclusive"]) != dataset_count
            or record_count != dataset_count
        ):
            raise ValueError(
                "all-record replay shards do not exactly cover the dataset"
            )
    manifest = {
        "schema": SHARDED_REPLAY_SCHEMA,
        "record_count": record_count,
        "shards": normalized,
        "provenance": dict(provenance),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    complete_path = destination / "COMPLETE.json"
    complete_path.write_text(
        json.dumps(
            {
                "complete": True,
                "manifest_sha256": _sha256(manifest_path),
                "schema": SHARDED_REPLAY_SCHEMA,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
