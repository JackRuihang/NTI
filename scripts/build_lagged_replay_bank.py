#!/usr/bin/env python3
"""Build an immutable sharded lagged-replay bank from a frozen checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nti.checkpoint import load_checkpoint  # noqa: E402
from nti.config import load_experiment_config  # noqa: E402
from nti.data import (  # noqa: E402
    PretokenizedImageNetDataset,
    sha256_file,
    verify_manifest_tokenizer_checkpoint,
)
from nti.distributed import (  # noqa: E402
    barrier,
    initialize_distributed,
    seed_everything,
    shutdown_distributed,
)
from nti.models import build_nti  # noqa: E402
from nti.tokenizer import load_maskgit_codebook  # noqa: E402
from nti.training import (  # noqa: E402
    BellCosineReplacementSchedule,
    LinearReplacementSchedule,
    TwoPassConfig,
    build_codebook_cosine_kernel,
    categorical_sample_detached_target,
    conditional_non_gt_sample_detached_target,
    deterministic_replay_class_drop_mask,
    sparsify_detached_target,
    two_pass_forward,
    write_replay_bank,
    write_sharded_replay_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-checkpoint-step", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--limit-to-training-steps",
        type=int,
        help=(
            "Only materialize selected records visited during the first N "
            "optimizer steps of epoch zero. The population selection probability "
            "is unchanged."
        ),
    )
    return parser.parse_args()


def contiguous_partition(count: int, rank: int, world_size: int) -> tuple[int, int]:
    if count <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid contiguous partition arguments")
    return count * rank // world_size, count * (rank + 1) // world_size


def estimated_bank_bytes(
    record_count: int,
    sequence_length: int,
    top_k: int,
    *,
    conditional_equal_weights: bool = False,
) -> int:
    """Estimate persistent bytes for the fixed production dtypes."""

    if record_count <= 0 or sequence_length <= 0 or top_k <= 0:
        raise ValueError("bank dimensions must be positive")
    per_record = (
        8  # record index int64
        + 2  # class int16
        + sequence_length * 2  # altered uint16
        + sequence_length  # active bool
        + sequence_length * 2  # effective mass float16
        + sequence_length * top_k * 2  # top indices uint16
        + sequence_length * top_k * (1 if conditional_equal_weights else 2)
    )
    return record_count * per_record


def planned_training_record_indices(
    dataset_size: int,
    train,
    *,
    world_size: int,
    optimizer_steps: int,
) -> np.ndarray:
    """Return the exact epoch-zero records consumed by a bounded DDP run."""

    if dataset_size <= 0 or world_size <= 0 or optimizer_steps <= 0:
        raise ValueError("planned training dimensions must be positive")
    local_batch = int(train.per_device_batch_size) * world_size
    if int(train.global_batch_size) % local_batch:
        raise ValueError(
            "global_batch_size must be divisible by per_device_batch_size * world_size"
        )
    accumulation_steps = int(train.global_batch_size) // local_batch
    records_per_rank = (
        optimizer_steps * accumulation_steps * int(train.per_device_batch_size)
    )
    sampler_records_per_rank = dataset_size // world_size
    if records_per_rank > sampler_records_per_rank:
        raise ValueError(
            "bounded replay bank currently supports only an epoch-zero training window"
        )

    rank_indices: list[np.ndarray] = []
    for rank in range(world_size):
        sampler = DistributedSampler(
            range(dataset_size),
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(train.seed),
            drop_last=True,
        )
        sampler.set_epoch(0)
        values = np.fromiter(
            itertools.islice(iter(sampler), records_per_rank),
            dtype=np.int64,
            count=records_per_rank,
        )
        if values.size != records_per_rank:
            raise RuntimeError("distributed sampler ended before the planned window")
        rank_indices.append(values)

    planned = np.sort(np.concatenate(rank_indices))
    if np.unique(planned).size != planned.size:
        raise RuntimeError("epoch-zero distributed sampler produced duplicate records")
    expected = optimizer_steps * int(train.global_batch_size)
    if planned.size != expected:
        raise RuntimeError(
            f"planned record count mismatch: expected {expected}, got {planned.size}"
        )
    return planned


def replay_selection_probability(train, snapshot_step: int) -> tuple[float, float, int]:
    """Derive the cached record fraction from the midpoint objective weight."""

    midpoint = min(
        snapshot_step + train.replay_refresh_steps // 2,
        train.replay_end_step - 1,
    )
    schedule = (
        LinearReplacementSchedule(
            start_ratio=train.replacement_max_ratio,
            max_ratio=train.replacement_max_ratio,
            peak_fraction=train.replacement_warmup_fraction,
        )
        if train.replacement_schedule == "constant"
        else BellCosineReplacementSchedule(
            max_ratio=train.replacement_max_ratio,
            start_fraction=train.replacement_start_fraction,
            peak_fraction=train.replacement_warmup_fraction,
            decay_start_fraction=train.replacement_decay_start_fraction,
            end_fraction=train.replacement_end_fraction,
        )
    )
    ratio = schedule(midpoint, train.max_steps)
    auxiliary_gate = ratio / train.replacement_max_ratio
    if train.replay_stream_mode == "all_records":
        probability = 1.0
    else:
        replay_weight = train.branch_loss_weight * auxiliary_gate
        probability = replay_weight / (train.clean_loss_weight + replay_weight)
    return probability, ratio, midpoint


def sparse_capture_totals(
    dense_target: torch.Tensor,
    target_ids: torch.Tensor,
    effective_non_gt_mass: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Return captured and dense non-GT mass summed over active tokens."""

    if dense_target.ndim != 3 or target_ids.shape != dense_target.shape[:2]:
        raise ValueError("dense target and target IDs have incompatible shapes")
    if (
        effective_non_gt_mass.shape != target_ids.shape
        or active_mask.shape != target_ids.shape
    ):
        raise ValueError("sparse mass and active mask must match target IDs")
    gt_mass = (
        dense_target.detach()
        .float()
        .gather(-1, target_ids.detach().long()[..., None])
        .squeeze(-1)
    )
    dense_non_gt = (1.0 - gt_mass).clamp(0.0, 1.0)
    captured = effective_non_gt_mass.detach().float().clamp(0.0, 1.0)
    mask = active_mask.detach().to(dtype=torch.float32)
    return torch.stack(((captured * mask).sum(), (dense_non_gt * mask).sum())).double()


def target_uses_configured_alpha(target_mode: str) -> bool:
    """Return whether the replay target formula consumes ``training.alpha``."""

    return target_mode == "support_gated_soft"


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast("cpu", enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16)


def main() -> None:
    args = parse_args()
    if args.expected_step < 0 or args.batch_size <= 0:
        raise ValueError("expected step and batch size must be valid")
    if args.limit_to_training_steps is not None and args.limit_to_training_steps <= 0:
        raise ValueError("limit-to-training-steps must be positive")
    expected_checkpoint_step = (
        args.expected_step
        if args.expected_checkpoint_step is None
        else args.expected_checkpoint_step
    )
    if expected_checkpoint_step < 0:
        raise ValueError("expected checkpoint step must be non-negative")
    config = load_experiment_config(args.config)
    train = config.training
    if train.history_source != "lagged_replay":
        raise ValueError("replay bank builder requires history_source=lagged_replay")
    if args.expected_step % train.replay_refresh_steps:
        raise ValueError("snapshot step must align to replay_refresh_steps")
    if not train.replay_start_step <= args.expected_step < train.replay_end_step:
        raise ValueError("snapshot step is outside the active replay interval")

    context = initialize_distributed()
    seed_everything(train.seed + args.expected_step, context.rank, deterministic=False)
    destination = args.output.expanduser().resolve()
    staging = destination.with_name(destination.name + ".partial")
    if context.is_main:
        if destination.exists() or staging.exists():
            raise FileExistsError(destination if destination.exists() else staging)
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
    barrier()

    dataset = PretokenizedImageNetDataset(config.data.train_manifest, split="train")
    tokenizer_sha256 = verify_manifest_tokenizer_checkpoint(
        dataset.manifest, config.data.tokenizer_checkpoint
    )
    derived_selection_probability, replacement_ratio, interval_midpoint = (
        replay_selection_probability(train, args.expected_step)
    )
    population_selected_count = max(
        1, round(len(dataset) * derived_selection_probability)
    )
    selection_probability = population_selected_count / len(dataset)
    selection_rng = np.random.default_rng(train.seed + args.expected_step + 91_337)
    if train.replay_stream_mode == "all_records":
        selected_indices = np.arange(len(dataset), dtype=np.int64)
    else:
        selected_indices = np.sort(
            selection_rng.choice(
                len(dataset), size=population_selected_count, replace=False
            )
        ).astype(np.int64, copy=False)
    planned_record_count = None
    if args.limit_to_training_steps is not None:
        planned_indices = planned_training_record_indices(
            len(dataset),
            train,
            world_size=context.world_size,
            optimizer_steps=args.limit_to_training_steps,
        )
        planned_record_count = int(planned_indices.size)
        selected_indices = np.intersect1d(
            selected_indices, planned_indices, assume_unique=True
        )
    selected_count = int(selected_indices.size)
    if selected_count <= 0:
        raise RuntimeError("bounded replay bank selected no records")
    selected_start, selected_end = contiguous_partition(
        selected_count, context.rank, context.world_size
    )
    local_indices = selected_indices[selected_start:selected_end]
    class_drop_seed = train.seed + args.expected_step * 1_000_003 + 814_067
    local_class_drop_mask = deterministic_replay_class_drop_mask(
        local_indices,
        probability=config.model.class_dropout,
        seed=class_drop_seed,
    )
    local_count = local_indices.size
    required_bytes = estimated_bank_bytes(
        selected_count,
        config.model.sequence_length,
        train.replay_top_k,
        conditional_equal_weights=(
            train.replay_target_encoding == "conditional_non_gt_mc"
        ),
    )
    if context.is_main and shutil.disk_usage(staging.parent).free < math.ceil(
        required_bytes * 1.2
    ):
        raise RuntimeError("insufficient disk for replay bank plus safety margin")

    codebook = load_maskgit_codebook(config.data.tokenizer_checkpoint)
    kernel = build_codebook_cosine_kernel(
        codebook, temperature=train.codebook_kernel_temperature
    ).to(context.device)
    model = build_nti(
        config.model,
        icb_config=config.icb,
    ).to(context.device)
    state = load_checkpoint(
        args.checkpoint,
        model=model,
        expected_config=config.to_dict(),
        required_config_sections=("model", "icb"),
        restore_rng=False,
    )
    if int(state["step"]) != expected_checkpoint_step:
        raise RuntimeError(
            "checkpoint step mismatch: expected "
            f"{expected_checkpoint_step}, got {state['step']}"
        )
    model.eval().requires_grad_(False)

    schedule = (
        LinearReplacementSchedule(
            start_ratio=train.replacement_max_ratio,
            max_ratio=train.replacement_max_ratio,
            peak_fraction=train.replacement_warmup_fraction,
        )
        if train.replacement_schedule == "constant"
        else BellCosineReplacementSchedule(
            max_ratio=train.replacement_max_ratio,
            start_fraction=train.replacement_start_fraction,
            peak_fraction=train.replacement_warmup_fraction,
            decay_start_fraction=train.replacement_decay_start_fraction,
            end_fraction=train.replacement_end_fraction,
        )
    )
    two_pass = TwoPassConfig(
        target_mode=train.target_mode,
        soft_alpha=train.alpha,
        sampling_temperature=train.history_temperature,
        sampling_top_k=train.history_top_k,
        sampling_top_p=train.history_top_p,
        replacement_ratio_sampling=train.replacement_ratio_sampling,
        branch_image_fraction=1.0,
        branch_loss_scope="active_prefix",
        clean_loss_weight=1.0,
        branch_loss_weight=1.0,
        schedule=schedule,
    )

    sequence = config.model.sequence_length
    top_k = train.replay_top_k
    categorical_generator = None
    categorical_seed = None
    if train.replay_target_encoding in {"categorical_mc", "conditional_non_gt_mc"}:
        categorical_seed = (
            train.seed
            + args.expected_step * 1_000_003
            + context.rank * 97_409
            + 260_826
        )
        categorical_generator = torch.Generator(device=context.device)
        categorical_generator.manual_seed(categorical_seed)
    arrays = {
        "record_indices": local_indices,
        "class_labels": np.empty(local_count, dtype=np.int16),
        "altered_tokens": np.empty((local_count, sequence), dtype=np.uint16),
        "active_mask": np.empty((local_count, sequence), dtype=np.bool_),
        "effective_non_gt_mass": np.empty((local_count, sequence), dtype=np.float16),
        "top_indices": np.empty((local_count, sequence, top_k), dtype=np.uint16),
        "top_probabilities": np.empty(
            (local_count, sequence, top_k),
            dtype=(
                np.uint8
                if train.replay_target_encoding == "conditional_non_gt_mc"
                else np.float16
            ),
        ),
    }
    started = time.time()
    capture_totals = torch.zeros(2, dtype=torch.float64, device=context.device)
    with torch.inference_mode():
        for offset in range(0, local_count, args.batch_size):
            stop = min(offset + args.batch_size, local_count)
            indices = local_indices[offset:stop].tolist()
            items = [dataset[index] for index in indices]
            tokens = torch.stack([item["image_tokens"] for item in items]).to(
                context.device
            )
            labels = torch.stack([item["class_labels"] for item in items]).to(
                context.device
            )
            drop = torch.as_tensor(
                local_class_drop_mask[offset:stop],
                dtype=torch.bool,
                device=context.device,
            )
            with _autocast(context.device, args.precision):
                output = two_pass_forward(
                    model,
                    tokens,
                    labels,
                    config=two_pass,
                    codebook_kernel=kernel,
                    replacement_ratio=replacement_ratio,
                    model_kwargs={"class_drop_mask": drop},
                )
            inverse = torch.empty_like(output.branch_indices)
            inverse[output.branch_indices] = torch.arange(
                output.branch_indices.numel(), device=context.device
            )
            target = output.branch_target
            if target is None:
                if train.target_mode != "hard_gt":
                    raise AssertionError("dynamic target builder returned no target")
                target = torch.nn.functional.one_hot(
                    tokens.index_select(0, output.branch_indices),
                    num_classes=config.model.vocab_size,
                ).float()
            target = target.index_select(0, inverse)
            if train.replay_target_encoding == "categorical_mc":
                sparse = categorical_sample_detached_target(
                    target,
                    tokens,
                    sample_count=top_k,
                    generator=categorical_generator,
                )
            elif train.replay_target_encoding == "conditional_non_gt_mc":
                sparse = conditional_non_gt_sample_detached_target(
                    target,
                    tokens,
                    sample_count=top_k,
                    generator=categorical_generator,
                )
            else:
                sparse = sparsify_detached_target(target, tokens, top_k=top_k)
            active_mask = output.active_prefix_mask.index_select(0, inverse)
            if train.replay_stream_mode == "all_records":
                sparse = replace(
                    sparse,
                    effective_non_gt_mass=(
                        sparse.effective_non_gt_mass * active_mask.float()
                    ).detach(),
                )
            capture_totals += sparse_capture_totals(
                target, tokens, sparse.effective_non_gt_mass, active_mask
            )
            target_slice = slice(offset, stop)
            arrays["class_labels"][target_slice] = labels.cpu().numpy().astype(np.int16)
            arrays["altered_tokens"][target_slice] = (
                output.mixed_tokens.index_select(0, inverse)
                .cpu()
                .numpy()
                .astype(np.uint16)
            )
            arrays["active_mask"][target_slice] = active_mask.cpu().numpy()
            arrays["effective_non_gt_mass"][target_slice] = (
                sparse.effective_non_gt_mass.cpu().numpy().astype(np.float16)
            )
            arrays["top_indices"][target_slice] = (
                sparse.top_indices.cpu().numpy().astype(np.uint16)
            )
            arrays["top_probabilities"][target_slice] = (
                sparse.top_probabilities.cpu()
                .numpy()
                .astype(arrays["top_probabilities"].dtype)
            )

    active_counts = torch.tensor(
        [int(arrays["active_mask"].sum()), int(arrays["active_mask"].size)],
        dtype=torch.float64,
        device=context.device,
    )
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(active_counts, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(capture_totals, op=torch.distributed.ReduceOp.SUM)
    active_token_fraction = float((active_counts[0] / active_counts[1]).item())
    if not 0.0 < active_token_fraction <= 1.0:
        raise RuntimeError("replay bank has no active target tokens")
    dense_non_gt_total = float(capture_totals[1].item())
    non_gt_mass_ratio = (
        float((capture_totals[0] / capture_totals[1]).item())
        if dense_non_gt_total > 0.0
        else 1.0
    )
    if not math.isfinite(non_gt_mass_ratio) or non_gt_mass_ratio < 0.0:
        raise RuntimeError("invalid replay target non-GT mass ratio")
    if train.replay_target_encoding == "sparse_topk" and non_gt_mass_ratio > 1.0 + 1e-6:
        raise RuntimeError("invalid sparse target capture fraction")
    sparse_capture_fraction = (
        non_gt_mass_ratio if train.replay_target_encoding == "sparse_topk" else None
    )
    sampled_non_gt_mass_ratio = (
        non_gt_mass_ratio if train.replay_target_encoding == "categorical_mc" else None
    )

    checkpoint_sha256 = sha256_file(args.checkpoint)
    shard_name = f"shard_{context.rank:04d}"
    write_replay_bank(
        staging / shard_name,
        arrays,
        provenance={
            "rank": context.rank,
            "world_size": context.world_size,
            "snapshot_step": args.expected_step,
            "teacher_checkpoint_step": expected_checkpoint_step,
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "target_mode": train.target_mode,
            "configured_alpha_used": target_uses_configured_alpha(train.target_mode),
            "top_k": top_k,
            "target_encoding": train.replay_target_encoding,
            "replay_stream_mode": train.replay_stream_mode,
            "history_draw_count_per_record": 1,
            "history_sampling_policy": "single_altered_sequence_per_record_v1",
            "target_weight_storage": (
                "implicit_equal_uint8"
                if train.replay_target_encoding == "conditional_non_gt_mc"
                else "explicit_float16"
            ),
            "categorical_seed": categorical_seed,
            "replacement_ratio": replacement_ratio,
            "selection_probability": selection_probability,
            "derived_selection_probability": derived_selection_probability,
            "selection_population_record_count": population_selected_count,
            "planned_training_steps": args.limit_to_training_steps,
            "planned_record_count": planned_record_count,
            "planned_sampler": (
                "torch_distributed_sampler_epoch0_shuffle_drop_last_v1"
                if args.limit_to_training_steps is not None
                else None
            ),
            "planned_world_size": (
                context.world_size if args.limit_to_training_steps is not None else None
            ),
            "planned_global_batch_size": (
                train.global_batch_size
                if args.limit_to_training_steps is not None
                else None
            ),
            "planned_per_device_batch_size": (
                train.per_device_batch_size
                if args.limit_to_training_steps is not None
                else None
            ),
            "planned_training_seed": (
                train.seed if args.limit_to_training_steps is not None else None
            ),
            "active_token_fraction": active_token_fraction,
            "sparse_non_gt_mass_capture_fraction": sparse_capture_fraction,
            "sampled_non_gt_mass_ratio": sampled_non_gt_mass_ratio,
            "interval_midpoint_step": interval_midpoint,
            "class_drop_policy": "splitmix64_record_index_v1",
            "class_drop_probability": config.model.class_dropout,
            "class_drop_seed": class_drop_seed,
            "tokenizer_sha256": tokenizer_sha256,
        },
    )
    barrier()
    if context.is_main:
        shards = []
        for rank in range(context.world_size):
            shard_offset_start, shard_offset_end = contiguous_partition(
                selected_count, rank, context.world_size
            )
            shard_indices = selected_indices[shard_offset_start:shard_offset_end]
            shards.append(
                {
                    "path": f"shard_{rank:04d}",
                    "start_record_index": int(shard_indices[0]),
                    "end_record_index_exclusive": int(shard_indices[-1]) + 1,
                    "record_count": int(shard_indices.size),
                }
            )
        write_sharded_replay_index(
            staging,
            shards=shards,
            provenance={
                "snapshot_step": args.expected_step,
                "teacher_checkpoint_step": expected_checkpoint_step,
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "target_mode": train.target_mode,
                "configured_alpha_used": target_uses_configured_alpha(
                    train.target_mode
                ),
                "top_k": top_k,
                "target_encoding": train.replay_target_encoding,
                "replay_stream_mode": train.replay_stream_mode,
                "history_draw_count_per_record": 1,
                "history_sampling_policy": "single_altered_sequence_per_record_v1",
                "target_weight_storage": (
                    "implicit_equal_uint8"
                    if train.replay_target_encoding == "conditional_non_gt_mc"
                    else "explicit_float16"
                ),
                "categorical_seed_formula": (
                    "training_seed + snapshot_step*1000003 + rank*97409 + 260826"
                    if train.replay_target_encoding
                    in {"categorical_mc", "conditional_non_gt_mc"}
                    else None
                ),
                "replacement_ratio": replacement_ratio,
                "selection_probability": selection_probability,
                "derived_selection_probability": derived_selection_probability,
                "selection_population_record_count": population_selected_count,
                "planned_training_steps": args.limit_to_training_steps,
                "planned_record_count": planned_record_count,
                "planned_sampler": (
                    "torch_distributed_sampler_epoch0_shuffle_drop_last_v1"
                    if args.limit_to_training_steps is not None
                    else None
                ),
                "planned_world_size": (
                    context.world_size
                    if args.limit_to_training_steps is not None
                    else None
                ),
                "planned_global_batch_size": (
                    train.global_batch_size
                    if args.limit_to_training_steps is not None
                    else None
                ),
                "planned_per_device_batch_size": (
                    train.per_device_batch_size
                    if args.limit_to_training_steps is not None
                    else None
                ),
                "planned_training_seed": (
                    train.seed if args.limit_to_training_steps is not None else None
                ),
                "active_token_fraction": active_token_fraction,
                "sparse_non_gt_mass_capture_fraction": sparse_capture_fraction,
                "sampled_non_gt_mass_ratio": sampled_non_gt_mass_ratio,
                "interval_midpoint_step": interval_midpoint,
                "class_drop_policy": "splitmix64_record_index_v1",
                "class_drop_probability": config.model.class_dropout,
                "class_drop_seed": class_drop_seed,
                "dataset_record_count": len(dataset),
                "tokenizer_sha256": tokenizer_sha256,
                "manifest_sha256": sha256_file(config.data.train_manifest),
                "world_size": context.world_size,
                "per_rank_batch_size": args.batch_size,
                "runtime_seconds": time.time() - started,
                "captured_target_tail_policy": (
                    "unbiased categorical Monte Carlo samples from the full target"
                    if train.replay_target_encoding == "categorical_mc"
                    else (
                        "exact non-GT dose with equal-weight conditional samples"
                        if train.replay_target_encoding == "conditional_non_gt_mc"
                        else "omitted non-GT mass returns to GT"
                    )
                ),
            },
        )
        os.replace(staging, destination)
        print(
            json.dumps(
                {
                    "event": "replay_bank_complete",
                    "output": str(destination),
                    "record_count": selected_count,
                    "selection_population_record_count": population_selected_count,
                    "planned_training_steps": args.limit_to_training_steps,
                    "planned_record_count": planned_record_count,
                    "dataset_record_count": len(dataset),
                    "teacher_checkpoint_step": expected_checkpoint_step,
                    "selection_probability": selection_probability,
                    "active_token_fraction": active_token_fraction,
                    "sparse_non_gt_mass_capture_fraction": sparse_capture_fraction,
                    "sampled_non_gt_mass_ratio": sampled_non_gt_mass_ratio,
                    "estimated_bytes": required_bytes,
                    "replacement_ratio": replacement_ratio,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    barrier()
    shutdown_distributed()


if __name__ == "__main__":
    main()
