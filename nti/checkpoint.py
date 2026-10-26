from __future__ import annotations

import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


_TRAINING_CHECKPOINT_PATTERN = re.compile(r"^step_(\d+)\.pt$")


def _config_section(config: Mapping[str, Any], section: str) -> object:
    value = config.get(section)
    if section == "token_interface" and section not in config:
        # Checkpoints written before the unified interface stored only icb.enabled.
        legacy_icb = config.get("icb")
        if not isinstance(legacy_icb, Mapping):
            return value
        return {
            "mode": "icb" if bool(legacy_icb.get("enabled", False)) else "free",
            "read_weight": 0.0,
            "write_weight": 0.0,
        }
    if section == "training" and isinstance(value, Mapping):
        # Pre-annealing checkpoints used the original ramp-and-hold schedule.
        normalized = dict(value)
        normalized.setdefault("replacement_schedule", "linear_hold")
        normalized.setdefault("replacement_ratio_sampling", "fixed")
        normalized.setdefault("replacement_start_fraction", 0.0)
        normalized.setdefault("replacement_decay_start_fraction", 0.5)
        normalized.setdefault("replacement_end_fraction", 1.0)
        normalized.setdefault("branch_active_only_when_replacement_positive", False)
        normalized.setdefault("auxiliary_loss_gate", "constant")
        normalized.setdefault("branch_image_fraction", 1.0)
        return normalized
    if section == "monitoring" and isinstance(value, Mapping):
        normalized = dict(value)
        preview_class_ids = normalized.get("preview_class_ids")
        if isinstance(preview_class_ids, (list, tuple)):
            normalized["preview_class_ids"] = tuple(preview_class_ids)
        return normalized
    return value


def checkpoint_reached_max_steps(state: Mapping[str, int], max_steps: int) -> bool:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    return int(state.get("step", 0)) >= max_steps


def prune_training_checkpoints(
    checkpoint_dir: str | Path,
    *,
    milestone_every: int = 0,
    keep_last: int = 0,
) -> tuple[Path, ...]:
    """Remove non-milestone checkpoints outside a bounded recent window."""

    if milestone_every < 0 or keep_last < 0:
        raise ValueError("checkpoint retention values must be non-negative")
    if milestone_every == 0 and keep_last == 0:
        return ()
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)

    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.iterdir():
        match = _TRAINING_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None and path.is_file() and not path.is_symlink():
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        return ()

    recent_count = max(keep_last, 1)
    preserved = {path for _, path in checkpoints[-recent_count:]}
    if milestone_every > 0:
        preserved.update(
            path for step, path in checkpoints if step % milestone_every == 0
        )

    removed: list[Path] = []
    for _, path in checkpoints:
        if path not in preserved:
            path.unlink()
            removed.append(path)
    return tuple(removed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def _seed_rng_for_rank(seed: int, rank: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("geometry migration RNG seed must be an integer")
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    rank_seed = int(seed) + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank_seed)


def _validate_geometry_migration_rng_seed(seed: object, world_size: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise RuntimeError("geometry migration RNG seed must be an integer")
    normalized = int(seed)
    if normalized < 0 or normalized + world_size - 1 >= 2**32:
        raise RuntimeError(
            "geometry migration RNG seed and target ranks must fit NumPy's "
            "32-bit seed range"
        )
    return normalized


def _validate_rng_state(
    state: object, *, label: str, require_cuda: bool = False
) -> None:
    if not isinstance(state, Mapping):
        raise RuntimeError(f"{label} RNG state is not a mapping")
    missing = {"python", "numpy", "torch"} - set(state)
    if require_cuda and "cuda" not in state:
        missing.add("cuda")
    if missing:
        raise RuntimeError(f"{label} RNG state is missing {sorted(missing)}")
    try:
        python_rng = random.Random()
        python_rng.setstate(state["python"])
        numpy_rng = np.random.RandomState()
        numpy_rng.set_state(state["numpy"])
        torch_rng = torch.Generator()
        torch_rng.set_state(state["torch"])
        if "cuda" in state:
            if not isinstance(state["cuda"], torch.Tensor):
                raise TypeError("CUDA RNG state is not a tensor")
            if torch.cuda.is_available():
                cuda_rng = torch.Generator(
                    device=torch.device("cuda", torch.cuda.current_device())
                )
                cuda_rng.set_state(state["cuda"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"{label} RNG state is invalid: {error}") from error


def _unwrapped(model: nn.Module) -> nn.Module:
    current = model
    while True:
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "_orig_mod"):
            current = current._orig_mod
            continue
        return current


def validate_checkpoint_config(
    checkpoint_config: object,
    expected_config: Mapping[str, Any],
    *,
    sections: Sequence[str],
    ignored_keys: Sequence[str] = (),
    allowed_migrations: Optional[Mapping[str, tuple[object, object]]] = None,
    require_migrations: bool = False,
) -> None:
    """Bind runtime behavior to the model and token-interface training config."""

    if not isinstance(checkpoint_config, Mapping):
        raise RuntimeError("checkpoint has no structured config")
    ignored = set(ignored_keys)
    migrations = dict(allowed_migrations or {})
    applied_migrations: set[str] = set()
    mismatches: Dict[str, Dict[str, object]] = {}
    for section in sections:
        actual = _config_section(checkpoint_config, section)
        expected = _config_section(expected_config, section)
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            mismatches[section] = {"expected": expected, "actual": actual}
            continue
        keys = sorted(set(actual) | set(expected))
        for key in keys:
            qualified_key = f"{section}.{key}"
            if qualified_key in ignored:
                continue
            actual_value = actual[key] if key in actual else "<missing>"
            expected_value = expected[key] if key in expected else "<missing>"
            if actual_value != expected_value:
                allowed_transition = migrations.get(qualified_key)
                if allowed_transition == (actual_value, expected_value):
                    applied_migrations.add(qualified_key)
                    continue
                mismatches[qualified_key] = {
                    "expected": expected_value,
                    "actual": actual_value,
                }
    if require_migrations:
        for qualified_key, transition in migrations.items():
            if qualified_key not in applied_migrations:
                mismatches[f"required_migration.{qualified_key}"] = {
                    "expected": transition,
                    "actual": "transition was not applied",
                }
    if mismatches:
        raise RuntimeError(
            "checkpoint inference config does not match the requested config: "
            f"{mismatches}"
        )


_TRAINING_GEOMETRY_KEYS = (
    "world_size",
    "per_device_batch_size",
    "accumulation_steps",
)


def _normalize_training_geometry(geometry: object, *, label: str) -> Dict[str, int]:
    if not isinstance(geometry, Mapping):
        raise RuntimeError(f"{label} training geometry is not a mapping")
    normalized: Dict[str, int] = {}
    for key in _TRAINING_GEOMETRY_KEYS:
        if key not in geometry:
            raise RuntimeError(f"{label} training geometry is missing {key}")
        value = geometry[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise RuntimeError(f"{label} training geometry {key} is not an integer")
        normalized[key] = int(value)
        if normalized[key] <= 0:
            raise RuntimeError(f"{label} training geometry {key} must be positive")
    return normalized


def _optimizer_steps_per_epoch(
    dataset_records: int, geometry: Mapping[str, int]
) -> int:
    world_size = geometry["world_size"]
    per_device_batch_size = geometry["per_device_batch_size"]
    accumulation_steps = geometry["accumulation_steps"]
    samples_per_rank = dataset_records // world_size
    micro_batches_per_rank = samples_per_rank // per_device_batch_size
    return micro_batches_per_rank // accumulation_steps


def migrate_training_state_geometry(
    state: Mapping[str, int],
    *,
    checkpoint_geometry: object,
    current_geometry: object,
    dataset_records: int,
) -> Dict[str, Any]:
    """Remap an optimizer-boundary cursor while preserving the global batch."""

    if dataset_records <= 0:
        raise ValueError("dataset_records must be positive for geometry migration")
    source = _normalize_training_geometry(checkpoint_geometry, label="checkpoint")
    target = _normalize_training_geometry(current_geometry, label="current")
    source_global_batch = math.prod(source.values())
    target_global_batch = math.prod(target.values())
    if source_global_batch != target_global_batch:
        raise RuntimeError(
            "training geometry migration must preserve effective global batch: "
            f"checkpoint={source_global_batch}, current={target_global_batch}"
        )

    source_steps_per_epoch = _optimizer_steps_per_epoch(dataset_records, source)
    target_steps_per_epoch = _optimizer_steps_per_epoch(dataset_records, target)
    if source_steps_per_epoch <= 0 or target_steps_per_epoch <= 0:
        raise RuntimeError("training geometry has no complete optimizer step per epoch")
    if source_steps_per_epoch != target_steps_per_epoch:
        raise RuntimeError(
            "training geometry migration changes optimizer steps per epoch: "
            f"checkpoint={source_steps_per_epoch}, current={target_steps_per_epoch}"
        )

    source_batch = int(state.get("batch_in_epoch", 0))
    if source_batch < 0:
        raise RuntimeError("checkpoint batch_in_epoch must be nonnegative")
    source_accumulation = source["accumulation_steps"]
    if source_batch % source_accumulation:
        raise RuntimeError(
            "checkpoint geometry migration requires an optimizer-step boundary"
        )
    completed_steps_in_epoch = source_batch // source_accumulation
    if completed_steps_in_epoch > source_steps_per_epoch:
        raise RuntimeError("checkpoint batch_in_epoch exceeds the source epoch")

    source_micro_batch = source["world_size"] * source["per_device_batch_size"]
    target_micro_batch = target["world_size"] * target["per_device_batch_size"]
    consumed_samples = source_batch * source_micro_batch
    target_batch, remainder = divmod(consumed_samples, target_micro_batch)
    if remainder:
        raise RuntimeError("training cursor cannot be represented by current geometry")
    if target_batch % target["accumulation_steps"]:
        raise RuntimeError("migrated cursor is not an optimizer-step boundary")
    target_complete_micro_batches = (
        target_steps_per_epoch * target["accumulation_steps"]
    )
    if target_batch > target_complete_micro_batches:
        raise RuntimeError("migrated batch_in_epoch exceeds the current epoch")

    source_epoch = int(state.get("epoch", 0))
    if source_epoch < 0:
        raise RuntimeError("checkpoint epoch must be nonnegative")
    if completed_steps_in_epoch == source_steps_per_epoch:
        target_epoch = source_epoch + 1
        target_batch = 0
    else:
        target_epoch = source_epoch

    expanding_world_size = target["world_size"] > source["world_size"]
    migrated = {
        "step": int(state.get("step", 0)),
        "epoch": target_epoch,
        "batch_in_epoch": target_batch,
        "geometry_migration": {
            "source_geometry": source,
            "target_geometry": target,
            "source_epoch": source_epoch,
            "target_epoch": target_epoch,
            "source_batch_in_epoch": source_batch,
            "target_batch_in_epoch": target_batch,
            "completed_optimizer_steps_in_epoch": completed_steps_in_epoch,
            "optimizer_steps_per_epoch": source_steps_per_epoch,
            "rng_policy": (
                "restore_existing_ranks_seed_added_ranks"
                if expanding_world_size
                else "restore_same_logical_rank"
            ),
        },
    }
    return migrated


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    config: Mapping[str, Any],
    rng_states: Optional[list[Mapping[str, Any]]] = None,
    training_geometry: Optional[Mapping[str, int]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": _unwrapped(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "config": dict(config),
        "rng_states": rng_states or [capture_rng_state()],
        "training_geometry": dict(training_geometry or {}),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    rank: int = 0,
    expected_training_geometry: Optional[Mapping[str, int]] = None,
    expected_config: Optional[Mapping[str, Any]] = None,
    required_config_sections: Sequence[str] = (),
    ignored_config_keys: Sequence[str] = (),
    allowed_config_migrations: Optional[Mapping[str, tuple[object, object]]] = None,
    require_config_migrations: bool = False,
    restore_rng: bool = True,
    allow_training_geometry_migration: bool = False,
    migration_dataset_records: Optional[int] = None,
    geometry_migration_rng_seed: Optional[int] = None,
    include_scheduler_state: bool = False,
    require_complete_rng_states: bool = False,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if required_config_sections and expected_config is None:
        raise ValueError("required_config_sections needs expected_config")
    if allow_training_geometry_migration and expected_training_geometry is None:
        raise ValueError("training geometry migration needs expected_training_geometry")
    if expected_config is not None:
        validate_checkpoint_config(
            checkpoint.get("config"),
            expected_config,
            sections=required_config_sections,
            ignored_keys=ignored_config_keys,
            allowed_migrations=allowed_config_migrations,
            require_migrations=require_config_migrations,
        )

    resume_state: Dict[str, Any] = {
        "step": int(checkpoint.get("step", 0)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "batch_in_epoch": int(checkpoint.get("batch_in_epoch", 0)),
    }
    if include_scheduler_state:
        scheduler_state = checkpoint.get("scheduler")
        if not isinstance(scheduler_state, Mapping):
            raise RuntimeError("checkpoint has no structured scheduler state")
        resume_state["checkpoint_scheduler_state"] = dict(scheduler_state)
    geometry_migrated = False
    states = checkpoint.get("rng_states", [])
    if require_complete_rng_states:
        if expected_training_geometry is None:
            raise ValueError(
                "complete RNG-state validation needs expected_training_geometry"
            )
        if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
            raise RuntimeError("checkpoint RNG states are not a sequence")
        expected_rng_count = int(expected_training_geometry["world_size"])
        if len(states) != expected_rng_count:
            raise RuntimeError(
                "checkpoint RNG state count does not match the expected world size: "
                f"states={len(states)}, world_size={expected_rng_count}"
            )
        for state_rank, rng_state in enumerate(states):
            _validate_rng_state(
                rng_state,
                label=f"checkpoint rank {state_rank}",
                require_cuda=True,
            )
        resume_state["rng_state_count"] = len(states)
    if expected_training_geometry is not None:
        actual_geometry = checkpoint.get("training_geometry", {})
        if dict(actual_geometry) != dict(expected_training_geometry):
            if not allow_training_geometry_migration:
                raise RuntimeError(
                    "resume training geometry changed: "
                    f"checkpoint={actual_geometry}, "
                    f"current={dict(expected_training_geometry)}"
                )
            if migration_dataset_records is None:
                raise ValueError(
                    "training geometry migration needs migration_dataset_records"
                )
            resume_state = migrate_training_state_geometry(
                resume_state,
                checkpoint_geometry=actual_geometry,
                current_geometry=expected_training_geometry,
                dataset_records=migration_dataset_records,
            )
            geometry_migrated = True

    if rank < 0:
        raise ValueError("rank must be nonnegative")
    if geometry_migrated and not restore_rng:
        raise RuntimeError("training geometry migration requires RNG restoration")
    if restore_rng and states:
        if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
            raise RuntimeError("checkpoint RNG states are not a sequence")
        if geometry_migrated:
            source_geometry = resume_state["geometry_migration"]["source_geometry"]
            target_geometry = resume_state["geometry_migration"]["target_geometry"]
            if len(states) != source_geometry["world_size"]:
                raise RuntimeError(
                    "checkpoint RNG state count does not match its world size: "
                    f"states={len(states)}, world_size={source_geometry['world_size']}"
                )
            if rank >= target_geometry["world_size"]:
                raise RuntimeError(
                    "rank exceeds the migrated training world size: "
                    f"rank={rank}, world_size={target_geometry['world_size']}"
                )
            for state_rank, rng_state in enumerate(states):
                _validate_rng_state(
                    rng_state,
                    label=f"checkpoint rank {state_rank}",
                    require_cuda=torch.cuda.is_available(),
                )
            if target_geometry["world_size"] > source_geometry["world_size"]:
                if geometry_migration_rng_seed is None:
                    raise RuntimeError(
                        "expanding training world size requires "
                        "geometry_migration_rng_seed"
                    )
                new_rank_seed_base = int(geometry_migration_rng_seed) + int(
                    resume_state["step"]
                )
                resume_state["geometry_migration"]["new_rank_seed_base"] = (
                    _validate_geometry_migration_rng_seed(
                        new_rank_seed_base, target_geometry["world_size"]
                    )
                )
        else:
            if rank >= len(states):
                raise RuntimeError(
                    "checkpoint has no RNG state for this distributed rank"
                )
            _validate_rng_state(states[rank], label=f"checkpoint rank {rank}")
    elif geometry_migrated:
        raise RuntimeError("training geometry migration checkpoint has no RNG states")

    _unwrapped(model).load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng and states:
        if geometry_migrated and rank >= len(states):
            new_rank_seed_base = resume_state["geometry_migration"][
                "new_rank_seed_base"
            ]
            _seed_rng_for_rank(new_rank_seed_base, rank)
        else:
            restore_rng_state(states[rank])
    return resume_state
