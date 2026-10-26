from __future__ import annotations
import argparse
import contextlib
from dataclasses import replace
import gc
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from nti.checkpoint import (
    capture_rng_state,
    checkpoint_reached_max_steps,
    load_checkpoint,
    prune_training_checkpoints,
    save_checkpoint,
)
from nti.config import TokenInterfaceConfig, TrainingConfig, load_experiment_config
from nti.data import (
    PretokenizedImageNetDataset,
    sha256_file,
    validate_manifest,
    verify_manifest_tokenizer_checkpoint,
)
from nti.distributed import (
    barrier,
    gather_rng_states,
    initialize_distributed,
    reduce_metrics,
    seed_everything,
    shutdown_distributed,
)
from nti.generation import SamplingConfig
from nti.gradient_monitoring import (
    collect_preclip_gradient_metrics,
    collect_rear_style_gradient_metrics,
)
from nti.monitoring import (
    generate_training_preview,
    iter_preview_manifest,
    preserve_all_rng_states,
)
from nti.models import build_nti
from nti.optim import build_lr_scheduler, build_optimizer
from nti.tokenizer import load_frozen_maskgit_vqgan, load_maskgit_codebook
from nti.training import (
    BellCosineReplacementSchedule,
    LinearReplacementSchedule,
    RampCosineReplacementSchedule,
    TwoPassConfig,
    align_available_replay_batch,
    build_codebook_cosine_affinity,
    build_codebook_cosine_kernel,
    hard_token_cross_entropy,
    masked_token_mean,
    one_pass_replay_forward,
    open_replay_bank,
    resolve_replay_bank_root,
    two_pass_forward,
)

_RESUME_CONFIG_SECTIONS = ("model", "icb", "token_interface", "training", "data", "run")
_RESUME_IGNORED_CONFIG_KEYS = (
    "training.per_device_batch_size",
    "training.log_every",
    "training.save_every",
    "training.checkpoint_milestone_every",
    "training.checkpoint_keep_last",
    "run.name",
    "run.output_dir",
    "run.resume",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NTI from pretokenized ImageNet")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-model-checkpoint",
        type=Path,
        default=None,
        help="initialize model weights from a training or eval checkpoint while starting a fresh optimizer, scheduler, RNG stream, and data cursor",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--stop-at-step",
        type=int,
        default=None,
        help="save and exit exactly at a replay refresh boundary",
    )
    parser.add_argument(
        "--allow-training-geometry-migration",
        action="store_true",
        help="allow a resume with a different world/micro-batch geometry",
    )
    parser.add_argument(
        "--allow-replacement-schedule-migration",
        action="store_true",
        help="resume at the configured replacement peak in a new run directory while changing only the replacement schedule and end ratio",
    )
    None
    parser.add_argument(
        "--validate-resume-only",
        action="store_true",
        help="validate and load a resume checkpoint without starting training",
    )
    return parser.parse_args()


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    raise ValueError("precision must be fp32, bf16, or fp16")


def _validate_replacement_schedule_migration(
    train: TrainingConfig, state: Dict[str, int]
) -> int:
    peak_step = round(train.max_steps * train.replacement_warmup_fraction)
    if train.replacement_schedule != "ramp_cosine_zero":
        raise RuntimeError(
            "replacement schedule migration target must use ramp_cosine_zero"
        )
    if int(state["step"]) != peak_step:
        raise RuntimeError(
            f"replacement schedule migration must resume exactly at the configured peak step {peak_step}, got {state['step']}"
        )
    return peak_step


def _validate_resume_data_cursor(
    state: Mapping[str, Any],
    *,
    epochs: int,
    complete_micro_batches: int,
    accumulation_steps: int,
) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for key in ("epoch", "batch_in_epoch"):
        raw_value = state.get(key)
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise RuntimeError(f"checkpoint {key} must be an integer")
        values[key] = int(raw_value)
    if not 0 <= values["epoch"] < epochs:
        raise RuntimeError(
            f"checkpoint epoch is outside the configured training range: {values['epoch']} not in [0, {epochs})"
        )
    if not 0 <= values["batch_in_epoch"] <= complete_micro_batches:
        raise RuntimeError(
            f"checkpoint batch_in_epoch exceeds complete DataLoader batches: {values['batch_in_epoch']} not in [0, {complete_micro_batches}]"
        )
    if values["batch_in_epoch"] % accumulation_steps:
        raise RuntimeError(
            f"checkpoint batch_in_epoch is not an optimizer-step boundary: {values['batch_in_epoch']} % {accumulation_steps} != 0"
        )
    return {
        **values,
        "complete_micro_batches": complete_micro_batches,
        "accumulation_steps": accumulation_steps,
    }


def _branch_is_enabled(train: TrainingConfig, replacement_ratio: float) -> bool:
    if not math.isfinite(replacement_ratio) or not 0.0 <= replacement_ratio <= 1.0:
        raise ValueError("replacement_ratio must be finite and in [0, 1]")
    return (
        train.branch_loss_weight > 0.0
        and train.branch_image_fraction > 0.0
        and (
            not train.branch_active_only_when_replacement_positive
            or replacement_ratio > 0.0
        )
    )


def _auxiliary_loss_gate(train: TrainingConfig, replacement_ratio: float) -> float:
    """Return the deterministic gate applied to all method-specific losses."""
    if not math.isfinite(replacement_ratio) or not 0.0 <= replacement_ratio <= 1.0:
        raise ValueError("replacement_ratio must be finite and in [0, 1]")
    if train.auxiliary_loss_gate == "constant":
        return 1.0
    if train.auxiliary_loss_gate != "normalized_replacement":
        raise AssertionError(
            f"unsupported auxiliary loss gate {train.auxiliary_loss_gate}"
        )
    maximum = float(train.replacement_max_ratio)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(
            "normalized_replacement auxiliary loss gate requires a positive replacement_max_ratio"
        )
    tolerance = max(1e-12, maximum * 1e-09)
    if replacement_ratio > maximum + tolerance:
        raise ValueError(
            "replacement_ratio exceeds replacement_max_ratio for the normalized_replacement auxiliary loss gate"
        )
    return min(replacement_ratio / maximum, 1.0)


def _compute_training_objective(
    model: torch.nn.Module,
    tokens: Tensor,
    labels: Tensor,
    class_drop_mask: Tensor,
    *,
    train: TrainingConfig,
    interface: TokenInterfaceConfig,
    two_pass: TwoPassConfig,
    codebook_kernel: Tensor | None,
    codebook_affinity: Tensor | None = None,
    replacement_ratio: float,
    record_indices: Tensor | None = None,
    replay_bank: Any = None,
) -> tuple[Tensor, Dict[str, Tensor]]:
    """Compute one micro-batch objective with a shared auxiliary-loss gate."""
    auxiliary_gate = _auxiliary_loss_gate(train, replacement_ratio)
    branch_enabled = _branch_is_enabled(train, replacement_ratio)
    if not branch_enabled:
        logits = model(tokens, class_labels=labels, class_drop_mask=class_drop_mask)
        clean_loss = hard_token_cross_entropy(logits, tokens)
        inactive_weight = (
            train.branch_loss_weight
            if train.history_source == "lagged_replay"
            and train.replay_stream_mode == "all_records"
            else train.clean_loss_weight
        )
        raw_loss = inactive_weight * clean_loss
        batch_metrics = {
            "loss": raw_loss.detach(),
            "clean_loss": clean_loss.detach(),
            "branch_loss": clean_loss.detach().new_zeros(()),
            "branch_enabled": clean_loss.detach().new_zeros(()),
            "branch_image_fraction_requested": clean_loss.detach().new_tensor(
                train.branch_image_fraction
            ),
            "branch_image_fraction": clean_loss.detach().new_zeros(()),
            "branch_image_count": clean_loss.detach().new_zeros(()),
            "replacement_ratio": clean_loss.detach().new_tensor(replacement_ratio),
            "replacement_sampled_probability_mean": clean_loss.detach().new_zeros(()),
            "replacement_selected_fraction": clean_loss.detach().new_zeros(()),
            "history_changed_fraction": clean_loss.detach().new_zeros(()),
            "active_prefix_fraction": clean_loss.detach().new_zeros(()),
            "clean_top1": logits.detach().argmax(dim=-1).eq(tokens).float().mean(),
            "branch_top1": clean_loss.detach().new_zeros(()),
        }
    elif train.history_source == "lagged_replay":
        if replay_bank is None or record_indices is None:
            raise RuntimeError(
                "active lagged replay requires a loaded bank and record indices"
            )
        if interface.mode != "free":
            raise RuntimeError("lagged replay only supports the free token interface")
        effective_branch_weight = train.branch_loss_weight * auxiliary_gate
        all_records = train.replay_stream_mode == "all_records"
        if all_records:
            if not replay_bank.contains(record_indices).all():
                raise RuntimeError(
                    "all-record replay bank miss; clean fallback is forbidden"
                )
            replay_batch = replay_bank.lookup(record_indices)
            replay_image_mask = torch.ones_like(record_indices, dtype=torch.bool)
        else:
            replay_batch, replay_image_mask = align_available_replay_batch(
                replay_bank, record_indices, tokens, labels
            )
        bank_manifest = getattr(replay_bank, "manifest", {})
        bank_provenance = bank_manifest.get("provenance", {})
        bank_replacement_ratio = float(
            bank_provenance.get("replacement_ratio", replacement_ratio)
        )
        replay_selection_probability = float(
            bank_provenance.get("selection_probability", -1.0)
        )
        replay_active_token_fraction = float(
            bank_provenance.get("active_token_fraction", -1.0)
        )
        if all_records and replay_selection_probability != 1.0:
            raise RuntimeError("all-record replay requires selection_probability=1")
        if not all_records and (not 0.0 < replay_selection_probability < 1.0):
            raise RuntimeError("replay bank selection probability is invalid")
        if not 0.0 < replay_active_token_fraction <= 1.0:
            raise RuntimeError("replay bank active token fraction is invalid")
        output = one_pass_replay_forward(
            model,
            tokens,
            labels,
            replay_batch,
            replay_image_mask,
            clean_loss_weight=train.clean_loss_weight,
            replay_loss_weight=effective_branch_weight,
            replay_all_tokens=all_records,
            model_kwargs={
                "class_drop_mask": torch.where(
                    replay_image_mask, replay_batch.class_drop_mask, class_drop_mask
                )
            },
        )
        counts = torch.stack(
            [
                output.clean_token_count.detach(),
                output.replay_token_count.detach(),
                output.clean_token_count.detach().new_tensor(tokens.numel()),
            ]
        ).float()
        if dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            gradient_scale = float(dist.get_world_size())
        else:
            gradient_scale = 1.0
        global_potential_token_count = counts[2].clamp_min(1.0)
        if all_records:
            clean_loss = hard_token_cross_entropy(output.logits, tokens)
            replay_loss = output.replay_loss
            raw_loss = replay_loss
        else:
            clean_loss = (
                output.clean_loss_sum
                * gradient_scale
                / (global_potential_token_count * (1.0 - replay_selection_probability))
            )
            replay_loss = (
                output.replay_loss_sum
                * gradient_scale
                / (
                    global_potential_token_count
                    * replay_selection_probability
                    * replay_active_token_fraction
                )
            )
            raw_loss = (
                train.clean_loss_weight * clean_loss
                + effective_branch_weight * replay_loss
            )
        replay_count = replay_image_mask.sum()
        replay_token_mask = replay_image_mask[:, None].expand_as(tokens)
        changed_mask = replay_batch.altered_tokens.ne(tokens) & replay_token_mask
        clean_top1 = masked_token_mean(
            output.logits.detach().argmax(dim=-1).eq(tokens).float(),
            None if all_records else output.clean_loss_mask,
        )
        branch_top1 = masked_token_mean(
            output.logits.detach().argmax(dim=-1).eq(tokens).float(),
            output.replay_loss_mask,
        )
        batch_metrics = {
            "loss": raw_loss.detach(),
            "clean_loss": clean_loss.detach(),
            "branch_loss": replay_loss.detach(),
            "branch_enabled": clean_loss.detach().new_ones(()),
            "branch_image_fraction_requested": clean_loss.detach().new_tensor(
                effective_branch_weight
                / (train.clean_loss_weight + effective_branch_weight)
            ),
            "branch_image_fraction": replay_image_mask.float().mean(),
            "branch_image_count": replay_count,
            "replacement_ratio": clean_loss.detach().new_tensor(replacement_ratio),
            "replacement_sampled_probability_mean": clean_loss.detach().new_tensor(
                bank_replacement_ratio / 2.0
                if train.replacement_ratio_sampling == "per_image_uniform"
                else bank_replacement_ratio
            ),
            "replay_bank_replacement_ratio": clean_loss.detach().new_tensor(
                bank_replacement_ratio
            ),
            "replacement_selected_fraction": masked_token_mean(
                changed_mask.float(), replay_token_mask
            ),
            "history_changed_fraction": masked_token_mean(
                changed_mask.float(), replay_token_mask
            ),
            "active_prefix_fraction": masked_token_mean(
                replay_batch.active_mask.float(), replay_token_mask
            ),
            "branch_active_count": output.replay_loss_mask.sum(),
            "global_clean_token_count": counts[0],
            "global_replay_token_count": counts[1],
            "global_potential_token_count": counts[2],
            "replay_selection_probability": clean_loss.detach().new_tensor(
                replay_selection_probability
            ),
            "replay_active_token_fraction": clean_loss.detach().new_tensor(
                replay_active_token_fraction
            ),
            "branch_target_non_gt_mass": masked_token_mean(
                replay_batch.effective_non_gt_mass, output.replay_loss_mask
            ),
            "branch_target_gt_mass": 1.0
            - masked_token_mean(
                replay_batch.effective_non_gt_mass, output.replay_loss_mask
            ),
            "target_non_gt_dose": masked_token_mean(
                replay_batch.effective_non_gt_mass, output.replay_loss_mask
            ),
            "clean_top1": clean_top1,
            "branch_top1": branch_top1,
            "student_forward_count": clean_loss.detach().new_ones(()),
            "replay_mode": clean_loss.detach().new_ones(()),
        }
    else:
        gated_two_pass = replace(
            two_pass, branch_loss_weight=train.branch_loss_weight * auxiliary_gate
        )
        output = two_pass_forward(
            model,
            tokens,
            labels,
            config=gated_two_pass,
            codebook_kernel=codebook_kernel,
            codebook_affinity=codebook_affinity,
            replacement_ratio=replacement_ratio,
            model_kwargs={"class_drop_mask": class_drop_mask},
        )
        clean_loss = output.clean_loss
        raw_loss = output.loss
        batch_metrics = dict(output.metrics)
        batch_metrics["branch_enabled"] = output.clean_loss.detach().new_ones(())
    batch_metrics.update(
        {
            "auxiliary_loss_gate": clean_loss.detach().new_tensor(auxiliary_gate),
            "effective_branch_loss_weight": clean_loss.detach().new_tensor(
                train.branch_loss_weight * auxiliary_gate
            ),
        }
    )
    return (raw_loss, batch_metrics)


def _append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _broadcast_rank_zero_object(value: object) -> object:
    payload = [value]
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _accumulate_metrics(
    total: Dict[str, Tensor], values: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    for name, value in values.items():
        scalar = value.detach().float().mean()
        total[name] = total.get(name, torch.zeros_like(scalar)) + scalar
    return total


def _reduce_base_metrics_with_rank_zero_gradients(
    base_metrics: Dict[str, Tensor],
    gradient_metrics: Dict[str, float],
    *,
    is_main: bool,
) -> Dict[str, float]:
    """Reduce fixed training metrics and append dynamic grad fields on rank zero.

    Gradient diagnostics deliberately stay out of distributed collectives.  Their
    key set can differ across ranks when only one rank observes a non-finite
    parameter; reducing such dictionaries key-by-key would mismatch collective
    order and deadlock.  DDP has already synchronized ordinary gradients before
    this helper is called, so rank zero is sufficient for these diagnostics.
    """
    reduced = reduce_metrics(base_metrics)
    if is_main:
        reduced.update(gradient_metrics)
    return reduced


def _validate_bounded_replay_provenance(
    provenance: Mapping[str, Any],
    *,
    bank_record_count: int,
    dataset_record_count: int,
    train: TrainingConfig,
    world_size: int,
    start_state: Mapping[str, int],
) -> None:
    """Reject a bounded replay bank outside its exact sampler window."""
    planned_steps_raw = provenance.get("planned_training_steps")
    if planned_steps_raw is None:
        return
    planned_steps = int(planned_steps_raw)
    if planned_steps <= 0:
        raise RuntimeError("bounded replay bank has invalid planned training steps")
    if train.max_steps > planned_steps:
        raise RuntimeError("bounded replay bank does not cover max_steps")
    if int(start_state.get("epoch", -1)) != 0:
        raise RuntimeError("bounded replay bank only covers epoch zero")
    if (
        provenance.get("planned_sampler")
        != "torch_distributed_sampler_epoch0_shuffle_drop_last_v1"
    ):
        raise RuntimeError("bounded replay bank sampler provenance mismatch")
    exact_fields = {
        "planned_world_size": world_size,
        "planned_global_batch_size": train.global_batch_size,
        "planned_per_device_batch_size": train.per_device_batch_size,
        "planned_training_seed": train.seed,
        "planned_record_count": planned_steps * train.global_batch_size,
    }
    for key, expected in exact_fields.items():
        if int(provenance.get(key, -1)) != int(expected):
            raise RuntimeError(f"bounded replay bank {key} mismatch")
    planned_records = int(provenance["planned_record_count"])
    if not 0 < bank_record_count <= planned_records <= dataset_record_count:
        raise RuntimeError("bounded replay bank record counts are invalid")
    probability = float(provenance.get("selection_probability", -1.0))
    expected_population = round(dataset_record_count * probability)
    if (
        int(provenance.get("selection_population_record_count", -1))
        != expected_population
    ):
        raise RuntimeError("bounded replay bank selection population mismatch")


def _validate_replay_window_provenance(
    provenance: Mapping[str, Any],
    *,
    bank_record_count: int,
    dataset_record_count: int,
    manifest_sha256: str,
    tokenizer_sha256: str,
    train: TrainingConfig,
    class_dropout: float,
    expected_snapshot: int,
    replacement_schedule: Any,
    expected_checkpoint: Path,
) -> None:
    """Bind an immutable replay bank to one exact production window."""
    expected_midpoint = min(
        expected_snapshot + train.replay_refresh_steps // 2, train.replay_end_step - 1
    )
    expected_ratio = float(replacement_schedule(expected_midpoint, train.max_steps))
    if train.replay_stream_mode == "all_records":
        derived_probability = 1.0
    else:
        expected_weight = train.branch_loss_weight * (
            expected_ratio / train.replacement_max_ratio
        )
        derived_probability = expected_weight / (
            train.clean_loss_weight + expected_weight
        )
    expected_population = max(1, round(dataset_record_count * derived_probability))
    expected_probability = expected_population / dataset_record_count
    expected_alpha_use = train.target_mode == "support_gated_soft"
    exact = {
        "snapshot_step": expected_snapshot,
        "target_mode": train.target_mode,
        "configured_alpha_used": expected_alpha_use,
        "top_k": train.replay_top_k,
        "target_encoding": train.replay_target_encoding,
        "replay_stream_mode": train.replay_stream_mode,
        "history_draw_count_per_record": 1,
        "history_sampling_policy": "single_altered_sequence_per_record_v1",
        "interval_midpoint_step": expected_midpoint,
        "selection_population_record_count": expected_population,
        "dataset_record_count": dataset_record_count,
        "manifest_sha256": manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "class_drop_policy": "splitmix64_record_index_v1",
        "class_drop_probability": class_dropout,
        "class_drop_seed": train.seed + expected_snapshot * 1000003 + 814067,
    }
    for key, expected in exact.items():
        if provenance.get(key) != expected:
            raise RuntimeError(
                f"replay bank {key} mismatch: expected {expected!r}, got {provenance.get(key)!r}"
            )
    numeric = {
        "replacement_ratio": expected_ratio,
        "derived_selection_probability": derived_probability,
        "selection_probability": expected_probability,
    }
    for key, expected in numeric.items():
        actual = float(provenance.get(key, float("nan")))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"replay bank {key} mismatch: expected {expected}, got {actual}"
            )
    if provenance.get("planned_training_steps") is None:
        if bank_record_count != expected_population:
            raise RuntimeError("production replay bank record count mismatch")
        if int(provenance.get("teacher_checkpoint_step", -1)) != expected_snapshot:
            raise RuntimeError("production replay teacher checkpoint step mismatch")
        actual_checkpoint = Path(str(provenance.get("checkpoint", ""))).resolve()
        if actual_checkpoint != expected_checkpoint.resolve():
            raise RuntimeError("production replay checkpoint path mismatch")
    if train.replay_target_encoding == "categorical_mc":
        if (
            provenance.get("categorical_seed_formula")
            != "training_seed + snapshot_step*1000003 + rank*97409 + 260826"
        ):
            raise RuntimeError("categorical replay seed formula mismatch")
        sampled_ratio = float(provenance.get("sampled_non_gt_mass_ratio", float("nan")))
        if not math.isfinite(sampled_ratio) or not 0.95 <= sampled_ratio <= 1.05:
            raise RuntimeError("categorical replay sampled mass ratio mismatch")
    if train.replay_target_encoding == "conditional_non_gt_mc":
        if (
            provenance.get("categorical_seed_formula")
            != "training_seed + snapshot_step*1000003 + rank*97409 + 260826"
        ):
            raise RuntimeError("conditional replay seed formula mismatch")
        if provenance.get("target_weight_storage") != "implicit_equal_uint8":
            raise RuntimeError("conditional replay weight storage mismatch")


def _validate_replay_selection_probability(
    selection_probability: float, replay_stream_mode: str
) -> None:
    if replay_stream_mode == "all_records":
        if selection_probability != 1.0:
            raise RuntimeError("all-record replay requires selection_probability=1")
        return
    if not 0.0 < selection_probability < 1.0:
        raise RuntimeError("replay bank selection probability is invalid")


class _OffsetSampler(Sampler[int]):
    """Skip already-consumed indices without loading and discarding samples."""

    def __init__(self, sampler: Sampler[int]) -> None:
        self.sampler = sampler
        self.offset = 0

    def set_start_batch(self, batch_index: int, batch_size: int) -> None:
        if batch_index < 0 or batch_size <= 0:
            raise ValueError("resume batch index and batch size must be valid")
        offset = batch_index * batch_size
        if offset > len(self.sampler):
            raise RuntimeError("checkpoint batch_in_epoch exceeds DataLoader")
        self.offset = offset

    def __iter__(self) -> Iterator[int]:
        return itertools.islice(iter(self.sampler), self.offset, None)

    def __len__(self) -> int:
        return max(len(self.sampler) - self.offset, 0)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    train = config.training
    monitoring = config.monitoring
    if args.max_steps is not None:
        train.max_steps = args.max_steps
    if args.train_manifest is not None:
        config.data.train_manifest = str(args.train_manifest)
    if args.global_batch_size is not None:
        train.global_batch_size = args.global_batch_size
    if args.per_device_batch_size is not None:
        train.per_device_batch_size = args.per_device_batch_size
    if args.num_workers is not None:
        train.num_workers = args.num_workers
    if args.output_dir is not None:
        config.run.output_dir = str(args.output_dir)
    if args.run_name is not None:
        config.run.name = args.run_name
    if args.smoke_steps is not None:
        train.max_steps = args.smoke_steps
        train.warmup_steps = min(train.warmup_steps, train.max_steps)
        train.log_every = 1
        train.save_every = max(args.smoke_steps, 1)
    train.validate()
    if args.stop_at_step is not None:
        if not 0 < args.stop_at_step <= train.max_steps:
            raise ValueError("--stop-at-step must be in (0, max_steps]")
        if (
            train.history_source == "lagged_replay"
            and args.stop_at_step < train.replay_end_step
            and args.stop_at_step % train.replay_refresh_steps
        ):
            raise ValueError("--stop-at-step must align to replay_refresh_steps")
    resume_path = args.resume or (
        Path(config.run.resume) if config.run.resume else None
    )
    initialization_path = args.init_model_checkpoint
    output_dir = Path(config.run.output_dir) / config.run.name
    migration_flags = (
        args.allow_training_geometry_migration,
        args.allow_replacement_schedule_migration,
        False,
    )
    if sum((bool(enabled) for enabled in migration_flags)) > 1:
        raise ValueError("training migration modes are mutually exclusive")
    context = initialize_distributed()
    seed_everything(train.seed, context.rank, config.run.deterministic)
    if context.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    directory_error = ""
    if context.is_main:
        if resume_path is not None and initialization_path is not None:
            directory_error = (
                "--resume and --init-model-checkpoint are mutually exclusive"
            )
        elif args.validate_resume_only and resume_path is None:
            directory_error = "--validate-resume-only requires --resume"
        elif args.allow_training_geometry_migration and resume_path is None:
            directory_error = "--allow-training-geometry-migration requires --resume"
        elif args.allow_replacement_schedule_migration and resume_path is None:
            directory_error = "--allow-replacement-schedule-migration requires --resume"
        elif resume_path is not None and (not resume_path.is_file()):
            directory_error = f"resume checkpoint does not exist: {resume_path}"
        elif initialization_path is not None and (not initialization_path.is_file()):
            directory_error = (
                f"model initialization checkpoint does not exist: {initialization_path}"
            )
        elif (
            args.allow_training_geometry_migration
            or args.allow_replacement_schedule_migration
        ):
            source_run_dir = resume_path.resolve().parent.parent
            if output_dir.resolve() == source_run_dir:
                directory_error = f"training migration must use a new run directory; source and target both resolve to {source_run_dir}"
            elif output_dir.exists() and any(output_dir.iterdir()):
                directory_error = (
                    f"training migration output directory is not empty: {output_dir}"
                )
        elif resume_path is None and output_dir.exists() and any(output_dir.iterdir()):
            directory_error = f"training output directory is not empty: {output_dir}; use a new run name or pass --resume"
    if torch.distributed.is_initialized():
        error_payload = [directory_error]
        torch.distributed.broadcast_object_list(error_payload, src=0)
        directory_error = str(error_payload[0])
    if directory_error:
        raise FileExistsError(directory_error)
    report = validate_manifest(config.data.train_manifest, max_records=32)
    if not report.ok:
        raise RuntimeError("invalid pretokenized manifest: " + "; ".join(report.errors))
    dataset = PretokenizedImageNetDataset(config.data.train_manifest, split="train")
    tokenizer_sha256 = verify_manifest_tokenizer_checkpoint(
        dataset.manifest, config.data.tokenizer_checkpoint
    )
    if dataset.codebook_size != config.model.vocab_size:
        raise ValueError("dataset codebook_size does not match model vocab_size")
    if dataset.num_tokens != config.model.sequence_length:
        raise ValueError("dataset num_tokens does not match model sequence_length")
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=train.seed,
        drop_last=True,
    )
    offset_sampler = _OffsetSampler(sampler)
    loader = DataLoader(
        dataset,
        batch_size=train.per_device_batch_size,
        sampler=offset_sampler,
        num_workers=train.num_workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=train.num_workers > 0,
        drop_last=True,
    )
    local_global_batch = train.per_device_batch_size * context.world_size
    if train.global_batch_size % local_global_batch:
        raise ValueError(
            "global_batch_size must be divisible by per_device_batch_size * world_size"
        )
    accumulation_steps = train.global_batch_size // local_global_batch
    complete_micro_batches = len(loader) // accumulation_steps * accumulation_steps
    if complete_micro_batches == 0:
        raise ValueError("dataset is too small for one effective global batch")
    optimizer_steps_per_epoch = complete_micro_batches // accumulation_steps
    if train.epochs * optimizer_steps_per_epoch < train.max_steps:
        raise ValueError(
            f"epochs do not cover max_steps: {train.epochs} * {optimizer_steps_per_epoch} < {train.max_steps}"
        )
    training_geometry = {
        "world_size": context.world_size,
        "per_device_batch_size": train.per_device_batch_size,
        "accumulation_steps": accumulation_steps,
    }
    codebook = load_maskgit_codebook(config.data.tokenizer_checkpoint)
    expected_codebook = (config.model.vocab_size, config.model.codebook_dim)
    if tuple(codebook.shape) != expected_codebook:
        raise ValueError(
            f"tokenizer codebook must be {expected_codebook}, got {tuple(codebook.shape)}"
        )
    model = build_nti(
        config.model,
        codebook=codebook if config.icb.enabled else None,
        icb_config=config.icb,
    ).to(context.device)
    if config.run.compile_model:
        model = torch.compile(model)
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    kernel = None
    affinity = None
    if train.target_mode in {"tv_history_conditioned"}:
        kernel = build_codebook_cosine_kernel(
            codebook, temperature=train.codebook_kernel_temperature
        ).to(context.device)
    if train.target_mode == "recoverability_static_g":
        affinity = build_codebook_cosine_affinity(
            codebook, temperature=train.codebook_kernel_temperature
        ).to(context.device)
    interface = config.token_interface
    if train.replacement_schedule == "constant":
        replacement_schedule = LinearReplacementSchedule(
            start_ratio=train.replacement_max_ratio,
            max_ratio=train.replacement_max_ratio,
            peak_fraction=train.replacement_warmup_fraction,
        )
    elif train.replacement_schedule == "linear_hold":
        replacement_schedule = LinearReplacementSchedule(
            start_ratio=0.0,
            max_ratio=train.replacement_max_ratio,
            peak_fraction=train.replacement_warmup_fraction,
        )
    elif train.replacement_schedule == "ramp_cosine_zero":
        replacement_schedule = RampCosineReplacementSchedule(
            start_ratio=0.0,
            max_ratio=train.replacement_max_ratio,
            peak_fraction=train.replacement_warmup_fraction,
        )
    elif train.replacement_schedule == "bell_cosine":
        replacement_schedule = BellCosineReplacementSchedule(
            max_ratio=train.replacement_max_ratio,
            start_fraction=train.replacement_start_fraction,
            peak_fraction=train.replacement_warmup_fraction,
            decay_start_fraction=train.replacement_decay_start_fraction,
            end_fraction=train.replacement_end_fraction,
        )
    else:
        raise AssertionError(
            f"unsupported replacement schedule {train.replacement_schedule}"
        )
    two_pass = TwoPassConfig(
        target_mode=train.target_mode,
        soft_alpha=train.alpha,
        clean_target_temperature=1.0,
        sampling_temperature=train.history_temperature,
        sampling_top_k=train.history_top_k,
        sampling_top_p=train.history_top_p,
        replacement_ratio_sampling=train.replacement_ratio_sampling,
        branch_image_fraction=train.branch_image_fraction,
        branch_loss_scope=train.branch_loss_scope,
        clean_loss_weight=train.clean_loss_weight,
        branch_loss_weight=train.branch_loss_weight,
        schedule=replacement_schedule,
    )
    optimizer = build_optimizer(model, train)
    scheduler = build_lr_scheduler(optimizer, train)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=context.device.type == "cuda" and train.precision == "fp16"
    )
    state = {"step": 0, "epoch": 0, "batch_in_epoch": 0}
    initialization_source_state = None
    continuation_migration = None
    target_continuation_resume = None
    if resume_path is not None:
        allowed_config_migrations = None
        if args.allow_replacement_schedule_migration:
            allowed_config_migrations = {
                "training.replacement_schedule": ("linear_hold", "ramp_cosine_zero")
            }
        required_config_sections = _RESUME_CONFIG_SECTIONS
        ignored_config_keys = _RESUME_IGNORED_CONFIG_KEYS
        state = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=context.rank,
            expected_training_geometry=training_geometry,
            expected_config=config.to_dict(),
            required_config_sections=required_config_sections,
            ignored_config_keys=ignored_config_keys,
            allowed_config_migrations=allowed_config_migrations,
            require_config_migrations=args.allow_replacement_schedule_migration,
            allow_training_geometry_migration=args.allow_training_geometry_migration,
            migration_dataset_records=len(dataset),
            geometry_migration_rng_seed=train.seed,
            include_scheduler_state=False,
            require_complete_rng_states=False,
        )
        if args.allow_replacement_schedule_migration:
            _validate_replacement_schedule_migration(train, state)
    elif initialization_path is not None:
        initialization_source_state = load_checkpoint(
            initialization_path,
            model=model,
            expected_config=config.to_dict(),
            required_config_sections=("model", "icb"),
            restore_rng=False,
        )
    if scheduler is None:
        raise AssertionError("training scheduler was not initialized")
    resume_data_cursor = (
        _validate_resume_data_cursor(
            state,
            epochs=train.epochs,
            complete_micro_batches=complete_micro_batches,
            accumulation_steps=accumulation_steps,
        )
        if resume_path is not None
        else None
    )
    barrier()
    start_state = {key: int(state[key]) for key in ("step", "epoch", "batch_in_epoch")}
    replay_bank = None
    replay_provenance: Mapping[str, Any] | None = None
    replay_resolved_root: Path | None = None
    if (
        train.history_source == "lagged_replay"
        and train.replay_start_step <= start_state["step"] < train.replay_end_step
    ):
        replay_resolved_root = resolve_replay_bank_root(train.replay_bank_root)
        replay_bank = open_replay_bank(replay_resolved_root)
        replay_provenance = replay_bank.manifest.get("provenance", {})
        expected_snapshot = (
            start_state["step"]
            // train.replay_refresh_steps
            * train.replay_refresh_steps
        )
        if int(replay_provenance.get("snapshot_step", -1)) != expected_snapshot:
            raise RuntimeError(
                f"replay snapshot does not match the current refresh interval: expected {expected_snapshot}, got {replay_provenance.get('snapshot_step')}"
            )
        if replay_provenance.get("target_mode") != train.target_mode:
            raise RuntimeError("replay bank target_mode mismatch")
        if int(replay_provenance.get("top_k", -1)) != train.replay_top_k:
            raise RuntimeError("replay bank top_k mismatch")
        if (
            replay_provenance.get("target_encoding", "sparse_topk")
            != train.replay_target_encoding
        ):
            raise RuntimeError("replay bank target_encoding mismatch")
        if replay_provenance.get("tokenizer_sha256") != tokenizer_sha256:
            raise RuntimeError("replay bank tokenizer mismatch")
        if int(replay_provenance.get("dataset_record_count", -1)) != len(dataset):
            raise RuntimeError("replay bank dataset cardinality mismatch")
        selection_probability = float(
            replay_provenance.get("selection_probability", -1.0)
        )
        _validate_replay_selection_probability(
            selection_probability, train.replay_stream_mode
        )
        active_token_fraction = float(
            replay_provenance.get("active_token_fraction", -1.0)
        )
        if not 0.0 < active_token_fraction <= 1.0:
            raise RuntimeError("replay bank active token fraction is invalid")
        _validate_replay_window_provenance(
            replay_provenance,
            bank_record_count=int(replay_bank.record_count),
            dataset_record_count=len(dataset),
            manifest_sha256=sha256_file(config.data.train_manifest),
            tokenizer_sha256=tokenizer_sha256,
            train=train,
            class_dropout=config.model.class_dropout,
            expected_snapshot=expected_snapshot,
            replacement_schedule=replacement_schedule,
            expected_checkpoint=resume_path
            if args.allow_training_geometry_migration
            and resume_path is not None
            and (expected_snapshot == start_state["step"])
            else Path(config.run.output_dir)
            / config.run.name
            / "checkpoints"
            / f"step_{expected_snapshot:07d}.pt",
        )
        _validate_bounded_replay_provenance(
            replay_provenance,
            bank_record_count=int(replay_bank.record_count),
            dataset_record_count=len(dataset),
            train=train,
            world_size=context.world_size,
            start_state=start_state,
        )
    initialization_checkpoint_sha256 = (
        sha256_file(initialization_path)
        if context.is_main and initialization_path is not None
        else None
    )
    geometry_migration = state.get("geometry_migration")
    resume_checkpoint_sha256 = (
        sha256_file(resume_path)
        if context.is_main and resume_path is not None
        else None
    )
    if args.validate_resume_only:
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "resume_validation_succeeded",
                        "checkpoint": str(resume_path),
                        "state": start_state,
                        "resume_data_cursor": resume_data_cursor,
                        "training_geometry": training_geometry,
                        "geometry_migration": geometry_migration,
                        "resume_checkpoint_sha256": resume_checkpoint_sha256,
                        "optimizer_lrs": [
                            float(group["lr"]) for group in optimizer.param_groups
                        ],
                        "scheduler_last_epoch": scheduler.last_epoch,
                        "scheduler_base_lrs": list(scheduler.base_lrs),
                        "scheduler_last_lrs": list(scheduler.get_last_lr()),
                        "continuation_migration": continuation_migration,
                        "target_continuation_resume": target_continuation_resume,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        shutdown_distributed()
        return
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path = output_dir / "resolved_config.json"
        if not resolved_config_path.exists():
            resolved_config_path.write_text(
                json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if continuation_migration is not None:
            migration_path = output_dir / "continuation_migration.json"
            migration_payload = {
                **continuation_migration,
                "source_checkpoint": str(resume_path),
                "source_checkpoint_sha256": resume_checkpoint_sha256,
                "target_run_directory": str(output_dir),
            }
            migration_path.write_text(
                json.dumps(migration_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    barrier()
    preview_enabled = monitoring.preview_every > 0
    preview_tokenizer = None
    preview_manifest_path = output_dir / "preview_manifest.jsonl"
    preview_sampling = SamplingConfig(
        num_tokens=config.model.sequence_length,
        temperature=monitoring.preview_temperature,
        top_k=monitoring.preview_top_k or None,
        top_p=monitoring.preview_top_p if monitoring.preview_top_p < 1.0 else None,
        do_sample=True,
        guidance_scale=monitoring.preview_guidance_scale,
        guidance_schedule="cosine",
        guidance_scale_pow=monitoring.preview_guidance_scale_pow,
        unconditional_class_id=config.model.num_classes,
    )
    preview_sampling.validate()
    if context.is_main:
        trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
        kernel_metrics = {}
        if kernel is not None:
            with torch.no_grad():
                kernel_entropy = -(kernel * kernel.clamp_min(1e-12).log()).sum(dim=-1)
                kernel_metrics = {
                    "kernel_diagonal_mass": float(kernel.diagonal().mean()),
                    "kernel_entropy": float(kernel_entropy.mean()),
                    "kernel_effective_support": float(kernel_entropy.exp().mean()),
                }
        print(
            json.dumps(
                {
                    "event": "start",
                    "arm": train.arm,
                    "target_mode": train.target_mode,
                    "history_source": train.history_source,
                    "replay_bank": str(replay_resolved_root)
                    if replay_resolved_root is not None
                    else None,
                    "replay_provenance": replay_provenance,
                    "icb": config.icb.enabled,
                    "token_interface": interface.mode,
                    "branch_image_fraction": train.branch_image_fraction,
                    "replacement_max_ratio": train.replacement_max_ratio,
                    "replacement_ratio_sampling": train.replacement_ratio_sampling,
                    "replacement_start_fraction": train.replacement_start_fraction,
                    "replacement_warmup_fraction": train.replacement_warmup_fraction,
                    "replacement_decay_start_fraction": train.replacement_decay_start_fraction,
                    "replacement_end_fraction": train.replacement_end_fraction,
                    "replacement_schedule": train.replacement_schedule,
                    "auxiliary_loss_gate": train.auxiliary_loss_gate,
                    "branch_active_only_when_replacement_positive": train.branch_active_only_when_replacement_positive,
                    "replacement_schedule_migration": args.allow_replacement_schedule_migration,
                    "continuation_migration": continuation_migration,
                    "world_size": context.world_size,
                    "accumulation_steps": accumulation_steps,
                    "effective_global_batch": local_global_batch * accumulation_steps,
                    "preview_every": monitoring.preview_every,
                    "preview_seed": monitoring.preview_seed,
                    "preview_class_ids": list(monitoring.preview_class_ids),
                    "preview_decode_batch_size": monitoring.preview_decode_batch_size,
                    "grad_norm_every": monitoring.grad_norm_every,
                    "trainable_parameters": trainable,
                    "dataset_records": len(dataset),
                    "tokenizer_sha256": tokenizer_sha256,
                    "resume_checkpoint": str(resume_path)
                    if resume_path is not None
                    else None,
                    "resume_checkpoint_sha256": resume_checkpoint_sha256,
                    "model_initialization_checkpoint": str(initialization_path)
                    if initialization_path is not None
                    else None,
                    "model_initialization_source_state": initialization_source_state,
                    "model_initialization_checkpoint_sha256": initialization_checkpoint_sha256,
                    "optimizer_initialization": "fresh"
                    if initialization_path is not None
                    else "checkpoint"
                    if resume_path is not None
                    else "fresh",
                    "start_state": start_state,
                    "training_geometry": training_geometry,
                    "geometry_migration": geometry_migration,
                    **kernel_metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if checkpoint_reached_max_steps(state, train.max_steps):
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "already_complete",
                        "checkpoint_step": state["step"],
                        "max_steps": train.max_steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        shutdown_distributed()
        return
    optimizer.zero_grad(set_to_none=True)
    step = state["step"]
    wall_start_step = step
    start_epoch = state["epoch"]
    wall_start = time.time()
    metrics_path = output_dir / "metrics.jsonl"
    metric_sums: Dict[str, Tensor] = {}
    metric_micro_batches = 0
    for epoch in range(start_epoch, train.epochs):
        sampler.set_epoch(epoch)
        batch_start = state["batch_in_epoch"] if epoch == start_epoch else 0
        if batch_start % accumulation_steps:
            raise RuntimeError("checkpoint must resume at an optimizer-step boundary")
        offset_sampler.set_start_batch(batch_start, train.per_device_batch_size)
        iterator = iter(loader)
        for batch_index in range(batch_start, complete_micro_batches):
            batch = next(iterator)
            tokens = batch["image_tokens"].to(context.device, non_blocking=True)
            labels = batch["class_labels"].to(context.device, non_blocking=True)
            record_indices = batch["record_index"].to(context.device, non_blocking=True)
            class_drop_mask = (
                torch.rand(labels.shape, device=context.device)
                < config.model.class_dropout
            ).detach()
            group_position = batch_index % accumulation_steps
            synchronize = group_position == accumulation_steps - 1
            if hasattr(model, "no_sync") and (not synchronize):
                synchronization_context = model.no_sync()
            else:
                synchronization_context = contextlib.nullcontext()
            with (
                synchronization_context,
                _autocast_context(context.device, train.precision),
            ):
                current_replacement_ratio = replacement_schedule(step, train.max_steps)
                if train.history_source == "lagged_replay" and (
                    not train.replay_start_step <= step < train.replay_end_step
                ):
                    current_replacement_ratio = 0.0
                raw_loss, batch_metrics = _compute_training_objective(
                    model,
                    tokens,
                    labels,
                    class_drop_mask,
                    train=train,
                    interface=interface,
                    two_pass=two_pass,
                    codebook_kernel=kernel,
                    codebook_affinity=affinity,
                    replacement_ratio=current_replacement_ratio,
                    record_indices=record_indices,
                    replay_bank=replay_bank,
                )
                loss = raw_loss / accumulation_steps
            scaler.scale(loss).backward()
            _accumulate_metrics(metric_sums, batch_metrics)
            metric_micro_batches += 1
            if not synchronize:
                continue
            scaler.unscale_(optimizer)
            completed_step = step + 1
            grad_monitor_due = (
                monitoring.grad_norm_every > 0
                and completed_step % monitoring.grad_norm_every == 0
            )
            gradient_metrics: Dict[str, float] = {}
            if grad_monitor_due and context.is_main:
                gradient_metrics.update(
                    collect_preclip_gradient_metrics(
                        model, max_grad_norm=train.max_grad_norm
                    )
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train.max_grad_norm
            )
            if grad_monitor_due and context.is_main:
                gradient_metrics.update(collect_rear_style_gradient_metrics(model))
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            skipped_update = scaler.get_scale() < previous_scale
            if skipped_update:
                continue
            scheduler.step()
            step += 1
            next_batch = batch_index + 1
            if (
                step % train.log_every == 0
                or grad_monitor_due
                or step == 1
                or (step == train.max_steps)
            ):
                averaged = {
                    name: value / metric_micro_batches
                    for name, value in metric_sums.items()
                }
                averaged["grad_norm"] = torch.as_tensor(
                    grad_norm, device=context.device
                )
                reduced = _reduce_base_metrics_with_rank_zero_gradients(
                    averaged, gradient_metrics, is_main=context.is_main
                )
                payload: Dict[str, object] = {
                    "step": step,
                    "epoch": epoch,
                    "batch_in_epoch": next_batch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.time() - wall_start,
                    **reduced,
                }
                elapsed = max(float(payload["elapsed_seconds"]), 1e-09)
                completed_steps = step - wall_start_step
                payload["optimizer_steps_per_second"] = completed_steps / elapsed
                payload["images_per_second"] = (
                    completed_steps * train.global_batch_size / elapsed
                )
                payload["image_tokens_per_second"] = (
                    completed_steps
                    * train.global_batch_size
                    * config.model.sequence_length
                    / elapsed
                )
                if context.device.type == "cuda":
                    payload["peak_memory_allocated_gib_rank0"] = (
                        torch.cuda.max_memory_allocated(context.device) / 1024**3
                    )
                    payload["peak_memory_reserved_gib_rank0"] = (
                        torch.cuda.max_memory_reserved(context.device) / 1024**3
                    )
                if context.is_main:
                    print(json.dumps(payload, sort_keys=True), flush=True)
                    _append_jsonl(metrics_path, payload)
                metric_sums = {}
                metric_micro_batches = 0
            stop_due = args.stop_at_step is not None and step == args.stop_at_step
            should_save = (
                step % train.save_every == 0 or step == train.max_steps or stop_due
            )
            if should_save:
                rng_states = gather_rng_states(capture_rng_state(), context)
                if context.is_main:
                    checkpoint_path = output_dir / "checkpoints" / f"step_{step:07d}.pt"
                    save_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        epoch=epoch,
                        batch_in_epoch=next_batch,
                        config=config.to_dict(),
                        rng_states=rng_states,
                        training_geometry=training_geometry,
                    )
                    latest = output_dir / "checkpoints" / "latest.txt"
                    latest.write_text(str(checkpoint_path) + "\n", encoding="utf-8")
                    removed = prune_training_checkpoints(
                        checkpoint_path.parent,
                        milestone_every=train.checkpoint_milestone_every,
                        keep_last=train.checkpoint_keep_last,
                    )
                    if removed:
                        print(
                            json.dumps(
                                {
                                    "event": "checkpoint_retention",
                                    "step": step,
                                    "removed": [path.name for path in removed],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                barrier()
            preview_due = preview_enabled and step % monitoring.preview_every == 0
            if preview_due:
                barrier()
                preview_status: Dict[str, object] = {"ok": True, "step": step}
                if context.is_main:
                    try:
                        existing_steps = {
                            record.step
                            for record in iter_preview_manifest(preview_manifest_path)
                        }
                        if step in existing_steps:
                            preview_status["event"] = "training_preview_exists"
                        else:
                            with preserve_all_rng_states():
                                if preview_tokenizer is None:
                                    preview_tokenizer = load_frozen_maskgit_vqgan(
                                        config.data.tokenizer_checkpoint,
                                        rear_repo=config.data.rear_source_root,
                                        strict=True,
                                    )
                                preview_tokenizer.to(context.device)
                                with _autocast_context(context.device, train.precision):
                                    record = generate_training_preview(
                                        model,
                                        preview_tokenizer,
                                        step=step,
                                        class_ids=monitoring.preview_class_ids,
                                        seed=monitoring.preview_seed,
                                        sampling=preview_sampling,
                                        run_dir=output_dir,
                                        rows=monitoring.preview_rows,
                                        decode_batch_size=monitoring.preview_decode_batch_size,
                                        manifest_path=preview_manifest_path,
                                        device=context.device,
                                    )
                            preview_status.update(
                                {
                                    "event": "training_preview_complete",
                                    "path": record.png_path,
                                    "png_sha256": record.png_sha256,
                                }
                            )
                    except Exception as error:
                        preview_status = {
                            "ok": False,
                            "event": "training_preview_disabled",
                            "step": step,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    finally:
                        if preview_tokenizer is not None:
                            try:
                                preview_tokenizer.to("cpu")
                            except Exception as cleanup_error:
                                preview_status = {
                                    "ok": False,
                                    "event": "training_preview_disabled",
                                    "step": step,
                                    "error": f"preview tokenizer cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}",
                                }
                        gc.collect()
                        if context.device.type == "cuda":
                            torch.cuda.empty_cache()
                    print(json.dumps(preview_status, sort_keys=True), flush=True)
                preview_status = _broadcast_rank_zero_object(preview_status)
                preview_enabled = bool(preview_status.get("ok", False))
                barrier()
            if stop_due:
                if context.is_main:
                    print(
                        json.dumps(
                            {"event": "replay_refresh_boundary", "step": step},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                shutdown_distributed()
                return
            if step >= train.max_steps:
                shutdown_distributed()
                return
        optimizer.zero_grad(set_to_none=True)
        state["batch_in_epoch"] = 0
    shutdown_distributed()
    raise RuntimeError("training exhausted epochs before reaching max_steps")


if __name__ == "__main__":
    main()
