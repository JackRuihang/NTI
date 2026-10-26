#!/usr/bin/env python3
"""Distributed, class-balanced ImageNet sampling for FID/IS evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nti.checkpoint import load_checkpoint  # noqa: E402
from nti.config import ExperimentConfig, load_experiment_config  # noqa: E402
from nti.distributed import (  # noqa: E402
    barrier,
    initialize_distributed,
    seed_everything,
    shutdown_distributed,
)
from nti.evaluation import (  # noqa: E402
    CUSTOM_PROTOCOL_NAME,
    REAR_PROTOCOL_NAME,
    create_adm_npz,
    file_record,
    rear_class_plan,
    validate_complete_plan,
    validate_rear_sampling_protocol,
)
from nti.generation import SamplingConfig, autoregressive_sample, decode_generation  # noqa: E402
from nti.models import build_nti  # noqa: E402
from nti.tokenizer import load_frozen_maskgit_vqgan, load_maskgit_codebook  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=22.0)
    parser.add_argument("--guidance-scale-pow", type=float, default=2.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--skip-npz", action="store_true")
    parser.add_argument(
        "--logical-world-size",
        type=int,
        default=None,
        help=(
            "sampling world size used for class planning and rank RNG; defaults "
            "to the physical torch.distributed world size"
        ),
    )
    parser.add_argument(
        "--logical-rank-offset",
        type=int,
        default=0,
        help="first logical rank assigned to physical rank zero in this wave",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="append a disjoint logical-rank wave to an unfinished output directory",
    )
    parser.add_argument(
        "--defer-finalize",
        action="store_true",
        help="write this wave's PNGs/rank metadata without merging samples or NPZ",
    )
    parser.add_argument(
        "--protocol",
        choices=("rear-formal", "custom"),
        default="custom",
        help="rear-formal enforces every released 50K sampling constant",
    )
    return parser.parse_args()


def resolve_logical_rank(
    *,
    physical_rank: int,
    physical_world_size: int,
    logical_world_size: int | None,
    logical_rank_offset: int,
) -> tuple[int, int]:
    resolved_world_size = (
        physical_world_size if logical_world_size is None else logical_world_size
    )
    if physical_world_size <= 0 or resolved_world_size <= 0:
        raise ValueError("physical and logical world sizes must be positive")
    if not 0 <= physical_rank < physical_world_size:
        raise ValueError("physical rank is outside the physical world")
    if logical_rank_offset < 0:
        raise ValueError("logical rank offset must be non-negative")
    if logical_rank_offset % physical_world_size:
        raise ValueError("logical rank offset must align to a physical-rank wave")
    if logical_rank_offset + physical_world_size > resolved_world_size:
        raise ValueError("logical-rank wave exceeds the logical world size")
    return logical_rank_offset + physical_rank, resolved_world_size


def validate_wave_output(
    output: Path,
    *,
    logical_ranks: range,
    plans: list[list[tuple[int, int]]],
    allow_existing: bool,
) -> None:
    if not output.exists():
        return
    if not allow_existing:
        raise FileExistsError(
            f"sampling output already exists: {output}; use a new directory"
        )
    if not output.is_dir():
        raise FileExistsError(f"sampling output is not a directory: {output}")
    finalized = [output / "samples.json", output.with_suffix(".npz")]
    if any(path.exists() for path in finalized):
        raise FileExistsError(f"sampling output is already finalized: {output}")
    for logical_rank, plan in zip(logical_ranks, plans):
        rank_metadata = output / f"samples.rank-{logical_rank:05d}.json"
        if rank_metadata.exists():
            raise FileExistsError(f"logical rank already exists: {rank_metadata}")
        conflict = next(
            (
                output / f"{sample_index:06d}.png"
                for sample_index, _ in plan
                if (output / f"{sample_index:06d}.png").exists()
            ),
            None,
        )
        if conflict is not None:
            raise FileExistsError(f"logical-rank output already exists: {conflict}")


def save_image(image: torch.Tensor, path: Path) -> None:
    array = (
        image.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(np.asarray(array)).save(path)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def source_provenance(
    args: argparse.Namespace, config: ExperimentConfig
) -> dict[str, object]:
    source_paths = [
        PROJECT_ROOT / "nti" / "models" / "backbone.py",
        PROJECT_ROOT / "nti" / "models" / "icb.py",
        PROJECT_ROOT / "nti" / "generation.py",
        PROJECT_ROOT / "nti" / "evaluation.py",
        PROJECT_ROOT / "nti" / "checkpoint.py",
        PROJECT_ROOT / "nti" / "tokenizer" / "maskgit_vqgan.py",
        Path(__file__).resolve(),
    ]
    return {
        "config": file_record(args.config),
        "checkpoint": file_record(args.checkpoint),
        "tokenizer_checkpoint": file_record(config.data.tokenizer_checkpoint),
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): file_record(path)
            for path in source_paths
        },
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.log_every_batches <= 0:
        raise ValueError("batch-size and log-every-batches must be positive")
    config = load_experiment_config(args.config)
    context = initialize_distributed()
    logical_rank, logical_world_size = resolve_logical_rank(
        physical_rank=context.rank,
        physical_world_size=context.world_size,
        logical_world_size=args.logical_world_size,
        logical_rank_offset=args.logical_rank_offset,
    )
    seed_everything(args.seed, logical_rank, deterministic=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    top_k = args.top_k if args.top_k != 0 else None
    top_p = args.top_p if args.top_p != 1.0 else None
    if args.protocol == "rear-formal":
        validate_rear_sampling_protocol(
            num_samples=args.num_samples,
            num_classes=config.model.num_classes,
            per_rank_batch_size=args.batch_size,
            world_size=logical_world_size,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p,
            guidance_scale=args.guidance_scale,
            guidance_scale_pow=args.guidance_scale_pow,
            seed=args.seed,
        )
    protocol_name = (
        REAR_PROTOCOL_NAME if args.protocol == "rear-formal" else CUSTOM_PROTOCOL_NAME
    )
    plan = rear_class_plan(
        args.num_samples,
        config.model.num_classes,
        per_rank_batch_size=args.batch_size,
        rank=logical_rank,
        world_size=logical_world_size,
    )
    output_error = ""
    if context.is_main:
        wave_ranks = range(
            args.logical_rank_offset,
            args.logical_rank_offset + context.world_size,
        )
        wave_plans = [
            rear_class_plan(
                args.num_samples,
                config.model.num_classes,
                per_rank_batch_size=args.batch_size,
                rank=rank,
                world_size=logical_world_size,
            )
            for rank in wave_ranks
        ]
        try:
            validate_wave_output(
                args.output,
                logical_ranks=wave_ranks,
                plans=wave_plans,
                allow_existing=args.allow_existing_output,
            )
        except (FileExistsError, ValueError) as error:
            output_error = str(error)
    if torch.distributed.is_initialized():
        error_payload = [output_error]
        torch.distributed.broadcast_object_list(error_payload, src=0)
        output_error = str(error_payload[0])
    if output_error:
        raise FileExistsError(output_error)
    barrier()

    provenance = source_provenance(args, config) if context.is_main else None
    barrier()

    codebook = load_maskgit_codebook(config.data.tokenizer_checkpoint)
    model = build_nti(
        config.model,
        codebook=codebook if config.icb.enabled else None,
        icb_config=config.icb,
    ).to(context.device)
    checkpoint_state = load_checkpoint(
        args.checkpoint,
        model=model,
        rank=context.rank,
        expected_config=config.to_dict(),
        required_config_sections=("model", "icb", "token_interface"),
        restore_rng=False,
    )
    model.eval()
    if context.is_main:
        assert provenance is not None
        runtime = provenance["runtime"]
        assert isinstance(runtime, dict)
        runtime.update(
            {
                "gpu_name": (
                    torch.cuda.get_device_name(context.device)
                    if context.device.type == "cuda"
                    else None
                ),
                "model_dtype": str(next(model.parameters()).dtype),
            }
        )
    tokenizer = load_frozen_maskgit_vqgan(
        config.data.tokenizer_checkpoint,
        rear_repo=config.data.rear_source_root,
        strict=True,
    ).to(context.device)
    if context.is_main and not args.output.exists():
        args.output.mkdir(parents=True, exist_ok=False)
    barrier()
    sampling = SamplingConfig(
        num_tokens=config.model.sequence_length,
        temperature=args.temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        guidance_scale=args.guidance_scale,
        guidance_schedule="cosine",
        guidance_scale_pow=args.guidance_scale_pow,
        unconditional_class_id=config.model.num_classes,
    )
    generator = torch.Generator(device=context.device).manual_seed(
        args.seed + logical_rank
    )
    records: list[dict[str, object]] = []
    started = time.monotonic()
    total_batches = (len(plan) + args.batch_size - 1) // args.batch_size
    for batch_number, start in enumerate(range(0, len(plan), args.batch_size), start=1):
        jobs = plan[start : start + args.batch_size]
        labels = torch.tensor(
            [class_id for _, class_id in jobs],
            dtype=torch.long,
            device=context.device,
        )
        generation = autoregressive_sample(
            model, labels, config=sampling, generator=generator
        )
        images = decode_generation(tokenizer, generation)
        for local_index, ((sample_index, class_id), image) in enumerate(
            zip(jobs, images)
        ):
            path = args.output / f"{sample_index:06d}.png"
            save_image(image, path)
            records.append(
                {
                    "sample_index": sample_index,
                    "class_id": class_id,
                    "path": path.name,
                    "mean_token_logprob": float(
                        generation.token_logprobs[local_index].float().mean().item()
                    ),
                }
            )
        if batch_number % args.log_every_batches == 0 or batch_number == total_batches:
            elapsed = max(time.monotonic() - started, 1.0e-9)
            print(
                json.dumps(
                    {
                        "event": "sample_progress",
                        "rank": logical_rank,
                        "batches_complete": batch_number,
                        "batches_total": total_batches,
                        "local_samples_complete": len(records),
                        "samples_per_second": len(records) / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    rank_metadata = args.output / f"samples.rank-{logical_rank:05d}.json"
    atomic_json(rank_metadata, records)
    barrier()
    if args.defer_finalize:
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "logical_rank_wave_complete",
                        "logical_rank_offset": args.logical_rank_offset,
                        "logical_world_size": logical_world_size,
                        "physical_world_size": context.world_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        shutdown_distributed()
        return
    if context.is_main:
        merged: list[dict[str, object]] = []
        for rank in range(logical_world_size):
            path = args.output / f"samples.rank-{rank:05d}.json"
            merged.extend(json.loads(path.read_text(encoding="utf-8")))
        merged.sort(key=lambda record: int(record["sample_index"]))
        validate_complete_plan(
            merged,
            num_samples=args.num_samples,
            num_classes=config.model.num_classes,
        )
        metadata = {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "num_samples": args.num_samples,
            "num_classes": config.model.num_classes,
            "world_size": logical_world_size,
            "execution": {
                "physical_world_size_per_wave": context.world_size,
                "finalizing_logical_rank_offset": args.logical_rank_offset,
            },
            "seed": args.seed,
            "sampling": sampling.__dict__,
            "checkpoint_state": checkpoint_state,
            "provenance": provenance,
            "protocol": {
                "name": protocol_name,
                "requested_mode": args.protocol,
                "per_rank_batch_size": args.batch_size,
                "class_stream": "0..999 repeated, contiguous slice per rank",
                "filename_order": "rank-interleaved within each global batch",
                "rank_seed": "seed + rank",
                "uint8_conversion": "clamp(image,0,1) * 255 then truncate",
            },
            "records": merged,
        }
        atomic_json(args.output / "samples.json", metadata)
        if not args.skip_npz:
            npz_path = create_adm_npz(
                args.output,
                num_samples=args.num_samples,
                image_size=256,
            )
            print(json.dumps({"event": "npz_complete", "path": str(npz_path)}))
    barrier()
    shutdown_distributed()


if __name__ == "__main__":
    main()
