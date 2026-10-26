#!/usr/bin/env python3
"""Pretokenize ImageNet with the frozen reAR MaskGIT VQGAN.

The preferred input is the Hugging Face parquet export because it carries the
official integer labels.  A flat JPEG directory is supported only together with
an explicit synset mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nti.data import (  # noqa: E402
    MANIFEST_FORMAT,
    JsonlShardWriter,
    build_raw_imagenet_dataset,
    sha256_file,
    shard_metadata_dict,
    write_manifest,
)
from nti.tokenizer import load_frozen_maskgit_vqgan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--parquet-glob", help="Quoted glob for Hugging Face parquet shards"
    )
    source.add_argument(
        "--image-root", type=Path, help="ImageFolder root or flat JPEG directory"
    )
    parser.add_argument(
        "--synset-mapping", type=Path, help="Required for flat JPEG input"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rear-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--records-per-shard", type=int, default=100_000)
    parser.add_argument(
        "--log-every-records",
        type=int,
        default=10_000,
        help="Emit one JSON progress record per rank at this interval; 0 disables it",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Single-process source-image limit for smoke tests",
    )
    parser.add_argument(
        "--views", choices=("original", "original,hflip"), default="original"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-unexpected-checkpoint-keys", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def distributed_context(device_arg: str) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if world_size > 1:
            device = torch.device("cuda", local_rank)
        else:
            requested = torch.device(device_arg)
            device = torch.device("cuda", requested.index or 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device(device_arg)
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        process_group_kwargs: dict[str, Any] = {
            "backend": backend,
            "init_method": "env://",
        }
        if device.type == "cuda":
            process_group_kwargs["device_id"] = device
        torch.distributed.init_process_group(**process_group_kwargs)
    return rank, world_size, local_rank, device


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    write_manifest(path, payload)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.records_per_shard <= 0:
        raise ValueError(
            "batch-size and records-per-shard must be positive; num-workers must be nonnegative"
        )
    if args.log_every_records < 0:
        raise ValueError("log-every-records must be nonnegative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    rank, world_size, _, device = distributed_context(args.device)
    if args.max_samples is not None and world_size != 1:
        raise ValueError("max-samples is a single-process smoke-test option")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    fragment_path = output_dir / f"{args.split}.rank-{rank:05d}.fragment.json"
    existing_rank_files = list(output_dir.glob(f"{args.split}-r{rank:05d}-s*.jsonl"))
    if not args.overwrite and (fragment_path.exists() or existing_rank_files):
        raise FileExistsError(
            f"Rank {rank} output already exists in {output_dir}; pass --overwrite only after checking it"
        )
    if args.overwrite:
        fragment_path.unlink(missing_ok=True)
        for path in existing_rank_files:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".idx").unlink(missing_ok=True)

    dataset = build_raw_imagenet_dataset(
        parquet_glob=args.parquet_glob,
        image_root=args.image_root,
        split=args.split,
        synset_mapping=args.synset_mapping,
        image_size=256,
        random_horizontal_flip=False,
        rank=rank,
        world_size=world_size,
    )
    # Striding partitions records exactly once; DistributedSampler would pad and duplicate.
    local_dataset = (
        dataset
        if isinstance(dataset, IterableDataset)
        else Subset(dataset, range(rank, len(dataset), world_size))
    )
    loader = DataLoader(
        local_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    tokenizer = load_frozen_maskgit_vqgan(
        args.checkpoint,
        rear_repo=args.rear_repo,
        strict=not args.allow_unexpected_checkpoint_keys,
    ).to(device)
    views = args.views.split(",")
    shard_metadata: list[dict[str, Any]] = []
    writer: JsonlShardWriter | None = None
    shard_index = 0
    local_records = 0
    source_images_seen = 0
    progress_start = time.monotonic()
    next_progress_record = args.log_every_records

    def open_writer() -> JsonlShardWriter:
        nonlocal shard_index
        path = output_dir / f"{args.split}-r{rank:05d}-s{shard_index:05d}.jsonl"
        shard_index += 1
        return JsonlShardWriter(path)

    try:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            if args.max_samples is not None:
                remaining = args.max_samples - source_images_seen
                if remaining <= 0:
                    break
                if images.shape[0] > remaining:
                    images = images[:remaining]
                    batch["class_labels"] = batch["class_labels"][:remaining]
                    batch["sample_id"] = batch["sample_id"][:remaining]
            labels = batch["class_labels"].tolist()
            sample_ids = list(batch["sample_id"])
            source_images_seen += images.shape[0]
            view_batches = [images]
            if "hflip" in views:
                view_batches.append(torch.flip(images, dims=(-1,)))
            all_tokens = tokenizer.encode(torch.cat(view_batches, dim=0)).cpu()

            batch_size = images.shape[0]
            for view_index, view_name in enumerate(views):
                tokens_for_view = all_tokens[
                    view_index * batch_size : (view_index + 1) * batch_size
                ]
                for item_index in range(batch_size):
                    if writer is None:
                        writer = open_writer()
                    sample_id = str(sample_ids[item_index])
                    writer.write(
                        {
                            "sample_id": sample_id
                            if view_name == "original"
                            else f"{sample_id}:{view_name}",
                            "class_id": int(labels[item_index]),
                            "tokens": tokens_for_view[item_index].tolist(),
                            "view": view_name,
                        }
                    )
                    local_records += 1
                    if (
                        args.log_every_records > 0
                        and local_records >= next_progress_record
                    ):
                        elapsed = max(time.monotonic() - progress_start, 1.0e-9)
                        print(
                            json.dumps(
                                {
                                    "event": "pretokenize_progress",
                                    "rank": rank,
                                    "split": args.split,
                                    "local_records": local_records,
                                    "source_images_seen": source_images_seen,
                                    "elapsed_seconds": elapsed,
                                    "records_per_second": local_records / elapsed,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        next_progress_record += args.log_every_records
                    if writer.num_records >= args.records_per_shard:
                        shard_metadata.append(shard_metadata_dict(writer.close()))
                        writer = None
        if writer is not None:
            shard_metadata.append(shard_metadata_dict(writer.close()))
            writer = None
    except BaseException:
        if writer is not None:
            writer.abort()
        raise

    atomic_json(
        fragment_path,
        {
            "rank": rank,
            "world_size": world_size,
            "split": args.split,
            "num_records": local_records,
            "shards": shard_metadata,
        },
    )
    elapsed = max(time.monotonic() - progress_start, 1.0e-9)
    print(
        json.dumps(
            {
                "event": "pretokenize_rank_complete",
                "rank": rank,
                "split": args.split,
                "local_records": local_records,
                "source_images_seen": source_images_seen,
                "elapsed_seconds": elapsed,
                "records_per_second": local_records / elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if world_size > 1:
        torch.distributed.barrier(
            device_ids=[device.index] if device.type == "cuda" else None
        )
    if rank == 0:
        fragments = []
        for process_rank in range(world_size):
            path = output_dir / f"{args.split}.rank-{process_rank:05d}.fragment.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed rank fragment: {path}")
            fragments.append(json.loads(path.read_text(encoding="utf-8")))
        shards = [shard for fragment in fragments for shard in fragment["shards"]]
        num_records = sum(int(fragment["num_records"]) for fragment in fragments)
        source_count = len(dataset)
        if args.max_samples is not None:
            source_count = min(source_count, args.max_samples)
        expected = source_count * len(views)
        if num_records != expected:
            raise RuntimeError(
                f"Pretokenization count mismatch: wrote {num_records}, expected {expected}"
            )
        checkpoint_path = str(args.checkpoint.expanduser().resolve())
        checkpoint_digest = sha256_file(args.checkpoint)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") != MANIFEST_FORMAT:
                raise RuntimeError(f"Cannot merge unsupported manifest {manifest_path}")
            old_tokenizer = manifest.get("tokenizer", {})
            expected = (
                tokenizer.spec.codebook_size,
                tokenizer.spec.codebook_dim,
                tokenizer.spec.num_tokens,
                checkpoint_digest,
            )
            actual = (
                old_tokenizer.get("codebook_size"),
                old_tokenizer.get("codebook_dim"),
                old_tokenizer.get("num_tokens"),
                old_tokenizer.get("checkpoint_sha256"),
            )
            if actual != expected:
                raise RuntimeError(
                    "Existing manifest uses a different tokenizer contract: "
                    f"existing={actual}, current={expected}"
                )
        else:
            manifest = {
                "format": MANIFEST_FORMAT,
                "dataset": "imagenet-1k",
                "tokenizer": {
                    "name": "maskgit-vqgan-imagenet-f16-256",
                    "checkpoint": checkpoint_path,
                    "checkpoint_sha256": checkpoint_digest,
                    "codebook_size": tokenizer.spec.codebook_size,
                    "codebook_dim": tokenizer.spec.codebook_dim,
                    "num_tokens": tokenizer.spec.num_tokens,
                    "image_size": tokenizer.spec.image_size,
                },
                "sources": {},
                "views_by_split": {},
                "splits": {},
            }
        manifest.setdefault("sources", {})[args.split] = {
            "kind": "parquet" if args.parquet_glob else "image-files",
            "parquet_glob": args.parquet_glob,
            "image_root": str(args.image_root.resolve()) if args.image_root else None,
            "synset_mapping": str(args.synset_mapping.resolve())
            if args.synset_mapping
            else None,
        }
        manifest.setdefault("views_by_split", {})[args.split] = views
        manifest.setdefault("splits", {})[args.split] = {
            "num_records": num_records,
            "shards": shards,
        }
        atomic_json(manifest_path, manifest)
        print(
            f"Wrote {num_records} records and {len(shards)} shards to {manifest_path}"
        )
    if world_size > 1:
        torch.distributed.barrier(
            device_ids=[device.index] if device.type == "cuda" else None
        )
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
