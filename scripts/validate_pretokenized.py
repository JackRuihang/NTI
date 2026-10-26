#!/usr/bin/env python3
"""Validate NTI pretokenized ImageNet manifests or index legacy reAR JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nti.data import (  # noqa: E402
    MANIFEST_FORMAT,
    build_jsonl_index,
    index_path_for,
    sha256_file,
    validate_manifest,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--legacy-jsonl", type=Path)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--tokenizer-checkpoint", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--check-checksums", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def legacy_to_manifest(args: argparse.Namespace) -> Path:
    jsonl = args.legacy_jsonl.expanduser().resolve()
    if not jsonl.is_file():
        raise FileNotFoundError(jsonl)
    if args.tokenizer_checkpoint is None:
        raise ValueError("Legacy JSONL conversion requires --tokenizer-checkpoint")
    tokenizer_checkpoint = args.tokenizer_checkpoint.expanduser().resolve()
    if not tokenizer_checkpoint.is_file():
        raise FileNotFoundError(tokenizer_checkpoint)
    tokenizer_sha256 = sha256_file(tokenizer_checkpoint)
    index = index_path_for(jsonl)
    if args.build_index:
        count = build_jsonl_index(jsonl, index)
    elif index.is_file() and index.stat().st_size % 8 == 0:
        count = index.stat().st_size // 8
    else:
        raise FileNotFoundError(f"Missing {index}; pass --build-index")
    if args.write_manifest is None:
        raise ValueError(
            "Legacy JSONL validation requires --write-manifest to make the inferred contract explicit"
        )
    manifest_path = args.write_manifest.expanduser().resolve()
    try:
        relative_jsonl = jsonl.relative_to(manifest_path.parent)
        relative_index = index.relative_to(manifest_path.parent)
    except ValueError as exc:
        raise ValueError(
            "Legacy JSONL and index must be under the manifest directory"
        ) from exc
    manifest = {
        "format": MANIFEST_FORMAT,
        "dataset": "imagenet-1k",
        "tokenizer": {
            "name": "maskgit-vqgan-imagenet-f16-256",
            "checkpoint": str(tokenizer_checkpoint),
            "checkpoint_sha256": tokenizer_sha256,
            "codebook_size": 1024,
            "codebook_dim": 256,
            "num_tokens": 256,
            "image_size": 256,
        },
        "source": {"kind": "legacy-rear-jsonl"},
        "views": ["unspecified"],
        "splits": {
            args.split: {
                "num_records": count,
                "shards": [
                    {
                        "path": relative_jsonl.as_posix(),
                        "index_path": relative_index.as_posix(),
                        "num_records": count,
                        "sha256": sha256_file(jsonl),
                    }
                ],
            }
        },
    }
    write_manifest(manifest_path, manifest)
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("max-records must be positive")
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else legacy_to_manifest(args)
    )
    report = validate_manifest(
        manifest_path,
        check_checksums=args.check_checksums,
        max_records=args.max_records,
    )
    print(json.dumps(report.__dict__, indent=2))
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
