#!/usr/bin/env python3
"""Atomically point a replay root at one complete immutable bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "nti.lagged_sparse_replay.pointer.v1"
REPLAY_SCHEMA = "nti.lagged_sparse_replay.v1"
SHARDED_REPLAY_SCHEMA = "nti.lagged_sparse_replay.sharded.v1"
REPLAY_SCHEMAS = {REPLAY_SCHEMA, SHARDED_REPLAY_SCHEMA}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-checkpoint", type=Path)
    parser.add_argument("--expected-target-mode", default="")
    parser.add_argument("--expected-target-encoding", default="")
    parser.add_argument("--expected-top-k", type=int)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    bank = args.bank.expanduser().resolve()
    if root not in bank.parents or bank.parent != root:
        raise ValueError("bank must be a direct child of replay root")
    manifest_path = bank / "manifest.json"
    complete_path = bank / "COMPLETE.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise ValueError("cannot publish an incomplete replay bank")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    manifest_sha256 = sha256(manifest_path)
    manifest_schema = manifest.get("schema")
    if manifest_schema not in REPLAY_SCHEMAS:
        raise ValueError("replay manifest has the wrong schema")
    if (
        complete.get("schema") != manifest_schema
        or complete.get("complete") is not True
    ):
        raise ValueError("replay completion marker is not complete")
    if complete.get("manifest_sha256") != manifest_sha256:
        raise ValueError("replay completion marker manifest checksum mismatch")
    provenance = manifest.get("provenance", {})
    if int(provenance.get("snapshot_step", -1)) != args.expected_step:
        raise ValueError("replay bank snapshot step mismatch")
    if (
        args.expected_target_mode
        and provenance.get("target_mode") != args.expected_target_mode
    ):
        raise ValueError("replay bank target mode mismatch")
    if (
        args.expected_target_encoding
        and provenance.get("target_encoding") != args.expected_target_encoding
    ):
        raise ValueError("replay bank target encoding mismatch")
    if (
        args.expected_top_k is not None
        and int(provenance.get("top_k", -1)) != args.expected_top_k
    ):
        raise ValueError("replay bank target width mismatch")
    if args.expected_checkpoint is not None:
        checkpoint = args.expected_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if int(provenance.get("teacher_checkpoint_step", -1)) != args.expected_step:
            raise ValueError("replay teacher checkpoint step mismatch")
        if Path(str(provenance.get("checkpoint", ""))).resolve() != checkpoint:
            raise ValueError("replay checkpoint path mismatch")
        if provenance.get("checkpoint_sha256") != sha256(checkpoint):
            raise ValueError("replay checkpoint checksum mismatch")
    root.mkdir(parents=True, exist_ok=True)
    pointer_path = root / "CURRENT.json"
    temporary = root / f"CURRENT.json.partial-{os.getpid()}"
    payload = {
        "schema": SCHEMA,
        "bank": bank.name,
        "manifest_sha256": manifest_sha256,
        "snapshot_step": args.expected_step,
    }
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, pointer_path)
    print(json.dumps({"event": "replay_pointer_published", **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
