"""Versioned, indexed JSONL storage for pretokenized ImageNet."""

from __future__ import annotations

import bisect
import hashlib
import hmac
import json
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


MANIFEST_FORMAT = "nti.imagenet_tokens.v1"
RECORD_KEYS = frozenset({"class_id", "tokens"})


@dataclass(frozen=True)
class ShardMetadata:
    path: str
    index_path: str
    num_records: int
    sha256: str


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    manifest_path: str
    num_records_declared: int
    num_records_checked: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_tokenizer_checkpoint(
    manifest: Mapping[str, Any], checkpoint_path: str | Path
) -> str:
    """Require the training tokenizer to match the pretokenization artifact."""

    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("Manifest must contain a tokenizer object")
    expected = tokenizer.get("checkpoint_sha256")
    if not isinstance(expected, str):
        raise ValueError("Manifest tokenizer must define checkpoint_sha256")
    expected = expected.strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("Manifest tokenizer checkpoint_sha256 must be a SHA-256 hex digest")
    actual = sha256_file(checkpoint_path)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            "Tokenizer checkpoint SHA-256 does not match the pretokenized manifest: "
            f"manifest={expected}, checkpoint={actual}"
        )
    return actual


def index_path_for(jsonl_path: str | Path) -> Path:
    path = Path(jsonl_path)
    return path.with_suffix(path.suffix + ".idx")


def build_jsonl_index(jsonl_path: str | Path, index_path: str | Path | None = None) -> int:
    """Build an atomic little-endian uint64 byte-offset index."""

    jsonl_path = Path(jsonl_path)
    index_path = Path(index_path) if index_path is not None else index_path_for(jsonl_path)
    temporary = index_path.with_name(index_path.name + f".tmp-{os.getpid()}")
    count = 0
    try:
        with jsonl_path.open("rb") as source, temporary.open("wb") as target:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if line.strip():
                    target.write(struct.pack("<Q", offset))
                    count += 1
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(index_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


class JsonlShardWriter:
    """Crash-safe writer that emits JSONL and its random-access index together."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.index_path = index_path_for(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        suffix = f".tmp-{os.getpid()}"
        self._temporary_path = self.path.with_name(self.path.name + suffix)
        self._temporary_index = self.index_path.with_name(self.index_path.name + suffix)
        self._data = self._temporary_path.open("wb")
        self._index = self._temporary_index.open("wb")
        self._digest = hashlib.sha256()
        self._count = 0
        self._closed = False

    @property
    def num_records(self) -> int:
        return self._count

    def write(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed shard")
        missing = RECORD_KEYS - set(record)
        if missing:
            raise ValueError(f"Pretokenized record is missing keys: {sorted(missing)}")
        offset = self._data.tell()
        encoded = (json.dumps(dict(record), separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
        self._index.write(struct.pack("<Q", offset))
        self._data.write(encoded)
        self._digest.update(encoded)
        self._count += 1

    def close(self) -> ShardMetadata:
        if self._closed:
            raise RuntimeError("Shard is already closed")
        self._closed = True
        try:
            for handle in (self._data, self._index):
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            self._temporary_path.replace(self.path)
            self._temporary_index.replace(self.index_path)
        except BaseException:
            self._temporary_path.unlink(missing_ok=True)
            self._temporary_index.unlink(missing_ok=True)
            raise
        return ShardMetadata(
            path=self.path.name,
            index_path=self.index_path.name,
            num_records=self._count,
            sha256=self._digest.hexdigest(),
        )

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._data.close()
        self._index.close()
        self._temporary_path.unlink(missing_ok=True)
        self._temporary_index.unlink(missing_ok=True)

    def __enter__(self) -> "JsonlShardWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    if payload.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"Unsupported manifest format {payload.get('format')!r}; expected {MANIFEST_FORMAT!r}")
    return payload


class _IndexedJsonlShard:
    def __init__(self, path: Path, index_path: Path, expected_records: int | None = None):
        self.path = path
        self.index_path = index_path
        if not path.is_file():
            raise FileNotFoundError(f"Pretokenized shard not found: {path}")
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Pretokenized index not found: {index_path}. Run validate_pretokenized.py --build-index."
            )
        if index_path.stat().st_size % 8:
            raise ValueError(f"Invalid uint64 index size: {index_path}")
        self.offsets = np.memmap(index_path, mode="r", dtype="<u8")
        if expected_records is not None and len(self.offsets) != expected_records:
            raise ValueError(
                f"Shard count mismatch for {path}: manifest={expected_records}, index={len(self.offsets)}"
            )
        self._handle: Any = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def read(self, index: int) -> dict[str, Any]:
        if self._handle is None:
            self._handle = self.path.open("rb")
        self._handle.seek(int(self.offsets[index]))
        line = self._handle.readline()
        if not line:
            raise RuntimeError(f"Index offset {index} points beyond EOF in {self.path}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Record {index} in {self.path} is not a JSON object")
        return payload

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None
        offsets = getattr(self, "offsets", None)
        if offsets is not None:
            mapping = getattr(offsets, "_mmap", None)
            if mapping is not None:
                mapping.close()
            self.offsets = None

    def __del__(self) -> None:  # pragma: no cover - timing is interpreter dependent
        self.close()


class PretokenizedImageNetDataset(Dataset[dict[str, Any]]):
    """Random-access dataset backed by an NTI manifest and indexed shards."""

    def __init__(self, manifest_path: str | Path, *, split: str = "train"):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest = load_manifest(self.manifest_path)
        tokenizer = self.manifest.get("tokenizer", {})
        self.codebook_size = int(tokenizer.get("codebook_size", -1))
        self.num_tokens = int(tokenizer.get("num_tokens", -1))
        if self.codebook_size <= 0 or self.num_tokens <= 0:
            raise ValueError("Manifest tokenizer must define positive codebook_size and num_tokens")

        split_payload = self.manifest.get("splits", {}).get(split)
        if not isinstance(split_payload, Mapping):
            raise KeyError(f"Split {split!r} not found in {self.manifest_path}")
        shards_payload = split_payload.get("shards")
        if not isinstance(shards_payload, Sequence) or not shards_payload:
            raise ValueError(f"Split {split!r} has no shards")

        base = self.manifest_path.parent
        self.shards: list[_IndexedJsonlShard] = []
        self.cumulative: list[int] = []
        total = 0
        for item in shards_payload:
            if not isinstance(item, Mapping):
                raise ValueError("Each shard manifest entry must be an object")
            shard = _IndexedJsonlShard(
                base / str(item["path"]),
                base / str(item["index_path"]),
                expected_records=int(item["num_records"]),
            )
            self.shards.append(shard)
            total += len(shard)
            self.cumulative.append(total)
        declared = int(split_payload.get("num_records", -1))
        if declared != total:
            raise ValueError(f"Split count mismatch: manifest={declared}, shards={total}")

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative, index)
        shard_start = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        record = self.shards[shard_index].read(index - shard_start)
        class_id = int(record["class_id"])
        tokens = torch.as_tensor(record["tokens"], dtype=torch.long).reshape(-1)
        if tokens.numel() != self.num_tokens:
            raise ValueError(
                f"Record {record.get('sample_id', index)!r} has {tokens.numel()} tokens, expected {self.num_tokens}"
            )
        if not 0 <= class_id < 1000:
            raise ValueError(f"ImageNet class ID out of range: {class_id}")
        if tokens.numel() and (int(tokens.min()) < 0 or int(tokens.max()) >= self.codebook_size):
            raise ValueError(f"Token ID outside [0,{self.codebook_size})")
        return {
            "image_tokens": tokens,
            "class_labels": torch.tensor(class_id, dtype=torch.long),
            "record_index": torch.tensor(index, dtype=torch.long),
            "sample_id": str(record.get("sample_id", f"record-{index}")),
            "view": str(record.get("view", "unspecified")),
        }

    def close(self) -> None:
        for shard in self.shards:
            shard.close()

    def __del__(self) -> None:  # pragma: no cover - timing is interpreter dependent
        self.close()


def _validate_record(
    record: Mapping[str, Any],
    *,
    num_tokens: int,
    codebook_size: int,
    location: str,
) -> list[str]:
    errors: list[str] = []
    missing = RECORD_KEYS - set(record)
    if missing:
        return [f"{location}: missing keys {sorted(missing)}"]
    try:
        class_id = int(record["class_id"])
        if not 0 <= class_id < 1000:
            errors.append(f"{location}: class_id={class_id} outside [0,1000)")
    except (TypeError, ValueError):
        errors.append(f"{location}: class_id is not an integer")
    tokens = record.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != num_tokens:
        errors.append(f"{location}: expected {num_tokens} token IDs")
    else:
        try:
            low, high = min(int(token) for token in tokens), max(int(token) for token in tokens)
            if low < 0 or high >= codebook_size:
                errors.append(f"{location}: token range [{low},{high}] outside [0,{codebook_size})")
        except (TypeError, ValueError):
            errors.append(f"{location}: tokens contain a non-integer value")
    if "sample_id" in record and (not isinstance(record["sample_id"], str) or not record["sample_id"]):
        errors.append(f"{location}: sample_id must be a nonempty string when present")
    return errors


def validate_manifest(
    manifest_path: str | Path,
    *,
    check_checksums: bool = False,
    max_records: int | None = None,
) -> ValidationReport:
    manifest_path = Path(manifest_path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    declared = 0
    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        return ValidationReport(False, str(manifest_path), 0, 0, (str(exc),), ())

    tokenizer = manifest.get("tokenizer", {})
    try:
        codebook_size = int(tokenizer["codebook_size"])
        codebook_dim = int(tokenizer["codebook_dim"])
        num_tokens = int(tokenizer["num_tokens"])
        if (codebook_size, codebook_dim, num_tokens) != (1024, 256, 256):
            errors.append(
                "MaskGIT tokenizer contract must be K=1024, d=256, N=256; "
                f"got {(codebook_size, codebook_dim, num_tokens)}"
            )
    except (KeyError, TypeError, ValueError):
        return ValidationReport(False, str(manifest_path), 0, 0, ("Invalid tokenizer contract",), ())

    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or not splits:
        errors.append("Manifest must contain at least one split")
        return ValidationReport(False, str(manifest_path), 0, 0, tuple(errors), tuple(warnings))

    remaining = max_records
    for split_name, split in splits.items():
        if not isinstance(split, Mapping):
            errors.append(f"Split {split_name!r} is not an object")
            continue
        split_declared = int(split.get("num_records", -1))
        declared += max(split_declared, 0)
        shard_total = 0
        for shard_item in split.get("shards", []):
            data_path = manifest_path.parent / str(shard_item.get("path", ""))
            idx_path = manifest_path.parent / str(shard_item.get("index_path", ""))
            expected = int(shard_item.get("num_records", -1))
            try:
                shard = _IndexedJsonlShard(data_path, idx_path, expected_records=expected)
            except Exception as exc:
                errors.append(str(exc))
                continue
            shard_total += len(shard)
            if check_checksums:
                actual = sha256_file(data_path)
                if actual != shard_item.get("sha256"):
                    errors.append(f"Checksum mismatch for {data_path}: {actual}")

            limit = len(shard) if remaining is None else min(len(shard), max(remaining, 0))
            for record_index in range(limit):
                try:
                    record = shard.read(record_index)
                    errors.extend(
                        _validate_record(
                            record,
                            num_tokens=num_tokens,
                            codebook_size=codebook_size,
                            location=f"{data_path.name}:{record_index}",
                        )
                    )
                except Exception as exc:
                    errors.append(f"{data_path.name}:{record_index}: {exc}")
                checked += 1
            shard.close()
            if remaining is not None:
                remaining -= limit
        if shard_total != split_declared:
            errors.append(f"Split {split_name!r} declares {split_declared} records but shards contain {shard_total}")
    if max_records is not None and checked < declared:
        warnings.append(f"Validation sampled {checked} of {declared} declared records")
    return ValidationReport(
        ok=not errors,
        manifest_path=str(manifest_path),
        num_records_declared=declared,
        num_records_checked=checked,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def shard_metadata_dict(metadata: ShardMetadata) -> dict[str, Any]:
    return asdict(metadata)
