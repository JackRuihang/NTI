"""Paper-evaluation sample planning and ADM-compatible NPZ packaging."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RearSamplingProtocol:
    num_samples: int = 50_000
    num_classes: int = 1_000
    per_rank_batch_size: int = 50
    world_size: int = 8
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    guidance_scale: float = 22.0
    guidance_scale_pow: float = 2.75
    seed: int = 42


REAR_SAMPLING_PROTOCOL = RearSamplingProtocol()
REAR_PROTOCOL_NAME = "rear_released_50k_v1"
CUSTOM_PROTOCOL_NAME = "custom_nti_sampling_v1"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_rear_sampling_protocol(
    *,
    num_samples: int,
    num_classes: int,
    per_rank_batch_size: int,
    world_size: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    guidance_scale: float,
    guidance_scale_pow: float,
    seed: int,
) -> None:
    """Reject a run mislabeled as the released REAR 50K protocol."""

    actual = {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "per_rank_batch_size": per_rank_batch_size,
        "world_size": world_size,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "guidance_scale": guidance_scale,
        "guidance_scale_pow": guidance_scale_pow,
        "seed": seed,
    }
    expected = asdict(REAR_SAMPLING_PROTOCOL)
    mismatches = {
        key: {"expected": expected[key], "actual": value}
        for key, value in actual.items()
        if value != expected[key]
    }
    if mismatches:
        raise ValueError(
            "rear-formal sampling arguments do not match the released protocol: "
            + json.dumps(mismatches, sort_keys=True)
        )


def balanced_class_plan(
    num_samples: int,
    num_classes: int,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> list[tuple[int, int]]:
    """Return this rank's `(global_sample_index, class_id)` assignments.

    Global index modulo class count makes the complete sample set exactly class
    balanced while striding indices across ranks avoids filename collisions.
    """

    if num_samples <= 0 or num_classes <= 0:
        raise ValueError("num_samples and num_classes must be positive")
    if num_samples % num_classes:
        raise ValueError("num_samples must be divisible by num_classes")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank/world_size are invalid")
    return [
        (index, index % num_classes) for index in range(rank, num_samples, world_size)
    ]


def rear_class_plan(
    num_samples: int,
    num_classes: int,
    *,
    per_rank_batch_size: int,
    rank: int = 0,
    world_size: int = 1,
) -> list[tuple[int, int]]:
    """Reproduce REAR's rank-local class stream and interleaved filenames.

    REAR constructs ``0..num_classes-1`` repeatedly, gives each rank one
    contiguous slice, then interleaves filenames across ranks within every
    batch.  The pairing between class labels and rank-local RNG streams matters
    for an exact protocol reproduction even though a strided plan is also
    globally class balanced.
    """

    if num_samples <= 0 or num_classes <= 0 or per_rank_batch_size <= 0:
        raise ValueError("sample counts, class count, and batch size must be positive")
    if num_samples % num_classes:
        raise ValueError("num_samples must be divisible by num_classes")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank/world_size are invalid")
    global_batch_size = per_rank_batch_size * world_size
    if num_samples % global_batch_size:
        raise ValueError(
            "REAR-compatible sampling requires num_samples divisible by "
            "per_rank_batch_size * world_size"
        )

    samples_per_rank = num_samples // world_size
    class_stream_start = rank * samples_per_rank
    plan: list[tuple[int, int]] = []
    for local_index in range(samples_per_rank):
        batch_index, within_batch = divmod(local_index, per_rank_batch_size)
        sample_index = (
            batch_index * global_batch_size + within_batch * world_size + rank
        )
        class_id = (class_stream_start + local_index) % num_classes
        plan.append((sample_index, class_id))
    return plan


def validate_complete_plan(
    records: list[dict[str, object]],
    *,
    num_samples: int,
    num_classes: int,
) -> None:
    """Reject missing, duplicate, or class-imbalanced merged sample metadata."""

    indices = [int(record["sample_index"]) for record in records]
    if sorted(indices) != list(range(num_samples)):
        raise RuntimeError("sample metadata has missing or duplicate global indices")
    counts = Counter(int(record["class_id"]) for record in records)
    expected = num_samples // num_classes
    if set(counts) != set(range(num_classes)) or any(
        count != expected for count in counts.values()
    ):
        raise RuntimeError("sample metadata is not exactly class balanced")


def validate_sample_metadata(
    path: str | Path,
    *,
    num_samples: int,
    num_classes: int,
    required_protocol: str | None = None,
) -> dict[str, Any]:
    """Validate the merged sampler manifest used by a metric run."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sample metadata root must be a JSON object")
    if int(payload.get("num_samples", -1)) != num_samples:
        raise ValueError("sample metadata num_samples does not match evaluation")
    if int(payload.get("num_classes", -1)) != num_classes:
        raise ValueError("sample metadata num_classes does not match evaluation")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping) or not isinstance(protocol.get("name"), str):
        raise ValueError("sample metadata has no protocol name")
    if required_protocol is not None and protocol["name"] != required_protocol:
        raise ValueError(
            f"sample protocol is {protocol['name']!r}, expected {required_protocol!r}"
        )
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != num_samples:
        raise ValueError("sample metadata records have the wrong length")
    typed_records: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("sample metadata record must be an object")
        sample_index = int(record["sample_index"])
        expected_name = f"{sample_index:06d}.png"
        if record.get("path") != expected_name:
            raise ValueError(
                f"sample {sample_index} path is not the canonical filename"
            )
        typed_records.append(record)
    validate_complete_plan(
        typed_records,
        num_samples=num_samples,
        num_classes=num_classes,
    )
    if not isinstance(payload.get("provenance"), Mapping):
        raise ValueError("sample metadata has no provenance record")
    return payload


def inspect_adm_npz(
    path: str | Path,
    *,
    num_samples: int,
    image_size: int,
) -> dict[str, object]:
    """Validate an ADM NPZ from its NPY header without loading image data."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with zipfile.ZipFile(resolved, mode="r") as archive:
        members = archive.infolist()
        if [member.filename for member in members] != ["arr_0.npy"]:
            raise ValueError("ADM NPZ must contain exactly arr_0.npy")
        member = members[0]
        if member.compress_type != zipfile.ZIP_STORED:
            raise ValueError("ADM NPZ arr_0.npy must use np.savez ZIP_STORED format")
        with archive.open(member, mode="r") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    stream
                )
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    stream
                )
            else:
                raise ValueError(f"unsupported NPY header version: {version}")
            header_bytes = stream.tell()
        expected_shape = (num_samples, image_size, image_size, 3)
        if tuple(shape) != expected_shape:
            raise ValueError(
                f"ADM sample array shape is {tuple(shape)}, expected {expected_shape}"
            )
        if np.dtype(dtype) != np.dtype(np.uint8):
            raise ValueError(f"ADM sample dtype is {dtype}, expected uint8")
        if fortran_order:
            raise ValueError("ADM sample array must use C order")
        expected_member_bytes = (
            header_bytes + int(np.prod(shape)) * np.dtype(dtype).itemsize
        )
        if member.file_size != expected_member_bytes:
            raise ValueError(
                "ADM sample member size does not match its declared shape/dtype"
            )
    return {
        "shape": list(expected_shape),
        "dtype": "uint8",
        "fortran_order": False,
        "zip_member": "arr_0.npy",
        "zip_compression": "stored",
    }


def create_adm_npz(
    sample_dir: str | Path,
    *,
    num_samples: int,
    output_path: str | Path | None = None,
    image_size: int | None = None,
) -> Path:
    """Pack `000000.png ...` as an uncompressed `arr_0` ADM/FID NPZ.

    A temporary NumPy memmap keeps the 50K-image path bounded in RAM. The final
    ZIP member is still the standard `arr_0.npy` produced by `numpy.savez`.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    sample_dir = Path(sample_dir).expanduser().resolve()
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else sample_dir.with_name(sample_dir.name + ".npz")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    first_path = sample_dir / "000000.png"
    if not first_path.is_file():
        raise FileNotFoundError(first_path)
    with Image.open(first_path) as image:
        first = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"expected RGB samples, got shape {first.shape}")
    if image_size is not None and first.shape[:2] != (image_size, image_size):
        raise ValueError(
            f"expected {image_size}x{image_size} samples, got {first.shape[:2]}"
        )

    npy_temporary = output.with_name(output.name + ".arr_0.npy.tmp")
    zip_temporary = output.with_name(output.name + ".tmp")
    npy_temporary.unlink(missing_ok=True)
    zip_temporary.unlink(missing_ok=True)
    try:
        array = np.lib.format.open_memmap(
            npy_temporary,
            mode="w+",
            dtype=np.uint8,
            shape=(num_samples, *first.shape),
        )
        for index in range(num_samples):
            path = sample_dir / f"{index:06d}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                sample = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if sample.shape != first.shape:
                raise ValueError(
                    f"sample {path.name} has shape {sample.shape}, expected {first.shape}"
                )
            array[index] = sample
        array.flush()
        del array
        with zipfile.ZipFile(
            zip_temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            archive.write(npy_temporary, arcname="arr_0.npy")
        os.replace(zip_temporary, output)
    except BaseException:
        zip_temporary.unlink(missing_ok=True)
        raise
    finally:
        npy_temporary.unlink(missing_ok=True)
    return output
