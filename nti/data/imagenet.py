"""Raw ImageNet readers with MaskGIT-compatible image preprocessing."""

from __future__ import annotations

import glob
import io
import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".JPEG"}


def center_crop_arr(image: Image.Image, image_size: int) -> Image.Image:
    """ADM/DiT center crop, matching the reAR pretokenization script."""

    while min(image.size) >= 2 * image_size:
        image = image.resize(tuple(size // 2 for size in image.size), resample=Image.Resampling.BOX)
    scale = image_size / min(image.size)
    image = image.resize(tuple(round(size * scale) for size in image.size), resample=Image.Resampling.BICUBIC)
    array = np.asarray(image.convert("RGB"))
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return Image.fromarray(array[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class MaskGITImageTransform:
    """Convert a PIL image to `[3,H,W]` float in `[0,1]`."""

    def __init__(self, image_size: int = 256, random_horizontal_flip: bool = False):
        self.image_size = image_size
        self.random_horizontal_flip = random_horizontal_flip

    def __call__(self, image: Image.Image) -> Tensor:
        image = center_crop_arr(image, self.image_size)
        if self.random_horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous().div_(255.0)
        return tensor


def load_synset_mapping(path: str | Path) -> dict[str, int]:
    """Load an explicit synset-to-class-id mapping from JSON or text."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Synset mapping not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            mapping = {str(key): int(value) for key, value in payload.items()}
        elif isinstance(payload, Sequence):
            mapping = {str(synset): index for index, synset in enumerate(payload)}
        else:
            raise ValueError("Synset JSON must be an object or ordered list")
    else:
        mapping: dict[str, int] = {}
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) == 1:
                synset, class_id = parts[0], len(mapping)
            elif len(parts) >= 2:
                synset, class_id = parts[0], int(parts[1])
            else:  # pragma: no cover - split always returns >= 1 here
                raise ValueError(f"Invalid mapping line {line_number}: {raw!r}")
            if synset in mapping:
                raise ValueError(f"Duplicate synset {synset!r} at line {line_number}")
            mapping[synset] = class_id
    if len(mapping) != 1000 or set(mapping.values()) != set(range(1000)):
        raise ValueError("ImageNet synset mapping must be a bijection onto class IDs 0..999")
    return mapping


class ImageNetFileDataset(Dataset[dict[str, Any]]):
    """ImageNet in class directories or a flat JPEG directory.

    Flat layouts require an explicit mapping because silently sorting synsets can
    assign labels that disagree with the official ImageNet class IDs.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str | None = None,
        synset_mapping: str | Path | Mapping[str, int] | None = None,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ):
        root = Path(root).expanduser().resolve()
        split_root = root / split if split and (root / split).is_dir() else root
        if not split_root.is_dir():
            raise FileNotFoundError(f"ImageNet image directory not found: {split_root}")
        self.root = split_root
        self.transform = transform or MaskGITImageTransform()

        nested_files = sorted(
            path for path in split_root.glob("*/*") if path.is_file() and path.suffix.lower() in {x.lower() for x in IMAGE_EXTENSIONS}
        )
        flat_files = sorted(
            path for path in split_root.iterdir() if path.is_file() and path.suffix.lower() in {x.lower() for x in IMAGE_EXTENSIONS}
        )
        if nested_files and flat_files:
            raise ValueError(f"Mixed flat and class-directory layouts are not supported: {split_root}")
        if not nested_files and not flat_files:
            raise ValueError(f"No supported images found under {split_root}")

        if isinstance(synset_mapping, Mapping):
            mapping = {str(key): int(value) for key, value in synset_mapping.items()}
        elif synset_mapping is not None:
            mapping = load_synset_mapping(synset_mapping)
        elif nested_files:
            synsets = sorted({path.parent.name for path in nested_files})
            if len(synsets) != 1000:
                raise ValueError(
                    f"Class-directory ImageNet must expose 1000 synsets, found {len(synsets)}. "
                    "Pass synset_mapping explicitly for a subset."
                )
            mapping = {synset: index for index, synset in enumerate(synsets)}
        else:
            raise ValueError("Flat ImageNet JPEG layout requires synset_mapping")

        files = nested_files or flat_files
        samples: list[tuple[Path, int, str]] = []
        for path in files:
            synset = path.parent.name if nested_files else path.name.split("_", 1)[0]
            if synset not in mapping:
                raise ValueError(f"No class ID for synset {synset!r} from {path.name}")
            class_id = int(mapping[synset])
            if not 0 <= class_id < 1000:
                raise ValueError(f"Class ID must be in [0,1000), got {class_id} for {synset}")
            samples.append((path, class_id, path.stem))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, class_id, sample_id = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "class_labels": torch.tensor(class_id, dtype=torch.long),
            "sample_id": sample_id,
            "source_path": str(path),
        }


def _decode_hf_image(value: Any, parquet_parent: Path) -> Image.Image:
    if isinstance(value, Mapping):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image_path:
            path = Path(str(image_path))
            if not path.is_absolute():
                path = parquet_parent / path
            return Image.open(path).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = parquet_parent / path
        return Image.open(path).convert("RGB")
    raise TypeError(f"Unsupported Hugging Face Image feature payload: {type(value).__name__}")


class ImageNetParquetDataset(IterableDataset[dict[str, Any]]):
    """Streaming reader for Hugging Face ImageNet parquet shards.

    `pyarrow` is imported lazily so pretokenized-only training does not require it.
    Shards are partitioned exactly once across distributed ranks and loader workers;
    images are decoded in bounded Arrow batches rather than rereading a row group
    for every sample.
    """

    def __init__(
        self,
        parquet_glob: str | Sequence[str],
        *,
        image_column: str = "image",
        label_column: str = "label",
        transform: Callable[[Image.Image], Tensor] | None = None,
        rank: int = 0,
        world_size: int = 1,
        parquet_batch_size: int = 64,
    ):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Reading ImageNet parquet requires pyarrow") from exc

        patterns = [parquet_glob] if isinstance(parquet_glob, str) else list(parquet_glob)
        files = sorted({Path(match).resolve() for pattern in patterns for match in glob.glob(pattern)})
        if not files:
            raise FileNotFoundError(f"No parquet files matched: {patterns}")
        self.files = files
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform or MaskGITImageTransform()
        self.rank = rank
        self.world_size = world_size
        self.parquet_batch_size = parquet_batch_size
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0,{world_size}), got {rank}")

        total = 0
        for path in files:
            metadata = pq.ParquetFile(path).metadata
            names = set(metadata.schema.names)
            # Struct leaves can make metadata names less informative, so label is mandatory here.
            if self.label_column not in names:
                raise ValueError(f"Parquet {path} lacks label column {self.label_column!r}; columns={sorted(names)}")
            total += metadata.num_rows
        self.num_rows = total

    def __len__(self) -> int:
        return self.num_rows

    def __iter__(self):
        import pyarrow.parquet as pq

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers_per_rank = 1 if worker is None else worker.num_workers
        global_worker = self.rank * workers_per_rank + worker_id
        global_workers = self.world_size * workers_per_rank
        assigned = self.files[global_worker::global_workers]
        for path in assigned:
            parquet = pq.ParquetFile(path)
            row_offset = 0
            for batch in parquet.iter_batches(
                batch_size=self.parquet_batch_size,
                columns=[self.image_column, self.label_column],
            ):
                rows = batch.to_pylist()
                for local_index, payload in enumerate(rows):
                    class_id = int(payload[self.label_column])
                    if not 0 <= class_id < 1000:
                        raise ValueError(f"Parquet class ID must be in [0,1000), got {class_id}")
                    image = _decode_hf_image(payload[self.image_column], path.parent)
                    try:
                        tensor = self.transform(image)
                    finally:
                        image.close()
                    yield {
                        "image": tensor,
                        "class_labels": torch.tensor(class_id, dtype=torch.long),
                        "sample_id": f"{path.stem}:{row_offset + local_index}",
                        "source_path": str(path),
                    }
                row_offset += len(rows)


def build_raw_imagenet_dataset(
    *,
    parquet_glob: str | Sequence[str] | None = None,
    image_root: str | Path | None = None,
    split: str | None = None,
    synset_mapping: str | Path | Mapping[str, int] | None = None,
    image_size: int = 256,
    random_horizontal_flip: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Dataset[dict[str, Any]]:
    if (parquet_glob is None) == (image_root is None):
        raise ValueError("Specify exactly one of parquet_glob or image_root")
    transform = MaskGITImageTransform(image_size, random_horizontal_flip)
    if parquet_glob is not None:
        return ImageNetParquetDataset(
            parquet_glob,
            transform=transform,
            rank=rank,
            world_size=world_size,
        )
    return ImageNetFileDataset(
        image_root,
        split=split,
        synset_mapping=synset_mapping,
        transform=transform,
    )
