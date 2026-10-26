from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nti.data import (  # noqa: E402
    MANIFEST_FORMAT,
    ImageNetFileDataset,
    JsonlShardWriter,
    MaskGITImageTransform,
    PretokenizedImageNetDataset,
    build_jsonl_index,
    index_path_for,
    shard_metadata_dict,
    validate_manifest,
    verify_manifest_tokenizer_checkpoint,
    write_manifest,
)


def make_manifest(directory: Path, shard: dict, count: int) -> Path:
    path = directory / "manifest.json"
    write_manifest(
        path,
        {
            "format": MANIFEST_FORMAT,
            "dataset": "imagenet-1k",
            "tokenizer": {
                "name": "maskgit-vqgan-imagenet-f16-256",
                "codebook_size": 1024,
                "codebook_dim": 256,
                "num_tokens": 256,
                "image_size": 256,
            },
            "splits": {"train": {"num_records": count, "shards": [shard]}},
        },
    )
    return path


class RawImageNetTests(unittest.TestCase):
    def test_maskgit_transform_is_rgb_chw_in_unit_range(self) -> None:
        image = Image.new("RGB", (500, 300), color=(255, 128, 0))
        tensor = MaskGITImageTransform(256)(image)
        self.assertEqual(tuple(tensor.shape), (3, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_flat_jpeg_requires_explicit_synset_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (256, 256)).save(root / "n00000001_1.JPEG")
            with self.assertRaisesRegex(ValueError, "requires synset_mapping"):
                ImageNetFileDataset(root)
            dataset = ImageNetFileDataset(root, synset_mapping={"n00000001": 7})
            item = dataset[0]
            self.assertEqual(int(item["class_labels"]), 7)
            self.assertEqual(tuple(item["image"].shape), (3, 256, 256))


class PretokenizedDataTests(unittest.TestCase):
    def test_training_checkpoint_must_match_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "tokenizer.bin"
            checkpoint.write_bytes(b"frozen tokenizer")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest = {"tokenizer": {"checkpoint_sha256": digest}}

            self.assertEqual(
                verify_manifest_tokenizer_checkpoint(manifest, checkpoint), digest
            )
            manifest["tokenizer"]["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_manifest_tokenizer_checkpoint(manifest, checkpoint)

    def test_indexed_roundtrip_and_manifest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = JsonlShardWriter(root / "train-00000.jsonl")
            writer.write(
                {
                    "sample_id": "a",
                    "class_id": 3,
                    "tokens": [1] * 256,
                    "view": "original",
                }
            )
            writer.write(
                {"sample_id": "b", "class_id": 4, "tokens": [2] * 256, "view": "hflip"}
            )
            metadata = writer.close()
            manifest = make_manifest(root, shard_metadata_dict(metadata), 2)

            report = validate_manifest(manifest, check_checksums=True)
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.num_records_checked, 2)
            dataset = PretokenizedImageNetDataset(manifest)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(dataset[1]["image_tokens"].shape), (256,))
            self.assertEqual(int(dataset[1]["class_labels"]), 4)
            self.assertEqual(dataset[1]["view"], "hflip")
            dataset.close()

    def test_legacy_records_without_sample_id_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "maskgitvq.jsonl"
            jsonl.write_text(
                json.dumps({"class_id": 1, "tokens": [5] * 256}) + "\n",
                encoding="utf-8",
            )
            count = build_jsonl_index(jsonl)
            self.assertEqual(count, 1)
            self.assertTrue(index_path_for(jsonl).is_file())
            shard = {
                "path": jsonl.name,
                "index_path": index_path_for(jsonl).name,
                "num_records": 1,
                "sha256": "unchecked",
            }
            manifest = make_manifest(root, shard, 1)
            dataset = PretokenizedImageNetDataset(manifest)
            self.assertEqual(dataset[0]["sample_id"], "record-0")
            dataset.close()

    def test_validation_catches_out_of_range_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = JsonlShardWriter(root / "bad.jsonl")
            writer.write({"sample_id": "bad", "class_id": 0, "tokens": [1024] * 256})
            metadata = writer.close()
            manifest = make_manifest(root, shard_metadata_dict(metadata), 1)
            report = validate_manifest(manifest)
            self.assertFalse(report.ok)
            self.assertTrue(any("token range" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
