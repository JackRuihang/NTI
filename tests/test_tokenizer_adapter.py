from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nti.tokenizer import MaskGITVQGANAdapter, load_maskgit_codebook  # noqa: E402


class StubQuantizer(nn.Module):
    def __init__(self, size: int = 1024, dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(size, dim)


class StubMaskGIT(nn.Module):
    def __init__(self, size: int = 1024, dim: int = 256):
        super().__init__()
        self.quantize = StubQuantizer(size, dim)
        self.offset = nn.Parameter(torch.tensor(0.25))

    def encode(self, images: torch.Tensor):
        tokens = torch.arange(256, device=images.device).remainder(
            self.quantize.embedding.num_embeddings
        )
        tokens = tokens.unsqueeze(0).expand(images.shape[0], -1)
        return images[:, :, :16, :16], {"min_encoding_indices": tokens}

    def decode_tokens(self, tokens: torch.Tensor):
        value = tokens[:, :1].float().view(-1, 1, 1, 1) * 0 + self.offset
        return value.expand(-1, 3, 256, 256)


class TokenizerAdapterTests(unittest.TestCase):
    def test_lightweight_codebook_loader_handles_wrapper_prefix(self) -> None:
        expected = torch.randn(1024, 256, dtype=torch.float16)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "maskgit.bin"
            torch.save(
                {
                    "state_dict": {
                        "module.tokenizer.quantize.embedding.weight": expected
                    }
                },
                checkpoint,
            )
            actual = load_maskgit_codebook(checkpoint)
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(tuple(actual.shape), (1024, 256))
        self.assertTrue(torch.equal(actual, expected.float()))

    def test_loads_nested_prefixed_checkpoint_and_freezes_every_parameter(self) -> None:
        source = StubMaskGIT()
        state = {
            f"module.tokenizer.{key}": value.clone()
            for key, value in source.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "maskgit.bin"
            torch.save({"state_dict": state, "epoch": 7}, checkpoint)
            adapter = MaskGITVQGANAdapter.from_checkpoint(
                checkpoint,
                backend_factory=StubMaskGIT,
            )

        self.assertEqual(adapter.checkpoint_report.payload_path, "root.state_dict")
        self.assertEqual(
            adapter.checkpoint_report.loaded_keys, len(source.state_dict())
        )
        self.assertEqual(tuple(adapter.codebook_weight.shape), (1024, 256))
        self.assertFalse(adapter.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in adapter.parameters())
        )
        adapter.train(True)
        self.assertFalse(adapter.training)
        self.assertFalse(adapter.backend.training)

    def test_encode_decode_and_embedding_shapes(self) -> None:
        adapter = MaskGITVQGANAdapter(StubMaskGIT())
        images = torch.rand(2, 3, 256, 256)
        tokens = adapter.encode(images)
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tuple(tokens.shape), (2, 256))
        self.assertEqual(tuple(adapter.token_embeddings(tokens).shape), (2, 256, 256))
        reconstructions = adapter.decode(tokens)
        self.assertEqual(tuple(reconstructions.shape), (2, 3, 256, 256))
        self.assertTrue(torch.all((reconstructions >= 0) & (reconstructions <= 1)))

    def test_rejects_wrong_codebook_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "codebook must have shape"):
            MaskGITVQGANAdapter(StubMaskGIT(size=512))

    def test_rejects_bad_image_range_and_token_count(self) -> None:
        adapter = MaskGITVQGANAdapter(StubMaskGIT())
        with self.assertRaisesRegex(ValueError, "expects images in"):
            adapter.encode(torch.full((1, 3, 256, 256), 2.0))
        with self.assertRaisesRegex(ValueError, "Expected tokens"):
            adapter.decode(torch.zeros(1, 255, dtype=torch.long))

    def test_missing_checkpoint_parameter_is_fatal_even_non_strict(self) -> None:
        backend = StubMaskGIT()
        state = backend.state_dict()
        state.pop("offset")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "maskgit.bin"
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "missing required parameters"):
                MaskGITVQGANAdapter.from_checkpoint(
                    checkpoint,
                    backend_factory=StubMaskGIT,
                    strict=False,
                )


if __name__ == "__main__":
    unittest.main()
