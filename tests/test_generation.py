from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nti.generation import (  # noqa: E402
    SamplingConfig,
    autoregressive_sample,
    filter_logits,
    guidance_scale_at,
    guidance_scale_schedule,
)


class CountingAR:
    def __init__(self, vocabulary_size: int = 8, unconditional_id: int = 1000):
        self.vocabulary_size = vocabulary_size
        self.unconditional_id = unconditional_id
        self.prefill_batches: list[int] = []
        self.decode_batches: list[int] = []

    def _logits(
        self, labels: torch.Tensor, last: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = torch.full(
            (labels.shape[0], self.vocabulary_size), -5.0, device=labels.device
        )
        if last is None:
            target = labels.remainder(self.vocabulary_size)
        else:
            target = (last + 1).remainder(self.vocabulary_size)
        logits.scatter_(1, target[:, None], 5.0)
        # The unconditional branch disagrees on the initial token, exercising CFG arithmetic.
        logits[labels == self.unconditional_id] = torch.tensor(
            [5.0] + [-5.0] * (self.vocabulary_size - 1), device=labels.device
        )
        return logits

    def prefill(self, class_labels: torch.Tensor):
        self.prefill_batches.append(class_labels.shape[0])
        cache = {"labels": class_labels.clone(), "steps": 0}
        return self._logits(class_labels), cache

    def decode_step(self, last_token: torch.Tensor, cache):
        self.decode_batches.append(last_token.shape[0])
        cache["steps"] += 1
        return self._logits(cache["labels"], last_token), cache


class GenerationTests(unittest.TestCase):
    def test_greedy_generation_uses_prefill_then_incremental_steps(self) -> None:
        model = CountingAR()
        output = autoregressive_sample(
            model,
            torch.tensor([2, 3]),
            config=SamplingConfig(num_tokens=4, do_sample=False, guidance_scale=1.0),
        )
        self.assertEqual(output.tokens.tolist(), [[2, 3, 4, 5], [3, 4, 5, 6]])
        self.assertEqual(model.prefill_batches, [2])
        self.assertEqual(model.decode_batches, [2, 2, 2])
        self.assertEqual(tuple(output.token_logprobs.shape), (2, 4))

    def test_cfg_concatenates_conditional_and_unconditional_batches(self) -> None:
        model = CountingAR(unconditional_id=99)
        config = SamplingConfig(
            num_tokens=3,
            do_sample=False,
            guidance_scale=2.0,
            guidance_schedule="constant",
            unconditional_class_id=99,
        )
        output = autoregressive_sample(model, torch.tensor([2, 3]), config=config)
        self.assertEqual(model.prefill_batches, [4])
        self.assertEqual(model.decode_batches, [4, 4])
        self.assertEqual(output.tokens[:, 0].tolist(), [2, 3])
        self.assertTrue(torch.all(output.guidance_scales == 2.0))

    def test_prefix_is_consumed_before_sampling_and_marked_unsupervised(self) -> None:
        model = CountingAR()
        prefix = torch.tensor([[6, 1]])
        output = autoregressive_sample(
            model,
            torch.tensor([4]),
            config=SamplingConfig(num_tokens=4, do_sample=False),
            prefix_tokens=prefix,
        )
        self.assertEqual(output.tokens.tolist(), [[6, 1, 2, 3]])
        self.assertTrue(torch.isnan(output.token_logprobs[0, :2]).all())
        # Two prefix tokens and one generated token are consumed; the final token is not.
        self.assertEqual(model.decode_batches, [1, 1, 1])

    def test_full_length_prefix_returns_without_cache_overrun(self) -> None:
        model = CountingAR()
        prefix = torch.tensor([[6, 1, 4, 2]])
        output = autoregressive_sample(
            model,
            torch.tensor([4]),
            config=SamplingConfig(num_tokens=4, do_sample=False),
            prefix_tokens=prefix,
        )
        torch.testing.assert_close(output.tokens, prefix)
        self.assertTrue(torch.isnan(output.token_logprobs).all())
        self.assertTrue(torch.isnan(output.guidance_scales).all())
        self.assertEqual(model.prefill_batches, [1])
        self.assertEqual(model.decode_batches, [])

    def test_filters_keep_top_k_and_at_least_one_nucleus_token(self) -> None:
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        top_k = filter_logits(logits, top_k=2, top_p=None)
        self.assertEqual(torch.isfinite(top_k).sum().item(), 2)
        nucleus = filter_logits(logits, top_k=None, top_p=0.01)
        self.assertEqual(torch.isfinite(nucleus).sum().item(), 1)

    def test_cosine_cfg_starts_at_one_and_increases(self) -> None:
        config = SamplingConfig(
            num_tokens=256, guidance_scale=22.0, guidance_scale_pow=2.75
        )
        self.assertAlmostEqual(guidance_scale_at(0, config), 1.0)
        self.assertGreater(
            guidance_scale_at(255, config), guidance_scale_at(128, config)
        )

    def test_cfg_schedule_matches_rear_torch_formula(self) -> None:
        config = SamplingConfig(
            num_tokens=256, guidance_scale=22.0, guidance_scale_pow=2.75
        )
        actual = guidance_scale_schedule(config, "cpu")
        positions = torch.arange(256, dtype=torch.float32)
        scale_pow = torch.ones(1, dtype=torch.float32) * 2.75
        expected = (
            1.0
            + 21.0
            * (1.0 - torch.cos(((positions / 256) ** scale_pow) * torch.pi))
            * 0.5
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_cfg_requires_unconditional_class_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "unconditional_class_id"):
            SamplingConfig(guidance_scale=2.0).validate()


if __name__ == "__main__":
    unittest.main()
