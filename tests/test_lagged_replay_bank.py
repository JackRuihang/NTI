from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nti.training.replay import (  # noqa: E402
    categorical_sample_detached_target,
    conditional_non_gt_sample_detached_target,
    deterministic_replay_class_drop_mask,
    LaggedReplayBank,
    ReplayBatch,
    SparseReplayTarget,
    ShardedLaggedReplayBank,
    balanced_replay_image_mask,
    open_replay_bank,
    sparse_replay_cross_entropy,
    sparsify_detached_target,
    one_pass_replay_forward,
    write_replay_bank,
    write_sharded_replay_index,
)


class CountingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.1, -0.2, 0.3, -0.4]))
        self.calls = 0
        self.last_tokens: torch.Tensor | None = None

    def forward(
        self, image_tokens: torch.Tensor, *, class_labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        del class_labels
        self.calls += 1
        self.last_tokens = image_tokens.detach().clone()
        return {
            "logits": self.bias.view(1, 1, -1).expand(
                image_tokens.shape[0], image_tokens.shape[1], -1
            )
        }


def test_sparse_target_moves_omitted_tail_back_to_gt() -> None:
    target = torch.tensor([[[0.40, 0.30, 0.20, 0.10]]])
    sparse = sparsify_detached_target(target, torch.tensor([[0]]), top_k=2)

    torch.testing.assert_close(sparse.effective_non_gt_mass, torch.tensor([[0.50]]))
    torch.testing.assert_close(
        sparse.captured_non_gt_fraction, torch.tensor([[5.0 / 6.0]])
    )
    torch.testing.assert_close(sparse.top_probabilities.sum(dim=-1), torch.ones((1, 1)))
    assert set(sparse.top_indices.flatten().tolist()) == {1, 2}


def test_sparse_cross_entropy_matches_explicit_conservative_target() -> None:
    logits = torch.tensor([[[0.2, -0.1, 0.8, -0.5]]], requires_grad=True)
    sparse = SparseReplayTarget(
        effective_non_gt_mass=torch.tensor([[0.5]]),
        top_indices=torch.tensor([[[1, 2]]]),
        top_probabilities=torch.tensor([[[0.6, 0.4]]]),
        captured_non_gt_fraction=torch.tensor([[0.8]]),
    )
    observed = sparse_replay_cross_entropy(logits, torch.tensor([[0]]), sparse)
    explicit = torch.tensor([[[0.5, 0.3, 0.2, 0.0]]])
    expected = -(explicit * logits.log_softmax(dim=-1)).sum()
    torch.testing.assert_close(observed, expected)
    observed.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_categorical_replay_cross_entropy_matches_sampled_hard_ce_average() -> None:
    target = torch.tensor([[[0.15, 0.25, 0.35, 0.25], [0.45, 0.15, 0.20, 0.20]]])
    target_ids = torch.tensor([[0, 2]])
    generator = torch.Generator().manual_seed(19)
    sampled = categorical_sample_detached_target(
        target, target_ids, sample_count=16, generator=generator
    )
    logits = torch.tensor(
        [[[0.2, -0.1, 0.8, -0.5], [-0.4, 0.6, 0.1, 0.3]]],
        requires_grad=True,
    )

    observed = sparse_replay_cross_entropy(logits, target_ids, sampled)
    non_gt = sampled.top_probabilities > 0
    realized_samples = torch.where(non_gt, sampled.top_indices, target_ids[..., None])
    expected = -logits.log_softmax(dim=-1).gather(-1, realized_samples).mean()

    torch.testing.assert_close(observed, expected)


def test_categorical_replay_is_deterministic_for_a_fixed_generator_seed() -> None:
    target = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]]).expand(2, 3, 4)
    target_ids = torch.zeros((2, 3), dtype=torch.long)
    first = categorical_sample_detached_target(
        target,
        target_ids,
        sample_count=12,
        generator=torch.Generator().manual_seed(123),
    )
    second = categorical_sample_detached_target(
        target,
        target_ids,
        sample_count=12,
        generator=torch.Generator().manual_seed(123),
    )

    torch.testing.assert_close(first.top_indices, second.top_indices)
    torch.testing.assert_close(
        first.effective_non_gt_mass, second.effective_non_gt_mass
    )
    torch.testing.assert_close(first.top_probabilities, second.top_probabilities)


def test_categorical_replay_non_gt_mass_is_unbiased() -> None:
    target = torch.tensor([[[0.7, 0.1, 0.1, 0.1]]]).expand(128, 64, 4)
    target_ids = torch.zeros((128, 64), dtype=torch.long)
    sampled = categorical_sample_detached_target(
        target,
        target_ids,
        sample_count=32,
        generator=torch.Generator().manual_seed(9),
    )

    assert sampled.effective_non_gt_mass.mean().item() == pytest.approx(0.3, abs=0.005)
    assert sampled.captured_non_gt_fraction.eq(1.0).all()


def test_conditional_replay_preserves_exact_dose_and_excludes_gt() -> None:
    target = torch.tensor([[[0.7, 0.1, 0.15, 0.05]]]).expand(64, 8, 4)
    target_ids = torch.zeros((64, 8), dtype=torch.long)
    sampled = conditional_non_gt_sample_detached_target(
        target,
        target_ids,
        sample_count=8,
        generator=torch.Generator().manual_seed(13),
    )

    torch.testing.assert_close(sampled.effective_non_gt_mass, torch.full((64, 8), 0.3))
    assert sampled.top_indices.ne(target_ids[..., None]).all()
    torch.testing.assert_close(sampled.top_probabilities, torch.ones((64, 8, 8)))


def test_conditional_replay_zero_dose_is_exactly_hard_gt() -> None:
    target = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    target_ids = torch.ones((1, 1), dtype=torch.long)
    sampled = conditional_non_gt_sample_detached_target(
        target,
        target_ids,
        sample_count=8,
        generator=torch.Generator().manual_seed(17),
    )
    logits = torch.tensor([[[0.2, -0.1, 0.7, -0.4]]])

    assert sampled.effective_non_gt_mass.eq(0.0).all()
    replay_loss = sparse_replay_cross_entropy(logits, target_ids, sampled)
    hard_gt_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_ids.flatten()
    )
    torch.testing.assert_close(replay_loss, hard_gt_loss, atol=0.0, rtol=0.0)


def test_one_pass_replay_routes_disjoint_examples_through_one_forward() -> None:
    model = CountingModel()
    clean = torch.tensor([[0, 1], [2, 3], [1, 0]])
    labels = torch.tensor([4, 5, 6])
    replay = ReplayBatch(
        record_indices=torch.tensor([10, 11, 12]),
        class_labels=labels.clone(),
        class_drop_mask=torch.tensor([False, True, False]),
        altered_tokens=torch.tensor([[3, 3], [1, 1], [2, 2]]),
        active_mask=torch.tensor([[True, True], [True, False], [False, True]]),
        effective_non_gt_mass=torch.full((3, 2), 0.25),
        top_indices=torch.tensor([[[1], [2]], [[0], [2]], [[2], [3]]]),
        top_probabilities=torch.ones((3, 2, 1)),
    )
    replay_mask = torch.tensor([False, True, False])

    output = one_pass_replay_forward(model, clean, labels, replay, replay_mask)

    assert model.calls == 1
    assert model.last_tokens is not None
    torch.testing.assert_close(
        model.last_tokens, torch.tensor([[0, 1], [1, 1], [1, 0]])
    )
    assert output.clean_loss_mask.sum().item() == 4
    assert output.replay_loss_mask.sum().item() == 1
    torch.testing.assert_close(
        output.loss, output.clean_loss + 0.5 * output.replay_loss
    )
    output.loss.backward()
    assert model.bias.grad is not None and torch.isfinite(model.bias.grad).all()


def test_all_record_replay_supervises_gt_before_active_prefix() -> None:
    model = CountingModel()
    clean = torch.tensor([[0, 1], [2, 3]])
    labels = torch.tensor([4, 5])
    replay = ReplayBatch(
        record_indices=torch.tensor([10, 11]),
        class_labels=labels.clone(),
        class_drop_mask=torch.tensor([False, True]),
        altered_tokens=torch.tensor([[3, 3], [1, 1]]),
        active_mask=torch.tensor([[False, True], [False, True]]),
        effective_non_gt_mass=torch.tensor([[0.0, 0.25], [0.0, 0.30]]),
        top_indices=torch.tensor([[[0], [2]], [[2], [1]]]),
        top_probabilities=torch.ones((2, 2, 1)),
    )
    output = one_pass_replay_forward(
        model,
        clean,
        labels,
        replay,
        torch.ones(2, dtype=torch.bool),
        clean_loss_weight=0.0,
        replay_loss_weight=1.0,
        replay_all_tokens=True,
    )

    assert output.replay_loss_mask.all()
    assert not output.clean_loss_mask.any()
    assert torch.isfinite(output.loss)


def test_balanced_replay_mask_uses_objective_weights_without_empty_subsets() -> None:
    generator = torch.Generator().manual_seed(7)
    mask = balanced_replay_image_mask(
        12,
        clean_loss_weight=1.0,
        replay_loss_weight=0.5,
        device="cpu",
        generator=generator,
    )
    assert mask.sum().item() == 4
    assert not mask.all()
    assert not mask.logical_not().all()
    assert not balanced_replay_image_mask(
        4,
        clean_loss_weight=1.0,
        replay_loss_weight=0.0,
        device="cpu",
    ).any()


def test_replay_class_drop_is_deterministic_and_order_independent() -> None:
    indices = np.asarray([9, 2, 100, 17], dtype=np.int64)
    first = deterministic_replay_class_drop_mask(indices, probability=0.25, seed=7)
    reordered = deterministic_replay_class_drop_mask(
        indices[[2, 0, 3, 1]], probability=0.25, seed=7
    )

    np.testing.assert_array_equal(reordered, first[[2, 0, 3, 1]])
    assert not deterministic_replay_class_drop_mask(
        indices, probability=0.0, seed=7
    ).any()


def _arrays() -> dict[str, np.ndarray]:
    return {
        "record_indices": np.asarray([3, 8], dtype=np.int64),
        "class_labels": np.asarray([7, 9], dtype=np.int16),
        "altered_tokens": np.asarray([[1, 2], [3, 4]], dtype=np.uint16),
        "active_mask": np.asarray([[False, True], [True, True]], dtype=np.bool_),
        "effective_non_gt_mass": np.asarray([[0.0, 0.4], [0.2, 0.3]], dtype=np.float16),
        "top_indices": np.asarray([[[0], [2]], [[1], [3]]], dtype=np.uint16),
        "top_probabilities": np.ones((2, 2, 1), dtype=np.float16),
    }


def test_replay_bank_round_trip_and_random_access(tmp_path: Path) -> None:
    root = write_replay_bank(
        tmp_path / "bank",
        _arrays(),
        provenance={"snapshot_step": 100_000, "target": "suffix_likelihood_sqrt"},
    )
    bank = LaggedReplayBank(root)
    batch = bank.lookup(torch.tensor([8, 3]))

    torch.testing.assert_close(batch.class_labels, torch.tensor([9, 7]))
    assert not batch.class_drop_mask.any()
    torch.testing.assert_close(batch.altered_tokens, torch.tensor([[3, 4], [1, 2]]))
    assert bank.manifest["provenance"]["snapshot_step"] == 100_000
    with pytest.raises(KeyError):
        bank.lookup(torch.tensor([4]))


def test_replay_bank_rejects_duplicate_indices_and_manifest_tampering(
    tmp_path: Path,
) -> None:
    arrays = _arrays()
    arrays["record_indices"] = np.asarray([3, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="strictly increasing"):
        write_replay_bank(tmp_path / "duplicates", arrays, provenance={})

    root = write_replay_bank(tmp_path / "bank", _arrays(), provenance={})
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        LaggedReplayBank(root)


def test_sharded_replay_bank_preserves_unsorted_lookup_order(tmp_path: Path) -> None:
    root = tmp_path / "sharded"
    root.mkdir()
    first = _arrays()
    first["record_indices"] = np.asarray([0, 1], dtype=np.int64)
    second = _arrays()
    second["record_indices"] = np.asarray([2, 3], dtype=np.int64)
    write_replay_bank(root / "shard_0000", first, provenance={"rank": 0})
    write_replay_bank(root / "shard_0001", second, provenance={"rank": 1})
    write_sharded_replay_index(
        root,
        shards=[
            {
                "path": "shard_0000",
                "start_record_index": 0,
                "end_record_index_exclusive": 2,
            },
            {
                "path": "shard_0001",
                "start_record_index": 2,
                "end_record_index_exclusive": 4,
            },
        ],
        provenance={"snapshot_step": 10_000},
    )

    bank = open_replay_bank(root)
    assert isinstance(bank, ShardedLaggedReplayBank)
    batch = bank.lookup(torch.tensor([3, 0, 2, 1]))
    torch.testing.assert_close(batch.record_indices, torch.tensor([3, 0, 2, 1]))
    torch.testing.assert_close(batch.class_labels, torch.tensor([9, 7, 7, 9]))
    assert bank.manifest["provenance"]["snapshot_step"] == 10_000
    torch.testing.assert_close(
        bank.contains(torch.tensor([3, 7, 0])),
        torch.tensor([True, False, True]),
    )


def test_all_record_sharded_bank_rejects_a_coverage_gap(tmp_path: Path) -> None:
    root = tmp_path / "sharded_gap"
    root.mkdir()
    first = _arrays()
    first["record_indices"] = np.asarray([0, 1], dtype=np.int64)
    second = _arrays()
    second["record_indices"] = np.asarray([3, 4], dtype=np.int64)
    write_replay_bank(root / "shard_0000", first, provenance={"rank": 0})
    write_replay_bank(root / "shard_0001", second, provenance={"rank": 1})

    with pytest.raises(ValueError, match="exactly cover"):
        write_sharded_replay_index(
            root,
            shards=[
                {
                    "path": "shard_0000",
                    "start_record_index": 0,
                    "end_record_index_exclusive": 2,
                    "record_count": 2,
                },
                {
                    "path": "shard_0001",
                    "start_record_index": 3,
                    "end_record_index_exclusive": 5,
                    "record_count": 2,
                },
            ],
            provenance={
                "replay_stream_mode": "all_records",
                "dataset_record_count": 5,
            },
        )
