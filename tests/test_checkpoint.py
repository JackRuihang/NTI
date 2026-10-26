import random

import numpy as np
import pytest
import torch

from nti.checkpoint import (
    capture_rng_state,
    checkpoint_reached_max_steps,
    load_checkpoint,
    migrate_training_state_geometry,
    prune_training_checkpoints,
    save_checkpoint,
    validate_checkpoint_config,
)


DATASET_RECORDS = 12_811_680
CHECKPOINT_GEOMETRY = {
    "world_size": 6,
    "per_device_batch_size": 128,
    "accumulation_steps": 3,
}
MIGRATED_GEOMETRY = {
    "world_size": 4,
    "per_device_batch_size": 96,
    "accumulation_steps": 6,
}
EXPANDING_SOURCE_GEOMETRY = {
    "world_size": 4,
    "per_device_batch_size": 128,
    "accumulation_steps": 4,
}
EXPANDING_TARGET_GEOMETRY = {
    "world_size": 8,
    "per_device_batch_size": 128,
    "accumulation_steps": 2,
}


def _write_geometry_checkpoint(path, *, config=None, batch_in_epoch=9_840):
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(1.25)
        model.bias.fill_(-0.75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=70_000,
        epoch=12,
        batch_in_epoch=batch_in_epoch,
        config=config or {},
        rng_states=[capture_rng_state() for _ in range(6)],
        training_geometry=CHECKPOINT_GEOMETRY,
    )
    return {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }


def _write_expanding_geometry_checkpoint(path, *, batch_in_epoch=19_680):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    rng_states = []
    for rank in range(EXPANDING_SOURCE_GEOMETRY["world_size"]):
        rank_seed = 100 + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(rank_seed)
        rng_states.append(capture_rng_state())
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=105_000,
        epoch=16,
        batch_in_epoch=batch_in_epoch,
        config={},
        rng_states=rng_states,
        training_geometry=EXPANDING_SOURCE_GEOMETRY,
    )


def _fill_and_snapshot(model, value):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
    return {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }


def _assert_model_state(model, expected):
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expected[name])


def test_checkpoint_completion_boundary():
    assert not checkpoint_reached_max_steps({"step": 3}, max_steps=4)
    assert checkpoint_reached_max_steps({"step": 4}, max_steps=4)
    assert checkpoint_reached_max_steps({"step": 5}, max_steps=4)


def test_checkpoint_retention_keeps_milestones_and_recent_files(tmp_path):
    steps = (5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000)
    for step in steps:
        (tmp_path / f"step_{step:07d}.pt").write_text(str(step))
    untouched = (
        tmp_path / "step_0030000.eval.pt",
        tmp_path / "step_0035000.pt.tmp",
        tmp_path / "notes.txt",
    )
    for path in untouched:
        path.write_text("keep")

    removed = prune_training_checkpoints(tmp_path, milestone_every=25_000, keep_last=3)

    assert {path.name for path in removed} == {
        "step_0005000.pt",
        "step_0010000.pt",
        "step_0015000.pt",
        "step_0020000.pt",
    }
    retained_training = {
        path.name
        for path in tmp_path.iterdir()
        if path.name in {f"step_{step:07d}.pt" for step in steps}
    }
    assert retained_training == {
        "step_0025000.pt",
        "step_0030000.pt",
        "step_0035000.pt",
    }
    assert all(path.is_file() for path in untouched)


def test_checkpoint_retention_is_disabled_by_default(tmp_path):
    checkpoint = tmp_path / "step_0005000.pt"
    checkpoint.write_text("keep")
    assert prune_training_checkpoints(tmp_path) == ()
    assert checkpoint.is_file()


def test_checkpoint_restores_training_and_rng(tmp_path):
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "step.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=4,
        epoch=2,
        batch_in_epoch=3,
        config={"name": "test"},
        training_geometry={"world_size": 1},
    )
    expected = (torch.rand(1), random.random(), np.random.rand())
    with torch.no_grad():
        model.weight.zero_()
    state = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_training_geometry={"world_size": 1},
    )
    actual = (torch.rand(1), random.random(), np.random.rand())
    assert state == {"step": 4, "epoch": 2, "batch_in_epoch": 3}
    assert torch.equal(expected[0], actual[0])
    assert expected[1:] == actual[1:]


def test_checkpoint_can_load_weights_without_restoring_training_rng(tmp_path):
    torch.manual_seed(11)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "weights.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        epoch=0,
        batch_in_epoch=1,
        config={},
    )
    torch.manual_seed(29)
    expected = torch.rand(1)
    torch.manual_seed(29)
    load_checkpoint(path, model=model, restore_rng=False)
    actual = torch.rand(1)
    assert torch.equal(expected, actual)


def test_checkpoint_can_require_complete_world_rng_states(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "complete_rng.pt"
    states = []
    for _ in range(2):
        state = capture_rng_state()
        state.setdefault("cuda", state["torch"].clone())
        states.append(state)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        epoch=0,
        batch_in_epoch=1,
        config={},
        rng_states=states,
        training_geometry={
            "world_size": 2,
            "per_device_batch_size": 1,
            "accumulation_steps": 1,
        },
    )

    loaded = load_checkpoint(
        path,
        model=model,
        rank=0,
        expected_training_geometry={
            "world_size": 2,
            "per_device_batch_size": 1,
            "accumulation_steps": 1,
        },
        require_complete_rng_states=True,
    )
    assert loaded["rng_state_count"] == 2

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rng_states"] = payload["rng_states"][:1]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="RNG state count"):
        load_checkpoint(
            path,
            model=model,
            rank=0,
            expected_training_geometry={
                "world_size": 2,
                "per_device_batch_size": 1,
                "accumulation_steps": 1,
            },
            require_complete_rng_states=True,
        )


def test_checkpoint_rejects_inference_config_drift_before_loading(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "weights.pt"
    saved_config = {
        "model": {"hidden_size": 2, "layer_norm_eps": 1.0e-6},
        "icb": {"enabled": True, "geometry_temperature": 16.0},
    }
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        epoch=0,
        batch_in_epoch=1,
        config=saved_config,
    )
    changed_config = {
        **saved_config,
        "icb": {"enabled": True, "geometry_temperature": 8.0},
    }
    with pytest.raises(RuntimeError, match="icb.geometry_temperature"):
        load_checkpoint(
            path,
            model=model,
            expected_config=changed_config,
            required_config_sections=("model", "icb"),
            restore_rng=False,
        )


def test_checkpoint_canonicalizes_legacy_token_interface(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "legacy_interface.pt"
    saved_config = {
        "model": {"hidden_size": 2},
        "icb": {"enabled": True},
    }
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        epoch=0,
        batch_in_epoch=1,
        config=saved_config,
    )
    canonical_config = {
        **saved_config,
        "token_interface": {
            "mode": "icb",
            "read_weight": 0.0,
            "write_weight": 0.0,
        },
    }
    load_checkpoint(
        path,
        model=model,
        expected_config=canonical_config,
        required_config_sections=("model", "icb", "token_interface"),
        restore_rng=False,
    )

    unsupported_config = {
        **canonical_config,
        "icb": {"enabled": False},
        "token_interface": {
            "mode": "unsupported",
            "read_weight": 0.01,
            "write_weight": 0.01,
        },
    }
    with pytest.raises(RuntimeError, match="token_interface.mode"):
        load_checkpoint(
            path,
            model=model,
            expected_config=unsupported_config,
            required_config_sections=("model", "icb", "token_interface"),
            restore_rng=False,
        )


def test_checkpoint_rejects_malformed_explicit_token_interface(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "malformed_interface.pt"
    saved_config = {
        "model": {"hidden_size": 2},
        "icb": {"enabled": False},
        "token_interface": "free",
    }
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        epoch=0,
        batch_in_epoch=1,
        config=saved_config,
    )
    expected_config = {
        **saved_config,
        "token_interface": {
            "mode": "free",
            "read_weight": 0.0,
            "write_weight": 0.0,
        },
    }

    with pytest.raises(RuntimeError, match="token_interface"):
        load_checkpoint(
            path,
            model=model,
            expected_config=expected_config,
            required_config_sections=("model", "icb", "token_interface"),
            restore_rng=False,
        )


def test_checkpoint_canonicalizes_legacy_replacement_schedule():
    legacy = {
        "training": {
            "replacement_max_ratio": 0.2,
            "replacement_warmup_fraction": 0.3,
            "max_steps": 250_000,
        }
    }
    hold = {
        "training": {
            **legacy["training"],
            "replacement_schedule": "linear_hold",
        }
    }
    validate_checkpoint_config(legacy, hold, sections=("training",))


def test_checkpoint_canonicalizes_legacy_full_branch_fraction():
    legacy = {"training": {"branch_loss_weight": 0.5}}
    explicit_full = {
        "training": {
            "branch_loss_weight": 0.5,
            "branch_image_fraction": 1.0,
        }
    }
    validate_checkpoint_config(legacy, explicit_full, sections=("training",))

    partial = {
        "training": {
            "branch_loss_weight": 0.5,
            "branch_image_fraction": 0.25,
        }
    }
    with pytest.raises(RuntimeError, match="branch_image_fraction"):
        validate_checkpoint_config(legacy, partial, sections=("training",))


def test_checkpoint_canonicalizes_legacy_auxiliary_loss_gate():
    legacy = {"training": {"branch_loss_weight": 0.5}}
    explicit_constant = {
        "training": {
            "branch_loss_weight": 0.5,
            "auxiliary_loss_gate": "constant",
        }
    }
    validate_checkpoint_config(legacy, explicit_constant, sections=("training",))

    normalized = {
        "training": {
            "branch_loss_weight": 0.5,
            "auxiliary_loss_gate": "normalized_replacement",
        }
    }
    with pytest.raises(RuntimeError, match="auxiliary_loss_gate"):
        validate_checkpoint_config(legacy, normalized, sections=("training",))
    validate_checkpoint_config(normalized, normalized, sections=("training",))


def test_checkpoint_canonicalizes_monitoring_preview_class_ids_sequence_type():
    checkpoint_config = {
        "monitoring": {
            "preview_every": 5_000,
            "preview_class_ids": (1, 7, 282, 604),
        }
    }
    yaml_config = {
        "monitoring": {
            "preview_every": 5_000,
            "preview_class_ids": [1, 7, 282, 604],
        }
    }

    validate_checkpoint_config(
        checkpoint_config,
        yaml_config,
        sections=("monitoring",),
    )


def test_schedule_migration_ignores_only_schedule_mode():
    hold = {
        "training": {
            "replacement_max_ratio": 0.2,
            "replacement_warmup_fraction": 0.3,
            "replacement_schedule": "linear_hold",
            "max_steps": 250_000,
            "branch_loss_weight": 0.5,
        }
    }
    anneal = {
        "training": {
            **hold["training"],
            "replacement_schedule": "ramp_cosine_zero",
        }
    }
    with pytest.raises(RuntimeError, match="replacement_schedule"):
        validate_checkpoint_config(hold, anneal, sections=("training",))
    validate_checkpoint_config(
        hold,
        anneal,
        sections=("training",),
        allowed_migrations={
            "training.replacement_schedule": (
                "linear_hold",
                "ramp_cosine_zero",
            )
        },
    )

    with pytest.raises(RuntimeError, match="required_migration.*replacement_schedule"):
        validate_checkpoint_config(
            anneal,
            anneal,
            sections=("training",),
            allowed_migrations={
                "training.replacement_schedule": (
                    "linear_hold",
                    "ramp_cosine_zero",
                )
            },
            require_migrations=True,
        )

    with pytest.raises(RuntimeError, match="replacement_schedule"):
        validate_checkpoint_config(
            anneal,
            hold,
            sections=("training",),
            allowed_migrations={
                "training.replacement_schedule": (
                    "linear_hold",
                    "ramp_cosine_zero",
                )
            },
        )

    changed_ratio = {
        "training": {
            **anneal["training"],
            "replacement_max_ratio": 0.3,
        }
    }
    with pytest.raises(RuntimeError, match="replacement_max_ratio"):
        validate_checkpoint_config(
            hold,
            changed_ratio,
            sections=("training",),
            allowed_migrations={
                "training.replacement_schedule": (
                    "linear_hold",
                    "ramp_cosine_zero",
                )
            },
        )


def test_checkpoint_rejects_geometry_drift_by_default_before_loading_model(tmp_path):
    path = tmp_path / "geometry.pt"
    _write_geometry_checkpoint(path)
    target = torch.nn.Linear(2, 2)
    before = _fill_and_snapshot(target, 0.125)

    with pytest.raises(RuntimeError, match="resume training geometry changed"):
        load_checkpoint(
            path,
            model=target,
            expected_training_geometry=MIGRATED_GEOMETRY,
            restore_rng=False,
        )

    _assert_model_state(target, before)


def test_checkpoint_migrates_six_gpu_cursor_to_four_gpu_geometry(tmp_path):
    path = tmp_path / "geometry.pt"
    source_model_state = _write_geometry_checkpoint(path)
    target = torch.nn.Linear(2, 2)
    _fill_and_snapshot(target, 0.0)

    state = load_checkpoint(
        path,
        model=target,
        expected_training_geometry=MIGRATED_GEOMETRY,
        allow_training_geometry_migration=True,
        migration_dataset_records=DATASET_RECORDS,
    )

    assert state["step"] == 70_000
    assert state["epoch"] == 12
    assert state["batch_in_epoch"] == 19_680
    assert 9_840 * 6 * 128 == state["batch_in_epoch"] * 4 * 96
    assert state["batch_in_epoch"] % MIGRATED_GEOMETRY["accumulation_steps"] == 0
    _assert_model_state(target, source_model_state)


def test_checkpoint_expands_four_gpu_rng_to_eight_with_seeded_new_rank(tmp_path):
    path = tmp_path / "expanding_geometry.pt"
    _write_expanding_geometry_checkpoint(path)
    target = torch.nn.Linear(2, 2)

    new_rank_seed = 42 + 105_000 + 4
    random.seed(new_rank_seed)
    np.random.seed(new_rank_seed)
    torch.manual_seed(new_rank_seed)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    state = load_checkpoint(
        path,
        model=target,
        rank=4,
        expected_training_geometry=EXPANDING_TARGET_GEOMETRY,
        allow_training_geometry_migration=True,
        migration_dataset_records=DATASET_RECORDS,
        geometry_migration_rng_seed=42,
    )
    actual = (random.random(), np.random.rand(), torch.rand(1))

    assert state["step"] == 105_000
    assert state["epoch"] == 16
    assert state["batch_in_epoch"] == 9_840
    assert state["geometry_migration"]["rng_policy"] == (
        "restore_existing_ranks_seed_added_ranks"
    )
    assert state["geometry_migration"]["new_rank_seed_base"] == 105_042
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_checkpoint_expansion_preserves_existing_rank_rng(tmp_path):
    path = tmp_path / "expanding_geometry.pt"
    _write_expanding_geometry_checkpoint(path)
    target = torch.nn.Linear(2, 2)

    random.seed(102)
    np.random.seed(102)
    torch.manual_seed(102)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    load_checkpoint(
        path,
        model=target,
        rank=2,
        expected_training_geometry=EXPANDING_TARGET_GEOMETRY,
        allow_training_geometry_migration=True,
        migration_dataset_records=DATASET_RECORDS,
        geometry_migration_rng_seed=42,
    )
    actual = (random.random(), np.random.rand(), torch.rand(1))

    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_checkpoint_expansion_new_rank_rng_is_repeatable_and_distinct(tmp_path):
    path = tmp_path / "expanding_geometry.pt"
    _write_expanding_geometry_checkpoint(path)

    def load_and_draw(rank):
        target = torch.nn.Linear(2, 2)
        load_checkpoint(
            path,
            model=target,
            rank=rank,
            expected_training_geometry=EXPANDING_TARGET_GEOMETRY,
            allow_training_geometry_migration=True,
            migration_dataset_records=DATASET_RECORDS,
            geometry_migration_rng_seed=42,
        )
        return (random.random(), np.random.rand(), torch.rand(1))

    rank4_first = load_and_draw(4)
    rank4_second = load_and_draw(4)
    rank5 = load_and_draw(5)

    assert rank4_first[0] == rank4_second[0]
    assert rank4_first[1] == rank4_second[1]
    assert torch.equal(rank4_first[2], rank4_second[2])
    assert rank4_first[0] != rank5[0]
    assert rank4_first[1] != rank5[1]
    assert not torch.equal(rank4_first[2], rank5[2])


def test_checkpoint_expansion_requires_new_rank_seed_before_loading_model(tmp_path):
    path = tmp_path / "expanding_geometry.pt"
    _write_expanding_geometry_checkpoint(path)
    target = torch.nn.Linear(2, 2)
    before = _fill_and_snapshot(target, 0.625)

    with pytest.raises(RuntimeError, match="geometry_migration_rng_seed"):
        load_checkpoint(
            path,
            model=target,
            rank=4,
            expected_training_geometry=EXPANDING_TARGET_GEOMETRY,
            allow_training_geometry_migration=True,
            migration_dataset_records=DATASET_RECORDS,
        )

    _assert_model_state(target, before)


def test_geometry_migration_rejects_changed_effective_global_batch():
    incompatible_geometry = {
        "world_size": 4,
        "per_device_batch_size": 128,
        "accumulation_steps": 4,
    }

    with pytest.raises(RuntimeError, match="effective global batch"):
        migrate_training_state_geometry(
            {"step": 70_000, "epoch": 12, "batch_in_epoch": 9_840},
            checkpoint_geometry=CHECKPOINT_GEOMETRY,
            current_geometry=incompatible_geometry,
            dataset_records=DATASET_RECORDS,
        )


def test_geometry_migration_rejects_non_optimizer_boundary():
    with pytest.raises(RuntimeError, match="optimizer-step boundary"):
        migrate_training_state_geometry(
            {"step": 70_000, "epoch": 12, "batch_in_epoch": 9_841},
            checkpoint_geometry=CHECKPOINT_GEOMETRY,
            current_geometry=MIGRATED_GEOMETRY,
            dataset_records=DATASET_RECORDS,
        )


def test_geometry_migration_normalizes_epoch_end():
    state = migrate_training_state_geometry(
        {"step": 72_280, "epoch": 12, "batch_in_epoch": 16_680},
        checkpoint_geometry=CHECKPOINT_GEOMETRY,
        current_geometry=MIGRATED_GEOMETRY,
        dataset_records=DATASET_RECORDS,
    )

    assert state["step"] == 72_280
    assert state["epoch"] == 13
    assert state["batch_in_epoch"] == 0


def test_checkpoint_config_drift_can_ignore_geometry_and_run_keys(tmp_path):
    path = tmp_path / "config.pt"
    saved_config = {
        "training": {
            "global_batch_size": 2_304,
            "per_device_batch_size": 128,
            "learning_rate": 4.0e-4,
        },
        "run": {"name": "six_gpu_run"},
    }
    expected_config = {
        "training": {
            "global_batch_size": 2_304,
            "per_device_batch_size": 96,
            "learning_rate": 4.0e-4,
        },
        "run": {"name": "four_gpu_run"},
    }
    _write_geometry_checkpoint(path, config=saved_config)
    target = torch.nn.Linear(2, 2)

    state = load_checkpoint(
        path,
        model=target,
        expected_training_geometry=MIGRATED_GEOMETRY,
        allow_training_geometry_migration=True,
        migration_dataset_records=DATASET_RECORDS,
        expected_config=expected_config,
        required_config_sections=("training", "run"),
        ignored_config_keys=("training.per_device_batch_size", "run.name"),
    )

    assert state["batch_in_epoch"] == 19_680


def test_checkpoint_config_drift_rejects_nonignored_key_before_loading(tmp_path):
    path = tmp_path / "config.pt"
    saved_config = {
        "training": {
            "global_batch_size": 2_304,
            "per_device_batch_size": 128,
            "learning_rate": 4.0e-4,
        },
        "run": {"name": "six_gpu_run"},
    }
    expected_config = {
        "training": {
            "global_batch_size": 2_304,
            "per_device_batch_size": 96,
            "learning_rate": 8.0e-4,
        },
        "run": {"name": "four_gpu_run"},
    }
    _write_geometry_checkpoint(path, config=saved_config)
    target = torch.nn.Linear(2, 2)
    before = _fill_and_snapshot(target, 0.25)

    with pytest.raises(RuntimeError, match="training.learning_rate"):
        load_checkpoint(
            path,
            model=target,
            expected_training_geometry=MIGRATED_GEOMETRY,
            allow_training_geometry_migration=True,
            migration_dataset_records=DATASET_RECORDS,
            expected_config=expected_config,
            required_config_sections=("training", "run"),
            ignored_config_keys=("training.per_device_batch_size", "run.name"),
            restore_rng=False,
        )

    _assert_model_state(target, before)


def test_geometry_migration_rejects_invalid_rng_before_loading_model(tmp_path):
    path = tmp_path / "invalid_rng.pt"
    _write_geometry_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rng_states"][2] = {"python": random.getstate()}
    torch.save(payload, path)
    target = torch.nn.Linear(2, 2)
    before = _fill_and_snapshot(target, 0.375)

    with pytest.raises(RuntimeError, match="rank 2 RNG state is missing"):
        load_checkpoint(
            path,
            model=target,
            expected_training_geometry=MIGRATED_GEOMETRY,
            allow_training_geometry_migration=True,
            migration_dataset_records=DATASET_RECORDS,
        )

    _assert_model_state(target, before)
