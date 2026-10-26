import torch
from nti.config import load_experiment_config
from nti.training.targets import tv_history_conditioned_target
from pathlib import Path


def test_release_config_contract():
    root = Path(__file__).resolve().parents[1]
    for size in ("s", "b", "l"):
        c = load_experiment_config(root / "configs" / f"nti_{size}.yaml")
        t = c.training
        assert t.max_steps == 250000
        assert t.global_batch_size == 2048
        assert t.target_mode == "tv_history_conditioned"
        assert t.replacement_max_ratio == 0.3
        assert t.replacement_schedule == "bell_cosine"
        assert t.branch_loss_weight == 0.5
        assert t.clean_loss_weight == 1.0
        assert t.replay_refresh_steps == 10000
        assert t.replay_start_step == 10000 and t.replay_end_step == 200000
        assert t.codebook_kernel_temperature == 0.05


def test_tv_normalized_detached_and_finite_gradient():
    torch.manual_seed(42)
    clean = torch.randn(2, 5, 8, requires_grad=True)
    branch = torch.randn(2, 5, 8, requires_grad=True)
    y = torch.randint(8, (2, 5))
    kernel = torch.softmax(torch.eye(8) * 2, dim=-1)
    result = tv_history_conditioned_target(clean, branch, y, kernel)
    q = result.target
    assert not q.requires_grad
    assert torch.isfinite(q).all() and (q >= 0).all()
    torch.testing.assert_close(q.sum(-1), torch.ones(2, 5))
    loss = -(q * branch.log_softmax(-1)).sum(-1).mean()
    loss.backward()
    assert torch.isfinite(branch.grad).all()
    assert clean.grad is None
