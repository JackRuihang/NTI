"""Small CPU tests of the released NTI context/target/student pipeline."""

import torch
import pytest
from nti.config import tiny_config, experiment_config_from_dict
from nti.models import NTI
from nti.training import build_codebook_cosine_kernel, TwoPassConfig, two_pass_forward


def test_context_target_backward_and_optimizer():
    torch.manual_seed(7)
    c = tiny_config()
    model = NTI(c.model)
    tokens = torch.randint(c.model.vocab_size, (4, 10))
    labels = torch.tensor([0, 1, 2, 3])
    kernel = build_codebook_cosine_kernel(
        torch.randn(c.model.vocab_size, c.model.codebook_dim), temperature=0.05
    )
    out = two_pass_forward(
        model,
        tokens,
        labels,
        config=TwoPassConfig(),
        codebook_kernel=kernel,
        replacement_ratio=0.3,
    )
    assert torch.isfinite(out.loss)
    assert not out.branch_target.requires_grad
    torch.testing.assert_close(out.branch_target.sum(-1), torch.ones(4, 10))
    old = model.lm_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out.loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
    assert not torch.equal(old, model.lm_head.weight)


def test_no_unsupported_auxiliary_interface():
    with pytest.raises(ValueError, match="standard free token interface"):
        experiment_config_from_dict({"token_interface": {"mode": "unsupported"}})


def test_no_historical_target_modes():
    with pytest.raises(ValueError, match="tv_history_conditioned"):
        TwoPassConfig(target_mode="unsupported")
