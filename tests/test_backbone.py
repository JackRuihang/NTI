import torch
from nti.config import tiny_config
from nti.models.backbone import NTI


def _make_model(icb=False):
    config = tiny_config(icb=icb)
    codebook = None
    if icb:
        codebook = torch.randn(config.model.vocab_size, config.model.codebook_dim)
    model = NTI(config.model, codebook=codebook, icb_config=config.icb)
    return (config, model.eval())


def test_forward_alignment_and_causality():
    config, model = _make_model()
    tokens = torch.randint(0, config.model.vocab_size, (2, 8))
    labels = torch.tensor([1, 3])
    changed = tokens.clone()
    changed[:, 3] = (changed[:, 3] + 1) % config.model.vocab_size
    with torch.no_grad():
        logits = model(tokens, labels)
        changed_logits = model(changed, labels)
    assert logits.shape == (2, 8, config.model.vocab_size)
    assert torch.allclose(logits[:, :4], changed_logits[:, :4], atol=1e-06)


def test_incremental_cache_matches_full_forward():
    config, model = _make_model()
    tokens = torch.randint(0, config.model.vocab_size, (2, 7))
    labels = torch.tensor([0, 4])
    with torch.no_grad():
        full = model(tokens, labels)
        first, cache = model.prefill(labels)
        incremental = [first]
        for position in range(tokens.shape[1] - 1):
            next_logits, cache = model.decode_step(tokens[:, position], cache)
            incremental.append(next_logits)
        incremental = torch.stack(incremental, dim=1)
    assert torch.allclose(full, incremental, atol=2e-05, rtol=2e-05)


def test_incremental_cache_is_preallocated_and_reused():
    config, model = _make_model()
    labels = torch.tensor([0, 4])
    tokens = torch.randint(0, config.model.vocab_size, (2, 3))
    with torch.no_grad():
        _, cache = model.prefill(labels)
        key_pointer = cache.layers[0].keys.data_ptr()
        assert cache.layers[0].keys.shape[-2] == config.model.sequence_length + 1
        assert cache.layers[0].length == 2
        for position in range(tokens.shape[1]):
            _, cache = model.decode_step(tokens[:, position], cache)
    assert cache.layers[0].keys.data_ptr() == key_pointer
    assert cache.layers[0].length == 2 + tokens.shape[1]


def test_backbone_uses_rear_conditioning_structure():
    config, model = _make_model()
    assert model.cls_token.shape == (1, 1, config.model.hidden_size)
    assert model.position_embedding.shape[1] == config.model.sequence_length + 1024
    assert model.target_aware_position_embedding.shape == model.position_embedding.shape
    assert model.timestep_embedding.shape[1] == config.model.sequence_length + 100
    assert model.blocks[0].norm1.elementwise_affine
    assert model.blocks[0].norm2.elementwise_affine
    assert model.blocks[0].mlp[1].approximate == "none"


def test_explicit_class_drop_mask_is_reused_deterministically():
    config, model = _make_model()
    model.train()
    tokens = torch.randint(0, config.model.vocab_size, (2, 6))
    labels = torch.tensor([1, 2])
    drop_mask = torch.tensor([True, False])
    first = model(tokens, labels, class_drop_mask=drop_mask)
    second = model(tokens, labels, class_drop_mask=drop_mask)
    assert torch.equal(first, second)
