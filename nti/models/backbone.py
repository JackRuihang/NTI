from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nti.config import ICBConfig, ModelConfig


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale) + shift


@dataclass
class LayerKVCache:
    keys: Tensor
    values: Tensor
    length: int


class QKNormAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_dropout = nn.Dropout(config.dropout)
        self.attention_dropout = config.attention_dropout
        self.use_qk_norm = config.qk_norm
        # reAR caches CLS + class + at most N-1 consumed image tokens.
        self.cache_capacity = config.sequence_length + 1

    def _qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k, v

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._qkv(x)
        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        result = result.transpose(1, 2).reshape(x.shape)
        return self.proj_dropout(self.proj(result))

    def step(
        self,
        x: Tensor,
        cached_kv: Optional[LayerKVCache],
    ) -> Tuple[Tensor, LayerKVCache]:
        if x.shape[1] != 1:
            raise ValueError("incremental attention consumes exactly one position")
        q, k, v = self._qkv(x)
        if cached_kv is None:
            cache_shape = (*k.shape[:-2], self.cache_capacity, k.shape[-1])
            cached_kv = LayerKVCache(
                keys=torch.empty(cache_shape, dtype=k.dtype, device=k.device),
                values=torch.empty(cache_shape, dtype=v.dtype, device=v.device),
                length=0,
            )
        if cached_kv.length >= self.cache_capacity:
            raise ValueError("incremental attention cache is full")
        next_length = cached_kv.length + 1
        cached_kv.keys[..., cached_kv.length : next_length, :].copy_(k)
        cached_kv.values[..., cached_kv.length : next_length, :].copy_(v)
        cached_kv.length = next_length
        active_k = cached_kv.keys[..., :next_length, :]
        active_v = cached_kv.values[..., :next_length, :]
        result = F.scaled_dot_product_attention(
            q, active_k, active_v, dropout_p=0.0, is_causal=False
        )
        result = result.transpose(1, 2).reshape(x.shape)
        return self.proj_dropout(self.proj(result)), cached_kv

    def prefill(self, x: Tensor) -> Tuple[Tensor, LayerKVCache]:
        """Causally process the reAR CLS/class prefix and initialize its cache."""

        if x.shape[1] < 1 or x.shape[1] > self.cache_capacity:
            raise ValueError("invalid incremental attention prefix length")
        q, k, v = self._qkv(x)
        cache_shape = (*k.shape[:-2], self.cache_capacity, k.shape[-1])
        cached_kv = LayerKVCache(
            keys=torch.empty(cache_shape, dtype=k.dtype, device=k.device),
            values=torch.empty(cache_shape, dtype=v.dtype, device=v.device),
            length=x.shape[1],
        )
        cached_kv.keys[..., : x.shape[1], :].copy_(k)
        cached_kv.values[..., : x.shape[1], :].copy_(v)
        result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        result = result.transpose(1, 2).reshape(x.shape)
        return self.proj_dropout(self.proj(result)), cached_kv


class AdaLNBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = QKNormAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _adaln_parameters(self, condition: Tensor) -> Sequence[Tensor]:
        return self.modulation(condition).chunk(6, dim=-1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self._adaln_parameters(
            condition
        )
        x = x + gate_a * self.attention(_modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))
        return x

    def step(
        self,
        x: Tensor,
        condition: Tensor,
        cached_kv: Optional[LayerKVCache],
    ) -> Tuple[Tensor, LayerKVCache]:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self._adaln_parameters(
            condition
        )
        attended, new_kv = self.attention.step(
            _modulate(self.norm1(x), shift_a, scale_a), cached_kv
        )
        x = x + gate_a * attended
        x = x + gate_m * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))
        return x, new_kv

    def prefill(self, x: Tensor, condition: Tensor) -> Tuple[Tensor, LayerKVCache]:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self._adaln_parameters(
            condition
        )
        attended, new_kv = self.attention.prefill(
            _modulate(self.norm1(x), shift_a, scale_a)
        )
        x = x + gate_a * attended
        x = x + gate_m * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))
        return x, new_kv


class AdaLNFinal(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, 2 * config.hidden_size)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return _modulate(self.norm(x), shift, scale)


@dataclass
class ARCache:
    layers: List[LayerKVCache]
    condition: Tensor
    next_position: int
    interface_transport: Optional[Tensor] = None


class NTI(nn.Module):
    """Raster-order autoregressive backbone used by NTI."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        codebook: Optional[Tensor] = None,
        icb_config: Optional[ICBConfig] = None,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.null_class_id = config.num_classes
        self.class_embedding = nn.Embedding(config.num_classes + 1, config.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        position_capacity = config.sequence_length + 1024
        self.position_embedding = nn.Parameter(
            torch.zeros(1, position_capacity, config.hidden_size)
        )
        self.target_aware_position_embedding = nn.Parameter(
            torch.zeros(1, position_capacity, config.hidden_size)
        )
        self.timestep_embedding = nn.Parameter(
            torch.zeros(
                1,
                config.sequence_length + 100,
                config.hidden_size,
            )
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.target_aware_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.timestep_embedding, std=0.02)

        self.icb_config = icb_config or ICBConfig(enabled=False)
        if self.icb_config.enabled:
            raise ValueError("NTI requires a standard token embedding and head")
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.blocks = nn.ModuleList([AdaLNBlock(config) for _ in range(config.depth)])
        self.final = AdaLNFinal(config)
        self.apply(self._initialize)
        # Re-assert AdaLN-Zero after the generic linear initialization.
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final.modulation[-1].weight)
        nn.init.zeros_(self.final.modulation[-1].bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def _embed_tokens(
        self, token_ids: Tensor, transport: Optional[Tensor] = None
    ) -> Tensor:
        assert self.token_embedding is not None
        return self.token_embedding(token_ids)

    def _head(self, hidden: Tensor, transport: Optional[Tensor] = None) -> Tensor:
        assert self.lm_head is not None
        return self.lm_head(hidden)

    def _condition(
        self,
        class_labels: Tensor,
        *,
        apply_dropout: bool,
        class_drop_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if class_labels.ndim != 1:
            raise ValueError("class_labels must have shape [batch]")
        labels = class_labels.long()
        if class_drop_mask is not None:
            if class_drop_mask.shape != labels.shape:
                raise ValueError("class_drop_mask must have shape [batch]")
            dropped = class_drop_mask.detach().to(
                device=labels.device, dtype=torch.bool
            )
            labels = torch.where(
                dropped, torch.full_like(labels, self.null_class_id), labels
            )
        elif apply_dropout and self.training and self.config.class_dropout > 0:
            dropped = (
                torch.rand(labels.shape, device=labels.device)
                < self.config.class_dropout
            )
            labels = torch.where(
                dropped, torch.full_like(labels, self.null_class_id), labels
            )
        return self.class_embedding(labels).unsqueeze(1), labels

    def forward(
        self,
        image_tokens: Tensor,
        class_labels: Tensor,
        *,
        class_drop_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if image_tokens.ndim != 2:
            raise ValueError("image_tokens must have shape [batch, sequence]")
        batch, length = image_tokens.shape
        if length < 1 or length > self.config.sequence_length:
            raise ValueError("invalid image token sequence length")
        if class_labels.shape[0] != batch:
            raise ValueError("class_labels batch size does not match image_tokens")

        condition, _ = self._condition(
            class_labels,
            apply_dropout=class_drop_mask is None,
            class_drop_mask=class_drop_mask,
        )
        transport = None
        hidden = torch.cat(
            (
                self.cls_token.expand(batch, -1, -1),
                condition,
                self._embed_tokens(image_tokens, transport),
            ),
            dim=1,
        )
        total_length = length + 2
        target_positions = torch.cat(
            (
                torch.zeros_like(hidden[:, :1]),
                self.target_aware_position_embedding[:, 2 : 2 + length].expand(
                    batch, -1, -1
                ),
                torch.zeros_like(hidden[:, -1:]),
            ),
            dim=1,
        )
        hidden = hidden + self.position_embedding[:, :total_length] + target_positions
        condition_states = condition + self.timestep_embedding[:, :total_length]
        for block in self.blocks:
            hidden = block(hidden, condition_states)

        # This is exactly reAR's shifted supervision: class predicts z0 and
        # image state z_t predicts z_{t+1}; the final image state is discarded.
        hidden = hidden[:, 1:-1]
        condition_states = condition_states[:, 1:-1]
        return self._head(self.final(hidden, condition_states), transport)

    @torch.no_grad()
    def prefill(self, class_labels: Tensor) -> Tuple[Tensor, ARCache]:
        condition, _ = self._condition(class_labels, apply_dropout=False)
        transport = None
        batch = class_labels.shape[0]
        hidden = torch.cat((self.cls_token.expand(batch, -1, -1), condition), dim=1)
        target_positions = torch.cat(
            (
                torch.zeros_like(hidden[:, :1]),
                self.target_aware_position_embedding[:, 2:3].expand(batch, -1, -1),
            ),
            dim=1,
        )
        hidden = hidden + self.position_embedding[:, :2] + target_positions
        condition_states = condition + self.timestep_embedding[:, :2]
        layers: List[LayerKVCache] = []
        for block in self.blocks:
            hidden, cached_kv = block.prefill(hidden, condition_states)
            layers.append(cached_kv)
        logits = self._head(
            self.final(hidden[:, 1:2], condition_states[:, 1:2]), transport
        )[:, 0]
        return logits, ARCache(
            layers=layers,
            condition=condition,
            next_position=2,
            interface_transport=transport,
        )

    @torch.no_grad()
    def decode_step(self, last_token: Tensor, cache: ARCache) -> Tuple[Tensor, ARCache]:
        if last_token.ndim != 1:
            raise ValueError("last_token must have shape [batch]")
        if cache.next_position >= self.config.sequence_length + 1:
            raise ValueError("KV cache already reached the configured sequence length")
        hidden = self._embed_tokens(last_token, cache.interface_transport).unsqueeze(1)
        position = cache.next_position
        hidden = (
            hidden
            + self.position_embedding[:, position : position + 1]
            + self.target_aware_position_embedding[:, position + 1 : position + 2]
        )
        condition = (
            cache.condition + self.timestep_embedding[:, position : position + 1]
        )
        new_layers: List[LayerKVCache] = []
        for block, cached_kv in zip(self.blocks, cache.layers):
            hidden, new_kv = block.step(hidden, condition, cached_kv)
            new_layers.append(new_kv)
        logits = self._head(self.final(hidden, condition), cache.interface_transport)[
            :, 0
        ]
        return logits, ARCache(
            layers=new_layers,
            condition=cache.condition,
            next_position=cache.next_position + 1,
            interface_transport=cache.interface_transport,
        )


def build_nti(
    config: ModelConfig,
    *,
    codebook: Optional[Tensor] = None,
    icb_config: Optional[ICBConfig] = None,
) -> NTI:
    return NTI(config, codebook=codebook, icb_config=icb_config)
