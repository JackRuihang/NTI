"""Target distributions and token-level losses for NTI training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TVHistoryTargetResult:
    """Detached TV-gated target and its directly interpretable statistics."""

    target: torch.Tensor
    clean_confidence: torch.Tensor
    tv_distance: torch.Tensor
    overlap: torch.Tensor
    clean_gt_probability: torch.Tensor
    non_gt_dose: torch.Tensor


@dataclass(frozen=True)
class _TargetComponents:
    """Detached distributions used to construct NTI targets."""

    clean: torch.Tensor
    branch: torch.Tensor
    hard: torch.Tensor
    target_affinity: torch.Tensor
    clean_gt_probability: torch.Tensor


def build_codebook_cosine_kernel(
    codebook_embeddings: torch.Tensor,
    *,
    temperature: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build the frozen row-stochastic cosine kernel used by soft-branch targets.

    The returned tensor is always detached and computed in float32.  Keeping the
    kernel outside autograd is intentional: the tokenizer codebook is geometry
    supplied to the AR objective, not a parameter updated through that objective.
    """

    if codebook_embeddings.ndim != 2:
        raise ValueError(
            "codebook_embeddings must have shape [vocabulary, embedding_dim]"
        )
    if not codebook_embeddings.is_floating_point():
        raise TypeError("codebook_embeddings must be floating point")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    with torch.no_grad():
        codebook = codebook_embeddings.detach().float()
        norms = torch.linalg.vector_norm(codebook, dim=-1)
        if torch.any(norms <= eps):
            raise ValueError("codebook_embeddings contains a zero-norm vector")
        normalized = F.normalize(codebook, p=2.0, dim=-1, eps=eps)
        cosine = normalized @ normalized.transpose(0, 1)
        kernel = F.softmax(cosine / float(temperature), dim=-1)
    return kernel.detach()


def build_codebook_cosine_affinity(
    codebook_embeddings: torch.Tensor,
    *,
    temperature: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Build the unnormalized GT-centred compatibility affinity.

    For normalized codebook vectors, this returns

    .. math::

        A_{y,k} = \exp((\cos(C_y, C_k)-1)/\tau).

    Every diagonal entry is one (up to floating-point roundoff), while all
    entries lie in ``(0, 1]``.  Unlike the row-stochastic kernel, this absolute
    scale is part of the compatibility gate and must not be normalized away.
    """

    if codebook_embeddings.ndim != 2:
        raise ValueError(
            "codebook_embeddings must have shape [vocabulary, embedding_dim]"
        )
    if not codebook_embeddings.is_floating_point():
        raise TypeError("codebook_embeddings must be floating point")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    with torch.no_grad():
        codebook = codebook_embeddings.detach().float()
        norms = torch.linalg.vector_norm(codebook, dim=-1)
        if torch.any(norms <= eps):
            raise ValueError("codebook_embeddings contains a zero-norm vector")
        normalized = F.normalize(codebook, p=2.0, dim=-1, eps=eps)
        cosine = (normalized @ normalized.transpose(0, 1)).clamp(-1.0, 1.0)
        affinity = torch.exp((cosine - 1.0) / float(temperature))
        affinity.diagonal().fill_(1.0)
    return affinity.detach()


def tv_history_conditioned_target(
    clean_logits: torch.Tensor,
    branch_logits: torch.Tensor,
    target_ids: torch.Tensor,
    codebook_kernel: torch.Tensor,
    *,
    clean_temperature: float = 1.0,
    eps: float = 1e-8,
) -> TVHistoryTargetResult:
    r"""Build the alpha-free TV history-conditioned target used by NTI.

    Let ``p_c`` and ``p_b`` be the clean- and altered-history predictions and
    let ``A_y`` be the GT-centred codebook affinity reconstructed from the
    row-normalized kernel.  The distribution overlap and clean confidence are

    .. math::

        g=1-\tfrac12\lVert p_c-p_b\rVert_1,
        \qquad c=1-H(p_c)/\log V.

    Altered history changes both the non-GT dose and its internal direction:

    .. math::

        r_H(k) \propto \mathbf 1[k\ne y]A_{y,k}
        ((1-g)p_c(k)+g p_b(k\mid H)),

        d=c(1-p_c(y))g, \qquad q=(1-d)e_y+d r_H.

    Thus disjoint predictions return exactly to hard GT, while matching but
    uninformative uniform predictions also return to GT through ``c=0``.  No
    alpha, threshold, learned gate, Top-K, or additional temperature is used.
    All target construction is detached from autograd.
    """

    components = _target_components(
        clean_logits,
        branch_logits,
        target_ids,
        codebook_kernel,
        clean_temperature=clean_temperature,
        eps=eps,
    )
    with torch.no_grad():
        vocabulary = components.clean.shape[-1]
        tiny = torch.finfo(torch.float32).tiny
        log_clean = components.clean.clamp_min(tiny).log()
        clean_entropy = -(components.clean * log_clean).sum(dim=-1)
        clean_confidence = (1.0 - clean_entropy / math.log(vocabulary)).clamp(0.0, 1.0)

        tv_distance = (
            0.5 * (components.clean - components.branch).abs().sum(dim=-1)
        ).clamp(0.0, 1.0)
        overlap = (1.0 - tv_distance).clamp(0.0, 1.0)

        hard = components.hard
        non_gt = 1.0 - hard
        history_probability = (1.0 - overlap[..., None]) * components.clean + overlap[
            ..., None
        ] * components.branch
        candidate_score = components.target_affinity * non_gt * history_probability
        candidate_mass = candidate_score.sum(dim=-1)
        has_candidate = candidate_mass > 0.0
        candidate = torch.where(
            has_candidate[..., None],
            candidate_score / candidate_mass[..., None].clamp_min(float(eps)),
            hard,
        )

        non_gt_dose = (
            clean_confidence * (1.0 - components.clean_gt_probability) * overlap
        ).clamp(0.0, 1.0)
        non_gt_dose = torch.where(
            has_candidate, non_gt_dose, torch.zeros_like(non_gt_dose)
        )
        target = (1.0 - non_gt_dose[..., None]) * hard + non_gt_dose[
            ..., None
        ] * candidate
        target = target.clamp_min(0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(float(eps))

    return TVHistoryTargetResult(
        target=target.detach(),
        clean_confidence=clean_confidence.detach(),
        tv_distance=tv_distance.detach(),
        overlap=overlap.detach(),
        clean_gt_probability=components.clean_gt_probability.detach(),
        non_gt_dose=non_gt_dose.detach(),
    )


def _target_components(
    clean_logits: torch.Tensor,
    branch_logits: torch.Tensor,
    target_ids: torch.Tensor,
    codebook_kernel: torch.Tensor,
    *,
    clean_temperature: float,
    eps: float,
) -> _TargetComponents:
    """Compute the detached probabilities and codebook affinities."""

    if clean_logits.ndim != 3 or branch_logits.ndim != 3:
        raise ValueError(
            "clean_logits and branch_logits must be [batch, tokens, vocabulary]"
        )
    if clean_logits.shape != branch_logits.shape:
        raise ValueError("clean_logits and branch_logits must have the same shape")
    if target_ids.shape != clean_logits.shape[:2]:
        raise ValueError("target_ids must match logits [batch, tokens]")
    if target_ids.is_floating_point():
        raise TypeError("target_ids must contain integer token IDs")
    if not math.isfinite(clean_temperature) or clean_temperature <= 0.0:
        raise ValueError("clean_temperature must be finite and positive")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    vocabulary = clean_logits.shape[-1]
    if vocabulary < 2:
        raise ValueError("logit vocabulary must contain at least two entries")
    if tuple(codebook_kernel.shape) != (vocabulary, vocabulary):
        raise ValueError("codebook_kernel shape must match the logit vocabulary")
    if target_ids.numel() and (
        int(target_ids.min()) < 0 or int(target_ids.max()) >= vocabulary
    ):
        raise ValueError("target_ids contains an out-of-vocabulary token")

    with torch.no_grad():
        clean = F.softmax(
            clean_logits.detach().float() / float(clean_temperature), dim=-1
        )
        branch = F.softmax(branch_logits.detach().float(), dim=-1)
        kernel = codebook_kernel.detach().to(device=clean.device, dtype=torch.float32)
        if not bool(torch.isfinite(kernel).all()) or bool((kernel < 0.0).any()):
            raise ValueError("codebook_kernel must be finite and non-negative")
        kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        diagonal = kernel.diagonal().clamp_min(float(eps))
        affinity = (kernel / diagonal[:, None]).clamp(0.0, 1.0)
        target_affinity = F.embedding(target_ids.detach().long(), affinity)

        hard = F.one_hot(target_ids.detach().long(), num_classes=vocabulary).float()
        target_index = target_ids.detach().long()[..., None]
        clean_gt_probability = clean.gather(-1, target_index).squeeze(-1)
    return _TargetComponents(
        clean=clean.detach(),
        branch=branch.detach(),
        hard=hard.detach(),
        target_affinity=target_affinity.detach(),
        clean_gt_probability=clean_gt_probability.detach(),
    )


def masked_token_mean(
    per_token_value: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average token values, returning differentiable zero for an empty mask."""

    if per_token_value.ndim != 2:
        raise ValueError("per_token_value must have shape [batch, tokens]")
    if loss_mask is None:
        if per_token_value.numel() == 0:
            return per_token_value.sum() * 0.0
        return per_token_value.mean()
    if loss_mask.shape != per_token_value.shape:
        raise ValueError("loss_mask must have the same [batch, tokens] shape")

    weights = loss_mask.detach().to(
        device=per_token_value.device, dtype=per_token_value.dtype
    )
    numerator = (per_token_value * weights).sum()
    denominator = weights.sum()
    return numerator / denominator.clamp_min(1.0)


def hard_token_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross entropy for logits already aligned with complete image-token targets."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocabulary]")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError("target_ids must match logits [batch, tokens]")
    per_token = F.cross_entropy(
        logits.transpose(1, 2), target_ids.long(), reduction="none"
    )
    return masked_token_mean(per_token, loss_mask)


def soft_token_cross_entropy(
    logits: torch.Tensor,
    target_probability: torch.Tensor,
    *,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Soft-label cross entropy with an optional token mask."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocabulary]")
    if target_probability.shape != logits.shape:
        raise ValueError("target_probability must have the same shape as logits")
    target = target_probability.detach().to(device=logits.device, dtype=torch.float32)
    per_token = -(target * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
    return masked_token_mean(per_token, loss_mask)
