"""Frozen visual-tokenizer adapters used by NTI."""

from .maskgit_vqgan import (
    CheckpointLoadReport,
    MaskGITVQGANAdapter,
    MaskGITVQGANSpec,
    load_frozen_maskgit_vqgan,
    load_maskgit_codebook,
)

__all__ = [
    "CheckpointLoadReport",
    "MaskGITVQGANAdapter",
    "MaskGITVQGANSpec",
    "load_frozen_maskgit_vqgan",
    "load_maskgit_codebook",
]
