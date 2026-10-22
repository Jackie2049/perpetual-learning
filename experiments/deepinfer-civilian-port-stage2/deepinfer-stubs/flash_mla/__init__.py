"""Minimal flash_mla stub for civilian-port smoke (SM89): keep imports alive, raise on use."""
import torch


def flash_mla_with_kvcache(*a, **k):
    raise NotImplementedError("flash_mla.flash_mla_with_kvcache not available on civilian port (SM89)")

def flash_mla_sparse_fwd(*a, **k):
    raise NotImplementedError("flash_mla.flash_mla_sparse_fwd not available on civilian port (SM89)")

def fused_norm_rope_attn_rope_cast(*a, **k):
    raise NotImplementedError("flash_mla.fused_norm_rope_attn_rope_cast not available on civilian port (SM89)")
