"""Minimal deep_gemm stub for civilian-port smoke (SM89): keep imports alive, raise on use."""
import torch


class _NotImplemented:
    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        raise NotImplementedError(f"deep_gemm.{self._name} not available on civilian port (SM89)")

    def __getattr__(self, k):
        return _NotImplemented(f"{self._name}.{k}")


class SymmBuffer:
    def __init__(self, *a, **k):
        pass


def get_num_sms():
    return 128


def set_num_sms(n):
    pass


def get_block_m_for_mega_moe(*a, **k):
    return 128


def get_tma_aligned_size(*a, **k):
    return a[0] if a else 0


def get_theoretical_mk_alignment_for_contiguous_layout():
    return 1


def set_mk_alignment_for_contiguous_layout(n):
    pass


def get_fp8_casting_granularity():
    return 32


def get_paged_mqa_logits_metadata(*a, **k):
    return None


def get_paged_sparse_mqa_logits_metadata(*a, **k):
    return None


def get_sparse_mqa_logits_metadata(*a, **k):
    return None


def transform_sf_into_required_layout(*a, **k):
    return a[0] if a else None


def tf32_hc_prenorm_gemm(*a, **k):
    raise NotImplementedError("deep_gemm.tf32_hc_prenorm_gemm not available on civilian port (SM89)")

def bf16_gemm_nt(*a, **k):
    raise NotImplementedError("deep_gemm.bf16_gemm_nt not available on civilian port (SM89)")

def m_grouped_bf16_gemm_nt_contiguous(*a, **k):
    raise NotImplementedError("deep_gemm.m_grouped_bf16_gemm_nt_contiguous not available on civilian port (SM89)")

def m_grouped_fp8_fp4_gemm_nt_contiguous(*a, **k):
    raise NotImplementedError("deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous not available on civilian port (SM89)")

def fp8_fp4_gemm_nt(*a, **k):
    raise NotImplementedError("deep_gemm.fp8_fp4_gemm_nt not available on civilian port (SM89)")

def fp8_einsum(*a, **k):
    raise NotImplementedError("deep_gemm.fp8_einsum not available on civilian port (SM89)")

def fp8_fp4_mega_moe(*a, **k):
    raise NotImplementedError("deep_gemm.fp8_fp4_mega_moe not available on civilian port (SM89)")

def bf16_mega_moe(*a, **k):
    raise NotImplementedError("deep_gemm.bf16_mega_moe not available on civilian port (SM89)")

def mega_mhc(*a, **k):
    raise NotImplementedError("deep_gemm.mega_mhc not available on civilian port (SM89)")

def fp8_fp4_mqa_logits(*a, **k):
    raise NotImplementedError("deep_gemm.fp8_fp4_mqa_logits not available on civilian port (SM89)")

def fp8_fp4_sparse_mqa_logits(*a, **k):
    raise NotImplementedError("deep_gemm.fp8_fp4_sparse_mqa_logits not available on civilian port (SM89)")

def fp8_fp4_paged_mqa_logits(*a, **k):
    raise NotImplementedError("deep_gemm.fp8_fp4_paged_mqa_logits not available on civilian port (SM89)")

def fp8_fp4_paged_sparse_mqa_logits(*a, **k):
    raise NotImplementedError("deep_gemm.fp8_fp4_paged_sparse_mqa_logits not available on civilian port (SM89)")
