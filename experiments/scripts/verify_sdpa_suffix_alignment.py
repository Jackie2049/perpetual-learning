"""Real CPU SDPA suffix controls, tested with torch 2.9.1.

No device discovery, model, KV allocator, GPU kernel or throughput benchmark.
"""

import sys

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right


CPU = torch.device("cpu")
DTYPE = torch.float64


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


def suffix_mask(query_positions, key_positions, valid_keys=None):
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    return allowed if valid_keys is None else allowed & valid_keys.unsqueeze(0)


def visible_mean_example():
    # Prefix positions 0,1; suffix queries 2,3,4; every logit is zero.
    query = torch.zeros(1, 1, 3, 2, dtype=DTYPE, device=CPU)
    key = torch.zeros(1, 1, 5, 2, dtype=DTYPE, device=CPU)
    value = torch.arange(5, dtype=DTYPE, device=CPU).reshape(1, 1, 5, 1)
    mask = suffix_mask(torch.arange(2, 5, device=CPU), torch.arange(5, device=CPU))
    upper = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    unmasked = F.scaled_dot_product_attention(query, key, value, is_causal=False)
    aligned = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    close(upper.flatten(), torch.tensor([0.0, 0.5, 1.0], dtype=DTYPE, device=CPU))
    close(unmasked.flatten(), torch.tensor([2.0, 2.0, 2.0], dtype=DTYPE, device=CPU))
    close(aligned.flatten(), torch.tensor([1.0, 1.5, 2.0], dtype=DTYPE, device=CPU))
    # Materialize on CPU only: do not invoke CausalBias's automatic backend dispatch.
    bias_mask = causal_lower_right(3, 5)._materialize(CPU)
    assert torch.equal(mask, bias_mask)
    print("3 suffix queries, 5 valid keys: upper-left", upper.flatten().tolist(),
          "unmasked", unmasked.flatten().tolist(), "position-aligned", aligned.flatten().tolist())

    # An allocated 7-slot cache has only 5 valid keys. Shape-only alignment is wrong
    # even after removing the two padding columns: it moves query positions too far.
    capacity_key = torch.zeros(1, 1, 7, 2, dtype=DTYPE, device=CPU)
    capacity_value = torch.tensor([0, 1, 2, 3, 4, 100, 200], dtype=DTYPE, device=CPU).reshape(1, 1, 7, 1)
    key_pos = torch.arange(7, device=CPU)
    valid = key_pos < 5
    correct = suffix_mask(torch.arange(2, 5, device=CPU), key_pos, valid)
    wrong_offset = causal_lower_right(3, 7)._materialize(CPU) & valid.unsqueeze(0)
    padded_result = F.scaled_dot_product_attention(query, capacity_key, capacity_value, attn_mask=correct)
    wrong_result = F.scaled_dot_product_attention(query, capacity_key, capacity_value, attn_mask=wrong_offset)
    close(padded_result, aligned)
    close(wrong_result.flatten(), unmasked.flatten())
    assert not torch.equal(wrong_result, aligned)
    print("7 allocated / 5 valid keys: actual positions", padded_result.flatten().tolist(),
          "capacity-aligned plus validity mask", wrong_result.flatten().tolist())


def compare_suffix_with_full(prefix, suffix):
    generator = torch.Generator(device=CPU).manual_seed(2000 + 10 * prefix + suffix)
    total = prefix + suffix
    query = torch.randn(1, 2, total, 4, generator=generator, dtype=DTYPE, device=CPU).requires_grad_()
    key = torch.randn(1, 2, total, 4, generator=generator, dtype=DTYPE, device=CPU).requires_grad_()
    value = torch.randn(1, 2, total, 3, generator=generator, dtype=DTYPE, device=CPU).requires_grad_()
    reference = F.scaled_dot_product_attention(query, key, value, is_causal=True)[..., prefix:, :]
    weights = torch.arange(1, reference.numel() + 1, dtype=DTYPE, device=CPU).reshape(reference.shape)
    reference_grads = torch.autograd.grad((reference * weights).sum(), (query, key, value))

    chunk_q = query.detach()[..., prefix:, :].clone().requires_grad_()
    chunk_k = key.detach().clone().requires_grad_()
    chunk_v = value.detach().clone().requires_grad_()
    mask = suffix_mask(torch.arange(prefix, total, device=CPU), torch.arange(total, device=CPU))
    result = F.scaled_dot_product_attention(chunk_q, chunk_k, chunk_v, attn_mask=mask)
    result_grads = torch.autograd.grad((result * weights).sum(), (chunk_q, chunk_k, chunk_v))
    close(result, reference)
    close(result_grads[0], reference_grads[0][..., prefix:, :])
    close(result_grads[1], reference_grads[1])
    close(result_grads[2], reference_grads[2])
    assert torch.equal(mask, causal_lower_right(suffix, total)._materialize(CPU))
    if suffix == 1:
        close(F.scaled_dot_product_attention(chunk_q, chunk_k, chunk_v), reference)
    if prefix == 0:
        close(F.scaled_dot_product_attention(chunk_q, chunk_k, chunk_v, is_causal=True), reference)
    error = float((result - reference).detach().abs().max())
    print(f"prefix={prefix} suffix={suffix}: output and Q/K/V gradients match full causal suffix; max output error={error:.3g}")


def main():
    print("Python", sys.version.split()[0], "torch", torch.__version__, "git", torch.version.git_version,
          "CPU float64 / MATH / dropout_p=0")
    with sdpa_kernel(SDPBackend.MATH):
        visible_mean_example()
        for prefix, suffix in [(4, 1), (2, 3), (0, 5), (3, 2)]:
            compare_suffix_with_full(prefix, suffix)
    print("PASS: operator semantics only; CausalBias CPU materialization, not its automatic backend dispatch")


if __name__ == "__main__":
    main()
