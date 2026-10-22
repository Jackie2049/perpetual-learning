from typing import Any

import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=None,
    target="cuda",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def sum_smaller_probs_tl(
    vocab_size: int,
) -> Any:
    num_tokens = T.dynamic("num_tokens")
    logprobs_stride = T.dynamic("logprobs_stride")

    threads = 512
    block_size = 8192 if vocab_size >= 8192 else tilelang.next_power_of_2(vocab_size)
    n_chunks = T.ceildiv(vocab_size, block_size)

    @T.prim_func
    def _sum_smaller_probs(
        logprobs: T.StridedTensor((num_tokens, vocab_size), (logprobs_stride, 1), T.float32),  # type: ignore
        sample_results: T.Tensor((num_tokens,), T.int32),  # type: ignore
        output: T.Tensor((num_tokens,), T.float32),  # type: ignore
    ) -> None:
        T.assume(logprobs_stride % 8 == 0)
        with T.Kernel(num_tokens, threads=threads) as bx:
            T.pdl_sync()
            sample_result = T.alloc_var(T.int32, sample_results[bx])
            thread_sum = T.alloc_reducer((1,), T.float32, replication="all")
            T.fill(thread_sum, 0.0)

            if sample_result < 0:
                if T.get_thread_binding() == 0:
                    output[bx] = 0.0
            else:
                logprob_limit = T.alloc_var(T.float32, logprobs[bx, sample_result])
                for i0 in T.serial(n_chunks):
                    logprobs_frag = T.alloc_fragment((block_size,), T.float32)
                    T.copy(logprobs[bx, i0 * block_size], logprobs_frag)
                    for j in T.Parallel(block_size):
                        col = i0 * block_size + j
                        if col < vocab_size:
                            val = T.alloc_var(T.float32, logprobs_frag[j])
                            if val < logprob_limit or (val == logprob_limit and col <= sample_result):
                                thread_sum[0] += T.exp(val)
                T.finalize_reducer(thread_sum)
                if T.get_thread_binding() == 0:
                    output[bx] = thread_sum[0]

    return _sum_smaller_probs
