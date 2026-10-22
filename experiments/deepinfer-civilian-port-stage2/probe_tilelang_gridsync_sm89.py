import tilelang, tilelang.language as T, torch, time

# two-block producer/consumer separated by grid sync (PDL-free pattern)
@tilelang.jit(out_idx=None, target="cuda")
def gridsync_test(N: int, dtype: str = "float16"):
    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(2, threads=64) as (bid,):
            for i in T.Parallel(N):
                pass
            T.sync_grid()
            tx = T.get_thread_binding()
            idx = bid * (N // 2) + tx
            if bid == 1 and idx < N:
                C[idx] = A[idx] + B[idx]
    return main

t0 = time.time()
try:
    k = gridsync_test(128)
    print("sync_grid kernel compiled OK in %.1fs" % (time.time()-t0))
    a = torch.ones(128, dtype=torch.float16, device="cuda")
    b = torch.ones(128, dtype=torch.float16, device="cuda")
    c = torch.empty(128, dtype=torch.float16, device="cuda")
    k(a, b, c)
    torch.cuda.synchronize()
    print("ran OK, sample:", c[64:68].tolist())
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:200])
