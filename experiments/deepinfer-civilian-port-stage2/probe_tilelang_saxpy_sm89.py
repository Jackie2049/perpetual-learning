import tilelang, tilelang.language as T, torch, time

@tilelang.jit(out_idx=None, target="cuda")
def saxpy(N: int, dtype: str = "float16"):
    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, 128), threads=128) as (bx,):
            tx = T.get_thread_binding()
            idx = bx * 128 + tx
            if idx < N:
                C[idx] = A[idx] * 2 + B[idx]
    return main

t0 = time.time()
k = saxpy(1024)
print("minimal kernel compiled OK in %.1fs" % (time.time()-t0))
a = torch.randn(1024, dtype=torch.float16, device="cuda")
b = torch.randn(1024, dtype=torch.float16, device="cuda")
c = torch.empty(1024, dtype=torch.float16, device="cuda")
k(a, b, c)
torch.cuda.synchronize()
print("ran OK, err:", (c.float() - (a*2+b).float()).abs().max().item())
