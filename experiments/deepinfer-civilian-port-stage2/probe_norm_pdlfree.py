"""探针 v2：真实 DeepInfer norm_tl kernel 文本级去 PDL（pdl_sync->sync_grid, pdl_trigger->删除）
在 SM89 (4090) 上编译并对照 torch 验证数值正确性。

v1 教训：xdim=2 时 kernel 只写 batch 0（bb=0），多 batch 必须 xdim=3。
方法：服务器上 cuda 包目录已有 norm_pdlfree.py（由 norm.py sed 生成，仅替换 3 处 PDL 调用，
未改动原文件；git 未跟踪）。这里直接导入该真实源码文件编译运行。
"""
import sys
import time

import torch

sys.path.insert(0, "/home/zxw/deepinfer-civilian")
from deepinfer.model_executor.layers.tilelang_ops.cuda import norm_pdlfree as npf  # noqa: E402

D = 64
t0 = time.time()
kernel = npf.norm_tl(
    d=D,
    dtype="float16",
    return_dtype="float16",
    reduce_dtype="float32",
    has_residual=False,
    eps=1e-5,
    out_scale=1.0,
    do_fp8_cast=False,
    num_per_channels=1,
    use_tma_aligned_col_major_sf=False,
    round_sf=False,
    use_packed_ue8m0=False,
    return_origin_out=True,
    xdim=3,
    shared_sf_block_m=0,
    x_stride_s=D,
    y_stride_s=D,
    residual_stride_s=D,
    residual_out_stride_s=D,
    y_origin_stride_s=D,
    sf_stride_s=0,
)
print(f"compiled in {time.time()-t0:.1f}s")

torch.manual_seed(0)
B, S = 2, 128
x = torch.randn(B, S, D, dtype=torch.float16, device="cuda")
w = torch.randn(D, dtype=torch.float16, device="cuda") * 0.5 + 1.0
y = torch.empty(B, S, D, dtype=torch.float16, device="cuda")
residual = torch.zeros(B, S, D, dtype=torch.float16, device="cuda")
residual_out = torch.empty(B, S, D, dtype=torch.float16, device="cuda")
y_origin = torch.empty(B, S, D, dtype=torch.float16, device="cuda")
kernel(x, y, w, residual, residual_out, x, x, y_origin)
torch.cuda.synchronize()

# torch 对照: y = x / sqrt(mean(x^2)+eps) * w * out_scale  (fp32 内部, fp16 输出)
x32 = x.float()
rstd = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
ref = (x32 * rstd * w.float() * 1.0).half()

err = (y.float() - ref.float()).abs().max().item()
mismatch = (y != ref).float().mean().item() * 100
print(f"max abs err vs torch: {err:.5f}  (fp16 精度下应 <~0.01)")
print(f"mismatch ratio: {mismatch:.2f}%")
print(f"y_origin == y: {torch.equal(y_origin, y)}")
# 按 batch 分开报告
for b in range(B):
    eb = (y[b].float() - ref[b].float()).abs().max().item()
    print(f"batch {b}: max err {eb:.5f}")
print("RESULT:", "PASS" if err < 0.05 else "FAIL")
