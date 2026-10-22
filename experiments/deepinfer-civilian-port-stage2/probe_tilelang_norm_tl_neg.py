import torch, time
from deepinfer.model_executor.layers.tilelang_ops.cuda.norm import norm_tl
t0 = time.time()
kernel = norm_tl(d=64, dtype="float16", return_dtype="float16", reduce_dtype="float32",
                 has_residual=False, eps=1e-5, out_scale=1.0, do_fp8_cast=False,
                 num_per_channels=1, use_tma_aligned_col_major_sf=False, round_sf=False,
                 use_packed_ue8m0=False, return_origin_out=True, xdim=2, shared_sf_block_m=0,
                 x_stride_s=64, y_stride_s=64, residual_stride_s=64, residual_out_stride_s=64,
                 y_origin_stride_s=64, sf_stride_s=0)
print(f"kernel compiled in {time.time()-t0:.1f}s")
