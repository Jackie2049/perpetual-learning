#!/usr/bin/env bash
# DeepInfer 平民版移植冒烟第二段（2026-10-22，服务器 4090-1，M5）
# 目标：替换四个 kernel import（stub） + 放宽架构检查后验证 llm_engine import 链
#       与 tilelang kernel 在 SM89 上的 JIT 编译可行性。
#
# 前置：服务器 4090-1（8xRTX4090 SM89），独立克隆 /home/zxw/deepinfer-civilian
#       （HEAD 774ce244，16 子模块全 init），conda 环境 deepinfer-smoke
#       （Python 3.12.15 + torch 2.13.0+cu129 + gcc13 conda-forge + tilelang 0.1.15）。
# 注意：stub 只保证 import 链通过，调用即抛 NotImplementedError，不代表 kernel 可用。

set -euo pipefail

CONDA_SH=/home/zxw/miniconda3/etc/profile.d/conda.sh
STUBS=/home/zxw/deepinfer-stubs
REPO=/home/zxw/deepinfer-civilian

source "$CONDA_SH"
conda activate deepinfer-smoke
cd "$REPO"

export PYTHONPATH="$STUBS:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib"

# 1) 架构检查放宽（已就地 patch deep_cluster/hardware/device.py，SM89 -> 82）
python - <<'EOF'
from deep_cluster.hardware import get_bf16_tflops_limit
import torch
print("tflops_limit:", get_bf16_tflops_limit())        # 期望 82 (SM89 4090)
print("capability:", torch.cuda.get_device_capability(0))  # 期望 (8, 9)
EOF

# 2) 核心模块 import 链验证（20 个模块应全部 OK）
python - <<'EOF'
import importlib
mods = [
    "deepinfer.engine.llm_engine", "deepinfer.engine.arg_utils", "deepinfer.engine.profile_server",
    "deepinfer.model_executor.parallel_utils.parallel_state",
    "deepinfer.model_executor.layers.sampler", "deepinfer.model_executor.layers.attention_backend",
    "deepinfer.model_executor.layers.linear", "deepinfer.model_executor.model_loader",
    "deepinfer.model_executor.models.deepseek", "deepinfer.model_executor.models.deepseek_vl",
    "deepinfer.model_executor.models.v4_modules.attn", "deepinfer.model_executor.models.v4_modules.mlp",
    "deepinfer.model_executor.models.v4_modules.hc", "deepinfer.model_executor.models.v4_modules.engram",
    "deepinfer.model_executor.models.v4_modules.mhc_ops", "deepinfer.model_executor.models.v4_modules.index_attn",
    "deepinfer.worker.worker", "deepinfer.worker.base",
    "deepinfer.sequence", "deepinfer.sampling_params",
]
ok = fail = 0
for m in mods:
    try:
        importlib.import_module(m); ok += 1
    except Exception as e:
        fail += 1; print("FAIL", m, type(e).__name__, e)
print(f"import chain: {ok} OK / {fail} FAIL (expect {len(mods)} OK / 0 FAIL)")
EOF

# 3) tilelang: 最小 saxpy 对照（SM89 上 tilelang 0.1.15 本身应可编译运行）
python /tmp/tl_saxpy.py

# 4) tilelang: sync_grid（PDL 替代方案）在 SM89 上应可编译运行
python /tmp/tl_gridsync.py

# 5) tilelang: DeepInfer norm_tl（含 T.pdl_sync/pdl_trigger）在 SM89 上预期失败
#    "PDL is not supported"——PDL 是 Hopper+（SM90+）硬件特性，SM89 不支持。
#    全部 43 个 cuda tilelang kernel 文件均显式使用 PDL，见专题笔记。
#    负验证：预期失败即通过；用 set +e 避免脚本中断。
set +e
python /tmp/tl_norm_tl.py 2>&1 | grep -q "PDL is not supported" && echo "norm_tl FAILS as expected (PDL is not supported on SM89)" || echo "norm_tl did NOT fail as expected"
set -e
EOF
