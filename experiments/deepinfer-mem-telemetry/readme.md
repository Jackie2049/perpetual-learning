# DeepInfer #151 显存遥测轻量核对（4090-1，2026-10-22，M5）

## 目的

对照 [handoff-m5 显存遥测条目](../handoff-m5.md#deepinfer-显存遥测的轻量核对)，在真实设备上核对
DeepInfer `utils.py` 的显存采样 helper（`get_occupied_gpu_memory`）与日志
（`show_gpu_memory_stats`）行为：一次查询（#151 head）vs 两次查询（base）的语义差异、
Allocated / Reserved / Occupied 三个口径、异常回退路径。为 [M1 源码笔记](../notebook/frameworks/deepinfer-initial-survey.md#显存遥测采样与容量规划)
的静态结论补真机证据；不对 #151 PR 做任何上游操作（已有作者，不竞争实现）。

## 方法

- **来源身份**：服务器独立克隆 `/home/zxw/deepinfer-civilian`（HEAD `774ce244`，即 #151 base）的
  `deepinfer/utils.py`；head 版 = 对 base 副本应用 [PR #151 diff](pr151-utils.diff)（`patch -p1` 干净应用）。
  SHA256 与 M1 记录的两版引用**逐字节一致**（base `8a0049b9…` / head `db14bca1…`），证明
  服务器源码与研究基线相同、patch 应用后与 PR head 完全相同。
- **执行方式**：`check_mem_telemetry.py` 从源码文件用 AST 提取原函数体（逐字节保持，不手改），
  在独立命名空间 `exec`；`show_gpu_memory_stats` 对 `deepinfer…parallel_state` 的导入用
  `sys.modules` stub 隔离（只 stub 该模块，函数体不改）。正常路径全部走真实 `torch.cuda` API；
  异常路径用受控 mock 触发（`torch.cuda.mem_get_info` 临时替换为抛错函数，不制造驱动故障）。
- **环境**：4090-1，GPU 0（`CUDA_VISIBLE_DEVICES=0`），conda `deepinfer-smoke`（Python 3.12.15 +
  torch 2.13.0+cu129 + psutil 7.2.2），`PYTHONPATH` 含 `/home/zxw/deepinfer-stubs`、
  `LD_LIBRARY_PATH` 含 conda gcc13 libstdc++ / torch / nvjitlink（与移植冒烟同一环境）。

## 结果（全部 PASS）

1. **采样身份与时序**：两次连续 `mem_get_info` 完全稳定（free/total 均为
   24838340608 / 25250627584，即 GPU 0 空闲，仅 CUDA context ~393 MiB）。base 版（两次调用
   `total(t1)-free(t2)`）与 head 版（单次 `total-free`）返回值相同（`412286976`），且
   `base == t2-f2`、`head == t1-f1` —— 确认 M1 笔记「稳定时旧结果即第二次采样对应的 occupied」。
2. **三个显存口径语义不同**：分配 1 GiB tensor 后 `Allocated == Reserved == 1.000 GiB`
   （1 GiB 恰为缓存分配器池粒度，是探针伪影，已在输出标注）；`Occupied`（设备全局 `total-free`）
   = 1.445 GiB（含 CUDA context 等非本进程 tensor 占用）。`del` 释放 tensor 后（不 empty_cache）：
   **Allocated 0.000 GiB、Reserved 1.000 GiB（分配器仍持有）、Occupied 1.445 GiB** —— 三口径两两不同。
3. **日志行为**（原函数体 + PG stub，`show_gpu_memory_stats("MEMCHECK base")`）：
   `RANK 0 MEMCHECK base; GPU Allocated 0.000/1.000 GiB, GPU Reserved 1.000/1.000 GiB, GPU Occupied 1.445/23.516 GiB, CPU Memory 41.29/503.53 GiB, RSS 41.47 GiB` —— 当前/峰值与设备容量字段如实输出。
4. **异常路径**（受控 mock）：base 版静默返回 0（新增日志 0 条）；head 版返回 0 并发 warning
   `failed to query GPU memory usage: controlled mock: query failure`（新增日志 1 条）——
   确认「查询异常两版都回退 0、#151 额外发 warning」。
5. **真实 import**：脚本同进程内 stub 注册会让 `deepinfer` 变成非包模块（伪影）；独立进程
   `cd /home/zxw/deepinfer-civilian` 后 `import deepinfer.utils` 成功，真实函数返回值
   `412286976` 与 AST 提取版一致。

## 复现

```bash
# 服务器 4090-1
source /home/zxw/miniconda3/etc/profile.d/conda.sh && conda activate deepinfer-smoke
cd /home/zxw/deepinfer-civilian
export PYTHONPATH=/home/zxw/deepinfer-stubs:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nvjitlink/lib
CUDA_VISIBLE_DEVICES=0 python experiments/deepinfer-mem-telemetry/check_mem_telemetry.py
# head 版对照：cp deepinfer/utils.py /tmp/di151patch/deepinfer/utils.py && patch -p1 < experiments/deepinfer-mem-telemetry/pr151-utils.diff
```

## 边界

- 只核对 GPU 0 空闲态的正常路径与受控异常路径；未制造驱动故障、未在占用 GPU 上采样、未验证
  NPU（昇腾 VPN 掉线，`transfer_to_npu` 的 CUDA 语义是另一环境）。
- `show_gpu_memory_stats` 的 rank 来自 stub（真实接线由 deepinfer 包在 worker 内提供），函数体未改。
- 本项是项目自留证据，不构成对 #151 PR 的验收/评论；作者未执行 GPU runtime 验证的现状不因此改变。
