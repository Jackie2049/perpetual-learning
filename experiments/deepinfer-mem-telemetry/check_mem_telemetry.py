#!/usr/bin/env python
"""
DeepInfer #151 显存遥测轻量核对 — 4090-1, deepinfer-smoke, GPU 0 (2026-10-22, M5)

对照 handoff-m5.md#deepinfer-显存遥测的轻量核对。方法：从服务器源码文件用 AST
提取原函数体（逐字节保持），在独立命名空间 exec；show_gpu_memory_stats 里对
deepinfer...parallel_state 的导入用 sys.modules stub 隔离（记录隔离方式，函数体不改）。
正常路径全部走真实 torch.cuda API；异常路径用受控 mock 触发（不制造驱动故障）。

base = /home/zxw/deepinfer-civilian/deepinfer/utils.py (HEAD 774ce244)
head = /tmp/di151patch/deepinfer/utils.py (对 base 应用 PR #151 diff 的副本)
"""
import sys, os, ast, types, hashlib, logging

print("=== identity ===")
print("python:", sys.version.split()[0])
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
print("current_device:", torch.cuda.current_device(), "| name:", torch.cuda.get_device_name(torch.cuda.current_device()))

try:
    import psutil
    print("psutil:", psutil.__version__)
except Exception as e:
    psutil = None
    print("psutil unavailable:", e)

BASE = "/home/zxw/deepinfer-civilian/deepinfer/utils.py"
HEAD = "/tmp/di151patch/deepinfer/utils.py"

def extract(src_path):
    src = open(src_path, "r").read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "get_occupied_gpu_memory", "get_cpu_memory_usage_info", "show_gpu_memory_stats",
        ):
            out[node.name] = ast.get_source_segment(src, node)
    return out

base_f = extract(BASE)
head_f = extract(HEAD)
print("=== source identity ===")
print("base sha256:", hashlib.sha256(open(BASE, "rb").read()).hexdigest())
print("head sha256:", hashlib.sha256(open(HEAD, "rb").read()).hexdigest())
print("base extracted fns:", sorted(base_f))
print("head extracted fns:", sorted(head_f))
# M1 记录引用：base 8a0049b9... / head db14bca1...（8b161497 与 base 无 diff，见笔记）
print("M1 base ref 8a0049b98ea6f7ef98555ad8b38b9e8dac2292c0d2a89c2090bbdb5e5ee7295a")
print("M1 head ref db14bca1ddc8ad3e9a5ae0f99b8aeb1b9ebebf3c2262eb2e0837b06988c73258")

recs = []
class Cap(logging.Handler):
    def emit(self, r):
        recs.append(self.format(r))
cap = Cap()
lf = logging.getLogger("di151.memcheck")
lf.setLevel(logging.DEBUG)
lf.handlers = [cap]
lf.propagate = False

# 隔离方式：只 stub deepinfer 包与 parallel_state 模块，show_gpu_memory_stats 函数体逐字节保持
pg_mod = types.ModuleType("deepinfer.model_executor.parallel_utils.parallel_state")
class _PG:
    @staticmethod
    def get_rank():
        return 0
pg_mod.PG = _PG
for name in ("deepinfer", "deepinfer.model_executor", "deepinfer.model_executor.parallel_utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["deepinfer.model_executor.parallel_utils.parallel_state"] = pg_mod

def mkns():
    return {"torch": torch, "psutil": psutil, "logging": logging, "logger": lf}

nsB = mkns(); exec(base_f["get_occupied_gpu_memory"], nsB)
nsH = mkns(); exec(head_f["get_occupied_gpu_memory"], nsH)

print("=== check 1: mem_get_info 采样身份与时序 ===")
f1, t1 = torch.cuda.mem_get_info()
f2, t2 = torch.cuda.mem_get_info()
print(f"call#1: free={f1} total={t1} (occupied={t1-f1})")
print(f"call#2: free={f2} total={t2} (occupied={t2-f2})")
print(f"free stable: {f1==f2}; total stable: {t1==t2}")
occB = nsB["get_occupied_gpu_memory"]()
occH = nsH["get_occupied_gpu_memory"]()
print(f"base 版(两次调用, total(t1)-free(t2)): {occB}")
print(f"head 版(单次调用, total-free):          {occH}")
print(f"base == t2-f2 ? {occB == t2 - f2} ; head == t1-f1 ? {occH == t1 - f1} ; base==head ? {occB == occH}")

print("=== check 2: 三个显存口径（Allocated/Reserved/Occupied 语义不同）===")
torch.cuda.empty_cache()
buf = torch.ones(1024 * 1024 * 256, dtype=torch.float32, device="cuda")  # 1 GiB
alloc = torch.cuda.memory_allocated()
maxalloc = torch.cuda.max_memory_allocated()
resv = torch.cuda.memory_reserved()
maxresv = torch.cuda.max_memory_reserved()
f3, t3 = torch.cuda.mem_get_info()
occ3 = nsB["get_occupied_gpu_memory"]()
print("-- 分配 1 GiB 后（此刻 1 GiB 恰好等于分配器池粒度，Allocated==Reserved 是伪影）--")
print(f"Allocated(current/max): {alloc/2**30:.3f} / {maxalloc/2**30:.3f} GiB")
print(f"Reserved(current/max):  {resv/2**30:.3f} / {maxresv/2**30:.3f} GiB")
print(f"Occupied(设备全局 total-free): {occ3/2**30:.3f} GiB (free={f3/2**30:.3f}, total={t3/2**30:.3f} GiB)")
del buf
alloc2 = torch.cuda.memory_allocated()
resv2 = torch.cuda.memory_reserved()
f4, t4 = torch.cuda.mem_get_info()
occ4 = nsB["get_occupied_gpu_memory"]()
print("-- del 释放 tensor 后（不 empty_cache：Allocated 归 0，Reserved 仍持有，Occupied 含设备全局）--")
print(f"Allocated: {alloc2/2**30:.3f} GiB ; Reserved: {resv2/2**30:.3f} GiB ; Occupied: {occ4/2**30:.3f} GiB")
print(f"三口径两两不同: {alloc2 != resv2 and resv2 != occ4 and alloc2 != occ4}")
occ3 = occ4

print("=== check 3: show_gpu_memory_stats 日志（原函数体 + PG stub）===")
recs.clear()
nsS = mkns()
exec(base_f["get_cpu_memory_usage_info"], nsS)
exec(base_f["get_occupied_gpu_memory"], nsS)
exec(base_f["show_gpu_memory_stats"], nsS)
nsS["show_gpu_memory_stats"]("MEMCHECK base", level=logging.INFO)
for r in recs:
    print("LOG:", r)

print("=== check 4: 查询异常路径（受控 mock，非驱动故障）===")
recs.clear()
orig_mgi = torch.cuda.mem_get_info
def boom():
    raise RuntimeError("controlled mock: query failure")
torch.cuda.mem_get_info = boom
try:
    oB = nsB["get_occupied_gpu_memory"]()
    nB = len(recs)
    oH = nsH["get_occupied_gpu_memory"]()
    nH = len(recs) - nB
finally:
    torch.cuda.mem_get_info = orig_mgi
print(f"base 版异常返回值: {oB}（静默，无日志）  新增日志条数={nB}")
print(f"head 版异常返回值: {oH}（warning）        新增日志条数={nH}")
for r in recs:
    print("LOG:", r)

print("=== check 5: 真实 import deepinfer.utils（PYTHONPATH 含 stub 时）===")
try:
    import deepinfer.utils as du
    occ_real = du.get_occupied_gpu_memory()
    print(f"real import OK; du.get_occupied_gpu_memory() = {occ_real}")
    print(f"与提取版一致? {occ_real == occ3}")
except Exception as e:
    print("real import 失败（不影响提取版结论）:", type(e).__name__, e)
