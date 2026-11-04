"""Three GPU probes for the torch_memory profiler construction path.

A: ProfilerConfig with the *inner* tool dataclass  -> constructs, reports OOM-observer state
B: same but tool_config = the *container* dict     -> reproduces the static claim (AttributeError)
C: enable=False                                    -> check_enable() False, start() must not set this_step

Run with the tree explicitly on PYTHONPATH (a plain `python /path/script.py` keeps sys.path[0]
on the script directory, so the venv's *installed* verl would be imported instead):

    PYTHONPATH=$TREE CUDA_VISIBLE_DEVICES=1 python probe-profiler-path.py
"""

import json
import traceback

import torch

import verl
from verl.utils.profiler.config import ProfilerConfig, TorchMemoryToolConfig
from verl.utils.profiler.profile import DistProfiler
from verl.utils.profiler.torch_memory_profile import TorchMemoryProfiler

res = {"verl_file": verl.__file__, "torch": torch.__version__, "cuda": torch.cuda.is_available()}
res["cuda_oom_observer_symbol"] = hasattr(torch._C, "_cuda_attach_out_of_memory_observer")
inner = TorchMemoryToolConfig()
res["tool_config_cls"] = type(inner).__name__

# A: inner dataclass (what the worker and trainer actually pass)
try:
    cfg = ProfilerConfig(
        tool="torch_memory", enable=True, save_path="/tmp/m5-torchmem/probeA", tool_config=inner, ranks=[0]
    )
    d = DistProfiler(rank=0, config=cfg, tool_config=inner)
    res["A_construct"] = "ok"
    res["A_impl"] = type(d._impl).__name__
    res["A_oom_observer_attached"] = bool(d._impl._oom_observer_attached)
    res["A_check_enable"] = bool(d.check_enable())
except Exception as exc:
    res["A_construct"] = f"raised {type(exc).__name__}: {exc}"
    res["A_trace"] = traceback.format_exc().strip().splitlines()[-3:]

# B: container dict (the shape the static read says two call sites pass through)
try:
    container = {"torch_memory": inner}
    cfg_b = ProfilerConfig(
        tool="torch_memory", enable=True, save_path="/tmp/m5-torchmem/probeB", tool_config=container, ranks=[0]
    )
    d_b = DistProfiler(rank=0, config=cfg_b, tool_config=container)
    res["B_construct"] = "ok"
except Exception as exc:
    res["B_construct"] = f"raised {type(exc).__name__}: {exc}"

# C: enable=False (official example shape)
try:
    cfg_c = ProfilerConfig(
        tool="torch_memory", enable=False, save_path="/tmp/m5-torchmem/probeC", tool_config=inner, ranks=[0]
    )
    d_c = DistProfiler(rank=0, config=cfg_c, tool_config=inner)
    res["C_check_enable"] = bool(d_c.check_enable())
    res["C_start_return"] = str(d_c.start(profile_step=1))
    res["C_impl_this_step"] = bool(getattr(d_c._impl, "this_step", None))
except Exception as exc:
    res["C_result"] = f"raised {type(exc).__name__}: {exc}"

print(json.dumps(res, indent=1))
