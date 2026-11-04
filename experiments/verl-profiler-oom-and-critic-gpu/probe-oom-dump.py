#!/usr/bin/env python3
"""Probe: when the allocator hits OOM, does TorchMemoryProfiler really dump a snapshot?

Two arms, run as separate processes (the observer is a process-local class attribute):

  --arm attached   rank 0, config.ranks=[0]  -> _attach_oom_observer runs
  --arm unranked   rank 3, config.ranks=[0]  -> observer NOT attached (negative control)

Both arms force the same allocator OOM, so a snapshot in one arm and none in the
other isolates the observer as the cause.

The dump is written by the *third-party-ish* real code path:
  TorchMemoryProfiler._on_out_of_memory -> MemorySnapshotSampler.dump_memory_snapshot
so this is the shipping implementation, not a re-implementation.

Pitfalls this script is written around (learned the hard way in the 11-04 16:5x round):
  * `python /path/script.py` puts the *script directory* on sys.path[0], so the tree
    under test must be pinned with PYTHONPATH and verl.__file__ must be recorded.
  * every reading is written to JSON; the log tail is not the evidence.
"""

import argparse
import json
import logging
import pickle
import shutil
import time
from pathlib import Path

import torch


class Capture(logging.Handler):
    """Keep profiler log lines in-process; the driver log drops them (see round 16:5x)."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(
            {"level": record.levelname, "logger": record.name, "msg": record.getMessage()[:500]}
        )


def list_files(root: Path):
    if not root.exists():
        return []
    return sorted(
        str(p.relative_to(root)) + (f" ({p.stat().st_size} B)" if p.is_file() else "/")
        for p in root.rglob("*")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["attached", "unranked"], required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--no-log-capture",
        action="store_true",
        help="leave logging at its default (root WARNING, no handler) to see what a user sees",
    )
    args = ap.parse_args()

    import verl
    from verl.utils.profiler.config import ProfilerConfig, TorchMemoryToolConfig
    from verl.utils.profiler.torch_memory_profile import TorchMemoryProfiler

    cap = Capture()
    if not args.no_log_capture:
        logging.getLogger().addHandler(cap)
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("verl").setLevel(logging.DEBUG)

    save = Path(args.save)
    shutil.rmtree(save, ignore_errors=True)
    save.mkdir(parents=True, exist_ok=True)

    rank = 0 if args.arm == "attached" else 3
    cfg = ProfilerConfig(tool="torch_memory", save_path=str(save), ranks=[0], all_ranks=False)
    prof = TorchMemoryProfiler(rank=rank, config=cfg, tool_config=TorchMemoryToolConfig())

    result = {
        "arm": args.arm + ("-nocapture" if args.no_log_capture else ""),
        "verl_file": verl.__file__,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "rank": rank,
        "config_ranks": [0],
        "should_profile_this_rank": prof._should_profile_this_rank(),
        "oom_observer_attached_after_init": bool(TorchMemoryProfiler._oom_observer_attached),
        "has_cuda_attach_api": hasattr(torch._C, "_cuda_attach_out_of_memory_observer"),
        "files_before": list_files(save),
    }

    torch.cuda.init()
    _ = torch.empty(8, device="cuda")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    result["mem_before"] = {"free_bytes": free, "total_bytes": total}

    # Force an allocator OOM: ask for 1 TiB on a 24 GiB card.
    oom_err = None
    t_request = time.time()
    try:
        _huge = torch.empty(int(1e12), dtype=torch.uint8, device="cuda")
        result["oom_raised"] = False
    except Exception as exc:  # torch.OutOfMemoryError is a RuntimeError subclass
        oom_err = exc
        result["oom_raised"] = True
        result["except_class"] = type(exc).__name__
        result["oom_msg"] = str(exc)[:600]
    result["oom_request_seconds"] = round(time.time() - t_request, 3)

    # The callback is invoked by the allocator while the OOM is being handled; poll the
    # output directory a few times so a slow dump is not mistaken for "no dump".
    marks = {}
    t0 = time.time()
    for delay in (0, 1, 3, 10, 20):
        time.sleep(max(0.0, delay - (time.time() - t0)))
        marks[f"{delay}s"] = list_files(save)
    result["files_after"] = marks
    result["final_files"] = list_files(save)

    # Inspect any snapshot that landed.
    snapshots = []
    for entry in result["final_files"]:
        path = entry.split(" (")[0].rstrip("/")
        full = save / path
        if full.is_file() and full.suffix == ".pickle":
            info = {"path": path, "bytes": full.stat().st_size}
            try:
                with full.open("rb") as f:
                    snap = pickle.load(f)
                info["keys"] = sorted(snap.keys()) if isinstance(snap, dict) else f"<{type(snap).__name__}>"
                traces = snap.get("device_traces") if isinstance(snap, dict) else None
                if traces is not None:
                    info["device_traces"] = (
                        [len(t) for t in traces] if isinstance(traces, list) else str(type(traces))
                    )
                segs = snap.get("segments") if isinstance(snap, dict) else None
                if segs is not None:
                    info["segments"] = len(segs) if hasattr(segs, "__len__") else str(type(segs))
            except Exception as exc:
                info["load_error"] = f"{type(exc).__name__}: {exc}"
            snapshots.append(info)
    result["snapshots"] = snapshots

    result["log_lines"] = cap.records
    result["log_counts_by_logger_level"] = {}
    for rec in cap.records:
        key = f"{rec['logger']}:{rec['level']}"
        result["log_counts_by_logger_level"][key] = result["log_counts_by_logger_level"].get(key, 0) + 1

    # Still alive after OOM?
    try:
        torch.cuda.synchronize()
        result["cuda_alive_after_oom"] = True
        result["mem_after"] = dict(zip(("free_bytes", "total_bytes"), torch.cuda.mem_get_info()))
    except Exception as exc:
        result["cuda_alive_after_oom"] = False
        result["cuda_alive_error"] = f"{type(exc).__name__}: {exc}"[:300]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in ("arm", "oom_raised", "except_class", "final_files", "snapshots")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
