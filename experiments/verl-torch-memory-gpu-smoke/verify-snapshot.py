import pickle, sys, os, json
import torch

p = sys.argv[1]
out = {"pickle": p}
out["has_cuda_oom_observer_symbol"] = hasattr(torch._C, "_cuda_attach_out_of_memory_observer")
out["torch"] = torch.__version__
out["cuda_available"] = torch.cuda.is_available()
out["pickle_bytes"] = os.path.getsize(p)
with open(p, "rb") as f:
    snap = pickle.load(f)
out["snapshot_type"] = type(snap).__name__
if isinstance(snap, dict):
    out["keys"] = sorted(snap.keys())
    for k in ("segments", "blocks", "device_traces"):
        v = snap.get(k)
        if v is not None:
            out[k + "_len"] = len(v)
            if k == "device_traces" and v:
                out["device_traces_0_len"] = len(v[0])
print(json.dumps(out, indent=1, default=str))
