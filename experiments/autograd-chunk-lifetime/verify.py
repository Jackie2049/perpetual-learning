"""Native CPU saved-tensor lifecycle controls, not a memory/performance benchmark."""

import hashlib
import inspect
import json
import platform
import weakref

import torch


class Packed:
    def __init__(self, tensor):
        # Share values/storage but drop grad_fn to avoid a graph reference cycle.
        self.tensor = tensor.detach()


class Tracker:
    def __init__(self):
        self.live = weakref.WeakSet()
        self.peak_objects = 0
        self.peak_element_references = 0

    def current(self):
        return {"objects": len(self.live),
                "element_references": sum(item.tensor.numel() for item in self.live)}

    def pack(self, tensor):
        item = Packed(tensor)
        self.live.add(item)
        current = self.current()
        self.peak_objects = max(self.peak_objects, current["objects"])
        self.peak_element_references = max(self.peak_element_references, current["element_references"])
        return item

    @staticmethod
    def unpack(item):
        return item.tensor


def run(mode):
    theta = torch.tensor(.3, dtype=torch.float64, device="cpu", requires_grad=True)
    x = torch.arange(1, 16, dtype=torch.float64, device="cpu").reshape(5, 3) / 10
    target = torch.cos(x)
    tracker = Tracker()
    trace = []

    def record(label):
        trace.append({"event": label, **tracker.current()})

    def loss_rows(start, end):
        return (torch.sin(theta * x[start:end]) - target[start:end]).square()

    with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
        if mode == "full":
            loss = loss_rows(0, 5).mean()
            reported_loss = loss.item()
            record("forward_all")
            loss.backward()
            record("backward_all")
        elif mode == "chunk_forward_concat":
            pieces = []
            for start, end in ((0, 2), (2, 4), (4, 5)):
                pieces.append(loss_rows(start, end))
                record(f"forward_{start}_{end}")
            loss = torch.cat(pieces, dim=0).mean()
            reported_loss = loss.item()
            record("concatenated")
            loss.backward()
            record("backward_all")
        else:
            assert mode == "chunk_immediate_backward"
            reported_loss = 0.
            for start, end in ((0, 2), (2, 4), (4, 5)):
                loss = loss_rows(start, end).mean() * ((end - start) / 5)
                reported_loss += loss.item()
                record(f"forward_{start}_{end}")
                loss.backward()
                record(f"backward_{start}_{end}")
                assert tracker.current() == {"objects": 0, "element_references": 0}
    assert tracker.current() == {"objects": 0, "element_references": 0}
    return {"loss": reported_loss, "theta_gradient": theta.grad.item(),
            "peak_saved_objects": tracker.peak_objects,
            "peak_saved_element_references": tracker.peak_element_references, "trace": trace}


def main():
    torch.set_num_threads(1)
    results = {mode: run(mode) for mode in
               ("full", "chunk_forward_concat", "chunk_immediate_backward")}
    for result in results.values():
        assert abs(result["loss"] - results["full"]["loss"]) < 1e-12
        assert abs(result["theta_gradient"] - results["full"]["theta_gradient"]) < 1e-12
    delayed = results["chunk_forward_concat"]
    immediate = results["chunk_immediate_backward"]
    assert delayed["peak_saved_element_references"] > immediate["peak_saved_element_references"]
    assert delayed["trace"][0]["element_references"] < delayed["trace"][2]["element_references"]
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "torch_git": torch.version.git_version,
                      "saved_tensors_hooks_source_sha256": hashlib.sha256(
                          inspect.getsource(torch.autograd.graph.saved_tensors_hooks).encode()).hexdigest(),
                      "settings": {"device": "cpu", "dtype": "float64", "theta": .3,
                                   "steps": 5, "batch": 3, "chunk_sizes": [2, 2, 1],
                                   "objective": "mean((sin(theta*x)-cos(x))**2)",
                                   "x": "arange(1,16).reshape(5,3)/10", "tolerance": 1e-12},
                      "results": results,
                      "scope": "Three native CPU autograd traces using pack/unpack hooks and weakly held "
                      "detached wrappers. Counts are live graph-owned saved objects and their numel references, "
                      "not unique storage, RAM, allocator, GPU memory or graph-node counts. Correct chunk "
                      "weights are controls, not a new UniRL loss test. No optimizer step, replay/model, "
                      "BF16, retain_graph, checkpointing, distributed communication or performance validation."}, indent=2))


if __name__ == "__main__":
    main()
