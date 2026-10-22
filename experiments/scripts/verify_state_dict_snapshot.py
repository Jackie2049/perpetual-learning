#!/usr/bin/env python3
"""Check live state_dict aliases against isolated CPU snapshots.

Python >=3.10 and torch. Real scalar Parameter, persistent buffer and SGD;
manual gradients, own BytesIO payload, no device discovery or disk checkpoint.
Run: python experiments/scripts/verify_state_dict_snapshot.py
"""
import copy
import hashlib
import inspect
import io
import sys

sys.dont_write_bytecode = True
import torch


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64, device="cpu"))
        self.register_buffer("seen", torch.tensor(0, dtype=torch.int64, device="cpu"))


def check_model(state, weight, seen):
    torch.testing.assert_close(state["weight"], torch.tensor(weight, dtype=torch.float64, device="cpu"))
    assert state["seen"].item() == seen


def momentum(state):
    return state["state"][state["param_groups"][0]["params"][0]]["momentum_buffer"]


def main():
    print("Python", sys.version.split()[0], "torch", torch.__version__, "torch git", torch.version.git_version)
    for method in (torch.nn.Module.state_dict, torch.nn.Module._save_to_state_dict,
                   torch.optim.Optimizer.state_dict):
        print(method.__qualname__, "source-sha256=" + hashlib.sha256(inspect.getsource(method).encode()).hexdigest())

    model = Tiny()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, foreach=False)

    def step(gradient):
        model.weight.grad = torch.tensor(gradient, dtype=torch.float64, device="cpu")
        optimizer.step()
        with torch.no_grad():
            model.seen.add_(1)

    step(1.0)
    raw = model.state_dict()
    shallow = dict(raw)
    detached = {key: value.detach() for key, value in raw.items()}
    cpu_only = {key: value.cpu() for key, value in raw.items()}
    deep = copy.deepcopy(raw)
    cloned = {key: value.clone() for key, value in raw.items()}
    opt_raw = optimizer.state_dict()
    opt_deep = copy.deepcopy(opt_raw)
    for key, live in (("weight", model.weight), ("seen", model.seen)):
        assert raw[key] is not live and not raw[key].requires_grad
        assert shallow[key] is raw[key] and cpu_only[key] is raw[key]
        for state in (raw, shallow, detached, cpu_only):
            assert state[key].data_ptr() == live.data_ptr()
        for state in (deep, cloned):
            assert state[key].data_ptr() != live.data_ptr()
    assert momentum(opt_raw) is optimizer.state[model.weight]["momentum_buffer"]
    assert momentum(opt_deep).data_ptr() != momentum(opt_raw).data_ptr()
    check_model(raw, 0.9, 1)

    # Serialization finishes BEFORE the next mutation; no concurrent writer.
    with io.BytesIO() as stream:
        torch.save({"model": raw, "optimizer": opt_raw}, stream)
        step(2.0)
        for label, state in (("raw", raw), ("dict", shallow), ("detach", detached), ("cpu", cpu_only)):
            check_model(state, 0.61, 2)
            print(label, "after second step: weight=0.61 seen=2 (live storage)")
        for label, state in (("deepcopy", deep), ("clone", cloned)):
            check_model(state, 0.9, 1)
            print(label, "after second step: weight=0.9 seen=1 (isolated storage)")
        torch.testing.assert_close(momentum(opt_raw), torch.tensor(2.9, dtype=torch.float64, device="cpu"))
        assert momentum(opt_deep).item() == 1.0
        stream.seek(0)
        restored = torch.load(stream, map_location="cpu", weights_only=True)
    check_model(restored["model"], 0.9, 1)
    assert momentum(restored["optimizer"]).item() == 1.0
    for key in raw:
        assert restored["model"][key].data_ptr() != raw[key].data_ptr()
    assert momentum(restored["optimizer"]).data_ptr() != momentum(opt_raw).data_ptr()
    print("SGD raw momentum=2.9; deepcopy and completed serialization momentum=1")
    print("completed serialization: weight=0.9 seen=1, independent of later updates")

    shallow["weight"] = torch.tensor(99.0, dtype=torch.float64, device="cpu")
    check_model(model.state_dict(), 0.61, 2)
    raw["seen"].add_(10)
    check_model(model.state_dict(), 0.61, 12)
    for state in (deep, cloned, restored["model"]):
        check_model(state, 0.9, 1)
    print("dictionary rebinding leaves parameter unchanged; in-place alias write changes live buffer to 12")
    print("All CPU parameter/buffer/SGD snapshot observations passed; not original upstream tests")
    print("NOT RUN: disk recovery/atomicity, async/concurrent staging, distributed checkpoint, model training or GPU/NPU")


if __name__ == "__main__":
    main()
