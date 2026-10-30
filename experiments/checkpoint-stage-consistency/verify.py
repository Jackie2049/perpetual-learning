"""Three controlled CPU Adam staging schedules; own temporary checkpoint files only."""

import copy
import hashlib
import inspect
import json
import platform
import tempfile
import threading
from pathlib import Path

import torch


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64, device="cpu"))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.int64, device="cpu"))


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=.1, betas=(.9, .999), eps=1e-8,
                            foreach=False, fused=False, capturable=False)


def update(model, optimizer, gradient):
    model.weight.grad = torch.tensor(gradient, dtype=torch.float64, device="cpu")
    optimizer.step()
    with torch.no_grad():
        model.updates.add_(1)


def describe(model_state, optimizer_state):
    state = optimizer_state["state"][optimizer_state["param_groups"][0]["params"][0]]
    return {"weight": model_state["weight"].item(), "model_updates": model_state["updates"].item(),
            "adam_step": state["step"].item(), "exp_avg": state["exp_avg"].item(),
            "exp_avg_sq": state["exp_avg_sq"].item()}


def run(case):
    model = Tiny()
    optimizer = make_optimizer(model)
    update(model, optimizer, 1.)
    step_one = describe(model.state_dict(), optimizer.state_dict())
    staged = {"reported_step": 1}
    if case != "late_both":
        staged["model"] = copy.deepcopy(model.state_dict())
    if case == "stage_both_before_update":
        staged["optimizer"] = copy.deepcopy(optimizer.state_dict())
    ready, proceed = threading.Event(), threading.Event()
    errors = []

    with tempfile.TemporaryDirectory(prefix="pl-stage-consistency-") as root:
        checkpoint = Path(root) / "own-state.pt"

        def writer():
            try:
                ready.set()
                if not proceed.wait(10):
                    raise TimeoutError("writer was not released")
                if "model" not in staged:
                    staged["model"] = copy.deepcopy(model.state_dict())
                if "optimizer" not in staged:
                    staged["optimizer"] = copy.deepcopy(optimizer.state_dict())
                # Every schedule owns independent storage, including the inconsistent ones.
                for name, live in model.state_dict().items():
                    assert staged["model"][name].data_ptr() != live.data_ptr()
                live_state = optimizer.state[model.weight]
                saved_state = staged["optimizer"]["state"][0]
                for name in ("step", "exp_avg", "exp_avg_sq"):
                    assert saved_state[name].data_ptr() != live_state[name].data_ptr()
                torch.save(staged, checkpoint)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=writer, name="controlled-checkpoint-writer")
        worker.start()
        try:
            assert ready.wait(10), "writer did not reach its barrier"
            update(model, optimizer, -2.)
            step_two = describe(model.state_dict(), optimizer.state_dict())
        finally:
            proceed.set()
            worker.join(10)
        assert not worker.is_alive(), "writer did not finish"
        if errors:
            raise errors[0]
        loaded = torch.load(checkpoint, weights_only=True, map_location="cpu")
        captured = describe(loaded["model"], loaded["optimizer"])
        restored = Tiny()
        keys = restored.load_state_dict(loaded["model"], strict=True)
        assert not keys.missing_keys and not keys.unexpected_keys
        resumed_optimizer = make_optimizer(restored)
        resumed_optimizer.load_state_dict(loaded["optimizer"])
        assert describe(restored.state_dict(), resumed_optimizer.state_dict()) == captured
        update(restored, resumed_optimizer, -2.)
        resumed = describe(restored.state_dict(), resumed_optimizer.state_dict())

    if case == "stage_both_before_update":
        assert captured == step_one
        assert resumed == step_two
    elif case == "split_staging":
        assert captured["weight"] == step_one["weight"]
        assert captured["model_updates"] == 1 and captured["adam_step"] == 2
        assert captured["exp_avg"] == step_two["exp_avg"]
        assert resumed["weight"] != step_two["weight"]
    else:
        assert case == "late_both" and captured == step_two
        assert resumed["weight"] != step_two["weight"]
    return {"reported_step": loaded["reported_step"], "captured": captured,
            "after_one_resumed_update": resumed,
            "expected_after_reported_step_plus_one": step_two,
            "next_update_matches_step_two": resumed == step_two,
            "all_staged_tensor_storages_independent": True,
            "native_strict_model_and_optimizer_loads_succeeded": True}


def main():
    torch.set_num_threads(1)
    results = {case: run(case) for case in
               ("stage_both_before_update", "split_staging", "late_both")}
    hashes = {method.__qualname__: hashlib.sha256(inspect.getsource(method).encode()).hexdigest()
              for method in (torch.nn.Module.state_dict, torch.optim.Optimizer.state_dict,
                             torch.optim.Adam.step)}
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "torch_git": torch.version.git_version, "source_sha256": hashes,
                      "settings": {"device": "cpu", "dtype": "float64", "lr": .1,
                                   "betas": [.9, .999], "eps": 1e-8,
                                   "manual_gradients": [1., -2.], "resumed_gradient": -2.},
                      "results": results,
                      "scope": "Three native CPU Adam schedules with one real writer thread each. "
                      "Event barriers place the complete second update before background staging/writing. "
                      "All snapshots are deep-copied; this is not a live-alias test. Own temporary files, "
                      "weights_only load and one manual-gradient resumed update. No forward/backward, "
                      "RNG/data-loader/scheduler recovery, crash/fsync/atomic publication, distributed "
                      "framework, DCP, GPU/NPU, device discovery, performance or quality validation."}, indent=2))


if __name__ == "__main__":
    main()
