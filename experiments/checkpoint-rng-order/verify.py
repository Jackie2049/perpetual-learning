"""Tiny native CPU restart controls, not a framework checkpoint or DataLoader test.

Python 3.12, torch 2.9.1, numpy 2.0.2. Prints JSON; no file or device access.
"""

import copy
import hashlib
import inspect
import json
import platform
import random

import numpy as np
import torch


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed_initial():
    random.seed(123)
    np.random.seed(234)
    # CPU default generator only; avoid manual_seed's accelerator hooks.
    torch.random.default_generator.manual_seed(345)
    return torch.Generator(device="cpu").manual_seed(456)


def rng_snapshot(generator):
    return copy.deepcopy({"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch_cpu": torch.get_rng_state(), "sampling": generator.get_state()})


def restore_rng(state, generator, include_sampling=True):
    random.setstate(state["python"])
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].clone())
    if include_sampling:
        generator.set_state(state["sampling"].clone())


def rng_hashes(generator):
    state = rng_snapshot(generator)
    numpy_state = state["numpy"]
    return {"python": hashlib.sha256(repr(state["python"]).encode()).hexdigest(),
            "numpy": hashlib.sha256(numpy_state[1].tobytes() + repr(
                (numpy_state[0], *numpy_state[2:])).encode()).hexdigest(),
            "torch_cpu": tensor_hash(state["torch_cpu"]),
            "sampling": tensor_hash(state["sampling"])}


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 4, dtype=torch.float64, device="cpu"),
            torch.nn.Tanh(), torch.nn.Dropout(0.5),
            torch.nn.Linear(4, 1, dtype=torch.float64, device="cpu"))

    def forward(self, value):
        return self.net(value)


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01,
                             foreach=False, fused=False, capturable=False)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def parameter_hash(model, gradients=False):
    values = [p.grad if gradients else p for p in model.parameters()]
    return tensor_hash(torch.cat([value.detach().reshape(-1) for value in values]))


def train_step(model, optim, generator, step):
    # Fixed data pool; generator-driven fresh subsets, not a real DataLoader.
    features = torch.arange(24, dtype=torch.float64, device="cpu").reshape(8, 3) / 10
    targets = features.sum(1, keepdim=True) / 3
    indices = torch.randperm(8, generator=generator, device="cpu")[:4]
    gain = 0.8 + 0.4 * random.random()
    noise = np.random.uniform(-0.2, 0.2, size=(4, 1))
    optim.zero_grad(set_to_none=True)
    prediction = model(features[indices] * gain)
    loss = (prediction - targets[indices] - torch.from_numpy(noise)).square().mean()
    loss.backward()
    gradient = parameter_hash(model, gradients=True)
    optim.step()
    return {"step": step, "indices": indices.tolist(), "python_gain": gain,
            "numpy_noise": noise.reshape(-1).tolist(), "prediction": prediction.detach().reshape(-1).tolist(),
            "loss": loss.item(), "gradient_sha256": gradient,
            "parameters_sha256": parameter_hash(model)}


def main():
    torch.set_num_threads(1)
    generator = seed_initial()
    model = Tiny().train()
    optim = make_optimizer(model)
    warmup = [train_step(model, optim, generator, step) for step in range(1, 3)]
    checkpoint = copy.deepcopy({"model": model.state_dict(), "optimizer": optim.state_dict(),
                                "rng": rng_snapshot(generator)})
    checkpoint_before = copy.deepcopy(checkpoint)
    saved_model_hash = parameter_hash(model)
    saved_rng_hashes = rng_hashes(generator)
    reference = [train_step(model, optim, generator, step) for step in range(3, 6)]
    final_model = copy.deepcopy(model.state_dict())
    final_optimizer = copy.deepcopy(optim.state_dict())
    final_rng_hashes = rng_hashes(generator)
    results = {}
    for case in ("reseed_initial", "global_rng_only", "restore_before_constructor", "restore_after_load"):
        current_generator = seed_initial()
        if case == "restore_before_constructor":
            restore_rng(checkpoint["rng"], current_generator)
        current = Tiny().train()
        current_optimizer = make_optimizer(current)
        current.load_state_dict(copy.deepcopy(checkpoint["model"]), strict=True)
        current_optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        assert equal(current.state_dict(), checkpoint["model"])
        assert equal(current_optimizer.state_dict(), checkpoint["optimizer"])
        assert parameter_hash(current) == saved_model_hash
        if case in ("global_rng_only", "restore_after_load"):
            restore_rng(checkpoint["rng"], current_generator,
                        include_sampling=case == "restore_after_load")
        entry_hashes = rng_hashes(current_generator)
        entry_matches = {key: value == saved_rng_hashes[key] for key, value in entry_hashes.items()}
        trace = [train_step(current, current_optimizer, current_generator, step) for step in range(3, 6)]
        result = {"loaded_model_equal": True, "loaded_optimizer_equal": True,
                  "rng_matches_checkpoint_before_step": entry_matches,
                  "trace": trace, "trace_equal_reference": trace == reference,
                  "final_model_equal": equal(current.state_dict(), final_model),
                  "final_optimizer_equal": equal(current_optimizer.state_dict(), final_optimizer),
                  "final_rng_equal": rng_hashes(current_generator) == final_rng_hashes}
        if case == "restore_after_load":
            assert all(entry_matches.values())
            assert all(result[key] for key in ("trace_equal_reference", "final_model_equal",
                                               "final_optimizer_equal", "final_rng_equal"))
        else:
            assert not result["trace_equal_reference"]
            assert not result["final_model_equal"]
            assert not result["final_optimizer_equal"]
        if case == "global_rng_only":
            assert entry_matches == {"python": True, "numpy": True, "torch_cpu": True, "sampling": False}
            assert trace[0]["indices"] != reference[0]["indices"]
            assert trace[0]["python_gain"] == reference[0]["python_gain"]
            assert trace[0]["numpy_noise"] == reference[0]["numpy_noise"]
        if case == "restore_before_constructor":
            assert entry_matches == {"python": True, "numpy": True, "torch_cpu": False, "sampling": True}
            assert trace[0]["indices"] == reference[0]["indices"]
            assert trace[0]["python_gain"] == reference[0]["python_gain"]
            assert trace[0]["numpy_noise"] == reference[0]["numpy_noise"]
        results[case] = result
    assert equal(checkpoint, checkpoint_before)
    sources = {"get_rng_state": torch.random.get_rng_state, "set_rng_state": torch.random.set_rng_state,
               "linear_reset_parameters": torch.nn.Linear.reset_parameters,
               "dropout_forward": torch.nn.Dropout.forward}
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "numpy": np.__version__, "torch_git": torch.version.git_version,
                      "source_sha256": {key: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                                        for key, fn in sources.items()},
                      "settings": {"dtype": "float64", "threads": 1, "warmup_steps": 2, "resume_steps": 3,
                                   "seeds": {"python": 123, "numpy": 234, "torch_cpu": 345, "sampling": 456},
                                   "model": "Linear(3,4), Tanh, Dropout(0.5), Linear(4,1)",
                                   "optimizer": "AdamW lr=.01 weight_decay=.01 foreach/fused/capturable=False"},
                      "warmup": warmup, "saved_model_sha256": saved_model_hash,
                      "saved_rng_sha256": saved_rng_hashes, "reference": reference, "results": results,
                      "scope": "Four native single-process CPU restart controls after two real updates; three resumed updates. In-memory deep copies, not disk serialization. Fixed data pool and explicit torch.Generator subsets, not DataLoader, workers or a framework sampler. No distributed checkpoint manager, device discovery, GPU/NPU, framework training, cross-version reproducibility or performance claim."}, indent=2))


if __name__ == "__main__":
    main()
