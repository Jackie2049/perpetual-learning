"""Real CPU AdamW parameter-order controls; no model checkpoint files or distributed framework."""

import copy
import hashlib
import inspect
import json
import platform

import torch


class Pair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64, device="cpu"))
        self.right = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64, device="cpu"))


def optimizer(model, names):
    return torch.optim.AdamW([{"params": [getattr(model, name) for name in names], "param_names": list(names)}],
                             lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.,
                             foreach=False, fused=False, capturable=False)


def snapshot(model, optim):
    return {name: {"parameter": parameter.item(), "step": optim.state[parameter]["step"].item(),
                   "exp_avg": optim.state[parameter]["exp_avg"].item(),
                   "exp_avg_sq": optim.state[parameter]["exp_avg_sq"].item()}
            for name, parameter in model.named_parameters()}


model = Pair()
original = optimizer(model, ["left", "right"])
model.left.grad = torch.tensor(1., dtype=torch.float64, device="cpu")
model.right.grad = torch.tensor(-2., dtype=torch.float64, device="cpu")
original.step()
model_state = copy.deepcopy(model.state_dict())
optimizer_state = copy.deepcopy(original.state_dict())
source_before = snapshot(model, original)
results = {}
for case in ("same_order", "reversed_order", "renamed_metadata_only", "explicitly_aligned_ids"):
    restored = Pair()
    restored.load_state_dict(copy.deepcopy(model_state), strict=True)
    for name, value in restored.state_dict().items():
        assert torch.equal(value, model_state[name])
    names = ["left", "right"] if case == "same_order" else ["right", "left"]
    current = optimizer(restored, names)
    payload = copy.deepcopy(optimizer_state)
    if case in ("renamed_metadata_only", "explicitly_aligned_ids"):
        group = payload["param_groups"][0]
        name_to_id = dict(zip(group["param_names"], group["params"], strict=True))
        group["param_names"] = names
        if case == "explicitly_aligned_ids":
            # Only this tiny unique-name single-group example; not a migration utility.
            group["params"] = [name_to_id[name] for name in names]
    current.load_state_dict(payload)
    before = snapshot(restored, current)
    for parameter in restored.parameters():
        parameter.grad = torch.zeros_like(parameter)
    current.step()
    after = snapshot(restored, current)
    if case in ("same_order", "explicitly_aligned_ids"):
        for name in names:
            assert before[name] == source_before[name]
    else:
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert before["left"][key] == source_before["right"][key]
            assert before["right"][key] == source_before["left"][key]
    results[case] = {"actual_parameter_order": names,
                     "loaded_param_names": current.param_groups[0]["param_names"],
                     "before_zero_grad_step": before, "after_zero_grad_step": after}
assert results["same_order"]["after_zero_grad_step"] == results["explicitly_aligned_ids"]["after_zero_grad_step"]
assert results["reversed_order"]["after_zero_grad_step"] == results["renamed_metadata_only"]["after_zero_grad_step"]
assert results["same_order"]["after_zero_grad_step"] != results["reversed_order"]["after_zero_grad_step"]
assert snapshot(model, original) == source_before
try:
    optimizer(Pair(), ["left"]).load_state_dict(copy.deepcopy(optimizer_state))
except ValueError as error:
    guard = str(error)
else:
    raise AssertionError("Different parameter count should be rejected")
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version,
                  "optimizer_load_state_dict_sha256": hashlib.sha256(inspect.getsource(torch.optim.Optimizer.load_state_dict).encode()).hexdigest(),
                  "settings": {"dtype": "float64", "lr": 0.1, "betas": [0.9, 0.999], "eps": 1e-8,
                               "weight_decay": 0., "foreach": False, "fused": False, "capturable": False,
                               "warmup_gradients": {"left": 1., "right": -2.},
                               "resume_gradients": {"left": 0., "right": 0.}},
                  "saved_ids": optimizer_state["param_groups"][0]["params"],
                  "source_after_warmup": source_before, "results": results, "different_count_guard": guard,
                  "scope": "Four native CPU AdamW load/step traces plus one native size guard. Model parameters are first restored by name; optimizer states are deep-copied in memory. Explicit ID alignment is a local single-group reference, not a general pre-hook or framework fix. No serialization/storage, scheduler, framework checkpoint manager, FSDP/DCP/ZeRO, AMP, GPU/NPU, model training or performance validation."}, indent=2))
