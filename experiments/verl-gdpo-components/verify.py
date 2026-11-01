"""#8424: original GDPO math/helpers, explicit own data carrier and adapter.

python verify.py /path/to/pre-cached/verl-git
No full V1 entry, DataProto/TensorDict, TransferQueue, original tests or training.
"""

import __future__
import ast
from collections import defaultdict
import copy
from enum import Enum
import hashlib
import json
import os
import platform
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

HEAD = "8e4571bec0656a0c68394f3c9fd0dd7b7909acd3"
BASE = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
CORE = "verl/trainer/ppo/core_algos.py"
TF = "verl/utils/torch_functional.py"
TRAINER = "verl/trainer/ppo/v1/trainer_base.py"
UTILS = "verl/trainer/ppo/v1/utils.py"
KEYS = ["acc", "fmt"]


class Carrier:
    """Own row-selection seam, not original DataProto or TensorDict."""
    def __init__(self, batch, non_tensor_batch):
        self.batch, self.non_tensor_batch = batch, non_tensor_batch

    def __len__(self):
        return self.batch["response_mask"].shape[0]

    def select_idxs(self, indices):
        return Carrier({key: value[indices].clone() for key, value in self.batch.items()},
                       {key: value[indices].copy() for key, value in self.non_tensor_batch.items()})


def main(cache):
    hashes = {}

    def read(ref, path):
        raw = subprocess.check_output(["git", "-C", cache, "show", f"{ref}:{path}"],
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
        return ast.parse(raw)

    def execute(nodes, ns):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_ast", "exec", flags=__future__.annotations.compiler_flag), ns)

    def named(tree, name):
        return next(node for node in tree.body if getattr(node, "name", None) == name)

    merge_base = subprocess.check_output(["git", "-C", cache, "merge-base", HEAD, BASE],
                                        env=dict(os.environ, GIT_NO_LAZY_FETCH="1")).decode().strip()
    assert merge_base == BASE
    functional = {"torch": torch}
    tree = read(HEAD, TF)
    execute([named(tree, name) for name in ("masked_sum", "masked_mean", "masked_var", "masked_whiten")], functional)
    ns = dict(torch=torch, np=np, defaultdict=defaultdict, Enum=Enum, verl_F=SimpleNamespace(**functional))
    core = read(HEAD, CORE)
    execute([named(core, "AdvantageEstimator")], ns)
    for name in ("compute_grpo_outcome_advantage", "compute_gdpo_outcome_advantage"):
        node = named(core, name)
        node.decorator_list = []
        execute([node], ns)
    trainer = named(read(HEAD, TRAINER), "PPOTrainer")
    methods = [node for node in trainer.body if getattr(node, "name", None) in {"_pop_gdpo_rewards", "_lengths_to_mask"}]
    assert len(methods) == 2
    execute([ast.ClassDef(name="Helpers", bases=[], keywords=[], body=methods, decorator_list=[])], ns)
    helper = ns["Helpers"]
    config = {"gdpo_reward_keys": KEYS, "gdpo_reward_weights": [1., .3]}
    components = {"acc": np.array([0., 1., 2., 2., 4., 0.]),
                  "fmt": np.array([1., 0., 0., 0., 2., 1.])}
    extra = [{"reward_extra_info": {key: components[key][i] for key in KEYS}} for i in range(6)]
    extracted = []
    for layout in (copy.deepcopy(components), {"extra_fields": copy.deepcopy(extra)}):
        values = helper._pop_gdpo_rewards(layout, KEYS)
        assert all(np.array_equal(values[key], components[key]) for key in KEYS)
        assert not layout
        extracted.append({key: values[key].tolist() for key in KEYS})
    none_layout = dict(copy.deepcopy(components), extra_fields=None)
    none_values = helper._pop_gdpo_rewards(none_layout, KEYS)
    assert all(np.array_equal(none_values[key], components[key]) for key in KEYS) and not none_layout
    missing = copy.deepcopy(extra)
    del missing[2]["reward_extra_info"]["fmt"]
    missing_values = helper._pop_gdpo_rewards({"extra_fields": missing}, KEYS)
    assert set(missing_values) == {"acc"}
    priority = dict(copy.deepcopy(components), extra_fields=[{} for _ in range(6)])
    assert helper._pop_gdpo_rewards(priority, KEYS) == {} and set(priority) == set(KEYS)

    lengths = torch.tensor([3, 4, 2, 5, 3, 4])
    attention = helper._lengths_to_mask(lengths, 5)
    mask = attention.clone()
    mask[1, 1] = 0  # tool/context inside full response
    mask[3, 3:5] = 0  # tool/context at the end, not padding
    uid = np.array([0, 0, 0, 1, 1, 1], dtype=object)
    placement = []
    grpo = ns["compute_grpo_outcome_advantage"]

    def record_component(*args, **kwargs):
        placement.append(kwargs["token_level_rewards"].clone())
        return grpo(*args, **kwargs)

    ns["compute_grpo_outcome_advantage"] = record_component
    gdpo = ns["compute_gdpo_outcome_advantage"]

    def estimate(data, **kwargs):
        advantages, returns = gdpo(data.batch["token_level_rewards"], data.batch["response_mask"],
                                   index=data.non_tensor_batch["uid"], config=config,
                                   non_tensor_batch=data.non_tensor_batch, batch=data.batch)
        data.batch.update(advantages=advantages, returns=returns)
        return data

    def data():
        rewards = torch.zeros(6, 5)
        rewards[torch.arange(6), lengths-1] = torch.tensor(components["acc"]+components["fmt"], dtype=torch.float32)
        return Carrier(dict(token_level_rewards=rewards, response_mask=mask.clone(),
                            prompts=torch.zeros(6, 0, dtype=torch.long), attention_mask=attention.clone()),
                       dict(uid=uid.copy(), **copy.deepcopy(components)))

    response_only = estimate(data()).batch["advantages"].clone()
    for index, key in enumerate(KEYS):
        expected = torch.zeros(6, 5)
        expected[torch.arange(6), lengths-1] = torch.tensor(components[key], dtype=torch.float32)
        assert torch.equal(placement[index], expected)
    padded = data()
    padded.batch["prompts"] = torch.zeros(6, 4, dtype=torch.long)
    padded.batch["attention_mask"] = torch.cat((torch.ones(6, 4, dtype=torch.long), attention), dim=1)
    torch.testing.assert_close(estimate(padded).batch["advantages"], response_only, rtol=0, atol=0)
    assert torch.equal(response_only[mask == 0], torch.zeros_like(response_only[mask == 0]))
    errors = []
    for kind in ("missing_prompts", "missing_component"):
        bad = data()
        if kind == "missing_prompts":
            bad.batch.pop("prompts")
        else:
            bad.non_tensor_batch = dict(uid=uid.copy(), **missing_values)
        try:
            estimate(bad)
            raise RuntimeError("expected failure was not raised")
        except (KeyError, AssertionError) as error:
            errors.append(dict(case=kind, exception=type(error).__name__, message=str(error)))

    sessions = {}
    row_indices = [0, 0, 1, 2, 3, 4, 5]  # session 0 emits one extra intermediate
    batch_keys = ["u_with_underscore_0_0", "u_with_underscore_0_9", "u_with_underscore_1_0",
                  "u_with_underscore_2_0", "v_0_0", "v_1_0", "v_2_0"]
    expected = response_only[row_indices]
    for ref in (BASE, HEAD):
        local = dict(torch=torch, core_algos=SimpleNamespace(AdvantageEstimator=ns["AdvantageEstimator"]),
                     compute_advantage=estimate)
        execute([named(read(ref, UTILS), "compute_advantage_for_multi_trajectories")], local)
        actual = local["compute_advantage_for_multi_trajectories"](
            data().select_idxs(row_indices), batch_keys=batch_keys, adv_estimator="gdpo", config=config)
        gap = float((actual.batch["advantages"]-expected).abs().max())
        if ref == HEAD:
            torch.testing.assert_close(actual.batch["advantages"], expected, rtol=0, atol=0)
            enum_result = local["compute_advantage_for_multi_trajectories"](
                data().select_idxs(row_indices), batch_keys=batch_keys,
                adv_estimator=ns["AdvantageEstimator"].GDPO, config=config)
            torch.testing.assert_close(enum_result.batch["advantages"], expected, rtol=0, atol=0)
            permutation = [4, 2, 0, 6, 1, 5, 3]
            permuted = local["compute_advantage_for_multi_trajectories"](
                data().select_idxs(row_indices).select_idxs(permutation),
                batch_keys=[batch_keys[i] for i in permutation], adv_estimator="gdpo", config=config)
            torch.testing.assert_close(permuted.batch["advantages"], expected[permutation], rtol=1e-6, atol=1e-6)
        else:
            assert gap > .01
        sessions[ref] = dict(max_abs_gap_to_final_session_reference=gap,
                             first_token_advantages=actual.batch["advantages"][:, 0].tolist())

    print(json.dumps(dict(head=HEAD, base=BASE, actual_merge_base=merge_base, python=platform.python_version(),
                         torch=torch.__version__, numpy=np.__version__, source_sha256=hashes,
                         own_layout_controls=extracted, own_missing_key_errors=errors,
                         own_extra_fields_priority="Non-None extra_fields wins; incomplete rows omit whole key, no per-key fallback.",
                         own_none_extra_fields_uses_top_level=True,
                         own_full_response_lengths=lengths.tolist(), own_loss_token_counts=mask.sum(1).tolist(),
                         own_component_positions_equal_full_response_end=True,
                         own_padded_response_only_exact_equal=True, own_permuted_session_control=True,
                         own_string_enum_equivalence=True,
                         own_multi_output_controls=sessions,
                         scope="Original complete GRPO/GDPO and four masked helpers, two original V1 helpers, "
                         "two original session functions via AST (registry decorators removed). Own plain dict/NumPy "
                         "layouts, row-selecting Carrier and compute_advantage adapter directly to GDPO; no original "
                         "tests, DataProto/TensorDict, full _compute_advantage, TQ, KL/IS, models or training."), indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
