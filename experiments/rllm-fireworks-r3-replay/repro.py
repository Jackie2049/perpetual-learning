"""Reproduction: the routing matrices rLLM's Fireworks backend records (R3) do not
reach the Tinker-style training datum in a form the trainer can use.

Stage A — data path (real rllm classes, real tinker types):
  ModelOutput(routing_matrices=...)   <- what FireworksEngine attaches
    -> Step.from_model_output         <- what every workflow stores
    -> trajectory_to_datums(R3)       <- Tinker/Fireworks trainer conversion
  Checks (a) the matrices survive, (b) what type they have, (c) whether the
  trainer frame (`tinker.ModelInput`) is able to carry them at all.

Stage B — FireworksEngine end to end, stubbed sampler + stub tokenizer/parser,
  asserting its response parsing (logprobs.content[*].routing_matrix) surfaces
  the matrices and where they end up.

Exit 0 = at least one stage reproduces a defect.
Exit 1 = nothing reproduced; re-check the conclusion.
"""
from __future__ import annotations

import asyncio
import inspect
import sys

from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.trainer.tinker.transform import trajectory_to_datums
from rllm.types import Step, Trajectory

PROMPT = [101, 102, 103, 104]
RESPONSE = [201, 202, 203]
MATRICES = ["m0", "m1", "m2"]


def _step(mats: list[str] | None) -> Step:
    s = Step(
        prompt_ids=list(PROMPT),
        response_ids=list(RESPONSE),
        logprobs=[-0.1, -0.2, -0.3],
        routing_matrices=mats,
    )
    s.advantage = 1.0
    return s


def stage_a() -> bool:
    defect = False

    print("[A] router_replay=disabled ->",
          getattr(trajectory_to_datums(Trajectory(steps=[_step(None)]), router_replay=False)[0].model_input, "routing_matrices", None))
    d = trajectory_to_datums(Trajectory(steps=[_step(list(MATRICES))]), router_replay=True)[0]
    mi = d.model_input
    mats = getattr(mi, "routing_matrices", None)
    print(f"[A] R3 datum len={mi.length} routing_matrices={mats}")
    print(f"[A] datum input type = {type(mi).__module__}.{type(mi).__name__}")

    # (b) type: rLLM's verl path feeds (len, num_layers, topk) tensors; the Tinker
    # path hands the strings straight through.
    print(f"[A] element types     = {sorted({type(m).__name__ for m in (mats or [])})}")

    # (c) can the trainer frame carry the field at all?
    declared = getattr(type(mi), "model_fields", {})
    carrier = "routing_matrices" in declared
    print(f"[A] ModelInput declared fields = {list(declared)}; can carry routing_matrices = {carrier}")
    if not carrier:
        print("[A] DEFECT: tinker.ModelInput has no routing_matrices field; the attribute rLLM sets")
        print("      is a plain python attribute that is not part of the serialized model input")
        defect = True

    targets = list(d.loss_fn_inputs["target_tokens"].data)
    mask = list(d.loss_fn_inputs["mask"].data)
    aligned = sum(
        1 for i in range(len(MATRICES))
        if (mats or [None] * len(mi))[len(PROMPT) - 1 + i] == MATRICES[i] and targets[len(PROMPT) - 1 + i] == RESPONSE[i]
    )
    print(f"[A] target_tokens={targets} mask={mask} aligned_matches={aligned}/{len(MATRICES)}")
    return defect


class _StubSampler:
    async def async_completions_stream(self, prompt, max_tokens=1024, **kwargs):
        raw = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "raw_output": {"completion_token_ids": list(RESPONSE)},
                    "logprobs": {
                        "content": [
                            {"token": t, "logprob": -0.1 * (i + 1), "routing_matrix": f"m{i}"}
                            for i, t in enumerate(RESPONSE)
                        ]
                    },
                }
            ]
        }
        return raw, None


class _Tokenizer:
    name_or_path = "stub-model-abc"  # no family keyword -> base ChatTemplateParser
    eos_token_id = 2
    bos_token = "<s>"

    def encode(self, text, add_special_tokens=False):
        return list(PROMPT)

    def decode(self, ids, skip_special_tokens=True):
        return "<text>"

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        # Per-message rendering is concatenation-stable so the base parser's
        # equivalence self-check passes.
        return "".join(f"<{m['role']}>{m.get('content')}</{m['role']}>" for m in messages) + (
            "<gen>" if add_generation_prompt else ""
        )


def stage_b() -> bool:
    from rllm.engine.rollout.fireworks_engine import FireworksEngine

    sampler = _StubSampler()
    engine = FireworksEngine(
        tokenizer=_Tokenizer(),
        sampler=sampler,
        max_prompt_length=64,
        max_response_length=64,
        max_model_length=4096,
    )
    # The base ChatTemplateParser leaves parse_completion unimplemented; a real
    # tokenizer resolves to a family parser instead. Only content parsing is
    # stubbed here — routing-matrix extraction below runs the real code.
    engine.chat_parser.parse_completion = lambda ids: {"content": "ok", "reasoning": "", "tool_calls": []}
    out = asyncio.run(engine.get_model_response([{"role": "user", "content": "hi"}]))
    print(f"[B] engine ModelOutput.routing_matrices = {out.routing_matrices}")
    step = Step.from_model_output(out)
    step.advantage = 1.0
    datum = trajectory_to_datums(Trajectory(steps=[step]), router_replay=True)[0]
    mats = getattr(datum.model_input, "routing_matrices", None)
    print(f"[B] workflow Step.routing_matrices       = {step.routing_matrices}")
    print(f"[B] R3 datum routing_matrices            = {mats}")
    if out.routing_matrices != MATRICES:
        print("[B] engine did not surface the matrices in this stub run — re-check")
        return False
    if mats is None:
        print("[B] DEFECT: matrices captured by the engine never reach the datum")
        return True
    # Same carrier question as stage A: the field is not part of ModelInput.
    carrier = "routing_matrices" in getattr(type(datum.model_input), "model_fields", {})
    print(f"[B] datum ModelInput can carry routing_matrices = {carrier}")
    return not carrier


if __name__ == "__main__":
    a = stage_a()
    print()
    b = stage_b()
    print()
    print(f"stage A defect: {a}   stage B defect: {b}")
    sys.exit(0 if (a or b) else 1)
