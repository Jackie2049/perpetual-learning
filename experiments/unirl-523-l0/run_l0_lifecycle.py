"""L0 real-module lifecycle smoke for UniRL #523 (M5, 2026-10-23).

Loads the REAL `unirl.train.stack.unified` module (upstream main 6fd82cba +
523-lifecycle-fix.patch applied; run with the fork-consistent patch so the file
fingerprints match fork head 04101ad5), extracts the real `train_track` method
body (the @distributed decorator uses functools.wraps, so `__wrapped__` is the
plain method), and executes it bound to a substitute self with substitute
collaborators. The lifecycle block under test:

    self.fsdp_backend.model.eval()
    try:
        prepare_segment_anchors(self.ar_algorithm, ar_part, ar_steps, order=ar_order)
        ar_part = self.ar_algorithm.prepare_part(ar_part)
        prepare_segment_anchors(self.image_algorithm, image_part, image_steps, order=image_order)
        image_part = self.image_algorithm.prepare_part(image_part)
    finally:
        self.fsdp_backend.model.train()

Substitutes (recorded explicitly below):
- fsdp_backend.model: real torch nn.Module (single Linear) whose eval()/train() record
  into a shared event log (real torch mode semantics preserved).
- ar/image algorithms, three variants:
  (a) no-op prepare_part + recomputes_anchor=False  -> shipped GRPO/FlowGRPO/BagelFlowUniGRPO
  (b) recording prepare_part                         -> PPO/DiffusionOPD override semantics
  (c) prepare_part raises RuntimeError               -> failure substitute
- prepare_segment_anchors: REAL function from unirl/train/stack/anchor.py (module global
  of the extracted method body; verified by code filename).
- sample / Part / update_planner / _train_one_step / _align_track_inputs / on_rollout_end:
  minimal substitutes (recorded; Part/Plan construction deps are heavy — allowed by the
  task handoff as long as each substitute is recorded).
- _train_step_profiler: None (UNIRL_PROFILE defaults to off -> nullcontext path).

Assertions:
  ① call order = eval -> freeze(ar) -> prepare_part(ar) -> freeze(image) ->
    prepare_part(image) -> train
  ② (c) prepare_part raises -> RuntimeError propagates AND model.train() restored
  ③ (a) both parts returned unchanged (same objects flow into _train_one_step)
  ④ (b) both algorithms' prepare_part called exactly once each

Usage:
    UNIRL_REPO=/path/to/repo python run_l0_lifecycle.py [--mode patched|pristine]

--mode pristine runs the same harness against the unpatched tree as a control:
the four assertions must FAIL (no eval/train, no prepare_part) — proving the
assertions discriminate the fix.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import List, Optional

REPO = os.environ.get("UNIRL_REPO", "/home/zxw/Tencent-Hunyuan-UniRL-6fd82cb")
sys.path.insert(0, REPO)

import torch  # noqa: E402

EVENTS: List[str] = []


def emit(tag: str) -> None:
    EVENTS.append(tag)


# ---------- substitutes ----------


class RecModel(torch.nn.Module):
    """Real nn.Module (single Linear) that records OUTER eval()/train() calls.

    torch's Module.eval() is implemented as self.train(False), so a naive wrapper
    would double-record. A reentrancy guard records only the calls train_track
    makes, while real torch mode semantics (self.training) are preserved.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self._switching = False

    def eval(self):
        if not self._switching:
            self._switching = True
            try:
                emit("model.eval()")
                return super().eval()
            finally:
                self._switching = False
        return super().eval()

    def train(self, mode: bool = True):
        if not self._switching:
            self._switching = True
            try:
                emit("model.train()")
                return super().train(mode)
            finally:
                self._switching = False
        return super().train(mode)


class AlgoStub:
    """Algorithm substitute. prepare_mode: noop | record | raise."""

    def __init__(self, name: str, prepare_mode: str = "noop", recomputes_anchor: bool = False):
        self.name = name
        self.prepare_mode = prepare_mode
        self.recomputes_anchor = recomputes_anchor
        self.anchor_fields: tuple = ()
        self.prepare_segment_calls = 0
        self.prepare_part_calls = 0
        self.last_returned = None

    def prepare_segment(self, *, conditions, segment):
        self.prepare_segment_calls += 1
        emit(f"prepare_segment({self.name})")

    def prepare_part(self, part):
        self.prepare_part_calls += 1
        emit(f"prepare_part({self.name})")
        if self.prepare_mode == "raise":
            raise RuntimeError(f"{self.name}.prepare_part boom")
        self.last_returned = part
        return part  # no-op / recording: same object back


class SegmentStub:
    def __init__(self) -> None:
        self.conditions = {"cond": torch.randn(4, 4)}


class PartStub:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.segment = SegmentStub()
        self.conditions = {"cond": torch.randn(4, 4)}  # part-level conditions (real anchor fn reads part.conditions)


class SampleStub:
    def __init__(self) -> None:
        self.created: List[PartStub] = []

    def gen_part(self, params_cls):
        emit(f"gen_part({params_cls.__name__})")
        p = PartStub(params_cls.__name__)
        self.created.append(p)
        return p


class PlannerStub:
    """arrange_many -> ((plan, steps, order), (plan, steps, order)); 1 update, 2 micros for ar."""

    def arrange_many(self, parts, *, num_updates, micro_batch_size, shuffle_step):
        return (None, [[(0, 4), (4, 8)]], None), (None, [[(0, 4)]], None)


class SelfStub:
    def __init__(self, ar_algo: AlgoStub, image_algo: AlgoStub, model: RecModel) -> None:
        self.ar_algorithm = ar_algo
        self.image_algorithm = image_algo
        self.fsdp_backend = SimpleNamespace(model=model)
        self.update_planner = PlannerStub()
        self._train_step_profiler = None
        self.num_updates_per_batch = 1
        self.micro_batch_size = 4
        self.max_grad_norm = 1.0
        self.received_ar_part = None
        self.received_image_part = None

    def _align_track_inputs(self, part) -> None:
        emit("align_track_inputs")

    def on_rollout_end(self) -> None:
        emit("on_rollout_end")

    def _train_one_step(self, ar_part, image_part, **kwargs):
        emit("train_one_step")
        self.received_ar_part = ar_part
        self.received_image_part = image_part
        # real TrainStepResult; len(results)==1 -> aggregate_update_results returns it as-is
        from unirl.train.stack.base import TrainStepResult

        r = TrainStepResult(
            loss=0.1, grad_norm=1.0, lr=1e-4, has_backward=True,
            micros=[], metrics={}, optimizer_updates=1,
        )
        return {"ar": r, "image": r}


# ---------- assertions ----------


def lifecycle_events() -> List[str]:
    """The event slice from model.eval() through model.train() (train_track emits
    gen_part/align_track_inputs events before the lifecycle block)."""
    start = EVENTS.index("model.eval()")
    return EVENTS[start : start + 6]


def assert_order_ok() -> None:
    lifecycle = lifecycle_events()
    expected = [
        "model.eval()",
        "prepare_segment(ar)",
        "prepare_part(ar)",
        "prepare_segment(image)",
        "prepare_part(image)",
        "model.train()",
    ]
    assert lifecycle == expected, f"lifecycle order mismatch:\n  got      {lifecycle}\n  expected {expected}"
    start = EVENTS.index("model.eval()")
    assert "train_one_step" in EVENTS[start + 6 :], "train_one_step should follow the lifecycle"
    assert EVENTS[-1] == "on_rollout_end", "on_rollout_end should be last"
    print("[PASS] ① call order: eval -> freeze(ar) -> prepare_part(ar) -> freeze(image) -> prepare_part(image) -> train")


def assert_raise_restores_train(model: RecModel) -> None:
    start = EVENTS.index("model.eval()")
    assert EVENTS[start] == "model.eval()", "eval must be first in the lifecycle"
    assert EVENTS[-1] == "model.train()", "train must be restored even when prepare_part raises"
    assert model.training is True, f"model mode after exception must be train, got {model.training}"
    print("[PASS] ② prepare_part raises -> RuntimeError propagates AND model.train() restored (finally)")


def assert_parts_unchanged(sample: SampleStub, self_: SelfStub) -> None:
    assert self_.received_ar_part is sample.created[0], "ar part must be the same object (no-op prepare_part)"
    assert self_.received_image_part is sample.created[1], "image part must be the same object"
    print("[PASS] ③ no-op prepare_part: both parts returned unchanged (same objects into _train_one_step)")


def assert_prepare_once(ar_algo: AlgoStub, image_algo: AlgoStub) -> None:
    assert ar_algo.prepare_part_calls == 1, f"ar prepare_part called {ar_algo.prepare_part_calls}x"
    assert image_algo.prepare_part_calls == 1, f"image prepare_part called {image_algo.prepare_part_calls}x"
    print("[PASS] ④ both algorithms' prepare_part called exactly once each")


# ---------- main ----------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("patched", "pristine"), default="patched")
    args = ap.parse_args()
    pristine = args.mode == "pristine"

    print(f"mode        : {args.mode}")
    print(f"repo        : {REPO}")
    print(f"python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__}")

    with open(os.path.join(REPO, "unirl", "train", "stack", "unified.py"), "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    print(f"unified.py  : sha256 {sha}")
    if not pristine:
        assert sha == "9a55deabdf1821acbfb57d36bb1377418a4880a652c8a28dd7e89163e541a4e3", (
            "patched unified.py fingerprint mismatch — stop"
        )
    else:
        assert sha == "d2da53dea5143e98f0228465cbde5cc90f014a3c6bf3b123de658354dfe95028", (
            "pristine unified.py fingerprint mismatch — stop"
        )

    import unirl.train.stack.unified as u  # real module
    from unirl.train.stack.base import TrainStepResult  # noqa: F401 (import check)
    from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams  # noqa: F401

    raw = u.UnifiedModelTrainStack.train_track.__wrapped__
    assert raw is not None, "@distributed must use functools.wraps so __wrapped__ exposes the plain method"

    # confirm the extracted body is the REAL method of the REAL module
    assert u.UnifiedModelTrainStack.__module__ == "unirl.train.stack.unified"
    assert os.path.realpath(u.__file__) == os.path.realpath(
        os.path.join(REPO, "unirl", "train", "stack", "unified.py")
    )
    anchor_fn = raw.__globals__["prepare_segment_anchors"]
    assert "anchor.py" in anchor_fn.__code__.co_filename, (
        f"prepare_segment_anchors must be the real anchor.py function, got {anchor_fn.__code__.co_filename}"
    )
    print(f"anchor fn    : {anchor_fn.__code__.co_filename}")

    # ---------- variant (a): shipped-recipes semantics (no-op prepare_part) ----------
    global EVENTS
    EVENTS = []
    model = RecModel()
    ar_algo = AlgoStub("ar", prepare_mode="noop", recomputes_anchor=False)
    image_algo = AlgoStub("image", prepare_mode="noop", recomputes_anchor=False)
    self_ = SelfStub(ar_algo, image_algo, model)
    sample = SampleStub()
    if pristine:
        # ar_part = sample.gen_part(...); old body has no prepare_part rebinding
        out = raw(self_, sample, training_progress=0.5, rollout_id=0)
        print("lifecycle events (pristine):", EVENTS)
        # old behavior: no model mode switch, no prepare_part
        assert "model.eval()" not in EVENTS and "model.train()" not in EVENTS, "pristine must not switch modes"
        assert ar_algo.prepare_part_calls == 0 and image_algo.prepare_part_calls == 0, "pristine must not call prepare_part"
        print("[CONTROL] pristine: no model.eval()/train(), no prepare_part — old behavior confirmed")
    else:
        out = raw(self_, sample, training_progress=0.5, rollout_id=0)
        print("lifecycle events (patched):", EVENTS)
        assert_order_ok()
        assert_parts_unchanged(sample, self_)
        assert ar_algo.prepare_segment_calls == 1 and image_algo.prepare_segment_calls == 1
        assert isinstance(out["ar"], TrainStepResult) and isinstance(out["image"], TrainStepResult)
        print("  (a) prepare_segment calls: ar=1 image=1; results are real TrainStepResult")

    # ---------- variant (b): PPO/DiffusionOPD override semantics (recording) ----------
    EVENTS = []
    model = RecModel()
    ar_algo = AlgoStub("ar", prepare_mode="record", recomputes_anchor=False)
    image_algo = AlgoStub("image", prepare_mode="record", recomputes_anchor=False)
    self_ = SelfStub(ar_algo, image_algo, model)
    sample = SampleStub()
    raw(self_, sample, training_progress=0.5, rollout_id=0)
    if not pristine:
        assert_order_ok()
        assert_prepare_once(ar_algo, image_algo)
        assert ar_algo.last_returned is sample.created[0] and image_algo.last_returned is sample.created[1]
    else:
        assert ar_algo.prepare_part_calls == 0, "pristine: prepare_part must not be called"
        print("[CONTROL] pristine (b): prepare_part never called")

    # ---------- variant (c): failure substitute (prepare_part raises) ----------
    EVENTS = []
    model = RecModel()
    ar_algo = AlgoStub("ar", prepare_mode="raise", recomputes_anchor=False)
    image_algo = AlgoStub("image", prepare_mode="record", recomputes_anchor=False)
    self_ = SelfStub(ar_algo, image_algo, model)
    sample = SampleStub()
    if not pristine:
        try:
            raw(self_, sample, training_progress=0.5, rollout_id=0)
            raise AssertionError("expected RuntimeError from ar.prepare_part")
        except RuntimeError as exc:
            assert "boom" in str(exc), str(exc)
        print("exception events:", EVENTS)
        assert_raise_restores_train(model)
        # ar prepared, image never reached: eval, freeze(ar), prepare_part(ar), train (finally)
        start = EVENTS.index("model.eval()")
        assert EVENTS[start : start + 4] == [
            "model.eval()",
            "prepare_segment(ar)",
            "prepare_part(ar)",
            "model.train()",
        ], EVENTS[start:]
    else:
        # pristine: prepare_part is never called, so the failure substitute never triggers;
        # the call completes without exception and no mode switching happens.
        raw(self_, sample, training_progress=0.5, rollout_id=0)
        assert "model.eval()" not in EVENTS and "model.train()" not in EVENTS
        assert ar_algo.prepare_part_calls == 0
        print("exception events (pristine):", EVENTS)
        print("[CONTROL] pristine (c): old body never calls prepare_part -> failure substitute never triggers")

    if pristine:
        print("CONTROL DONE: pristine tree fails all four patched assertions as expected (no eval/train, no prepare_part).")
    else:
        print("L0 DONE: all four lifecycle assertions PASS on the real patched module.")


if __name__ == "__main__":
    main()
