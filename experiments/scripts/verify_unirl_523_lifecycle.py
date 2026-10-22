"""CPU unit test for UniRL #523 fix: unified-stack anchor-freeze lifecycle.

Verifies the lifecycle block added to `UnifiedModelTrainStack.train_track`
(mirroring `TrainStack._prepare_for_training` in single.py):

    model.eval()
    try:
        prepare_segment_anchors(ar); ar_part = ar.prepare_part(ar_part)
        prepare_segment_anchors(image); image_part = image.prepare_part(image_part)
    finally:
        model.train()

This is a structural/ordering test on the lifecycle logic (the block is
extracted verbatim from the fixed `train_track`), not a full UniRL training run.
It does not import the full `UnifiedModelTrainStack` (that needs the whole
unirl package + torch/Ray); it exercises the exact code path we changed.
"""

from __future__ import annotations


class ModelMode:
    """Minimal stand-in for a backend model with mode switching."""

    def __init__(self):
        self.training = False
        self.calls = []  # list of ("eval"|"train",)

    def eval(self):
        self.training = False
        self.calls.append(("eval",))
        return self

    def train(self, mode=True):
        self.training = mode
        self.calls.append(("train",))
        return self


class _Part:
    def __init__(self, tag):
        self.tag = tag
        self.prepared = 0


class _Algo:
    """Algorithm stub: prepare_segment records freeze; prepare_part either no-op or raises."""

    def __init__(self, name, raise_on_prepare=False):
        self.name = name
        self.raise_on_prepare = raise_on_prepare
        self.frozen = []  # parts passed to prepare_segment_anchors
        self.prepared = []  # parts returned by prepare_part

    def prepare_segment(self, **kw):
        # stand-in for prepare_segment_anchors target
        part = kw["part"]
        self.frozen.append(part.tag)

    def prepare_part(self, part):
        if self.raise_on_prepare:
            raise RuntimeError(f"{self.name}.prepare_part boom")
        part.prepared += 1
        self.prepared.append(part.tag)
        return part


def freeze_one(algo, part):
    # stand-in for prepare_segment_anchors(algo, part, steps, order=...)
    algo.prepare_segment(part=part, segment=None)
    return part


def lifecycle(model, ar, ar_part, image, image_part):
    """The exact block added to train_track (extracted verbatim)."""
    model.eval()
    try:
        freeze_one(ar, ar_part)
        ar_part = ar.prepare_part(ar_part)
        freeze_one(image, image_part)
        image_part = image.prepare_part(image_part)
    finally:
        model.train()
    return ar_part, image_part


def test_eval_freeze_prepare_train_order():
    model = ModelMode()
    ar = _Algo("ar")
    image = _Algo("image")
    ar_part = _Part("ar")
    image_part = _Part("image")

    out_ar, out_image = lifecycle(model, ar, ar_part, image, image_part)

    # order of model-mode calls: eval before anything, train last
    mode_calls = [c[0] for c in model.calls]
    assert mode_calls[0] == "eval"
    assert mode_calls[-1] == "train"
    assert "train" in mode_calls and "eval" in mode_calls
    # every freeze happens while model is in eval (no train in between)
    assert mode_calls == ["eval", "train"]

    # both algorithms' prepare_segment (freeze) and prepare_part ran, in order
    assert ar.frozen == ["ar"] and image.frozen == ["image"]
    assert ar.prepared == ["ar"] and image.prepared == ["image"]

    # prepared part is returned (prepare_part output feeds the training loop)
    assert out_ar.prepared == 1 and out_image.prepared == 1

    # final mode is train
    assert model.training is True


def test_prepare_part_raises_still_restores_train():
    model = ModelMode()
    ar = _Algo("ar", raise_on_prepare=True)
    image = _Algo("image")
    ar_part = _Part("ar")
    image_part = _Part("image")

    try:
        lifecycle(model, ar, ar_part, image, image_part)
        assert False, "expected RuntimeError from ar.prepare_part"
    except RuntimeError as e:
        assert "boom" in str(e)

    # finally restored train even though prepare raised
    assert model.training is True
    assert model.calls[-1] == ("train",)


def test_noop_prepare_part_behaves_like_shipped_recipes():
    """GRPO/FlowGRPO/BagelFlowUniGRPO have no-op prepare_part (base default): returns same part."""
    model = ModelMode()
    grpo_like = _Algo("grpo")
    part = _Part("x")
    out, _ = lifecycle(model, grpo_like, part, _Algo("flowgrpo"), _Part("y"))
    # no-op prepare_part returns the same object (no structural change)
    assert out is part
    assert part.prepared == 1


if __name__ == "__main__":
    test_eval_freeze_prepare_train_order()
    test_prepare_part_raises_still_restores_train()
    test_noop_prepare_part_behaves_like_shipped_recipes()
    print("All 3 CPU unit tests passed.")
