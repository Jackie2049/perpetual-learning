"""CPU PNG and request-shape controls on pinned original UniRL wire functions.

Requires Pillow; does not import UniRL, Torch, a processor, or SGLang.
"""

import argparse
import ast
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, List

import PIL
from PIL import Image

HEAD = "fe40529f3634cc51ee592064aeef0ce969bd7573"
REQUIRE = "unirl/config/require.py"
IMAGES = "unirl/rollout/engine/sglang/utils/images.py"
VLM = "unirl/rollout/engine/sglang/adapters/vlm.py"
READING = ("unirl/rollout/engine/sglang/adapters/base.py",
           "unirl/rollout/engine/sglang/utils/conversations.py", "unirl/data/data_source.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    sources, hashes = {}, {}
    for path in (REQUIRE, IMAGES, VLM, *READING):
        raw = subprocess.check_output(["git", "show", f"{HEAD}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        compile(ast.parse(raw), path, "exec")
        sources[path] = raw
        hashes[path] = hashlib.sha256(raw).hexdigest()
    namespace = {"Any": Any, "List": List}
    for path in (REQUIRE, IMAGES):
        exec(compile(sources[path], path, "exec"), namespace)
    function = next(n for n in ast.parse(sources[VLM]).body if isinstance(n, ast.FunctionDef)
                    and n.name == "_wire_image_data")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(function)], type_ignores=[])),
                 VLM, "exec"), namespace)
    wire = namespace["_wire_image_data"]
    images = [Image.new("RGB", (2, 1), (255, 0, 0)), Image.new("RGBA", (3, 2), (0, 255, 0, 71)),
              Image.new("L", (4, 1), 123)]

    def check(selected, candidates):
        encoded = wire(selected, parallel_samples=candidates)
        if isinstance(encoded, str):
            shape, flat = "scalar", [encoded]
        elif isinstance(encoded[0], list):
            assert len(encoded) == 1
            shape, flat = f"nested[1,{len(encoded[0])}]", encoded[0]
        else:
            shape, flat = f"flat[{len(encoded)}]", encoded
        assert len(flat) == len(selected)
        decoded = []
        for uri, original in zip(flat, selected, strict=True):
            assert uri.startswith("data:image/png;base64,")
            raw = base64.b64decode(uri.split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                assert image.format == "PNG" and image.mode == original.mode and image.size == original.size
                assert image.tobytes() == original.tobytes()
                decoded.append({"mode": image.mode, "size": list(image.size), "png_sha256": hashlib.sha256(raw).hexdigest()})
        return {"images": len(selected), "candidates": candidates, "wire_shape": shape, "decoded": decoded}

    cases = [check(images[:n], r) for n in (1, 3) for r in (1, 2, 5)]
    cases.append(check([images[2], images[0], images[1]], 2))
    cases.append(check([images[0], images[0]], 1))
    errors = []
    for selected, r in (([], 1), (images, 0), (images, -1)):
        try:
            wire(selected, parallel_samples=r)
        except ValueError as error:
            errors.append({"images": len(selected), "candidates": r, "error": str(error)})
        else:
            raise AssertionError("Expected original precondition rejection")
    print(json.dumps({"python": platform.python_version(), "pillow": PIL.__version__, "head": HEAD,
                      "source_sha256": hashes, "own_valid_controls": cases, "own_invalid_controls": errors,
                      "scope": "Eight own CPU wire controls with real in-memory PIL images, original PNG/base64 "
                      "encoder and require modules, and the complete unchanged _wire_image_data AST function. "
                      "Three original precondition rejections. Six full source inputs compile; remaining "
                      "adapter/data/conversation code read only. No original author harness, ImageSets/Part/Sample "
                      "construction, fake or real processor, Torch, SGLang request/backend, replay, model, "
                      "training, GPU/NPU or performance execution."}, indent=2))


if __name__ == "__main__":
    main()
