"""Resolve #8446's two pins without installing packages or building sdists.

Fixed Git blobs; uv cache required in default offline mode. --online explicitly
allows PyPI metadata access. No Docker, pip install, imports of target packages,
GPU discovery, model or source builds. The lock is not the full Docker image.
Original Dockerfile/requirements: verl-project/verl, Apache-2.0.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess


BASE = "a08ea02a784d7d4982b1d81df36d188a476192bb"
HEAD = "f9fd7888400df03034482f2f68691046d32cb53d"
DOCKER = "docker/Dockerfile.stable.vllm"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    hashes = {}

    def source(revision, path):
        raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    docker = {label: source(rev, DOCKER) for label, rev in [("base", BASE), ("candidate", HEAD)]}
    assert docker["base"].replace("ARG TRANSFORMERS_VERSION=5.3.0", "ARG TRANSFORMERS_VERSION=5.5.3") == docker["candidate"]
    pins = {label: dict(re.findall(r"^ARG (\w+)=([^\n]+)$", raw, re.M)) for label, raw in docker.items()}
    requirements = source(BASE, "requirements.txt")
    tf_requirement = next(line for line in requirements.splitlines() if line.startswith("transformers"))
    env = dict(os.environ, UV_CACHE_DIR=str(args.cache_dir.resolve()), UV_PYTHON_DOWNLOADS="never", NO_COLOR="1")
    for key in ["UV_INDEX", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX", "UV_FIND_LINKS"]:
        env.pop(key, None)
    results = {}
    cases = [("base_manylinux_2_39", "base", "x86_64-manylinux_2_39", 1),
             ("candidate_manylinux_2_28", "candidate", "x86_64-manylinux_2_28", 1),
             ("candidate_manylinux_2_39", "candidate", "x86_64-manylinux_2_39", 0)]
    for name, label, target, expected_exit in cases:
        values = pins[label]
        inputs = f"vllm=={values['VLLM_VERSION']}\ntransformers=={values['TRANSFORMERS_VERSION']}\n"
        command = [args.uv, "--no-config", "pip", "compile", "-", "--python-version", values["PYTHON_VERSION"],
                   "--python-platform", target, "--default-index", "https://pypi.org/simple",
                   "--only-binary", ":all:", "--no-header", "--no-annotate"]
        if not args.online:
            command.append("--offline")
        run = subprocess.run(command, input=inputs, text=True, env=env,
                             capture_output=True, timeout=55)
        assert run.returncode == expected_exit, run.stderr[-6000:]
        if name.startswith("base"):
            assert "transformers>=5.5.3" in run.stderr
        elif expected_exit:
            assert "llguidance" in run.stderr and "no usable wheels" in run.stderr
        else:
            assert "torch==2.11.0" in run.stdout and "llguidance==1.7.6" in run.stdout
        # Keep diagnostic order but deduplicate repeated metadata warnings;
        # omit the variable resolver duration, not the dependency error.
        diagnostics = []
        seen_warnings = set()
        for line in run.stderr.splitlines():
            if re.match(r"Resolved \d+ packages in ", line):
                continue
            if line.startswith("WARN"):
                if line in seen_warnings:
                    continue
                seen_warnings.add(line)
            diagnostics.append(line)
        results[name] = {"input": inputs.splitlines(), "python": values["PYTHON_VERSION"],
                         "target": target, "exit_code": run.returncode,
                         "requirements": run.stdout.splitlines(), "diagnostic_lines": diagnostics}
    print(json.dumps({"base": BASE, "head": HEAD, "source_sha256": hashes,
        "host_python": platform.python_version(),
        "uv": subprocess.check_output([args.uv, "--version"], text=True).strip(),
        "binary_only": True, "index": "https://pypi.org/simple",
        "original_args": {label: {key: values[key] for key in ["PYTHON_VERSION", "TORCH_VERSION",
            "VLLM_VERSION", "TRANSFORMERS_VERSION", "VERL_VERSION"]} for label, values in pins.items()},
        "current_main_transformers_requirement_not_image_input": tf_requirement,
        "cases": results,
        "scope": "Two top-level pins only; no package installation/source build, full image/ARM/ABI/model/training validation"},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
