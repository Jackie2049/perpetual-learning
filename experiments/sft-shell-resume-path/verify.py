"""Check fixed upstream SFT shell scripts with their isolated command substitutes.

Usage: python verify.py /path/to/verl-git
Needs pytest, Bash and cached Git blobs. Does not import verl, Torch or Hydra.
Only 12 selected upstream parameter cases are called directly; the eight
space-path cases use the upstream helper but deliberately omit Hydra parsing.
Source blobs are materialized verbatim through an in-memory tar, never patched.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile

import pytest


HEAD = "4a0679602e0a56b13a49e477e279cef5f03324aa"
BASE = "c76d46ea6474f9fc9837a469c2ab74b58d921bbc"
SCRIPT = "tests/special_e2e/sft/run_sft_engine.sh"
TEST = "tests/trainer/test_sft_engine_cleanup_on_cpu.py"


def main(repo: Path) -> dict:
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(repo), *args], env=env)

    hashes = {
        f"{rev}:{path}": hashlib.sha256(git("show", f"{rev}:{path}")).hexdigest()
        for rev, path in [(HEAD, SCRIPT), (HEAD, TEST), (BASE, SCRIPT)]
    }
    # Select exactly the read blobs: git archive can traverse unrelated missing
    # blobs in a partial clone. This does not create a checkout or run setup.
    def extract(root: Path, rev: str, *paths: str) -> None:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as output:
            for path in paths:
                assert path in {SCRIPT, TEST}
                content = git("show", f"{rev}:{path}")
                entry = tarfile.TarInfo(path)
                entry.size = len(content)
                entry.mode = 0o644
                output.addfile(entry, io.BytesIO(content))
        subprocess.run(["tar", "-x", "-C", str(root)], input=archive.getvalue(), check=True)

    result = {
        "python": platform.python_version(),
        "pytest": pytest.__version__,
        "bash": subprocess.check_output([shutil.which("bash"), "--version"], text=True).splitlines()[0],
        "sha256": hashes,
        "scope": "12 upstream parameter cases plus 8 local shell-only space-path cases per script; no Hydra or training",
        "scripts": {},
    }
    for label, revision in [("base", BASE), ("head", HEAD)]:
        with tempfile.TemporaryDirectory(prefix="pl-sft-resume-") as directory:
            root = Path(directory).resolve()
            extract(root, HEAD, SCRIPT, TEST)
            if revision != HEAD:
                extract(root, revision, SCRIPT)
            assert hashlib.sha256((root / SCRIPT).read_bytes()).hexdigest() == hashes[f"{revision}:{SCRIPT}"]
            subprocess.run([shutil.which("bash"), "-n", str(root / SCRIPT)], check=True)
            ns = runpy.run_path(str(root / TEST))
            groups = {}
            empty_array_observations = {}
            counter = itertools.count()

            def case_dir() -> Path:
                path = root / f"case-{next(counter)}"
                path.mkdir()
                return path

            def check(group: str, name: str, invoke) -> None:
                row = groups.setdefault(group, {"passed": [], "failed": []})
                try:
                    invoke()
                except AssertionError:
                    row["failed"].append(name)
                else:
                    row["passed"].append(name)

            for mode, resume_path, keep in itertools.product(["spmd", "ray"], [None, ""], [None, "False"]):
                args = (case_dir(), mode, resume_path, keep)
                check("original_missing_path", f"{mode}/{resume_path!r}/{keep!r}",
                      lambda args=args: ns["test_resume_path_requires_path_before_training_or_cleanup"](*args))
            for mode, resume_mode in itertools.product(["spmd", "ray"], ["disable", "auto"]):
                args = (case_dir(), mode, resume_mode)
                check("original_other_modes", f"{mode}/{resume_mode}",
                      lambda args=args: ns["test_resume_path_variable_does_not_change_other_modes"](*args))

            for mode, async_save, exit_code in itertools.product(["spmd", "ray"], ["False", "True"], [0, 37]):
                path = case_dir()

                def space_case(path=path, mode=mode, async_save=async_save, exit_code=exit_code):
                    source = path / "resume inputs/global_step_2"
                    source.mkdir(parents=True)
                    marker = source / "input.marker"
                    marker.write_text("external synthetic resume input\n")
                    work, checkpoint, completed = ns["_run_sft_engine"](
                        path, mode, exit_code, "resume_path", "False", str(source), async_save
                    )
                    if label == "base" and async_save == "False" and exit_code == 0:
                        empty_array_observations[mode] = {
                            "returncode": completed.returncode,
                            "recorded_training_arguments": (work / "training-arguments.json").is_file(),
                            "empty_array_unbound_error": "checkpoint_overrides[@]: unbound variable" in completed.stderr,
                        }
                    assert completed.returncode == exit_code
                    assert (work / "training-arguments.json").is_file(), "no recorded training arguments"
                    arguments = json.loads((work / "training-arguments.json").read_text())
                    assert [arg for arg in arguments if arg.startswith("trainer.resume_from_path=")] == [
                        f"trainer.resume_from_path={source}"
                    ]
                    assert marker.read_text() == "external synthetic resume input\n"
                    assert ns["_state"](checkpoint) == {
                        "input": False, "generated": False, "notes": False, "hidden": True, "tracker": False
                    }
                    assert (work / "outside.marker").read_text() == "outside checkpoint directory\n"

                check("local_space_path_no_hydra", f"{mode}/{async_save}/{exit_code}", space_case)
            result["scripts"][label] = {
                group: {"passed": len(row["passed"]), "failed": len(row["failed"]), "failed_cases": row["failed"]}
                for group, row in groups.items()
            }
            if empty_array_observations:
                result["base_empty_array_observations"] = empty_array_observations
    expected = {"base": [(0, 8), (4, 0), (0, 8)], "head": [(8, 0), (4, 0), (8, 0)]}
    for label, counts in expected.items():
        actual = [(row["passed"], row["failed"]) for row in result["scripts"][label].values()]
        assert actual == counts, (label, actual, counts)
    return result


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]).resolve()), ensure_ascii=False, indent=2))
