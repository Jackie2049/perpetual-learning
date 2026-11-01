"""Run #8056's original 156 shell tests with real Hydra, on new/previous shell.

Use a dedicated interpreter with pytest 8.4.2, Hydra 1.3.2, OmegaConf 2.3.1:
python verify_hydra.py /cached/verl-git
Only the original test's guarded disposable files and command shims run.
No torch/verl import, launcher, model, checkpoint payload or training.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

BASE = "4a0679602e0a56b13a49e477e279cef5f03324aa"
HEAD = "fa95b88e69c4ed335fd4a8e9791fd1ad47b59633"
SCRIPT = "tests/special_e2e/sft/run_sft_engine.sh"
TEST = "tests/trainer/test_sft_engine_cleanup_on_cpu.py"


def run_one(root: Path):
    import pytest

    class Reports:
        def __init__(self):
            self.rows = []
            self.infrastructure_errors = []

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.rows.append({"case": report.nodeid.split("::", 1)[1], "outcome": report.outcome,
                                  **({"error": str(report.longrepr.reprcrash.message)[:500]}
                                     if report.failed else {})})
            elif report.failed:
                self.infrastructure_errors.append({"case": report.nodeid, "stage": report.when})

        def pytest_collectreport(self, report):
            if report.failed:
                self.infrastructure_errors.append({"case": report.nodeid, "stage": "collection"})

    reports = Reports()
    code = pytest.main([str(root / TEST), "-q", "-c", os.devnull, "--confcutdir", str(root),
                        "-p", "no:cacheprovider", "--tb=line", "--disable-warnings",
                        "--basetemp", str(root / "pytest-cases")], plugins=[reports])
    counts = Counter(row["outcome"] for row in reports.rows)
    groups = {}
    for row in reports.rows:
        group = row["case"].split("[", 1)[0]
        groups.setdefault(group, Counter())[row["outcome"]] += 1
    print("PL_RESULT=" + json.dumps({"exit_code": int(code), "collected_call_reports": len(reports.rows),
                                      "counts": dict(counts), "groups": groups,
                                      "failures": [row for row in reports.rows if row["outcome"] == "failed"],
                                      "infrastructure_errors": reports.infrastructure_errors}))


def main(repo: Path):
    git_env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], env=git_env)

    assert git("merge-base", BASE, HEAD).decode().strip() == BASE
    assert git("show", "-s", "--format=%P", HEAD).decode().strip() == BASE
    sources = {(HEAD, TEST): git("show", f"{HEAD}:{TEST}"),
               **{(ref, SCRIPT): git("show", f"{ref}:{SCRIPT}") for ref in (BASE, HEAD)}}
    compile(sources[HEAD, TEST], TEST, "exec")
    results = {}
    for label, ref in (("base", BASE), ("head", HEAD)):
        with tempfile.TemporaryDirectory(prefix="pl-sft-hydra-") as temporary:
            root = Path(temporary).resolve()
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w") as output:
                for path, data in ((SCRIPT, sources[ref, SCRIPT]), (TEST, sources[HEAD, TEST])):
                    assert path in (SCRIPT, TEST)
                    entry = tarfile.TarInfo(path)
                    entry.mode, entry.size = 0o644, len(data)
                    output.addfile(entry, io.BytesIO(data))
            subprocess.run(["tar", "-x", "-C", str(root)], input=archive.getvalue(), check=True)
            assert (root / SCRIPT).read_bytes() == sources[ref, SCRIPT]
            assert (root / TEST).read_bytes() == sources[HEAD, TEST]
            subprocess.run([shutil.which("bash"), "-n", str(root / SCRIPT)], check=True)
            child_env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", str(root)],
                                       capture_output=True, text=True, env=child_env, check=True, timeout=240)
            lines = [line for line in completed.stdout.splitlines() if line.startswith("PL_RESULT=")]
            assert len(lines) == 1, completed.stdout[-2000:]
            result = json.loads(lines[0].removeprefix("PL_RESULT="))
            assert not result["infrastructure_errors"], result
            assert result["collected_call_reports"] == 156, result
            results[label] = result
    assert results["head"]["counts"] == {"passed": 156}, results["head"]
    assert results["head"]["exit_code"] == 0
    assert results["base"]["counts"].get("failed", 0) > 0
    assert results["base"]["exit_code"] == 1

    unchanged = {}
    for path in ("verl/trainer/ppo/async_checkpoint.py", "verl/trainer/ppo/v1/trainer_base.py",
                 "verl/trainer/sft_trainer.py", "verl/trainer/sft_trainer_ray.py",
                 "verl/utils/checkpoint/checkpoint_handler.py",
                 "verl/utils/checkpoint/megatron_checkpoint_manager.py"):
        def blob(ref):
            line = git("ls-tree", ref, "--", path).decode().strip()
            assert line and line.split()[1] == "blob", path
            return line.split()[2]
        assert blob(BASE) == blob(HEAD), path
        unchanged[path] = blob(HEAD)
    return {
        "base_and_actual_parent": BASE, "head": HEAD,
        "environment": {"python": platform.python_version(),
                        **{name: importlib.metadata.version(name) for name in ("pytest", "hydra-core", "omegaconf")},
                        "bash": subprocess.check_output([shutil.which("bash"), "--version"], text=True).splitlines()[0]},
        "source_sha256": {f"{ref}:{path}": hashlib.sha256(data).hexdigest()
                          for (ref, path), data in sources.items()},
        "unchanged_core_blobs_not_executed": unchanged, "original_pytest_results": results,
        "scope": "Original HEAD test module, native pytest collection and assertions unchanged; "
                 "only shell revision differs. Original guarded-rm and fake Python command, real minimal Hydra "
                 "composition/resolution. No torch/verl import, real launcher, model, checkpoint content, "
                 "resume state, async writer, GPU/NPU, server access or training.",
    }


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        run_one(Path(sys.argv[2]))
    else:
        print(json.dumps(main(Path(sys.argv[1]).resolve()), indent=2))
