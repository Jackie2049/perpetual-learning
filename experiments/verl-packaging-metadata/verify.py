"""Generate pinned verl egg-info from original packaging inputs, offline.

Python 3.12+, setuptools, wheel and packaging. Requires cached Git blobs.
No dependency resolver, install, package import, wheel or device execution.
The legacy-only mode deliberately omits pyproject.toml; it is a counterfactual,
not another supported installation path. No upstream source is changed.
"""

import argparse
import ast
from email.parser import Parser
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

from packaging.requirements import Requirement

REV = "a08ea02a784d7d4982b1d81df36d188a476192bb"
INPUTS = ("setup.py", "pyproject.toml", "README.md", "LICENSE",
          "verl/version/version", "verl/__init__.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="local read-only Git object repository")
    args = parser.parse_args()
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", PIP_NO_INDEX="1",
               PYTHONDONTWRITEBYTECODE="1")
    sources = {p: subprocess.check_output(["git", "show", f"{REV}:{p}"],
                                          cwd=args.repo, env=env) for p in INPUTS}
    tree = ast.parse(sources["setup.py"])
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                literals[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    extras_node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "extras_require" for t in n.targets))
    legacy = {ast.literal_eval(k): literals[v.id] for k, v in zip(extras_node.keys, extras_node.values)}
    config = tomllib.loads(sources["pyproject.toml"].decode())
    project = config["project"]
    assert "optional-dependencies" not in project["dynamic"]

    def generate(include_pyproject, pep517=False):
        # Export exact original files with a separate temporary Git index.
        # git archive may prefetch ALL blobs in a partial clone, even with a
        # pathspec. checkout-index only needs these six cached blobs and never
        # changes the source repository's real index or checkout.
        paths = list(INPUTS) if include_pyproject else [p for p in INPUTS if p != "pyproject.toml"]
        with tempfile.TemporaryDirectory(prefix="pl-verl-metadata-") as tmp:
            copy_env = dict(env, GIT_INDEX_FILE=str(Path(tmp) / "snapshot.index"))
            subprocess.run(["git", "read-tree", REV], cwd=args.repo, env=copy_env, check=True)
            subprocess.run(["git", "checkout-index", f"--prefix={tmp}/", "--", *paths],
                           cwd=args.repo, env=copy_env, check=True)
            command = [sys.executable, "-B", "setup.py", "-q", "egg_info"]
            pattern = "*.egg-info/PKG-INFO"
            if pep517:
                metadata_dir = Path(tmp) / "metadata"
                metadata_dir.mkdir()
                code = ("from setuptools.build_meta import prepare_metadata_for_build_wheel; "
                        "import sys; prepare_metadata_for_build_wheel(sys.argv[1]); "
                        "assert not {'torch', 'ray', 'verl'} & sys.modules.keys()")
                command = [sys.executable, "-B", "-c", code, str(metadata_dir)]
                pattern = "metadata/*.dist-info/METADATA"
            build = subprocess.run(command,
                                   cwd=tmp, env=env, text=True, capture_output=True, timeout=60)
            if build.returncode:
                raise RuntimeError(build.stdout + build.stderr)
            files = list(Path(tmp).glob(pattern))
            assert len(files) == 1
            metadata = Parser().parsestr(files[0].read_text())
            requirements = [Requirement(r) for r in metadata.get_all("Requires-Dist", [])]
            extra_names = sorted(metadata.get_all("Provides-Extra", []))
            by_extra = {
                name: sorted(str(r).split(";")[0].strip() for r in requirements
                             if r.marker is not None and r.marker.evaluate({"extra": name})
                             and not r.marker.evaluate({"extra": ""}))
                for name in ("vllm", "sglang", "prime", "trtllm") if name in extra_names
            }
            return {"version": metadata["Version"], "provides_extra": extra_names,
                    "base_requires_dist": sorted(str(r) for r in requirements if r.marker is None),
                    "selected_extra_requires_dist": by_extra,
                    "extras_overwrite_warning": "`extras_require` overwritten" in build.stderr}

    current = generate(True)
    pep517 = generate(True, pep517=True)
    fallback = generate(False)
    assert current == pep517
    assert set(current["provides_extra"]) == set(project["optional-dependencies"])
    assert set(fallback["provides_extra"]) == set(legacy)
    missing = sorted(set(legacy) - set(current["provides_extra"]))
    assert missing == ["geo", "gpu", "mcore", "prime", "trl", "trtllm"]
    assert current["extras_overwrite_warning"] and not fallback["extras_overwrite_warning"]
    assert current["base_requires_dist"] == fallback["base_requires_dist"]
    assert len(current["base_requires_dist"]) == len(literals["install_requires"])
    assert fallback["selected_extra_requires_dist"]["prime"] == []
    assert "tensorrt-llm>=1.2.0rc6" in fallback["selected_extra_requires_dist"]["trtllm"]
    print(json.dumps({"revision": REV, "python": sys.version.split()[0],
                      "tools": {p: version(p) for p in ("setuptools", "wheel", "packaging")},
                      "project_dynamic": project["dynamic"], "legacy_declared_extras": sorted(legacy),
                      "missing_legacy_extras": missing, "current": current,
                      "pep517_matches_egg_info": True,
                      "counterfactual_without_pyproject": fallback,
                      "source_sha256": {p: hashlib.sha256(v).hexdigest() for p, v in sources.items()},
                      "scope": "six original packaging inputs; real egg_info and PEP517 metadata hook; no full source tree, wheel build, resolver or install"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
