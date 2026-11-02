#!/usr/bin/env python3
"""Push local HEAD to GitHub over the REST API when `git push` cannot reach github.com.

Why this exists: on 2026-11-03 github.com:443 was unreachable from the M1 host
while api.github.com answered and the gh token had `repo` scope, so REST was the
only usable channel. The repo also has a giant top-level directory (experiments/
with thousands of blobs), and POST /git/trees rejects a single request of that
size with HTTP 502. This script therefore rebuilds only the directories on the
path of each changed file and reuses the base tree's entry for everything else.

Usage:
    push_via_api.py <base_commit_sha> [-m MESSAGE_FILE]

Caveats:
  * Squashes every local commit above <base> into one commit on the remote.
  * The resulting remote commit has a different SHA than the local one; fetch
    and reset locally afterwards so the two agree.
"""

import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = "Jackie2049/perpetual-learning"
API = f"https://api.github.com/repos/{REPO}"
ROOT = pathlib.Path(__file__).resolve().parents[2]
TOKEN = subprocess.check_output(["gh", "auth", "token"], text=True).strip()


def call(method, url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {url} -> {e.code}: {e.read().decode(errors='replace')[:400]}")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def retry_tree(prefix, level, attempts=4):
    """GitHub's trees endpoint intermittently 500s; retry with backoff."""
    last = None
    for n in range(attempts):
        data = json.dumps({"tree": level}).encode()
        req = urllib.request.Request(f"{API}/git/trees", data=data, method="POST")
        req.add_header("Authorization", f"Bearer {TOKEN}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = f"{e.code}: {e.read().decode(errors='replace')[:200]}"
            print(f"  retry {prefix or '<root>'} after {e.code} ({n + 1}/{attempts})")
            time.sleep(2 ** n)
    raise SystemExit(f"tree {prefix} failed: {last}")


def main():
    base = sys.argv[1]
    msg = git("log", "-1", "--format=%B", "HEAD").decode()
    if "-m" in sys.argv:
        msg = open(sys.argv[sys.argv.index("-m") + 1]).read()

    parts = [p for p in git("diff", "--name-status", "-z", base, "HEAD").decode().split("\0") if p]
    changed = {parts[i + 1]: parts[i] for i in range(0, len(parts), 2)}
    print(f"{len(changed)} path(s) changed vs {base[:8]}")

    new_blobs = {}
    for path, status in sorted(changed.items()):
        if status == "D":
            print(f"  delete {path}")
            continue
        blob = call("POST", f"{API}/git/blobs", {
            "content": git("show", f"HEAD:{path}").decode("utf-8", errors="surrogateescape"),
            "encoding": "utf-8",
        })
        new_blobs[path] = blob["sha"]
        print(f"  write  {path} -> {blob['sha'][:8]}")

    base_tree = call("GET", f"{API}/commits/{base}")["commit"]["tree"]["sha"]
    # directories whose own entry list changes: every ancestor of a changed file
    touched = set()
    for path in changed:
        bits = path.split("/")[:-1]
        for i in range(len(bits) + 1):
            touched.add("/".join(bits[:i]))

    def rebuild(prefix):
        url = f"{API}/git/trees/{base_tree}" + (f":{prefix}" if prefix else "")
        entries = call("GET", url)["tree"]
        level = []
        for entry in entries:
            child = f"{prefix}/{entry['path']}" if prefix else entry["path"]
            if entry["type"] == "tree":
                sha = rebuild(child) if child in touched else entry["sha"]
                level.append({"path": entry["path"], "mode": "040000", "type": "tree", "sha": sha})
            elif child not in changed:  # changed files are re-added below
                level.append({"path": entry["path"], "mode": entry["mode"],
                              "type": "blob", "sha": entry["sha"]})
        seen_trees = {e["path"] for e in level if e["type"] == "tree"}
        for path, sha in new_blobs.items():
            parent = "/".join(path.split("/")[:-1])
            if parent == prefix:
                level.append({"path": path.split("/")[-1], "mode": "100644",
                              "type": "blob", "sha": sha})
            elif parent.startswith(prefix + "/") if prefix else parent:
                child = (parent[len(prefix) + 1:] if prefix else parent).split("/")[0]
                if child not in seen_trees:
                    seen_trees.add(child)
                    level.append({"path": child, "mode": "040000", "type": "tree",
                                  "sha": rebuild(f"{prefix}/{child}" if prefix else child)})
        level.sort(key=lambda e: e["path"])
        result = retry_tree(prefix, level)
        print(f"  tree   {prefix or '<root>'}: {len(level)} entries -> {result['sha'][:8]}")
        return result["sha"]

    root = rebuild("")
    commit = call("POST", f"{API}/git/commits",
                  {"message": msg, "tree": root, "parents": [base]})
    print("commit:", commit["sha"])
    upd = call("PATCH", f"{API}/git/refs/heads/main", {"sha": commit["sha"], "force": False})
    print("main ->", upd["object"]["sha"])


if __name__ == "__main__":
    main()
