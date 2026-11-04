#!/usr/bin/env python3
"""Where do the three inert flags still appear in user-facing docs/examples?

Reads a verl checkout and reports every non-code mention, so the "docs still teach it"
part of the claim is a script output rather than a hand-built list.

Run: python3 check_docs.py [--verl-root /tmp/verl-6067-w3] [--json raw/docs.json]
Exit 0 always (report only); exit 2 if a flag has ZERO non-code mentions, which would
mean the doc-debt part of the claim no longer holds.
"""
import argparse, json, os, subprocess, sys

FLAGS = ["tiled_mlp", "use_prefix_grouper", "freeze_vision_tower"]
EXTS = [".md", ".rst", ".yaml", ".yml", ".sh", ".py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verl-root", default=os.environ.get("VERL_ROOT", "/tmp/verl-6067-w3"))
    ap.add_argument("--json")
    args = ap.parse_args()
    out = {}
    for flag in FLAGS:
        r = subprocess.run(
            ["grep", "-rn", flag, "--include=*.md", "--include=*.rst", "--include=*.yaml",
             "--include=*.yml", "--include=*.sh", args.verl_root],
            capture_output=True, text=True,
        )
        hits = [ln.replace(args.verl_root + "/", "") for ln in r.stdout.splitlines() if ln]
        # split doc prose from generated config dumps (generated yaml is machine output)
        out[flag] = {
            "n_hits": len(hits),
            "docs": [h for h in hits if "/docs/" in h],
            "examples": [h for h in hits if "/examples/" in h],
            "generated_yaml": [h for h in hits if "_generated_" in h],
            "config_yaml": [h for h in hits if "/trainer/config/" in h and "_generated_" not in h],
        }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        open(args.json, "w").write(text + "\n")
    print(text)
    missing = [f for f in FLAGS if not (out[f]["docs"] or out[f]["examples"] or out[f]["config_yaml"])]
    return 2 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
