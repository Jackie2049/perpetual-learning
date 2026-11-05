#!/usr/bin/env bash
# Do the two TM-008 patches apply to the trees we will actually hand to the window?
#
# `poll-count.patch` was authored against one commit; the window runs a different one
# (`/home/zxw/venvs/rollout-g8-src/verl` = 24f25b03). A hunk that only matches main
# fails in prep-tree *before* any GPU is touched, but it still costs the window. Worse,
# a hunk with fuzz would apply somewhere else and the log would carry no counter.
#
# For each clean tree: build the instrumented copy with prep-tree.sh (both patches,
# --fuzz=0, four markers), run the counter harness on that exact tree, and verify the
# key routing. Output is one line per tree per stage.
#
# Usage: PYBIN=/tmp/py312/python/bin/python3 bash verify-patch-matrix.sh <tree> [<tree>...]
#        PYBIN must be >= 3.10 with numpy (the tree's replay_buffer.py uses `match`).
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PYBIN=${PYBIN:-python3}
SIM=$HERE/../verl-v1-poll-quantum-polls/sim-loop.py

if [ $# -lt 1 ]; then
  echo "usage: bash verify-patch-matrix.sh <clean-tree> [<clean-tree>...]" >&2
  exit 2
fi

fail=0
for src in "$@"; do
  tmp=$(mktemp -d)
  dst=$tmp/verl
  label=$(basename "$src")
  printf '== %s\n' "$src"
  if bash "$HERE/prep-tree.sh" "$src" "$dst" >/dev/null 2>&1; then
    echo "   prep-tree   OK"
  else
    echo "   prep-tree   FAIL"
    fail=1
    rm -rf "$tmp"
    continue
  fi
  sim=$("$PYBIN" "$SIM" --tree "$dst" --selftest 2>&1 | tail -1)
  echo "   sim         $sim"
  case "$sim" in
    *"12/12 passed"*) ;;
    *) fail=1 ;;
  esac
  route=$("$PYBIN" "$HERE/verify-counter-routing.py" --tree "$dst" 2>&1 | tail -1)
  echo "   routing     $route"
  case "$route" in
    *"instrumented=True failed=0"*) ;;
    *) fail=1 ;;
  esac
  rm -rf "$tmp"
done

echo "matrix: $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
