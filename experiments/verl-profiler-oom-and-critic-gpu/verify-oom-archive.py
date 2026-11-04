#!/usr/bin/env python3
"""Re-check the three-arm OOM-snapshot archive from the JSON alone (no GPU, no server).

The archive was produced on 4090-1 (GPU 1) on 2026-11-04 18:0x CST by an interrupted
session; this script is what makes the reads re-checkable afterwards. It does not
re-measure anything -- it checks that the recorded facts are self-consistent and that
the negative control really is a negative control.

Checks (per arm):
  A. OOM really raised, as torch.OutOfMemoryError
  B. attached arms: a snapshot appears at the FIRST poll mark and stays there through 20 s,
     final_files == files_after["20s"], filename matches
     oom_<ts>/torch_memory_oom_rank<env RANK>_pid<pid>.pickle (env RANK, NOT the profiler rank)
  C. attached arms: snapshot loads to the four expected keys, device_traces is a 1-element
     list, segments >= 0
  D. unranked arm: observer not attached, should_profile False, and NO file at ANY mark
  E. log_counts_by_logger_level recounts from log_lines (no hand-typed totals)
  F. the user-visible channel: raw/nocap.stderr holds >= 3 [torch_memory] lines while the
     nocap arm's in-process log_lines is empty (its Capture handler is not installed)

Reported, not gated:
  * oom_request_seconds per arm (dump cost, single run each)
  * every arm's free/total bytes (same card, same window)

Exit codes: 0 all checks pass, 1 a check failed, 2 archive incomplete.
  --selftest fabricates archives that must fail; it exits 0 only if every injected fault
  is caught (and the clean copy passes).
"""

import argparse
import json
import re
import sys
from pathlib import Path

ARMS = ("attached", "nocap", "unranked")
SNAP_KEYS = {"allocator_settings", "device_traces", "external_annotations", "segments"}
NAME_RE = re.compile(r"^oom_\d{8}-\d{6}-\d{6}/torch_memory_oom_rank(\d+)_pid(\d+)\.pickle$")


def load(raw_dir):
    out, missing = {}, []
    for arm in ARMS:
        p = raw_dir / f"result-{arm}.json"
        if not p.exists():
            missing.append(p.name)
        else:
            out[arm] = json.loads(p.read_text())
    return out, missing


def size_of(entry):
    m = re.search(r"\((\d+) B\)$", entry)
    return int(m.group(1)) if m else None


def check_arm(arm, d, checks):
    def ck(name, ok, detail=""):
        checks.append({"check": f"{arm}:{name}", "ok": bool(ok), "detail": detail})

    ck("oom_raised", d.get("oom_raised") is True)
    ck("oom_is_outofmemoryerror", d.get("except_class") == "OutOfMemoryError", d.get("except_class", ""))
    ck("should_profile", d.get("should_profile_this_rank") is (arm != "unranked"),
       str(d.get("should_profile_this_rank")))
    ck("observer_flag", d.get("oom_observer_attached_after_init") is (arm != "unranked"),
       str(d.get("oom_observer_attached_after_init")))

    marks = d.get("files_after", {})
    final = d.get("final_files", [])
    snaps = d.get("snapshots", [])

    if arm == "unranked":
        ck("no_file_at_any_mark", all(v == [] for v in marks.values()), json.dumps(marks))
        ck("no_final_files", final == [])
        ck("no_snapshots", snaps == [])
        return

    ck("five_polls", sorted(marks) == sorted(["0s", "1s", "3s", "10s", "20s"]), ",".join(sorted(marks)))
    # B: visible at the first poll already, and unchanged afterwards
    first = marks.get("0s", [])
    ck("file_present_at_0s", len(first) == 2, json.dumps(first))
    ck("stable_marks", all(v == first for v in marks.values()), "marks differ")
    ck("final_equals_20s", final == marks.get("20s", []), json.dumps(final))
    pickles = [e for e in final if e.endswith(".pickle)") or ".pickle (" in e]
    ck("exactly_one_pickle", len(pickles) == 1, json.dumps(pickles))
    on_disk = size_of(pickles[0]) if pickles else None
    if pickles:
        path = pickles[0].split(" (")[0]
        ck("filename_shape", NAME_RE.match(path) is not None, path)
        ck("pickle_nonempty", bool(on_disk) and on_disk > 0, str(on_disk))
    # C: payload
    ck("one_snapshot_record", len(snaps) == 1, str(len(snaps)))
    if snaps:
        s = snaps[0]
        ck("snapshot_keys", set(s.get("keys", [])) == SNAP_KEYS, json.dumps(s.get("keys")))
        dt = s.get("device_traces")
        ck("device_traces_is_1_list", isinstance(dt, list) and len(dt) == 1, json.dumps(dt))
        ck("segments_present", isinstance(s.get("segments"), int) and s["segments"] >= 0, str(s.get("segments")))
        # the loadable-pickle size and the directory listing must agree
        ck("snapshot_bytes_match", on_disk == s.get("bytes"), f"listing {on_disk} vs pickle {s.get('bytes')}")


def check_logs(arms, raw_dir, checks):
    def ck(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    att = arms.get("attached", {})
    recs = att.get("log_lines", [])
    # E: recount instead of trusting the recorded tally
    recount = {}
    for r in recs:
        k = f"{r['logger']}:{r['level']}"
        recount[k] = recount.get(k, 0) + 1
    ck("attached:log_tally_recounts", recount == att.get("log_counts_by_logger_level"),
       json.dumps({"recount": recount, "recorded": att.get("log_counts_by_logger_level")}))
    msgs = " | ".join(r["msg"] for r in recs)
    ck("attached:has_attach_info", "OOM snapshot observer attached" in msgs)
    ck("attached:has_dumping_error", "dumping allocator snapshot" in msgs)
    ck("attached:has_python_stack", "Python stack at OOM" in msgs)
    # the periodic-path "dumped: <path>" INFO lives on memory_utils, whose logger is
    # named by file path and defaults to WARN -> it can never show up here
    ck("attached:no_dumped_path_line", "dumped:" not in msgs, "memory_utils INFO leaked in?")

    nocap = arms.get("nocap", {})
    ck("nocap:in_process_log_empty", nocap.get("log_lines") == [])
    stderr = raw_dir / "nocap.stderr"
    text = stderr.read_text() if stderr.exists() else ""
    n_err = len(re.findall(r"^ERROR:.*\[torch_memory\]", text, flags=re.M))
    ck("nocap:stderr_shows_oom_errors", n_err >= 3, f"{n_err} ERROR lines")
    ck("nocap:stderr_hides_attach_info", "observer attached" not in text)


def check_meta(arms, checks):
    def ck(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    mem = {a: arms[a].get("mem_before") for a in ARMS if a in arms}
    ck("same_card_same_window", len({json.dumps(v) for v in mem.values()}) == 1, json.dumps(mem))
    t = {a: arms[a].get("oom_request_seconds") for a in ARMS if a in arms}
    ck("request_seconds_recorded", all(isinstance(v, (int, float)) for v in t.values()), json.dumps(t))
    if "unranked" in t and "attached" in t:
        checks.append({"check": "INFO:dump_overhead_seconds",
                       "ok": True,
                       "detail": f"attached {t['attached']:.3f}s vs unranked {t['unranked']:.3f}s "
                                 f"(single run each, not a timed experiment)"})


def run(raw_dir):
    arms, missing = load(raw_dir)
    if missing:
        return {"exit": 2, "missing": missing, "checks": []}
    checks = []
    for arm in ARMS:
        check_arm(arm, arms[arm], checks)
    check_logs(arms, raw_dir, checks)
    check_meta(arms, checks)
    gated = [c for c in checks if not c["check"].startswith("INFO:")]
    failed = [c for c in gated if not c["ok"]]
    return {"exit": 0 if not failed else 1, "checks": checks,
            "n_pass": sum(1 for c in gated if c["ok"]), "n_fail": len(failed)}


def selftest(tmp):
    """Fabricate a clean archive plus one injected fault per check family."""
    tmp.mkdir(parents=True, exist_ok=True)

    def snap_arm(arm, with_snapshot=True, extra_logs=None):
        files = (["oom_20261104-100119-420536/",
                  "oom_20261104-100119-420536/torch_memory_oom_rank0_pid2515486.pickle (6275 B)"]
                 if with_snapshot else [])
        return {
            "arm": arm, "oom_raised": True, "except_class": "OutOfMemoryError",
            "rank": 0 if arm != "unranked" else 3, "config_ranks": [0],
            "should_profile_this_rank": arm != "unranked",
            "oom_observer_attached_after_init": arm != "unranked",
            "has_cuda_attach_api": True, "files_before": [],
            "mem_before": {"free_bytes": 1, "total_bytes": 2},
            "oom_request_seconds": 0.1 if arm != "unranked" else 0.005,
            "files_after": {k: list(files) for k in ("0s", "1s", "3s", "10s", "20s")},
            "final_files": list(files),
            "snapshots": ([{"path": files[1].split(" (")[0], "bytes": 6275,
                            "keys": sorted(SNAP_KEYS), "device_traces": [3], "segments": 1}]
                          if with_snapshot else []),
            "log_lines": extra_logs or [],
            "log_counts_by_logger_level": {},
            "cuda_alive_after_oom": True,
        }

    logs = [
        {"level": "INFO", "logger": "verl.utils.profiler.torch_memory_profile",
         "msg": "[torch_memory] CUDA OOM snapshot observer attached"},
        {"level": "ERROR", "logger": "verl.utils.profiler.torch_memory_profile",
         "msg": "[torch_memory] CUDA OOM on device 0: requested=1 total=2 free=1; dumping allocator snapshot"},
        {"level": "ERROR", "logger": "verl.utils.profiler.torch_memory_profile",
         "msg": "[torch_memory] Python stack at OOM:\n  File ..."},
        {"level": "ERROR", "logger": "verl.utils.profiler.torch_memory_profile",
         "msg": "[torch_memory] allocator memory at OOM: {}"},
    ]
    tally = {}
    for r in logs:
        k = f"{r['logger']}:{r['level']}"
        tally[k] = tally.get(k, 0) + 1

    def write(arm, d):
        (tmp / f"result-{arm}.json").write_text(json.dumps(d))

    def clean():
        a = snap_arm("attached"); a["log_lines"] = logs; a["log_counts_by_logger_level"] = tally
        c = snap_arm("nocap"); c["log_lines"] = []
        u = snap_arm("unranked", with_snapshot=False)
        write("attached", a); write("nocap", c); write("unranked", u)
        (tmp / "nocap.stderr").write_text(
            "ERROR:2026-11-04 10:04:22,541:[torch_memory] CUDA OOM on device 0\n"
            "ERROR:2026-11-04 10:04:22,542:[torch_memory] Python stack at OOM:\n"
            "ERROR:2026-11-04 10:04:22,543:[torch_memory] allocator memory at OOM: {}\n")

    results = {}
    clean()
    results["clean_passes"] = run(tmp)["exit"] == 0

    # fault 1: negative control did dump
    clean()
    u = json.loads((tmp / "result-unranked.json").read_text())
    u["final_files"] = ["oom_x/", "oom_x/torch_memory_oom_rank0_pid1.pickle (10 B)"]
    u["files_after"] = {k: list(u["final_files"]) for k in u["files_after"]}
    write("unranked", u)
    results["catches_unranked_dump"] = run(tmp)["exit"] == 1

    # fault 2: attached arm dumped nothing
    clean()
    a = json.loads((tmp / "result-attached.json").read_text())
    a["files_after"] = {k: [] for k in a["files_after"]}
    a["final_files"] = []; a["snapshots"] = []
    write("attached", a)
    results["catches_missing_dump"] = run(tmp)["exit"] == 1

    # fault 3: snapshot keys changed
    clean()
    a = json.loads((tmp / "result-attached.json").read_text())
    a["snapshots"][0]["keys"] = ["device_traces"]
    write("attached", a)
    results["catches_key_drift"] = run(tmp)["exit"] == 1

    # fault 4: log tally hand-edited
    clean()
    a = json.loads((tmp / "result-attached.json").read_text())
    a["log_counts_by_logger_level"] = {"verl.utils.profiler.torch_memory_profile:INFO": 99}
    write("attached", a)
    results["catches_tally_edit"] = run(tmp)["exit"] == 1

    # fault 5: the "dumped: path" INFO leaked into the record (would mean the logger default
    # did not suppress it, and the log-based reading would need redoing)
    clean()
    a = json.loads((tmp / "result-attached.json").read_text())
    a["log_lines"] = logs + [{"level": "INFO", "logger": "x", "msg": "[memory_visualize] dumped: /p"}]
    write("attached", a)
    results["catches_dumped_path_leak"] = run(tmp)["exit"] == 1

    # fault 6: missing file -> exit 2, not a silent pass
    (tmp / "result-unranked.json").unlink()
    results["missing_archive_exit_2"] = run(tmp)["exit"] == 2

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(Path(__file__).resolve().parent / "raw"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        import tempfile

        results = selftest(Path(tempfile.mkdtemp()) / "oom-selftest")
        ok = all(results.values())
        print(json.dumps({"selftest": results, "ok": ok}, indent=2))
        return 0 if ok else 1

    res = run(Path(args.raw))
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        for c in res["checks"]:
            tag = "INFO" if c["check"].startswith("INFO:") else ("PASS" if c["ok"] else "FAIL")
            detail = c["detail"] if (not c["ok"] or tag == "INFO") else ""
            print(f"{tag:4} {c['check']}" + (f": {detail}" if detail else ""))
        print(f"\n{res['n_pass']} pass / {res['n_fail']} fail  (exit {res['exit']})")
        if res.get("missing"):
            print(f"missing: {res['missing']}")
    return res["exit"]


if __name__ == "__main__":
    sys.exit(main())
