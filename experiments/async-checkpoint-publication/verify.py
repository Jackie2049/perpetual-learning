"""Stdlib controls of #8056's full original driver coordinator, not Ray tests.

Local workers return explicit acknowledgements or injected failures; pruning
only records calls. Real temporary tracker files and os.replace are exercised.
No role checkpoint payload, retention deletion, Megatron queue or trainer runs.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

HEAD = "c76d46ea6474f9fc9837a469c2ab74b58d921bbc"
PREVIOUS = "d91f870e3565be4bddfbf680f9eb5e11fda6324c"
SOURCE = "verl/trainer/ppo/async_checkpoint.py"


def main(repo):
    def read(rev):
        return subprocess.check_output(["git", "-C", repo, "show", f"{rev}:{SOURCE}"],
                                       env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})

    raw, previous = read(HEAD), read(PREVIOUS)
    assert raw == previous
    module = types.ModuleType("_pl_original_async_checkpoint")
    exec(compile(raw, SOURCE, "exec"), module.__dict__)
    results = {}
    with tempfile.TemporaryDirectory(prefix="pl-checkpoint-publication-") as temporary:
        root = Path(temporary)

        def setup(name):
            folder = root / name
            folder.mkdir()
            tracker = folder / "latest_checkpointed_iteration.txt"
            tracker.write_text("1")
            events = []

            class Worker:
                def __init__(self, role):
                    self.role = role
                    self.ack = [True, True]
                    self.finalize_error = None
                    self.prune_error = None

                def finalize_async_checkpointing(self, blocking=False):
                    events.append(["finalize", self.role, blocking])
                    if self.finalize_error:
                        raise self.finalize_error
                    return self.ack

                def prune_checkpoints(self, max_ckpt_to_keep):
                    events.append(["prune", self.role, max_ckpt_to_keep, tracker.read_text()])
                    if self.prune_error:
                        raise self.prune_error

            actor, critic = Worker("actor"), Worker("critic")
            coordinator = module.AsyncCheckpointCoordinator([actor, critic], str(folder),
                                                           [(actor, 1), (critic, 2)])
            return coordinator, actor, critic, tracker, events

        coordinator, actor, critic, tracker, events = setup("idle")
        coordinator.finalize(blocking=True)
        assert events == [] and tracker.read_text() == "1"
        results["no_pending_is_noop"] = True

        coordinator, actor, critic, tracker, events = setup("partial")
        coordinator.pending_step = 3
        actor.ack = [True, False]
        coordinator.finalize()
        assert events == [["finalize", "actor", False], ["finalize", "critic", False]]
        assert tracker.read_text() == "1" and coordinator.pending_step == 3
        results["incomplete_return_still_progresses_other_role"] = events.copy()

        coordinator, actor, critic, tracker, events = setup("truthy")
        coordinator.pending_step = 3
        actor.ack = [1]
        try:
            coordinator.finalize(blocking=True)
        except RuntimeError as error:
            assert "pending writes" in str(error)
        else:
            raise AssertionError("truthy integer is not exact True")
        assert tracker.read_text() == "1" and coordinator.pending_step == 3
        assert events == [["finalize", "actor", True], ["finalize", "critic", True]]
        results["blocking_requires_exact_true"] = True

        coordinator, actor, critic, tracker, events = setup("exception")
        coordinator.pending_step = 3
        actor.finalize_error = RuntimeError("writer failed")
        try:
            coordinator.finalize(blocking=True)
        except RuntimeError as error:
            assert str(error) == "writer failed"
        else:
            raise AssertionError("missing writer failure")
        assert events == [["finalize", "actor", True]]
        assert tracker.read_text() == "1" and coordinator.pending_step == 3
        results["raised_rpc_aborts_before_later_roles"] = events.copy()

        coordinator, actor, critic, tracker, events = setup("replace_failure")
        coordinator.pending_step = 3
        with patch.object(module.os, "replace", side_effect=PermissionError("replace blocked")):
            try:
                coordinator.finalize(blocking=True)
            except PermissionError:
                pass
            else:
                raise AssertionError("missing replace failure")
        assert tracker.read_text() == "1" and coordinator.pending_step == 3
        assert Path(str(tracker) + ".tmp").read_text() == "3"
        assert all(event[0] == "finalize" for event in events)
        results["replace_failure"] = {"tracker": "1", "temporary_tracker": "3", "pending": 3}
        coordinator.finalize(blocking=True)
        assert tracker.read_text() == "3" and coordinator.pending_step is None
        assert not Path(str(tracker) + ".tmp").exists()
        assert events[-2:] == [["prune", "actor", 1, "3"], ["prune", "critic", 2, "3"]]
        results["replace_retry_publishes_before_pruning"] = events.copy()

        coordinator, actor, critic, tracker, events = setup("prune_failure")
        coordinator.pending_step = 3
        critic.prune_error = RuntimeError("cleanup failed")
        try:
            coordinator.finalize(blocking=True)
        except RuntimeError as error:
            assert str(error) == "cleanup failed"
        else:
            raise AssertionError("missing cleanup failure")
        assert tracker.read_text() == "3" and coordinator.pending_step == 3
        assert events[-2:] == [["prune", "actor", 1, "3"], ["prune", "critic", 2, "3"]]
        results["cleanup_failure_already_published"] = {"tracker": "3", "pending": 3, "events": events.copy()}
        critic.prune_error = None
        coordinator.finalize(blocking=True)
        assert tracker.read_text() == "3" and coordinator.pending_step is None
        assert events.count(["prune", "actor", 1, "3"]) == 2
        assert events.count(["prune", "critic", 2, "3"]) == 2
        finished_events = events.copy()
        coordinator.finalize(blocking=True)
        assert events == finished_events
        results["cleanup_retry_repeats_pruning_then_clears_pending"] = events.copy()
    print(json.dumps({"python": platform.python_version(), "head": HEAD, "previous": PREVIOUS,
                      "module_unchanged_from_previous": True,
                      "source_sha256": hashlib.sha256(raw).hexdigest(),
                      "source_git_blob": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
                      "local_control_groups": len(results), "results": results,
                      "scope": "Eight local stdlib control groups using the complete original coordinator module, real temporary tracker writes/replace, and explicit worker acknowledgement/error recorders. No original OmegaConf tests, prepare_async_checkpoint configuration construction, actual role payloads, retention deletion, fsync/power-loss durability, shared-storage semantics, Ray/Megatron collectives, SFT/RL trainer, GPU/NPU or author H20 acceptance runs."}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
