"""Check original #8316 topology and original pure guards, never import NIXL."""
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

BASE = "619c397dc9917fed278b6e47c52a6620aef05466"
HEAD = "3a60b8710f8e1001110820f7a14cd0e71d9d5e07"


def main():
    repo = Path(sys.argv[1]).resolve()

    def read(rev, path):
        data = subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=repo,
                                       env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        print(hashlib.sha256(data).hexdigest(), rev, path)
        return ast.parse(data)

    def method(tree, cls, name):
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)

    def block(nodes):
        return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "original-guard", "exec")

    manager = method(read(HEAD, "verl/checkpoint_engine/base.py"), "CheckpointEngineManager", "build_process_group")
    guards = [n for n in manager.body if isinstance(n, ast.For) and len(n.body) == 1
              and isinstance(n.body[0], ast.Assert)]
    assert len(guards) == 2
    manager_guards = block(guards)
    outputs = {}
    for revision in (BASE, HEAD):
        tree = read(revision, "verl/checkpoint_engine/nixl_checkpoint_engine.py")
        topology = method(tree, "NIXLCheckpointEngine", "build_topology")
        namespace = {}
        # The class shell contains only the unchanged original classmethod node.
        shell = ast.ClassDef(name="OriginalTopology", bases=[], keywords=[], body=[topology], decorator_list=[])
        exec(block([shell]), namespace)
        build = namespace["OriginalTopology"].build_topology
        init = method(tree, "NIXLCheckpointEngine", "init_process_group")
        init_guard = block([next(n for n in init.body if isinstance(n, ast.If))])
        passed = failed = 0
        for actors in (1, 2, 4, 8):
            for rollouts in (1, 2, 3, 4, 8):
                actor_meta = [{"actor": i} for i in range(actors)]
                rollout_meta = [{"rollout": i} for i in range(rollouts)]
                actor_kwargs, rollout_kwargs = build(actors, rollouts, actor_meta + rollout_meta)
                outputs[revision, actors, rollouts] = actor_kwargs, rollout_kwargs
                scope = dict(actor_wg=SimpleNamespace(world_size=actors), rollout=SimpleNamespace(world_size=rollouts),
                             actor_wg_kwargs=actor_kwargs, rollout_kwargs=rollout_kwargs)
                try:
                    exec(manager_guards, scope)
                except AssertionError as error:
                    assert revision == BASE and rollouts == 1
                    assert str(error) == "rollout_kwargs[next_agent_metadata] must have length of 1"
                    assert len(rollout_kwargs["next_agent_metadata"]) == actors + 2
                    failed += 1
                    continue
                assert actor_kwargs["rank"] == [0] + [-1] * (actors - 1)
                assert rollout_kwargs["rank"] == list(range(1, rollouts + 1))
                assert actor_kwargs["next_agent_metadata"] == [rollout_meta[0]] + [None] * (actors - 1)
                assert rollout_kwargs["prev_agent_metadata"] == [actor_meta[0]] + rollout_meta[:-1]
                assert rollout_kwargs["next_agent_metadata"] == rollout_meta[1:] + [None]
                for group in (actor_kwargs, rollout_kwargs):
                    for index in range(len(group["rank"])):
                        exec(init_guard, {key: value[index] for key, value in group.items()})
                passed += 1
        assert (passed, failed) == ((16, 4) if revision == BASE else (20, 0))
        print(revision[:8], "20 synthetic topology cases:", passed, "pass", failed, "expected old guard failures")
    for actors in (1, 2, 4, 8):
        for rollouts in (2, 3, 4, 8):
            assert outputs[BASE, actors, rollouts] == outputs[HEAD, actors, rollouts]
    assert all(name not in sys.modules for name in ("torch", "ray", "nixl", "cupy", "zmq", "verl"))
    print("16 multi-rollout outputs unchanged; no prepare/init/transfer, metadata contains labels only")


if __name__ == "__main__":
    main()
