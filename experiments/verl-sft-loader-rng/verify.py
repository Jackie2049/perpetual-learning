"""PR #8434 CPU RNG isolation with real Torch/torchdata and original tests.

python verify.py /path/to/verl-git --deps /path/to/isolated-deps
No device discovery, all-device seeding, workers, model or checkpoint manager.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

HEAD = "d941a6a81fd79b4261bf1784d66a372a75c183db"
BASE = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
SOURCE = "verl/trainer/sft_trainer.py"
TEST = "tests/trainer/test_sft_dataloader_resume_rng_on_cpu.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache")
    parser.add_argument("--deps", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.deps)
    import torch
    import torchdata
    from torch.utils.data.distributed import DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    import omegaconf

    hashes = {}

    def read(ref, path):
        raw = subprocess.check_output(["git", "-C", args.cache, "show", f"{ref}:{path}"],
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
        return ast.parse(raw)

    def execute(nodes, ns):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_ast", "exec"), ns)

    merge_base = subprocess.check_output(["git", "-C", args.cache, "merge-base", HEAD, BASE],
                                        env=dict(os.environ, GIT_NO_LAZY_FETCH="1")).decode().strip()
    assert merge_base == BASE
    namespaces, original_results = {}, {}
    # The original tests call torch.manual_seed. Redirect only that entry to the
    # real default CPU generator so no CUDA/MPS/XPU seed hooks are called.
    with patch.object(torch, "manual_seed", torch.random.default_generator.manual_seed):
        for ref in (BASE, HEAD):
            ns = dict(torch=torch, StatefulDataLoader=StatefulDataLoader, DistributedSampler=DistributedSampler,
                      get_device_name=lambda: "cpu", SFTTensorCollator=lambda _mode: None)
            cls = next(n for n in read(ref, SOURCE).body if isinstance(n, ast.ClassDef) and n.name == "SFTTrainer")
            method = next(n for n in cls.body if getattr(n, "name", None) == "_build_dataloader")
            execute([ast.ClassDef(name="SFTTrainer", bases=[], keywords=[], body=[method], decorator_list=[])], ns)
            tests = read(HEAD, TEST)
            execute([node for node in tests.body if not (isinstance(node, ast.ImportFrom)
                     and node.module == "verl.trainer.sft_trainer")], ns)
            results = []
            for name in sorted(name for name in ns if name.startswith("test_")):
                try:
                    ns[name]()
                    results.append(dict(test=name, passed=True))
                except (AssertionError, AttributeError) as error:
                    results.append(dict(test=name, passed=False, exception=type(error).__name__,
                                        traceback=traceback.format_exc()))
            assert len(results) == 3 and sum(item["passed"] for item in results) == (0 if ref == BASE else 3)
            original_results[ref] = results
            namespaces[ref] = ns

        # Own control: trainer.seed seeds loader base-seed streams, not the
        # DistributedSampler's independent shuffle stream (default seed=0).
        orders = {}
        for ref, ns in namespaces.items():
            version = {}
            for seed in (1, 97):
                trainer = ns["_build_trainer"]()
                trainer.config.trainer.seed = seed
                ns["SFTTrainer"]._build_dataloader(trainer)
                trainer.train_dataloader.collate_fn = torch.utils.data.default_collate
                epochs = []
                for epoch in (0, 1):
                    trainer.train_sampler.set_epoch(epoch)
                    epochs.append([batch.tolist() for batch in trainer.train_dataloader])
                version[str(seed)] = dict(sampler_seed=trainer.train_sampler.seed, epochs=epochs)
            assert version["1"] == version["97"]
            orders[ref] = version
        assert orders[BASE] == orders[HEAD]

        ns = namespaces[HEAD]
        trainer = ns["_build_trainer"](dp_rank=1, dp_size=2)
        train_before = trainer.train_dataloader.generator.get_state().clone()
        val_before = trainer.val_dataloader.generator.get_state().clone()
        cpu_before = torch.get_rng_state().clone()
        iter(trainer.val_dataloader)
        assert torch.equal(train_before, trainer.train_dataloader.generator.get_state())
        assert not torch.equal(val_before, trainer.val_dataloader.generator.get_state())
        assert torch.equal(cpu_before, torch.get_rng_state())
        split_control = dict(train_seed=trainer.train_dataloader.generator.initial_seed(),
                             val_seed=trainer.val_dataloader.generator.initial_seed(),
                             val_advanced=True, train_unchanged=True, global_cpu_unchanged=True)

        # Own later-epoch scope check, still zero-worker integer dataset. Match
        # current resumed batches/CPU RNG separately from private loader RNG.
        trainer = ns["_build_trainer"]()
        list(trainer.train_dataloader)  # completed first iterator advances private generator
        iterator = iter(trainer.train_dataloader)
        for _ in range(3):
            next(iterator)
        state = copy.deepcopy(trainer.train_dataloader.state_dict())
        cpu_state = torch.get_rng_state().clone()
        expected_batch = next(iterator).tolist()
        private_state = trainer.train_dataloader.generator.get_state().clone()
        future_base_seed = iter(trainer.train_dataloader)._base_seed
        resumed = ns["_build_trainer"]()
        resumed.train_dataloader.load_state_dict(state)
        torch.set_rng_state(cpu_state)
        resumed_iterator = iter(resumed.train_dataloader)
        actual_batch = next(resumed_iterator).tolist()
        assert actual_batch == expected_batch and torch.equal(cpu_state, torch.get_rng_state())
        private_equal = torch.equal(private_state, resumed.train_dataloader.generator.get_state())
        resumed_future_base_seed = iter(resumed.train_dataloader)._base_seed
        assert not private_equal and future_base_seed != resumed_future_base_seed
        later_control = dict(current_batch_equal=True, global_cpu_equal=True, loader_generator_equal=private_equal,
                             future_iterator_base_seed=future_base_seed,
                             resumed_future_iterator_base_seed=resumed_future_base_seed,
                             scope="Own zero-worker integer data control after a prior complete iterator; "
                             "private RNG/base-seed difference is not an observed worker augmentation or model divergence.")

    loader_path = Path(sys.modules[StatefulDataLoader.__module__].__file__)
    order_report = {
        ref: {seed: dict(sampler_seed=entry["sampler_seed"],
                         epoch_batch_counts=[len(epoch) for epoch in entry["epochs"]],
                         epoch_first_batches=[epoch[0] for epoch in entry["epochs"]],
                         complete_order_sha256=hashlib.sha256(json.dumps(entry["epochs"]).encode()).hexdigest())
              for seed, entry in version.items()}
        for ref, version in orders.items()
    }
    print(json.dumps(dict(base=BASE, head=HEAD, actual_merge_base=merge_base, python=platform.python_version(),
                         torch=torch.__version__, torch_git=torch.version.git_version,
                         torchdata=torchdata.__version__, omegaconf=omegaconf.__version__,
                         source_sha256=hashes, torchdata_loader_sha256=hashlib.sha256(loader_path.read_bytes()).hexdigest(),
                         original_test_bodies=original_results, own_sampler_order=order_report,
                         own_train_val_stream=split_control, own_later_iterator_scope=later_control,
                         scope="Complete original _build_dataloader, three original tests manually invoked, "
                         "real OmegaConf, Torch DistributedSampler, StatefulDataLoader, original mock engine. "
                         "Device resolver fixed CPU; collator constructor returns None then original test "
                         "replaces it with default_collate; torch.manual_seed redirected to default CPU only. "
                         "No pytest collection, actual trainer/manager, workers, dataset randomness, model, "
                         "optimizer, device RNG, distributed collectives or training."), indent=2))


if __name__ == "__main__":
    main()
