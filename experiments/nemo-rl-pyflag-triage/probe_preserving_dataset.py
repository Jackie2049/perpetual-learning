#!/usr/bin/env python3
"""`use_preserving_dataset`：两档在同一份异构 tool 数据上到底输出什么，以及差异随 `datasets` 版本怎么走。

取材方式：从 raw/snapshot-*.py（上游 d82b79c 的逐字节副本）用 AST 取出真实类/函数体**原样 exec**。
为本机装不上的依赖提供替身，替身清单打印进读数：
  * `RawDataset`（上游基类）、`torch.Tensor`（注解）、`nemo_rl.data.multimodal_utils` 四个名字、
    `nemo_rl.distributed.batched_data_dict.BatchedDataDict`、tokenizer 的 `return_tensors="pt"` 包装
真实对象：`OpenAIFormatDataset`、`PreservingDataset`、`get_formatted_message_log`、`sft_processor`、
`datasets.load_dataset`、`transformers` 的 tokenizer 与 chat template。

判据分组：
  CF-* 当前锁定版本（datasets 5.0.1）下两档的实际输出
  CH-* 用户的 YAML 键能不能走到这个形参（静态）
  HV-* 同一份数据在历史锁定版本（datasets 4.0.0，即 #1181 引入时的锁）下的输出
退出码：0 全 PASS / 1 有 FAIL / 2 自检未通过。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import tempfile
import types
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
UPSTREAM_SHA = "d82b79c58f175917182ed7f53283ae590af67924"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exec_nodes(path: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    found = {node.name for node in nodes}
    if found != names:
        raise AssertionError(f"{path.name} 缺定义: 期望 {names}, 实得 {found}")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def build_namespace() -> tuple[dict, list[str]]:
    import datasets
    import numpy as np
    from transformers import PreTrainedTokenizerBase

    stubs: list[str] = []

    class RawDatasetStub:
        pass

    stubs.append("RawDataset（上游基类，只提供 set_task_spec/set_processor）")

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    stubs.append("torch.Tensor（仅类注解）")

    class _PackedTensor:
        pass

    stubs.append("nemo_rl.data.multimodal_utils 四个名字（本案走纯文本路径，不触多模态）")
    stubs.append("nemo_rl.distributed.batched_data_dict.BatchedDataDict")

    class BatchedDataDict(dict):
        pass

    ns: dict = {
        "__name__": "nemo_rl.data.llm_message_utils",
        "warnings": warnings,
        "Mapping": dict,
        "Sequence": list,
        "Any": object,
        "Optional": object,
        "Union": object,
        "cast": lambda t, v: v,
        "np": np,
        "torch": torch_stub,
        "Dataset": datasets.Dataset,
        "PreTrainedTokenizerBase": PreTrainedTokenizerBase,
        "load_dataset": datasets.load_dataset,
        "json": json,
        "Callable": object,
        "RawDataset": RawDatasetStub,
        "PackedTensor": _PackedTensor,
        "extract_multimodal_model_inputs": lambda *a, **k: {},
        "get_multimodal_default_settings_from_processor": lambda processor: {},
        "load_media_from_message": lambda message, **k: {},
        "BatchedDataDict": BatchedDataDict,
        "defaultdict": __import__("collections").defaultdict,
        "os": __import__("os"),
        "hashlib": hashlib,
        "re": __import__("re"),
    }

    iface = exec_nodes(
        RAW / "snapshot-interfaces.py",
        {"TaskDataSpec"},
        {
            "Optional": __import__("typing").Optional,
            "PathLike": str,
            "os": __import__("os"),
            "dataclass": __import__("dataclasses").dataclass,
            "Any": object,
        },
    )
    ns["TaskDataSpec"] = iface["TaskDataSpec"]
    ns["LLMMessageLogType"] = list
    ns["FlatMessagesType"] = list

    ns.update(
        exec_nodes(
            RAW / "snapshot-llm_message_utils.py",
            {"get_first_index_that_differs", "get_formatted_message_log"},
            ns,
        )
    )
    ns.update(exec_nodes(RAW / "snapshot-processors.py", {"sft_processor"}, ns))
    exec_nodes(
        RAW / "snapshot-oai_format_dataset.py",
        {"PreservingDataset", "OpenAIFormatDataset"},
        ns,
    )
    return ns, stubs


def sample_hetero_tools():
    """两条样本的 tool 定义异构：样本 2 多一个 `strict` 键、参数多一个 `precision`。"""
    return [
        {
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query":"x"}'},
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                }
            ],
        },
        {
            "messages": [
                {"role": "user", "content": "1+1?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "calc", "arguments": '{"expr":"y"}'},
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calc",
                        "description": "calculator",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expr": {"type": "string"},
                                "precision": {"type": "integer"},
                            },
                        },
                    },
                }
            ],
        },
    ]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_tokenizer():
    """真实的 transformers tokenizer；本机无 torch ⇒ 只把 `return_tensors="pt"` 换成 numpy 形状。"""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
    )
    fast.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}"
        "{% if m.get('tool_calls') %}|TC={{ m['tool_calls'] }}{% endif %}\n{% endfor %}"
        "{% if tools %}|TOOLS={{ tools }}{% endif %}"
    )

    class NumpyTokenIds:
        def __init__(self, ids):
            self._ids = ids

        def __getitem__(self, key):
            if key != "input_ids":
                raise KeyError(key)
            return [self._ids]

    class NoTorchTokenizer:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def __call__(self, *args, **kwargs):
            kwargs.pop("return_tensors", None)
            enc = self._inner(*args, **kwargs)
            return NumpyTokenIds(list(enc["input_ids"]))

    return NoTorchTokenizer(fast)


def tool_functions(rows) -> list[dict]:
    return [row["tools"][0]["function"] for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    import datasets as hf_datasets
    import transformers

    ns, stubs = build_namespace()
    OFD = ns["OpenAIFormatDataset"]
    PreservingDataset = ns["PreservingDataset"]
    sft_processor = ns["sft_processor"]
    TaskDataSpec = ns["TaskDataSpec"]
    tokenizer = make_tokenizer()

    results = []

    def check(name: str, ok: bool, got, want) -> None:
        results.append({"name": name, "ok": bool(ok), "got": got, "want": want})

    with tempfile.TemporaryDirectory() as tmp:
        hetero = Path(tmp) / "hetero.jsonl"
        write_jsonl(hetero, sample_hetero_tools())

        def load(preserving: bool):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                obj = OFD(str(hetero), use_preserving_dataset=preserving)
            return obj, [str(item.message) for item in caught]

        off, off_warn = load(False)
        on, on_warn = load(True)
        off_rows = [off.dataset[i] for i in range(len(off.dataset))]
        on_rows = [on.dataset[i] for i in range(len(on.dataset))]

        is_lock = hf_datasets.__version__.startswith("5.")
        tag = "CF(锁 5.0.1)" if is_lock else "CF(非锁定版本)"

        check(
            f"{tag}-01 两档都构造成功、样本数正确（读数不是 <absent>）",
            len(off_rows) == 2 and len(on_rows) == 2,
            [len(off_rows), len(on_rows)],
            2,
        )
        check(
            f"{tag}-02 开启档确实走 PreservingDataset，关闭档走 HF Dataset",
            isinstance(on.dataset, PreservingDataset)
            and not isinstance(off.dataset, PreservingDataset),
            [type(on.dataset).__name__, type(off.dataset).__name__],
            ["PreservingDataset", "Dataset"],
        )
        check(
            f"{tag}-03 关闭档在检测到 tools 时告警指名该开关、开启档不告警",
            any("use_preserving_dataset=True" in m for m in off_warn)
            and not any("use_preserving_dataset=True" in m for m in on_warn),
            {"off": len(off_warn), "on": len(on_warn)},
            {"off": "≥1", "on": 0},
        )

        # 两档输出是否逐字节相同 —— 这才是"开关有没有作用"的判据
        same_tools = [json.dumps(r["tools"], sort_keys=True) for r in off_rows] == [
            json.dumps(r["tools"], sort_keys=True) for r in on_rows
        ]
        if is_lock:
            check(
                "CF-04 锁定版本下两档输出逐字节相同 ⇒ 开关不改变任何数据",
                same_tools,
                {"same": same_tools,
                 "off_fn0_keys": sorted(tool_functions(off_rows)[0]),
                 "on_fn0_keys": sorted(tool_functions(on_rows)[0])},
                "same=True",
            )
            check(
                "CF-05 关闭档也未给样本 1 补出 `strict`（datasets 5.x 对异构 dict 退回 Json 类型）",
                "strict" not in tool_functions(off_rows)[0],
                sorted(tool_functions(off_rows)[0]),
                "无 strict",
            )
        else:
            check(
                "CF-04 旧版 datasets：关闭档给样本 1 补出 `strict=None`（异构 schema 被破坏）",
                "strict" in tool_functions(off_rows)[0]
                and tool_functions(off_rows)[0]["strict"] is None,
                {"off_fn0": tool_functions(off_rows)[0]},
                'strict 存在且为 None',
            )
            check(
                "CF-05 旧版 datasets：开启档保留原结构（样本 1 无 strict）",
                "strict" not in tool_functions(on_rows)[0],
                sorted(tool_functions(on_rows)[0]),
                "无 strict",
            )

        # 差异是否传到训练样本
        spec = TaskDataSpec(task_name="probe")
        proc_off = sft_processor(dict(off_rows[0]), spec, tokenizer, 4096, 0)
        proc_on = sft_processor(dict(on_rows[0]), spec, tokenizer, 4096, 0)
        ids_off = [list(m.get("token_ids") or []) for m in proc_off["message_log"]]
        ids_on = [list(m.get("token_ids") or []) for m in proc_on["message_log"]]
        blob_off = json.dumps(proc_off["message_log"], default=str)
        blob_on = json.dumps(proc_on["message_log"], default=str)
        check(
            "DS-01 tools 确实进了 chat template（渲染文本含 TOOLS=）",
            "TOOLS=" in blob_off and "TOOLS=" in blob_on,
            {"off": "TOOLS=" in blob_off, "on": "TOOLS=" in blob_on},
            True,
        )
        if is_lock:
            check(
                "DS-02 锁定版本下两档训练样本逐字节相同（开关对 loss 输入无影响）",
                blob_off == blob_on and ids_off == ids_on,
                {"blob_same": blob_off == blob_on, "ids_same": ids_off == ids_on},
                {"blob_same": True, "ids_same": True},
            )
        else:
            check(
                "DS-02 旧版 datasets：两档训练样本不同（默认档凭空多出 strict:null）",
                blob_off != blob_on,
                {"off_has_strict_null": "strict': None" in blob_off or '"strict": null' in blob_off,
                 "on_has_strict": "strict" in blob_on},
                "不同",
            )

        # ---- CH：可达性 ----
        init_src = (RAW / "snapshot-response_datasets-init.py").read_text()
        check(
            "CH-01 注册表把 'openai_format' 直接绑到 OpenAIFormatDataset（无 partial/包装）",
            '"openai_format": OpenAIFormatDataset' in init_src,
            "在位",
            "在位",
        )
        check(
            "CH-02 构造点是 dataset_class(**data_config)，配置键直接成关键字实参",
            "**data_config" in init_src,
            "在位",
            "在位",
        )
        utils_src = (RAW / "snapshot-datasets-utils.py").read_text()
        check(
            "CH-03 上游 unsupported-key 告警清单（_BEHAVIORAL_DATASET_CONFIG_KEYS）里没有这个键",
            "use_preserving_dataset" not in utils_src,
            "不在清单里",
            "不在清单里（⇒ 写错也不会有提示，只能靠本条探针）",
        )
        data_init = (RAW / "snapshot-data-init.py").read_text()
        check(
            "CH-04 ResponseDatasetConfig(TypedDict) 无该字段，但 TypedDict 不做运行时校验 ⇒ 不拦截",
            "use_preserving_dataset" not in data_init,
            "无字段",
            "无字段",
        )
        docs = (RAW / "snapshot-sft-doc-excerpt.md").read_text()
        check(
            "CH-05 官方 sft 指南把它写成可用的公开开关",
            "use_preserving_dataset: true" in docs,
            "出现",
            "出现",
        )

        if args.selftest:
            print("=== selftest 1：把开关改成恒走 HF 分支，判据必须变红 ===")
            target = RAW / "snapshot-oai_format_dataset.py"
            original = target.read_text()
            injected = original.replace(
                "        if not use_preserving_dataset:", "        if True:  # injected", 1
            )
            if injected == original:
                print("注入未生效 → 自检失败")
                return 2
            target.write_text(injected)
            try:
                ns2, _ = build_namespace()
                obj2, _ = (
                    ns2["OpenAIFormatDataset"](str(hetero), use_preserving_dataset=True),
                    None,
                )
                rows2 = [obj2.dataset[i] for i in range(len(obj2.dataset))]
                # 注入后：请求 True 却走了 HF 分支 ⇒ CF-02（类型）必须红
                red_type = not isinstance(obj2.dataset, ns2["PreservingDataset"])
                # 且如果版本会补 None，CF-05 也应红
                red_fill = "strict" in tool_functions(rows2)[0]
                print(f"CF-02 类型判据变红: {red_type}；CF-05 补齐判据变红: {red_fill}")
                if not red_type:
                    print("自检失败：注入后 CF-02 仍绿")
                    return 2
                if is_lock and red_fill:
                    print("注：锁定版本下注入会让 CF-04/CF-05 的 same 结论翻转")
            finally:
                target.write_text(original)

            print("=== selftest 2：把告警文案里的键名删掉，CF-03 必须变红 ===")
            injected2 = original.replace(
                "Set use_preserving_dataset=True to preserve heterogeneous tool schemas. ", "", 1
            )
            if injected2 == original:
                print("注入未生效 → 自检失败")
                return 2
            target.write_text(injected2)
            try:
                ns3, _ = build_namespace()
                _, warns3 = ns3["OpenAIFormatDataset"](str(hetero), use_preserving_dataset=False), None
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    ns3["OpenAIFormatDataset"](str(hetero), use_preserving_dataset=False)
                msgs = [str(w.message) for w in caught]
                red = not any("use_preserving_dataset=True" in m for m in msgs)
                print(f"CF-03 告警判据变红: {red}")
                if not any("Tools detected" in m for m in msgs):
                    print("自检失败：连告警本身都没出现，注入前的前提不成立")
                    return 2
                if not red:
                    print("自检失败：注入后 CF-03 仍绿")
                    return 2
            finally:
                target.write_text(original)
            print("selftest OK")

    failed = [item for item in results if not item["ok"]]
    report = {
        "upstream_commit": UPSTREAM_SHA,
        "repo": "NVIDIA-NeMo/RL",
        "env": {
            "python": sys.version.split()[0],
            "datasets": hf_datasets.__version__,
            "transformers": transformers.__version__,
        },
        "snapshots": {p.name: sha256(p) for p in sorted(RAW.glob("snapshot-*.py"))},
        "substitutes": stubs,
        "checks": results,
        "failed": len(failed),
    }
    for item in results:
        print(f"[{'PASS' if item['ok'] else 'FAIL'}] {item['name']}")
        print(f"        got={json.dumps(item['got'], ensure_ascii=False, default=str)[:500]}")
        print(f"        want={json.dumps(item['want'], ensure_ascii=False, default=str)[:300]}")
    print(f"\ndatasets={hf_datasets.__version__} transformers={transformers.__version__}")
    print(f"{len(results)} checks / {len(failed)} failed")
    print("替身:", "; ".join(stubs))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
