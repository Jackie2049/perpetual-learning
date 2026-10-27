# 推理调度器原函数控制

配合[调度器源码地图](../../notebook/rollout-performance-research.md#89-vllm-token预算与抢占)检查固定版本的计数、排序和预算规则。两条代码阅读路径不按文件或用例数计数；这里不是两个引擎的正确性、吞吐或RL复现验收。

源码须为 vLLM `eaf1627c01f553ea9a26b3d77f3cff8495dbf404` 与 SGLang `cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1`。脚本核对HEAD及九份所选文件与Git blob一致，只解析／提取原函数体，不导入框架或修改检出。

使用已有Python 3.12建立隔离环境，无第三方依赖。从项目根目录运行，两个源码路径由调用者提供：

```bash
uv --cache-dir /tmp/perpetual-c2-uv-cache venv --offline --python /path/to/python3.12 /tmp/perpetual-c2-venv
/tmp/perpetual-c2-venv/bin/python experiments/rollout-scheduler-controls/verify.py --vllm /path/to/vllm --sglang /path/to/sglang
```

2026-10-28使用uv创建的Python 3.12.14环境执行，[结果](result.json)为9份AST、8组原函数体控制通过，含8,700个MTP chunk边界组合。请求、队列、缓存和容量接口均为替身；不覆盖GPU分配、kernel、真实请求生命周期、并发／故障、跨节点或性能。首版替身不可hash导致集合操作失败，修正替身后通过，不是源码缺陷。
