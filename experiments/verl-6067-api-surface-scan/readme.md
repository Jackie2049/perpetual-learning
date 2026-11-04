# verl #6067 重构的接口面扫描（实验）

**目的**：上一批 [`verl-6067-config-consumer-scan`](../verl-6067-config-consumer-scan/readme.md) 查的是"配置字段还在、没人读"。
本批换一个信号族：**被删掉的接口名和文档引用**——#6067 整文件删除的 10 个生产模块里，哪些公开方法/函数名在当前 main 上已经没有了，而文档还在教用户用。

**结论（详见[证据文档](../../open-source/verl/evidence/engine-refactor-api-surface/readme.md)）**：50 个公开名字里 32 个仍有同名定义；13 条"无定义但非历史文档仍在引用"去重后是 3 个名字（`update_policy`、`update_critic`、`compute_values`）；**代码侧调用点 0 处 ⇒ 无静默失效**。另查出一条结构性缺口：`@DistProfiler.annotate(role="compute_values"|"critic_update")` 随旧 worker 删除，engine 侧 0 处，但前端白名单仍接受这两个 stage。

## 文件

| 文件 | 作用 |
|---|---|
| `scan_api_surface.py` | 从 `044bbba2^` 的 10 个被删模块抽公开方法/函数名，在当前 main 查 `def` 与 `docs/**` 引用，四档判定 |
| `verify_findings.py` | 对 7 条判定逐条写死证据（正则／`main_uses` 断言／装饰器 role 计数），跑一遍必须 0 不一致；`--selftest` 往 main 副本种同名定义验证判据会翻转 |
| `scan-result.json` | `scan_api_surface.py` 输出（50 条，可复现） |
| `findings.json` | `verify_findings.py` 输出（7 条判定，0 不一致） |

## 复现

```bash
# 需要一台有 verl 仓库的机器（本目录脚本只读该仓库，不改它）
python3 scan_api_surface.py --repo <verl 仓库> --json scan-result.json
python3 verify_findings.py --repo <verl 仓库> --json findings.json   # 期望 exit 0
python3 verify_findings.py --repo <verl 仓库> --selftest             # 期望"自检通过"
```

本轮实测环境：verl main `74543265`（2026-11-04 fetch），父提交 `044bbba2^`（#6067 于 2026-04-20 合入），Python 3.12.13，macOS arm64。**未跑 verl、未上 GPU/NPU**。

## 判据与边界

- **历史文档排除**：`docs/blog/` 下的发布说明写"当年有什么"，引用已删接口是对的，扫描器单独排除、不参与判定（本轮命中 0 条）。
- **"有定义"一律算没问题**：`__init__`／`forward` 这类跨文件重名会掩盖真问题，宁可漏报；本轮 `IGNORED_NAMES` 显式剔除了协议样板。
- **"无定义"不等于"功能没了"**：本批 3 个名字都属**改名/重新分层**，代码侧无调用点；profiler stage 那条才是"功能可能真的没接上"，但静态只能说到"无装饰器触发点"。
- **看不见的**：装饰器 role 这类**字符串常量**（`scan_api_surface.py` 抓不到，靠人工读 diff + `verify_findings.py` 单独计数）；`docs/blog` 之外的发布类材料；`recipe` 子模块（`160000 commit`，未检出）。

## 脚本自身的坑（写在这里免得下次重犯）

1. **`update_critic`／`compute_values` 不是"被删了"，是"没有同名 RPC 了"**：trainer 改成 `critic_wg.train_mini_batch` / `critic_wg.infer_batch`。把它们和 `update_policy`（确实改名成 `update_actor`）混成一条写，会把重新分层说成删除。
2. **文档引用的正则必须从实际文件抄**：第一版三条锚点凭记忆写（中英文混写、路径拼错、行首符号写错），全部 MISMATCH；`verify_findings.py` 的逐条判据正是靠这三次失败才校准。
3. **`verify_findings.py --selftest` 只覆盖 4 条**（`update_policy`／`update_critic`／`compute_values`／`critic_update`）；另两条 `doc_drift` 用的注入是单独手工跑的（把文档旧名去掉 ⇒ 判据翻 False；给 main 种一个 `annotate(role="compute_values")` ⇒ role 计数判据翻 False），**未固化进脚本**。
