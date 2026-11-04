# SkyRL pyflag 候选逐条取证（2026-11-04）

## 一句话结论

2026-11-02 用 [pyflag](../../tools/pyflag/readme.md) 扫 SkyRL 报的 7 条「有消费者、无生产者」，
**在当前 HEAD（`1089e18`，2026-11-04）上逐条查证：真正失效的只有 1 条——`use_liger_kernel`，
而且它不是「配置项被写坏」，是「形参从没接过线、config 里连字段都没有」。**
其余 6 条分别是：1 条内部形参（工具的假阳性）、1 条弃用开关、2 条属性传参（工具已知盲区）、
1 条公开 helper API、1 条「分支靠第二个或项恒成立」。

⇒ TM-007 此前记的「`use_liger_kernel`、`use_remove_padding` **确认为真**」，
**`use_remove_padding` 那条要降级**：它确实从没被传过，但分支里还有
`or ulysses_sp_size > 1`，sp>1 时无条件进分支，而唯一调用点又在 `sp > 1` 的守卫里
⇒ 这个形参在本仓库**不产生任何观测差异**，谈不上「用户设了没用」。

## 为什么值得做这一轮

上一轮（2026-11-02）的结论只有一句「确认为真」，既没写**怎么确认的**，也没写
**在当时哪个 commit 上确认**，两轮之后没人能复核。这一轮把「确认」变成可复跑的东西：
每条判定绑定 2–3 条**可在源码里查证的字符串事实**，脚本任一事实对不上就 `exit 1`。

## 七条逐条

| # | 开关 | 判定 | 依据（都可 grep 到） |
|---|---|---|---|
| 1 | `use_liger_kernel` | **DEAD** | 全仓零关键字实参；`config.py` 无该字段；分支 `if use_liger_kernel:` + `AutoLigerKernelForCausalLM` 真实存在 |
| 2 | `use_remove_padding` | **NOT_USER_FACING** | `monkey_patch.py:153` 是 `if use_remove_padding or ulysses_sp_size > 1:`；唯一调用点 `worker.py:168` 在 `if self.sequence_parallel_size > 1:` 内 |
| 3 | `enable_sample_support_replay` | INDIRECT_PRODUCER | `worker.py:1122` 读配置字段、`:1166` 传到形参（pyflag 盲区） |
| 4 | `disable_print` | PUBLIC_HELPER_API | `skyrl_gym/envs/registration.py:113` docstring 明写「返回字符串而不是打印」；skyrl-gym 是独立发布包 |
| 5 | `use_combine` | FALSE_POSITIVE | 不在任何 `*Config` 里；子类 `Qwen4ExpFinalMixer` 走 **位置参数** `super().__init__(config, use_combine=False)` |
| 6 | `enable_pd` | INDIRECT_PRODUCER | `inference_servers/setup.py:94` 读 `ie_cfg.enable_pd`，`:133`／`:155` 两处 `enable_pd=True` |
| 7 | `use_signals` | DEPRECATED_BY_DESIGN | `py_functional.py:57` docstring 标 `(Deprecated)`，默认走多进程 |

`mode` 与判定不是一回事：第 2 条是 `always`（"关不掉"），第 5 条也是 `always`，
但两者的原因完全不同——一个是**第二个或项**，一个是**位置参数传 True 的默认值**。

## 唯一真失效的那条，到底能不能开

不能，而且不是「开了不生效」这种软失败：

- `HFModelWrapper.__init__` 有 `use_liger_kernel=False` 形参、有 `if use_liger_kernel:` 分支、
  分支里 import `liger_kernel.transformers.AutoLigerKernelForCausalLM`；
- `fsdp_worker.py` 构造 `HFModelWrapper` 时传了 15 个关键字（`use_flash_attention_2`、
  `lora_rank`、`remove_microbatch_padding` …），**没有** `use_liger_kernel`；
- `skyrl/train/config/config.py` 里**没有** `use_liger_kernel` 字段 ⇒ 连 `+use_liger_kernel=True`
  这种绕路都没有（Hydra/OmegaConf 侧也不接受）。
- 依赖侧：`pyproject.toml` 与 `uv.lock` 里都没有 `liger`。

历史归属：`model_wrapper.py` 自 2026-02-16 起的**全部 18 次提交**（GitHub contents API 逐版本取文件）里，
它的 `use_liger_kernel` 计数**恒为 3**（docstring + 默认值 + 分支），而同一版本 `fsdp_worker.py`
恒为 **0**。⇒ 在**这 18 个 ref 上**从 2026-02-16 到 2026-10-30 **一次都没有接过线**，不像某次重构漏掉的。

**影响面**：配不了、也没人配，所以**当前没有任何用户的训练受影响**。
这不是回归，是一段「看起来能配」的死开关。⇒ 对外提 issue 的价值评级 **3/10**：
只有「文档/docstring 说 `Defaults to False`，读者会以为可以传」这一点值得说，
本身与 verl 那种「字段在、消费者没了」的回归不是同一类问题。

## 脚本

```bash
# 需要一份 SkyRL checkout
python3 verify_findings.py /path/to/SkyRL --json findings.json   # 判定，不一致 exit 1
python3 verify_findings.py /path/to/SkyRL --selftest             # 8 条注入，全部必须翻转
```

`--selftest` 的 8 条里有一条是**本轮真正踩到的坑**：v1 的「无生产者」判据用
`"use_liger_kernel=" in text` 做子串匹配，把 `model_wrapper.py:148` 的**默认值声明行**
算成了生产者，读数当场反了（脚本报「有生产者」，而实际零调用）。改成
「跳过 `def` 行 + 要求前面有 `调用名(`」后恢复正常，selftest 第 6 条专门钉这三种形状：
声明行、`_v2` 这类前缀名、`cfg.use_liger_kernel=1` 这种属性赋值。

## 没做到的（别当结论引用）

- **扫描只查了 `1089e18` 这一个 HEAD**（浅克隆深度 1）。历史归属是走 GitHub contents API 逐版本取文件的：
  覆盖了 `model_wrapper.py` 的全部 18 次提交，但**没有**对 `fsdp_worker.py` 自己的提交历史做独立遍历，
  也不是全仓 bisect。同类查询对 `skyrl/backends/skyrl_train/workers/model_wrapper.py` 的第 2 页返回 0 条，
  说明这 18 个就是该路径可查的全部提交（GitHub 按路径查历史有 100 条上限，这里远未触顶）。
  ⇒ 能说「该文件自 2026-02-16 起没接过线」，不能说「整个仓库历史上从没有过」。
- **没有跑任何训练**。`use_liger_kernel` 那条的「配不了」是读代码读出来的，
  没有真跑一次确认 Hydra 报错——本仓库的配置系统不是 Hydra 直配，构造路径比 verl 长。
- **`skyrl-agent/examples/run_verl/*.sh` 里的 `actor_rollout_ref.model.use_remove_padding=True`
  是 verl 的键**，与 SkyRL 的 `apply_monkey_patch(use_remove_padding=…)` 不是同一个东西。
  v1 判据差点把这两个混算成「有生产者」——正是本轮把「同名字符串」与「同一个形参」分开的原因。
- 上游 `git clone` 直连失败（超时），副本经 `gh-proxy.com` 拉取；`git ls-remote` 返回 0 引用，
  说明只读代理的 git 协议不通但 HTTP 文件代理可用。**这属于本机网络状况，不构成对 SkyRL 的任何陈述。**

## 产物

- `scan-skyrl-1089e18.json` — pyflag 原始扫描（38 个开关、7 条 silently_inert，`root` 已改成副本目录名）
- `findings-skyrl-1089e18.json` — 本脚本输出的逐条判定与事实表
- `verify_findings.py` — 判定 + selftest

入口：[pyflag 工具说明](../../tools/pyflag/readme.md) · [TM-007](../../queue_tech-miner.md) · [diary](../../diary/2026-11-04.md)
