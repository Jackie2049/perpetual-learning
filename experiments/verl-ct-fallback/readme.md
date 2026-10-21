# verl 连续 token builder 回退验证

复查已有[#8311](https://github.com/verl-project/verl/pull/8311)的原CPU测试，区分model_type选择、token存在性与真实模板边界。结论见[原主题笔记新版补充](../../notebook/frameworks/verl-6779-continuous-token-agentic-rollout-reading.md#11-新版自动选择与-tokenizer-能力回退)。

## 输入与运行

候选：`9fc9e2646e4543c3cba1eef8f490622499583911`；目标base：`cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`；共同祖先/实跑基线：`5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517`。五个加载的实现文件在祖先与目标base没有diff；真实PR三文件85增0删，不能用目标分支的其他变化当本PR改动。

使用Python3.12.13/pytest8.4.2，无新增安装。准备含固定对象的上游克隆后，在本项目根执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python experiments/verl-ct-fallback/verify.py /path/to/verl
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python experiments/verl-ct-fallback/verify.py /path/to/verl 5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517
```

[verify.py](verify.py)以GIT_NO_LAZY_FETCH=1读固定Git对象，无自动fetch或源码落盘。注册空包壳跳过verl根初始化，原样加载tokenizer.py、continuous_token.py、deepseek.py、continuous_token_wiring.py。现有环境缺Transformers，chat_template.py只AST选原apply_chat_template全函数，启用延迟注解而不导入其两个仅用于类型标注的Transformers类型；函数体不改、无模板替身。其余模块及原测试完整compile/exec，保留原参数化/断言。

Pytest自定义collector从内存测试模块收集，以脚本路径作锚点；禁用上游conftest、外部插件、缓存和字节码。audit hook拒绝socket操作。测试使用原合成tokenizer/processor；不导入torch/transformers/ray/vllm，不下载真实模型/tokenizer，不调用HF加载函数或运行AgentLoop。

## 2026年10月21日结果

原tests/utils/test_continuous_token_on_cpu.py共172项：

| 输入 | 结果 | 差异 |
|---|---|---|
| 祖先实现＋候选原测试 | 169过、3失败，退出1 | 缺im_end自动回退抛错、unk映射未选默认builder、单元素unk列表未拒绝 |
| 候选实现＋同一原测试 | 172过，退出0 | 原断言全部满足 |

五个新增测试中，显式Qwen缺token仍抛错、unk token自身合法两项基线也通过。三项基线失败原样保留，不改xfail。原suite还检查有效Qwen边界换行、非前缀稳定模板拒绝、mask/logprob编辑对齐与合成VL工厂等；这些不是对真实R1-Distill/Jinja/HF processor的集成证明。

候选只在auto、Qwen文本家族、没有图像processor且im_end不可用时回退；显式家族和VL路径不由该guard吞掉错误。require_token_id的unk校验则作用于所有调用该helper的builder，不只auto分支。成功构造不检验真实模板的多轮边界；默认builder保留suffix-diff拒绝非前缀稳定模板的条件。

## 输入校验

SHA256：continuous_token.py基线 `1943099099f30b6a47e146ec8af7053be215d03bf23dd8776083712b2a2d01c0`、候选 `256f1c7e483eb55fd0f4e75e370d0f207cc424838d40f2b89453772ee6f9ce54`；wiring基线 `fc6733fd020dfa085acf112d33e3b71e73c4f805da0e1aa86c07ecb26e37e1f6`、候选 `7628c29f6d3787f7b600a8a82cf16e838b97db83bc123adb56033a7c06925582`；原测试 `1b290b6aba1ce19be3a1740f3e0de130757e40e3f1913ae0d25ffe45fbe597fa`。脚本同时打印三个未改辅助文件hash。

作者报告的真实R1 tokenizer和A100训练结果不是本机复跑；本题不新增M5 GPU任务、不自动采用补丁或竞争作者实现。
