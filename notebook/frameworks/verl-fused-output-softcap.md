# verl 融合输出头与 logit softcap

通用 fused output head 直接消费hidden states和输出权重，绕过模型原始forward中的输出处理，因此必须保留原模型对logits的变换。省去 `c*tanh(z/c)` 不只是更换计算精度，会改变 log-prob、熵和梯度。verl [#7991](https://github.com/verl-project/verl/pull/7991)选择在不支持时拒绝通用 fused 路径，而不是补上 softcap kernel。

## 固定版本

2026-10-23读取API时PR为open，head `b145ec9b405ca93092c45b5f9e7995fb4604dbd2`，base及共同祖先 `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`（ahead2/behind0）。实际差异为4文件、61行新增，其中生产逻辑仅 `monkey_patch.py` 的7行检查；其余是44行CPU测试、性能文档提示和2026年贡献者license头支持。

[`dense_common.py`](https://github.com/verl-project/verl/blob/b145ec9b405ca93092c45b5f9e7995fb4604dbd2/verl/models/transformers/dense_common.py)和[`torch_functional.py`](https://github.com/verl-project/verl/blob/b145ec9b405ca93092c45b5f9e7995fb4604dbd2/verl/utils/experimental/torch_functional.py)在base/head逐字节相同。因此，下面对原torch输出头的观察不是“修复后kernel验收”。

## 拒绝范围

[`patch_forward_with_backends`](https://github.com/verl-project/verl/blob/b145ec9b405ca93092c45b5f9e7995fb4604dbd2/verl/models/transformers/monkey_patch.py#L240)先判断是否启用、backend是否在torch/triton/liger列表，再选择模型专用或通用adapter。新检查只在通用分支读取 `model.config.get_text_config()`，发现 `final_logit_softcapping` 为真值便抛ValueError，提示 `use_fused_kernels=False`。

| 入口条件 | 这段函数的行为 | 不能推出的结论 |
|---|---|---|
| fused关闭或backend不在列表 | 开头直接返回 | 不认证配置合法，也不撤销先前patch |
| 模型专用Qwen/GLM分支 | 沿原专用adapter选择 | 新guard没有审计这些adapter的softcap语义 |
| 通用分支，softcap为真值 | 在import通用adapter、设置backend属性和替换forward之前抛错 | 不是融合softcap实现或完整模型可用性证明 |
| 通用分支，softcap缺失或为假值 | 继续原backend选择 | 不认证模型不存在其他特殊输出变换 |

torch与liger都选 `forward_with_torch_backend`，后者再按实例backend属性分派；triton选另一入口。三者都会经过同一个通用guard，改backend名称不能绕过限制。检查不是一般的softcap数值范围校验，条件使用Python真值。

该函数成功后修改的是 `model.__class__.forward`，backend标记则在实例上。关闭fused只跳过这次操作，不自动恢复已修改的类。外层 `apply_monkey_patch` 还可能先安装TiledMLP、PrefixGrouper或attention patch，最后才调用这个检查；故“抛错前未替换此处forward”不能外推为整个初始化没有副作用或具备回滚。

## 为什么不只是舍入差异

通用torch路径直接用hidden states和 `lm_head.weight` 计算 `z=hWᵀ`，再除temperature后计算选中token的log-prob和熵，没有调用原模型的输出处理，也没有softcap参数。[HF v4.57.3 Gemma2源码](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/gemma2/modeling_gemma2.py#L565)则在LM-head投影后先除c、tanh、再乘c，之后才交给loss或作为logits返回。

本对照先执行这一模型输出变换，再施加temperature，比较 `softmax(c*tanh(z/c)/T)` 与 `softmax(z/T)`。softcap的导数还带有 `1-tanh²(z/c)`，不能只在最终log-prob上修一个常数。

本机合成CPU例子，`z=[0,60,120]`、选token2、T=1、c=30：

| 计算 | 选中token log-prob | 熵 |
|---|---|---|
| 原torch输出头，FP32 | 0 | 0 |
| raw logits参考，FP64 | 0（舍入） | 约5.25e-25 |
| softcap参考，FP64 | −0.297720 | 0.570416 |

这里没有模型、采样或训练收益测量；只是相同hidden/weight/label在两种分布定义下的直接对照，不将合成logits说成Gemma实测。

## CPU 对照与数值限制

[复查脚本](../../experiments/scripts/verify_verl_fused_softcap.py)从本地固定Git对象执行完整原 `torch_functional.py`，不删函数或替换反向实现。Liger/FlashAttention均须未安装，原模块正常进入torch fallback；不创建verl包壳、假模型或配置，也不导入Transformers/设备后端。

对照矩阵为2D/3D输入、chunk=1/2、T=0.7/1/2及log-prob单独、熵单独、二者组合三种反向目标，共36组。每组都用真实autograd核hidden/weight的梯度，并与独立FP64 raw/softcap参考比较。36组中，softcap语义差异均大于各自raw数值误差的100倍；每组分别取前向和梯度的最大绝对差，跨组最小值为29.8434和1.10977。这是本组输入的观察，不是全输入或每个token的误差下界。

最初对FP32参考设 `atol=rtol=2e-5` 时，熵相关梯度有一例超限（绝对差约1.57e-4）。改用FP64参考量化后，没有把容差调宽称为全过：36组中只有28组的四类最大绝对误差都在2e-5内。各类跨组最大值如下：

| 原FP32实现相对raw FP64参考 | 最大绝对差 |
|---|---|
| 选中token log-prob | 2.18e-6 |
| 熵 | 2.73e-6 |
| hidden梯度 | 1.61e-4 |
| weight梯度 | 1.34e-6 |

原熵公式含 `logsumexp(z) - sum(p*z)` 的大项相减；这里保留实际误差，不认证原手写反向与autograd数值精确等价，也不据此另立训练缺陷或修改误差容许标准。它与缺少softcap导致的上述分布变化是两个问题。

Python3.12.13 / torch2.9.1（git `5811a8d7da873dd699ff6687092c225caffcf1bb`），只运行CPU FP32原输出头和FP64参考：

```bash
python experiments/scripts/verify_verl_fused_softcap.py /path/to/verl
```

脚本还对base/head四个完整Python文件及head原测试模块做9次AST/内存compile，并核两个输出头文件未变。完整原实现SHA256为 `812d6db39676454f6bdb4881eedc9eceeef28c8e08a1b505bb54346c7b6a7485`；其他指纹随命令打印。

## 原测试与接续

作者[原测试](https://github.com/verl-project/verl/blob/b145ec9b405ca93092c45b5f9e7995fb4604dbd2/tests/models/test_fused_softcap_on_cpu.py)以真实HF Gemma2/Gemma3 config及无权重的PreTrainedModel覆盖3 backend×2嵌套布局×2softcap值×2启用状态，共24例。它检查拒绝时这段forward/backend选择未修改，不运行模型或三种kernel；作者报告24过，不是本机结果。本机缺Transformers，未执行这些测试，不用假配置替代。

原生guard回归首先需要匹配HF及verl导入依赖，不需要先申请GPU。若日后在已有获准模型任务中核对真实融合路径，还需匹配模型config/权重、原始forward及backend环境，比较相同hidden/label/temperature下的输出和梯度。真机参考补入[M5原一致性入口](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动部署候选、下载模型或增加训练队列；与#8250没有已证因果关系，AReaL/#8313/#523分工不变。
