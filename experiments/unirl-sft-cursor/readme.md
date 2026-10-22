# UniRL SFT 预取与数据游标复查

固定官方 `6fd82cba13f52f394747832344039408699ea378`，结论集中在[原调研](../../notebook/frameworks/unirl-survey.md#sft-预取与数据消费游标)。这是本项目的学习实验，不是修改 UniRL 或给它恢复 tests 目录。

```bash
python -B experiments/unirl-sft-cursor/verify.py /path/to/UniRL
```

Python3.12.13、torch2.9.1（既有CPU环境）；不安装完整框架。脚本核原data/sft.py与trainer/sft.py hash，真实读取此目录的5行训练、3行验证和同长度内容变更JSONL。所有夹具只读，不写 checkpoint 或修改源仓库；`-B` 避免字节码文件。

仅用带真实 `__path__` 的包壳跳过 `unirl.distributed.tensor.__init__` 的 Ray worker-local transport 汇总导入；原 Batch、dataset、normalizer 和完整 SupervisedDataSource 仍按原代码加载，没有假类、方法摘取或修改逻辑。没有导入 Ray/Hydra/OmegaConf，也不调用设备发现。

观察包括跨epoch peek/commit/恢复、三个重入保护、shuffle开关各10批与直读对照、全局random状态、seed/shuffle/内容身份拒绝、缺fingerprint的legacy兼容警告、验证2+1行不影响训练pending。全部通过；legacy警告是预期结果，不是报错或身份已验证。

没有运行上游原测试、trainer、worker-side prefetch、optimizer、磁盘checkpoint/故障恢复或设备任务。游标表示数据消费，不表示optimizer成功，也不提供exactly-once训练保证。
