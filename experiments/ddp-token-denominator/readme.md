# DDP 有效 token 分母验证

用两个真实本机进程核对默认DDP梯度平均与全局token均值，避免把单进程微批算术当分布式验证。原理与四条结果见[归一化笔记](../../notebook/fundamentals/gradient-accumulation-benchmark-rtx4090.md#ddp-的全局-token-分母)，[result.json](result.json)保留每rank输出。

## 运行

```bash
python -B experiments/ddp-token-denominator/verify.py
```

依赖现有Torch CPU/Gloo支持；实跑Python3.12.13、Torch2.9.1、git5811a8d7da873dd699ff6687092c225caffcf1bb。脚本强制Darwin lo0／其他系统lo，file rendezvous位于独立临时目录，process-group超时20秒；只需本机回环端口权限，不连接远端。

两spawn子进程验证PID互异、真实SUM token计数、DDP梯度归约及all_gather_object结果，不用后端替身。每条路径重建FP64单参数Linear／SGD，所有标签有效，单进程全量autograd作为参考。两次独立运行结果JSON完全相同；首个沙箱运行uv_bind报operation not permitted、子进程SIGABRT，不能算通过；获准仅本机CPU运行后通过。

本实验不运行历史OPT、verl／UniRL、GPU／NPU／SSH／VPN、性能测试、FSDP／SP／TP／AMP／自定义通信hook或全局空批；没有改动训练配方或新增M5任务。
