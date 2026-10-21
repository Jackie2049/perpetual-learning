# DeepInfer 关闭控制流检查

固定#175的13项原测试断言，针对base和候选的三个隔离原方法执行；资源均沿用原测试Mock。目的是检查清理顺序、异常保留及不重试标志，不验证真实线程/进程、原生扩展或设备释放。分析和结果见[主笔记](../../notebook/frameworks/deepinfer-initial-survey.md#关闭失败后的独立清理与一次性标志)。

Python≥3.12、已有pytest、git和本地DeepInfer仓库，预先具有BASE `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` 与PR `1dc73e9e8bb8d6099a50b163b9fc3a8351c955f2` 对象；缺对象时停止，不自动fetch/安装/检出。本轮实际用3.12.13/pytest8.4.2。

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/deepinfer-close/verify.py /path/to/DeepInfer --implementation base
PYTHONDONTWRITEBYTECODE=1 python experiments/deepinfer-close/verify.py /path/to/DeepInfer --implementation pr175
```

stdout打印源码SHA256、pytest摘要与选择范围；stderr可能保留基线析构异常。脚本核对基线13项中恰有指定10项失败、pytest退出1，候选13全过、pytest退出0；脚本自身正常结束表示观测符合此固定对照，不表示基线测试通过。1项真实本地子进程清理测试明确排除，不启动其sleep进程。禁用上游conftest、外部pytest插件与缓存；诊断文件名只存linecache，不写进源码checkout。
