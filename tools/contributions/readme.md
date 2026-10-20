# 开源文档抬头检查

规范唯一来源：[open-source/WRITING.md](../../open-source/WRITING.md)。`summary.py` 由提交 hook 和网页构建共同使用，检查固定字段、十分制理由、来源链接及 What/Why/How 三点格式；网页读取同一解析结果，不另写一份摘要。

```bash
python3 tools/contributions/summary.py
python3 tools/contributions/summary.py --staged
```

默认检查工作区已跟踪的开源 Markdown；`--staged` 只检查暂存内容，未暂存的正确版本不能掩盖错误提交。删除文件无需抬头。新文件被暂存后也会检查。格式检查不判断技术结论真假、链接是否仍有效或文案是否足够易懂，作者仍需核对证据。

AReaL 的三份既有文档由 M5 负责，本次自动审批拒绝改写，暂列显式例外；新文档不豁免。一旦这三份加入规范抬头，也会正常检查。移除例外须在获准完成整改后进行。
