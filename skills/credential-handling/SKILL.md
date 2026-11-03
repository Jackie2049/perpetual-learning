---
name: credential-handling
description: 在仓库里处理密码 / token / 私钥的实操规则。适用于发现明文凭据、把凭据移出代码、以及记录这类工作的时候。源自 2026-11-03 一次真实的凭据清理（两个 SSH 密码、6 个文件、最久已入库 5 个月）。
---

# 凭据处理：清理、改造、记录

## 什么时候用

- 发现仓库里有明文密码 / token / 私钥
- 要把硬编码的凭据改成从环境变量读
- **要写一段"我清理了凭据"的记录**

## 一、发现：用模式扫，不要搜已知值

**案例**：第一次发现一个 SSH 密码后，我只搜了那一个值、清掉了它。第二轮改用模式扫描，**又找到另一个完全不同的密码**，连的是另一台主机。

```bash
# 对（模式）
grep -rInE "sshpass -p '[^']+'|password\s*[:=]\s*['\"][^'\"]{4,}" --exclude-dir=.git .

# 错（只搜已知值）—— 只能找到你已经知道的那个
grep -rn "<已知密码>" .
```

或者直接用 [`tools/anchors/credential_scan.py`](../tools/anchors/credential_scan.py)。

**知道一个泄露 ≠ 知道全部泄露。**

## 二、改造：从环境变量读，缺省时安全地不做

**不要**直接删掉密码——那会让脚本失效。

```python
def _ssh_cmd():
    pw = os.environ.get('GPU_SSH_PASS')
    if not pw:
        return None          # 调用方显式跳过并提示
    return 'sshpass -p ' + shlex.quote(pw) + ' ssh ...'
```

调用方必须处理 `None`：

```python
if cmd is None:
    print(f"  {name}: SKIP（GPU_SSH_PASS 未设置）")
    continue                 # 不能让它落到 subprocess.run(None)
```

**默认行为应该是"安全地不做"，不是"报错"，更不是"拿空密码去连"。**

## 三、shell 里的引号陷阱

```bash
# 危险：密码不含空格时碰巧正确
SSH_CMD="sshpass -p "$GPU_SSH_PASS" ssh ..."

# 正确
SSH_CMD="sshpass -p '$GPU_SSH_PASS' ssh ..."
```

bash 会拼接相邻引号，所以第一行在**密码没有空格时**看起来正常——含空格就断成多个参数。

## 四、验证：两个分支都要测

改了凭据处理之后，**"语法通过"和"跑起来没报错"都不够**：

| 必测 | 为什么 |
|---|---|
| 未设环境变量 | 应安全跳过，不是崩溃、不是误报"离线" |
| 已设环境变量 | 命令应正确拼出（含空格的值也要测） |

**真实教训**：一次改造里有三个 bug **语法完全通过**——`shlex` 没导入（要 import 才暴露）、shell 引号（要含空格才暴露）、`None` 被 `except Exception` 吞掉后误报成 "OFFLINE"。

## 五、记录：写"我清理了 X"时不要把 X 再写一遍

**这是最容易犯的错。** 一次清理中，我在日记里写「全仓 grep `<密码>` 零命中」，把密码又写了进去；下一轮又写了 5 次（标题、正文、引用块、历史说明）。

**规则**：

> 记录凭据相关工作时，正文里一律只用占位符，**不复述原值**——哪怕是为了说明"我找到了它"。

```markdown
✅ 全仓 grep 原密码，零命中
❌ 全仓 grep mypassword123，零命中
```

## 六、工作区干净 ≠ 历史干净

移除工作区里的明文**不能**消除 git 历史中的暴露。

处置顺序：

1. **轮换凭据**（最要紧，通常只有人能操作）
2. 视需要决定是否 `git filter-repo` 重写历史——**会改变所有 commit hash**，必须由人明确同意
3. Agent 不应擅自重写历史

## 七、不要用 `--insecure` 绕过证书错误

网络出问题时 `git`/`curl` 报 `self signed certificate`，说明有中间人代理。**不要**用 `http.sslVerify=false` 或 `curl -k` 绕过——那等于接受未验证的证书。

正确做法：记录受阻，等网络恢复。

## 边界

- 本文只覆盖**本仓库实际遇到过**的形态（SSH 密码、环境变量改造、记录泄露）。
- 不是完整的安全指南：不涉及密钥管理服务、轮换策略、审计日志。
- 「宁漏不误报」是 `credential_scan.py` 的取舍，不是通用原则——安全场景通常反过来，但**误报率高到会被关掉的哨兵等于没有**，两者要权衡。

## 来源

[diary/2026-11-03.md](../../diary/2026-11-03.md) 18:0x–18:4x 的六条记录；
[tools/anchors/credential_scan.py](../tools/anchors/credential_scan.py)。
