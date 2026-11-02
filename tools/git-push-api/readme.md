# git-push-api — 网络不通时用 REST API 推送

## 使用场景

`git push` 连不上 `github.com:443`，但 `api.github.com` 正常、且 `gh` 已登录（token 有 `repo` 权限）时用来把本地提交推上去。2026-11-03 实际遇到：本机到 github.com 直连超时 75s，`gh api` 与 `api.github.com` 均返回 200。

## 用法

```bash
# 以远端当前 main 为父提交，把本地 HEAD 的差异推上去
python3 tools/git-push-api/push_via_api.py <远端当前 main 的 sha> -m /tmp/commit-msg.txt
```

跑完远端 main 会指向一个新的 commit。**它的 SHA 与本地不同**（消息、树相同，父提交与提交时间可能不同），所以之后要：

```bash
# 用 gh-proxy 把新提交取下来，再让本地 main 对齐
git init -q --bare /tmp/remote-main && cd /tmp/remote-main
git fetch -q https://gh-proxy.com/https://github.com/Jackie2049/perpetual-learning.git <新 sha>
cd - && git index-pack --stdin --fix-thin < <(cat /tmp/remote-main/.git/objects/pack/*.pack)  # 或直接 reset 后 pull
git reset --hard <新 sha>
```

## 为什么要自己重建树

本仓库的 `experiments/` 是一个巨型目录。实测 `POST /git/trees` 单请求提交整体树时：

| 条目数 | 结果 |
|---|---|
| 1500 | OK |
| 1915 | OK |
| 1916 | **HTTP 502** |
| 2621（完整树） | **502** |

所以脚本只重建"改动文件所在路径上的目录"，其余目录直接复用父提交里的 tree sha；每个目录一层请求，并对 500/502 做指数退避重试。

## 输入输出与依赖

- 输入：远端父提交 sha、可选 commit message 文件。
- 输出：新的 commit sha 与更新后的 `refs/heads/main`。
- 依赖：`gh` 已登录、`python3`（仅标准库）、GitHub token 有 `repo` scope。

## 已知限制（务必先读）

- **会压平提交历史**：本地在父提交之上的所有提交会被合成远端的一个提交。要保持多条提交，就逐个父提交调用。
- **不是原子的**：脚本先建 blob、再建树、最后 commit + 更新 ref。中途失败不会留下错误引用，但已创建的 blob/tree 会悬空（无害，GitHub 会回收）。
- **不适用于删除整个目录后重建同名目录的场景**：树的 diff 是增量的。
- **并发写会失败**：如果另一个会话在此期间推了 main，PATCH ref 会返回 422 `not a fast forward`——此时用新的远端 sha 重跑即可（脚本是幂等的）。

## 验证范围

2026-11-03 在 perpetual-learning 上真实跑通一次：14 个改动路径 + 5 个目录层，远端 main 从 `ee977b39` 前进到 `60869da2`，随后本地 fetch 对齐。**未在含文件删除/重命名的变更上验证**，也**未在多提交（>1 层）的差异上验证**。
