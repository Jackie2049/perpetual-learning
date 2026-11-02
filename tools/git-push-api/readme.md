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

2026-11-03 在 perpetual-learning 上真实跑通一次：14 个改动路径 + 5 个目录层，远端 main 从 `ee977b39` 前进到 `60869da2`，随后本地 fetch 对齐。**未在含文件删除/重命名的变更上验证**，也**未在多提交（>1 层）的差异上验证**。新增整个新目录的路径已验证（`tools/git-push-api/` 目录在父提交里不存在时，GET 该层会 404，脚本按空层处理）。

## 核验远端内容时的一个坑（2026-11-03 踩过）

推完之后要核验"内容真的上去了"，**不要用 Contents API**：

```bash
# 错：对 >1MB 的文件，content 返回空串，解码后是空内容
curl "https://api.github.com/repos/<owner>/<repo>/contents/<path>?ref=<sha>"

# 对：先取 blob sha，再用 Blob API
curl "https://api.github.com/repos/<owner>/<repo>/git/trees/<sha>?recursive=1"   # 找 path 的 sha
curl "https://api.github.com/repos/<owner>/<repo>/git/blobs/<blob_sha>"          # base64 content
```

**症状**：Contents API 报 `size: 1516668`，但 `base64.b64decode(content)` 得到空串，
md5 为 `d41d8cd98f00b204e9800998ecf8427e`（空串的 md5）。这会让人误判"内容丢了"。

**最简单的判据**：远端 blob 的 `git hash-object` 与本地
`git hash-object <file>` 相同即一致——不必解码内容。

**教训**：核验工具本身也会骗人。出现"内容丢失"这种严重结论前，先换一种方式复核一遍。
