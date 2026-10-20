# vLLM NixlConnector lease 契约与跨请求 KV 消费正确性（#61229 / #62065）

> 源码版本：本地 vLLM main (2026-06) + 上游 #62065 (2026-10) diff
> 关联：[[vllm-kv-transfer-connector]] · [[vllm-pd-disaggregation-reading]] · [[vllm-pd-separation-kv-transfer-architecture]]
> 上游：issue [#61229](https://github.com/vllm-project/vllm/issues/61229) · 修复 PR [#62065](https://github.com/vllm-project/vllm/pull/62065)（vadiklyutiy，OPEN，B200 硬件验证）· 竞争 PR #61549（sawsa307，CLOSED）

## 一、问题：lease 过期后无声消费跨请求 KV

P/D 分离 + NIXL pull 模式下，prefill（P）把某请求的 KV blocks 通过 lease 机制借给 decode（D）读取。本地 6 月版源码（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/`）：

- **lease 机制**（`base_worker.py`）：`kv_lease_duration`（默认 30s）定义 P 侧 block 的可持有期；D 每 `kv_lease_duration // 6` 秒发 heartbeat，每个 heartbeat 把 deadline 延长到 `now + kv_lease_duration * 2 // 3`（默认 5s 心跳、20s 续期，约 20s 心跳沉默即被回收）。
- **无条件释放**（`base_worker.py:1844-1861`）：`_reap_expired_send_leases` 到点即把 block 移入 `done_sending`，**不查询是否有 consumer READ 在途/待发**。日志行诚实写明了这一点："Releasing expired KV blocks for request %s which were retrieved by %d remote worker(s) before lease expired."
- **无声消费**（`pull_worker.py` 的 `_read_blocks`）：D 按 descriptor（`remote_block_ids`/`remote_request_id`，**不含任何内容/代际标识**）post READ，传输完成即加载给模型，无任何 producer 侧有效性校验。

两条合起来：lease 过期 → P 释放并复用 block → D 的 READ 仍按旧地址读取（新请求已占用）→ 静默解码**另一个请求的内容**。reporter 在 2×? 上 5/5 复现（默认 `kv_lease_duration=30`，解码引擎停顿 33.6s 触发）。MEMBER `vadiklyutiy` 指出这是**跨请求机密内容泄露**（一个用户的 KV 被另一个请求消费），即便罕见也是严重问题。

## 二、根因：正确性契约缺失，不是单纯的时序 bug

- descriptor 只携带 `block_ids`，**没有与字节绑定的身份/代际**；layout hash 校验哈希的是布局不是内容。
- consumer 侧唯一的检测器是 P 侧一个 post-hoc ERROR（"Potentially invalid KV blocks ... retrieved by a decode worker"），fire-and-forget，且常在字节已消费后才触发，多数运行根本不触发。
- 因此 lease 过期从"设计上的块回收"退化为"静默返回错误 token + 跨请求泄露"。

这是典型的 **P/D 远程 KV 传输的正确性契约**问题：lease 定义了"P 什么时候可以安全复用 block"，但没有任何机制把"复用发生"这件事传达到已发出 READ 的 D。本地 `pull_worker.py` 的 `_read_blocks` 里注释也承认"having the staging blocks on the READER side is not going to work well ... if we want to make READ happen cleanly"，但没解决过期竞态。

## 三、修复：#62065 的 lease slot + nonce 校验

#62065 新增 `nixl/lease_slots.py`，把 reporter 建议的"transfer 末尾查 request id + verdict 送达 consumer"落实为 DRAM 级 nonce 校验：

- **Producer 侧**：每个 prefill 请求分配 `(slot, nonce)` lease slot，nonce 由 `secrets.randbelow` 生成（避免重启后重复 / 请求 id 复用的代际混淆）。slot 把 nonce 写入 DRAM 并导出给 reader。block 释放前先 `clear(req_id)` 把 slot 置 0。
- **Consumer 侧**（`_verify_read_leases`）：READ 完成后，用 `LeaseVerifier` 读 producer 的 slot nonce，与 descriptor 携带的 nonce 比对；不匹配 → `Declining KV pulled ...`，走 `kv_load_failure_policy`（`fail` → 请求失败 HTTP 500 / `recompute` → 本地重算），把"无声错误 token"变成"响亮的失败或正确重算"。
- **第二个修复**：read notif 携带 lease nonce，P 忽略属于**更早 lease** 的 late notif（request id 随机化关闭 + 路由器用同 id 重试时，早期尝试的晚到 notif 不再误释放重试的 block）。
- **开关**：`kv_lease_verification` 默认开，`kv_lease_slots` 可调 slot 数，关掉还原旧行为。
- **验证**：B200 2P1D 复现器，base 在 34s 停顿+复用后 HTTP 200 返回他人文本；本 PR `fail` 态 HTTP 500、`recompute` 态匹配基准；TTFT/ITL 在 concurrency 1 不变，concurrency 16 时 median TTFT +1.3ms。

关键设计点：**用"读写共享的 DRAM 计数器/非ce"做 lease 有效期校验，完全不用时钟或心跳会计**——校验发生在 READ 完成之后，能覆盖"lease 在 READ 期间过期"和"READ 在 lease 过期后发出"两个时相；nonce 的单调性区分了请求代际，不依赖全局时钟同步。

## 四、方法学启示（P/D 分离系统的通用正确性模式）

1. **lease 的"可复用性"断言必须传达到消费者**。生产端能安全释放 ≠ 消费端能安全使用；块回收事件要能逆流通知到所有引用者，否则"lease 过期"与"in-flight READ"两个事件之间的窗口就是静默错误区。
2. **把内容身份绑进传输描述符**（代际 tag / nonce）并在消费端校验，是最小、最可靠的 seam；比 producer 端 refcount / fence 握手成本低，且对网络分区、引擎停顿这类"非故障"的假死也有效。
3. **"布局校验 ≠ 内容校验"**：layout hash 只保证字节形状一致，不保证字节属于谁。校验目标应是请求/代际身份而非几何。
4. 竞态窗口类 bug 用**确定性复现（人为制造停顿）+ 对照（本地计算/完好 pull）** 才能坐实，reporter 的方法（5/5 + 三组对照）是教科书式。

## 五、贡献轨道判断（2026-10-20 核验）

#61229 已被 #62065（完整修复+硬件验证）与 #61549（fake-clock 单测）认领，另有 sawsa307 的 KV checksum 长线系列（#61578/#61488）。修复方向与 reporter 建议一致且实现更完整。**结论：无 PR 竞争空间，不再投入。** 本地旧版源码 + #62065 diff 作为理解材料保留，支撑后续对 P/D 远程 KV 正确性、KV connector 子系统的贡献判断。
