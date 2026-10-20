# vLLM-Ascend：NPU 进程间通信的反序列化与设备边界

<!-- contribution-summary:start -->
**【仓库】** [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)（昇腾推理适配项目）

**【贡献点】** NPU 进程间通信的反序列化与设备边界。[PR #10592](https://github.com/vllm-project/vllm-ascend/pull/10592)

**【类型】** 安全审阅

**【价值评估】** 5 / 10 — 可提供状态边界与验证思路，但历史推断需要适配 NPU 环境和当前源码重新核验。

**【进展】** 历史审阅草稿（2026-06）：尚无本文已发布或修复已验证的记录；原文社区状态不作为当前事实。

**【大白话讲明白】**

进程间传递张量时，既要确认数据来自可信来源，也要确认它属于正确设备。草稿指出直接解析外部 pickle 数据及设备标识不匹配可能带来风险；是否可被利用还取决于接口暴露范围和信任边界。

**【What–Why–How】**

- **What：** NPUIPC 的 HTTP 数据解析与 UntypedStorage 设备匹配。
- **Why：** 反序列化执行能力与底层设备归属是不同风险点，不能省略可达性与权限条件。
- **How：** 建议审查输入来源、认证边界和设备校验；本文无已验证的远程利用或修复部署结论。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> 2026-06-19 | Comment draft for posting on vllm-ascend #10592
> ★★★★★★★★ CRITICAL: pickle.loads RCE vulnerability on HTTP endpoint
> ★★★★★★★★ CRITICAL: UntypedStorage device mismatch → cross-device memory corruption
> ★★★★★★★★ Pattern family: Insecure Deserialization (same as SGLang #28582)

---

## Comment Body Draft

```markdown
## Two Critical Security/Correctness Bugs in NPUIPC Weight Transfer

This PR introduces important zero-copy weight sync capability for Ascend NPU, but has two critical bugs that need fixing before merge.

### Bug 1: RCE Vulnerability via pickle.loads on HTTP Endpoint

The `ipc_handles_pickled` mode uses `pickle.loads()` on data received from HTTP endpoints. This is a **Remote Code Execution vulnerability** — same pattern as SGLang #28582 (RCE via pickle IPC).

**Attack path:**
1. Attacker sends crafted HTTP request with malicious pickle payload
2. Server base64-decodes and calls `pickle.loads()`
3. Arbitrary code executed on the server

**Security gate analysis:**
- `VLLM_ALLOW_INSECURE_SERIALIZATION` is used as a gate
- This gate is an environment variable — easy to misconfigure
- The HTTP endpoint still EXISTS even when gated — any misconfiguration enables RCE
- Same environment variable is used for model loading in vLLM main → conflating weight sync security with model loading security

**Recommended fix:** Replace `pickle.loads` with safe serialization:
- **safetensors**: Already used by vLLM for model loading — proven safe, zero-code-execution guarantee
- **JSON + raw binary**: Metadata as JSON, weight tensors as raw bytes — safe and simple
- **Remove HTTP+pickle mode entirely**: Keep Ray IPC mode (already safe) — simplest fix

### Bug 2: UntypedStorage Device Mismatch → Cross-Device Memory Corruption

`pickle.loads` deserializes `UntypedStorage` with the **sender's device index baked in**. The current code only updates `list_args[6]` (tensor's logical device metadata), but the storage still points to the sender's physical NPU device.

**What happens:**
```python
tensor = pickle.loads(pickled_data)
# tensor.untyped_storage().device = sender_device_index (e.g., 220)
list_args[6] = local_device_index  # Only changes tensor metadata!
# But storage still reads from sender's physical device → WRONG device!
# → cross-device memory access → corruption or crash
```

**Fix direction:**
```python
# After deserialization, must rebind storage to local device:
storage = tensor.untyped_storage()
new_storage = storage._new_shared_npu(local_device_index)
tensor.set_(new_storage)
```

Or better: avoid pickle entirely (safetensors or raw binary), which eliminates both bugs at once.

### Cross-Framework Pattern: Insecure Deserialization

| Framework | Issue | Attack Vector | Severity | Fix |
|-----------|-------|--------------|----------|-----|
| vLLM-Ascend | **#10592** | pickle.loads on HTTP | **CRITICAL** (network) | safetensors or remove mode |
| SGLang | #28582 | pickle.loads on IPC | CRITICAL (local) | Remove pickle path |
| SGLang | #28588 | PIL decompression bomb | MEDIUM (local) | PIL pixel guard |

**Pattern**: Using `pickle` for data that crosses trust boundaries (network, IPC). `pickle` can execute arbitrary Python code during deserialization. This is a well-known vulnerability class (OWASP A8: Insecure Deserialization).

### verl Ascend Integration Relevance

NPUIPC is the **Ascend equivalent** of verl's ZMQ IPC weight sync. After security bugs are fixed, NPUIPC becomes the optimal path for verl Ascend integration:
- Zero-copy saves significant memory bandwidth (Ascend HBM ~1.2 TB/s vs H100 ~3.35 TB/s)
- But RCE vulnerability MUST be fixed first — weight sync happens every training step in RLHF
- Training clusters MUST NOT have pickle-based network endpoints

### Suggested Priority Fix

**Immediate**: Remove `ipc_handles_pickled` HTTP+pickle mode entirely. Keep only Ray IPC mode (safe by design).

**Long-term**: Add safetensors-based HTTP endpoint as safe alternative (if standalone deployment without Ray is needed).

Thanks for this important PR — zero-copy weight sync is critical for Ascend NPU performance!
```

---

## Posting Strategy

1. ★★★★★★★★ MUST get user authorization before posting on vllm-project/vllm-ascend #10592
2. Post this comment → provides 2 critical bug analysis + fix suggestions
3. Security focus: RCE + device mismatch → high-priority for merge review
4. Highlight verl integration pathway
5. Pattern family connection to SGLang #28582

## Priority: P6 C18 (HIGH) — NPUIPC security and correctness bugs

★★★★★★★★★ This is a UNIQUE security contribution:
  → RCE vulnerability identification with attack path
  → UntypedStorage device mismatch root cause analysis
  → Cross-framework Insecure Deserialization pattern family
  → verl Ascend integration impact analysis
  → Concrete fix suggestions (safetensors, remove HTTP+pickle mode)
  → SGLang #28582 cross-reference (same vulnerability class)

---

## References

- PR: https://github.com/vllm-project/vllm-ascend/pull/10592
- SGLang #28582: RCE via pickle IPC
- SGLang #28588: image decompression bomb
- Deep reading: notebook/frameworks/vllm-ascend-10592-npuipc-weight-transfer-reading.md
- OWASP A8: Insecure Deserialization
