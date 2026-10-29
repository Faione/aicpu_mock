# 原型输入输出协议 v1

公开接口为 UTF-8 JSON 文件或等价 Python dict。首版使用 JSON 配置，原设计稿中的 YAML 尚未实现；CPU 路径无第三方 JSON/YAML 库依赖。拒绝未知字段、重复 JSON 字段及不支持的属性组合。

## 配置

`--config <path>` 或环境变量 `QFA_METADATA_CONFIG` 指定配置。见 `examples/target_arch92.json` 和 `examples/target_arch35.json`。

| 字段 | 类型 / 要求 |
|---|---|
| backend | 必填，只接受 `mock_aicpu` |
| target.name | 必填，非空字符串；用于标识，不能决定调度策略 |
| target.soc_version | 必填，真实算子 SoC 标识；首版只支持 README 所列系列 |
| target.kernel_arch | 必填，`arch92` 或 `arch35`；与 SoC 策略核对 |
| target.aic_core_num | 必填，整数 1–1024，实际有效核数 |
| target.aiv_core_num | 必填，整数 1–2048，与架构核数比例匹配 |
| cache.enabled | 可选，布尔值，默认 true |
| cache.max_entries | 可选，整数 1–128，默认 8；仅进程内缓存，无持久化缓存 |

CLI 是专用 mock 工具，缺配置即报错。TTK 插件没有配置时才委托原 native 流程。

## 请求

| 字段 | 类型 / 要求 |
|---|---|
| protocol_version | 必填，整数 1 |
| execution_mode | 必填，字符串 `one_shot`；不接受历史 cache 或追加状态 |
| num_heads_q | 必填，整数 1–1024 |
| num_heads_kv | 必填，正整数，不大于 num_heads_q，且整除 num_heads_q |
| head_dim | 必填，整数 128 |
| head_dim_v | 可选，默认 128；null 等价于默认值，当前必须等于 128 |
| cu_seqlens_q | 必填，B+1 个非负 int32 范围整数，首项 0，非递减 |
| cu_seqlens_kv | KV=TND 时必填，B+1 项；PA_BNBD 时必须为空或不提供 |
| seqused_q / seqused_kv | 可选，B 项非负整数；PA_BNBD 必须提供 seqused_kv |
| batch_size | 可选，null 或正整数；由 cu_seqlens_q 推导 B，显式值须一致 |
| max_seqlen_q / max_seqlen_kv | 必填，正整数；不接受 -1，不自行补写长度 |
| quant_mode | 可选，默认且仅支持整数 1 |
| mask_mode | 可选，默认 3，仅支持 0、3、5 |
| win_left / win_right | 可选，默认且仅支持 -1 |
| layout_q / layout_q_descale / layout_out | 可选，默认且仅支持 TND |
| layout_kv | 必填，TND 或 PA_BNBD |
| is_grad_enabled | 可选，默认且仅支持 false |

规则：B 在 1–4096 内；空数组与 null 视作未提供（必填数组仍会失败）。`seqused` 存在时作为有效长度，否则取累积长度差分。有效长度不得超过分配长度，分配长度不得超过对应 max 属性。保留原始累积数组和有效数组，不因有效长度相同而合并请求。PA_BNBD 的 KV 分配上界以 max 属性描述。

请求中没有 Q/K/V/scale 数值，它们继续走原测试流程。Q descale 布局保留在请求中，因为它影响模板选择；首版只接受 TND，暂不支持 N2TGD/BN2SGD。这是首版组合限制，不是从“无历史 KV cache”推导出的限制。

## 结果 bundle

| 文件 | 内容 |
|---|---|
| request.json | 完整规范化请求，默认值显式化，保留原始长度 |
| config.json | 完整规范化目标配置与缓存设置 |
| metadata.bin | 2×rowSize 个 little-endian 32-bit word，按行连续 |
| result.json | 协议和算法版本、shape、Header、活跃核数、有效区长度、校验结果及摘要 |

Tensor 接口为 CPU contiguous `torch.int32`，shape `(2, rowSize)`。协议字段按 uint32 位模式编码，交付 int32 不改变位模式。

```text
rowSize = align_up(16 + (AIC+AIV)*16*B*num_heads_q, 4096)
used_words = 16 + sectionNum*(AIC+AIV)*16
used_words <= rowSize
```

4096 的单位为元素。首版完整二进制最多 256 MiB，超出即拒绝。

第一行布局为 Header[16]、FA[sectionNum][AIC][16]、FD[sectionNum][AIV][16]；第一行余量和完整第二行清零。FA/FD 具体字段使用原 `quant_flash_attn_metadata.h` 常量编码。

Header 槽位：0=sectionNum，1=isFD，2=M，3=S2，4=AIC，5=AIV，15=needInitOutput；其余槽位清零。

`result.json` 包含：

- `protocol_version=1`、`algorithm_version=qfa-host-v1`。
- `shape`、`header`、`used_words`、`active_aic`、`active_aiv`、`target`。
- `source_digest`：调度头文件、相关 QFA 源码、原 decoder 与 Host C++ 适配代码的 SHA-256 指纹。
- `request_digest`：目标、规范化请求、算法版本及源码指纹的规范 JSON 的 SHA-256。
- `metadata_digest`：完整二进制 SHA-256。
- `validation`：结构/原 decoder 覆盖校验状态、FD 任务数、有效 Q/KV 长度。
- `cache_hit`：本次是否命中当前 worker 的进程内缓存。

## 失败

CLI 非零退出，stderr 输出单个 JSON 错误对象：

```json
{"error":{"code":"UNSUPPORTED_REQUEST","message":"v1 requires execution_mode='one_shot', got 'incremental'"}}
```

主要错误码：`MISSING_FIELD`、`INVALID_INPUT`、`UNSUPPORTED_FIELD`、`UNSUPPORTED_REQUEST`、`UNSUPPORTED_TARGET`、`TARGET_MISMATCH`、`REQUEST_MISMATCH`、`CAPACITY_EXCEEDED`、`HELPER_NOT_BUILT`、`STALE_BUILD`、`HELPER_FAILED`、`HELPER_TIMEOUT`、`INVALID_METADATA`、`COVERAGE_FAILED`、`GOLDEN_FAILED`。IO/JSON 格式错误由 CLI 输出 `IO_OR_FORMAT_ERROR`。库接口抛出 `MockError`（含 `.code`），IO 错误保留 Python 异常。

失败不提交真实 qfa_metadata，不使用固定样例替代，不回退全局参考计算。

## C++ 私有传输协议

用户不直接构造此协议。Python 校验公开 JSON 后，将它编码为版本化的定长顺序文本，传给 C++ stdin，避免引入 C++ JSON 解析依赖。

```text
QFA_HOST_V1
arch92_flag AIC AIV B Nq Nkv D Dv maxQ maxKV maskMode paged_flag
cu_q_count cu_q_values...
cu_kv_count cu_kv_values...
used_q_count used_q_values...
used_kv_count used_kv_values...
```

缺失数组写 `-1`；布尔 flag 写 0/1；不接受尾随数据。固定受支持属性由协议版本约束，不在 wire 中重复传递。C++ 再检查范围、长度和容量。启动形式为 `metadata_helper <output.bin>`，成功 stdout 输出 JSON 摘要，失败非零退出并在 stderr 输出错误。每次调用使用独立临时目录。
