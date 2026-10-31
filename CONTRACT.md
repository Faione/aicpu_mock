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

CLI generate 与旧外置插件是专用 mock 入口，缺配置即报错。当前推荐的 assets 补丁使用原 Spec：QFA_METADATA_CONFIG 非空进入 mock，未设置时保持 native；给定的配置无效时仍直接失败。

共享 worker 首次调用固定解析后的仓库路径、规范化配置、helper 路径和构建身份。字段顺序/JSON 排版变化或相同内容迁移配置文件不改变身份；语义变化报 `WORKER_CONFIG_CHANGED`。新 assets 分支直接传入真实形参和分段计划，无原实现加载器或函数替换；旧外置方案的路径/签名检查保留兼容。平台证据工具降为可选诊断，简化路径不要求启动回执；目标 tiling 和设备精度仍须实际验收。

build 不调用 Git。编译前必须按清单检查编译头文件、assets/golden 和 AICPU 参考 cpp；缺少任一项都阻止构建，不提供跳过选项。缺失时返回 SOURCE_MISSING，error.repo 是解析后的根目录，error.missing 数组逐项包含 group、relative_path、path、required_for；message 同时列出全部缺失项。分组为 compile、assets、aicpu_reference。检查失败不调用编译器、不覆盖旧产物。

C++ 编译完成后生成 build_format=3 清单。dependency_files 记录实际编译的项目文件、@helper 和 @mock/cpp/ 下的共享核心；source_digest 只按这些编译依赖计算。behavior_files/behavior_digest 另行记录已核对的上游 AICPU、分块/调度定义及 decoder；adapter_version=qfa-host-v2 标识适配实现，algorithm_version=qfa-host-v1 保留原分核算法身份。编译和行为身份均进入 worker 校验；行为摘要另进入请求/缓存/bundle 身份。未知行为变更报 UPSTREAM_INCOMPATIBLE，重编译不自动授予兼容结论。旧清单和 bundle 需重新生成。

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
- `source_digest` / `dependency_files`：实际编译依赖及 Host C++ 适配代码的 SHA-256 指纹/清单，不代表已验证生产包装逻辑或实际 tiling。
- `behavior_digest`：已核对源码组合和适配版本的摘要，独立于编译依赖。
- `request_digest`：目标、规范化请求、算法版本、编译指纹及行为摘要的规范 JSON 的 SHA-256。
- `metadata_digest`：完整二进制 SHA-256。
- `validation`：结构/原 decoder 覆盖校验状态、FD 任务数、有效 Q/KV 长度。
- `cache_hit`：本次是否命中当前 worker 的进程内缓存。

## 失败

CLI 非零退出，stderr 输出单个 JSON 错误对象：

```json
{"error":{"code":"UNSUPPORTED_REQUEST","message":"v1 requires execution_mode='one_shot', got 'incremental'"}}
```

主要错误码：`MISSING_FIELD`、`INVALID_INPUT`、`UNSUPPORTED_FIELD`、`UNSUPPORTED_REQUEST`、`UNSUPPORTED_TARGET`、`TARGET_MISMATCH`、`REQUEST_MISMATCH`、`CAPACITY_EXCEEDED`、`HELPER_NOT_BUILT`、`STALE_BUILD`、`HELPER_FAILED`、`HELPER_TIMEOUT`、`INVALID_METADATA`、`COVERAGE_FAILED`、`GOLDEN_FAILED`。IO/JSON 格式错误由 CLI 输出 `IO_OR_FORMAT_ERROR`。库接口抛出 `MockError`（含 `.code`），IO 错误保留 Python 异常。

旧接入层另有 `MODULE_SOURCE_CONFLICT`、`SOURCE_CHANGED`、`UPSTREAM_INCOMPATIBLE`、`WORKER_CONFIG_CHANGED`、`TARGET_UNVERIFIED`、`PREPROCESS_FAILED`、`RESTORE_FAILED`。新补丁在显式 metadata_plan 模式中直接传播数值异常，不使用内部 BaseException 桥接；未设置配置的 native 行为不变。新路径日志以 patched_golden_plan / patched_preprocess_plan 关联请求与 metadata 摘要。

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

## extension_cpp 安装模式

原 Torch schema、C++ native metadata 函数体和主 qfa 绑定不变。对外 metadata PyBind callable 由 MakeMetadataBinding 持有原函数指针：native 调原指针，mock 调同一 Host 核心；不做强弱符号抢占或 Python provider 回调。Python 仅处理配置、目标容量和完整主 qfa 用例范围；原 assets 不改，因此原 golden 的回退行为保留，独立审计负责识别分段是否实际成功。

默认 output 按目标上界分配，C++ 也接受显式其他行容量，只要实际 `16 + sectionNum*(AIC+AIV)*16 <= rowSize`。检查 dtype=int32、二维两行、连续、设备和容量，完整校验后写入并返回原 output，第二行和剩余区域清零。核心覆盖验证设 20,000,000 次任务遍历上限，超过显式失败；单个 metadata 仍限制为 256 MiB。

扩展目前不缓存计划。显式 QFA_METADATA_BACKEND=mock_aicpu 或兼容的非空 QFA_METADATA_CONFIG 启用 mock；显式 native 与非空 mock 配置冲突失败。worker 固定规范化目标/配置，C++ 固定 PID/模式/目标/构建身份；fork 后重新初始化，修改后重启。目标 JSON 格式沿用前文，CSV 目标配置仍未实现。

独立 vendor wheel 内置共享核心、相关上游头文件、参考身份和 Python 适配；不依赖开发仓库绝对路径。`.run` 组合工具要求实际 CANN custom 包、同 vendor wheel、同源行为身份和完整安装文件清单。安装过程核对原生文件、provider 来源和真实 JIT 扩展后才报告成功，失败保留 INSTALL_INCOMPLETE。编译依赖和目标环境步骤见 EXTENSION_USAGE.md。
