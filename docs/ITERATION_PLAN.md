# QFA Mock AICPU 下一轮迭代计划

主设计第 25 节的可执行副本。当前实现基线为 eef4af5；以下除 P0 文档同步外均待实施，不替代 README 的现有部署命令。

## 25. 下一轮实施计划：统一入口、版本兼容与 CSV 参数来源（2026-10-31）

本节将第 23—24 节的候选分析及后续讨论整理为下一轮实施顺序。它是待实施计划，不把共同接口、CSV 目标配置或跨版本兼容记为现有能力。当前可部署基线仍为 `eef4af5` 的三文件 assets 补丁；2026-10-30 的 55 项 CPU 测试证据仅属于该基线。本轮只更新文档，未新增代码验证或目标设备结论。

### 25.1 已确定的边界与改动范围

- 仅处理一次完整输入；保留现有 MXFP8、quant_mode=1、正向 Eager、D=Dv=128、Q/scale/Out=TND、KV=TND 或 PA_BNBD、默认窗口、显式正最大长度等限制。mask=0/3/5 的计划检查不等于全部组合已经完成真实 golden 或设备数值验收。
- metadata 包含 Header、FA 分核区、FD 归并区及保留的第二行；调度依赖形状、有效/分配长度、算子属性、目标和算法版本，不读取 Q/K/V 的具体浮点数值。多 section、FD、PA 布局均不意味着多轮 KV cache。
- C++ helper 不解析 CSV。TTK 解析 CSV 并生成实际输入，Python 适配层读取实际形参与长度张量，校验 sidecar 后编码给 helper。独立 generate 入口继续接受 JSON。
- build 仍不调用 Git，必需头文件、assets/golden 和 AICPU 参考 cpp 均强制检查；缺任一项停止并列出全部缺失项，无跳过选项。源码身份由内容摘要确定，不把 Git 历史重新变成前置条件。
- 下一轮只设计一个共同 Python metadata 实现入口与必要的 assets 保护。候选补丁可包含 Torch 扩展 Python 实现及必要 assets 文件，不改 Torch schema/签名、生产 C++/AI Core 或 TTK。扩大补丁范围是本节待落地的设计，不表示当前三文件补丁已经替换。
- 现有 `assets_explicit` 与候选 `metadata_api` 是部署/测试方案标签。每个 checkout/worker 只启用其中一套；工具需识别、拒绝叠加，明确卸载旧补丁后再应用候选补丁。保留基线回退是显式部署选择，运行失败不自动切换方案、native 或全局 golden。

### 25.2 helper 职责拆分与版本约束

当前 helper 由三类代码构成，不能统称为完整复用原 AICPU 算子：

| 类别 | 当前代码 | 下一轮处理 |
|---|---|---|
| 输入输出与安全检查 | wire 解析、范围/容量检查、文件与摘要输出 | 留在通用入口；main 仅编排，不加入 CSV、TTK 或 SDK 对象 |
| 上游包装行为的适配 | BaseInfo/DeviceInfo、架构与核配比限制、FD/L2 策略、needInitOutput、FA/FD 字段组织 | 拆为集中且有来源说明的版本适配模块；核配比属于当前支持限制，不能宣称为全部版本的通用定律 |
| 实际算法复用 | AdjustSinnerAndSouter*、SectionStreamK::Compute、FaMetaData 编码类 | 继续从目标源码编译；接口/语义变化需重新适配 |

先做行为不变的模块拆分，不修改现有结果，也不把 L2、FD 阈值等算法细节转成 CSV 可调参数。长期共用纯计算接口可以消除重复策略，但需要原 AICPU 生产源码配合，单列为后续选项，不作为本轮前置条件。直接编译完整 AICPU cpp 还需要 CpuKernelContext/SDK 适配，不能仅增加 include 就认为已复用完整实现。

版本兼容由三层分别约束：

1. 编译依赖摘要：继续记录真正参与编译的 helper/头文件与二进制摘要；头文件变化要求重建。
2. 行为依赖摘要：新增记录原 AICPU cpp、共同 Python 实现、decoder、相关 assets 和补丁/适配版本的内容身份。未参与编译的源码不得混称编译依赖；记录变更后的状态必须进入缓存、bundle 与 worker 身份判断。
3. 适配兼容结论：为已核对源码组合登记适配版本、支持字段和 metadata 格式。源码变化但适配兼容性尚未确认时停止 mock，明确指出变更文件和待核对项。重新 build 只更新编译产物，不能自动把新的 ParamsInit、needInitOutput 或编码语义判为兼容。

改动识别需区分原始源码与已应用补丁的运行态摘要，明确先检查/应用候选补丁、再登记运行态依赖的顺序；不得要求补丁后的文件匹配补丁前摘要。重复应用应幂等；同样内容的源码压缩包可使用，未知内容不因文件名相同而获得兼容结论。Python 模块首次加载时核对实际 `__file__` 和内容来源，防止只修改 checkout、实际却加载了另一份安装包。

验证格式变化时需同时核对 C++ 编码、Python 中固定的偏移/保留字段/容量规则和 golden decoder。变更 decoder 后，不允许命中旧缓存绕过新检查。现有 `qfa-host-v1` 是 mock 自身标识，不能当作上游 qfa_metadata 版本。

### 25.3 共同接口的最小落地路径

目标生产入口为 `torch_extension/quant_flash_attn.py::quant_flash_attn_metadata` 的同一个 Python 实现函数：包导出函数直调、PrivateUse1 dispatcher 调用必须都进入此处；无 Tensor 参数时的 fallback 也应按相同模式拒绝不支持的请求，不得绕回真实 AICPU。首版不新增 CPU dispatcher、图模式或 Meta/Fake 实际计算能力。

```text
实际调用形参 → 用例支持范围检查
                 ↓
共同 metadata Python 实现：规范化 metadata 实参
                 ↓
与本次用例请求比对 → 已固定目标/构建身份的 provider
                 ↓
Host helper → Header/FA/FD/容量/decoder 校验
                 ↓
按原接口设备语义返回 contiguous int32 metadata
```

- mock 分支必须在原 `_calculate_max_schedule_size`、NPU 输出分配及 `op_builder.load()` 之前选择；按目标核数计算容量，不能先按物理设备默认核数分配。模块级导入/注册/`_ensure_initialized()` 是否仍需 SDK 能力也要核实，不能把函数内绕过 load 等同于整个模块可脱离 CANN 导入。
- 不覆盖 `torch.ops` 属性、不替换已导入别名、不做全局函数 monkeypatch。包直调和 dispatcher 通过修改共同实现函数体收敛；只注册另一条 dispatcher kernel 不足以覆盖 golden。
- metadata 接口实参与 QFA 主调用分别正规化后比对；保留调用侧完整范围校验。原 build_metadata_arguments 未完整传递窗口等信息，需显式补齐或在调用侧拒绝。非空 V-tail、sinks、非默认窗口等不能因关闭 golden 被漏检。 独立 metadata API 调用只校验其可见实参；完整 QFA e2e 必须提供本次用例上下文，不能假定 metadata 接口能获知未传入的 QKV/V-tail 信息。
- 两个消费方的入口记录 hook、规范化 request、target、适配身份及 metadata 摘要。同一请求即使命中缓存，也记录消费事件；“调用过 provider”不能代替“golden 分段计算成功”。
- npu_preprocess 继续处理 None→字典回填、(0,)→resize/copy、固定形状→严格匹配/copy。已有非零 Tensor 仍生成计划；不截断、不把 CPU Tensor 填入 NPU 动态槽。
- metadata_api 模式的 golden 必须实际走共同接口，不能保留旧补丁的显式计划短路后宣称接口已覆盖它。保留原数值函数与 Out/LSE 整理；mock 下导入失败、_HAS_NPU=False、计划为空、decoder 或分段数值失败均报错，绕过原分段缓存或补足目标与版本身份，禁止全局参考回退。
- 目标 Runtime 支持 NPU 分配、传输、同步、读回时，候选模式可先保留 golden 的这些操作；它们不等同于 AICPU metadata 调度，但仍要检查是否引入其他不可用 AICPU 辅助任务。若 Runtime 不具备这些能力，使用现有 CPU golden 直接消费 provider 的已验收方案，保持候选模式待验收，不隐式改变接口返回设备。
- worker 首次调用前确定 native/mock 模式，首次调用固定目标、源码、已加载实现与 helper 身份。运行中配置或模式变化失败；fork 后独立初始化，重启才可更换目标。

### 25.4 CSV 能提供什么，以及如何减少配置

| 信息 | 首选来源 | 约束 |
|---|---|---|
| Nq/Nkv、D/Dv、分配形状 | TTK 生成的真实 Q/K/V shape | CSV 的 N_q/N_kv/D 等只作一致性检查，不读 testcase_name 猜参数 |
| batch、cu_seqlens、seqused | 实际长度张量，必要时使用 CSV *_values | 两者同时存在必须一致；PA KV 不传 cu_seqlens_kv；保留分配长度与有效长度差异 |
| quant/mask/window/layout/max_seqlen | CSV attributes 装配后的实际形参 | 与主 qfa 同源；max 继续显式为正，不单独推导并改写 metadata 侧默认值 |
| SoC、编译架构、有效 AIC/AIV | 外部目标配置，或新增完整 CSV 目标块 | 配置声明需与实际 tiling 核对；普通设备总核数、short_soc_version/--sim-soc 不直接替代有效核数或构建架构 |
| 分块、FD/L2 策略、编码格式 | 已适配的对应版本源码 | 不允许 CSV 自由覆盖算法策略 |
| 源码根目录、helper 位置 | checkout/部署环境与构建清单 | 保留部署配置；CSV 不消除源码或 C++ 编译器依赖 |

既有用例参数已经来自 TTK，无需另维护 request.json；它仅用于独立 CLI。CSV 目标块是后续可选能力，当前代码尚不识别。计划使用 attributes 中的 `qfa_mock_target`，复用现有 target 的完整字段（name、soc_version、kernel_arch、aic_core_num、aiv_core_num），不新增 TTK 顶层列或修改算子 schema。

拟定的解析规则如下，落地前写入独立契约与示例，不提前改写当前部署说明：

1. 新增可选显式模式开关 `QFA_METADATA_BACKEND=mock_aicpu`。未设置新开关时兼容现有非空 QFA_METADATA_CONFIG 启用 mock 的行为；两者都未设置仍走 native。显式 native 与非空 mock 配置/CSV 目标冲突时失败，不静默选一方。
2. mock 已启用时，完整 CSV 目标可替代外部目标 JSON；同时给出两者必须按规范化目标逐字段一致，禁止不透明覆盖或拼接部分字段。缺目标、缺必需字段、未知字段均失败。
3. CSV 目标本身不静默切换 backend。native 下存在 mock 目标也需由用例校验报告配置冲突；纯 CLI 不因 CSV 功能改变行为。cache 使用默认值或兼容配置，不将缓存参数重复到每个用例。
4. CSV 中的额外属性需要在 golden 与预处理边界显式消费和传递，当前仅写入 CSV 不会生效。metadata 接口没有该参数，不能靠修改其 schema 或临时改环境变量透传；通过一个明确的 worker 目标绑定入口初始化同一 provider，并在后续两端重复核对。
5. 首版一个 worker 只绑定一个目标；同一运行中不同用例声明不同目标时明确失败，按目标拆开启动 TTK。不引入同 worker 多目标 provider 池；同目标字段排版变化不算目标变化。
6. 为完整用例请求建立调用范围内的显式上下文，并与 metadata 接口实际收到的参数核对；上下文在成功/失败后清理，不能泄漏至下一用例或线程。不从另一用例的模块全局状态补缺字段。具体传递方式在接入阶段定稿，不影响公开算子签名。

CSV 可减少目标 JSON 文件，但不能自行证明物理/仿真平台、安装包与实际构建一致。核数比限制先集中到当前版本适配配置，保留拒绝未知目标的行为，不因 CSV 可填任意值而放开。

### 25.5 分阶段任务、依赖和完成条件

| 阶段 | 范围与交付 | 前置条件 | 完成条件 | 当前状态 |
|---|---|---|---|---|
| P0 | 同步第 22—24 节及本节；整理基线、待办、入口图和现有证据 | 无 | 设计/本地副本/进度相符，未把候选当作已实现 | 本轮文档同步完成；无新增运行证据 |
| P1 | helper 行为不变拆分；编译/行为身份分离；兼容清单与源码来源检测 | P0 | 当前固定样本与六组 CPU 结果不变；仅改 AICPU cpp 或 decoder 也能拒绝失配；必需文件仍全检 | 待实现 |
| P2 | 共同 Python metadata 接口候选补丁、独立 adapter 与部署模式检测 | P1 | 真包直调与真 dispatcher 均命中同一实现；原生 metadata 调度入口触发即失败；模式不叠加 | 待实现 |
| P3 | golden 严格失败、真实参数一致性、缓存、槽位与生命周期回归 | P2 | 关闭 golden 仍检查完整用例；分段计算成功；所有错误不回退；候选独立得到 CPU 接入证据 | 待实现 |
| P4 | CSV 目标块、显式开关与配置来源规则；新示例/使用说明 | P3 | 双入口同目标同计划；无需目标 JSON 的 CSV 用例通过；来源冲突、缺字段、worker 换目标均失败 | 待实现 |
| P5 | 目标 Runtime 冒烟与实际 AI Core 精度；登记可部署组合 | P2/P3，CSV 场景另需 P4 | 先关闭 golden 验证 metadata 写入/实际 tiling，再开启 golden 验证两端及 Out/LSE；AICPU 同配置对拍具备条件时单列完成证据 | 待目标环境 |

P2 前先识别模块导入与 Runtime 能力边界；发现环境不支持时停止候选设备路径，P1 及 CPU 侧验证仍可推进。P5 不等待 CSV 优化才能开始，但必须使用已确认一致的目标来源。P1 不要求维护任意历史版本，只登记已完成适配与验证的源码组合；未知版本明确失败。

### 25.6 验收矩阵与上线条件

- 保留现有 55 项基线回归，候选新增证据单独记录模式、补丁摘要、适配版本、源码/安装模块来源及环境；不把旧测试数量累加后称为目标验收。
- 入口：包函数、PrivateUse1 dispatcher、fallback 路由；native 原行为；mock 配置失败；未启用/中途改模式；旧补丁叠加；错误 checkout/安装包来源。
- 请求：相同 shape 不同有效长度、padding、零长、FD、多 section；原始累计长度与有效长度均参与身份；改变 QKV 数值而元信息不变时计划不变。不同核数使用独立 worker。
- 版本：头文件修改触发重建；仅修改原 ParamsInit、初始化规则或 decoder 也触发行为失配；未知格式和适配版本失败；行为身份变化后不能复用缓存或旧 bundle。
- golden：关闭时预处理仍拒绝非支持用例；开启时实际通过共同接口、decoder 与原分段数值函数；注入获取/解码/数值失败，不执行真实 AICPU 或全局参考。验证旧分段缓存不能绕过共同接口。
- 槽位：None、(0,)、固定匹配、固定不匹配、已有非零内容；设备、dtype、连续性和容量与目标一致。CPU 复制替身只能证明槽位行为，真实 H2D 另验。
- CSV：既有用例不依赖 request.json；目标来自文件/CSV 的等价性、冲突、缺省、未知字段及 worker 固定；验证黄金和预处理收到同样的属性来源。
- 目标：记录真实 SoC/架构/有效核数、Runtime 版本、实际安装/构建身份和 tiling；实际 Out/LSE 比较才升级为设备通过。无法运行真实 AICPU时保留对拍未完成，不伪造与原算子一致性结论。

完成 P2/P3 的独立入口和严格失败验证后，才评估是否替换推荐接入；正式提升候选需目标环境 P5 证据。未到此条件前，README 的推荐部署命令继续对应现有 assets_explicit 基线，候选仅在隔离 checkout/worker 验证。
