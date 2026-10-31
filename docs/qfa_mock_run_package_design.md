# QFA metadata mock：`.run` 交付评估与迭代计划

日期：2026-10-31。本文保留设计基线，实际开发进度见主设计第 27 节：共享核心、绑定包装与交付工具已开发，63 项 CPU 回归及 CPU libtorch 验证通过；真实 CANN 原生包、安装与设备尚未验收。下文“待实现”表示设计阶段状态，以最新开发记录为准。

## 1. 结论与选择

**可以设计为：安装一个 `.run`，配置环境变量，原 TTK、CSV 和 assets 继续调用原 `quant_flash_attn_metadata` 接口。** 推荐首期将第 25 节的 `extension_cpp` 做成完整安装包：包内提供真实 qfa、Host mock 核心、带分派的 Torch 扩展源码及其 Python 包。安装后由原 metadata 入口调用内部 `mock_qfa_metadata`，主 qfa 继续调用真实 ACLNN/AI Core。

这里的“上层不改”指 TTK、assets、公开 Torch schema 和业务调用代码不改；底层安装的 Torch 适配包会更新。如果要求连已安装的 Torch 适配包也保持原样，则需采用下述 CANN 原接口方案，其执行机制尚需目标 SDK 验证。

| 方案 | 原调用能否自动走 mock | 评估 |
|---|---|---|
| 只新增不同名字的 CANN `MockQfaMetadata` 并安装 | 不能，原调用仍寻找 `aclnnQuantFlashAttnMetadata` | 不采用 |
| 原 CANN ACLNN 接口内分派到 Host mock | 可以保持 Torch 适配包也不改；需解决两阶段执行、设备数据访问和 executor 生命周期 | 目标 SDK 验证后可升级采用 |
| 原 Torch C++ metadata 入口内分派，随 `.run` 配送更新后的适配包 | 可保持 TTK、assets、调用接口不改；沿用现有 Torch Tensor 复制与扩展机制 | **首期推荐** |

首期 `mock_qfa_metadata` 是 Host C++ 实现的内部名称，不要求新增 Torch schema、GE OpType 或 AICPU 注册。若交付要求必须包含一个独立 CANN 算子，需完成第 8 节验证，不能把内部函数包装成“独立 CANN 算子已完成”。

补丁粒度进一步按第 10 节收敛：优先保留原 native 函数体，通过 PyBind 绑定包装接入独立实现；第 5 节所述“入口分派”不再要求把 mock 逻辑直接写入原 metadata 函数。

## 2. 仓库证据

| 已确认行为 | 源码位置 | 设计影响 |
|---|---|---|
| Torch C++ metadata 入口调用固定的 `aclnnQuantFlashAttnMetadata`，随后返回传入的 output | [quant_flash_attn.cpp](/Users/fanghaolei/Workplace/agent/ops-transformer/attention/quant_flash_attn/torch_extension/csrc/quant_flash_attn.cpp:64) | 另装不同名字的算子不会改变调用 |
| 原 ACLNN 第一阶段校验、创建 executor，并加入 metadata 任务；第二阶段执行 executor | [aclnn_quant_flash_attn_metadata.cpp](/Users/fanghaolei/Workplace/agent/ops-transformer/attention/quant_flash_attn_metadata/op_host/op_api/aclnn_quant_flash_attn_metadata.cpp:37) | CANN 接入必须保持两阶段 API 契约 |
| metadata L0 明确调用 `ADD_TO_LAUNCHER_LIST_AICPU` | [l0_quant_flash_attn_metadata.cpp](/Users/fanghaolei/Workplace/agent/ops-transformer/attention/quant_flash_attn_metadata/op_host/op_api/l0_quant_flash_attn_metadata.cpp:46) | 改库名、安装位置或 AICPU `.so` 名称不能消除 AICPU 执行依赖 |
| 自定义 OpAPI 按 `ASCEND_CUSTOM_OPP_PATH`、`LD_LIBRARY_PATH` 搜索；库句柄和 API 地址使用静态缓存 | [aclnn_common.h](/Users/fanghaolei/Workplace/agent/ops-transformer/torch_extension/cann_ops_transformer/common/aclnn_common.h:353)、[ACLNN_CMD](/Users/fanghaolei/Workplace/agent/ops-transformer/torch_extension/cann_ops_transformer/common/aclnn_common.h:1048) | 环境须在启动 worker 前设置；两个 ACLNN 符号必须来自匹配版本，不能进程中途换库 |
| 内置包可收集 `cann_ops_transformer-*.whl`，安装脚本尝试安装到目标 Python 目录 | [package.cmake](/Users/fanghaolei/Workplace/agent/ops-transformer/cmake/package.cmake:130)、[opp_install.sh](/Users/fanghaolei/Workplace/agent/ops-transformer/scripts/package/ops_transformer/scripts/opp_install.sh:366) | 已有单 `.run` 携带 Torch 包的基础，但当前失败处理不足以保证可用安装 |
| custom vendor 安装路径升级 `op_impl`、`op_api` 等并生成环境脚本，未见同等的 wheel 安装逻辑 | [custom/install.sh](/Users/fanghaolei/Workplace/agent/ops-transformer/cmake/scripts/custom/install.sh:275) | 首期 custom `.run` 需要补齐 wheel 配送、安装和激活 |
| 现有 wheel 复制算子 Python 和顶层 C++ 源文件；扩展在首次调用时通过 Torch JIT 编译 | [setup.py](/Users/fanghaolei/Workplace/agent/ops-transformer/torch_extension/setup.py:110)、[builder.py](/Users/fanghaolei/Workplace/agent/ops-transformer/torch_extension/cann_ops_transformer/op_builder/builder.py:179) | wheel 已安装不等于扩展已编译；共享核心、头文件和适配文件也必须入包 |
| 原 Python 在调用 C++ 前，按设备查询的核数分配 metadata | [quant_flash_attn.py](/Users/fanghaolei/Workplace/agent/ops-transformer/attention/quant_flash_attn/torch_extension/quant_flash_attn.py:86)、[分配入口](/Users/fanghaolei/Workplace/agent/ops-transformer/attention/quant_flash_attn/torch_extension/quant_flash_attn.py:413) | 目标核数配置与 output 容量必须协同，安装 mock 本身不能解决容量问题 |

## 3. 推荐执行链

```text
安装：一个 .run
       ├─ 原 qfa OpAPI、tiling、AI Core kernel
       ├─ 更新后的 Torch 适配 wheel：原接口 + metadata C++ 分派 + Host 核心
       └─ 环境脚本、配置模板、构建清单、安装检查工具

运行：原 ttk e2e + 原 CSV + 原 assets
       ├─ golden：包导出的 quant_flash_attn_metadata ──┐
       └─ 预处理：torch.ops.quant_flash_attn_metadata ─┤
                                                     ↓
                                          原 Python metadata API
                                            配置绑定、容量分配
                                                     ↓
                                          原 C++ metadata 入口
                         ┌───────────────────────────┴───────────────────────┐
                       native                                             mock
                  原 metadata ACLNN                             内部 mock_qfa_metadata
                       AICPU                                  读取实际长度 → Host 计算
                                                                   校验 → 写入 output
                         └───────────────────────────┬───────────────────────┘
                                             原设备 metadata Tensor
                              ┌──────────────────────┴───────────────────┐
                           golden                                    预处理
                       读回、解码、CPU 数值计算                  原 None/(0,)/固定槽回填
                                                                         ↓
                                                           原 qfa ACLNN → AI Core
                                                                         ↓
                                                              Out、LSE 比较及结果 CSV
```

共享核心沿用第 25 节拆分：标准 C++ Request/Target/Result、版本适配、原分块/SectionStreamK、编码和覆盖校验；helper 与扩展编译同一份核心。mock 分支不启动 helper 子进程，不提交 metadata ACLNN/AICPU，不读取 CSV 或 Q/K/V 数值。

长度输入仍是本次真实 Tensor。配置只提供目标设备信息，不能代替变长输入。Host 读取长度及写入 metadata 仍依赖仿真 Runtime 的设备复制、同步能力。

## 4. 容量契约：收紧上一版的“精确目标 shape”要求

需要区分**调度使用的目标核数**和**输出分配容量**。Header/FA/FD 必须使用用户配置的目标核数；output 可以比有效计划大，不能因其分配时查询了另一组核数就直接判定计划错误。

实际需要的单行 int32 元素数为：

```text
required = 16 + sectionNum × (target.aicCoreNum + target.aivCoreNum) × 16
```

该公式来自原 [GenMetaData](/Users/fanghaolei/Workplace/agent/ops-transformer/attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp:1021)。`rowSize` 是物理行容量；不能用行容量反推实际核数或 section 数。

| 路径 | 容量策略 |
|---|---|
| 推荐的更新后 Torch 包 | 按配置目标计算原公式的安全上界，分配 `(2,rowSize)`；先完整校验再写入，返回同一个 output |
| Torch 包完全不改的 CANN 路径 | 接受已有 `(2,rowSize)`，按实际 `required <= rowSize` 检查；使用已有行距编码，剩余区域按协议初始化。容量不足立即报错，不截断、不修改 Header 冒充其他核数 |

核心接口应接受显式 `rowSize`，不将所有调用限制为唯一的目标上界 shape。默认分配可以固定采用目标上界，bridge 对调用方提供的 output 则检查 dtype、维度、连续性、设备、实际容量和协议。

**CSV 保持原样的首期用例优先选 metadata 动态声明 `(-1,-1)` 或空占位 `(0,)`。** 原 assets 会检查固定槽与返回 Tensor 的 shape 相等；如果 CSV 已固定为旧容量，而更新后的包分配新容量，会在原预处理中失败。因此不能对任意固定 shape CSV 承诺只改环境变量即可运行。首期记录兼容条件；如要兼容全部旧固定槽，需要另设计可选容量约定，不能静默 resize 或绕过原 shape 检查。

## 5. `.run` 包内容与安装工作

优先交付隔离的 custom vendor 包，避免直接替换公共 CANN 安装；包内包含 qfa 及其所需依赖。native metadata 是否随包保留，按 native/mock 双模式验收需要明确列入清单。不能假定 `--ops=quant_flash_attn` 自动覆盖所有依赖。

| 部分 | 计划改动 |
|---|---|
| `attention/quant_flash_attn/torch_extension/csrc/quant_flash_attn.cpp` | metadata 函数加入 native/mock 分派；完整保留主 qfa 导出与量化路径 |
| 同目录 Python metadata API | 配置绑定、目标容量分配、能力检查；公开签名与返回值不变 |
| 共享核心及版本适配 | 纳入 wheel，安装后不依赖开发机的 mock 仓库或 ops-transformer 源码绝对路径 |
| 算子 OpBuilder 与 `torch_extension/setup.py` | 收集所有新增 `.cpp/.h` 和嵌套文件，声明 JIT sources/includes、重建依赖与版本身份 |
| `build.sh`、打包 CMake | 让所选 `.run` 构建分支显式构建并收集唯一匹配 wheel；不能遗漏 custom 分支，也不能沿用“缺 Python build 模块则跳过成功” |
| 实际启用的安装脚本 | 安装到包自身 Python 前缀，校验安装完成；缺 pip、缺 wheel 或导入失败必须报错，不能仅复制 wheel 后宣称完成 |
| `set_env.bash` | 激活包的 Python 路径、OpAPI/依赖库路径、OPP 路径；mock 默认关闭，目标配置由用户选择 |
| 检查工具与清单 | 检查实际包路径、entry point 来源、扩展路径/摘要、ABI、core/adapter/protocol 身份、目标和模式；成功事件记录实际 metadata 生产路径 |

这里还需处理两个现有细节：

- `setup.py` 在选算子时将 wheel 命名为 `cann_ops_transformer_<vendor>`，现有内置打包/安装的硬编码文件名匹配不能直接照搬。安装器应读取本次构建清单，禁止随意选一个历史 wheel。
- assets golden 仍导入 `cann_ops_transformer.ops`；vendor wheel 需要通过现有 entry point 机制正确提供 qfa 和 metadata。同一进程存在多个同名 provider 时必须检查实际来源并拒绝歧义，不能只设置 `PYTHONPATH` 就假定选中了新实现。

首期可沿用安装后首次调用 JIT，但必须在交付说明中列出编译器、Torch、torch_npu、CANN 依赖，并在部署验收时预热扩展。若要求运行机器没有编译器，另增 ABI 匹配的预编译扩展及加载逻辑；当前 wheel 不是这种交付方式。

## 6. 环境变量和原 assets 的限制

以下是**待实现的统一开关约定**，不是当前可运行的新命令：

```bash
source <安装前缀>/bin/set_env.bash
export QFA_METADATA_BACKEND=mock_aicpu
export QFA_METADATA_CONFIG=<绝对路径>/mk001.json
# 然后执行原 ttk e2e 命令，--plugin 继续指向原 assets。
```

兼容旧配置：未设新开关而 `QFA_METADATA_CONFIG` 非空时启用 mock；两者均未指定时 native；显式 native 与非空 mock 配置冲突报错。每个 worker 固定 backend、target 和构建身份；切换后重新启动，不热切换动态库、目标或模式。

目标包含 SoC 调度标识、qfa 编译架构、有效 AIC/AIV 核数。已知 mk001=80/80，仍须确认实际架构映射；该配置不改变主 qfa 的编译或 Runtime 控核设置。首期限定一次完整输入、mode=1、forward Eager、D=Dv=128、已适配布局及默认窗口。可见参数校验放在包内；完整用例的限制还需在主 qfa C++ 入口或外部验收清单检查，metadata 接口看不到 V-tail、sinks 等主算子输入。

本轮撤销第 25 节“新模式必须再打 assets 严格保护补丁”的部署要求，以满足原 assets 不改。代价需明确：

1. 原 golden 的 metadata/解码异常、分段数值异常仍可能被捕获并回退全局参考。mock 层能传播自身错误，不能让上层已有 catch 自动消失。
2. 原 golden 的分段缓存不含 target/backend/version；一个 worker 一个固定目标和版本，迁移后重启。首次调用与重复命中分开验收，不以固定调用次数判断是否覆盖两个入口。
3. 发布验收必须独立记录 golden 成功取得并消费分段计划，以及真实 qfa 使用了同身份 metadata；仅有 TTK PASS 或 mock 生成日志不够。可由独立测试入口/执行追踪采集，不修改交付的 assets。若要求任意异常都禁止 golden 回退，则仍需单列 assets 或 golden 上游改进，不能同时承诺完全原样。

独立验收需特别包含“metadata 合法，但分段数值函数抛错”的用例，证明能识别原 golden 回退；不能靠更换异常类型或进程强退掩盖这个问题。

## 7. 推荐方案分阶段计划

| 阶段 | 工作与交付 | 完成条件 |
|---|---|---|
| R0：契约与安装探测 | 确认目标架构、有效核数、实际 `.run` 构建/安装分支、现有 CSV 槽位、Runtime 复制及同步能力 | 记录可复现的目标环境和包布局；明确动态槽/固定槽适用范围 |
| R1：共享 Host 核心 | 按第 25 节抽离核心与适配，补全纯 C++ 编码/覆盖校验，输出接受显式行容量 | 原 CPU 基线保持；不同容量、80/80、变长、零长、FD、多 section 独立回归；架构未确认时不标记目标适配完成 |
| R2：同名兼容入口 | 真 Torch C++ 分派、配置绑定、目标分配；主 qfa 用例范围校验；不依赖 assets 补丁 | 包直调、torch.ops、扩展直调都覆盖 mock；返回实际 output；mock 不调用原 metadata ACLNN；主 qfa 仍调真实 ACLNN |
| R3：完整 `.run` | wheel 内容收集、custom 安装、环境激活、依赖与构建清单、安装检查、互斥迁移 | 干净目标环境仅安装 `.run` 即可解析并加载所有组件；不引用开发仓库，TTK/assets 文件摘要不变；缺依赖不能假成功 |
| R4：无改动链路回归 | 原 golden、原预处理的动态/空/固定槽，错误与缓存，多 provider 和旧库检测 | 两种上层调用均到新入口；golden 分段结果实际成功；固定槽不兼容场景明确失败；旧三文件补丁不得叠加 |
| R5：目标仿真 e2e | 先关闭 golden 检查 metadata 写入、实际 qfa tiling/核数，再开启 golden 比较 Out、LSE | 记录 metadata 调度无 AICPU、真实 qfa 有 AI Core 执行、Out/LSE 通过及结果 CSV；辅助 Runtime 能力单独观察 |

native 回归与 mock 回归都覆盖同一包；主 qfa 不能因替换 metadata 丢失导出或改变执行路径。与真实 AICPU 的同配置逐字节对拍另列证据，不能由 55 项旧测试替代。

## 8. 独立 CANN 算子的升级路线

如果要求现有 Torch 适配包也完全不更新，先做 **C0 执行机制验证**，再决定是否投入独立 `MockQfaMetadata`：

1. 保留原 `aclnnQuantFlashAttnMetadataGetWorkspaceSize` 和 `aclnnQuantFlashAttnMetadata` ABI，在它们内部选择 native/mock。两个入口及依赖须来自同一个匹配包；不依赖 `LD_PRELOAD` 只覆盖其中一个。
2. 第一阶段只校验/保存请求及配置并返回合法 workspace/executor；不能把设备长度地址当 Host 指针，不能忽略同 stream 上尚未完成的长度生成。设备数据读取与 Host 计算应安排在能取得正确执行 stream 和依赖的位置。
3. 第二阶段经目标 SDK 支持的 executor/Host 执行机制读取长度、调用共享核心、写入原 output。首版允许同步执行，以保证输入就绪、输出可见及 Host 缓冲生命周期；先验证线程、设备上下文、内存释放、异常清理和重复调用，再优化异步。
4. 必须使用真实 SDK 支持的生命周期与执行接口。当前仓库已确认的 metadata L0 只有 AICPU launcher，尚未找到可直接复用的 Host-only metadata executor 示例。因此不能预设某个 Host launcher API 存在，不能自造 `aclOpExecutor*`、将私有结构强转成 executor，或在 callback 内再次调用同一 stream 的同步操作。
5. C0 用真实 Torch 调用和普通 ACLNN 两阶段调用，至少覆盖“前序异步写长度→metadata→后序读取 metadata”、两 stream、非默认 device、失败释放及容量不足。记录所有任务类型，证明 mock 不需要 AICPU。

C0 通过后，再实现内部 `l0op::MockQfaMetadata` 或对应 SDK 执行组件，由原 ACLNN 入口分派；独立公开的 `aclnnMockQfaMetadata*` 仅在确有调用需求时增加。输出使用原调用方提供的容量，按第 4 节处理。打包进同一个 custom `.run`，这样才有依据承诺 Torch 适配包也不改。

仅把现有 AICPU kernel 换名打包仍需 AICPU；把分核算法搬到 AI Core 则是另一种设备算子开发，不是本轮 Host mock 的最小改造。Host tiling 生成计划再用 AI Core 写回也需单独验证长度值依赖和可变计划传输容量，不作为无需验证的捷径。

**执行决策：先完成 R0—R5 的 `.run` 打包方案；C0 是可并行评估的后续选项，未验证前不替代已定位清楚的 Torch C++ 接入点。** 两条路线复用同一 Host 核心，一次部署只选择一个 metadata 生产入口。

## 9. 耦合度与跨版本兼容策略

推荐方案对 TTK/assets 的源码侵入较低，对 Torch 适配与构建存在直接耦合，对 QFA 调度及 metadata 协议存在强语义耦合。改动行数少、公开接口不变，均不能作为跨版本兼容证据。

| 耦合面 | 程度 | 升级风险与处理 |
|---|---|---|
| TTK、assets 调用契约 | 低，但存在行为依赖 | 返回 Tensor、槽位规则、golden 导入/回退行为变化时重新验证两个入口及结果消费 |
| Torch Python/C++ bridge | 中 | 参数顺序、默认值、分配策略、PyBind 导出变化可能导致补丁失败或传参错误；按目标源码适配 |
| Torch、torch_npu、CANN、编译器 | 强构建/运行依赖 | JIT 扩展与链接的库必须匹配；升级后重新构建并验证实际加载对象，不承诺旧 `.so` 通用 |
| QFA 算法、架构策略与 metadata 协议 | 高，核心兼容风险 | 字段偏移、FA/FD、分块、长度规则、初始化标志、tiling 及 kernel 消费方式必须匹配同一已验证适配 |
| `.run`、wheel、vendor 加载机制 | 中 | 安装目录、命名、entry point 与依赖搜索变化时调整打包适配并做干净安装验证 |

一个具体风险是：mock 复用了 SectionStreamK 头文件，但原 AICPU `.cpp` 中的策略发生变化。当前 [ParamsInit](/Users/fanghaolei/Workplace/agent/ops-transformer/attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp:298) 包含架构分支、FD/L2 策略及分块参数设置，[初始化逻辑](/Users/fanghaolei/Workplace/agent/ops-transformer/attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp:341) 也会影响输出。只重编译头文件依赖，可能继续得到可编译、shape 正确但与新 qfa 不匹配的计划。

因此兼容性分三项登记，不能互相替代：

- **补丁兼容**：能否准确修改目标版本的集成点。使用小范围补丁及源上下文/内容检查，不整文件覆盖，不强行模糊应用。
- **构建兼容**：能否在该 Torch/torch_npu/CANN/编译器组合中编译、加载并正确复制 Tensor。JIT 重编译解决的是这一部分。
- **行为兼容**：生成的计划是否匹配该版本 qfa 的协议、调度及数值行为。接口相同、编译成功和安装成功都不能证明这一项。

后续实现将版本适配集中在 `metadata_adapter`，Torch/Runtime 调用集中在 bridge，纯调度核心继续复用目标源码。构建清单同时记录编译依赖和行为依赖：原 AICPU 参数/初始化策略、分块/调度/编码、主 qfa 的 metadata 消费和 tiling、Python 容量及 golden decoder。清单是身份与变化检测依据，首次兼容结论仍需验证，不能由内容摘要自动推导。

升级处理约定：

1. 目标源/行为依赖变化时，旧适配的验收结论失效，先比较变化并决定复用还是新增适配；未知组合不自动启用 mock，也不静默回退 native。
2. Torch/Runtime ABI 或扩展源变化时重建扩展；调度/协议变化还必须重新验证核心和消费者，不能只重新 build。
3. 适配或二进制身份变化时使缓存失效并重启 worker；同一个 `.run` 的 qfa、mock 核心和 Torch 适配来自一致的发布清单，禁止单独混换旧核心。
4. 每个发布组合分别登记补丁应用、干净构建/安装、CPU 协议与覆盖、原 golden 分段执行、仿真 qfa Out/LSE，以及具备条件时的真实 AICPU 同配置对拍。不同核数平台的结果不能当成 80/80 的逐字节对拍。

当前还没有新方案的跨版本验收矩阵。首期承诺范围应限定为明确的 QFA 源码适配、目标架构和工具链组合；mk001 的 80/80 配置本身不能授予新版本或新架构兼容性。

## 10. 降低补丁结构依赖：绑定包装与独立文件

### 10.1 推荐接入方式

首期采用**一个绑定包装点、一个容量接入点、独立实现文件和构建声明**。保留原 C++ metadata 函数及其 native ACLNN 调用，不复制原函数体，不改其强弱属性；在原 PyBind 注册处将原函数指针交给新包装器。包装器在运行时决定调用捕获的 native 函数还是 Host mock。

下面只是接口示意，不是已实现代码：

```cpp
// 原注册
m.def("quant_flash_attn_metadata", &quant_flash_attn_metadata, ...);

// 候选注册：公开名称不变，包装器持有原函数指针
m.def("quant_flash_attn_metadata",
      qfa_mock::MakeMetadataBinding(&quant_flash_attn_metadata), ...);
```

包装器返回 PyBind 可识别的、有明确参数类型的 callable；可以用函数指针类型推导保留调用签名，不使用无法推导签名的泛型 lambda 直接注册。新参数出现时，请求转换适配必须显式支持或编译失败，不自动忽略。原函数通过捕获的指针调用，不依靠 `RTLD_NEXT` 或再次寻找公开 metadata 导出，避免递归。

| 位置 | 原文件内的最小改动 | 独立文件职责（名称待实现时定稿） |
|---|---|---|
| C++ 模块绑定 | include 新头文件；修改 metadata 的一个 `m.def`；增加必要的私有控制接口注册 | `qfa_metadata_binding.h/.cpp`：类型化包装、native 函数指针、模式分派、Tensor 读取/写入、私有配置/容量查询 |
| Python metadata API | 分配前调用一个准备函数，完成配置绑定和容量取得 | `_qfa_metadata_backend.py`：解析/固定配置、取得同一扩展、校验能力和目标；native 复用原容量函数 |
| 算子 OpBuilder | 该算子的 sources/includes 增加独立文件 | 显式收集所有实现与头文件；不修改全局编译器行为 |
| wheel 与 `.run` | 增加文件收集和安装声明 | 使用前述安装与身份检查机制 |
| 调度核心 | 无须将算法插入原函数 | `metadata_core`、`metadata_adapter`：沿用第 9 节的纯核心与版本隔离 |

这样，主 qfa 的绑定与函数体、原 native metadata 函数体和原 assets 都可以保持不变。原 native 函数本身仍是 native；对外导出的 PyBind metadata callable 才是分派入口。当前仓库两种上层调用都经过该导出，单独验证新增的原生 C++ 内部直调是否绕过包装；不要把“任意 C++ 调用都被覆盖”作为保证。

容量准备函数需要在实际 Eager 分配 output 前完成配置绑定，与包装器使用同一上下文。不能只包装 C++ 执行而忽略原 Python 的容量计算。当前不扩展 graph/Fake 支持；若后续支持，形状推导另行接入相同契约，不能在 Fake 路径隐式读取设备数据或启动真实计划计算。

### 10.2 强弱符号是否适用

可以使用，但更适合**新设计的可选 hook**。现有 [C++ metadata 函数](/Users/fanghaolei/Workplace/agent/ops-transformer/attention/quant_flash_attn/torch_extension/csrc/quant_flash_attn.cpp:64) 没有 weak 声明；不能靠再增加一个 weak 实现替换它。若在同一次链接加入两个同名普通强定义，则存在重复定义问题。GCC 的 weak 机制要求覆盖符号类型匹配，见 [GCC 属性说明](https://gcc.gnu.org/onlinedocs/gcc/Common-Attributes.html)。

可选方案是在独立接入层预留命名明确的 backend hook：

```text
分派层 → QfaMetadataBackendHookV1
          ├─ 默认文件：weak，报告 mock 能力不可用
          └─ mock 文件：strong，提供 Host 核心适配
```

该方式用于**同一个最终 `.so` 的受控链接**。启用 mock 的构建显式加入 strong 实现的 object/source；默认 weak 属性只放在默认定义上，不能通过公共声明把 strong 实现也标成 weak。不要依赖静态库中“恰好被抽取”的对象或链接顺序，发布时检查 link map 和符号表。

链接期决定 hook 能力是否存在；运行时环境变量仍决定选择 native/mock。显式请求 mock 而只链接默认 hook 时必须报错，不能转 native。执行 hook 和容量能力必须来自同一后端版本，mock 错误也不能被当作“未处理”退回 native。

相较绑定包装，该方案增加了符号与链接验收工作，未减少容量接入、版本适配或算法协议依赖。因此不作为首期默认；如果上游愿意长期维护统一 hook，可作为可选后端扩展方式。

### 10.3 不依赖全局符号覆盖

当前 [Torch builder](/Users/fanghaolei/Workplace/agent/ops-transformer/torch_extension/cann_ops_transformer/op_builder/builder.py:144) 使用 `-fvisibility=hidden`；若尝试在 CANN 层覆盖，[custom_build.cmake](/Users/fanghaolei/Workplace/agent/ops-transformer/cmake/custom_build.cmake:101) 还为 `cust_opapi` 指定 `-Bsymbolic`。后者会使库内引用绑定本库定义，不能将同一最终链接内的强弱选择等同于运行时跨库抢占。[GNU ld 说明](https://sourceware.org/binutils/docs/ld/Options.html)

`--wrap` 主要替换未定义符号引用；当前 native metadata 定义与 PyBind 引用在同一编译单元，不能假定加链接参数就能接管。另一个候选目标 `aclnnQuantFlashAttnMetadata` 在现有 `ACLNN_CMD` 中是按名字动态查找后通过函数指针调用，也不是普通的未定义直接调用。[GNU ld 的 --wrap 规则](https://sourceware.org/binutils/docs/ld/Options.html)

因此不以全局 `LD_PRELOAD`、同名 C++ 符号抢占、宏替换整文件 include、二进制符号改名或允许重复强定义作为主方案。这些方式会额外依赖加载顺序、名字修饰及内部结构，仍不能解决调度语义兼容。

### 10.4 后续可选：稳定的显式插件接口

若希望 mock 核心独立升级，可在上述包装器之后增加显式 backend 插件接口：通过唯一、带版本的 C 导出取得接口表，bridge 主动链接或加载该插件。接口仅传 Host 上的固定宽度标量、长度数组、容量和结果缓冲区；Torch Tensor、stream、设备复制留在 bridge。约定 ABI 版本、struct size、能力、缓冲所有权、错误码；不跨边界传 STL/Torch 对象或 C++ 异常。

这可以隔离大部分 Torch C++ ABI 依赖，使插件不必随每次 Torch 包重编译，但仍受 Host 平台/运行库和 QFA 语义版本约束。ABI 版本与 metadata 协议/策略版本分别检查。该插件是 Torch 包装器的后端，不需要等待第 8 节的 CANN executor C0 验证；也不等于独立 CANN 算子已实现。首期仍可直接编译独立源文件，暂不增加动态插件部署复杂度。

### 10.5 补丁应用与验收调整

- 补丁工具按文件身份、作用域、函数签名和唯一绑定定位修改点；Python 可使用语法树定位，C++ 有工具链时使用编译器语法树，否则采用明确版本的小范围补丁。禁止仅靠固定行号、全局字符串替换或正则匹配任意大括号范围来修改函数。
- 限定原文件改动区域并保存差异。目标不唯一、签名/调用契约不匹配时先报告，不强行应用；原文件不要求全文字节一致，非相关函数变化不应自动导致补丁不可应用，但行为依赖变更仍须按第 9 节复核。
- R2 增加绑定 callable 的真实验证：包直调、torch.ops、扩展导出直调均命中包装；native 指针调用次数正确，mock 不调用 native；两种错误均向上传播且不递归。原 qfa 导出和 native metadata 函数体保持原样。
- R3 校验新 Python/C++ 文件、include 和版本清单全部入包，缺包装对象或后端时不能构建/安装假成功。若选弱 hook 分支，还需记录弱默认/强实现各自的符号与最终选中对象。
- R4/R5 继续验证容量、槽位、golden 分段、Runtime 与仿真 Out/LSE。这一优化降低补丁对文件布局与函数体的依赖，不授予新的 QFA 语义兼容结论。

本节为补丁设计调整，未添加实现文件或运行新的符号/设备测试；R2/R3 的具体实现以本节为准细化。


## 11. 开发记录与交付边界（2026-10-31）

第 10 节的绑定包装与容量准备已实现；原 native C++ 函数体、主绑定、assets 和 TTK 保持原样。helper 与扩展使用同一 Host 核心；core/bridge 不回调 Python provider 或启动 helper 子进程。源码接入工具、独立 vendor wheel、基础加载框架配送、单 .run 组合器和严格安装代码已交付；vendor 与行为身份、真实原生安装文件摘要及 provider/JIT 均有检查。

本机以实际 setuptools vendor wheel、归档完整性和 CPU libtorch 验证代码，不以 trap fixture 宣称构建了 CANN .run。真正原生包需由目标构建产生，作为组合器输入。环境探测记录缺 Torch NPU/CANN，因此 R2—R5 的真实目标验收尚未完成。原 golden 回退保持不变，新增独立审计可拒绝分段未成功的结果。详见主设计第 27 节及独立仓库 EXTENSION_USAGE.md / PROGRESS.md。
