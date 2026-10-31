# QFA Mock AICPU 下一轮迭代计划

主设计第 25 节的当前副本；首选 extension_cpp，在 Torch 加载的真实 C++ 扩展内分派 metadata。当前已实现基线仍为 eef4af5 的 assets_explicit，以下除文档更新外均待实施。

## 25. 下一轮实施计划：在 Torch C++ 扩展中 mock metadata（2026-10-31）

本节按用户最新决策更新，取代上一版第 25 节“共同 Python 函数直接调用 provider”的实施路线；第 23—24 节保留为分析依据和 Python 入口备选，不再是下一轮首选接入点。当前已交付实现仍是 `eef4af5` 的三文件 assets 补丁，55 项 CPU 证据不变。本轮仅优化设计，未改运行代码、未编译新扩展、未新增设备结论。

### 25.1 决策、范围与真实接入点

首选方案标签为 `extension_cpp`：修改现有 Torch 扩展中 `op_api::quant_flash_attn_metadata` 的 Host C++ 函数，在调用 `ACLNN_CMD(aclnnQuantFlashAttnMetadata, ...)` 前选择 native/mock。PyBind 导出名、C++ 入口签名、Torch schema、包导出 API 和主 qfa 调用方式保持兼容。它替代 metadata 的执行入口，不替换 CANN 全局 ACLNN 符号、不部署伪 AICPU kernel、不依赖全局函数 monkeypatch，也不靠 Meta/Fake 返回空数据冒充计划。

源码接入点：`attention/quant_flash_attn/torch_extension/csrc/quant_flash_attn.cpp` 的 metadata 函数与 `PYBIND11_MODULE`。当前同一扩展同时导出 metadata 和主 quant_flash_attn，不能把整个模块替换成只含 metadata 的假模块。保留主函数、量化分支和真实 ACLNN/AI Core 路径，只在 metadata 函数中加入 Host mock 分支。

候选交付一个统一补丁包，范围允许扩大到 Torch 扩展 C++ 绑定、Python 容量/配置适配、OpBuilder 构建声明及必要的 assets 严格失败保护。共享计算核心放在 mock 仓库，补丁通过明确构建路径接入，不复制第二份算法。TTK、Torch schema、生产 AI Core 和 AICPU kernel 算法不改；修改的是 Torch 扩展的 Host 调用胶水。

支持范围不扩大：一次完整输入、MXFP8 quant_mode=1、正向 Eager、D=Dv=128、Q/scale/Out=TND、KV=TND 或 PA_BNBD、默认窗口、显式正最大长度。mask=0/3/5 的计划支持与每个组合的数值/设备验收分开记录。不新增历史 KV cache、图模式、反向或同 worker 多目标。

### 25.2 运行链路与责任划分

```text
TTK CSV → 原输入生成 → assets 对本次真实用例作范围/来源校验
                                      ↓
golden 包函数直调 ──┐        Python metadata API
                    ├──────→ 参数规范化、固定目标与构建身份
预处理 torch.ops ──┘        按目标计算容量、分配原设备 output
                                      ↓
                         op_builder.load() 的真实扩展
                                      ↓
                      C++ quant_flash_attn_metadata
                        ├─ native → 原 ACLNN → AICPU
                        └─ mock   → 长度向量复制到 Host
                                     → 共享 Host 核心计算与校验
                                     → 写入调用方的设备 output
                                     → 返回同一个 output
                                      ↓
             golden：同步/读回/decoder → 原分段 CPU Out/LSE
             预处理：原 metadata 槽位回填 → 真实 qfa AI Core
```

Python 不再调用 MetadataProvider.generate 或启动 metadata_helper 来生产本模式的计划；它负责数据来源、支持范围、配置绑定、容量计算、调用与结果契约检查。C++ 是唯一 native/mock 计算分派点，mock 分支不得回调 Python provider 或再次调用 torch.ops metadata 形成递归。

golden 与预处理必须真实经过同一个已加载的 C++ metadata 导出函数。移除候选模式下旧 assets 显式 metadata_plan 的短路；可保留原数值函数的参数兼容性，但不能把旧路径成功记成新扩展接入成功。关闭 golden 时预处理独立工作。两个入口可以重复计算；若开启缓存，只复用同身份 CPU 主副本，不要求跨进程共享 provider 对象。

### 25.3 共享 Host 计算核心及 helper 拆分

建议职责如下；名称是规划，尚无这些新增文件/接口：

| 组件 | 计划职责 | 依赖 |
|---|---|---|
| `metadata_core.h/.cpp` | 类型化 Request、Target、Result；长度正规化、分块/调度、编码、容量/覆盖校验 | 标准 C++ 与目标 QFA 头文件；无 Torch/CANN Runtime/CSV/Python |
| 集中的版本适配模块 | BaseInfo/DeviceInfo、架构选择、FD/L2 策略、needInitOutput、格式约束及能力清单 | 对应版本源码；记录来源与适配身份 |
| `metadata_helper.cpp` | 解析现有 wire，调用核心，输出 bin/摘要 | 同一套 Host 核心；独立 CLI 继续可用 |
| Torch C++ bridge | 读取实际标量/长度 Tensor，调用核心，验证并写入 output | Torch、torch_npu/CANN 及同一套 Host 核心 |
| Python 配置/容量适配 | 解析 JSON/后续 CSV、绑定上下文、取得核心规定的 output shape | 不承担第二套分核计算 |

共享指同一份源文件、类型契约和适配版本被 helper 与扩展各自编译使用，不要求将同一个预编译静态库跨平台复用。核心没有 main、进程启动、JSON/CSV 解析或设备 Tensor；扩展不启动 helper 子进程。当前 Python provider 继续服务既有部署与独立 CLI，不强行把它塞入 C++ 计算链。

首轮以现有样本逐字节等价为目标拆分，不改分核算法；AdjustSinnerAndSouter*、SectionStreamK::Compute、编码类继续来自目标源码。核配比、D=128 等是当前适配的能力限制，不是所有 QFA 版本的通用定律。长期让原 AICPU 也共用这套纯计算入口仍是单列选项，不作为本轮前置条件。

### 25.4 配置绑定、输出容量与校验契约

1. **先固定配置，再分配输出。** Python 解析现有目标 JSON 或后续 CSV 目标，向真实扩展的内部配置入口绑定类型化 backend、target、适配/构建身份。内部 PyBind 控制接口可以新增，但公开算子签名不加 mock 参数。模式和目标首次绑定后不可改变；C++ 执行入口复核，mock 请求未初始化时失败，不能默认落到 native。具体内部函数名在实现时定稿。
2. **Python 仅适配容量。** 原 `_calculate_max_schedule_size` 使用设备查询核数，mock 下必须改为使用已绑定目标；由核心提供 shape-only 查询，或共享生成的容量契约，避免 Python/C++ 各维护一套不一致公式。此查询不是 metadata 生产入口。native 保留原容量语义；mock 不先按物理默认值分配，也不先分配错误尺寸后改 Header。
3. **C++ 写入前严格检查。** 验证 request、Target、dtype=int32、形状 `(2,rowSize)`、连续性、设备和容量；当前约定精确 shape 匹配，输入/输出设备符合原调用语义。错误不 resize、不截断、不写半成品、不发起真实 metadata ACLNN。候选仍返回传入的 output 对象；公开 Python 返回语义不变。
4. **小输入读取与完整结果写入。** 只将 cu_seqlens/seqused 等长度向量读取到 Host，不复制 Q/K/V 数值。使用目标后端支持的 Tensor 复制/同步接口和正确设备/stream，不直接解引用 NPU data_ptr；Host 计算结果先完整验证再写入 output。写入错误直接失败、不返回成功结果；Runtime 可能已部分复制，不能承诺设备 copy 的事务回滚或无设备依赖。
5. **校验进入实际 C++ 链路。** 核心/bridge 必须覆盖 Header、FA/FD、初始化标志、容量、保留区和任务覆盖；关闭 golden 时也执行。Python 原 decoder/validation 继续作为独立回归参考，golden 实际消费时照常解码。不能因为绕过 Python provider 而漏掉原有覆盖检查，也不在持锁计算路径中隐式回调 Python 校验。
6. **单一上下文与可选缓存。** 以实际 PID、backend、target、适配版本和扩展构建身份固定上下文，fork 后重新初始化，禁止 worker 中途改环境或目标；同 worker 不混跑 native/mock。首轮允许无缓存确保正确；如保留性能缓存，只设一层有界 C++ CPU bytes 缓存，键含规范化请求及完整版本身份，每次独立物化输出，失败结果不缓存。
7. **完整用例检查留在边界。** metadata 函数只有其实际实参，不知道未传入的 V-tail/sinks 或主 qfa 参数。assets 仍校验完整 QFA 用例，显式补齐窗口等参数并比对双入口请求；独立 metadata 调用只检查可见实参。不借另一用例的模块全局状态补参数，调用范围上下文在成功/异常后清理。

现有 NPU Runtime、分配、H2D/D2H 与同步仍然必需；C++ mock 只消除 metadata AICPU 调度，不能保证仿真 Runtime 的辅助任务都不需要 AICPU。先在目标环境核实这些能力。原 npu_preprocess 的 None→字典回填、(0,)→resize/copy、固定槽→严格匹配/copy 保留；这些调用侧动作与 C++ 对已分配 output 的禁止 resize 不是同一层。

### 25.5 编译、版本身份与部署

编译分为两类产物，不把现有 helper 构建成功记成扩展部署成功：

| 产物 | 编译方式与依赖 | 使用方 |
|---|---|---|
| Host helper | 保留 `python3 -m qfa_mock build --repo ...`，普通 C++17 编译同一核心；不依赖 Git 历史或 Torch/CANN Runtime | 独立 generate/validate、原方案与纯 CPU 回归 |
| 修改后的真实 Torch 扩展 | 在目标项目现有 OpBuilder/JIT 或已确认的安装构建链中，加入共享核心 source/include 和明确适配配置；依赖目标 Python、Torch、torch_npu/CANN、兼容编译器/ABI | 新 extension_cpp 接入，metadata mock 与主 qfa 同模块 |

build 的必需头文件、assets/golden 和 AICPU 参考 cpp 强制全检继续有效，缺失一次列全且不提供跳过。候选扩展构建额外检查绑定源码、builder、共享核心、Torch/CANN 工具链；没有扩展环境时明确只能完成 Host 核心阶段，不静默生成伪完整产物。helper/扩展使用独立构建清单。

Python metadata API 在 mock 模式也需要加载真实扩展，因此不再声称绕过 op_builder.load 或无需 CANN。OpBuilder 需把共享核心及适配身份纳入其依赖/重建判定，避免源核心已更新而 JIT 仍命中旧 .so。构建清单记录 core/adapter/schema/metadata 格式与行为依赖摘要、补丁模式、扩展模块来源、实际二进制摘要及目标工具链信息；核验加载对象的实际 __file__，不靠 .so 文件名推断匹配，不清空全局 Torch 缓存作为常规步骤。

版本分为编译依赖、行为依赖与明确的适配兼容结论。原 AICPU cpp 中 ParamsInit/needInitOutput、decoder、容量/字段语义、assets 或 bridge 的变化都必须触发核对；仅存在性检查或重新 build 不授予跨版本兼容。行为依赖变化也使缓存、bundle 和 worker 身份失效；原始源码与补丁后运行态身份分别记录。相同内容源码压缩包可用，不引入 Git 提交前置条件。

一个统一补丁包交付 bridge、容量适配、构建声明和必要 assets 保护；不是在二进制加载后修改模块属性，也不另造缺少 quant_flash_attn 的同名模块。使用目标项目已有构建方式重建/安装并重启 worker；安装步骤需记录实际命令，当前不虚构已存在的 extension build CLI。

`assets_explicit`（已实现）、`metadata_api`（上一轮 Python 备选）、`extension_cpp`（本轮首选）互斥，补丁工具须检测旧方案并拒绝叠加。迁移先在隔离 checkout/worker 撤销旧补丁，再应用候选、编译/安装、核验模块和开始验收；失败只报告，不自动切换 native、全局 golden 或另一模式。尚未验收时 README 当前推荐命令保持原样。

### 25.6 golden、TTK 与 CSV 的协同

TTK 不改，--plugin 仍指原 assets，--backend npusim/--sim-soc 仍描述实际仿真环境。TTK 读取 CSV、生成张量与实际调用；golden 和预处理经过原 Python API 后落入同一个 C++ metadata 函数。golden 保留设备长度传输、metadata 读回和原分段数值计算；主 qfa 使用原 kernel，比较 Out/LSE。

必要 assets 补丁继续负责：完整支持范围、双入口参数一致性、_HAS_NPU/导入失败显式报错、计划为空失败、decoder/数值异常向上传播。禁用或完善旧分段缓存，防止绕过 C++ 入口或复用另一目标/版本结果；禁止元数据获取失败后回退全局参考。成功日志同时记录 request/target/core/extension/metadata 身份与分段数值成功事件，不能仅证明调用过扩展。

CSV 规则沿用上轮并适配新核心：

| 信息 | 来源与处理 |
|---|---|
| Nq/Nkv、D/Dv、分配形状 | 实际 Q/K/V shape；CSV 属性只作一致性检查 |
| batch、cu_seqlens、seqused | 真实小张量；*_values 为补充，两者存在必须一致；PA KV 不传 cu_seqlens_kv |
| quant/mask/window/layout/max | CSV 装配后的真实形参；与主 qfa 一致，最大长度显式为正 |
| SoC、架构、有效核数 | 当前外部 JSON；后续可用 attributes.qfa_mock_target 的完整目标块替代 |
| 分块、FD/L2、编码规则 | 已适配版本源码；不作为 CSV 自由配置 |
| 源码根、共享核心、helper/扩展路径 | 部署/构建配置；CSV 不消除这些依赖 |

helper 和共享核心均不解析 CSV。独立 CLI 仍接受 request/config JSON；TTK 无需另维护 request.json。CSV 目标块、显式 QFA_METADATA_BACKEND 开关仍待 P4，不能仅填写字段就宣称生效。

P4 约定：未设新开关时兼容非空 QFA_METADATA_CONFIG 启用 mock；显式 native 与 mock 配置冲突失败。CSV 目标不隐式切换 backend，mock 已启用后完整目标块可代替 JSON，同时给出两种来源必须一致，缺字段/未知字段/目标变化失败，不拼接覆盖。Python 在两端消费目标、初始化同一 C++ 上下文，不能临时改环境变量或给公开 schema 添参。一个 worker 一个目标，不同目标拆开运行；CSV 的声明不等于实际 tiling/有效核数证明。

### 25.7 分阶段交付与验收

| 阶段 | 交付 | 前置与验收 | 状态 |
|---|---|---|---|
| P0 | C++ 接入设计、依赖和进度同步 | 区分现有基线与候选；识别目标扩展/Runtime 能力 | 本轮文档已更新；环境实测待做 |
| P1 | 抽出共享 Host 核心、版本适配与纯 C++ 校验；helper 变薄 | 普通 C++ 环境；固定样本/六组结果保持一致，原 Python 校验独立比对，必需文件全检，行为变化可发现 | 待实现 |
| P2 | 真扩展 metadata 分派、内部上下文/容量查询、Python 分配适配、OpBuilder 声明与清单 | P1；目标 Torch/CANN 编译实际 bridge，包直调/dispatcher/扩展直调都到真实 mock C++ 分支；真实 metadata ACLNN 触发即失败，主 qfa 正常 | 待实现，需目标工具链 |
| P3 | 严格 golden、用例上下文、生命周期/缓存/槽位和错误回归；互斥补丁工具 | P2；golden 开关均完整校验；分段成功且错误不回退；output 身份/形状/设备正确；源或二进制变化拒绝复用 | 待实现 |
| P4 | CSV 目标绑定、开关、冲突规则与示例 | P3；无目标 JSON 的 CSV 用例通过，来源一致/冲突、未知字段、worker 换目标均覆盖 | 待实现 |
| P5 | 目标 Runtime 与 AI Core 验收，记录实际部署命令 | P2/P3，CSV 场景另需 P4；先关 golden 检查写入/tiling，再开 golden 比较 Out/LSE；同配置 AICPU 对拍具备条件时另记 | 待目标环境 |

P1 可在当前普通 CPU 环境推进；没有 Torch/CANN 时不得将 CPU bridge 替身记为 P2 完成。P5 不必等待 P4，但应使用已核对的同一目标 JSON。原 55 项基线继续回归；新模式证据独立登记，不沿用旧测试结论。

必须补充的验收项：

- 路由：golden 包直调、torch.ops、C++ 模块导出直调；参数不支持、缺上下文、模式中途变化、native 行为、补丁叠加、错误安装模块来源。
- 核心：变长、padding、零长、FD、多 section、同形状不同有效长度；仅改变 QKV 数值时计划不变；所有 active/保留槽及第二行检查完整。
- 容量：目标核数与设备总核数不同，仍按目标分配；output 过小/过大、dtype/stride/device 不匹配时写入前失败；返回传入 output，现有 None/(0,)/固定槽/非零内容行为不变。
- 计算隔离：mock C++ 分支不调用原 metadata ACLNN、Python provider 或 helper 子进程；主 qfa 仍执行真实路径；缓存也执行版本/配置检查。
- 版本：helper 与扩展共用核心版本；只改原 AICPU cpp/decoder 会失配；核心变化会重建扩展，不加载旧 JIT 产物；未知格式、旧 bundle、fork/worker 身份变化有明确结果。
- golden：获取/解码/数值失败均不走全局参考；旧分段缓存不能掩盖入口；两端请求与 metadata 一致，分段数值确实成功。
- 设备：正确 stream/设备上的长度读取、H2D、同步和输出可消费；单列真实安装/工具链、SoC/有效核数/tiling、Out/LSE 及 AICPU 对拍状态。

完成真实 P2/P3 接入验证后才评估升级；推荐部署改为 extension_cpp 还需 P5 目标证据。当前 README 仍对应已实现的 assets_explicit，Python API 方案仅作未实施备选，不与 C++ 方案叠加。C++ 接入减少了 Python 生产入口适配，但增加目标扩展编译依赖，也没有自动消除上游包装行为的版本漂移。
