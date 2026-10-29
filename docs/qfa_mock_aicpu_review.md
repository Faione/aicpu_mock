# QFA Mock AICPU：未解决事项评估与下一轮方案

日期：2026-10-29  
主设计：[qfa_mock_aicpu_design.md](/Users/fanghaolei/Workplace/agent/ops-transformer/scratch/qfa_mock_aicpu_design.md)  
实现目录：`/Users/fanghaolei/Workplace/agent/aicore_mock`

## 1. 评估结论

外置 Spec、共享 provider、Host helper 的方向可以继续。当前主要缺口在接入可靠性和实际链路验收，无需重做调度算法，也不必为满足原文“两处替换”的数量限制而删除已有异常保护。

本轮建议收敛为：**两个计划接管点，加一个 golden 异常保护点；固定 worker 的模块与目标身份；分开验证计划、原钩子、数值计算和设备执行。** 原 assets、TTK 与生产算子继续保持零修改。

评估依据包括当前 PROGRESS、ASSETS_USAGE、CONTRACT，`ttk_plugin.py`、`qfa_mock` 实现、现有测试，以及上游 golden、预处理和 TTK 仿真配置代码。实现工作区含 `7072d5d` 之后的未提交修改，结论针对读取时的文件内容。

| 事项 | 评估 | 下一轮处理 |
|---|---|---|
| Host helper、协议与 CPU 计划校验 | 已有可复用实现，未发现需要整体重写的证据 | 继续复用；真实 AICPU 对拍补充独立证据 |
| 第三处 golden 包装 | 原设计约束过严；包装有明确必要性 | 保留并限定用途，补严格失败与恢复检查 |
| 模块加载 | 固定别名且未校验来源，可能混用 checkout | 作为设备接入前的必修项 |
| worker 配置 | 当前变更时重建 provider，与已加载原模块可能不一致 | 固定身份，拒绝运行中切换 |
| 运行时平台检查 | 当前查询方式和直接字符串比较不足以覆盖仿真 | 引入明确的平台证据与规范化比较 |
| 钩子和数值验收 | 替身测试不能验证原函数体行为 | 先执行原 hook 和 CPU golden，再进入设备精度验证 |
| YAML、持久化缓存 | 对当前接入没有必要 | 后置，首版沿用 JSON 和进程内缓存 |

## 2. golden 异常保护：保留第三处包装

### 问题依据

原 [cpu_mxfp8_golden](/Users/fanghaolei/Workplace/agent/ops-transformer/attention/quant_flash_attn/tests/assets/quant_flash_attn_golden.py:1431) 先获取分段，再用 `try/except Exception` 调用 `_cpu_mxfp8_golden_sectioned`。分段计算失败后，它会继续计算全局参考结果。

因此，只替换 `get_qfa_section_info` 可以阻止设备计划获取，但不能阻止之后的数值异常被吞掉。现有“计划回调被调用过”的标记，也不能证明分段 golden 成功。

### 推荐行为

保留现有第三处包装，明确三处职责：

| 局部替换 | 职责 |
|---|---|
| `get_qfa_section_info` | 使用 mock 计划并调用原 decoder |
| `run_metadata` | 将 mock 计划写入目标设备 |
| `_cpu_mxfp8_golden_sectioned` | 原样调用数值函数，仅阻止异常触发全局回退 |

异常保护仅捕获数值函数的普通 `Exception`，转换成内部专用异常以跨过上游捕获；在外置 `run_golden` 边界立即转换为普通 `MockError`，保留原始异常链。内部异常可以继续继承 `BaseException`，但不能逃到 TTK，也不能捕获或转换 `KeyboardInterrupt`、`SystemExit`。

在共享锁内维护本次调用的状态：计划已消费、解码结果非空、分段函数已成功返回。只有满足这些条件才允许返回 golden；返回结构和 LSE 仍沿用原实现。当前串行 MXFP8 路径可要求一次计划消费、一次分段计算，上游调用结构变化时应重新核对。

成功、decoder 失败、分段计算失败和外层输出处理失败，都必须恢复进入前的原函数对象。测试中同时断言全局参考分支没有执行。

不推荐通过替换 logger、修改函数 AST 或捕获日志文本来禁用 fallback；也不推荐等全局参考计算完再拒绝结果。它们扩大了对上游细节的依赖，或执行了本应禁止的回退。

## 3. 模块与 worker 身份：固定来源，拒绝中途切换

当前 `ttk_plugin.py` 每次使用固定别名加载原 Spec；原 assets 又使用固定的 `qfa_assets_impl_*` 名称。另一 checkout 已经加载时，仅修改原 Spec 别名仍不足以避免实现模块串用。

建议在现有文件中补加载职责，不要求新增指定文件名：

1. 注册时解析并固定原仓库绝对路径；原 Spec 别名包含该路径的稳定摘要，同一路径重复发现时复用已验证模块。
2. 加载或复用原 Spec、impl、数值模块和 `qfa_mock` 支持包时，检查实际 `__file__`。路径经 `resolve()` 后比较；不匹配就报错，不覆盖已有模块后继续执行。
3. 首版一个解释器只服务一个原 checkout；遇到第二个 checkout 使用独立进程。无需改造原固定模块名体系以支持同进程多仓库。
4. 首次 worker 调用固定：PID、原仓库路径、规范化配置、helper 路径和构建身份。后续语义变化报 `WORKER_CONFIG_CHANGED`，不清缓存重建 provider 后继续。
5. JSON 格式、字段顺序变化不算配置变化；内容相同的配置文件迁移可记录路径变化，不改变目标身份。源代码或 helper 被替换则沿用构建指纹检查报错，要求重新启动 worker。
6. 子进程按新 PID 创建自己的状态；注册阶段不建立 provider 或安装函数替换，避免将父进程的运行中状态带入 worker。

兼容检查至少覆盖两个真实入口的参数绑定、计划函数和 decoder 是否存在，以及异常保护点的调用约定。原数值函数可能引用模块全局变量，不能仅凭相同函数名确认替换对象正确。

## 4. 目标校验：区分标识与真实执行参数

### 当前不足

当前 [metadata_for_preprocess](/Users/fanghaolei/Workplace/agent/aicore_mock/qfa_mock/integrations.py:131) 强制要求设备属性查询，并在 `get_soc_version` 存在时，将其结果转成字符串与配置的 `soc_version` 直接比较。这段逻辑尚无实际 SDK 和仿真证据。

TTK 自身已经区分仿真器 SoC、CANN 平台配置名和 camodel 模型名；同一目标的这些名称可以不同。查询 API 的返回类型与命名也必须在部署环境核实，不能假定其返回 metadata 使用的 SoC 长名称。现有测试没有覆盖 SoC 查询分支。

另外，原 metadata 的 Host API 优先查询**当前线程有效核数**，失败才回退平台核数。设备属性中的总核数不能单独证明本次 qfa 使用的有效核数。

### 推荐校验流程

| 信息 | 用途与来源 |
|---|---|
| 目标名称 mk001 | 用户标识；当前已知设计核数 80/80 |
| 调度 SoC 与架构 | 决定 mock 的分块、FD 等策略；与 qfa 实际构建匹配 |
| 仿真器标识、模型及 CANN 平台 | 由本次启动参数和安装的仿真配置确认映射 |
| 有效 AIC、AIV | 与本次 qfa tiling 使用的核数一致；区分物理总量与运行限制 |
| 模板信息 | 核对 M、S2、核配比、FD 分支及工作空间约束 |

处理顺序：

1. 用户配置继续作为生成 mock 计划的输入。当前 SoC 白名单保持显式，不因名称是 mk001 或核数为 80/80而自动选择 arch92。
2. 查询能力存在且语义明确时，将结果规范化后比较。别名映射来自当前 CANN、仿真安装或明确的项目配置，并记录来源；不能用模糊字符串匹配掩盖冲突。
3. 查询不可用时，允许使用与本次运行绑定的离线平台记录。记录至少关联仿真模型和版本、平台配置及摘要、实际 qfa 构建产物与架构、有效核数，以及分块规则或已提取的 tiling 信息。
4. 离线记录必须与实际启动路径和构建产物匹配，不能只是再复制一份用户目标配置。核数限制未知时，首版禁止启用额外控核。
5. 两类证据冲突时停止执行；均无法证明兼容性时也停止设备执行，但仍允许 CPU 独立生成与结构检查。
6. 已获得实际 qfa tiling 时，优先直接核对 metadata 与 tiling。无法事前获取完整 tiling 时，先按匹配的构建和平台记录做预检，首次仿真保留 tiling 产物复核，通过后再登记该平台组合。

不增加笼统的 `skip_target_check` 开关。配置能决定 CPU mock 如何计算，不能改变 qfa 实际使用的平台与模板。

mk001 对应的调度 SoC、编译架构及 CANN 仿真注册项仍是目标部署的待确认输入。它们不阻塞加载器、CPU 钩子和异常恢复的开发。

## 5. 验收调整：先验证真实钩子，再验证设备

当前 18 项测试验证了 helper、协议、缓存与部分替身控制流。`make_call` 使用简化签名，预处理部分测试替换了真正负责设备适配的函数，因此不能据此认定原槽位流程和平台检查已经通过。

| 阶段 | 环境与具体检查 | 退出条件 |
|---|---|---|
| A：加载与失败恢复 | 路径冲突、重复发现、worker 配置变化；helper、decoder、分段计算异常；三处恢复 | 来源固定，错误向上抛出，无原生 metadata 或全局参考回退 |
| B：原 hook | 有 CPU Torch；加载原 `npu_preprocess.run`，只替换计划生产依赖；覆盖 None、`(0,)`、静态匹配和不匹配 | 字典返回、resize、copy、错误和恢复均符合原约定 |
| C：原数值链路 | 兼容的 CPU Torch 与原 golden；执行原输入适配、decoder、分段函数和输出整理 | Out、LSE 的结构、dtype 和用例上下文正确，数值函数成功；不调用设备 metadata |
| D：同配置对拍 | 可运行 AICPU 的设备，使用该设备的真实目标与相同请求 | 有效 Header、FA、FD 和分段语义匹配；忽略未初始化区差异 |
| E：目标仿真 | 实际 TTK discovery、metadata 写入、原 qfa AI Core 和输出比较 | 两个钩子计划摘要一致；Out、LSE 通过容差；无 qfa_metadata 提交 |

阶段 B 可使用 CPU Tensor 执行原槽位分支，平台证据与写入能力另设适配边界；该结果只记为 CPU 槽位验收。测试环境中真实 `torch.ops...quant_flash_attn_metadata` 必须设置为触发即失败，覆盖 golden 和预处理两条路径。

阶段 C 验证控制流和真实数值实现可用；最终精度以阶段 E 对比为准。覆盖至少包括基线、变长、多 section、FD、padding、零长；新增 mask 和布局的“计划生成支持”与“端到端支持”分开登记。

真实对拍设备与目标仿真可以是不同平台，但必须各自使用匹配配置。其他平台对拍只能增加移植正确性的证据，不能记为 mk001 对拍已通过。无目标硬件时，单列“目标对拍未完成”，保留实际仿真精度记录。

无需等待真实对拍设备才开始 TTK discovery 和目标仿真排障；各阶段独立保存结论，不将阶段 E 通过自动替代阶段 D。

## 6. 实施顺序与交付

1. **先收敛接入。** 保留第三处保护，补成功标志与恢复；固定模块来源和 worker 配置。补原 hook 的槽位矩阵、SoC 分支及平台证据测试。
2. **再跑真实 Python 链路。** 在有兼容 Torch 的环境完成原数值链路和 TTK 插件发现，开启及关闭 golden 各验证一次。目标平台确认并行推进。
3. **完成设备闭环。** 同配置真实对拍与目标仿真分别留存证据；先跑现有 LSE 基线，再扩展用例。

每次验收保留：Spec 和原模块路径、源码摘要、配置和 helper 摘要、平台证据、请求及计划摘要、失败阶段、替换恢复结果、Out 与 LSE 比较结论。相同请求在两个钩子中的摘要要可关联。

首版不增加 YAML、持久化缓存、通用 AICPU 劫持或新图模式。可选优化是让预处理本次调用复用已取得的 MetadataResult，避免再次查 provider 和重复校验；不以此阻塞正确性验收。

实现 agent 应同步独立仓库的 PROGRESS、ASSETS_USAGE 和主设计副本，明确“第三处保护允许保留”与新的平台校验策略。本轮只更新本仓库设计文档，没有修改独立仓库的代码或状态文件。

## 7. 本次实际检查记录

- 复跑 `python3 -B -m unittest discover -s tests -v`：18 项通过，0 失败。
- 从原 `cpu_qfa_mxfp8` 和 `npu_preprocess.run` 提取真实签名，绑定 `scratch/qfa-metadata-mock/qfa_mxfp8_lse.csv` 的基线参数：两个适配器生成的规范化请求一致。
- 签名检查通过 AST 提取完成，未执行原函数体，也没有执行 Torch 数值计算。
- 未运行真实 TTK discovery、原 CPU golden 数值链路、AICPU 对拍、设备写入或 AI Core 仿真；本文方案优化尚待实现。
