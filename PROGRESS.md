# QFA Mock AICPU 迭代进度

更新日期：2026-10-29。对齐最新主设计第 17–19 节与评审文档；实现基线包含 `7072d5d` 后的全部接入改进。

**本机可完成的实现与 CPU 接入验收已完成。41 项测试通过，0 失败、0 错误、0 跳过；260 个上游 Python 文件测试前后摘要一致。真实 AICPU 对拍、H2D 和 AI Core 仿真精度仍待目标环境。**

## 本轮完成

| 事项 | 实现与证据 |
|---|---|
| 模块加载与来源固定 | 原 Spec 用解析后绝对路径摘要生成别名，重复发现复用；校验已缓存 impl、数值模块、支持包路径和源码摘要；第二 checkout 或签名不兼容立即失败 |
| worker 生命周期 | 固定 PID、源仓库、规范化配置、helper 路径和构建身份；语义变化报错；JSON 排版/字段顺序变化及相同配置迁移可复用；fork 重置锁与 provider 状态 |
| 三处受控替换 | 两个计划接管点与第三处分段数值异常保护均按最新决策保留；共享 RLock，成功/异常恢复原对象 |
| golden 严格成功条件 | 要求一次计划消费、非空解码、一次分段函数成功返回；异常桥接在边界转普通 MockError，保留原因链；KeyboardInterrupt/SystemExit 不转换 |
| 平台预检 | 离线模型、平台、内核、构建与映射记录绑定实际 TTK argv/环境；核对摘要、NpuArch、有效核数和调度规则；支持真实 tiling 复核与 FD 容量检查；无证据不执行设备路径 |
| 查询兼容 | SoC 类型化别名及来源记录；区分物理总核数与有效核数；查询不可用可使用离线证据，明确冲突则失败；不提供跳过校验开关 |
| 原 hook 槽位矩阵 | 使用真实 CPU Tensor 执行原 run：动态返回、(0,) resize/copy、静态匹配及不匹配错误/恢复；只绕过设备平台证据边界 |
| 原 CPU 数值链路 | 实际执行原输入生成、输入适配、decoder、分段 Attention/LSE 与输出整理；覆盖基线、变长、padding、零长、FD、多 section，保留原 FP32 Out/LSE |
| TTK 插件发现 | 使用实际 TTK SpecLoader 发现外置类，验证输入/compare/容差继承；原 native Spec 仍独立可发现 |
| 诊断与复现 | 结构化源模块路径/摘要、worker、阶段、请求/计划摘要、恢复结果；验证脚本保存测试记录、CPU Out/LSE 摘要和上游完整性摘要 |

配置继续为 JSON，缓存仅进程内。一次性输入范围不变：无历史 KV cache 追加、跨轮复用或请求加入。D=128、正向 Eager 等已支持组合见 README/CONTRACT；YAML、持久化缓存和新布局/图模式按设计后置。

## 验收阶段

| 阶段 | 状态 | 范围与限制 |
|---|---|---|
| A：加载与失败恢复 | 完成 | 路径冲突、复用、配置/构建变化、helper/decoder/数值/外层输出错误、三处恢复和中断传播均有测试 |
| B：原 hook | CPU 完成 | 真正运行原 run 和真实 Tensor；不是 H2D 验收；平台检查另外用带文件摘要的夹具验证 |
| C：原 CPU golden | 完成 | 6 组真实数值用例；测试中的 torch_npu 仅为导入边界，设备/metadata 调用触发即失败，没有替代数值运算 |
| TTK discovery | 完成 | 实际原 SpecLoader、外置注册及继承；不等于完整 TTK 设备 e2e |
| D：同配置真实 AICPU 对拍 | 未完成 | 当前没有可执行 AICPU 的目标设备证据 |
| E：目标 AI Core 仿真 | 未完成 | 未验证真实 H2D、tiling 和 Out/LSE 精度；mk001 的 SoC/架构/仿真注册仍需部署方确认 |

普通设备属性不足以独立证明构建和有效核数；首版默认使用已绑定的离线记录。尚无经真实 SDK 验收的纯运行时证明适配器，不影响离线预检路径的使用。离线记录必须来自真实构建与安装，夹具测试不能作为 mk001 平台登记。

## 可复核证据

- [verification_latest.json](docs/verification_latest.json)：Python 3.12.16、Torch 2.9.0、NumPy 1.26.4；41 项通过、无跳过，260 个上游文件摘要不变。
- [cpu_tests_latest.txt](docs/cpu_tests_latest.txt)：完整测试结果。
- 机器记录含动态/静态槽位结论、实际 TTK discovery 结论和 6 组 CPU Out/LSE 的 shape、dtype、数值摘要及请求/计划摘要。
- arch92 80/80 固定基线 SHA-256 回归通过；本机还与旧二进制逐字节比较。
- 平台夹具覆盖别名整数/字符串、查询不可用、容量不足、有效核数/架构/源码冲突、文件变更、启动绑定、实际 tiling 差异及证据打包 CLI。

## 目标项目的剩余工作

1. 确认真实调度 SoC、编译架构、有效核数、CANN 平台和仿真器/模型映射。
2. 用实际安装与 qfa 构建生成可信记录，在目标机器编译 helper、获取 metadata，再通过 launch 拉起 TTK；流程见 [ASSETS_USAGE.md](ASSETS_USAGE.md) 和 [PLATFORM_EVIDENCE.md](PLATFORM_EVIDENCE.md)。
3. 保存首次真实 tiling 复核、H2D、Out/LSE 比较结果；独立记录同配置 AICPU 对拍，不用 CPU 测试替代设备验收。
