# QFA Mock AICPU 进度（2026-10-31）

已同步主设计第 26 节及 [`.run` 详细设计](docs/qfa_mock_run_package_design.md)，按第 10 节实现“独立文件 + PyBind 绑定包装 + 分配前准备”。旧第 25 节的精确行容量与必须修改 assets 的要求由第 26 节覆盖。当前用户流程见 [EXTENSION_USAGE.md](EXTENSION_USAGE.md)，旧 assets_explicit 命令仍保留于 README；两种接入互斥。

## 本轮完成

- 抽出 cpp/metadata_core.h/.cpp 和 metadata_adapter.h。helper 缩减为协议与文件 I/O；架构、分块和调度适配集中管理，helper 与扩展共同编译同一核心。
- 核心支持显式 rowSize，按真实 section/核数校验容量，接受更大容量或足够的较小容量，剩余与第二行清零。补齐纯 C++ Header、FA/FD 覆盖、初始化、保留槽及工作区校验。
- 新增真实类型化 MakeMetadataBinding 包装器，捕获原 native 指针；不修改原 native metadata、主 qfa 函数体与主绑定。mock 不调用原 metadata ACLNN、Python provider 或 helper 子进程，返回原 output。
- Python 新模块负责配置/目标绑定、行容量、payload/扩展二进制身份与主 qfa 范围校验；不生产 metadata。扩展初版不缓存计划，worker 模式/目标固定，fork 后重新初始化。
- 接入工具按 AST/唯一绑定和已知签名小范围修改，只有三个原文件变化：Torch C++ 绑定、Torch Python 构建/容量入口、setup 文件收集；原 assets 和 TTK 不改。支持 --check、幂等、逐字节撤销、编辑保护和旧补丁互斥。
- build 保持不依赖 Git，必需文件强制全检；build_format=3 将编译依赖与 behavior_files/behavior_digest 分开。未知上游组合拒绝自动兼容；缓存、worker 和独立 bundle 均核对行为身份。
- 已构建真实 vendor wheel 并校验嵌套核心/头文件/运行时/许可证/entry point/清单，同时嵌入原基础算子发现框架，避免 vendor wheel 缺少基础 COT 加载器。安装后不依赖开发仓库绝对路径。
- 新增单 `.run` 组合与严格安装器：需要真实 CANN custom .run、匹配 wheel、同源声明和完整安装文件摘要；校验 vendor/架构/来源，逐项检查原生安装、独立进程检查 provider 并预热扩展，失败不报告成功。归档测试使用明确的 trap fixture，不将其作为原生包交付。
- 新增 GoldenAudit 独立执行审计，不修改原 golden 或其 fallback；真实原函数回归证明“合法 metadata 但分段数值抛错后返回全局结果”可被审计拒绝。

## 本机验证

**63 项测试通过，0 失败、0 错误、0 跳过；260 个上游 Python 文件测试前后摘要不变。** 旧补丁与旧外置路径六组 Out/LSE 摘要仍逐项一致。详见 [verification_latest.json](docs/verification_latest.json)。

另外用真实 CPU libtorch 编译同一个 C++ 包装器/核心，验证 PyBind 与 CPU dispatcher、native 指针/异常传播、模式切换拒绝、原 output 返回、较大容量、写入前错误拒绝；基线、FD、多 section 与独立 helper 逐字节一致。记录见 [cpp_bridge_verification.json](docs/cpp_bridge_verification.json)。该构建有显式 CPU acceptance 标记，生产运行时会拒绝将其当作 CANN 扩展。

新接入与 wheel/安装归档仅在临时源码副本测试；没有给本地 ops-transformer 打生产补丁，也没有安装或替换原算子包。硬件探测见 [target_probe.json](docs/target_probe.json)：本机 Darwin/arm64，缺 torch_npu/CANN。

## R0—R5 状态

| 阶段 | 已完成 | 尚未完成 |
|---|---|---|
| R0 环境/契约 | 新设计同步、只读探测、槽位与容量规则 | 目标架构/有效核数、真实安装与 Runtime 探测 |
| R1 共享核心 | 实现及 CPU/容量/损坏检测回归 | mk001 目标平台仍未确认 |
| R2 同名包装 | 原函数保留、绑定/配置/容量/构建实现；真实 CPU libtorch 验证 | 目标 CANN 扩展、PrivateUse1 与真实主 qfa 路由验收 |
| R3 单 .run | 真实 vendor wheel、载荷/组合工具、严格安装代码及归档检查 | 实际 CANN .run 构建输入、干净 Linux 安装/JIT与真实文件检查 |
| R4 原 assets 回归 | 原 CPU golden/fallback 审计、旧槽位、互斥/来源与打包回归 | 新 CANN 扩展配合原 assets 的完整真实设备链路 |
| R5 仿真精度 | 无目标运行结论 | H2D/stream、实际 tiling、AI Core Out/LSE；AICPU 对拍另记 |

CSV 目标嵌入、弱 hook/稳定插件、CANN Host-only executor C0、图模式、反向与历史 KV cache 均未实现，本轮没有以这些后续能力作为完成项。
