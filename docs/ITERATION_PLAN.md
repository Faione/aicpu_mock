# 当前实施与后续验收计划（2026-10-31）

主设计第 26 节及 [`.run` 详细设计第 10 节](qfa_mock_run_package_design.md)是当前依据；第 25 节保留前序计划。特别是“原 assets 不改”“PyBind 包装原函数指针”“容量只需容纳实际计划”覆盖前序要求。新增进度见主设计第 27 节和 [PROGRESS.md](../PROGRESS.md)。

## 当前执行结构

```text
原 TTK/CSV/assets → 原 metadata Python API
                   → 新增配置/容量准备 → 原 output 分配
                   → PyBind 包装 callable
                       ├─ native → 捕获的原函数指针 → 原 ACLNN/AICPU
                       └─ mock → 共享 Host 核心 → 覆盖/容量校验 → 写原 output
原主 qfa 绑定与函数体 → 原 ACLNN/AI Core
```

helper 与扩展共同编译 metadata_core/metadata_adapter；helper 不解析 CSV。新模式不回调 Python provider，不启动 helper 子进程，不改 schema，不替换全局符号。行为依赖、适配能力与二进制身份分开登记，不因编译成功就自动授予跨版本兼容。

## 已开发与后续步骤

| 阶段 | 本轮代码/证据 | 接下来完成条件 |
|---|---|---|
| R0 | 环境探测、槽位/容量契约 | 在实际目标重复 probe，确认 SoC/有效核数和 Runtime；当前 macOS 无 CANN |
| R1 | 共享核心及纯 C++ 校验；默认与显式容量回归 | 已完成本机阶段，保持样本与 Python 独立参考一致 |
| R2 | 类型化 PyBind 包装、配置冻结/行容量查询、Python 准备与 OpBuilder 源声明 | 目标 Torch/CANN 编译原扩展，验证包函数、PrivateUse1、扩展直调与主 qfa；CPU dispatcher 证据不可替代 |
| R3 | vendor wheel 内容检查、单 .run 组合、严格安装器、环境激活与身份核查 | 用真实同源 CANN 包及安装清单，在干净目标 prefix 安装并预热；当前仅归档机制已测 |
| R4 | 原 golden 审计、数值 fallback 检测、旧槽位/缓存与接入撤销回归 | 新扩展 + 原 assets 的真设备链路；记录实际分段消费，不能只看 TTK PASS |
| R5 | 保留目标验收脚本与记录要求 | 先关 golden 核对 metadata/tiling，再开 golden 比较 Out/LSE，单列 AICPU 对拍 |

## 部署与证据约束

- 现有 assets_explicit 与 extension_cpp 互斥。候选在隔离 checkout/worker 应用，只有一个 metadata 生产入口；SDK 不可用时只完成 Host/CPU 阶段，不生成伪原生 `.run`。
- 安装包只接收一个匹配的实际原生包与实际 wheel；检查同源行为声明、vendor/架构、完整安装文件清单。缺依赖或 JIT 失败非零退出，保留 INSTALL_INCOMPLETE，不覆盖旧 prefix。
- 原 assets 的异常回退保留，GoldenAudit 独立拒绝分段未成功的结果，不宣称生产 golden 任意错误都向上传播。原 CSV 优先动态/零长槽；旧固定 shape 不符时明确失败。
- 当前 63 项 CPU 回归与真实 CPU libtorch 包装证据分开记录；PrivateUse1、CANN 安装、H2D、实际 AI Core、mk001 兼容性仍待目标环境。
- CSV 目标配置、CANN executor C0、稳定 backend ABI、弱 hook 等保持后置，不属于本轮已实现能力。

操作入口：[EXTENSION_USAGE.md](../EXTENSION_USAGE.md)。源设计和状态已同步入 docs，外部主设计保留完整历史与新增实际开发记录。
