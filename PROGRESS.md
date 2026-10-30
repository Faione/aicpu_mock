# QFA Mock AICPU 进度（2026-10-30）

最新决策见主设计第 21 节及 [SIMPLIFICATION_PLAN.md](docs/SIMPLIFICATION_PLAN.md)。主方案已改为“一个 assets 补丁 + 共享 provider + Host helper”，旧零源码修改方案保留兼容，不再是使用前置条件。

## 本轮完成

- build 不调用 Git；按用户最新要求，强制检查编译头文件、assets/golden、AICPU 参考 cpp 的完整性，逐项列出所有缺失路径及用途，缺少任一项即失败，不提供跳过选项。实际编译指纹仍只包含真正引用的头文件和 helper。
- 已在无 .git、路径含空格的完整源码副本编译并运行 helper，基线摘要正确；逐类删除头文件、assets/golden 或 AICPU cpp，均在调用编译器前失败并报告缺失项，旧产物保持不变。
- 交付唯一的 patches/qfa_assets_mock.patch，修改三个测试资产。应用工具支持检查、重复应用、撤销，保留 LF/CRLF；无 Git 仓库的目录也可应用（需 git 可执行程序）。
- impl/golden.py 传递显式分段计划，数值函数跳过设备 metadata 获取并直接传播异常；没有临时替换函数或异常桥接。
- 预处理选择本地 metadata_builder，沿用原动态槽返回、零长占位、静态检查和 copy。配置缺省走 native，给定无效配置失败；其他量化模式提前拒绝。
- 原 Spec 直接运行，主路径不导入旧 integrations/runtime_loader；worker 和请求整理提取为小型共享模块。
- 平台离线记录/启动器降为旧兼容路径的工具，新主路径无需生成这些文件。目标由配置显式声明，已知容量冲突会失败，真实 tiling / Out / LSE 仍在目标环境验收。

## 验证

**55 项测试通过，0 失败、0 错误、0 跳过；260 个上游 Python 文件摘要不变。补丁与旧路径六组 Out/LSE 摘要逐项完全一致。**

- 补丁在临时复制的上游 assets 应用，不改本地原 assets 或 TTK。
- 直接调用补丁后的原 Spec，执行基线、变长、padding、零长、FD、多 section 六组数值；mock 计划函数没有被测试替换。
- 验证动态 metadata、(0,)、静态匹配/不匹配；双入口请求和计划一致；decoder/数值错误不调用原生 metadata 或全局参考。
- 无配置时原计划入口及原预处理仍被选择。真实 TTK SpecLoader 发现原 Spec。
- 保留旧外置方案与平台证据的兼容回归。完整测试结果与补丁/旧路径 Out/LSE 的逐项摘要比较见 [verification_latest.json](docs/verification_latest.json)。

## 使用与部署

更新代码后重新 build helper，再执行 apply_assets_patch.py。设置 PYTHONPATH、QFA_METADATA_CONFIG、QFA_MOCK_HELPER，TTK 的 --plugin 指向目标项目原 assets。完整命令见 [ASSETS_USAGE.md](ASSETS_USAGE.md)。

本地上游 assets 尚未自动打补丁，目标部署需要显式应用。源码不匹配时工具会失败，不应强行忽略。旧外置 ttk_plugin.py 不与新补丁叠加。

## 尚待目标环境

mk001 对应真实 SoC/架构/有效核数、实际 tiling、AICPU 对拍、H2D、AI Core 仿真 Out/LSE 精度仍需在目标机器完成。CPU 验证不能替代这些结论。首版仍是一次性输入、MXFP8 正向 Eager、D=128；未扩展历史 KV cache、YAML、持久化缓存或图模式。
