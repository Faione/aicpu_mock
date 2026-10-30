### 21.1 简化方案实施结果（2026-10-30）

主方案已改为一个 assets 补丁与共享 provider；51 项 CPU 测试通过。当前决策与实现见第 21 节，真实设备验收未完成。

- 已交付一个三文件 assets 补丁、显式适配模块和应用/撤销工具；原 Spec 直接执行，不依赖外置函数替换或异常桥接。
- build 直接编译，使用编译器实际依赖清单；无 .git、assets、AICPU cpp 的最小头文件目录也能构建并运行。旧 SOURCE_MISSING 来自完整源码检查，不是 Git 检查；此编译前检查已移除。
- worker 与请求规范化提取为独立小模块，简化路径不导入旧 integrations 或 runtime_loader。
- 51 项 CPU 测试通过，0 失败/错误/跳过。补丁路径与旧路径六组 Out/LSE 摘要逐项一致；动态/静态槽、native 选择、失败不回退、补丁字节级撤销已验证。
- 260 个原上游 Python 文件测试前后摘要不变，补丁只在临时副本应用。目标项目部署时显式执行 apply_assets_patch.py；真实 AICPU、H2D 和 AI Core 精度仍待目标环境。
- 更新后的使用流程见 ASSETS_USAGE.md，测试记录见 docs/verification_latest.json。第 21 节覆盖旧方案中的零源码修改和强制启动证据约定。


### 21.2 build 缺失检查修正（2026-10-30）

按用户最新要求，恢复默认源码完整性检查；本节覆盖 21.1 中“编译前检查已移除”的构建行为说明。

- build 仍不调用 Git，不要求 .git 或提交历史。默认检查编译头文件、assets/golden 和 AICPU 参考 cpp，共三组。
- 检查失败返回 SOURCE_MISSING，一次列出所有缺失项；error.repo 为源码根目录，error.missing 每项包含 group、relative_path、绝对 path 和 required_for。message 同时给出可读列表和源码根目录提示。
- 编译头文件用于生成 helper；assets/golden 用于后续 TTK 和 CPU 校验；AICPU cpp 作为参数包装行为参考，不参与 helper 编译。
- 显式 --compile-only 仅检查编译头文件，允许最小头文件目录构建；未列入清单的额外 include 缺失由编译器报告。实际编译指纹仍只覆盖真正引用的依赖。
- 检查或编译失败不覆盖已有 helper/清单。无 .git 的完整源码检查、缺失清单、CLI 结构化错误、最小目录编译均已验证。
- 当前 54 项 CPU 测试通过，0 失败/错误/跳过；260 个上游 Python 文件不变。TTK 流程及目标设备待验收事项保持原计划。


### 21.3 必需源码不可跳过（2026-10-30）

按用户最新纠正，移除 21.2 引入的 --compile-only。所有 build 入口必须检查编译头文件、assets/golden、AICPU 参考 cpp，缺少任一项就停止构建，不调用编译器，不修改已有 helper 或清单，并返回包含全部缺失项的 SOURCE_MISSING。此前仅有头文件目录可构建的说明已失效。

这三组文件是交付构建的强制前置条件；实际 C++ 编译依赖摘要仍由编译器生成。build 不调用 Git，也不要求 .git 或提交历史。TTK 使用流程保持不变。

验收覆盖完整源码无 Git 构建、带空格路径、逐类删除必需文件时禁止启动编译器、旧产物保留、CLI 拒绝跳过参数。当前 55 项 CPU 测试通过，0 失败/错误/跳过；260 个上游 Python 文件不变。真实设备验收仍未完成。
