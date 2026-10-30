# 简化接入计划（2026-10-30）

本轮按用户授权，将“原 assets 零修改”改为“允许维护一个 assets 补丁文件”。以下为新的主方案，覆盖旧方案中外置 Spec 和平台启动器为必选项的约定。

## 决策

1. 保留 Host C++ helper、统一请求、共享 provider、metadata 协议和 CPU 校验，不重写分核算法。
2. 新增一个统一补丁 `patches/qfa_assets_mock.patch`，只修改三个测试资产文件：impl/golden.py、quant_flash_attn_golden.py、impl/npu_preprocess.py。TTK、Torch 注册、生产 Host/AI Core 均不改。
3. impl/golden.py 从本次真实形参整理请求，取得并解码 CPU metadata，将分段计划作为可选显式参数传给原数值函数；数值函数在显式计划模式下直接抛出异常，不进入原设备计划获取或全局参考回退。原 Attention、LSE 和输出整理保持原样。
4. 预处理在 mock 分支从本次真实形参整理请求，选择一个本地 metadata 生成函数，沿用原 run 的动态/静态槽位处理；不改模块函数、不依赖 golden 先执行。
5. `QFA_METADATA_CONFIG` 非空即启用补丁中的 mock 分支；缺省保持原 native。给定配置但文件无效、请求不支持或 helper 失败则报错。两个分支延迟导入 qfa_mock，不初始化 NPU。
6. TTK 使用原 assets 的 --plugin 路径。无需外置 Spec、运行时函数替换、异常桥接、原类加载器或启动回执。旧外置方案保留兼容，不与新补丁叠加。
7. 主方案目标平台由显式配置提供；可用设备查询仅用于已知的容量/明确 SoC 冲突检查，不能自动证明有效核数或编译架构。首次目标仿真仍核对真实 tiling / Out / LSE。旧离线平台证据工具降为可选诊断，不再是简化路径的启动前置条件，不宣称目标已验收。
8. 按用户最新纠正，build 不调用 Git，但默认检查编译头文件、assets/golden 和 AICPU 参考 cpp，必须一次明确列出全部缺失文件、绝对路径、分组及用途。仅编译时用 --compile-only 跳过非编译文件完整性检查。实际编译清单和运行时指纹仍只包含真正编译的依赖，不绑定 Git 或整棵仓库。

## 实施与验收

- 先同步本计划到主设计，再实现 assets 适配、统一补丁及应用工具。
- 在无 .git 的完整源码副本验证默认检查通过；最小头文件目录的默认检查应列出 assets/AICPU 缺失，用 --compile-only 验证编译成功；检查或编译失败均不覆盖已有可执行文件。
- 在临时复制的上游 assets 应用和撤销同一个补丁；直接执行补丁后的原函数，禁止依赖 monkeypatch 计划生产函数。
- mock 下验证双入口请求/计划相同、数值基线/变长/padding/零长/FD/多 section、动态/静态槽，以及 decoder/数值/helper 错误不回退。
- 无配置时验证原计划入口和原预处理仍被选择；真实 metadata 在 mock 测试中触发即失败。
- 更新操作说明、进度和 CPU 证据，提交并推送远端。补丁交付后由目标项目显式应用；本地原上游 assets 不自动改写。
- 真实 AICPU、H2D 和 AI Core 精度继续单列为目标环境验收，不能用 CPU 结果替代。
