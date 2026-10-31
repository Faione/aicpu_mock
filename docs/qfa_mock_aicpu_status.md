# QFA Mock AICPU：当前实现与验收状态

更新日期：2026-10-31  
实现目录：`/Users/fanghaolei/Workplace/agent/aicore_mock`  
实现基线：`eef4af53c868dc3eaa23b81f54b3510f0e323a95`；2026-10-30 独立复核时工作区干净。2026-10-31 仅更新计划文档，未重新执行测试或变更实现。

## 当前方案

主方案已由外置 Spec 改为 **一个三文件 assets 补丁＋共享 provider＋Host C++ helper**。

```text
原 assets Spec
  ├─ golden：真实形参 → 统一请求 → provider → CPU metadata
  │           → 原 decoder → 显式 metadata_plan → CPU Out、LSE
  └─ 预处理：真实形参 → 同一 provider → CPU metadata → 写入目标设备
              → 原槽位回填 → 原 qfa AI Core → 输出比较
```

新主路径不再临时替换模块函数，不需要 BaseException 异常桥接、外置 Spec 或专门启动器。旧方案保留兼容，不与补丁叠加。

## 已实现并核对

| 项目 | 实现与本次证据 |
|---|---|
| CPU 分核计算 | Host helper 复用原分块函数与 SectionStreamK；生成 Header、FA、FD、初始化标志及完整 metadata |
| 请求与缓存 | 两个入口从真实形参整理请求；共享 provider；CPU bytes 主副本不可变，独立交付 Tensor；缓存区分长度、目标与版本 |
| worker 生命周期 | 固定规范化配置、源码目录、helper 路径和构建身份；运行中改变语义配置失败；fork 后重置 |
| golden 接入 | 显式传入 `metadata_plan`，跳过设备计划获取；decoder 或数值异常直接抛出，不进入全局参考回退 |
| 预处理接入 | 本次调用选择 metadata_builder，沿用动态字典返回、零长 resize、静态形状检查与 copy |
| 模式切换 | 未设置或配置为空时保留 native；非空配置无效时失败；mock 拒绝其他量化模式 |
| 平台处理 | 显式目标配置驱动计划；已知设备核数容量不足时失败；旧离线平台证据工具保留可选诊断 |
| 补丁交付 | 一个补丁修改三个 assets 文件；支持检查、重复应用检测、撤销及原行尾保留；源目录不要求有 .git，应用工具需要 git 程序 |
| 构建检查 | build 不调用 Git；强制检查编译头文件、assets/golden 和 AICPU 参考 cpp；一次列出全部缺失路径与用途，无跳过选项；失败不覆盖旧产物 |
| 编译身份 | 使用编译器实际依赖清单和内容摘要，构建身份不绑定 Git 提交或整棵源码目录 |
| 插件发现 | 真实 TTK SpecLoader 能发现补丁后的原 QuantFlashAttnSpec；原 spec.py、输入生成与 compare 不改 |

补丁文件：[qfa_assets_mock.patch](/Users/fanghaolei/Workplace/agent/aicore_mock/patches/qfa_assets_mock.patch)。

| 修改文件 | 具体变化 |
|---|---|
| `assets/impl/golden.py` | 从实际参数构造请求，获取 CPU 分段计划并显式传入数值函数 |
| `assets/quant_flash_attn_golden.py` | 新增可选 metadata_plan 参数；显式计划模式不获取设备 metadata，异常不回退 |
| `assets/impl/npu_preprocess.py` | 根据配置选择本地 metadata 生成函数，保留原槽位处理 |

## 本次独立复跑

环境：Python 3.12.16、CPU Torch 2.9.0、NumPy 1.26.4。

```bash
cd /Users/fanghaolei/Workplace/agent/aicore_mock
PYTHONDONTWRITEBYTECODE=1 .venv-cpu/bin/python -B scripts/verify_cpu.py \
  --repo /Users/fanghaolei/Workplace/agent/ops-transformer \
  --ttk /Users/fanghaolei/Workplace/agent/ops-test-kit \
  --output /Users/fanghaolei/Workplace/agent/ops-transformer/scratch/qfa_mock_verification_20261030
```

结果：**55 项通过，0 失败、0 错误、0 跳过；260 个上游 Python 文件摘要未变。**

- 原输入生成、原 CPU golden 的输入适配、decoder、分段数值函数及输出整理真实执行；补丁路径没有替换计划生产函数。
- 六组用例覆盖基线、变长、padding、零长、FD、多 section；补丁路径与旧外置路径的 Out、LSE 摘要逐项一致。
- 原预处理用真实 CPU Tensor 验证动态槽、`(0,)` 占位、静态匹配和不匹配；这证明槽位行为，不代表设备写入已验证。
- 故障注入验证 decoder、数值计算及配置错误传播；mock 测试中真实 metadata 和设备调用设置为触发即失败。
- 使用真实 TTK SpecLoader 验证发现，不等于执行完整 `ttk e2e`。
- 无 Git、带空格路径、源码缺失报告、禁止跳过检查和补丁应用撤销等用例均通过。

记录：[verification_latest.json](/Users/fanghaolei/Workplace/agent/ops-transformer/scratch/qfa_mock_verification_20261030/verification_latest.json)、[cpu_tests_latest.txt](/Users/fanghaolei/Workplace/agent/ops-transformer/scratch/qfa_mock_verification_20261030/cpu_tests_latest.txt)。

## 当前部署状态

**补丁已交付并在临时副本验收；当前 ops-transformer 的原 assets 尚未正式应用。** 本次运行补丁工具 `--check` 返回 `Patch can be applied`，未改写原 assets 或 TTK。

部署时：在目标平台 build helper，显式应用补丁，设置 PYTHONPATH、QFA_REPO、QFA_METADATA_CONFIG、QFA_MOCK_HELPER；TTK 的 `--plugin` 指向原 assets。完整命令见 [ASSETS_USAGE.md](/Users/fanghaolei/Workplace/agent/aicore_mock/ASSETS_USAGE.md)。

首版范围仍为一次性输入、MXFP8 quant_mode=1、正向 Eager、D=Dv=128、Q/scale/Out=TND、KV=TND 或 PA_BNBD、默认窗口，最大长度显式为正。mask=0/3/5 的计划生成已有检查；本次六组真实 golden 数值验收采用 PA_BNBD、mask=3，不将其他组合记为已完成端到端验证。

## 尚未完成

| 项目 | 当前状态 |
|---|---|
| mk001 平台确认 | 已知设计核数 80/80；实际调度 SoC、编译架构、仿真注册项与有效核数仍待确认 |
| 完整 TTK e2e | 尚未运行目标环境完整命令 |
| 实际设备写入 | CPU Tensor 复制已验证；将 metadata 写入仿真设备尚未验证 |
| qfa 实际 tiling | 配置不会改变实际编译与 Runtime；模板和有效核数兼容性仍待目标验收 |
| AICPU 同配置对拍 | 未执行；CPU 结构覆盖和新旧路径数值相等不能替代真实对拍 |
| AI Core 仿真精度 | 未执行真实 qfa；Out、LSE 与 CPU golden 的设备精度比较待完成 |

YAML、持久化缓存、历史 KV cache 与图模式未纳入本轮实现。

## 对上轮评审的处理

- 模块来源、worker 固定配置和三处替换恢复：旧兼容路径已补齐并继续回归；新路径直接调用原函数，无需替换恢复。
- 第三处 golden 异常保护：新路径通过显式计划参数直接传播异常，已消除临时包装需求。
- 平台证据：新主路径不再强制离线记录和启动器；完整目标兼容性保留为设备验收项，不能标记自动确认。
- 原 hook、CPU golden 与 SpecLoader：已从待验收升级为本次实际通过；AICPU 与设备项目仍保留待验收。


## 后续入口收敛设计

预处理可只替换 run_metadata 的生产职责，原动态、空占位和静态回填逻辑保持不变；当前 metadata_builder 已经实现这一结构。完整 e2e 仍保留 golden 的独立 CPU 计划入口。若把分派移入 run_metadata，必须保留真实参数整理与支持范围校验，不能只依据原 arguments 字典猜测缺失字段。

该项是设计建议，不是新增实现；详见[主设计第 23 节](/Users/fanghaolei/Workplace/agent/ops-transformer/scratch/qfa_mock_aicpu_design.md)。

另已补充算子接口统一 mock 的候选设计：共同 Python 实现覆盖包导出函数与 torch.ops 两种路由；仅覆盖 dispatcher 不足。golden 的错误回退、缓存和设备传输仍需处理。详见主设计第 24 节，尚未实现。

统一接口 mock 的环境评估已补入主设计第 24.5 节：能替代已确认的 metadata AICPU 调度计算，但仍需仿真 Runtime、严格 golden 路径和实际 tiling/Out/LSE 验收。共同接口尚未实现，既有 55 项测试不自动作为其验收。


## 下一轮实施计划（2026-10-31，仅文档）

主设计第 25 节已按用户最新决策调整为 extension_cpp：在 Torch 加载的真实 C++ 扩展 metadata 函数内选择 native/mock，保留主 qfa；helper 与扩展共同编译同一 Host 核心，扩展不启动 helper 子进程、不回调 Python provider。Python 仅作配置、容量和调用侧保护；第 23—24 节的共同 Python 生产入口降为备选。

| 阶段 | 当前状态 |
|---|---|
| P0 设计/依赖与进度更新 | 文档已同步，目标环境实测未做 |
| P1 共享 Host 核心与版本/覆盖校验 | 待实现，可在普通 CPU 环境推进 |
| P2 真 C++ 扩展分派、容量适配、构建清单 | 待实现，需要目标 Torch/CANN 工具链 |
| P3 严格 golden、路由/槽位/生命周期、补丁互斥 | 待实现 |
| P4 CSV 目标绑定 | 待实现，当前仍需目标 JSON |
| P5 Runtime、实际 tiling 与 Out/LSE | 待目标环境，AICPU 对拍另记 |

已交付的三文件 assets_explicit 仍是当前部署基线；与 extension_cpp 及 Python metadata_api 备选互斥。2026-10-30 的 55 项 CPU 记录不覆盖新扩展，不能证明新的二进制已编译、加载或设备已验收。本轮没有改动运行代码、补丁或生产源码，也未重跑测试。

新方案的 Python 输出必须先按目标核数分配；C++ 写入前检查精确 shape/dtype/device/连续性，返回同一个 output。native 保留原 metadata ACLNN；mock 计算失败不回退。TTK 不改，--plugin 仍是原 assets，主 qfa 仍执行真实 AI Core。

计划副本：[ITERATION_PLAN.md](/Users/fanghaolei/Workplace/agent/aicore_mock/docs/ITERATION_PLAN.md)。当前 README 部署命令不变，未来扩展需按目标项目已有构建链重新编译/安装。
