# 设备平台证据与启动约定

> 兼容方案说明：2026-10-30 起，主方案改为一个 assets 补丁，见 ASSETS_USAGE.md。以下严格启动器/离线记录保留给未应用补丁的旧外置 Spec，或主动启用的诊断路径；它们不再是简化路径的前置条件。设置 QFA_PLATFORM_EVIDENCE 时仍执行本文的严格校验，切换简化流程前可清除旧变量。

CPU `generate` / `validate` 不需要设备证据。进入设备预处理前，必须证明配置与本次仿真模型、平台及 qfa 构建一致。首版提供**离线记录 + 实际启动绑定 + 可用查询冲突检查**，不将普通设备属性当作线程有效核数证明，也不提供 `skip_target_check`。

## 记录来源

部署方先从实际安装、构建和提取工具准备两份可信记录；模板中的占位符必须替换，不能把 target.json 原样复制后充当证据。

- `examples/platform_build.template.json`：内核二进制 SHA-256、真实调度 SoC、编译架构、NpuArch、有效 AIC/AIV、无额外控核的约定，以及匹配上游源码的 helper `source_digest`。此记录应由实际构建/部署过程确认。
- `examples/platform_mapping.template.json`：仿真器 `--sim-soc`、安装的 model_name、CANN 平台 ini 名和调度 SoC 的明确映射；附安装 SOC_INFO 或明确项目映射源文件的绝对路径和摘要。可选 `runtime_soc_aliases` 按类型精确映射 SDK 返回值，数字与字符串不混同，不做模糊匹配。

`source_digest` 是与实际 qfa 源码对应的 helper 构建清单指纹。它包括调度头文件、相关包装与 decoder，不能使用另一个 checkout 生成的指纹。当前工具校验已有记录与文件的一致性，不能替代构建系统对记录内容的可信生成，也不能自动证明设备已加载该构建；最终仍以真实 tiling 和仿真验收闭环。

## 生成运行证据

在已配置 CANN 的目标环境中，设置 `QFA_REPO`、`QFA_METADATA_CONFIG`、`QFA_MOCK_HELPER`，并切换到本仓库。以下变量均为目标机器的实际路径：

```bash
export QFA_MOCK_ROOT=/path/to/aicpu_mock
export QFA_TTK_REPO=/path/to/ops-test-kit
export QFA_DEVICE_PYTHON=/absolute/path/to/device/environment/bin/python
export SIM_SOC='<实际仿真器标识>'

ttk_command=("$QFA_DEVICE_PYTHON" -m ttk e2e
  --backend npusim --sim-soc "$SIM_SOC"
  -i "$QFA_MOCK_ROOT/examples/qfa_mxfp8_lse.csv"
  --plugin "$QFA_MOCK_ROOT/ttk_plugin.py"
  --golden-mode Enable)

python3 scripts/prepare_platform_evidence.py \
  --model /actual/toolkit/tools/simulator/MODEL/camodel/libmodel.so \
  --model-version '<实际安装版本>' \
  --platform /actual/toolkit/aarch64-linux/data/platform_config/PLATFORM.ini \
  --qfa-binary /actual/opp/vendors/custom/path/to/qfa-kernel-binary \
  --build-record /actual/build-record.json \
  --mapping /actual/platform-mapping.json \
  --cwd "$QFA_TTK_REPO" \
  --output run.local.json \
  -- "${ttk_command[@]}"

python3 -m qfa_mock launch --evidence run.local.json -- "${ttk_command[@]}"
```

`prepare_platform_evidence.py` 只打包已有记录和实际文件，计算摘要并校验，不推断 mk001 架构。输出文件必须尚不存在。它记录当前 ASCEND_TOOLKIT_HOME、ASCEND_OPP_PATH、可选自定义 OPP 路径和初始 LD_LIBRARY_PATH。内核文件必须位于记录的 OPP 搜索根；platform ini 必须位于对应 CANN 平台目录；模型必须位于对应 toolkit/model/camodel 目录。

`launch` 只执行记录中完全相同的 argv，绑定 run_id、记录摘要及命令摘要，并向子进程传递 `QFA_PLATFORM_EVIDENCE` / `QFA_RUN_CONTEXT`。不要手工构造启动回执。worker 允许并核对上游 TTK 向 LD_LIBRARY_PATH 首部注入对应 camodel 目录这一转换；其他路径漂移失败。原始路径中已有 camodel 时也必须处于首位，避免先加载其他 Runtime。

首版有效核数必须来自实际构建/部署记录，禁止额外控核；`--sim-cores` 是记录/报告核选择，不能用它证明或设置有效 AIC/AIV。配置、运行证据、产物或查询结果冲突时立即失败。

## 实际 tiling

已有提取的真实 tiling JSON 时，在准备命令中增加 `--tiling /actual/tiling.json`。其字段为：

```json
{
  "request_digest": "<当前统一请求摘要>",
  "kernel_arch": "arch92",
  "section_num": 1,
  "is_fd": 0,
  "m_base": 128,
  "s2_base": 256,
  "aic_core_num": 80,
  "aiv_core_num": 80
}
```

FD 用例还需 `fd_workspace_slots`，不得小于 metadata 需要的归约存储槽数。该文件应是实际 tiling/执行记录的提取结果，不是从 mock Header 复制出的“期望值”。

没有事前 tiling 时，按绑定的构建、平台和调度源码记录预检，日志记录 `tiling_review_required`。首次仿真必须保留实际 tiling 复核，未复核的组合不能登记为已验收。tiling 与单次请求绑定；跨参数使用时重新生成运行证据。

## 运行时查询

原始 SoC 返回值可为字符串或枚举整数，只有规范化后才比较。物理总核数大于记录的有效核数不构成冲突；总容量小于有效核数则失败。查询报错或语义不明时记录诊断，使用有效离线证据继续预检；没有有效证据时停止设备执行。

当前未接入一个经真实 SDK 验证、能独立证明有效核数和构建身份的查询适配器，因此默认设备入口要求离线记录和启动绑定。无需等待该适配器即可使用上述路径；不会把 `get_device_properties()` 的总核数当作完整替代证据。
