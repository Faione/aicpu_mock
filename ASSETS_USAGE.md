# 在目标项目使用 Mock AICPU

## 1. 编译 helper

```bash
export QFA_MOCK_ROOT=/path/to/aicpu_mock
export QFA_REPO=/path/to/ops-transformer
export QFA_TTK_REPO=/path/to/ops-test-kit
cd "$QFA_MOCK_ROOT"
python3 -m qfa_mock build --repo "$QFA_REPO"
```

build 不依赖 Git 仓库，不要求 assets 或 AICPU cpp；--repo 只提供 C++ include 根。若缺少真正引用的头文件，编译器会指出具体路径。产物为 build/metadata_helper 和实际依赖清单。默认 c++，可加 --compiler g++；在目标操作系统/架构重建，不直接复制不同平台的可执行文件。

## 2. 应用唯一的 assets 补丁

```bash
python3 scripts/apply_assets_patch.py --repo "$QFA_REPO" --check
python3 scripts/apply_assets_patch.py --repo "$QFA_REPO"
```

工具使用 Git 的 patch 功能，但目录不必有 .git；重复应用会报告 Already applied。不匹配上游版本时失败，不能强行忽略补丁错误后继续。

一个补丁只修改三个测试资产：

| 文件 | 变化 |
|---|---|
| impl/golden.py | 从当前真实形参构造 mock 请求，取得 CPU 分段计划，显式传给原数值函数；其他量化模式在 mock 下提前拒绝 |
| quant_flash_attn_golden.py | 增加可选 metadata_plan 参数；给定计划时不调用设备 metadata，数值异常直接抛出 |
| impl/npu_preprocess.py | 根据当前请求选择本地 metadata 生成函数，保留原动态槽返回、resize、形状检查和 copy |

不改原 spec.py、输入生成、compare、TTK、生产算子或 Torch 注册；不替换模块函数，不使用 BaseException 桥接。补丁中的 import 仅在 mock 启用时执行。

## 3. 配置并导出环境

```bash
cp examples/target_arch92.json target.local.json
# 编辑 target.local.json：填入真实 SoC、编译架构和有效 AIC/AIV 核数。
export PYTHONPATH="$QFA_MOCK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export QFA_METADATA_CONFIG="$QFA_MOCK_ROOT/target.local.json"
export QFA_MOCK_HELPER="$QFA_MOCK_ROOT/build/metadata_helper"
unset QFA_PLATFORM_EVIDENCE QFA_RUN_CONTEXT
```

arch35 从对应示例开始。mk001 与 80/80 不能自动确定架构。配置文件只提供调度输入，不改变 CANN 平台；首次执行仍应核对真实 qfa tiling、有效核数及输出。普通设备属性仅检查容量等已知冲突，不能代替完整 tiling 验收。

## 4. 可选：先独立查看 metadata

```bash
python3 -m qfa_mock generate --repo "$QFA_REPO" \
  --request examples/request.json --output out/first-plan
python3 -m qfa_mock validate --repo "$QFA_REPO" --bundle out/first-plan
```

输出目录必须是新的。metadata.bin 是完整二进制，result.json 给出 Header、shape、摘要和校验结论；request.json/config.json 保存复现输入。build 只需头文件，generate/validate 还会读取原 golden decoder 来做覆盖校验。

## 5. 直接运行 TTK

在既有 Torch/CANN/仿真环境中：

```bash
export SIM_SOC='<CANN仿真器实际支持的SoC标识>'
cd "$QFA_TTK_REPO"
python3 -m ttk e2e \
  --backend npusim --sim-soc "$SIM_SOC" \
  -i "$QFA_MOCK_ROOT/examples/qfa_mxfp8_lse.csv" \
  --plugin "$QFA_REPO/attention/quant_flash_attn/tests/assets" \
  --golden-mode Enable
```

使用原 assets 的插件路径。无需 ttk_plugin.py、qfa_mock launch、build/mapping 运行记录或手工填 metadata 槽。TTK 从实际用例生成请求，golden 与预处理共享相同 CPU 计划内容，qfa 仍执行真实 AI Core kernel。

日志检查 patched_golden_plan 与 patched_preprocess_plan 的 request_digest / metadata_digest 是否一致；随后检查原 compare 的 Out/LSE 结论。关闭 golden 时，预处理仍可独立生成计划。

首版只支持 README 所列组合。新增参数必须真实传到两端，不能只因 helper 可计算就声称端到端已支持。

## 6. 关闭或撤销

```bash
unset QFA_METADATA_CONFIG
# 若需要移除源码改动：
cd "$QFA_MOCK_ROOT"
python3 scripts/apply_assets_patch.py --repo "$QFA_REPO" --reverse
```

未设置配置时补丁保留原 native 分支；设置了无效配置则失败，不回退 AICPU。工具保留目标 LF/CRLF 行尾，应用后撤销已做字节级回归。

旧外置 Spec/严格平台证据路径仍可在未应用补丁的 checkout 使用，作为兼容方案；说明见 PLATFORM_EVIDENCE.md。它不再是当前主方案。

本机已验证补丁后的真实 CPU 函数和槽位路径。真实 H2D、AICPU 对拍与 AI Core 仿真 Out/LSE 精度仍需目标环境完成。
