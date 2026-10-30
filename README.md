# QFA Mock AICPU

在 Host CPU 上生成 QFA metadata，供原 CPU golden 和真实 AI Core kernel 消费。C++ 直接复用原分块函数、SectionStreamK 调度及 metadata 编码；Python 负责请求、缓存和校验。

**当前推荐：应用一个 assets 补丁，继续使用原 Spec 运行 TTK。** 不再要求外置 Spec、运行时函数替换、异常桥接或平台启动回执。设计决策见 [简化计划](docs/SIMPLIFICATION_PLAN.md)，当前状态见 [PROGRESS.md](PROGRESS.md)。

## 快速使用

以下在目标项目既有的 Torch/CANN 环境执行；CPU helper 的编译仅需 Python 3.9+ 和 C++17 编译器。

```bash
export QFA_MOCK_ROOT=/path/to/aicpu_mock
export QFA_REPO=/path/to/ops-transformer
export QFA_TTK_REPO=/path/to/ops-test-kit
cd "$QFA_MOCK_ROOT"

python3 -m qfa_mock build --repo "$QFA_REPO"
python3 scripts/apply_assets_patch.py --repo "$QFA_REPO" --check
python3 scripts/apply_assets_patch.py --repo "$QFA_REPO"
cp examples/target_arch92.json target.local.json
```

将 target.local.json 改为真实 SoC、编译架构及有效核数；arch35 可使用另一份示例。示例 80/80 **不等于已确认的 mk001 配置**。

```bash
export PYTHONPATH="$QFA_MOCK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export QFA_METADATA_CONFIG="$QFA_MOCK_ROOT/target.local.json"
export QFA_MOCK_HELPER="$QFA_MOCK_ROOT/build/metadata_helper"
unset QFA_PLATFORM_EVIDENCE QFA_RUN_CONTEXT
export SIM_SOC='<实际仿真器SoC标识>'

cd "$QFA_TTK_REPO"
python3 -m ttk e2e \
  --backend npusim --sim-soc "$SIM_SOC" \
  -i "$QFA_MOCK_ROOT/examples/qfa_mxfp8_lse.csv" \
  --plugin "$QFA_REPO/attention/quant_flash_attn/tests/assets" \
  --golden-mode Enable
```

完整说明和撤销命令见 [ASSETS_USAGE.md](ASSETS_USAGE.md)。源码压缩包也能编译和应用补丁：build 不调用 Git；补丁工具需要 Git 可执行程序，但不要求目标目录有 .git。

## build 实际依赖

--repo 只是包含 attention/ 的头文件根目录。编译直接调用 C++，不检查 Git 历史、assets、AICPU cpp、golden 或完整源码树；缺少头文件时输出编译器的具体错误。编译成功后根据 -MMD 记录实际头文件和 helper 的摘要，用于运行时失效检查。

主要依赖是 load_balance 头文件、qfa_adjust_sinner_souter.h、quant_flash_attn_metadata.h。产物为 build/metadata_helper 和 .json 构建清单。升级后重新编译一次；新的 build_format=2 清单只包含编译依赖，旧 metadata bundle/平台摘要需要重新生成。

## 独立获取 metadata

```bash
cd "$QFA_MOCK_ROOT"
python3 -m qfa_mock generate --repo "$QFA_REPO" \
  --config target.local.json --request examples/request.json --output out/first-plan
python3 -m qfa_mock validate --repo "$QFA_REPO" --bundle out/first-plan
```

输出目录须尚不存在。metadata.bin 为 little-endian int32 `(2, rowSize)`，result.json 含 Header、请求/计划摘要和校验结果。TTK 按当前用例动态生成计划，不读取这份独立输出。协议见 [CONTRACT.md](CONTRACT.md)。

## 支持范围

- 一次完整输入，无历史 KV cache 追加或跨轮状态。
- MXFP8、quant_mode=1、正向、Eager，D=Dv=128。
- Q / Q descale / Out=TND；KV=TND 或 PA_BNBD。
- mask=0/3/5、默认窗口；可沿用原 assets 生成的标准 causal mask。
- batch、head 数、有效/分配长度、有效核数均可变化；max_seqlen 必须显式为正。
- 不支持 sinks、V-tail、其他量化模式、decode descale 或新图模式。

配置存在即进入 mock，错误直接失败；未设置 QFA_METADATA_CONFIG 时原 native 行为保留。生产 Host、AI Core、Torch 算子注册和 TTK 不改。

目标配置是调度输入声明，不代表已验证实际 kernel tiling。可用查询检查容量等已知冲突；首次目标仿真仍需核对真实 tiling 与 Out/LSE。旧 [平台证据工具](PLATFORM_EVIDENCE.md) 和 ttk_plugin.py 保留为兼容/可选诊断，勿与新补丁叠加为两套入口。

## CPU 验收

```bash
python3.12 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements-cpu.txt
.venv-cpu/bin/python scripts/verify_cpu.py --repo "$QFA_REPO" --ttk "$QFA_TTK_REPO"
```

测试在临时复制的 assets 应用补丁，不改本地上游源文件。包含最小头文件目录编译、补丁幂等/撤销、无函数替换的直接执行、动态/静态槽、六组原 CPU 数值以及 native 选择检查；另保留旧方案兼容回归。补丁与旧路径的六组 Out/LSE 摘要逐项比较。

记录见 [verification_latest.json](docs/verification_latest.json)。torch_npu 仅为测试导入边界，metadata/设备调用触发即失败；真实 AICPU 对拍、H2D、AI Core 仿真精度仍需目标环境验收。
