# QFA Mock AICPU

在 Host CPU 上生成 QFA metadata，供原 CPU golden 和真实 AI Core kernel 消费。C++ 直接复用原分块函数、SectionStreamK 调度及 metadata 编码；Python 负责请求、缓存和校验。

**当前推荐：应用一个 assets 补丁，继续使用原 Spec 运行 TTK。** 不再要求外置 Spec、运行时函数替换、异常桥接或平台启动回执。设计决策见 [简化计划](docs/SIMPLIFICATION_PLAN.md)，当前状态见 [PROGRESS.md](PROGRESS.md)。

下一轮设计见 [迭代计划](docs/ITERATION_PLAN.md)：在 Torch 加载的真实 C++ 扩展中分派 metadata 的 native/mock，helper 与扩展共享 Host 核心，并补容量适配、版本检查与 CSV 目标配置。以上仍待实施，下面命令对应当前已交付方案。

## 在目标项目使用

以下在目标项目既有的 Torch/CANN/TTK 仿真环境执行。mock 工具使用 Python 3.9+ 和 C++17 编译器；应用补丁需要 Git 可执行程序。目标项目需包含完整的必需源码，但不要求有 .git 或提交历史。

### 1. 获取代码并编译

首次获取本工具：

```bash
git clone git@github.com:Faione/aicpu_mock.git /path/to/aicpu_mock
```

将下面三个目录改为目标机器上的绝对路径。QFA_REPO 必须指向包含 attention/ 的源码根目录，QFA_TTK_REPO 指向能执行 python3 -m ttk 的 TTK 项目目录。

```bash
export QFA_MOCK_ROOT=/path/to/aicpu_mock
export QFA_REPO=/path/to/ops-transformer
export QFA_TTK_REPO=/path/to/ops-test-kit
cd "$QFA_MOCK_ROOT"

python3 -m qfa_mock build --repo "$QFA_REPO"
```

构建前强制检查头文件、assets/golden 和 AICPU 参考 cpp；缺少任一项都会停止，并列出全部缺失路径。成功后生成 build/metadata_helper 和 build/metadata_helper.json。默认使用 c++，需要指定编译器时添加 --compiler g++。在实际运行的操作系统和架构上编译 helper。

### 2. 应用 assets 补丁

仍在 QFA_MOCK_ROOT 中执行：

```bash
python3 scripts/apply_assets_patch.py --repo "$QFA_REPO" --check
python3 scripts/apply_assets_patch.py --repo "$QFA_REPO"
```

--check 只检查是否可应用；第二条命令修改目标项目的三个 assets 文件。重复应用会报告 Already applied；源码版本不匹配时先解决补丁错误再继续。TTK 使用补丁后的原 assets 插件。

### 3. 配置目标并启用 mock

```bash
cp examples/target_arch92.json target.local.json
```

编辑 target.local.json；arch35 从 examples/target_arch35.json 复制。以下字段必须与真实目标匹配：

| 配置字段 | 含义 |
|---|---|
| target.name | 自定义目标名称，用于记录 |
| target.soc_version | 实际目标 SoC；当前支持 Ascend960* 或 Ascend910_95* |
| target.kernel_arch | 对应 arch92 或 arch35 |
| target.aic_core_num / target.aiv_core_num | 本次调度使用的有效 AIC/AIV 核数；arch92 要求 1:1，arch35 要求 1:2 |

示例 80/80 **不等于已确认的 mk001 配置**。cache 配置控制 CPU metadata 结果缓存，不表示支持历史 KV cache 或多轮追加输入。

```bash
export PYTHONPATH="$QFA_MOCK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export QFA_METADATA_CONFIG="$QFA_MOCK_ROOT/target.local.json"
export QFA_MOCK_HELPER="$QFA_MOCK_ROOT/build/metadata_helper"
unset QFA_PLATFORM_EVIDENCE QFA_RUN_CONTEXT
```

这些环境变量需要传递给 TTK 及其 worker：QFA_REPO 提供源码位置，PYTHONPATH 使 assets 能导入 qfa_mock，QFA_METADATA_CONFIG 选择目标配置，QFA_MOCK_HELPER 指向编译产物。配置非空即启用 mock；配置错误会直接失败。

### 4. 运行 TTK

确认目标项目既有的 Torch/CANN/仿真环境已加载，再执行：

```bash
export SIM_SOC='<实际仿真器SoC标识>'

cd "$QFA_TTK_REPO"
python3 -m ttk e2e \
  --backend npusim --sim-soc "$SIM_SOC" \
  -i "$QFA_MOCK_ROOT/examples/qfa_mxfp8_lse.csv" \
  --plugin "$QFA_REPO/attention/quant_flash_attn/tests/assets" \
  --golden-mode Enable
```

替换 SIM_SOC 为仿真器支持的真实 SoC 标识。示例 CSV 用于首轮验证，随后将 -i 替换为自己的用例文件，并确认参数处于下方支持范围。--plugin 始终指向目标项目原 assets。

TTK 按当前用例自动生成 metadata：CPU golden 使用解码后的分段计划，预处理生成并回填 metadata 张量，然后执行真实 QFA AI Core kernel。无需手工生成 metadata.bin 或填写 metadata 槽。

启用 INFO 日志时，核对同一用例的 patched_golden_plan 与 patched_preprocess_plan 中 request_digest、metadata_digest 是否一致，再查看 TTK compare 的 Out/LSE 结论。关闭 golden 时预处理仍可独立生成计划。当前本机只完成 CPU 验证，首次目标运行仍需验收真实 tiling 与 Out/LSE。

### 5. 关闭 mock 或撤销补丁

```bash
unset QFA_METADATA_CONFIG
```

后续运行会走原 native metadata 分支。需要恢复 assets 源码时执行：

```bash
cd "$QFA_MOCK_ROOT"
python3 scripts/apply_assets_patch.py --repo "$QFA_REPO" --reverse
```

更多补丁细节见 [ASSETS_USAGE.md](ASSETS_USAGE.md)。

## build 实际依赖

--repo 指向包含 attention/ 的源码根目录，不要求有 Git 历史。编译前必须检查编译头文件、assets/golden 和 AICPU cpp 是否存在；若缺失，一次列出全部文件的分组、绝对路径和用途，并返回 SOURCE_MISSING，不再只输出笼统的源码缺失提示。这里检查的是文件完整性，不代表源码行为或设备已验收。

这些文件均为 build 的强制前置条件，缺少任一项都停止构建，不提供跳过选项。其他实际引用但未列入预检查的缺失头文件，由编译器给出具体错误。编译成功后根据 -MMD 记录实际头文件和 helper 摘要；assets/golden、AICPU cpp 不进入编译指纹。

主要依赖是 load_balance 头文件、qfa_adjust_sinner_souter.h、quant_flash_attn_metadata.h。产物为 build/metadata_helper 和 .json 构建清单。升级后重新编译一次；新的 build_format=2 清单只包含编译依赖，旧 metadata bundle/平台摘要需要重新生成。

## 独立获取 metadata

```bash
cd "$QFA_MOCK_ROOT"
python3 -m qfa_mock generate --repo "$QFA_REPO" \
  --config target.local.json --request examples/request.json --output out/first-plan
python3 -m qfa_mock validate --repo "$QFA_REPO" --bundle out/first-plan
```

输出目录须尚不存在。metadata.bin 为 little-endian int32 `(2, rowSize)`，result.json 含 Header、请求/计划摘要和校验结果。TTK 按当前用例动态生成计划，不读取这份独立输出。协议见 [CONTRACT.md](CONTRACT.md)。

修改 examples/request.json 中的请求参数可生成其他单次输入的 metadata；request.json/config.json 会一并保存到输出目录，便于复现。此步骤可用于先检查调度结果，也可以直接跳过并运行 TTK。

## 常见问题

| 现象 | 处理方式 |
|---|---|
| SOURCE_MISSING | 按 error.missing 中的绝对路径补齐文件；确认 --repo 指向包含 attention/ 的根目录。检查不可跳过 |
| BUILD_FAILED | 查看编译器错误，确认 C++17 编译器可用，并补齐编译器指出的额外 include 依赖 |
| 无法导入 qfa_mock | 在启动 TTK 的同一环境导出 PYTHONPATH，确保包含 QFA_MOCK_ROOT |
| MISSING_CONFIG | 确认 QFA_METADATA_CONFIG 和 QFA_REPO 已导出并传递到 worker |
| STALE_BUILD | 重新执行 build；若已有 worker，重启 TTK 后再运行。独立生成的旧 bundle 也应重新生成 |
| 补丁无法应用 | 核对目标 assets 版本及已有改动，解决冲突后重新执行 --check |

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

测试在临时复制的 assets 应用补丁，不改本地上游源文件。包含完整源码无 Git 构建、缺失必需文件阻止构建、补丁幂等/撤销、无函数替换的直接执行、动态/静态槽、六组原 CPU 数值以及 native 选择检查；另保留旧方案兼容回归。补丁与旧路径的六组 Out/LSE 摘要逐项比较。

记录见 [verification_latest.json](docs/verification_latest.json)。torch_npu 仅为测试导入边界，metadata/设备调用触发即失败；真实 AICPU 对拍、H2D、AI Core 仿真精度仍需目标环境验收。
