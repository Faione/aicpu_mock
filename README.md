# QFA Mock AICPU 原型

在 Host CPU 上生成一次完整 QFA 调用所需的 metadata，供 CPU golden 和真实 AI Core kernel 消费。Python 适配输入；C++ 直接复用 `ops-transformer` 的分块函数、`SectionStreamK::Compute` 和 metadata 编码类型，不执行 AICPU，也不计算 Attention 数值。

本目录独立于生产仓库。构建时指定 `ops-transformer` 源码路径，不复制或修改该仓库的生产文件。使用上游源码时遵循其 `LICENSE`。

## 首版范围

- 单次自包含请求，不维护历史 KV cache，不支持追加、跨轮复用或请求加入/退出。
- MXFP8、`quant_mode=1`、正向、Eager，Q/K/V 的 D 均为 128。
- Q、Q descale、Out 为 TND；KV 为 TND 或 PA_BNBD。PA 在这里仅表示本次完整 KV 的存储布局，不提供跨轮状态管理。
- `mask_mode=0/3/5`，`win_left=win_right=-1`；TTK 可沿用原 assets 生成的标准 causal mask（mask=3，须声明匹配的 attn_mask_shape），不支持任意自定义 mask、V-tail、sinks、反向及 decode descale 布局。
- batch、head 数、实际长度、分配长度、最大长度和有效核数均可变化。`max_seqlen_q/kv` 须显式为正，首版拒绝 `-1`，不会自行替换默认值。
- 显式支持 `Ascend960* / arch92 / AIC:AIV=1:1` 与 `Ascend910_95* / arch35 / 1:2` 两组策略。示例配置仅用于验证算法，**不是已确认的 mk001 配置**。

即使是一次性请求，仍可能生成 FD 或多个 L2 section；它们是单次调度内部的任务划分。

## 本地运行

要求 Python 3.9+、支持 C++17 的编译器，以及已有的 `ops-transformer` 源码。CPU CLI 和测试只用 Python 标准库，不需要 Torch、CANN 或 NPU，不访问网络。

在本目录运行，先显式编译一次：

```bash
export QFA_REPO=/Users/fanghaolei/Workplace/agent/ops-transformer
python3 -m qfa_mock build --repo "$QFA_REPO"
python3 -m qfa_mock generate \
  --repo "$QFA_REPO" \
  --config examples/target_arch92.json \
  --request examples/request.json \
  --output out/my-baseline
python3 -m qfa_mock validate --repo "$QFA_REPO" --bundle out/my-baseline
python3 -m unittest discover -s tests -v
```

上述标准库环境会跳过需要 Torch 的原 Python 链路测试。完整 CPU 验收使用 Python 3.10–3.12（本轮为 3.12）：

```bash
python3.12 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements-cpu.txt
.venv-cpu/bin/python scripts/verify_cpu.py --repo "$QFA_REPO" --ttk /path/to/ops-test-kit
```

脚本保存测试输出、真实 CPU Out/LSE 摘要和上游源码完整性记录；出现跳过项不会登记为完整验收通过。

输出目录必须尚不存在，防止失败后的旧结果被误用。每个 bundle 包含 `request.json`、`config.json`、`metadata.bin`、`result.json`。协议详见 [CONTRACT.md](CONTRACT.md)。

生成实际包含 FD 或多个 section 的计划：

```bash
python3 -m qfa_mock generate --repo "$QFA_REPO" \
  --config examples/target_arch35.json --request examples/request_fd.json --output out/my-fd
python3 -m qfa_mock generate --repo "$QFA_REPO" \
  --config examples/target_arch35.json --request examples/request_multisection.json --output out/my-sections
```

## Python 消费接口

```python
from qfa_mock import MetadataProvider
from qfa_mock.model import load_json

provider = MetadataProvider(
    load_json("examples/target_arch92.json"),
    repo="/path/to/ops-transformer",
    helper="build/metadata_helper",
)
plan = provider.generate(load_json("examples/request.json"))
cpu_metadata = plan.cpu_tensor()              # 此处才导入 Torch
device_metadata = plan.device_tensor(anchor)  # 与设备锚点同设备
```

主副本为不可变 bytes；每次 Tensor 交付独立存储。进程内缓存有数量上限，键包括完整目标、规范化请求（含原始长度）、算法版本与源码指纹。源码或二进制改变后，即使缓存命中也要求重新构建，不会在用例运行期间编译。

## 可选 TTK 适配

`ttk_plugin.py` 是当前零源码修改接入的主方案：通过外置 Spec 在 worker 中局部替换两个计划入口，保留原始输入生成、golden 数值计算、metadata 槽位回填与 compare。无需修改生产 assets。

完整编译、配置、TTK 命令和 assets 调用路径见 [ASSETS_USAGE.md](ASSETS_USAGE.md)。

在具备 Torch/CANN/仿真运行环境的机器上：

```bash
export QFA_REPO=/path/to/ops-transformer
export QFA_METADATA_CONFIG=/path/to/aicore_mock/examples/target_arch92.json
export QFA_MOCK_HELPER=/path/to/aicore_mock/build/metadata_helper

# 先按实际目标修改配置、确认编译架构和仿真平台，再构建 helper。
# 在 ops-test-kit 的现有 e2e 命令中使用：
# --plugin /path/to/aicore_mock/ttk_plugin.py
# --backend npusim --sim-soc <实际SoC> --golden-mode Enable
```

- 先从两个实际调用签名读取字段，不依赖 golden 上一个用例遗留的全局参数；只读回小型长度 Tensor，不读回 Q/K/V 数值。
- golden 获取 CPU 计划后调用原 decoder。原分段 golden 发生异常时直接判失败，阻止它原有的全局参考回退。
- 预处理核对原 `build_metadata_arguments` 与当前请求一致，复用本次 MetadataResult，验证平台/构建证据后传输；动态槽、静态槽继续由原 hook 完成。设备执行使用绑定实际启动命令的离线记录，见 [PLATFORM_EVIDENCE.md](PLATFORM_EVIDENCE.md)。普通设备属性不能单独证明有效核数与编译架构。
- 选中本插件后必须提供有效 `QFA_METADATA_CONFIG`，缺失或错误时失败，不回退 native。native 模式改用原 assets 的 `--plugin` 路径。关闭 golden 时，预处理仍独立生成计划。
- 临时模块替换在本插件中串行化；同一 worker 串行执行 mock 用例，不与 native 混跑。显式修改两个 assets 的 provider 分支保留为将来的可选方案。

加载器按绝对路径固定 checkout，并检查原 Spec、impl、数值模块和支持包来源及函数签名。同 worker 固定规范化配置、helper 路径和构建身份；语义变化失败，不清缓存后继续。相同配置的 JSON 排版变化或文件迁移允许复用，子进程按新 PID 建立状态。设备校验区分仿真器名、CANN 平台名、调度 SoC 和编译架构；别名来自有摘要的安装/项目映射，不比较不同命名体系的原始字符串。

设计及当前验收状态见 [PROGRESS.md](PROGRESS.md) 和 [同步设计稿](docs/qfa_mock_aicpu_design.md)。最新方案明确允许两个计划接管点加一个 golden 异常保护点。普通数值异常在外置边界转成 MockError，用户中断/退出保持原样；只有一次计划消费、非空解码和一次成功分段计算都完成才返回结果。三处替换在成功及异常时均恢复，并记录路径、摘要、阶段和恢复结果。

## 已验证与尚未验证

本机已完成 Host C++ 构建、加载/平台校验、实际原 hook、原 CPU golden 与实际 TTK SpecLoader 的验收，记录见 [verification_latest.json](docs/verification_latest.json)：

- arch92 80/80 基线 `(2, 167936)`，M=128、S2=256、64 个活跃 AIC；与旧原型 `arch92_80_80.bin` **逐字节一致**。
- arch35 36/72 的 FD 用例，生成 8 个归约任务；多 section 用例生成 4 个 section、137 个归约任务。
- 原 golden decoder 的任务完整覆盖与无重复检查，变长 batch、mask、零长、padding、初始化标志、FA/FD 保留字段、未使用空间、FD 行区间与 workspace 范围检查。
- 缓存隔离、错误请求拒绝、失效 helper、损坏结果拒绝、双入口请求一致、golden 禁用时预处理独立运行，以及分段 golden 异常不进入全局回退。
- 实际运行原输入生成、CPU golden 的输入适配、decoder、分段函数和输出整理，覆盖基线、变长、padding、零长、FD 和多 section。原 CPU Out/LSE 为 FP32，保留其原始返回结构。
- 使用真实 CPU Torch Tensor 执行原预处理的动态槽、`(0,)`、静态匹配和不匹配路径；只在这类测试中绕过设备平台证据边界。
- 使用实际 TTK SpecLoader 发现外置 Spec，确认输入、compare、容差继承；原 native Spec 仍可独立发现。测试中原生 metadata 和设备入口被设为“触发即失败”。

**本机已安装独立 CPU Torch 环境，但没有 CANN/NPU：真实 AICPU 对拍、H2D、AI Core 仿真 Out/LSE 精度比较仍未完成。** 测试中的 torch_npu 仅为导入边界替身，未模拟任何数值计算；不能据此登记设备验收通过。
