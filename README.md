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

`ttk_plugin.py` 沿用旧原型的自定义 Spec 方式，在 worker 中局部替换两个入口，保留原始输入生成、golden 数值计算、metadata 槽位回填与 compare。无需修改生产 assets。

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
- 预处理核对原 `build_metadata_arguments` 与当前请求一致，检查运行时核数（SoC 查询可用时也检查），然后传输同一 metadata 内容；动态槽、静态槽的处理继续由原 hook 完成。
- 未设置 `QFA_METADATA_CONFIG` 时，自定义 Spec 委托原 native 方法。mock 模式不回退真实 AICPU。关闭 golden 时，预处理仍独立生成计划。
- 临时模块替换在本插件中串行化；这是 Eager 原型接入方式，其他代码不能同时替换相同模块函数。后续正式接入可改成两个 assets 的显式 provider 分支。

配置中的编译架构是用户声明，原型无法从实际 kernel 二进制自动验证；运行仿真前仍需核对 tiling 日志。两个入口使用相同规范化规则与 provider；请求摘要可用于对比，本版没有建立基于 TTK testcase 生命周期的跨入口差异追踪器。

## 已验证与尚未验证

本机已完成 Host C++ 构建与 18 项 CPU 测试：

- arch92 80/80 基线 `(2, 167936)`，M=128、S2=256、64 个活跃 AIC；与旧原型 `arch92_80_80.bin` **逐字节一致**。
- arch35 36/72 的 FD 用例，生成 8 个归约任务；多 section 用例生成 4 个 section、137 个归约任务。
- 原 golden decoder 的任务完整覆盖与无重复检查，变长 batch、mask、零长、padding、初始化标志、FA/FD 保留字段、未使用空间、FD 行区间与 workspace 范围检查。
- 缓存隔离、错误请求拒绝、失效 helper、损坏结果拒绝、双入口请求一致、golden 禁用时预处理独立运行，以及分段 golden 异常不进入全局回退。
- 两个集成入口的控制流程使用替身验证；未启用配置时委托 native 的行为也有测试。

**本机没有 Torch/CANN/NPU：未做真实 AICPU 对拍、真实 H2D、TTK 仿真或 Out/LSE 精度比较。** CPU 检查通过不等于目标平台验收完成。后续需要用已确认的 SoC、架构和有效核数完成这些验证。
