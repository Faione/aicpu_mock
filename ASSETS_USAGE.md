# 编译与 assets 接入

## 1. 编译 Host helper

在运行测试的机器上准备本仓库和完整的 `ops-transformer` 源码。CPU 工具需要 Python 3.9+ 与 C++17 编译器；接入 TTK 还需要已有的 Torch、CANN、QFA 扩展和仿真环境。

```bash
export QFA_MOCK_ROOT=/path/to/aicpu_mock
export QFA_REPO=/path/to/ops-transformer

cd "$QFA_MOCK_ROOT"
python3 -m qfa_mock build --repo "$QFA_REPO" --compiler c++
python3 -m unittest discover -s tests -v
```

产物是 `build/metadata_helper` 和 `build/metadata_helper.json`（源码与二进制指纹）。只编译 Host C++，不编译 AICPU、不使用 Torch C++ 扩展。不要把本机编译的可执行文件直接搬到不同操作系统/架构；在目标测试机器重建。

修改 helper 或调度源码后重新执行 build。生成计划时不会自动编译。回归检查使用已记录的基线 SHA-256；旧 `arch92_80_80.bin` 存在时还会逐字节比较。完整原 Python 链路验收需安装 `requirements-cpu.txt`，并运行 `scripts/verify_cpu.py`，见 README。

## 2. 配置实际目标

```bash
cp examples/target_arch92.json target.local.json
```

编辑 `target.local.json` 中的 `target.soc_version`、`kernel_arch`、`aic_core_num`、`aiv_core_num`，使其与实际 QFA 编译产物和仿真运行平台一致。arch35 使用 `examples/target_arch35.json` 作为起点。

`Ascend960 / arch92 / 80+80` 是示例，尚未确认它对应 mk001。仅知道 80/80 不能决定架构。`*.local.json` 不进入 Git。

先在 Host 上验证一个计划：

```bash
export QFA_METADATA_CONFIG="$QFA_MOCK_ROOT/target.local.json"
export QFA_MOCK_HELPER="$QFA_MOCK_ROOT/build/metadata_helper"

python3 -m qfa_mock generate \
  --repo "$QFA_REPO" \
  --request examples/request.json \
  --output out/first-plan
python3 -m qfa_mock validate --repo "$QFA_REPO" --bundle out/first-plan
```

输出目录须尚不存在。公开输入和二进制协议详见 [CONTRACT.md](CONTRACT.md)。

## 3. 用插件接入现有 assets

当前主方案通过本仓库的 `ttk_plugin.py` 使用原 assets，**不需要修改原 assets 文件**。插件继承原 Spec，保留输入生成、compare 及原预处理中的 metadata 槽位回填逻辑。目录命名与设计建议不同，但文件可以合并，公开入口就是这个 `.py` 文件。

在已经能运行 QFA Eager 的 TTK 环境中，保留上述环境变量，准备与实际构建和启动绑定的平台证据。完整生成和启动命令见 [PLATFORM_EVIDENCE.md](PLATFORM_EVIDENCE.md)。最终 TTK argv 如下，必须通过 `qfa_mock launch` 绑定后执行：

```bash
export SIM_SOC='<仿真器支持且与目标一致的实际SoC名称>'
cd /path/to/ops-test-kit

python3 -m ttk e2e \
  --backend npusim \
  --sim-soc "$SIM_SOC" \
  -i "$QFA_MOCK_ROOT/examples/qfa_mxfp8_lse.csv" \
  --plugin "$QFA_MOCK_ROOT/ttk_plugin.py" \
  --golden-mode Enable
```

该 CSV 随本仓库提供，来自原型基线；也可换成满足首版范围的自有 CSV。CLI 的 `examples/request.json` 用于独立生成测试，**TTK 不会读取这份请求**；插件从当前用例的真实调用提取参数。

两个入口的实际路径如下：

```text
原 Spec 的 golden
  → 本插件 QuantFlashAttnMockSpec.golden
  → 原 impl/golden.py 的 cpu_qfa_mxfp8
  → mock get_qfa_section_info
  → provider → CPU metadata → 原 decoder → 原 CPU 分段 golden

原 Spec 的 npu_preprocess
  → 本插件 QuantFlashAttnMockSpec.npu_preprocess
  → 原 impl/npu_preprocess.py 的 run
  → mock run_metadata
  → provider → CPU metadata → 写入 anchor.device
  → 原动态槽返回 / resize_ / copy_ 回填
  → 原 qfa AI Core 主算子
```

不要继续使用旧的 `scratch/qfa-metadata-mock/mock_spec.py`：它读取固定二进制，新插件根据每次调用生成计划。两处替换都需要启用，才能消除 golden 和设备预处理的 AICPU 依赖。

插件在 worker 首次调用时才初始化 provider。相同请求复用不可变的 CPU 计划，交付 Tensor 时使用独立存储；关闭 golden 时，预处理仍会独立生成。

## 4. 用例要求与诊断

- 一次完整输入，无历史 KV cache 追加或跨轮状态。
- MXFP8、quant_mode=1、D=Dv=128、正向、Eager。
- Q / Q descale / Out 为 TND；KV 为 TND 或 PA_BNBD。
- max_seqlen_q/kv 显式为正，长度 Tensor 与 `*_values` sidecar 一致。
- mask_mode=0/3/5，窗口均为 -1。标准 mask=3 用例可沿用原输入生成器填充的 causal mask，`attn_mask_shape` 须与 Tensor shape 一致；不能绕过输入生成器传任意 mask。
- 不支持 sinks、V-tail、反向、N2TGD/BN2SGD descale 或图模式。

mock 不会修改 Runtime 核数或 qfa tiling。预处理核对绑定的模型/平台/内核文件摘要、构建记录中的有效核数、架构及调度源码；可用时核对真实 tiling。普通设备属性仅校验容量，SoC 查询按有来源的类型化别名映射比较。查询不可用时仍可用有效离线证据完成预检；无证据则停止设备执行。首版不支持额外控核，不提供跳过校验开关。

TTK 日志中查找 `mock_aicpu`，可看到目标、请求摘要、Header、活跃 AIC/AIV 和 cache_hit。相同输入的 golden 与预处理应使用相同摘要。错误会使当前用例失败，不回退真实 qfa_metadata 或全局参考计算。

取消 mock 时，将 `--plugin` 改为原 `$QFA_REPO/attention/quant_flash_attn/tests/assets`。选中外置 mock 插件时，取消设置 `QFA_METADATA_CONFIG` 会立即报错，不会恢复 native。

同一 worker 串行运行 mock 用例，不与 native 混跑；并发使用独立 worker。新版生命周期和验收项的当前完成情况见 [PROGRESS.md](PROGRESS.md)。

已完成实际原 hook 的 CPU 槽位矩阵、原 CPU golden 数值链路及实际 TTK 插件发现。测试对 torch_npu 只提供导入边界，原生 metadata/设备函数触发即失败；不是 NPU 仿真。真实 H2D、AICPU 对拍及 AI Core Out/LSE 精度仍需在目标环境验收。
