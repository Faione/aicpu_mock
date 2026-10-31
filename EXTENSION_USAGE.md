# 原 assets 不改的 C++ 扩展接入

本模式按主设计第 26 节与 `.run` 详细设计第 10 节实现。交付代码与 CPU 验证已完成，真实 CANN 扩展编译、安装、NPU 传输和 AI Core 精度尚未在本机验收。当前机器是 macOS/arm64，没有 torch_npu/CANN；下面的目标步骤在已有 Torch/CANN 仿真环境执行。

## 1. 检查目标并编译独立 helper

```bash
export QFA_MOCK_ROOT=/path/to/aicpu_mock
export QFA_REPO=/path/to/ops-transformer
cd "$QFA_MOCK_ROOT"
python3 scripts/probe_target.py --repo "$QFA_REPO"
python3 -m qfa_mock build --repo "$QFA_REPO"
```

helper 仍由普通 C++17 编译器构建，build 不依赖 Git。必需头文件、assets/golden 和 AICPU cpp 缺失时一次列全，不提供跳过选项。构建清单已升级为 build_format=3：真正编译的依赖和行为兼容身份分别记录。首次升级须重新 build 并重新生成旧 bundle。

compatibility/qfa_host_v2.json 登记已核对源码摘要。未知上游版本会报 UPSTREAM_INCOMPATIBLE，不能只改清单就宣称适配完成；先复核 ParamsInit、分块、FD、初始化与编码规则，再做独立回归。helper 和扩展共同编译 cpp/metadata_core.cpp，不在扩展内启动 helper 子进程。

## 2. 在隔离 checkout 应用扩展接入

不要与旧 assets_explicit 补丁叠加。若原 checkout 已应用旧补丁，先用旧工具撤销；建议在隔离 checkout 中验证新模式。

```bash
python3 scripts/apply_assets_patch.py --repo "$QFA_REPO" --reverse
python3 scripts/apply_extension_patch.py --repo "$QFA_REPO" --check
python3 scripts/apply_extension_patch.py --repo "$QFA_REPO"
```

如果没有应用过旧补丁，省略第一条。新工具只修改三个原文件：

| 原文件 | 修改 |
|---|---|
| Torch C++ quant_flash_attn.cpp | 增加 include、包装 metadata 的唯一 PyBind 绑定、注册私有控制接口；native metadata 与主 qfa 的原函数体不改，主 qfa 绑定不改 |
| Torch Python quant_flash_attn.py | 用独立准备函数取得 mock 容量；在原主 qfa 调用前校验当前支持范围；补充该算子的 sources/includes |
| torch_extension/setup.py | 收集嵌套 C++/头文件、Python 资源和身份清单 |

其余逻辑放在新增文件。原 assets、TTK、Torch schema 和 CANN 算子实现不改。工具检查唯一绑定及已知函数签名，Python 修改用 AST 定位；不按固定行号改写函数。已应用时幂等，撤销逐字节还原；文件被再次编辑时拒绝覆盖，错误版本不会强行应用。

## 3. 构建独立 vendor wheel

```bash
python3 scripts/build_extension_wheel.py --repo "$QFA_REPO" \
  --vendor qfamock --output "$QFA_MOCK_ROOT/out/qfamock-wheel"
```

输出目录须尚不存在。工具调用目标项目原 setup.py 构建实际 vendor wheel，并输出 wheel_manifest.json，核对共享核心、上游头文件、bridge、运行时模块、entry point 和清单是否齐全。

vendor wheel 同时内置原基础 cann_ops_transformer.ops 发现框架，干净环境不需另装基础 COT wheel；只保留 vendor 的 qfa provider，不配送第二份基础 qfa 实现。wheel 内包含源码，首次在目标环境调用时由原 OpBuilder JIT 编译扩展；构建 wheel 成功不代表 CANN 扩展已编译。目标需要匹配的 Python、Torch、torch_npu/CANN、C++ 编译器和 Ninja。安装后不需要开发仓库或 QFA_REPO/QFA_MOCK_HELPER；新增源码和行为依赖已内置。mock 当前不缓存计划，每次执行共享核心；旧 JSON 的合法 cache 配置可读取，但不启用第二层 Python 缓存。

## 4. 生成单个安装 .run

先按目标项目已有构建流程取得真正的 CANN custom .run，vendor 必须与 wheel 一致，且包含真实 quant_flash_attn、native quant_flash_attn_metadata 及其所需依赖。本工具不在缺 CANN 的机器上伪造原算子包。

复制 examples/opp_release.template.json，填入实际包的 SHA-256、vendor、Host 架构、kernel 架构、实际 operator 清单及完整 installed_files（相对 vendor 根目录的所有交付文件与 SHA-256）。behavior_digest 必须由同源的原生构建记录提供，并与 wheel_manifest.json 中的行为身份一致。清单是本次原生包的构建声明，摘要校验不代替真实算子执行验证。

```bash
python3 scripts/build_run_package.py \
  --opp-package /path/to/actual-custom-ops.run \
  --opp-manifest /path/to/actual-opp-release.json \
  --wheel /path/to/cann_ops_transformer_qfamock-1.0.0-py3-none-any.whl \
  --output "$QFA_MOCK_ROOT/out/qfa-mock-bundle.run"

# 只核对归档及清单，不安装、不执行内含 CANN 包：
sh "$QFA_MOCK_ROOT/out/qfa-mock-bundle.run" --inspect
```

单个外层 .run 携带已有 CANN 包、唯一匹配的 wheel 和严格安装器。--inspect 仅证明内容/摘要完整，不证明 SDK、依赖闭包或仿真通过。构建输入缺 wheel、核心文件、entry point、匹配 vendor 或 native metadata 声明时失败。

## 5. 在目标 Linux 环境安装并运行原 TTK

```bash
# 先加载目标 Torch/CANN 环境，使用与 TTK 相同的 Python：
export QFA_PACKAGE_PYTHON=python3
sh /path/to/qfa-mock-bundle.run --prefix /absolute/new/qfa-mock-install
source /absolute/new/qfa-mock-install/bin/set_env.bash
export QFA_METADATA_BACKEND=mock_aicpu
export QFA_METADATA_CONFIG=/absolute/path/target.local.json

cd /path/to/ops-test-kit
python3 -m ttk e2e \
  --backend npusim --sim-soc "$SIM_SOC" \
  -i /path/to/original-cases.csv \
  --plugin "$QFA_REPO/attention/quant_flash_attn/tests/assets" \
  --golden-mode Enable
```

prefix 必须是新的绝对路径。安装器先检查 Linux/Host 架构和 Torch/torch_npu/pip，再安装 CANN 包并逐项核对 installed_files，随后安装 wheel，在独立进程核对 provider 来源、完整载荷并预热真正的 C++ 扩展。成功后才写激活脚本并移除 INSTALL_INCOMPLETE；失败非零退出，不把“复制 wheel 成功”当作安装成功，不覆盖已有 prefix。

激活脚本只设置本次安装的 Python、OpAPI 和 OPP 路径，默认不主动打开 mock。安装包模式不再要求 QFA_REPO 或 QFA_MOCK_HELPER；上例的 QFA_REPO 仅供 --plugin 指向原 assets，也可直接填绝对路径。目标 JSON 格式沿用 examples/target_arch92.json/target_arch35.json，示例核数不代表 mk001 架构已经确认。暂不从 CSV 读取目标配置。

未设置新开关时，兼容非空 QFA_METADATA_CONFIG 启用 mock；两者均空为 native。显式 native 与非空配置冲突失败。每个 worker 固定模式/目标/构建身份，修改后重启 TTK。原 golden 的导入和 NPU Runtime 仍然必须可用。

bridge 接受真实 output 的显式 rowSize，只要求实际 Header/FA/FD 计划能放下，剩余与第二行清零。Python 默认按目标上界分配。CSV 的动态 (-1,-1) 或 (0,) 占位可沿用；旧固定槽与新 shape 不同仍被原 assets 拒绝，不能通过截断或静默 resize 兼容。

## 6. 验收与错误边界

```bash
cd "$QFA_MOCK_ROOT"
.venv-cpu/bin/python scripts/verify_cpu.py --repo "$QFA_REPO"
.venv-cpu/bin/python scripts/verify_cpp_bridge.py
```

第一条做旧路径及核心/接入/真实 vendor wheel 回归；第二条实际编译同一 C++ 包装器与核心，通过测试专用 CPU 设备边界验证 PyBind、CPU dispatcher、native 指针、返回同一 output、FD/多 section 与容量。CPU acceptance 构建带显式标记，安装运行时会拒绝把它当作真实 CANN 扩展。记录见 docs/cpp_bridge_verification.json，不能据此标记 PrivateUse1、H2D 或 AI Core 已通过。

原 assets 完全保留，因此其 golden 仍可能捕获错误并回退全局参考。独立验收工具 qfa_mock.attestation.GoldenAudit（同时随 wheel 配送为算子包内的 _qfa_metadata_audit.GoldenAudit）用执行 profile 记录原 decoder 的 metadata 摘要及原分段数值函数成功返回，不替换原函数；require_consumed() 可拒绝“TTK 返回了输出但分段实际失败”的记录。测试特别覆盖合法 metadata 配合数值异常的真实回退。它是独立验收手段，不改变生产 assets 的异常语义；已有其他 profiler 时需使用独立 worker。

目标仍需单独证明：包函数直调/PrivateUse1/扩展直调均命中同一包装器、原 metadata ACLNN 未执行、真实 QFA 有 AI Core 执行、长度复制与 stream 顺序正确、实际 tiling/有效核数一致、golden 消费相同 metadata 并成功完成分段、Out/LSE 精度通过。同配置 AICPU 对拍另记。

需要撤销开发 checkout 的接入时：

```bash
python3 scripts/apply_extension_patch.py --repo "$QFA_REPO" --reverse
```

此命令不卸载已经安装的 .run 或清空全局 Torch 缓存；更换部署在新 prefix/worker 验证，避免旧 provider 或旧 .so 混用。
