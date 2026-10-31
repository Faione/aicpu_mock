"""Compile the Host helper; fingerprint only its actual compiler dependencies."""

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import shlex

from .model import ALGORITHM_VERSION, PROTOCOL_VERSION, MockError, canonical, fail, load_json


ROOT = Path(__file__).resolve().parents[1]
HELPER_SOURCE = ROOT / "cpp/metadata_helper.cpp"
CORE_SOURCE = ROOT / "cpp/metadata_core.cpp"

# These checks describe readiness, not Git identity or compiler fingerprints.
COMPILE_SOURCES = (
    "attention/common/op_kernel/load_balance/base_info.h",
    "attention/common/op_kernel/load_balance/load_balance_common.h",
    "attention/common/op_kernel/load_balance/section_stream_k/section_stream_k.h",
    "attention/common/op_kernel/load_balance/section_stream_k/section_stream_k_def.h",
    "attention/common/op_kernel/load_balance/section_stream_k/section_stream_k_impl.h",
    "attention/quant_flash_attn/op_host/qfa_adjust_sinner_souter.h",
    "attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata.h",
)
ASSET_SOURCES = tuple("attention/quant_flash_attn/tests/assets/" + name for name in (
    "spec.py", "impl/golden.py", "impl/npu_preprocess.py", "impl/inputs.py", "impl/compare.py",
    "impl/graph.py", "impl/metadata_inputs.py", "quant_flash_attn_golden.py",
    "quant_flash_attn_fp8_golden.py", "quant_flash_attn_hif8_golden.py", "quant_flash_attn_hifp4_golden.py",
))
AICPU_SOURCES = ("attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp",)


def check_sources(repo):
    """List every missing expected file, with purpose and absolute location."""
    repo = Path(repo).resolve()
    groups = [("compile", "Host C++ compilation", COMPILE_SOURCES),
              ("assets", "TTK assets, CPU golden and decoder", ASSET_SOURCES),
              ("aicpu_reference", "reference for metadata wrapper behavior", AICPU_SOURCES)]
    missing = [{"group": group, "relative_path": name, "path": str(repo / name), "required_for": purpose}
               for group, purpose, names in groups for name in names if not (repo / name).is_file()]
    if missing:
        lines = [f"Missing {len(missing)} required source file(s) under {repo}:"]
        lines += [f"[{item['group']}] {item['path']} ({item['required_for']})" for item in missing]
        lines += ["--repo must point to the source root containing attention/."]
        error = MockError("SOURCE_MISSING", "\n".join(lines))
        error.missing = missing
        error.repo = str(repo)
        raise error


def source_digest(repo, dependencies=None):
    repo = Path(repo).resolve()
    if dependencies is None:
        manifest = ROOT / "build/metadata_helper.json"
        if not manifest.is_file():
            fail("HELPER_NOT_BUILT", "build the helper or provide its dependency list")
        dependencies = load_json(manifest).get("dependency_files")
    if not isinstance(dependencies, list) or not dependencies or not all(isinstance(p, str) for p in dependencies):
        fail("STALE_BUILD", "helper has no compiler dependency list; rebuild it")
    digest = hashlib.sha256()
    for name in sorted(dependencies):
        if name == "@helper":
            path = HELPER_SOURCE
        elif name.startswith("@mock/") and ".." not in Path(name).parts:
            path = ROOT / name[len("@mock/"):]
        else:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                fail("STALE_BUILD", f"invalid compiler dependency: {name}")
            path = repo / relative
        if not path.is_file():
            fail("SOURCE_MISSING", f"missing compiled dependency: {path}")
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def binary_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_helper(repo, output, compiler="c++"):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    check_sources(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Compile to a private path so failed builds cannot damage a working executable.
    with tempfile.TemporaryDirectory(prefix="qfa-build-", dir=output.parent) as directory:
        compiled = Path(directory) / "metadata_helper"
        objects, entries = [], []
        for source in (HELPER_SOURCE, CORE_SOURCE):
            obj = Path(directory) / (source.stem + ".o")
            depfile = Path(directory) / (source.stem + ".d")
            command = [compiler, "-O2", "-std=c++17", "-I", str(repo), "-I", str(ROOT / "cpp"),
                       "-MMD", "-MF", str(depfile), "-c", str(source), "-o", str(obj)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                fail("BUILD_FAILED", "compiler exceeded 120 seconds")
            if result.returncode:
                fail("BUILD_FAILED", result.stderr.strip())
            entries += shlex.split(depfile.read_text().partition(":")[2].replace("\\\n", " "))
            objects.append(str(obj))
        try:
            result = subprocess.run([compiler, *objects, "-o", str(compiled)], capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            fail("BUILD_FAILED", "linker exceeded 120 seconds")
        if result.returncode:
            fail("BUILD_FAILED", result.stderr.strip())
        # Keep compile dependencies separate from semantic compatibility evidence.
        dependencies = ["@helper", "@mock/cpp/metadata_core.cpp"]
        for entry in entries:
            path = Path(os.path.abspath(entry))
            if path.resolve() == HELPER_SOURCE.resolve():
                dependencies.append("@helper")
            elif path.is_relative_to(ROOT / "cpp"):
                dependencies.append("@mock/" + path.relative_to(ROOT).as_posix())
            else:
                try:
                    dependencies.append(path.relative_to(repo).as_posix())
                except ValueError:
                    # Some Clang SDKs also list SDKSettings.json under -MMD.
                    # Toolchain-managed files are not project source dependencies.
                    continue
        dependencies = sorted(set(dependencies))
        from .extension import compatibility
        behavior = compatibility(repo)
        manifest = {"build_format": 3, "behavior_files": behavior["files"], "adapter_version": behavior["adapter_version"], "behavior_digest": hashlib.sha256(canonical(behavior).encode()).hexdigest(), "protocol_version": PROTOCOL_VERSION, "algorithm_version": ALGORITHM_VERSION,
                    "repo": str(repo), "dependency_files": dependencies,
                    "source_digest": source_digest(repo, dependencies), "binary_digest": binary_digest(compiled),
                    "compiler": compiler}
        manifest_path = Path(str(output) + ".json")
        staged = Path(directory) / "manifest.json"
        staged.write_text(canonical(manifest) + "\n", encoding="utf-8")
        os.replace(compiled, output)
        os.replace(staged, manifest_path)
    return manifest


def verify_build(helper, repo):
    helper = Path(helper).resolve()
    if not helper.is_file() or not Path(str(helper) + ".json").is_file():
        fail("HELPER_NOT_BUILT", "run python3 -m qfa_mock build before generating metadata")
    manifest = load_json(str(helper) + ".json")
    expected = {"protocol_version": PROTOCOL_VERSION, "algorithm_version": ALGORITHM_VERSION,
                "build_format": 3,
                "source_digest": source_digest(repo, manifest.get("dependency_files", [])), "binary_digest": binary_digest(helper)}
    from .extension import compatibility
    behavior = compatibility(repo)
    expected.update(behavior_files=behavior["files"], adapter_version=behavior["adapter_version"],
                    behavior_digest=hashlib.sha256(canonical(behavior).encode()).hexdigest())
    for field, value in expected.items():
        if manifest.get(field) != value:
            fail("STALE_BUILD", f"helper {field} differs; rebuild explicitly")
    return manifest
