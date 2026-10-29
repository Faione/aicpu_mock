"""Explicit pre-test compilation and source identity tracking."""

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from .model import ALGORITHM_VERSION, PROTOCOL_VERSION, canonical, fail, load_json


ROOT = Path(__file__).resolve().parents[1]
HELPER_SOURCE = ROOT / "cpp/metadata_helper.cpp"


def source_digest(repo):
    repo = Path(repo).resolve()
    balance = repo / "attention/common/op_kernel/load_balance"
    paths = sorted(balance.rglob("*.h"))
    paths += [repo / "attention/quant_flash_attn/op_host/qfa_adjust_sinner_souter.h",
              repo / "attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata.h",
              repo / "attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp",
              repo / "attention/quant_flash_attn/tests/assets/quant_flash_attn_golden.py"]
    if not balance.is_dir() or any(not p.is_file() for p in paths):
        fail("SOURCE_MISSING", f"incomplete ops-transformer source tree: {repo}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repo)).encode() + b"\0" + path.read_bytes() + b"\0")
    digest.update(b"metadata_helper.cpp\0" + HELPER_SOURCE.read_bytes())
    return digest.hexdigest()


def binary_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_helper(repo, output, compiler="c++"):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    before = source_digest(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Compile to a private path so failed builds cannot damage a working executable.
    with tempfile.TemporaryDirectory(prefix="qfa-build-", dir=output.parent) as directory:
        compiled = Path(directory) / "metadata_helper"
        command = [compiler, "-O2", "-std=c++17", "-I", str(repo), str(HELPER_SOURCE), "-o", str(compiled)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            fail("BUILD_FAILED", "compiler exceeded 120 seconds")
        if result.returncode:
            fail("BUILD_FAILED", result.stderr.strip())
        if before != source_digest(repo):
            fail("SOURCE_CHANGED", "source changed during compilation; build again")
        manifest = {"protocol_version": PROTOCOL_VERSION, "algorithm_version": ALGORITHM_VERSION,
                    "repo": str(repo), "source_digest": before, "binary_digest": binary_digest(compiled),
                    "compiler": compiler}
        os.replace(compiled, output)
        manifest_path = Path(str(output) + ".json")
        staged = Path(directory) / "manifest.json"
        staged.write_text(canonical(manifest) + "\n", encoding="utf-8")
        os.replace(staged, manifest_path)
    return manifest


def verify_build(helper, repo):
    helper = Path(helper).resolve()
    if not helper.is_file() or not Path(str(helper) + ".json").is_file():
        fail("HELPER_NOT_BUILT", "run python3 -m qfa_mock build before generating metadata")
    manifest = load_json(str(helper) + ".json")
    expected = {"protocol_version": PROTOCOL_VERSION, "algorithm_version": ALGORITHM_VERSION,
                "source_digest": source_digest(repo), "binary_digest": binary_digest(helper)}
    for field, value in expected.items():
        if manifest.get(field) != value:
            fail("STALE_BUILD", f"helper {field} differs; rebuild explicitly")
    return manifest
