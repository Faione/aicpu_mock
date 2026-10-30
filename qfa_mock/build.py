"""Compile the Host helper; fingerprint only its actual compiler dependencies."""

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import shlex

from .model import ALGORITHM_VERSION, PROTOCOL_VERSION, canonical, fail, load_json


ROOT = Path(__file__).resolve().parents[1]
HELPER_SOURCE = ROOT / "cpp/metadata_helper.cpp"


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
    output.parent.mkdir(parents=True, exist_ok=True)
    # Compile to a private path so failed builds cannot damage a working executable.
    with tempfile.TemporaryDirectory(prefix="qfa-build-", dir=output.parent) as directory:
        compiled = Path(directory) / "metadata_helper"
        depfile = Path(directory) / "metadata_helper.d"
        command = [compiler, "-O2", "-std=c++17", "-I", str(repo), "-MMD", "-MF", str(depfile),
                   "-MT", "qfa_metadata_helper", str(HELPER_SOURCE), "-o", str(compiled)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            fail("BUILD_FAILED", "compiler exceeded 120 seconds")
        if result.returncode:
            fail("BUILD_FAILED", result.stderr.strip())
        # No Git/full-tree preflight: the compiler decides which sources are required.
        entries = shlex.split(depfile.read_text().partition(":")[2].replace("\\\n", " "))
        dependencies = []
        for entry in entries:
            path = Path(os.path.abspath(entry))
            if path.resolve() == HELPER_SOURCE.resolve():
                dependencies.append("@helper")
            else:
                try:
                    dependencies.append(path.relative_to(repo).as_posix())
                except ValueError:
                    # Some Clang SDKs also list SDKSettings.json under -MMD.
                    # Toolchain-managed files are not project source dependencies.
                    continue
        dependencies = sorted(set(dependencies))
        manifest = {"build_format": 2, "protocol_version": PROTOCOL_VERSION, "algorithm_version": ALGORITHM_VERSION,
                    "repo": str(repo), "dependency_files": dependencies,
                    "source_digest": source_digest(repo, dependencies), "binary_digest": binary_digest(compiled),
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
                "build_format": 2,
                "source_digest": source_digest(repo, manifest.get("dependency_files", [])), "binary_digest": binary_digest(helper)}
    for field, value in expected.items():
        if manifest.get(field) != value:
            fail("STALE_BUILD", f"helper {field} differs; rebuild explicitly")
    return manifest
