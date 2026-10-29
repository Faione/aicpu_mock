"""Path-pinned upstream imports. No device queries or helper execution."""

import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import threading

from .build import ROOT
from .model import fail
from .telemetry import event, phase


CALL_LOCK = threading.RLock()
_checkout = None
_digests = {}


def _after_fork():
    global CALL_LOCK
    CALL_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_module(module, expected):
    expected = Path(expected).resolve()
    actual = getattr(module, "__file__", None)
    if actual is None or Path(actual).resolve() != expected:
        fail("MODULE_SOURCE_CONFLICT", f"{getattr(module, '__name__', module)}: expected {expected}, got {actual}")
    digest = _file_digest(expected)
    pinned = _digests.get(str(expected))
    if pinned is not None and pinned != digest:
        fail("SOURCE_CHANGED", f"loaded module source changed: {expected}; restart the worker")
    _digests[str(expected)] = digest
    event("module_verified", module=getattr(module, "__name__", None), path=str(expected), sha256=digest)
    return module


def verify_support_package():
    for name, module in list(sys.modules.items()):
        if name == "qfa_mock" or name.startswith("qfa_mock."):
            suffix = name.split(".")[1:]
            expected = ROOT / "qfa_mock" / ("/".join(suffix) + ".py" if suffix else "__init__.py")
            verify_module(module, expected)


def _check_cached_upstream(repo):
    assets = Path(repo) / "attention/quant_flash_attn/tests/assets"
    for name, module in list(sys.modules.items()):
        if name.startswith("qfa_assets_impl_"):
            verify_module(module, assets / "impl" / (name[len("qfa_assets_impl_"):] + ".py"))
        elif name.startswith("quant_flash_attn") and name.endswith("_golden"):
            verify_module(module, assets / (name + ".py"))


def require_signature(function, names, path=None):
    if not callable(function):
        fail("UPSTREAM_INCOMPATIBLE", f"expected callable with parameters {names}")
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError) as error:
        fail("UPSTREAM_INCOMPATIBLE", f"cannot inspect callable signature: {error}")
    missing = set(names) - set(parameters)
    if missing:
        fail("UPSTREAM_INCOMPATIBLE", f"{function.__name__}: missing parameters {sorted(missing)}")
    if path is not None:
        try:
            source = inspect.getsourcefile(function)
        except TypeError:
            source = None
        if source is None or Path(source).resolve() != Path(path).resolve():
            fail("MODULE_SOURCE_CONFLICT", f"{function.__name__} was defined outside {path}")


def load_original_spec(repo, plugin_path):
    global _checkout
    repo = Path(repo).resolve()
    with CALL_LOCK, phase("load_spec", repo=str(repo)):
        verify_support_package()
        if _checkout is not None and _checkout != repo:
            fail("MODULE_SOURCE_CONFLICT", f"this interpreter is bound to {_checkout}, requested {repo}")
        _check_cached_upstream(repo)
        path = repo / "attention/quant_flash_attn/tests/assets/spec.py"
        alias = "qfa_original_assets_spec_" + hashlib.sha256(str(path).encode()).hexdigest()[:20]
        module = sys.modules.get(alias)
        if module is None:
            spec = importlib.util.spec_from_file_location(alias, path)
            if spec is None or spec.loader is None:
                fail("UPSTREAM_INCOMPATIBLE", f"cannot load {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[alias]
                raise
        verify_module(module, path)
        if not isinstance(getattr(module, "QuantFlashAttnSpec", None), type):
            fail("UPSTREAM_INCOMPATIBLE", "original QuantFlashAttnSpec is missing")
        _check_cached_upstream(repo)
        _checkout = repo
        event("plugin_registered", plugin_path=str(Path(plugin_path).resolve()),
              plugin_sha256=_file_digest(Path(plugin_path)), original_spec=str(path),
              original_spec_sha256=_file_digest(path), alias=alias)
        return module


def load_impl(base, repo, stem):
    repo = Path(repo).resolve()
    assets = repo / "attention/quant_flash_attn/tests/assets"
    with CALL_LOCK, phase("load_implementation", implementation=stem):
        verify_support_package()
        verify_module(base, assets / "spec.py")
        _check_cached_upstream(repo)
        require_signature(getattr(base, "load_impl_module", None), ("stem",), assets / "spec.py")
        module = base.load_impl_module(stem)
        path = assets / "impl" / (stem + ".py")
        verify_module(module, path)
        suffix = "_t" if stem == "golden" else ""
        names = ("q", "k", "v", "quant_mode", "cu_seqlens_q" + suffix, "cu_seqlens_kv" + suffix,
                 "seqused_q" + suffix, "seqused_kv" + suffix, "mask_mode", "win_left", "win_right",
                 "max_seqlen_q", "max_seqlen_kv", "layout_q", "layout_q_descale", "layout_kv", "layout_out")
        if stem == "golden":
            require_signature(getattr(module, "cpu_qfa_mxfp8", None), names, path)
            numeric = getattr(module, "mxfp8_golden_mod", None)
            numeric_path = assets / "quant_flash_attn_golden.py"
            verify_module(numeric, numeric_path)
            if module.cpu_qfa_mxfp8.__globals__.get("mxfp8_golden_mod") is not numeric:
                fail("UPSTREAM_INCOMPATIBLE", "CPU golden references a different numeric module")
            require_signature(getattr(numeric, "get_qfa_section_info", None), ("actual_seq_q", "actual_seq_kv"), numeric_path)
            require_signature(getattr(numeric, "_decode_qfa_sections", None), ("meta_flat", "seqused_q", "seqused_kv"), numeric_path)
            require_signature(getattr(numeric, "_cpu_mxfp8_golden_sectioned", None),
                              ("q_fp8", "k_fp8", "v_fp8", "section_info"), numeric_path)
        elif stem == "npu_preprocess":
            require_signature(getattr(module, "run", None), names, path)
            require_signature(getattr(module, "run_metadata", None), ("arguments", "metadata"), path)
            if module.run.__globals__.get("run_metadata") is not module.run_metadata:
                fail("UPSTREAM_INCOMPATIBLE", "preprocess references a different run_metadata")
        else:
            fail("UPSTREAM_INCOMPATIBLE", f"unsupported implementation: {stem}")
        _check_cached_upstream(repo)
        return module
