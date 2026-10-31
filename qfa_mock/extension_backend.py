"""Installed as an operator-local module; no dependency on the development repo."""
import hashlib
import json
import logging
import os
import re
from pathlib import Path

_state = None
_pid = None
_binary = None


def _configuration():
    mode, path = os.environ.get("QFA_METADATA_BACKEND", ""), os.environ.get("QFA_METADATA_CONFIG", "")
    if mode not in ("", "native", "mock_aicpu"):
        raise ValueError("invalid QFA_METADATA_BACKEND")
    if mode == "native" and path:
        raise ValueError("native backend conflicts with mock configuration")
    enabled = mode == "mock_aicpu" or (not mode and bool(path))
    if not enabled:
        return None
    if not path:
        raise ValueError("mock backend requires QFA_METADATA_CONFIG")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate configuration field: " + key)
            result[key] = value
        return result
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream, object_pairs_hook=unique)
    if set(config) - {"backend", "target", "cache"} or config.get("backend") != "mock_aicpu":
        raise ValueError("unsupported mock configuration")
    target = config.get("target")
    expected = {"name", "soc_version", "kernel_arch", "aic_core_num", "aiv_core_num"}
    if not isinstance(target, dict) or set(target) != expected:
        raise ValueError("mock target requires exactly " + ", ".join(sorted(expected)))
    if any(not isinstance(target[k], str) or not target[k].strip() for k in ("name", "soc_version", "kernel_arch")):
        raise ValueError("invalid mock target identity")
    arch = "arch92" if "Ascend960" in target["soc_version"] else "arch35" if target["soc_version"].startswith("Ascend910_95") else None
    aic, aiv = target["aic_core_num"], target["aiv_core_num"]
    if arch != target["kernel_arch"] or type(aic) is not int or type(aiv) is not int or not 1 <= aic <= 1024 or aiv != aic * (1 if arch == "arch92" else 2):
        raise ValueError("unsupported mock architecture/core configuration")
    release_identity = None
    release_path = os.environ.get("QFA_MOCK_RELEASE")
    if release_path:
        with open(release_path, encoding="utf-8") as stream:
            release = json.load(stream)
        release_identity = hashlib.sha256(json.dumps(release, sort_keys=True).encode()).hexdigest()
        if release["opp"]["kernel_arch"] != arch:
            raise ValueError("mock target differs from the installed QFA kernel architecture")
    cache = config.get("cache", {})
    if not isinstance(cache, dict) or set(cache) - {"enabled", "max_entries"}:
        raise ValueError("unsupported cache configuration")
    if type(cache.get("enabled", True)) is not bool or type(cache.get("max_entries", 8)) is not int or not 1 <= cache.get("max_entries", 8) <= 128:
        raise ValueError("invalid cache configuration")
    # The extension prototype deliberately does not cache plans.
    return {"target": target, "release_identity": release_identity, "cache": {"enabled": cache.get("enabled", True), "max_entries": cache.get("max_entries", 8)}}


def _freeze():
    global _pid, _state, _binary
    config = _configuration()
    identity = json.dumps(config, sort_keys=True, separators=(",", ":"))
    if _pid != os.getpid():
        _pid, _state, _binary = os.getpid(), (identity, config), None
    elif _state[0] != identity:
        raise ValueError("worker backend/target/configuration changed; restart")
    return config


def _data_root():
    return Path(__file__).resolve().parents[3] / "csrc/attention/qfa_mock"


def verify_payload(root=None):
    root = Path(root) if root is not None else _data_root()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != 1 or manifest.get("mode") != "extension_cpp":
        raise ValueError("invalid extension mock manifest")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest["runtime_sha256"]:
        raise ValueError("extension runtime changed; rebuild package")
    if hashlib.sha256(Path(__file__).with_name("_qfa_metadata_audit.py").read_bytes()).hexdigest() != manifest["audit_sha256"]:
        raise ValueError("extension audit module changed; rebuild package")
    cpp = root.parent / "quant_flash_attn.cpp"
    if hashlib.sha256(cpp.read_bytes()).hexdigest() != manifest["entry_cpp_sha256"]:
        raise ValueError("extension binding source changed; rebuild package")
    python = Path(__file__).with_name("quant_flash_attn.py").read_text().replace("\r\n", "\n")
    python = re.sub(r"\b(from|import) cann_ops_transformer_[A-Za-z0-9_]+\.op_builder", r"\1 cann_ops_transformer.op_builder", python)
    if hashlib.sha256(python.encode()).hexdigest() != manifest["entry_python_sha256"]:
        raise ValueError("extension Python entry changed; rebuild package")
    for name, expected in manifest["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError("extension payload changed: " + name)
    if (root.parents[2] / "ops").is_dir():
        framework = root.parents[2].parent / "cann_ops_transformer"
        for name, expected in manifest["framework_files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or hashlib.sha256((framework / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("base loader framework changed: " + name)
    return manifest


def prepare_metadata(builder, batch, nq, native_allocation):
    global _binary
    config = _freeze()
    if config is None:
        return native_allocation(batch, nq)
    manifest = verify_payload()
    module = builder.load()
    info = module._qfa_mock_info()
    if info["build_id"] != manifest["build_id"] or info["cpu_acceptance_only"]:
        raise ValueError("loaded extension is stale or is CPU acceptance only; rebuild/restart")
    path = Path(module.__file__).resolve()
    binary = (str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    if _binary is not None and _binary != binary:
        raise ValueError("loaded extension binary changed; restart")
    _binary = binary
    target = config["target"]
    module._qfa_mock_configure(True, target["kernel_arch"] == "arch92", target["aic_core_num"], target["aiv_core_num"], manifest["build_id"])
    row = module._qfa_mock_row_size(batch, nq)
    logging.getLogger("qfa_mock.extension").info("extension_cpp metadata target=%s build=%s row=%s module=%s", target, manifest["build_id"], row, getattr(module, "__file__", "unknown"))
    return row


def validate_qfa_call(parameters):
    if _freeze() is None:
        return
    p = parameters
    for key, expected in (("quant_mode", 1), ("layout_q", "TND"), ("layout_q_descale", "TND"),
                          ("layout_out", "TND"), ("win_left", -1), ("win_right", -1)):
        if p[key] != expected:
            raise ValueError("unsupported QFA mock parameter: " + key)
    if p["layout_kv"] not in ("TND", "PA_BNBD") or p["mask_mode"] not in (0, 3, 5):
        raise ValueError("unsupported QFA mock layout/mask")
    for name in ("sinks", "v_tail", "block_table_tail", "seqused_v_tail"):
        if p.get(name) is not None and p[name].numel():
            raise ValueError("unsupported QFA mock input: " + name)
    for name in ("max_seqlen_q", "max_seqlen_kv"):
        if type(p[name]) is not int or p[name] <= 0:
            raise ValueError("mock requires explicit positive " + name)
    q, k, v = p["q"], p["k"], p["v"]
    if q.dim() != 3 or q.shape[-1] != 128 or k.shape != v.shape or k.dim() != (3 if p["layout_kv"] == "TND" else 4) or k.shape[-1] != 128:
        raise ValueError("unsupported QFA mock tensor shape")
