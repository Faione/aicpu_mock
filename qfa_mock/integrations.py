"""Adapters for the two existing assets entry points (small tensors only)."""

import inspect
import os
from pathlib import Path
import threading
from unittest.mock import patch

from .build import ROOT
from .model import MockError, canonical, effective_lengths, fail, load_json, normalize_request
from .provider import MetadataProvider


_providers = {}
_provider_lock = threading.Lock()
_patch_lock = threading.RLock()


def get_provider():
    path = os.environ.get("QFA_METADATA_CONFIG")
    repo = os.environ.get("QFA_REPO")
    helper = os.environ.get("QFA_MOCK_HELPER", str(ROOT / "build/metadata_helper"))
    if not path or not repo:
        fail("MISSING_CONFIG", "TTK adapter requires QFA_METADATA_CONFIG and QFA_REPO")
    config = load_json(path)
    key = (os.getpid(), str(Path(repo).resolve()), str(Path(helper).resolve()), canonical(config))
    with _provider_lock:
        if key not in _providers:
            _providers.clear()
            _providers[key] = MetadataProvider(config, repo, helper)
        return _providers[key]


def _small_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "reshape"):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        fail("INVALID_INPUT", "length input must be a small tensor or array")
    return list(value) or None


def _nonempty(value):
    if value is None:
        return False
    if hasattr(value, "numel"):
        return value.numel() != 0
    if hasattr(value, "size") and not callable(value.size):
        return value.size != 0
    return bool(value)


def request_from_call(function, args, kwargs, golden=False):
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    p = dict(bound.arguments)
    extras = p.pop("kwargs", {})
    suffix = "_t" if golden else ""
    r = {"protocol_version": 1, "execution_mode": "one_shot"}
    for name in ("quant_mode", "mask_mode", "win_left", "win_right", "max_seqlen_q", "max_seqlen_kv",
                 "layout_q", "layout_q_descale", "layout_kv", "layout_out"):
        if name not in p:
            fail("MISSING_FIELD", f"call adapter requires explicit function parameter {name}")
        r[name] = p[name]
        alias = extras.get(f"pytest_{name}")
        if alias is not None and alias != p[name]:
            fail("REQUEST_MISMATCH", f"{name} differs between invocation and pytest sidecar")
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        live = _small_list(p.get(name + suffix))
        sidecar = _small_list(extras.get(name + "_values"))
        if live is not None and sidecar is not None and live != sidecar:
            fail("REQUEST_MISMATCH", f"{name} tensor differs from sidecar")
        r[name] = live if live is not None else sidecar
    # Match the native preprocessing branch: PA does not pass cu_seqlens_kv.
    if r["layout_kv"] == "PA_BNBD":
        r["cu_seqlens_kv"] = None
    q_shape, k_shape, v_shape = (tuple(p[name].shape) for name in ("q", "k", "v"))
    if len(q_shape) != 3 or len(k_shape) != (3 if r["layout_kv"] == "TND" else 4) or k_shape != v_shape:
        fail("UNSUPPORTED_REQUEST", "v1 requires TND Q and matching TND/PA_BNBD K/V shapes")
    r.update(num_heads_q=q_shape[1], num_heads_kv=k_shape[1], head_dim=q_shape[-1], head_dim_v=v_shape[-1])
    for name, alias in (("num_heads_q", "N_q"), ("num_heads_kv", "N_kv"), ("head_dim", "D"), ("head_dim_v", "head_dim_v")):
        for attr in {name, alias, f"pytest_{name}"}:
            if extras.get(attr) is not None and extras[attr] != r[name]:
                fail("REQUEST_MISMATCH", f"{attr} differs from input tensor shape")
    if k_shape[-1] != r["head_dim"]:
        fail("REQUEST_MISMATCH", "K head dimension differs from Q")
    for name, actual in (("input_layout", r["layout_q"]), ("q_scale_layout", r["layout_q_descale"]),
                         ("kv_cache_layout", r["layout_kv"])):
        if extras.get(name) and extras[name] != actual:
            fail("REQUEST_MISMATCH", f"{name} sidecar differs from actual invocation")
    if extras.get("graph_path", 0):
        fail("UNSUPPORTED_REQUEST", "v1 TTK adapter supports Eager only")
    for name in ("v_tail", "block_table_tail", "seqused_v_tail", "sinks" + suffix):
        if _nonempty(p.get(name)):
            fail("UNSUPPORTED_REQUEST", f"v1 does not support nonempty {name}")
    mask = p.get("attn_mask" + suffix)
    if _nonempty(mask):
        # Original customize_inputs fills this slot with a canonical causal mask.
        # metadata itself never reads its values; preserve the original data path.
        declared_shape = extras.get("attn_mask_shape")
        shape = tuple(mask.shape)
        if r["mask_mode"] != 3 or len(shape) != 2 or not declared_shape or tuple(declared_shape) != shape:
            fail("UNSUPPORTED_REQUEST", "a nonempty attn_mask requires mask_mode=3 and matching attn_mask_shape from the original assets generator")
    r["batch_size"] = extras.get("batch_size")
    r["is_grad_enabled"] = extras.get("is_grad_enabled", False)
    r = normalize_request(r)
    if q_shape[0] != r["cu_seqlens_q"][-1]:
        fail("REQUEST_MISMATCH", "Q token allocation differs from cu_seqlens_q")
    if r["layout_kv"] == "TND" and k_shape[0] != r["cu_seqlens_kv"][-1]:
        fail("REQUEST_MISMATCH", "K/V token allocation differs from cu_seqlens_kv")
    return r, p


def assert_same_request(expected, actual):
    differences = {name: {"expected": expected.get(name), "actual": actual.get(name)}
                   for name in expected.keys() | actual.keys() if expected.get(name) != actual.get(name)}
    if differences:
        fail("REQUEST_MISMATCH", canonical(differences))


def decode_for_golden(result, request, module, q_lens, kv_lens):
    import struct

    if list(q_lens) != effective_lengths(request, "q") or list(kv_lens) != effective_lengths(request, "kv"):
        fail("REQUEST_MISMATCH", "golden effective lengths differ from original invocation")
    words = struct.unpack(f"<{len(result.data) // 4}I", result.data)
    return module._decode_qfa_sections(words, q_lens, kv_lens)


def metadata_for_preprocess(arguments, request, provider, anchor):
    import torch

    adapted = {**arguments, "protocol_version": 1, "execution_mode": "one_shot",
               "win_left": request["win_left"], "win_right": request["win_right"],
               "is_grad_enabled": False}
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        adapted[name] = _small_list(adapted.get(name))
    assert_same_request(request, normalize_request(adapted))
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "get_device_properties"):
        fail("TARGET_UNVERIFIED", "device preprocessing requires runtime core information")
    props = npu.get_device_properties(anchor.device)
    actual = (int(props.cube_core_num), int(props.vector_core_num))
    target = provider.config["target"]
    expected = (target["aic_core_num"], target["aiv_core_num"])
    if actual != expected:
        fail("TARGET_MISMATCH", f"runtime AIC/AIV={actual}, configured={expected}")
    if hasattr(npu, "get_soc_version"):
        soc = str(npu.get_soc_version())
        if soc != target["soc_version"]:
            fail("TARGET_MISMATCH", f"runtime SoC={soc}, configured={target['soc_version']}")
    return provider.generate(request).device_tensor(anchor)


class _SectionFailure(BaseException):
    """Escape the original golden's `except Exception` reference fallback."""


def run_golden(function, module, args, kwargs):
    provider = get_provider()
    request, _ = request_from_call(function, args, kwargs, golden=True)
    result = provider.generate(request)
    # Read the original function under the same lock used for temporary replacement.
    with _patch_lock:
        sectioned = module._cpu_mxfp8_golden_sectioned
        called = [False]

        def section_info(q_lens, kv_lens):
            called[0] = True
            return decode_for_golden(result, request, module, q_lens, kv_lens)

        def strict_sectioned(*inner_args, **inner_kwargs):
            try:
                return sectioned(*inner_args, **inner_kwargs)
            except Exception as error:
                raise _SectionFailure(error) from error

        with patch.object(module, "get_qfa_section_info", section_info), \
                patch.object(module, "_cpu_mxfp8_golden_sectioned", strict_sectioned):
            try:
                output = function(*args, **kwargs)
            except _SectionFailure as error:
                raise MockError("GOLDEN_FAILED", f"sectioned golden failed; fallback disabled: {error}") from error
        if not called[0]:
            fail("GOLDEN_FAILED", "golden did not consume the generated metadata")
        return output


def run_preprocess(module, args, kwargs):
    provider = get_provider()
    request, _ = request_from_call(module.run, args, kwargs)
    provider.generate(request)  # Validate before entering the existing hook.

    def mock_metadata(arguments, anchor):
        return metadata_for_preprocess(arguments, request, provider, anchor)

    with _patch_lock, patch.object(module, "run_metadata", mock_metadata):
        return module.run(*args, **kwargs)
