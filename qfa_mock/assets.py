"""Small explicit adapter called by the maintained assets patch.

No original-module loading, function replacement, exception bridge or launch
receipt is needed on this path. The original Spec owns its normal imports.
"""

import os
import struct

from .worker import get_provider
from .requests import request_from_parameters, assert_same_request, _small_list
from .model import effective_lengths, fail, normalize_request
from .telemetry import event, phase


def request(parameters, golden=False):
    with phase("parameter_binding", hook="patched_golden" if golden else "patched_preprocess"):
        normalized, _ = request_from_parameters(parameters, golden=golden)
        return normalized


def golden_plan(current_request, decoder, q_lengths, kv_lengths):
    if list(q_lengths) != effective_lengths(current_request, "q") or list(kv_lengths) != effective_lengths(current_request, "kv"):
        fail("REQUEST_MISMATCH", "golden lengths differ from this invocation")
    provider = get_provider()
    result = provider.generate(current_request)
    with phase("decode", hook="patched_golden"):
        words = struct.unpack(f"<{len(result.data) // 4}I", result.data)
        decoded = decoder(words, q_lengths, kv_lengths)
        if not decoded:
            fail("GOLDEN_FAILED", "mock decoder returned an empty plan")
    event("patched_golden_plan", request_digest=result.summary["request_digest"],
          metadata_digest=result.summary["metadata_digest"])
    return decoded


def _check_available_device_capacity(target, anchor):
    """Check known conflicts, without claiming a physical count proves tiling."""
    import torch

    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "get_device_properties"):
        try:
            props = npu.get_device_properties(anchor.device)
            physical = (int(props.cube_core_num), int(props.vector_core_num))
        except Exception as error:
            event("target_query_unavailable", error_type=type(error).__name__)
        else:
            if physical[0] < target["aic_core_num"] or physical[1] < target["aiv_core_num"]:
                fail("TARGET_MISMATCH", f"configured effective cores exceed device capacity {physical}")
            event("target_capacity", physical=physical, configured=target)
    event("target_tiling_review_required", target=target,
          reason="configuration declares scheduling target; verify real qfa tiling and Out/LSE on first simulation")


def device_metadata(current_request, arguments, anchor):
    # Original build_metadata_arguments still owns native argument preparation.
    adapted = {**arguments, "protocol_version": 1, "execution_mode": "one_shot",
               "win_left": current_request["win_left"], "win_right": current_request["win_right"],
               "is_grad_enabled": False}
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        adapted[name] = _small_list(adapted.get(name))
    assert_same_request(current_request, normalize_request(adapted))
    provider = get_provider()
    result = provider.generate(current_request)
    # The old evidence path is optional diagnostics, never a required launcher.
    if os.environ.get("QFA_PLATFORM_EVIDENCE"):
        from .platform_check import check_device_target
        check_device_target(provider, current_request, result, anchor)
    else:
        _check_available_device_capacity(provider.config["target"], anchor)
    with phase("device_write", hook="patched_preprocess"):
        output = result.device_tensor(anchor)
    event("patched_preprocess_plan", request_digest=result.summary["request_digest"],
          metadata_digest=result.summary["metadata_digest"])
    return output
