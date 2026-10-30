"""Adapters for the two existing assets entry points (small tensors only)."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from .model import MockError, effective_lengths, fail, normalize_request
from . import runtime_loader
from .telemetry import event, phase
from .worker import get_provider, _providers
from .requests import request_from_call, request_from_parameters, assert_same_request, _small_list


@contextmanager
def _replacements(module, values, hook):
    originals = {name: getattr(module, name) for name in values}
    try:
        with ExitStack() as stack:
            for name, value in values.items():
                stack.enter_context(patch.object(module, name, value))
            event("replacements_installed", hook=hook, names=list(values))
            yield
    finally:
        restored = all(getattr(module, name) is value for name, value in originals.items())
        event("replacements_restored", hook=hook, restored=restored, names=list(values))
        if not restored:
            fail("RESTORE_FAILED", f"{hook}: original function objects were not restored")


def decode_for_golden(result, request, module, q_lens, kv_lens):
    import struct

    if list(q_lens) != effective_lengths(request, "q") or list(kv_lens) != effective_lengths(request, "kv"):
        fail("REQUEST_MISMATCH", "golden effective lengths differ from original invocation")
    words = struct.unpack(f"<{len(result.data) // 4}I", result.data)
    return module._decode_qfa_sections(words, q_lens, kv_lens)


def metadata_for_preprocess(arguments, request, provider, anchor, result=None):
    adapted = {**arguments, "protocol_version": 1, "execution_mode": "one_shot",
               "win_left": request["win_left"], "win_right": request["win_right"],
               "is_grad_enabled": False}
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        adapted[name] = _small_list(adapted.get(name))
    assert_same_request(request, normalize_request(adapted))
    result = provider.generate(request) if result is None else result
    from .platform_check import check_device_target

    with phase("platform_check", hook="npu_preprocess"):
        check_device_target(provider, request, result, anchor)
    with phase("device_write", hook="npu_preprocess"):
        return result.device_tensor(anchor)


class _SectionFailure(BaseException):
    """Escape the original golden's `except Exception` reference fallback."""


def run_golden(function, module, args, kwargs):
    provider = get_provider()
    with phase("parameter_binding", hook="golden"):
        request, _ = request_from_call(function, args, kwargs, golden=True)
    with phase("plan_generation", hook="golden"):
        result = provider.generate(request)
    with runtime_loader.CALL_LOCK:
        sectioned = module._cpu_mxfp8_golden_sectioned
        state = {"plan_calls": 0, "decoded": False, "section_calls": 0, "section_success": False}

        def section_info(q_lens, kv_lens):
            with phase("decode", hook="golden"):
                state["plan_calls"] += 1
                if state["plan_calls"] != 1:
                    fail("GOLDEN_FAILED", "expected exactly one plan consumption")
                decoded = decode_for_golden(result, request, module, q_lens, kv_lens)
                if not decoded:
                    fail("GOLDEN_FAILED", "decoder returned an empty plan")
                state["decoded"] = True
                return decoded

        def strict_sectioned(*inner_args, **inner_kwargs):
            try:
                with phase("sectioned_compute", hook="golden"):
                    state["section_calls"] += 1
                    if state["section_calls"] != 1 or not state["decoded"]:
                        fail("GOLDEN_FAILED", "expected one sectioned computation after decoding")
                    output = sectioned(*inner_args, **inner_kwargs)
                    state["section_success"] = True
                    return output
            except Exception as error:
                raise _SectionFailure(error) from error

        try:
            with _replacements(module, {"get_qfa_section_info": section_info,
                                       "_cpu_mxfp8_golden_sectioned": strict_sectioned}, "golden"), \
                    phase("golden_output", hook="golden"):
                output = function(*args, **kwargs)
        except _SectionFailure as error:
            cause = error.__cause__
            converted = MockError("GOLDEN_FAILED", f"sectioned golden failed; fallback disabled: {cause}")
            converted.stage = getattr(cause, "stage", "sectioned_compute")
            raise converted from cause
        if state != {"plan_calls": 1, "decoded": True, "section_calls": 1, "section_success": True}:
            fail("GOLDEN_FAILED", f"golden did not complete the required sectioned path: {state}")
        event("golden_accepted", request_digest=result.summary["request_digest"],
              metadata_digest=result.summary["metadata_digest"], state=state)
        return output


def run_preprocess(module, args, kwargs):
    provider = get_provider()
    with phase("parameter_binding", hook="npu_preprocess"):
        request, _ = request_from_call(module.run, args, kwargs)
    with phase("plan_generation", hook="npu_preprocess"):
        result = provider.generate(request)
    calls = [0]

    def mock_metadata(arguments, anchor):
        calls[0] += 1
        if calls[0] != 1:
            fail("PREPROCESS_FAILED", "expected exactly one metadata production")
        return metadata_for_preprocess(arguments, request, provider, anchor, result=result)

    with runtime_loader.CALL_LOCK, _replacements(module, {"run_metadata": mock_metadata}, "npu_preprocess"), \
            phase("slot_processing", hook="npu_preprocess"):
        output = module.run(*args, **kwargs)
    if calls[0] != 1:
        fail("PREPROCESS_FAILED", "preprocess did not consume mock metadata")
    event("preprocess_accepted", request_digest=result.summary["request_digest"],
          metadata_digest=result.summary["metadata_digest"])
    return output
