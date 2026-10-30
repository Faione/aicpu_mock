"""Normalize actual QFA call parameters without loading original modules."""

import inspect

from .model import canonical, fail, normalize_request


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
    return request_from_parameters(dict(bound.arguments), golden=golden)


def request_from_parameters(parameters, golden=False):
    """Adapt an explicit function's locals without inspecting/replacing the function."""
    p = dict(parameters)
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
