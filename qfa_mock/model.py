"""Versioned JSON contract; no Torch, CANN, or device queries at import time."""

import hashlib
import json


PROTOCOL_VERSION = 1
ALGORITHM_VERSION = "qfa-host-v1"
MAX_METADATA_BYTES = 256 * 1024 * 1024


class MockError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def fail(code, message):
    raise MockError(code, message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def plan_digest(config, request, source):
    identity = {"target": config["target"], "request": request,
                "algorithm_version": ALGORITHM_VERSION, "source_digest": source}
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail("INVALID_INPUT", f"duplicate JSON field: {key}")
            result[key] = value
        return result

    with open(path, encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique)


def fields(value, allowed, required, label):
    if not isinstance(value, dict):
        fail("INVALID_INPUT", f"{label} must be an object")
    if set(value) - set(allowed):
        fail("UNSUPPORTED_FIELD", f"{label}: unknown fields {sorted(set(value) - set(allowed))}")
    if set(required) - set(value):
        fail("MISSING_FIELD", f"{label}: missing fields {sorted(set(required) - set(value))}")


def integer(value, label, minimum=0, maximum=2**31 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        fail("INVALID_INPUT", f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def normalize_config(config):
    fields(config, ("backend", "target", "cache"), ("backend", "target"), "config")
    if config["backend"] != "mock_aicpu":
        fail("UNSUPPORTED_MODE", "this prototype requires backend=mock_aicpu")
    t = dict(config["target"]) if isinstance(config["target"], dict) else config["target"]
    keys = ("name", "soc_version", "kernel_arch", "aic_core_num", "aiv_core_num")
    fields(t, keys, keys, "target")
    for key in ("name", "soc_version", "kernel_arch"):
        if not isinstance(t[key], str) or not t[key].strip():
            fail("INVALID_INPUT", f"target.{key} must be a nonempty string")
    aic = integer(t["aic_core_num"], "aic_core_num", 1, 1024)
    aiv = integer(t["aiv_core_num"], "aiv_core_num", 1, 2048)
    # Same SoC branch as the current production ParamsInit; never infer from name/cores.
    if "Ascend960" in t["soc_version"]:
        arch, ratio = "arch92", 1
    elif t["soc_version"].startswith("Ascend910_95"):
        arch, ratio = "arch35", 2
    else:
        fail("UNSUPPORTED_TARGET", "v1 accepts Ascend960* / arch92 or Ascend910_95* / arch35; confirm the real target")
    if t["kernel_arch"] != arch or aiv != ratio * aic:
        fail("TARGET_MISMATCH", f"{t['soc_version']} requires {arch} with AIV={ratio}*AIC in this prototype")
    cache = config.get("cache", {})
    fields(cache, ("enabled", "max_entries"), (), "cache")
    enabled = cache.get("enabled", True)
    if type(enabled) is not bool:
        fail("INVALID_INPUT", "cache.enabled must be boolean")
    return {"backend": "mock_aicpu", "target": t,
            "cache": {"enabled": enabled, "max_entries": integer(cache.get("max_entries", 8), "cache.max_entries", 1, 128)}}


def _array(value, name, cumulative=False):
    if value is None or value == []:
        return None
    if not isinstance(value, list):
        fail("INVALID_INPUT", f"{name} must be an integer array or null")
    result = [integer(x, name) for x in value]
    if cumulative and (len(result) < 2 or result[0] != 0 or any(a > b for a, b in zip(result, result[1:]))):
        fail("INVALID_INPUT", f"{name} must start at zero, be nondecreasing, and contain B+1 entries")
    return result


def normalize_request(request):
    allowed = ("protocol_version", "execution_mode", "num_heads_q", "num_heads_kv", "head_dim", "head_dim_v",
               "batch_size", "cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv", "max_seqlen_q",
               "max_seqlen_kv", "quant_mode", "mask_mode", "win_left", "win_right", "layout_q",
               "layout_q_descale", "layout_kv", "layout_out", "is_grad_enabled")
    required = ("protocol_version", "execution_mode", "num_heads_q", "num_heads_kv", "head_dim",
                "cu_seqlens_q", "max_seqlen_q", "max_seqlen_kv", "layout_kv")
    fields(request, allowed, required, "request")
    r = dict(request)
    fixed = {"protocol_version": 1, "execution_mode": "one_shot", "head_dim": 128,
             "head_dim_v": 128, "quant_mode": 1, "win_left": -1, "win_right": -1,
             "layout_q": "TND", "layout_q_descale": "TND", "layout_out": "TND", "is_grad_enabled": False}
    for key, expected in fixed.items():
        value = r.get(key, expected)
        if value is None and key == "head_dim_v":
            value = 128
        if type(value) is not type(expected) or value != expected:
            fail("UNSUPPORTED_REQUEST", f"v1 requires {key}={expected!r}, got {value!r}")
        r[key] = value
    if r["layout_kv"] not in ("TND", "PA_BNBD"):
        fail("UNSUPPORTED_REQUEST", "v1 supports layout_kv=TND or PA_BNBD")
    r["mask_mode"] = integer(r.get("mask_mode", 3), "mask_mode", 0, 5)
    if r["mask_mode"] not in (0, 3, 5):
        fail("UNSUPPORTED_REQUEST", "v1 supports mask_mode=0,3,5 (original decoder supported modes)")
    nq = integer(r["num_heads_q"], "num_heads_q", 1, 1024)
    nk = integer(r["num_heads_kv"], "num_heads_kv", 1, nq)
    if nq % nk:
        fail("INVALID_INPUT", "num_heads_q must be divisible by num_heads_kv")
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        r[name] = _array(r.get(name), name, name.startswith("cu_"))
    if r["cu_seqlens_q"] is None:
        fail("MISSING_FIELD", "TND requires cu_seqlens_q")
    batch = len(r["cu_seqlens_q"]) - 1
    integer(batch, "derived batch_size", 1, 4096)
    explicit = r.get("batch_size")
    if explicit is not None and integer(explicit, "batch_size", 1, 4096) != batch:
        fail("INVALID_INPUT", "batch_size differs from cu_seqlens_q")
    r["batch_size"] = batch
    if r["layout_kv"] == "TND" and r["cu_seqlens_kv"] is None:
        fail("MISSING_FIELD", "TND KV requires cu_seqlens_kv")
    if r["layout_kv"] == "PA_BNBD":
        if r["cu_seqlens_kv"] is not None:
            fail("UNSUPPORTED_REQUEST", "PA_BNBD uses seqused_kv, not cu_seqlens_kv")
        if r["seqused_kv"] is None:
            fail("MISSING_FIELD", "PA_BNBD requires explicit seqused_kv")
    for side in ("q", "kv"):
        maximum = integer(r[f"max_seqlen_{side}"], f"max_seqlen_{side}", 1)
        cu, used = r[f"cu_seqlens_{side}"], r[f"seqused_{side}"]
        if cu is not None and len(cu) != batch + 1:
            fail("INVALID_INPUT", f"cu_seqlens_{side} must have B+1 entries")
        if used is not None and len(used) != batch:
            fail("INVALID_INPUT", f"seqused_{side} must have B entries")
        allocated = [b - a for a, b in zip(cu, cu[1:])] if cu else [maximum] * batch
        effective = used if used is not None else allocated
        if any(u > a for u, a in zip(effective, allocated)) or max(allocated) > maximum:
            fail("INVALID_INPUT", f"{side}: require effective length <= allocated length <= max_seqlen; v1 requires explicit positive maxima")
    return r


def effective_lengths(request, side):
    used, cu = request[f"seqused_{side}"], request[f"cu_seqlens_{side}"]
    return used if used is not None else [b - a for a, b in zip(cu, cu[1:])]


def row_size(config, request):
    t = config["target"]
    size = 16 + (t["aic_core_num"] + t["aiv_core_num"]) * 16 * request["batch_size"] * request["num_heads_q"]
    row = (size + 4095) // 4096 * 4096
    if row * 8 > MAX_METADATA_BYTES:
        fail("CAPACITY_EXCEEDED", "metadata exceeds the v1 256 MiB limit")
    return row


def wire_request(config, request):
    t, r = config["target"], request
    scalars = [int(t["kernel_arch"] == "arch92"), t["aic_core_num"], t["aiv_core_num"],
               r["batch_size"], r["num_heads_q"], r["num_heads_kv"], r["head_dim"], r["head_dim_v"],
               r["max_seqlen_q"], r["max_seqlen_kv"], r["mask_mode"], int(r["layout_kv"] == "PA_BNBD")]
    lines = ["QFA_HOST_V1", " ".join(map(str, scalars))]
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        values = r[name]
        lines.append("-1" if values is None else " ".join(map(str, [len(values)] + values)))
    return "\n".join(lines) + "\n"
