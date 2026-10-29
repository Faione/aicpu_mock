"""Check the wire protocol and reuse the original golden's coverage decoder."""

import ast
from collections import defaultdict
import logging
from pathlib import Path
import struct

from .model import ALGORITHM_VERSION, PROTOCOL_VERSION, effective_lengths, fail, row_size


def _check(ok, message):
    if not ok:
        fail("INVALID_METADATA", message)


def load_decoder(repo, request):
    path = Path(repo) / "attention/quant_flash_attn/tests/assets/quant_flash_attn_golden.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"_decode_qfa_sections", "_qfa_calc_s2_block_range"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    _check({n.name for n in nodes} == wanted, "original decoder functions missing")
    namespace = {"N_q": request["num_heads_q"], "N_kv": request["num_heads_kv"],
                 "SPARSE_MODE": request["mask_mode"], "logger": logging.getLogger("qfa_mock.decoder"),
                 "resolve_q_scale_layout": lambda: (request["layout_q_descale"], request["num_heads_q"] // request["num_heads_kv"])}
    # Only these two pure functions are loaded; do not import Torch/CANN/golden globals.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_decode_qfa_sections"]


def expected_need_init(request):
    q_lens, kv_lens = effective_lengths(request, "q"), effective_lengths(request, "kv")
    q_cu, kv_cu = request["cu_seqlens_q"], request["cu_seqlens_kv"]
    for i, (q, kv) in enumerate(zip(q_lens, kv_lens)):
        allocated = q_cu[i + 1] - q_cu[i]
        if allocated == 0 or q == 0 or q < allocated or kv == 0:
            return 1
        if kv_cu is not None and kv_cu[i + 1] == kv_cu[i]:
            return 1
        if request["mask_mode"] == 3 and q > kv:
            return 1
    return 0


def validate_metadata(data, summary, config, request, repo):
    row = row_size(config, request)
    _check(len(data) == row * 8, "metadata byte length does not match target allocation")
    words = struct.unpack(f"<{row * 2}I", data)
    header = words[:16]
    sections, is_fd, m_base, s2_base, aic, aiv = header[:6]
    target = config["target"]
    if "target" in summary:
        _check(summary["target"] == target, "summary target differs from configured target")
    _check(summary.get("protocol_version") == PROTOCOL_VERSION and summary.get("algorithm_version") == ALGORITHM_VERSION,
           "helper protocol/algorithm version mismatch")
    _check(summary.get("shape") == [2, row] and summary.get("header") == list(header), "helper summary differs from binary")
    _check((aic, aiv) == (target["aic_core_num"], target["aiv_core_num"]), "header cores differ from target")
    _check(sections > 0 and m_base > 0 and s2_base > 0 and is_fd in (0, 1), "invalid header")
    _check(all(v == 0 for v in header[6:15]), "reserved header fields must be zero")
    _check(header[15] == expected_need_init(request), "incorrect needInitOutput")
    used = 16 + sections * (aic + aiv) * 16
    _check(used <= row and summary.get("used_words") == used, "invalid schedule capacity or used length")
    _check(not any(words[used:]), "unused capacity and backward row must be zero")
    active_aic, active_aiv = summary.get("active_aic", []), summary.get("active_aiv", [])
    _check(len(active_aic) == len(active_aiv) == sections, "missing per-section core counts")
    q_lens, kv_lens = effective_lengths(request, "q"), effective_lengths(request, "kv")
    head_count = request["num_heads_q"]
    bn_total = len(q_lens) * head_count
    fd_base = 16 + sections * aic * 16
    fd_tasks = defaultdict(list)
    fd_workspaces = {}
    previous_end = (0, 0, 0)
    for section in range(sections):
        _check(type(active_aic[section]) is int and 1 <= active_aic[section] <= aic, "invalid active AIC count")
        _check(type(active_aiv[section]) is int and 0 <= active_aiv[section] <= aiv, "invalid active AIV count")
        for core in range(aic):
            base = 16 + (section * aic + core) * 16
            record = words[base:base + 16]
            _check(not any(record[7:]), "reserved FA words must be zero")
            if core >= active_aic[section]:
                _check(not any(record), "unused FA slot must be zero")
                continue
            _check(record[:3] == previous_end, "FA chunks must chain across cores and sections")
            for bn, mi, s2 in (record[:3], record[3:6]):
                _check(bn <= bn_total, "FA bn out of range")
                if bn == bn_total:
                    _check(mi == s2 == 0, "invalid terminal FA endpoint")
                else:
                    b = bn // head_count
                    _check(mi <= (q_lens[b] + m_base - 1) // m_base, "FA m out of range")
                    _check(s2 <= (kv_lens[b] + s2_base - 1) // s2_base, "FA s2 out of range")
            _check(record[:3] <= record[3:6], "FA endpoints must be ordered")
            _check(record[6] < 2 * aic, "FA workspace index out of range")
            previous_end = record[3:6]
        for core in range(aiv):
            base = fd_base + (section * aiv + core) * 16
            record = words[base:base + 16]
            _check(not any(record[6:]), "reserved FD words must be zero")
            if core >= active_aiv[section]:
                _check(not any(record), "unused FD slot must be zero")
                continue
            bn, mi, workspace, parts, start, count = record[:6]
            _check(bn < bn_total and parts >= 2 and count > 0, "invalid FD task")
            rows = min(m_base, q_lens[bn // head_count] - mi * m_base)
            _check(0 <= start < start + count <= rows, "FD row interval out of range")
            _check(workspace + parts <= 2 * aic, "FD workspace span out of range")
            key = (section, bn, mi)
            value = (workspace, parts)
            _check(key not in fd_workspaces or fd_workspaces[key] == value, "inconsistent FD workspace/parts")
            fd_workspaces[key] = value
            fd_tasks[key].append((start, start + count))
    _check(is_fd == int(bool(fd_tasks)), "isFD differs from FD records")
    _check(not is_fd or target["kernel_arch"] != "arch92", "current arch92 policy disables FD")
    try:
        decoded = load_decoder(repo, request)(words, q_lens, kv_lens)
    except Exception as error:
        fail("COVERAGE_FAILED", f"original decoder rejected metadata: {error}")
    _, _, segments, decoded_heads, is_decode = decoded
    _check(decoded_heads == head_count and not is_decode, "decoder template mismatch")
    split_tasks = {(bn, mi) for bn, groups in enumerate(segments) for mi, pieces in enumerate(groups) if len(pieces) > 1}
    _check({(bn, mi) for _, bn, mi in fd_tasks} == split_tasks, "FD tasks differ from cross-core FA splits")
    for (section, bn, mi), intervals in fd_tasks.items():
        cursor = 0
        for start, end in sorted(intervals):
            _check(start == cursor, "FD row coverage has gaps or overlaps")
            cursor = end
        _check(cursor == min(m_base, q_lens[bn // head_count] - mi * m_base), "incomplete FD row coverage")
        _check(fd_workspaces[(section, bn, mi)][1] == len(segments[bn][mi]), "FD split count differs from FA splits")
    for section in range(sections):
        spans = sorted((workspace, workspace + parts) for key, (workspace, parts) in fd_workspaces.items()
                       if key[0] == section)
        _check(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "FD task workspace spans overlap")
    return {"structure": "passed", "original_decoder_coverage": "passed", "fd_tasks": len(fd_tasks),
            "q_lengths": q_lens, "kv_lengths": kv_lens}
