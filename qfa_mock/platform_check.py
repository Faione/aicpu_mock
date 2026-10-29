"""Bind offline platform/build evidence to the actual launched simulation.

Device properties alone describe physical capacity, not thread-effective cores or
the compiled kernel. They are diagnostics; the default adapter uses a pinned
offline build record plus a launch receipt, with no skip-check switch.
"""

import configparser
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .build import ROOT
from .model import ALGORITHM_VERSION, canonical, fail, load_json
from .telemetry import event


_worker_evidence = {}


def digest_file(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _require(ok, message, code="TARGET_UNVERIFIED"):
    if not ok:
        fail(code, message)


def _artifact(record, name):
    item = record["artifacts"].get(name)
    _require(isinstance(item, dict) and isinstance(item.get("path"), str), f"missing artifact: {name}")
    path = Path(item["path"])
    _require(path.is_absolute() and path.is_file(), f"artifact {name} must name an existing absolute file")
    _require(digest_file(path) == item.get("sha256"), f"artifact digest changed: {name}", "TARGET_MISMATCH")
    return path.resolve()


def verify_record(path, worker=False):
    path = Path(path).resolve()
    r = load_json(path)
    _require(isinstance(r, dict) and type(r.get("protocol_version")) is int and r["protocol_version"] == 1, "unsupported platform evidence version")
    for key in ("run_id", "launch", "artifacts"):
        _require(bool(r.get(key)), f"missing evidence field: {key}")
    _require(isinstance(r["run_id"], str), "run_id must be a string")
    launch = r["launch"]
    command = launch.get("command")
    _require(isinstance(command, list) and len(command) > 4 and all(isinstance(x, str) for x in command), "launch.command must be an argv array")
    _require(Path(command[0]).is_absolute() and Path(command[0]).is_file(), "launch executable must be an absolute file")
    _require(command[1:4] == ["-m", "ttk", "e2e"], "evidence launcher only supports python -m ttk e2e")
    def option(name):
        values = [command[i + 1] for i, value in enumerate(command[:-1]) if value == name]
        _require(len(values) == 1, f"launch requires exactly one {name}")
        return values[0]
    _require(option("--backend") == "npusim", "evidence is for npusim only")
    _require(option("--sim-soc") == launch.get("sim_soc"), "simulator SoC differs from command", "TARGET_MISMATCH")
    _require(Path(option("--plugin")).resolve() == ROOT / "ttk_plugin.py", "launch must select this mock plugin")
    _require(not any(x.startswith(("--aic-core-num", "--aiv-core-num", "--core-num")) for x in command), "additional core restrictions are not supported")
    cwd = Path(launch.get("cwd", ""))
    _require(cwd.is_absolute() and cwd.is_dir(), "launch.cwd must be an existing absolute directory")
    env = launch.get("environment")
    _require(isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), "launch.environment must contain string values")
    for key, value in env.items():
        expected_value = value
        if key == "LD_LIBRARY_PATH" and worker:
            model_dir = str(Path(env["ASCEND_TOOLKIT_HOME"]) / "tools/simulator" / launch["model_name"] / "camodel")
            # Match the verified TTK _inject_camodel_env transformation exactly.
            if model_dir not in value.split(":"):
                expected_value = f"{model_dir}:{value}" if value else model_dir
            _require(expected_value.split(":")[0] == model_dir, "camodel must precede other runtime library paths", "TARGET_MISMATCH")
        _require(os.environ.get(key, "") == expected_value, f"launch environment differs at {key}", "TARGET_MISMATCH")
    _require(bool(env.get("ASCEND_TOOLKIT_HOME")) and bool(env.get("ASCEND_OPP_PATH")), "record toolkit/OPP paths in launch.environment")
    _require("LD_LIBRARY_PATH" in env, "record the initial LD_LIBRARY_PATH, including an empty value")
    paths = {name: _artifact(r, name) for name in ("model", "platform", "qfa_binary", "build_record", "mapping")}
    mapping, build = load_json(paths["mapping"]), load_json(paths["build_record"])
    aliases = mapping.get("runtime_soc_aliases", [])
    _require(isinstance(aliases, list) and all(isinstance(a, dict) and type(a.get("value")) in (str, int)
             and isinstance(a.get("soc_version"), str) for a in aliases), "invalid typed runtime SoC aliases")
    _require(len({(type(a["value"]).__name__, canonical(a["value"])) for a in aliases}) == len(aliases), "ambiguous runtime SoC aliases")
    for key in ("sim_soc", "model_name", "cann_platform"):
        _require(mapping.get(key) == launch.get(key) and bool(mapping.get(key)), f"mapping differs at {key}", "TARGET_MISMATCH")
        _require(isinstance(mapping[key], str), f"mapping.{key} must be a string")
    _require(all(Path(mapping[key]).name == mapping[key] and mapping[key] not in (".", "..")
                 for key in ("model_name", "cann_platform")), "model/platform names must be simple path components")
    provenance = mapping.get("source", {})
    _require(isinstance(provenance.get("path"), str) and Path(provenance["path"]).is_absolute()
             and Path(provenance["path"]).is_file(), "mapping requires installed CANN/project provenance file")
    _require(digest_file(provenance["path"]) == provenance.get("sha256"), "mapping provenance changed", "TARGET_MISMATCH")
    toolkit = Path(env["ASCEND_TOOLKIT_HOME"]).resolve()
    model_dir = toolkit / "tools/simulator" / mapping["model_name"] / "camodel"
    _require(model_dir in paths["model"].parents, "model artifact is outside recorded toolkit/model directory", "TARGET_MISMATCH")
    _require(isinstance(r["artifacts"]["model"].get("version"), str) and bool(r["artifacts"]["model"]["version"]), "model version is required")
    _require(paths["platform"].stem == mapping["cann_platform"], "CANN platform name differs from platform file", "TARGET_MISMATCH")
    platform_root = Path(env["ASCEND_OPP_PATH"]).resolve().parent / "aarch64-linux/data/platform_config"
    _require(paths["platform"].parent == platform_root.resolve(), "platform file is outside the recorded CANN platform directory", "TARGET_MISMATCH")
    # The kernel must be in the actual standard/custom OPP search roots, not an unrelated file.
    roots = [Path(env["ASCEND_OPP_PATH"]).resolve()]
    custom = env.get("ASCEND_CUSTOM_OPP_PATH", "")
    roots.extend(Path(p).resolve() for p in custom.split(os.pathsep) if p)
    _require(any(root in paths["qfa_binary"].parents for root in roots), "qfa binary is outside recorded OPP lookup roots", "TARGET_MISMATCH")
    _require(build.get("qfa_binary_sha256") == r["artifacts"]["qfa_binary"]["sha256"], "build record is not bound to qfa binary", "TARGET_MISMATCH")
    _require(build.get("core_limit_policy") == "no_additional_limits", "effective core restrictions are unknown or unsupported")
    for key in ("effective_aic_core_num", "effective_aiv_core_num"):
        _require(type(build.get(key)) is int and build[key] > 0, f"build record requires {key}")
    ini = configparser.ConfigParser()
    ini.read(paths["platform"])
    architecture_codes = {value for section in ini.sections() for key, value in ini.items(section)
                          if key.replace("_", "").lower() == "npuarch"}
    _require(architecture_codes == {str(build.get("npu_arch"))}, "platform NpuArch differs from compiled build record", "TARGET_MISMATCH")
    _require(mapping.get("soc_version") == build.get("soc_version"), "mapping scheduling SoC differs from build record", "TARGET_MISMATCH")
    return r, build, mapping, paths


def launch(evidence_path, command):
    record, _, _, _ = verify_record(evidence_path)
    _require(command == record["launch"]["command"], "actual argv differs from evidence command", "TARGET_MISMATCH")
    receipt = {"run_id": record["run_id"], "evidence_digest": digest_file(evidence_path),
               "command_digest": hashlib.sha256(canonical(command).encode()).hexdigest()}
    env = dict(os.environ, QFA_PLATFORM_EVIDENCE=str(Path(evidence_path).resolve()), QFA_RUN_CONTEXT=canonical(receipt))
    event("simulation_launch", evidence=str(Path(evidence_path).resolve()), receipt=receipt, command=command)
    return subprocess.call(command, cwd=record["launch"]["cwd"], env=env)


def normalize_runtime_soc(value, aliases):
    if isinstance(value, str) and value in ("Ascend960", "Ascend910_9599"):
        return value
    for alias in aliases:
        if type(alias.get("value")) is type(value) and alias["value"] == value:
            return alias["soc_version"]
    return None


def check_device_target(provider, request, result, anchor):
    device_type = getattr(anchor.device, "type", str(anchor.device).split(":")[0])
    _require(device_type == "npu", "device path requires an NPU anchor; CPU slot acceptance has a separate test boundary")
    path, context = os.environ.get("QFA_PLATFORM_EVIDENCE"), os.environ.get("QFA_RUN_CONTEXT")
    _require(bool(path) and bool(context), "device execution requires launch-bound platform evidence; use qfa_mock launch")
    record, build, mapping, paths = verify_record(path, worker=True)
    try:
        receipt = json.loads(context)
    except ValueError:
        fail("TARGET_UNVERIFIED", "invalid launch receipt")
    expected_receipt = {"run_id": record["run_id"], "evidence_digest": digest_file(path),
                        "command_digest": hashlib.sha256(canonical(record["launch"]["command"]).encode()).hexdigest()}
    _require(receipt == expected_receipt, "evidence is not bound to this launch", "TARGET_MISMATCH")
    identity = canonical(expected_receipt)
    previous = _worker_evidence.get(os.getpid())
    _require(previous is None or previous == identity, "worker platform evidence changed", "WORKER_CONFIG_CHANGED")
    target = provider.config["target"]
    for field in ("soc_version", "kernel_arch"):
        _require(build.get(field) == target[field], f"build differs from target at {field}", "TARGET_MISMATCH")
    _require((build["effective_aic_core_num"], build["effective_aiv_core_num"]) ==
             (target["aic_core_num"], target["aiv_core_num"]), "effective core counts differ", "TARGET_MISMATCH")
    _require(build.get("schedule_source_digest") == result.summary["source_digest"] and
             build.get("algorithm_version") == ALGORITHM_VERSION, "build schedule rules differ from helper", "TARGET_MISMATCH")
    header = result.summary["header"]
    if "tiling" in record["artifacts"]:
        tiling = load_json(_artifact(record, "tiling"))
        expected = dict(zip(("section_num", "is_fd", "m_base", "s2_base", "aic_core_num", "aiv_core_num"), header[:6]))
        expected.update(request_digest=result.summary["request_digest"], kernel_arch=target["kernel_arch"])
        for key, value in expected.items():
            _require(tiling.get(key) == value, f"tiling differs at {key}", "TARGET_MISMATCH")
        if header[1]:
            import struct
            words = struct.unpack(f"<{len(result.data) // 4}I", result.data)
            start = 16 + header[0] * header[4] * 16
            needed = max((words[i + 2] + words[i + 3] for i in range(start, result.summary["used_words"], 16)), default=0)
            _require(type(tiling.get("fd_workspace_slots")) is int and tiling["fd_workspace_slots"] >= needed,
                     "tiling FD workspace capacity is insufficient", "TARGET_MISMATCH")
    else:
        event("tiling_review_required", request_digest=result.summary["request_digest"],
              reason="preflight uses bound build/platform rules; retain first simulation tiling for review")
    import torch
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "get_device_properties"):
        try:
            props = npu.get_device_properties(anchor.device)
        except Exception as error:
            event("runtime_query_unavailable", query="physical_cores", error_type=type(error).__name__)
        else:
            capacity = (int(props.cube_core_num), int(props.vector_core_num))
            _require(capacity[0] >= target["aic_core_num"] and capacity[1] >= target["aiv_core_num"],
                     "effective cores exceed runtime physical capacity", "TARGET_MISMATCH")
            event("runtime_capacity", physical_cores=capacity, effective_cores=[target["aic_core_num"], target["aiv_core_num"]])
    if npu is not None and hasattr(npu, "get_soc_version"):
        try:
            raw = npu.get_soc_version()
        except Exception as error:
            event("runtime_query_unavailable", query="soc", error_type=type(error).__name__)
        else:
            normalized = target["soc_version"] if type(raw) is str and raw == target["soc_version"] else normalize_runtime_soc(raw, mapping.get("runtime_soc_aliases", []))
            if normalized is not None:
                _require(normalized == target["soc_version"], "normalized runtime SoC conflicts with offline evidence", "TARGET_MISMATCH")
            event("runtime_soc", raw_type=type(raw).__name__, raw=raw if type(raw) in (str, int) else repr(raw), normalized=normalized,
                  mapping_source=str(paths["mapping"]))
    _worker_evidence[os.getpid()] = identity
    event("platform_verified", evidence=str(Path(path).resolve()), evidence_digest=receipt["evidence_digest"],
          run_id=record["run_id"], target=target, build_record=str(paths["build_record"]))
