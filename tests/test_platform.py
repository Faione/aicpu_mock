import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qfa_mock import MetadataProvider, MockError
from qfa_mock.build import ROOT
from qfa_mock.model import canonical, load_json, normalize_request
from qfa_mock import platform_check as platform


REPO = Path(os.environ.get("QFA_REPO", str(ROOT.parent / "ops-transformer")))


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.provider = MetadataProvider(load_json(ROOT / "examples/target_arch92.json"), REPO, ROOT / "build/metadata_helper")
        self.request = normalize_request(load_json(ROOT / "examples/request.json"))
        self.result = self.provider.generate(self.request)
        toolkit = self.root / "toolkit"
        opp = toolkit / "opp"
        self.env = {"ASCEND_TOOLKIT_HOME": str(toolkit), "ASCEND_OPP_PATH": str(opp), "LD_LIBRARY_PATH": ""}
        def artifact(name, path, content):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return {"path": str(path), "sha256": platform.digest_file(path)}
        model = artifact("model", toolkit / "tools/simulator/model-alias/camodel/libmodel.so", "model fixture")
        model["version"] = "fixture-1"
        ini = artifact("platform", toolkit / "aarch64-linux/data/platform_config/cann-alias.ini", "[SoCInfo]\nNpuArch=9201\n")
        binary = artifact("qfa_binary", opp / "vendors/test/qfa.so", "qfa binary fixture")
        provenance = artifact("source", toolkit / "python/soc_info.py", "# installed mapping fixture\n")
        build = {"qfa_binary_sha256": binary["sha256"], "core_limit_policy": "no_additional_limits",
                 "soc_version": "Ascend960", "kernel_arch": "arch92", "npu_arch": "9201",
                 "effective_aic_core_num": 80, "effective_aiv_core_num": 80,
                 "schedule_source_digest": self.result.summary["source_digest"], "algorithm_version": "qfa-host-v1"}
        mapping = {"sim_soc": "sim-alias", "model_name": "model-alias", "cann_platform": "cann-alias",
                   "soc_version": "Ascend960", "source": provenance,
                   "runtime_soc_aliases": [{"value": 1234, "soc_version": "Ascend960"}]}
        self.record = {"protocol_version": 1, "run_id": "fixture-run", "launch": {
            "command": [sys.executable, "-m", "ttk", "e2e", "--backend", "npusim", "--sim-soc", "sim-alias", "--plugin", str(ROOT / "ttk_plugin.py")],
            "cwd": str(self.root), "environment": self.env,
            "sim_soc": "sim-alias", "model_name": "model-alias", "cann_platform": "cann-alias"},
            "artifacts": {"model": model, "platform": ini, "qfa_binary": binary,
                          "build_record": artifact("build", self.root / "build.json", canonical(build)),
                          "mapping": artifact("mapping", self.root / "mapping.json", canonical(mapping))}}
        self.path = self.root / "evidence.json"
        self.write_record()
        self.environment = patch.dict(os.environ, {**self.env, "LD_LIBRARY_PATH": str(toolkit / "tools/simulator/model-alias/camodel"),
                                                   "QFA_PLATFORM_EVIDENCE": str(self.path), "QFA_RUN_CONTEXT": self.receipt()})
        self.environment.start()
        self.state = patch.dict(platform._worker_evidence, {}, clear=True)
        self.state.start()
        self.torch_patch = patch.dict(sys.modules, torch=SimpleNamespace())
        self.torch_patch.start()
        self.anchor = SimpleNamespace(device="npu:0")

    def tearDown(self):
        self.torch_patch.stop()
        self.state.stop()
        self.environment.stop()
        self.directory.cleanup()

    def write_record(self):
        self.path.write_text(canonical(self.record))

    def receipt(self):
        return canonical({"run_id": self.record["run_id"], "evidence_digest": platform.digest_file(self.path),
                          "command_digest": hashlib.sha256(canonical(self.record["launch"]["command"]).encode()).hexdigest()})

    def update_artifact(self, name, updates):
        path = Path(self.record["artifacts"][name]["path"])
        data = load_json(path)
        data.update(updates)
        path.write_text(canonical(data))
        self.record["artifacts"][name]["sha256"] = platform.digest_file(path)
        self.write_record()
        os.environ["QFA_RUN_CONTEXT"] = self.receipt()

    def check(self):
        return platform.check_device_target(self.provider, self.request, self.result, self.anchor)

    def assert_code(self, code, fn=None, *args):
        with self.assertRaises(MockError) as caught:
            (self.check if fn is None else fn)(*args)
        self.assertEqual(caught.exception.code, code)

    def test_bound_offline_evidence_works_without_runtime_queries(self):
        self.check()

    def test_queries_are_not_required_but_explicit_conflicts_fail(self):
        def unavailable(*args):
            raise RuntimeError("query unavailable in simulator")
        sys.modules["torch"].npu = SimpleNamespace(get_device_properties=unavailable, get_soc_version=unavailable)
        self.check()
        sys.modules["torch"].npu = SimpleNamespace(get_device_properties=lambda dev: SimpleNamespace(cube_core_num=96, vector_core_num=96), get_soc_version=lambda: 1234)
        self.check()  # Physical 96 is not confused with effective 80.
        sys.modules["torch"].npu.get_device_properties = lambda dev: SimpleNamespace(cube_core_num=16, vector_core_num=16)
        self.assert_code("TARGET_MISMATCH")

    def test_soc_aliases_are_typed_and_provenance_backed(self):
        sys.modules["torch"].npu = SimpleNamespace(get_soc_version=lambda: 1234)
        self.check()
        platform._worker_evidence.clear()
        self.update_artifact("mapping", {"runtime_soc_aliases": [{"value": 1234, "soc_version": "Ascend910_9599"}]})
        self.assert_code("TARGET_MISMATCH")
        self.assertIsNone(platform.normalize_runtime_soc("1234", [{"value": 1234, "soc_version": "Ascend960"}]))

    def test_missing_evidence_receipt_or_changed_artifacts_fail(self):
        with patch.dict(os.environ, QFA_RUN_CONTEXT=""):
            self.assert_code("TARGET_UNVERIFIED")
        with patch.dict(os.environ, QFA_RUN_CONTEXT='{}'):
            self.assert_code("TARGET_MISMATCH")
        Path(self.record["artifacts"]["qfa_binary"]["path"]).write_text("replaced build")
        self.assert_code("TARGET_MISMATCH")

    def test_target_arch_effective_cores_and_rules_are_all_checked(self):
        for field, bad in (("kernel_arch", "arch35"), ("effective_aic_core_num", 16),
                           ("schedule_source_digest", "wrong"), ("npu_arch", "3510")):
            with self.subTest(field=field):
                old = load_json(self.record["artifacts"]["build_record"]["path"])
                self.update_artifact("build_record", {field: bad})
                self.assert_code("TARGET_MISMATCH")
                self.update_artifact("build_record", old)

    def test_platform_environment_and_worker_evidence_cannot_change(self):
        with patch.dict(os.environ, ASCEND_OPP_PATH="/another/opp"):
            self.assert_code("TARGET_MISMATCH")
        self.check()
        self.record["run_id"] = "another-run"
        self.write_record()
        os.environ["QFA_RUN_CONTEXT"] = self.receipt()
        self.assert_code("WORKER_CONFIG_CHANGED")

    def test_actual_tiling_is_compared_to_header_and_request(self):
        h = self.result.summary["header"]
        tiling = dict(zip(("section_num", "is_fd", "m_base", "s2_base", "aic_core_num", "aiv_core_num"), h[:6]))
        tiling.update(request_digest=self.result.summary["request_digest"], kernel_arch="arch92")
        path = self.root / "tiling.json"
        path.write_text(canonical(tiling))
        self.record["artifacts"]["tiling"] = {"path": str(path), "sha256": platform.digest_file(path)}
        self.write_record()
        os.environ["QFA_RUN_CONTEXT"] = self.receipt()
        self.check()
        platform._worker_evidence.clear()
        self.update_artifact("tiling", {"m_base": h[2] * 2})
        self.assert_code("TARGET_MISMATCH")

    def test_launcher_executes_only_the_recorded_command(self):
        command = self.record["launch"]["command"]
        with patch("qfa_mock.platform_check.subprocess.call", return_value=0) as run, patch.dict(os.environ, LD_LIBRARY_PATH=""):
            self.assertEqual(platform.launch(self.path, command), 0)
            called = run.call_args
            self.assertEqual(called.args[0], command)
            self.assertEqual(json.loads(called.kwargs["env"]["QFA_RUN_CONTEXT"])["run_id"], "fixture-run")
        with patch.dict(os.environ, LD_LIBRARY_PATH=""):
            self.assert_code("TARGET_MISMATCH", platform.launch, self.path, command + ["--extra"])

    def test_evidence_preparation_cli_packages_existing_records(self):
        output = self.root / "prepared.json"
        command = [sys.executable, str(ROOT / "scripts/prepare_platform_evidence.py")]
        for option, artifact in (("model", "model"), ("platform", "platform"), ("qfa-binary", "qfa_binary"),
                                 ("build-record", "build_record"), ("mapping", "mapping")):
            command.extend(["--" + option, self.record["artifacts"][artifact]["path"]])
        command.extend(["--model-version", "fixture-1", "--cwd", str(self.root), "--output", str(output),
                        "--", *self.record["launch"]["command"]])
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        prepared, _, _, _ = platform.verify_record(output)
        self.assertEqual(prepared["launch"]["command"], self.record["launch"]["command"])
