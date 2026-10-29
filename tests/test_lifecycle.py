from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qfa_mock import MetadataProvider, MockError
from qfa_mock.build import ROOT
from qfa_mock import integrations, runtime_loader
from qfa_mock.model import canonical, load_json
from test_prototype import FakeTensor, make_call


REPO = Path(os.environ.get("QFA_REPO", str(ROOT.parent / "ops-transformer")))
HELPER = ROOT / "build/metadata_helper"


@contextmanager
def clean_upstream_modules():
    names = lambda: [n for n in sys.modules if n.startswith(("qfa_assets_impl_", "qfa_original_assets_spec_"))
                     or (n.startswith("quant_flash_attn") and n.endswith("_golden"))]
    previous = {n: sys.modules.pop(n) for n in names()}
    with patch.object(runtime_loader, "_checkout", None), patch.dict(runtime_loader._digests, {}, clear=True):
        try:
            yield
        finally:
            for name in names():
                sys.modules.pop(name)
            sys.modules.update(previous)


class LifecycleTests(unittest.TestCase):
    def assert_code(self, code, fn, *args, **kwargs):
        with self.assertRaises(MockError) as caught:
            fn(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_worker_identity_is_semantic_and_configuration_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(integrations._providers, {}, clear=True):
            path = Path(directory) / "config.json"
            config = load_json(ROOT / "examples/target_arch92.json")
            path.write_text(canonical(config))
            env = {"QFA_REPO": str(REPO), "QFA_METADATA_CONFIG": str(path), "QFA_MOCK_HELPER": str(HELPER)}
            with patch.dict(os.environ, env):
                first = integrations.get_provider()
                path.write_text(json.dumps(dict(reversed(list(config.items()))), indent=4))
                self.assertIs(integrations.get_provider(), first)
                moved = Path(directory) / "moved.json"
                moved.write_text(path.read_text())
                with patch.dict(os.environ, QFA_METADATA_CONFIG=str(moved)):
                    self.assertIs(integrations.get_provider(), first)
                config["target"].update(aic_core_num=16, aiv_core_num=16)
                path.write_text(canonical(config))
                self.assert_code("WORKER_CONFIG_CHANGED", integrations.get_provider)

    def test_worker_rejects_checkout_helper_and_build_changes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(integrations._providers, {}, clear=True):
            directory = Path(directory)
            path = directory / "config.json"
            path.write_text((ROOT / "examples/target_arch92.json").read_text())
            helper = directory / "helper"
            helper.write_bytes(HELPER.read_bytes())
            manifest = load_json(str(HELPER) + ".json")
            Path(str(helper) + ".json").write_text(canonical(manifest))
            with patch.dict(os.environ, QFA_REPO=str(REPO), QFA_METADATA_CONFIG=str(path), QFA_MOCK_HELPER=str(helper)):
                integrations.get_provider(expected_repo=REPO)
                with patch.dict(os.environ, QFA_REPO=directory.as_posix()):
                    self.assert_code("WORKER_CONFIG_CHANGED", integrations.get_provider, expected_repo=REPO)
                with patch.dict(os.environ, QFA_MOCK_HELPER=str(HELPER)):
                    self.assert_code("WORKER_CONFIG_CHANGED", integrations.get_provider)
                manifest["compiler"] = "changed compiler identity"
                Path(str(helper) + ".json").write_text(canonical(manifest))
                self.assert_code("STALE_BUILD", integrations.get_provider)

    def test_new_pid_gets_its_own_provider_and_fork_reset_clears_locks(self):
        with patch.dict(integrations._providers, {}, clear=True), \
                patch.dict(os.environ, QFA_REPO=str(REPO), QFA_METADATA_CONFIG=str(ROOT / "examples/target_arch92.json"), QFA_MOCK_HELPER=str(HELPER)):
            with patch("qfa_mock.integrations.os.getpid", return_value=100):
                first = integrations.get_provider()
            with patch("qfa_mock.integrations.os.getpid", return_value=101):
                second = integrations.get_provider()
            self.assertIsNot(first, second)
            old_lock = integrations._provider_lock
            integrations._after_fork()
            self.assertEqual(integrations._providers, {})
            self.assertIsNot(integrations._provider_lock, old_lock)

    def test_path_pinned_spec_reuse_and_second_checkout_rejected(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second, clean_upstream_modules():
            for directory in (first, second):
                path = Path(directory) / "attention/quant_flash_attn/tests/assets/spec.py"
                path.parent.mkdir(parents=True)
                path.write_text("class QuantFlashAttnSpec: pass\n")
            one = runtime_loader.load_original_spec(first, ROOT / "ttk_plugin.py")
            two = runtime_loader.load_original_spec(Path(first) / ".", ROOT / "ttk_plugin.py")
            self.assertIs(one, two)
            self.assertTrue(one.__name__.startswith("qfa_original_assets_spec_"))
            self.assert_code("MODULE_SOURCE_CONFLICT", runtime_loader.load_original_spec, second, ROOT / "ttk_plugin.py")

    def test_cached_impl_from_other_checkout_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, clean_upstream_modules():
            path = Path(directory) / "attention/quant_flash_attn/tests/assets/spec.py"
            path.parent.mkdir(parents=True)
            path.write_text("class QuantFlashAttnSpec: pass\n")
            foreign = SimpleNamespace(__file__="/another/checkout/impl/golden.py", __name__="qfa_assets_impl_golden")
            with patch.dict(sys.modules, qfa_assets_impl_golden=foreign):
                self.assert_code("MODULE_SOURCE_CONFLICT", runtime_loader.load_original_spec, directory, ROOT / "ttk_plugin.py")
                self.assertIs(sys.modules["qfa_assets_impl_golden"], foreign)

    def test_module_source_change_and_missing_signature_fail(self):
        with tempfile.TemporaryDirectory() as directory, clean_upstream_modules():
            path = Path(directory) / "module.py"
            path.write_text("def call(q): return q\n")
            spec = importlib.util.spec_from_file_location("fixture", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            runtime_loader.verify_module(module, path)
            self.assert_code("UPSTREAM_INCOMPATIBLE", runtime_loader.require_signature, module.call, ("q", "k"), path)
            path.write_text("def call(q): return None\n")
            self.assert_code("SOURCE_CHANGED", runtime_loader.verify_module, module, path)

    def _golden_call(self, mode):
        provider = MetadataProvider(load_json(ROOT / "examples/target_arch92.json"), REPO, HELPER)
        fallback = []
        module = SimpleNamespace()
        module.get_qfa_section_info = lambda q, kv: None
        def decoder(words, q, kv):
            if mode == "decoder_error":
                raise ValueError("decoder failure")
            return None if mode == "empty" else "decoded"
        module._decode_qfa_sections = decoder
        def compute(*args):
            if mode == "section_error":
                raise ValueError("section failure")
            if mode == "interrupt":
                raise KeyboardInterrupt()
            if mode == "exit":
                raise SystemExit(9)
            return ["Out", "LSE"]
        module._cpu_mxfp8_golden_sectioned = compute
        originals = (module.get_qfa_section_info, module._cpu_mxfp8_golden_sectioned)
        def body():
            info = module.get_qfa_section_info([128], [1024])
            if mode == "skip_section":
                return ["Out", "LSE"]
            try:
                output = module._cpu_mxfp8_golden_sectioned(info)
                if mode == "twice":
                    module._cpu_mxfp8_golden_sectioned(info)
            except Exception:
                fallback.append(True)
                return "global reference"
            if mode == "output_error":
                raise RuntimeError("output processing failure")
            return output
        args = (FakeTensor((128, 64, 128)), FakeTensor((2, 8, 512, 128)), FakeTensor((2, 8, 512, 128)))
        kwargs = {"cu_seqlens_q_t": [0, 128], "seqused_q_t": [128], "seqused_kv_t": [1024]}
        try:
            with patch("qfa_mock.integrations.get_provider", return_value=provider):
                return integrations.run_golden(make_call(body, golden=True), module, args, kwargs)
        finally:
            self.assertEqual(fallback, [])
            self.assertIs(module.get_qfa_section_info, originals[0])
            self.assertIs(module._cpu_mxfp8_golden_sectioned, originals[1])

    def test_golden_requires_successful_section_and_restores_all_failure_paths(self):
        self.assertEqual(self._golden_call("success"), ["Out", "LSE"])
        for mode in ("empty", "skip_section", "section_error", "twice"):
            with self.subTest(mode=mode):
                error = self.assert_code("GOLDEN_FAILED", self._golden_call, mode)
                if mode in ("section_error", "twice"):
                    self.assertIsInstance(error.__cause__, Exception)
        with self.assertRaises(ValueError):
            self._golden_call("decoder_error")
        with self.assertRaises(RuntimeError):
            self._golden_call("output_error")

    def test_user_interrupts_are_never_converted(self):
        with self.assertRaises(KeyboardInterrupt):
            self._golden_call("interrupt")
        with self.assertRaises(SystemExit) as caught:
            self._golden_call("exit")
        self.assertEqual(caught.exception.code, 9)

    def test_replacement_and_failure_stage_logs_are_structured(self):
        with self.assertLogs("qfa_mock", level="INFO") as captured:
            self.assert_code("GOLDEN_FAILED", self._golden_call, "section_error")
        events = [json.loads(line.split("qfa_mock ", 1)[1]) for line in captured.output if "qfa_mock {" in line]
        self.assertTrue(any(e["event"] == "replacements_restored" and e["restored"] for e in events))
        self.assertTrue(any(e["event"] == "stage_failed" and e["stage"] == "sectioned_compute" for e in events))
