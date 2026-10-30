"""Run the patched upstream functions directly, without replacing plan functions."""

import importlib.util
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qfa_mock.build import ROOT
from qfa_mock import integrations
from qfa_mock.model import load_json
from cpu_support import cpu_import_boundary, pa_inputs
from test_lifecycle import clean_upstream_modules


REPO = Path(os.environ.get("QFA_REPO", str(ROOT.parent / "ops-transformer")))
TTK = Path(os.environ.get("QFA_TTK_REPO", str(ROOT.parent / "ops-test-kit")))
HAS_TORCH = importlib.util.find_spec("torch") is not None
PATCHED_RECORDS = {"numerical": [], "slots": [], "native_selection": False, "original_spec_discovered": False}


@unittest.skipUnless(HAS_TORCH, "CPU Torch is required for patched assets acceptance")
class PatchedAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="qfa-patched-assets-")
        cls.repo = Path(cls.temporary.name)
        cls.assets = cls.repo / "attention/quant_flash_attn/tests/assets"
        shutil.copytree(REPO / "attention/quant_flash_attn/tests/assets", cls.assets,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        cls.before = {str(p.relative_to(cls.assets)): p.read_bytes() for p in cls.assets.rglob("*.py")}
        cls.apply = [sys.executable, str(ROOT / "scripts/apply_assets_patch.py"), "--repo", str(cls.repo)]
        subprocess.run([*cls.apply, "--check"], check=True, capture_output=True)
        subprocess.run(cls.apply, check=True, capture_output=True)
        cls.clean = clean_upstream_modules()
        cls.clean.__enter__()
        cls.boundary = cpu_import_boundary(torch)
        cls.boundary.__enter__()
        cls.old_path = list(sys.path)
        spec = importlib.util.spec_from_file_location("patched_qfa_spec", cls.assets / "spec.py")
        cls.base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.base)
        cls.golden = cls.base.load_impl_module("golden")
        cls.numeric = cls.golden.mxfp8_golden_mod
        cls.preprocess = cls.base.load_impl_module("npu_preprocess")
        cls.inputs = cls.base.load_impl_module("inputs")
        logging.getLogger().setLevel(logging.WARNING)

    @classmethod
    def tearDownClass(cls):
        sys.path[:] = cls.old_path
        cls.boundary.__exit__(None, None, None)
        cls.clean.__exit__(None, None, None)
        cls.temporary.cleanup()

    def setUp(self):
        self.environment = patch.dict(os.environ, QFA_REPO=str(REPO),
            QFA_METADATA_CONFIG=str(ROOT / "examples/target_arch92.json"),
            QFA_MOCK_HELPER=str(ROOT / "build/metadata_helper"))
        self.environment.start()
        self.old_optional = {k: os.environ.pop(k, None) for k in ("QFA_PLATFORM_EVIDENCE", "QFA_RUN_CONTEXT")}
        self.providers = patch.dict(integrations._providers, {}, clear=True)
        self.providers.start()

    def tearDown(self):
        self.providers.stop()
        for key, value in self.old_optional.items():
            if value is not None:
                os.environ[key] = value
        self.environment.stop()

    def test_direct_numeric_baseline_varlen_padding_zero_fd_and_sections(self):
        cases = [([128], [128], [1024], 64, 8, 512, "arch92"),
                 ([17, 33], [17, 33], [64, 96], 4, 1, 64, "arch92"),
                 ([16, 16], [8, 0], [64, 0], 4, 1, 64, "arch92"),
                 ([8], [8], [64], 2, 1, 64, "arch92"),
                 ([1], [1], [32768], 1, 1, 512, "arch35"),
                 ([129], [129], [16384], 24, 24, 512, "arch35")]
        objects = (self.numeric.get_qfa_section_info, self.numeric._cpu_mxfp8_golden_sectioned, self.preprocess.run_metadata)
        for allocated, used, kv, nq, nk, bs, arch in cases:
            with self.subTest(q=used, kv=kv, arch=arch):
                integrations._providers.clear()  # Independent worker lifecycle for each target case.
                os.environ["QFA_METADATA_CONFIG"] = str(ROOT / f"examples/target_{arch}.json")
                args, kwargs = pa_inputs(self.torch, self.inputs, allocated, used, kv, nq, nk, bs)
                output = self.base.QuantFlashAttnSpec.golden(*args, **kwargs)
                self.assertEqual(tuple(output[0].shape), (sum(allocated), nq, 128))
                self.assertEqual(tuple(output[1].shape), (nq, sum(allocated)))
                self.assertEqual(output[0].dtype, self.torch.float32)
                self.assertTrue(self.torch.isfinite(output[0]).all())
                self.assertIs(self.numeric.get_qfa_section_info, objects[0])
                self.assertIs(self.numeric._cpu_mxfp8_golden_sectioned, objects[1])
                self.assertIs(self.preprocess.run_metadata, objects[2])
                PATCHED_RECORDS["numerical"].append({"q": used, "kv": kv, "arch": arch,
                    "out_shape": list(output[0].shape), "lse_shape": list(output[1].shape), "no_function_replacements": True,
                    "out_sha256": hashlib.sha256(output[0].contiguous().numpy().tobytes()).hexdigest(),
                    "lse_sha256": hashlib.sha256(output[1].contiguous().numpy().tobytes()).hexdigest()})

    def test_direct_preprocess_slots_share_the_golden_plan(self):
        torch = self.torch
        args, kwargs = pa_inputs(torch, self.inputs, [16], [16], [64])
        # No mocked plan producer, platform checker, or device_tensor: real CPU transfer.
        original = self.preprocess.run_metadata
        for kind in ("dynamic", "placeholder", "matching", "mismatch"):
            with self.subTest(kind=kind):
                values = list(args)
                provider = integrations.get_provider()
                from qfa_mock.assets import request
                bound = __import__("inspect").signature(self.preprocess.run).bind(*args, **kwargs)
                bound.apply_defaults()
                expected = provider.generate(request(bound.arguments)).cpu_tensor()
                slot = {"dynamic": None, "placeholder": torch.empty(0, dtype=torch.int32),
                        "matching": torch.empty_like(expected), "mismatch": torch.zeros(2, 4, dtype=torch.int32)}[kind]
                values[15] = slot
                if kind == "mismatch":
                    with self.assertRaises(ValueError):
                        self.base.QuantFlashAttnSpec.npu_preprocess(*values, **kwargs)
                else:
                    result = self.base.QuantFlashAttnSpec.npu_preprocess(*values, **kwargs)
                    actual = result["metadata"] if kind == "dynamic" else slot
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(actual.is_contiguous())
                self.assertIs(self.preprocess.run_metadata, original)
                PATCHED_RECORDS["slots"].append({"kind": kind, "passed": True, "no_function_replacements": True})

    def test_decoder_and_numeric_errors_propagate_without_native_or_global_fallback(self):
        args, kwargs = pa_inputs(self.torch, self.inputs, [8], [8], [64], 2, 1)
        for name in ("_decode_qfa_sections", "_cpu_mxfp8_golden_sectioned"):
            with self.subTest(name=name), patch.object(self.numeric, name, side_effect=RuntimeError(name)), \
                    patch.object(self.numeric, "get_qfa_section_info", side_effect=AssertionError("native plan called")), \
                    patch.object(self.numeric, "broadcast_kv", side_effect=AssertionError("global fallback called")):
                with self.assertRaises(RuntimeError) as caught:
                    self.base.QuantFlashAttnSpec.golden(*args, **kwargs)
                self.assertEqual(str(caught.exception), name)

    def test_bad_config_and_unsupported_quant_mode_fail_early(self):
        args, kwargs = pa_inputs(self.torch, self.inputs, [8], [8], [64], 2, 1)
        with patch.dict(os.environ, QFA_METADATA_CONFIG="/nonexistent/mock-config.json"):
            with self.assertRaises(OSError):
                self.base.QuantFlashAttnSpec.golden(*args, **kwargs)
        for mode in (0, 6, 10):
            values = list(args)
            values[6] = mode
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.base.QuantFlashAttnSpec.golden(*values, **kwargs)

    def test_unset_config_keeps_native_plan_and_preprocess_selection(self):
        args, kwargs = pa_inputs(self.torch, self.inputs, [8], [8], [64], 2, 1)
        from qfa_mock.assets import request, golden_plan
        import inspect
        bound = inspect.signature(self.golden.cpu_qfa_mxfp8).bind(*args, **kwargs)
        bound.apply_defaults()
        current = request(bound.arguments, golden=True)
        info = golden_plan(current, self.numeric._decode_qfa_sections, [8], [64])
        metadata = integrations.get_provider().generate(current).cpu_tensor()
        with patch.dict(os.environ, QFA_METADATA_CONFIG=""), \
                patch.object(self.numeric, "get_qfa_section_info", return_value=info) as native_plan, \
                patch.object(self.preprocess, "run_metadata", return_value=metadata) as native_metadata:
            self.base.QuantFlashAttnSpec.golden(*args, **kwargs)
            self.base.QuantFlashAttnSpec.npu_preprocess(*args, **kwargs)
            native_plan.assert_called_once()
            native_metadata.assert_called_once()
        PATCHED_RECORDS["native_selection"] = True

    def test_patch_is_idempotent_reversible_and_does_not_need_git_repository(self):
        self.assertFalse((self.repo / ".git").exists())
        again = subprocess.run(self.apply, capture_output=True, text=True)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("Already applied", again.stdout)
        subprocess.run([*self.apply, "--reverse"], check=True, capture_output=True)
        after = {str(p.relative_to(self.assets)): p.read_bytes() for p in self.assets.rglob("*.py")}
        self.assertEqual(after, self.before)
        subprocess.run(self.apply, check=True, capture_output=True)

    def test_actual_ttk_discovers_the_original_patched_assets_spec(self):
        sys.path.insert(0, str(TTK))
        try:
            from ttk.core_modules.common.plugin.spec_loader import SpecLoader
            found = SpecLoader([str(self.assets)]).load("torch.ops.cann_ops_transformer.quant_flash_attn")
            self.assertIsNotNone(found)
            self.assertEqual(found.__name__, "QuantFlashAttnSpec")
            self.assertTrue(callable(found.golden) and callable(found.npu_preprocess))
            PATCHED_RECORDS["original_spec_discovered"] = True
        finally:
            sys.path.remove(str(TTK))
