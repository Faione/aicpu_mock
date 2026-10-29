import importlib.util
import hashlib
import logging
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from qfa_mock import MetadataProvider, MockError
from qfa_mock.build import ROOT
from qfa_mock.integrations import request_from_call, run_golden, run_preprocess
from qfa_mock.model import load_json
from qfa_mock import runtime_loader

from cpu_support import cpu_import_boundary, forbidden, pa_inputs


HAS_TORCH = importlib.util.find_spec("torch") is not None
REPO = Path(os.environ.get("QFA_REPO", str(ROOT.parent / "ops-transformer")))
TTK = Path(os.environ.get("QFA_TTK_REPO", str(ROOT.parent / "ops-test-kit")))
ACCEPTANCE_RECORDS = {"slots": [], "numerical": [], "discovery": {}}


def numerical_record(name, output, summary):
    ACCEPTANCE_RECORDS["numerical"].append({"case": name, "header": summary["header"],
        "request_digest": summary["request_digest"], "metadata_digest": summary["metadata_digest"],
        "out_shape": list(output[0].shape), "lse_shape": list(output[1].shape),
        "out_dtype": str(output[0].dtype), "lse_dtype": str(output[1].dtype),
        "out_sha256": hashlib.sha256(output[0].contiguous().numpy().tobytes()).hexdigest(),
        "lse_sha256": hashlib.sha256(output[1].contiguous().numpy().tobytes()).hexdigest()})


@unittest.skipUnless(HAS_TORCH, "install requirements-cpu.txt for real upstream Python acceptance")
class RealPythonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)
        cls.boundary = cpu_import_boundary(torch)
        cls.boundary.__enter__()
        try:
            cls.base = runtime_loader.load_original_spec(REPO, ROOT / "ttk_plugin.py")
            cls.golden = runtime_loader.load_impl(cls.base, REPO, "golden")
            cls.preprocess = runtime_loader.load_impl(cls.base, REPO, "npu_preprocess")
            cls.inputs = cls.base.load_impl_module("inputs")
            runtime_loader.verify_module(cls.inputs, REPO / "attention/quant_flash_attn/tests/assets/impl/inputs.py")
            logging.getLogger().setLevel(logging.WARNING)
        except BaseException:
            cls.boundary.__exit__(*sys.exc_info())
            raise

    @classmethod
    def tearDownClass(cls):
        cls.boundary.__exit__(None, None, None)

    def setUp(self):
        self.provider = MetadataProvider(load_json(ROOT / "examples/target_arch92.json"), REPO, ROOT / "build/metadata_helper")

    def test_original_preprocess_slot_matrix_with_cpu_tensors(self):
        torch = self.torch
        args, kwargs = pa_inputs(torch, self.inputs, [16], [16], [64])
        request, _ = request_from_call(self.preprocess.run, args, kwargs)
        expected = self.provider.generate(request).cpu_tensor()
        original = self.preprocess.run_metadata
        for kind in ("dynamic", "placeholder", "matching", "mismatch"):
            with self.subTest(kind=kind):
                values = list(args)
                slot = {"dynamic": None, "placeholder": torch.empty(0, dtype=torch.int32),
                        "matching": torch.full(expected.shape, -99, dtype=torch.int32),
                        "mismatch": torch.zeros((2, 4), dtype=torch.int32)}[kind]
                values[15] = slot
                # Only the platform-evidence boundary is bypassed for CPU slot acceptance.
                # Parameter extraction, argument comparison, plan generation, Tensor copy,
                # and the complete original run body execute unchanged.
                with patch("qfa_mock.integrations.get_provider", return_value=self.provider), \
                        patch("qfa_mock.platform_check.check_device_target"):
                    if kind == "mismatch":
                        with self.assertRaises(ValueError):
                            run_preprocess(self.preprocess, tuple(values), kwargs)
                    else:
                        output = run_preprocess(self.preprocess, tuple(values), kwargs)
                        actual = output["metadata"] if kind == "dynamic" else slot
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertEqual(actual.dtype, torch.int32)
                        self.assertTrue(actual.is_contiguous())
                        if kind != "dynamic":
                            self.assertIsNone(output)
                self.assertIs(self.preprocess.run_metadata, original)
                ACCEPTANCE_RECORDS["slots"].append({"kind": kind, "passed": True, "restored": True})

    def test_real_golden_baseline_varlen_padding_zero_and_no_stale_context(self):
        torch = self.torch
        cases = [([128], [128], [1024], 64, 8, 512),
                 ([17, 33], [17, 33], [64, 96], 4, 1, 64),
                 ([16, 16], [8, 0], [64, 0], 4, 1, 64),
                 ([8], [8], [64], 2, 1, 64)]
        for allocations, used, kv, nq, nk, bs in cases:
            with self.subTest(q=used, kv=kv):
                args, kwargs = pa_inputs(torch, self.inputs, allocations, used, kv, nq, nk, bs)
                module = self.golden.mxfp8_golden_mod
                original = (module.get_qfa_section_info, module._cpu_mxfp8_golden_sectioned)
                with patch("qfa_mock.integrations.get_provider", return_value=self.provider):
                    output = run_golden(self.golden.cpu_qfa_mxfp8, module, args, kwargs)
                self.assertEqual(len(output), 2)
                self.assertEqual(tuple(output[0].shape), (sum(allocations), nq, 128))
                self.assertEqual(tuple(output[1].shape), (nq, sum(allocations)))
                self.assertEqual(output[0].dtype, torch.float32)  # Original CPU reference accumulates/returns FP32.
                self.assertEqual(output[1].dtype, torch.float32)
                self.assertEqual(output[0].device.type, "cpu")
                self.assertEqual(output[1].device.type, "cpu")
                self.assertTrue(torch.isfinite(output[0].float()).all())
                cursor = 0
                for allocation, length in zip(allocations, used):
                    self.assertTrue((output[0][cursor + length:cursor + allocation] == 0).all())
                    self.assertTrue(torch.isinf(output[1][:, cursor + length:cursor + allocation]).all())
                    cursor += allocation
                self.assertIs(module.get_qfa_section_info, original[0])
                self.assertIs(module._cpu_mxfp8_golden_sectioned, original[1])
                request, _ = request_from_call(self.golden.cpu_qfa_mxfp8, args, kwargs, golden=True)
                numerical_record(f"q={used},kv={kv}", output, self.provider.generate(request).summary)

    def test_real_golden_fd_and_multisection(self):
        torch = self.torch
        provider = MetadataProvider(load_json(ROOT / "examples/target_arch35.json"), REPO, ROOT / "build/metadata_helper")
        cases = [([1], [1], [32768], 1, 1, 512),
                 ([129], [129], [16384], 24, 24, 512)]
        for index, (allocated, used, kv, nq, nk, bs) in enumerate(cases):
            with self.subTest(case=index):
                args, kwargs = pa_inputs(torch, self.inputs, allocated, used, kv, nq, nk, bs)
                request, _ = request_from_call(self.golden.cpu_qfa_mxfp8, args, kwargs, golden=True)
                summary = provider.generate(request).summary
                if index == 0:
                    self.assertEqual(summary["header"][1], 1)
                else:
                    self.assertGreater(summary["header"][0], 1)
                with patch("qfa_mock.integrations.get_provider", return_value=provider):
                    out, lse = run_golden(self.golden.cpu_qfa_mxfp8, self.golden.mxfp8_golden_mod, args, kwargs)
                self.assertEqual(tuple(out.shape), (sum(allocated), nq, 128))
                self.assertEqual(tuple(lse.shape), (nq, sum(allocated)))
                self.assertTrue(torch.isfinite(out.float()).all())
                self.assertTrue(torch.isfinite(lse).all())
                numerical_record("fd" if index == 0 else "multisection", [out, lse], summary)

    def test_actual_ttk_discovery_registration_and_inherited_hooks(self):
        sys.path.insert(0, str(TTK))
        try:
            from ttk.core_modules.common.plugin.spec_loader import SpecLoader
            logging.getLogger().setLevel(logging.WARNING)
            loader = SpecLoader([str(ROOT / "ttk_plugin.py")])
            with patch.dict(os.environ, QFA_REPO=str(REPO)):
                found = loader.load("torch.ops.cann_ops_transformer.quant_flash_attn")
            self.assertIsNotNone(found)
            self.assertEqual(found.__name__, "QuantFlashAttnMockSpec")
            self.assertTrue(issubclass(found, self.base.QuantFlashAttnSpec))
            self.assertIs(found.customize_inputs, self.base.QuantFlashAttnSpec.customize_inputs)
            self.assertIs(found.compare, self.base.QuantFlashAttnSpec.compare)
            self.assertEqual(found.tolerance, self.base.QuantFlashAttnSpec.tolerance)
            # Separate native discovery must still select the original class.
            native = SpecLoader([str(REPO / "attention/quant_flash_attn/tests/assets")])
            self.assertEqual(native.load("torch.ops.cann_ops_transformer.quant_flash_attn").__name__, "QuantFlashAttnSpec")
            ACCEPTANCE_RECORDS["discovery"] = {"actual_ttk_loader": True, "mock_class": found.__name__,
                "inherited_inputs_compare_tolerance": True, "native_discovery": True,
                "plugin_path": str(ROOT / "ttk_plugin.py")}
        finally:
            sys.path.remove(str(TTK))

    def test_real_function_faults_restore_and_never_submit_metadata(self):
        args, kwargs = pa_inputs(self.torch, self.inputs, [8], [8], [64], 2, 1, 64)
        module = self.golden.mxfp8_golden_mod
        cases = [("_decode_qfa_sections", ValueError("decoder injected"), ValueError),
                 ("_cpu_mxfp8_golden_sectioned", RuntimeError("numeric injected"), MockError),
                 ("convert_q_bnsd_to_layout", RuntimeError("output injected"), RuntimeError)]
        for name, failure, expected in cases:
            with self.subTest(stage=name):
                originals = (module.get_qfa_section_info, module._cpu_mxfp8_golden_sectioned)
                with patch.object(module, name, side_effect=failure), \
                        patch.object(module, "broadcast_kv", side_effect=AssertionError("global fallback executed")), \
                        patch("qfa_mock.integrations.get_provider", return_value=self.provider):
                    protected = module._cpu_mxfp8_golden_sectioned
                    with self.assertRaises(expected):
                        run_golden(self.golden.cpu_qfa_mxfp8, module, args, kwargs)
                    self.assertIs(module.get_qfa_section_info, originals[0])
                    self.assertIs(module._cpu_mxfp8_golden_sectioned, protected)
                self.assertIs(module._cpu_mxfp8_golden_sectioned, originals[1])
