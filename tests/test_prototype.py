import copy
import importlib.util
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qfa_mock import MetadataProvider, MockError
from qfa_mock.build import ROOT, verify_build
from qfa_mock.integrations import (metadata_for_preprocess, request_from_call, run_golden,
                                   run_preprocess)
from qfa_mock.model import canonical, load_json, normalize_config, normalize_request
from qfa_mock.validation import validate_metadata


REPO = Path(os.environ.get("QFA_REPO", str(ROOT.parent / "ops-transformer")))
HELPER = ROOT / "build/metadata_helper"


class FakeTensor:
    def __init__(self, shape, device="npu:0"):
        self.shape, self.device = shape, device

    def detach(self):
        raise AssertionError("Q/K/V contents must never be read by the adapter")


def make_call(body=None, golden=False):
    # Simulate the actual assets signatures without importing Torch/CANN.
    if golden:
        def call(q, k, v, quant_mode=1, cu_seqlens_q_t=None, cu_seqlens_kv_t=None,
                 seqused_q_t=None, seqused_kv_t=None, attn_mask_t=None, mask_mode=3, win_left=-1, win_right=-1,
                 max_seqlen_q=128, max_seqlen_kv=1024, layout_q="TND", layout_q_descale="TND",
                 layout_kv="PA_BNBD", layout_out="TND", **kwargs):
            return body() if body else None
    else:
        def call(q, k, v, quant_mode=1, cu_seqlens_q=None, cu_seqlens_kv=None,
                 seqused_q=None, seqused_kv=None, attn_mask=None, mask_mode=3, win_left=-1, win_right=-1,
                 max_seqlen_q=128, max_seqlen_kv=1024, layout_q="TND", layout_q_descale="TND",
                 layout_kv="PA_BNBD", layout_out="TND", **kwargs):
            return body() if body else None
    return call


class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_build(HELPER, REPO)

    def setUp(self):
        self.request = load_json(ROOT / "examples/request.json")
        self.config92 = load_json(ROOT / "examples/target_arch92.json")
        self.config35 = load_json(ROOT / "examples/target_arch35.json")
        self.provider = MetadataProvider(self.config92, REPO, HELPER)

    def assert_error(self, code, function, *args, **kwargs):
        with self.assertRaises(MockError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_baseline_matches_existing_binary_exactly(self):
        result = self.provider.generate(self.request)
        reference = REPO / "scratch/qfa-metadata-mock/arch92_80_80.bin"
        self.assertEqual(result.data, reference.read_bytes())
        self.assertEqual(result.summary["header"][:6], [1, 0, 128, 256, 80, 80])
        self.assertEqual(result.summary["active_aic"], [64])

    def test_batch_varlen_tnd_and_masks(self):
        for mask in (0, 3, 5):
            with self.subTest(mask=mask):
                request = {**self.request, "layout_kv": "TND", "cu_seqlens_q": [0, 73, 330],
                           "cu_seqlens_kv": [0, 129, 1154], "seqused_q": None, "seqused_kv": None,
                           "max_seqlen_q": 257, "max_seqlen_kv": 1025, "num_heads_q": 12,
                           "num_heads_kv": 3, "mask_mode": mask}
                result = self.provider.generate(request)
                self.assertEqual(result.summary["validation"]["q_lengths"], [73, 257])
                self.assertEqual(result.summary["validation"]["kv_lengths"], [129, 1025])
                self.assertEqual(result.summary["header"][15], 0)

    def test_padding_zero_lengths_and_full_mask_need_init(self):
        variants = [
            {"cu_seqlens_q": [0, 128, 256], "seqused_q": [64, 0], "seqused_kv": [1024, 0]},
            {"cu_seqlens_q": [0, 0], "seqused_q": [0], "seqused_kv": [0]},
            {"cu_seqlens_q": [0, 128], "seqused_q": [128], "seqused_kv": [8]},
        ]
        for updates in variants:
            for config in (self.config92, self.config35):
                with self.subTest(updates=updates, target=config["target"]):
                    result = MetadataProvider(config, REPO, HELPER).generate({**self.request, **updates})
                    self.assertEqual(result.summary["header"][15], 1)

    def test_real_fd_and_multisection_are_generated(self):
        provider = MetadataProvider(self.config35, REPO, HELPER)
        cases = [
            {"num_heads_q": 8, "num_heads_kv": 1, "cu_seqlens_q": [0, 32], "seqused_q": [32],
             "seqused_kv": [65536], "max_seqlen_q": 32, "max_seqlen_kv": 65536},
            {"cu_seqlens_q": [0, 512, 1024], "seqused_q": [512, 512], "seqused_kv": [65536, 65536],
             "max_seqlen_q": 512, "max_seqlen_kv": 65536},
        ]
        for index, updates in enumerate(cases):
            result = provider.generate({**self.request, **updates})
            self.assertEqual(result.summary["header"][1], 1)
            self.assertGreater(result.summary["validation"]["fd_tasks"], 0)
            if index:
                self.assertGreater(result.summary["header"][0], 1)

    def test_cache_is_immutable_and_includes_raw_allocations(self):
        first = self.provider.generate(self.request)
        second = self.provider.generate(copy.deepcopy(self.request))
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        altered_summary = second.summary
        altered_summary["header"][0] = 999
        self.assertEqual(self.provider.generate(self.request).summary["header"][0], 1)
        # Effective lengths are identical, but the second request has padded Q allocation.
        changed = self.provider.generate({**self.request, "cu_seqlens_q": [0, 256], "max_seqlen_q": 256})
        self.assertFalse(changed.cache_hit)
        self.assertNotEqual(first.summary["request_digest"], changed.summary["request_digest"])
        self.assertEqual(changed.summary["header"][15], 1)
        changed_length = self.provider.generate({**self.request, "seqused_kv": [999]})
        self.assertNotEqual(first.summary["request_digest"], changed_length.summary["request_digest"])

    def test_disabled_cache_and_changed_cores(self):
        config = copy.deepcopy(self.config92)
        config["cache"]["enabled"] = False
        provider = MetadataProvider(config, REPO, HELPER)
        self.assertFalse(provider.generate(self.request).cache_hit)
        self.assertFalse(provider.generate(self.request).cache_hit)
        config["target"].update(aic_core_num=16, aiv_core_num=16)
        small = MetadataProvider(config, REPO, HELPER).generate(self.request)
        self.assertEqual(small.summary["header"][4:6], [16, 16])
        self.assertNotEqual(small.summary["shape"], self.provider.generate(self.request).summary["shape"])

    def test_unsupported_and_invalid_requests_fail_before_helper(self):
        for updates, code in [({"execution_mode": "incremental"}, "UNSUPPORTED_REQUEST"),
                              ({"kv_cache": {}}, "UNSUPPORTED_FIELD"),
                              ({"win_left": 64}, "UNSUPPORTED_REQUEST"),
                              ({"head_dim": 256}, "UNSUPPORTED_REQUEST"),
                              ({"layout_q_descale": "N2TGD"}, "UNSUPPORTED_REQUEST"),
                              ({"max_seqlen_q": -1}, "INVALID_INPUT"),
                              ({"seqused_q": [129]}, "INVALID_INPUT"),
                              ({"cu_seqlens_q": [1, 128]}, "INVALID_INPUT"),
                              ({"num_heads_kv": 3}, "INVALID_INPUT"),
                              ({"is_grad_enabled": True}, "UNSUPPORTED_REQUEST")]:
            with self.subTest(updates=updates), patch("qfa_mock.provider.subprocess.run") as run:
                self.assert_error(code, self.provider.generate, {**self.request, **updates})
                run.assert_not_called()
        config = copy.deepcopy(self.config92)
        config["target"]["kernel_arch"] = "arch35"
        self.assert_error("TARGET_MISMATCH", normalize_config, config)
        config["target"].update(soc_version="mk001", kernel_arch="arch92")
        self.assert_error("UNSUPPORTED_TARGET", normalize_config, config)

    def test_corrupt_metadata_is_rejected(self):
        result = self.provider.generate(self.request)
        request = normalize_request(self.request)
        # Corrupt a task endpoint, an unused slot, and the backward row independently.
        for index, value in ((16 + 3, 999999), (16 + 64 * 16 + 7, 1), (result.summary["shape"][1], 1), (15, 1)):
            with self.subTest(index=index):
                data = bytearray(result.data)
                struct.pack_into("<I", data, index * 4, value)
                summary = result.summary
                summary["header"] = list(struct.unpack_from("<16I", data))
                self.assert_error("INVALID_METADATA", validate_metadata, bytes(data), summary,
                                  self.provider.config, request, REPO)

    def test_fd_corruption_is_rejected(self):
        request = normalize_request({**self.request, "num_heads_q": 8, "num_heads_kv": 1,
                                     "cu_seqlens_q": [0, 32], "seqused_q": [32], "seqused_kv": [65536],
                                     "max_seqlen_q": 32, "max_seqlen_kv": 65536})
        provider = MetadataProvider(self.config35, REPO, HELPER)
        result = provider.generate(request)
        s = result.summary
        fd_start = 16 + s["header"][0] * s["header"][4] * 16
        data = bytearray(result.data)
        struct.pack_into("<I", data, (fd_start + 3) * 4, 999999)
        self.assert_error("INVALID_METADATA", validate_metadata, bytes(data), s, provider.config, request, REPO)

    def test_missing_or_stale_helper_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "helper"
            self.assert_error("HELPER_NOT_BUILT", MetadataProvider, self.config92, REPO, helper)
            helper.write_bytes(HELPER.read_bytes())
            manifest = load_json(str(HELPER) + ".json")
            manifest["source_digest"] = "stale"
            Path(str(helper) + ".json").write_text(canonical(manifest))
            self.assert_error("STALE_BUILD", MetadataProvider, self.config92, REPO, helper)

    def test_helper_failure_and_malformed_wire(self):
        import subprocess

        with patch("qfa_mock.provider.subprocess.run", return_value=SimpleNamespace(returncode=1, stderr="injected failure")):
            self.assert_error("HELPER_FAILED", self.provider.generate, self.request)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metadata.bin"
            process = subprocess.run([str(HELPER), str(output)], input="QFA_HOST_V1\n-1\n", text=True, capture_output=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse(output.exists())

    def call_inputs(self, golden=False):
        args = (FakeTensor((128, 64, 128)), FakeTensor((2, 8, 512, 128)), FakeTensor((2, 8, 512, 128)))
        suffix = "_t" if golden else ""
        kwargs = {"cu_seqlens_q" + suffix: [0, 128], "seqused_q" + suffix: [128], "seqused_kv" + suffix: [1024]}
        return args, kwargs

    def test_both_adapters_derive_identical_request_without_reading_qkv(self):
        args, kwargs = self.call_inputs()
        preprocess, _ = request_from_call(make_call(), args, kwargs)
        args, kwargs = self.call_inputs(golden=True)
        golden, _ = request_from_call(make_call(golden=True), args, kwargs, golden=True)
        self.assertEqual(golden, preprocess)
        self.assertEqual(golden, normalize_request(self.request))
        kwargs["seqused_q_values"] = [64]
        self.assert_error("REQUEST_MISMATCH", request_from_call, make_call(golden=True), args, kwargs, golden=True)

    def test_device_adapter_copies_same_plan_and_rejects_core_mismatch(self):
        args, kwargs = self.call_inputs()
        request, _ = request_from_call(make_call(), args, kwargs)
        arguments = {key: value for key, value in request.items()
                     if key not in ("protocol_version", "execution_mode", "win_left", "win_right", "is_grad_enabled")}
        arguments["batch_size"] = None  # Native TND call derives B from lengths.
        fake_torch = SimpleNamespace(npu=SimpleNamespace(get_device_properties=lambda device:
                                    SimpleNamespace(cube_core_num=80, vector_core_num=80)))
        result = SimpleNamespace(device_tensor=lambda anchor: (self.provider.generate(request).data, anchor.device))
        fake_provider = SimpleNamespace(config=self.provider.config, generate=lambda r: result)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            data, device = metadata_for_preprocess(arguments, request, fake_provider, args[0])
            self.assertEqual(data, self.provider.generate(request).data)
            self.assertEqual(device, "npu:0")
            fake_torch.npu.get_device_properties = lambda device: SimpleNamespace(cube_core_num=36, vector_core_num=72)
            self.assert_error("TARGET_MISMATCH", metadata_for_preprocess, arguments, request, fake_provider, args[0])

    def test_original_assets_causal_mask_is_passed_through(self):
        for golden in (False, True):
            with self.subTest(golden=golden):
                args, kwargs = self.call_inputs(golden=golden)
                suffix = "_t" if golden else ""
                kwargs["attn_mask" + suffix] = FakeTensor((2048, 2048))
                kwargs["attn_mask_shape"] = (2048, 2048)
                request, _ = request_from_call(make_call(golden=golden), args, kwargs, golden=golden)
                self.assertEqual(request, normalize_request(self.request))
                kwargs["attn_mask_shape"] = (128, 128)
                self.assert_error("UNSUPPORTED_REQUEST", request_from_call, make_call(golden=golden), args, kwargs, golden=golden)
                kwargs["attn_mask_shape"] = (2048, 2048)
                kwargs["mask_mode"] = 0
                self.assert_error("UNSUPPORTED_REQUEST", request_from_call, make_call(golden=golden), args, kwargs, golden=golden)

    def test_golden_cannot_silently_fall_back_and_patches_restore(self):
        module = SimpleNamespace(get_qfa_section_info=lambda q, kv: None,
                                 _decode_qfa_sections=lambda words, q, kv: "decoded")
        fallback = []

        def broken(*args):
            raise RuntimeError("injected sectioned golden error")

        module._cpu_mxfp8_golden_sectioned = broken
        original_entry = module.get_qfa_section_info

        def body():
            info = module.get_qfa_section_info([128], [1024])
            try:
                return module._cpu_mxfp8_golden_sectioned(info)
            except Exception:
                fallback.append(True)
                return "wrong global reference"

        args, kwargs = self.call_inputs(golden=True)
        with patch("qfa_mock.integrations.get_provider", return_value=self.provider):
            self.assert_error("GOLDEN_FAILED", run_golden, make_call(body, golden=True), module, args, kwargs)
        self.assertEqual(fallback, [])
        self.assertIs(module.get_qfa_section_info, original_entry)
        self.assertIs(module._cpu_mxfp8_golden_sectioned, broken)

    def test_preprocess_works_with_golden_disabled(self):
        args, kwargs = self.call_inputs()
        module = SimpleNamespace(run_metadata=lambda a, anchor: "native must not run")
        original = module.run_metadata
        module.run = make_call(lambda: {"metadata": module.run_metadata({}, args[0])})
        with patch("qfa_mock.integrations.get_provider", return_value=self.provider), \
                patch("qfa_mock.integrations.metadata_for_preprocess", return_value="mock plan"):
            self.assertEqual(run_preprocess(module, args, kwargs), {"metadata": "mock plan"})
        self.assertIs(module.run_metadata, original)

    def test_plugin_without_config_delegates_to_native(self):
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "attention/quant_flash_attn/tests/assets/spec.py"
            spec_path.parent.mkdir(parents=True)
            spec_path.write_text("class QuantFlashAttnSpec:\n"
                                 "    @staticmethod\n    def golden(*args, **kwargs): return 'native golden'\n"
                                 "    @staticmethod\n    def npu_preprocess(*args, **kwargs): return 'native preprocess'\n")
            environment = dict(os.environ)
            environment.pop("QFA_METADATA_CONFIG", None)
            environment["QFA_REPO"] = directory
            spec = importlib.util.spec_from_file_location("test_qfa_plugin", ROOT / "ttk_plugin.py")
            module = importlib.util.module_from_spec(spec)
            with patch.dict(os.environ, environment, clear=True), patch.dict("sys.modules"):
                spec.loader.exec_module(module)
                self.assertEqual(module.QuantFlashAttnMockSpec.golden(), "native golden")
                self.assertEqual(module.QuantFlashAttnMockSpec.npu_preprocess(), "native preprocess")

    def test_bundle_is_reproducible_and_never_overwrites(self):
        result = self.provider.generate(self.request)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case"
            result.write(output)
            self.assertEqual((output / "metadata.bin").read_bytes(), result.data)
            self.assertEqual(load_json(output / "request.json"), normalize_request(self.request))
            with self.assertRaises(FileExistsError):
                result.write(output)


if __name__ == "__main__":
    unittest.main()
