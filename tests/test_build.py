import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from qfa_mock.build import ROOT, build_helper, verify_build
from qfa_mock.model import MockError, load_json, normalize_config, normalize_request, wire_request


class BuildTests(unittest.TestCase):
    def test_primary_adapter_does_not_import_legacy_patch_or_loader_modules(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import qfa_mock.assets; assert 'qfa_mock.integrations' not in sys.modules; "
            "assert 'qfa_mock.runtime_loader' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_only_actual_headers_are_needed_without_git_or_assets(self):
        original = load_json(ROOT / "build/metadata_helper.json")
        source = Path(original["repo"])
        with tempfile.TemporaryDirectory(prefix="qfa-minimal-") as directory:
            directory = Path(directory)
            headers = directory / "headers only with spaces"
            headers.mkdir()
            for name in original["dependency_files"]:
                if name == "@helper":
                    continue
                (headers / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, headers / name)
            self.assertFalse((headers / ".git").exists())
            self.assertFalse((headers / "attention/quant_flash_attn/tests/assets").exists())
            self.assertFalse(list(headers.rglob("*.cpp")))
            helper = directory / "metadata_helper"
            built = build_helper(headers, helper)
            self.assertEqual(built["source_digest"], original["source_digest"])
            verify_build(helper, headers)
            data = directory / "metadata.bin"
            request = wire_request(normalize_config(load_json(ROOT / "examples/target_arch92.json")),
                                   normalize_request(load_json(ROOT / "examples/request.json")))
            result = subprocess.run([str(helper), str(data)], input=request, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(),
                             "e504413ef5b29d24d6742f588d70fa20e4909814605d6f9455fec20096139c9f")

    def test_missing_headers_report_compiler_error_and_preserve_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            helper = directory / "helper"
            helper.write_bytes(b"previous working binary")
            manifest = Path(str(helper) + ".json")
            manifest.write_text("previous manifest")
            with self.assertRaises(MockError) as caught:
                build_helper(directory, helper)
            self.assertEqual(caught.exception.code, "BUILD_FAILED")
            self.assertIn("section_stream_k.h", str(caught.exception))
            self.assertEqual(helper.read_bytes(), b"previous working binary")
            self.assertEqual(manifest.read_text(), "previous manifest")
