import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qfa_mock.build import ROOT, build_helper, verify_build, check_sources, COMPILE_SOURCES, ASSET_SOURCES, AICPU_SOURCES
from qfa_mock.model import MockError, load_json, normalize_config, normalize_request, wire_request


class BuildTests(unittest.TestCase):
    def test_primary_adapter_does_not_import_legacy_patch_or_loader_modules(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import qfa_mock.assets; assert 'qfa_mock.integrations' not in sys.modules; "
            "assert 'qfa_mock.runtime_loader' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_complete_sources_build_without_git(self):
        original = load_json(ROOT / "build/metadata_helper.json")
        source = Path(original["repo"])
        with tempfile.TemporaryDirectory(prefix="qfa-minimal-") as directory:
            directory = Path(directory)
            headers = directory / "sources with spaces"
            headers.mkdir()
            for name in (*COMPILE_SOURCES, *ASSET_SOURCES, *AICPU_SOURCES):
                (headers / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, headers / name)
            self.assertFalse((headers / ".git").exists())
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

    def test_all_missing_sources_are_listed_and_previous_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            helper = directory / "helper"
            helper.write_bytes(b"previous working binary")
            manifest = Path(str(helper) + ".json")
            manifest.write_text("previous manifest")
            with self.assertRaises(MockError) as caught:
                build_helper(directory, helper)
            self.assertEqual(caught.exception.code, "SOURCE_MISSING")
            self.assertIn("section_stream_k.h", str(caught.exception))
            self.assertIn("quant_flash_attn_golden.py", str(caught.exception))
            self.assertIn("quant_flash_attn_metadata_aicpu.cpp", str(caught.exception))
            self.assertEqual({item["group"] for item in caught.exception.missing}, {"compile", "assets", "aicpu_reference"})
            self.assertTrue(all(Path(item["path"]).is_absolute() and item["required_for"] for item in caught.exception.missing))
            self.assertEqual(helper.read_bytes(), b"previous working binary")
            self.assertEqual(manifest.read_text(), "previous manifest")

    def test_any_required_source_missing_blocks_compilation(self):
        import os
        original = Path(os.environ.get("QFA_REPO", str(ROOT.parent / "ops-transformer")))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in (*COMPILE_SOURCES, *ASSET_SOURCES, *AICPU_SOURCES):
                (source / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original / name, source / name)
            self.assertFalse((source / ".git").exists())
            check_sources(source)
            helper = source / "helper"
            helper.write_bytes(b"previous binary")
            manifest = Path(str(helper) + ".json")
            manifest.write_text("previous manifest")
            for group, name in (("compile", COMPILE_SOURCES[0]), ("assets", ASSET_SOURCES[0]),
                                ("assets", ASSET_SOURCES[-1]), ("aicpu_reference", AICPU_SOURCES[0])):
                with self.subTest(group=group, source=name):
                    missing = source / name
                    content = missing.read_bytes()
                    missing.unlink()
                    try:
                        with patch("qfa_mock.build.subprocess.run") as compiler:
                            with self.assertRaises(MockError) as caught:
                                build_helper(source, helper)
                        compiler.assert_not_called()
                        self.assertEqual(caught.exception.code, "SOURCE_MISSING")
                        self.assertEqual([(item["group"], item["relative_path"]) for item in caught.exception.missing],
                                         [(group, name)])
                        self.assertEqual(helper.read_bytes(), b"previous binary")
                        self.assertEqual(manifest.read_text(), "previous manifest")
                    finally:
                        missing.write_bytes(content)

    def test_additional_missing_include_is_reported_by_compiler(self):
        original = Path(load_json(ROOT / "build/metadata_helper.json")["repo"])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in (*COMPILE_SOURCES, *ASSET_SOURCES, *AICPU_SOURCES):
                (source / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original / name, source / name)
            header = source / COMPILE_SOURCES[2]
            header.write_text('#include "missing_extra_dependency.h"\n' + header.read_text())
            with self.assertRaises(MockError) as caught:
                build_helper(source, source / "helper")
            self.assertEqual(caught.exception.code, "BUILD_FAILED")
            self.assertIn("missing_extra_dependency.h", str(caught.exception))

    def test_cli_missing_report_is_structured_and_complete(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-m", "qfa_mock", "build", "--repo", directory],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            details = json.loads(result.stderr)["error"]
            self.assertEqual(details["code"], "SOURCE_MISSING")
            self.assertGreater(len(details["missing"]), 3)
            self.assertEqual({item["group"] for item in details["missing"]}, {"compile", "assets", "aicpu_reference"})

    def test_cli_does_not_allow_skipping_required_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-m", "qfa_mock", "build", "--repo", directory,
                                     "--compile-only"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unrecognized arguments: --compile-only", result.stderr)
