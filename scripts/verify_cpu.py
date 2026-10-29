"""Run CPU acceptance and preserve source integrity, case evidence and test results."""

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("QFA_REPO", str(ROOT.parent / "ops-transformer")))
    parser.add_argument("--ttk", default=os.environ.get("QFA_TTK_REPO", str(ROOT.parent / "ops-test-kit")))
    parser.add_argument("--output", default=str(ROOT / "docs"))
    args = parser.parse_args()
    repo, ttk, output = Path(args.repo).resolve(), Path(args.ttk).resolve(), Path(args.output).resolve()
    if importlib.util.find_spec("torch") is None:
        parser.error("CPU acceptance requires Torch; install requirements-cpu.txt in a Python 3.10-3.12 venv")
    os.environ["QFA_REPO"], os.environ["QFA_TTK_REPO"] = str(repo), str(ttk)
    paths = [("ops-transformer/" + str(p.relative_to(repo)), p)
             for p in sorted((repo / "attention/quant_flash_attn/tests/assets").rglob("*.py"))]
    paths += [("ops-test-kit/" + str(p.relative_to(ttk)), p) for p in sorted((ttk / "ttk").rglob("*.py"))]
    def hashes():
        return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths}
    before = hashes()
    output.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = hashes()
    import torch
    import numpy
    from test_real_python import ACCEPTANCE_RECORDS
    from qfa_mock.build import verify_build
    report = {"date": "2026-10-29", "scope": "Host, loader/lifecycle, offline platform checks, real upstream CPU hooks/golden, actual TTK SpecLoader",
              "python": sys.version.split()[0], "torch": torch.__version__, "numpy": numpy.__version__,
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "passed": result.wasSuccessful() and not result.skipped,
              "upstream_file_count": len(before), "upstream_unchanged": before == after,
              "upstream_sha256_before": before, "upstream_sha256_after": after,
              "helper_build": verify_build(ROOT / "build/metadata_helper", repo),
              "cpu_acceptance": ACCEPTANCE_RECORDS,
              "cpu_boundaries": ["torch_npu import-only shim; device functions/real metadata raise immediately",
                                 "original CPU slot tests bypass only platform proof; real Tensor transfer stays on CPU"],
              "aicpu_parity": "not_run", "real_h2d": "not_run", "aicore_simulation_out_lse": "not_run",
              "target_mk001_confirmed": False}
    (output / "verification_latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "cpu_tests_latest.txt").write_text(stream.getvalue())
    print(json.dumps({key: report[key] for key in ("tests_run", "failures", "errors", "skipped", "passed", "upstream_file_count", "upstream_unchanged", "python", "torch", "numpy")}, indent=2))
    return 0 if report["passed"] and before == after else 1


if __name__ == "__main__":
    sys.exit(main())
