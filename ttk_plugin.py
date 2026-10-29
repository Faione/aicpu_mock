"""Optional Eager TTK plugin; original assets are unchanged. See README.md."""

import importlib.util
import os
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from qfa_mock.integrations import run_golden, run_preprocess
from qfa_mock.model import fail

_repo = os.environ.get("QFA_REPO")
if not _repo:
    fail("MISSING_CONFIG", "set QFA_REPO before loading the TTK plugin")
_path = Path(_repo) / "attention/quant_flash_attn/tests/assets/spec.py"
_spec = importlib.util.spec_from_file_location("qfa_original_assets_spec", _path)
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)


class QuantFlashAttnMockSpec(_base.QuantFlashAttnSpec):
    @staticmethod
    def golden(*args, **kwargs):
        if not os.environ.get("QFA_METADATA_CONFIG"):
            return _base.QuantFlashAttnSpec.golden(*args, **kwargs)
        impl = _base.load_impl_module("golden")
        return run_golden(impl.cpu_qfa_mxfp8, impl.mxfp8_golden_mod, args, kwargs)

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        if not os.environ.get("QFA_METADATA_CONFIG"):
            return _base.QuantFlashAttnSpec.npu_preprocess(*args, **kwargs)
        return run_preprocess(_base.load_impl_module("npu_preprocess"), args, kwargs)


__spec__ = {"torch.ops.cann_ops_transformer.quant_flash_attn": "QuantFlashAttnMockSpec"}
