"""Explicit mock-only Eager TTK plugin; choose original assets for native mode."""

import os
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
for _name, _module in list(sys.modules.items()):
    if _name == "qfa_mock" or _name.startswith("qfa_mock."):
        _suffix = _name.split(".")[1:]
        _expected = _HERE / "qfa_mock" / ("/".join(_suffix) + ".py" if _suffix else "__init__.py")
        if not getattr(_module, "__file__", None) or Path(_module.__file__).resolve() != _expected.resolve():
            raise RuntimeError(f"qfa_mock support package source conflict: {_name}")
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from qfa_mock.integrations import get_provider, run_golden, run_preprocess
from qfa_mock.model import fail
from qfa_mock.runtime_loader import load_impl, load_original_spec

_repo = os.environ.get("QFA_REPO")
if not _repo:
    fail("MISSING_CONFIG", "set QFA_REPO before loading the TTK plugin")
_repo = Path(_repo).resolve()
_base = load_original_spec(_repo, __file__)


class QuantFlashAttnMockSpec(_base.QuantFlashAttnSpec):
    @staticmethod
    def golden(*args, **kwargs):
        get_provider(expected_repo=_repo)
        impl = load_impl(_base, _repo, "golden")
        return run_golden(impl.cpu_qfa_mxfp8, impl.mxfp8_golden_mod, args, kwargs)

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        get_provider(expected_repo=_repo)
        return run_preprocess(load_impl(_base, _repo, "npu_preprocess"), args, kwargs)


__spec__ = {"torch.ops.cann_ops_transformer.quant_flash_attn": "QuantFlashAttnMockSpec"}
