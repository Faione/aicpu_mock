"""Test-only import boundary; all tensor math and upstream function bodies are real.

The CPU Torch wheel cannot import torch_npu. This shim supplies only availability
and trap functions: any attempted device operation or metadata submission fails.
It is never imported by the production plugin.
"""

from contextlib import ExitStack, contextmanager
import importlib.machinery
import sys
import types
from unittest.mock import patch


def forbidden(*args, **kwargs):
    raise AssertionError("native metadata or device operation must not execute in CPU acceptance")


@contextmanager
def cpu_import_boundary(torch):
    # Keep extension modules outside patch.dict(sys.modules)' restoration snapshot.
    # Reinitializing ml_dtypes against the same NumPy runtime is not supported.
    import ml_dtypes  # noqa: F401

    npu = types.ModuleType("torch_npu")
    npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    npu.npu = types.SimpleNamespace(is_available=lambda: False)
    package = types.ModuleType("cann_ops_transformer")
    package.__path__ = []
    ops = types.ModuleType("cann_ops_transformer.ops")
    ops.quant_flash_attn_metadata = forbidden
    ops.quant_flash_attn = forbidden
    package.ops = ops
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {"torch_npu": npu, "cann_ops_transformer": package,
                                                    "cann_ops_transformer.ops": ops}))
        stack.enter_context(patch.object(torch, "npu", npu.npu, create=True))
        stack.enter_context(patch.object(torch.ops.cann_ops_transformer, "quant_flash_attn_metadata", forbidden, create=True))
        stack.enter_context(patch.object(torch.Tensor, "npu", forbidden, create=True))
        yield


def pa_inputs(torch, input_module, q_alloc, q_used, kv_used, nq=4, nk=1, block_size=64):
    """Allocate actual native-layout slots, then call the original input generator."""
    def cu(lengths):
        result = [0]
        for length in lengths:
            result.append(result[-1] + length)
        return result
    blocks = max(1, sum((length + block_size - 1) // block_size for length in kv_used))
    max_blocks = max(1, max((length + block_size - 1) // block_size for length in kv_used))
    total_q = sum(q_alloc)
    slots = [torch.empty((total_q, nq, 128), dtype=torch.float8_e4m3fn),
             torch.empty((blocks, nk, block_size, 128), dtype=torch.float8_e4m3fn),
             torch.empty((blocks, nk, block_size, 128), dtype=torch.float8_e4m3fn),
             torch.empty((total_q, nq, 2, 2), dtype=torch.float8_e8m0fnu),
             torch.empty((blocks, nk, block_size, 2, 2), dtype=torch.float8_e8m0fnu),
             torch.empty((blocks, nk, block_size // 64, 128, 2), dtype=torch.float8_e8m0fnu),
             torch.empty((len(q_alloc), max_blocks), dtype=torch.int32),
             torch.ones((1,), dtype=torch.float32),
             torch.tensor(cu(q_alloc), dtype=torch.int32), torch.tensor(cu(kv_used), dtype=torch.int32),
             torch.tensor(q_used, dtype=torch.int32), torch.tensor(kv_used, dtype=torch.int32),
             None, torch.empty((2048, 2048), dtype=torch.int8), None,
             None, None, None]
    args = tuple(slots[:6] + [1] + slots[6:])
    kwargs = {"N_q": nq, "N_kv": nk, "D": 128, "head_dim_v": None, "batch_size": None,
              "cu_seqlens_q_values": cu(q_alloc), "cu_seqlens_kv_values": cu(kv_used),
              "seqused_q_values": q_used, "seqused_kv_values": kv_used,
              "max_seqlen_q": max(1, max(q_alloc)), "max_seqlen_kv": max(1, max(kv_used)),
              "mask_mode": 3, "win_left": -1, "win_right": -1,
              "layout_q": "TND", "layout_q_descale": "TND", "layout_out": "TND", "layout_kv": "PA_BNBD",
              "enable_pa": True, "kv_cache_layout": "PA_BNBD", "q_scale_layout": "TND",
              "block_size": block_size, "return_softmax_lse": True, "softmax_scale": 128**-0.5,
              "attn_mask_shape": (2048, 2048), "attn_mask_dtype": "int8", "p_scale_value": 1.0}
    input_module.generate_qfa_mxfp8_inputs(*args, **kwargs)
    return args, kwargs
