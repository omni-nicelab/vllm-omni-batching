import torch
from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import quack.utils as utils
from quack.reduce import warp_reduce
from quack.reduction_base import ReductionBase
from quack.cute_dsl_utils import torch2cute_dtype_map


# PyTorch has its own NVRTC, which may have a lower version than the system
# So try to disable PyTorch's NVRTC, or import NVRTC before PyTorch
import cuda.bindings.nvrtc as nvrtc
# print(f'NVRTC version: {nvrtc.nvrtcVersion()[1:]}')

import nvtx

from mh_cute_ops.attn_fp8.cute_preprocess_QK import PreprocessQK



@torch.compiler.disable
def preprocess_XW(
    x: torch.Tensor,
    int8: bool = True,
    chunk_size: int = 64,
) -> torch.Tensor:
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16 ], "Unsupported dtype"

    m, k = x.shape
    assert k % chunk_size == 0, "k must be divisible by chunk_size"

    device = x.device

    out = torch.empty(
        m*k, device=device, dtype=torch.int8 if int8 == True else torch.float8_e4m3fn
    ) 
    out_scale = torch.empty(
        m*k//chunk_size, device=device, dtype=torch.float32
    )

    dtype = torch2cute_dtype_map[x.dtype]

    convert_from_dlpack = lambda x: (
        from_dlpack(x.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=2, stride_order=(0, 1, 2), divisibility=128 // 16
        ).mark_compact_shape_dynamic(mode=1)
    )
    def convert_from_dlpack_fp8(x):
        torch_tensor_view = x.view(torch.uint8)
        cute_tensor = from_dlpack(torch_tensor_view, assumed_align=16).mark_compact_shape_dynamic(
            mode=2, stride_order=(0, 1, 2), divisibility=128 // 8
        ).mark_compact_shape_dynamic(mode=1)
        if int8 == True:
            cute_tensor.element_type = cutlass.Int8
        else:
            cute_tensor.element_type = cutlass.Float8E4M3FN
        return cute_tensor

    x_tensor = convert_from_dlpack(x.view(1, m, k))
    # print("x dim order", x.view(1, m, k).dim_order())
    # exit(0)
    out_tensor = convert_from_dlpack_fp8(out.view(1, m, k))
    out_scale_tensor = convert_from_dlpack(out_scale.view(1, k//chunk_size, m))

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, int8, chunk_size)
    if compile_key not in preprocess_XW.compile_cache:
        preprocess_XW_op = PreprocessQK(dtype, int8, chunk_size)
        preprocess_XW.compile_cache[compile_key] = cute.compile(
            preprocess_XW_op,
            x_tensor,
            out_tensor,
            out_scale_tensor,
            current_stream,
        )
    preprocess_XW.compile_cache[compile_key](
        x_tensor, out_tensor, out_scale_tensor, current_stream, 
    )

    # print("...... out shape", out.shape, "stride", out.stride())
    out_final = out.view(m, k)
    out_scale_final = out_scale.view(k//chunk_size, m)
    # print("out_final shape", out_final.shape, "stride", out_final.stride())

    return out_final, out_scale_final


preprocess_XW.compile_cache = {}


def test_preprocess_XW(m, k, chunk_size: int = 64, int8: bool = True):
    # torch.manual_seed(0)

    x = torch.randn(m, k, device="cuda", dtype=torch.float16)

    # PyTorch native conversion
    x1 = torch.empty(m, k, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for i in range(40):
        with nvtx.annotate("torch fp16 copy"):
            x1.copy_(x)
            torch.cuda.synchronize()
        
    def per_token_cast_to_fp8(x0: torch.Tensor, int8: bool = False, chunk_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        m, n = x0.shape

        x = x0.view(m, n//chunk_size, chunk_size)

        range_max = 448.0
        if int8 == True:
            range_max = 127.0

        x_amax = x.abs().float().amax(dim=-1).clamp(1e-4)
        sf = x_amax / range_max

        x_new = x.float() * (1.0 / sf.unsqueeze(-1))

        if int8 == True:
            x_new = torch.round(x_new).clamp(-127., 127.).to(torch.int8)
        else:
            x_new = x_new.to(torch.float8_e4m3fn)

        x_new = x_new.reshape(m, n)

        sf = sf.view(m, n//chunk_size)

        return x_new, sf.transpose(0, 1).contiguous()

    out_ref, out_ref_scale = per_token_cast_to_fp8(x, int8=int8, chunk_size=chunk_size)

    torch.cuda.synchronize()
    for i in range(40):
        with nvtx.annotate("cute preprocess V"):
            out, out_scale = preprocess_XW(x, int8=int8, chunk_size=chunk_size)
            torch.cuda.synchronize()
    out, out_scale = preprocess_XW(x, int8=int8, chunk_size=chunk_size)

    def compare_out(out_ref, out, flag=True):
        for i in range(10):
            print(f"out_ref[{i}] = {out_ref.flatten()[i]}, out[{i}] = {out.flatten()[i]}")

        out_ref_fp32 = out_ref.to(torch.float32)
        out_fp32 = out.to(torch.float32)

        print("0st element: ref ", out_ref_fp32.flatten()[0].item(), " vs out ", out_fp32.flatten()[0].item())
        print("max error: ", (out_ref_fp32 - out_fp32).abs().max())
        print("mean error: ", (out_ref_fp32 - out_fp32).abs().mean())

        if flag == True:
            torch.testing.assert_close(out_fp32, out_ref_fp32, atol=1e-3, rtol=1e-3)
    
    print("---------out--------------------")
    compare_out(out_ref, out, flag=True)
    print("--------------------------------")
    compare_out(out_ref_scale, out_scale, flag=True)
    print("--------------------------------")

    print("pass")


if __name__ == "__main__":
    m = 32000
    n = 3072
    # m = 16384
    # n = 128 * 24

    for chunk_size in [32, 64, 128]:
        for int8 in [False]:
            test_preprocess_XW(m, n, chunk_size, int8)
            print("--------------------------------")