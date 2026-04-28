# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import operator
import torch
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
from cutlass import Constexpr, Float32, Int32, const_expr, Boolean
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from quack.reduction_base import ReductionBase
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.reduce import warp_reduce

import nvtx

from mh_cute_ops.attn_fp8.utils import round_fp32_rn

# import deep_gemm
# from deep_gemm.utils.math import per_token_cast_to_fp8, per_block_cast_to_fp8
# # from DeepGEMM.tests.generators import generate_normal
# # import sys
# # sys.path.append("../DeepGEMM/tests/generators")
# # from tests.generators import generate_normal, MajorTypeAB
# # from tests.generators import MajorTypeAB

# from deep_gemm.testing import calc_diff
import enum


# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class PreprocessQK(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, int8: bool = False, chunk_size: int = 64):
        super().__init__(dtype, N=128, stage=1) 
        assert dtype.width == 16, "Input should be fp16/bf16"
        self.cluster_n = 1
        self.int8 = const_expr(True) if int8 == True else const_expr(False)

        self.chunk_size = const_expr(32) if chunk_size == 32 else const_expr(64) if chunk_size == 64 else const_expr(128)

    def _get_tv_layout2(self):

        tiler_mn = (128, 128)

        tv_layout_input = cute.make_layout(
            ((8,128), (8,2)),
            stride=(
                (128*8,1),
                (128, 128*8*8)
            ),
        )

        return tiler_mn, tv_layout_input, 

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        mScale: cute.Tensor,
        stream: cuda.CUstream,
    ):
        b, s, hd = mX.shape
        _b, s_padding, _hd = mO.shape

        assert mX.element_type == self.dtype

        tiler_mn, tv_layout_input = self._get_tv_layout2()

        num_threads = cute.size(tv_layout_input, mode=[0])
        # print("num_threads", num_threads)
        num_warps = num_threads // cute.arch.WARP_SIZE

        grid = [
            cute.ceil_div(s_padding, tiler_mn[0]),
            cute.ceil_div(hd, tiler_mn[1]),
            b]
        self.kernel(self.int8, mX, mO, mScale, tv_layout_input, tiler_mn, self.chunk_size).launch(
            grid=grid,
            block=[num_threads, 1, 1],
            cluster=None,
            # smem=self._smem_size_in_bytes((tiler_mn[0], tiler_mn[1]), num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        int8: cutlass.Constexpr[bool],
        mX: cute.Tensor,
        mO: cute.Tensor,
        mScale: cute.Tensor,
        tv_layout_input: cute.Layout,
        tiler_mn: cute.Shape,
        chunk_size: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, batch = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sX0 = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout((tiler_mn[0], tiler_mn[1]), order=(1, 0)), byte_alignment=2
        )
        sX = cute.local_tile(
            sX0, tiler_mn, (0, 0)
        )

        shape = (mX.shape[1], mX.shape[2])
        idX = cute.make_identity_tensor(shape)
        cX = cute.local_tile(idX, tiler_mn, (bidx, bidy))

        gX = cute.local_tile(mX[(batch, None, None)], tiler_mn, (bidx, bidy))
        gO = cute.local_tile(mO[(batch, None, None)], tiler_mn, (bidx, bidy))
        num_scale_per_128 = const_expr(128 // chunk_size)
        gScale = cute.local_tile(mScale[(batch, None, None)], (num_scale_per_128, 128), (bidy, bidx))

        # declare the atoms which will be used later for memory copy
        copy_atom_load_X_async = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mO.element_type, num_bits_per_copy=64
        )

        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X_async, tv_layout_input, tiler_mn).get_slice(
            tidx
        )
        thr_copy_O = cute.make_tiled_copy(copy_atom_store_O, tv_layout_input, tiler_mn).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]
        tXrO_fp16 = cute.make_fragment_like(tXrO, mX.element_type)

        row = tXcX[0][0]
        if row < shape[0]:
            cute.copy(copy_atom_load_X_async, tXgX, tXsX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.arch.sync_threads()

        cute.autovec_copy(tXsX, tXrX)

        tXrX_fp32_SSA = tXrX.load().to(cutlass.Float32)
        tXrX_fp32_SSA_abs = cute.where(tXrX_fp32_SSA > 0., tXrX_fp32_SSA, -tXrX_fp32_SSA)

        # print("tXrX_fp32_SSA_abs shape", tXrX_fp32_SSA_abs.shape)
        max_SSA = tXrX_fp32_SSA_abs.reduce(cute.ReductionOp.MAX, init_val=1e-4, reduction_profile=((1,1 if const_expr(chunk_size > 64) else None),None,None))
        num_threads = 4 if const_expr(chunk_size == 32) else 8 
        max_SSA = warp_reduce(max_SSA, cute.arch.fmax, width=num_threads)

        if const_expr(int8):
            one_over_range_max = cute.Float32(1.0 / 127.0)
        else:
            one_over_range_max = cute.Float32(1.0 / 448.0)
        scale_SSA = max_SSA * one_over_range_max
        # print("scale_SSA shape", scale_SSA.shape)

        scale_SSA_broadcast = cute.TensorSSA(scale_SSA.ir_value(), ((1,scale_SSA.shape[0]),1,1), scale_SSA.dtype).broadcast_to(tXrX_fp32_SSA.shape)
        tXrO_fp32_SSA = tXrX_fp32_SSA * (cute.Float32(1.0) / scale_SSA_broadcast)
        if const_expr(int8 == True):
            tXrO_fp32_SSA = cute.where(tXrO_fp32_SSA > 127., 127., tXrO_fp32_SSA)
            tXrO_fp32_SSA = cute.where(tXrO_fp32_SSA < -127., -127., tXrO_fp32_SSA)
            # round the fp32 values to nearest integer
            tmp = cute.make_fragment_like(tXrO, cutlass.Float32)
            tmp.store(tXrO_fp32_SSA)
            for i in cutlass.range_constexpr(cute.size(tmp)):
                tmp[i] = round_fp32_rn(tmp[i])
            tXrO.store(tmp.load().to(cutlass.Int8))
        else:
            tXrO.store(tXrO_fp32_SSA.to(cutlass.Float8E4M3FN))

        scale = cute.make_fragment(scale_SSA.shape, scale_SSA.dtype)
        scale.store(scale_SSA)

        if row >= shape[0]:
            tXrO.fill(0)
            scale.fill(0)

        cute.copy(copy_atom_store_O, tXrO, tXgO)

        if const_expr(chunk_size == 32):
            if tidx % 4 == 0:
                if tidx % 8 == 0:
                    gScale[0, tidx // 8] = scale[0]
                    gScale[2, tidx // 8] = scale[1]
                else:
                    gScale[1, tidx // 8] = scale[0]
                    gScale[3, tidx // 8] = scale[1]
        else:
            if tidx % num_threads == 0:
                for i in cutlass.range_constexpr(num_scale_per_128):
                    gScale[i, tidx // num_threads] = scale[i]
        


@torch.compiler.disable
def preprocess_QK(
    x: torch.Tensor,
    int8: bool = True,
    chunk_size: int = 64, # this is for token dimension
    padding_size: int = 128, # this is for sequence length dimension
) -> torch.Tensor:
    assert x.dim() == 4, "Input must be 4D"
    assert x.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16 ], "Unsupported dtype"

    b, s, h, d = x.shape
    assert d == chunk_size, "chunk_size must be equal to d"

    s_padding = (s + padding_size - 1) // padding_size * padding_size

    device = x.device

    out = torch.empty(
        b*s_padding*h*d, device=device, dtype=torch.int8 if int8 == True else torch.float8_e4m3fn
    ) 
    out_scale = torch.empty(
        b*s_padding*h, device=device, dtype=torch.float32
    )

    dtype = torch2cute_dtype_map[x.dtype]

    convert_from_dlpack = lambda x: (
        from_dlpack(x.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        ).mark_compact_shape_dynamic(
            mode=1
        ).mark_compact_shape_dynamic(
            mode=2, divisibility = 128 // 16
        )
    )
    def convert_from_dlpack_fp8(x):
        torch_tensor_view = x.view(torch.uint8)
        cute_tensor = from_dlpack(torch_tensor_view, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        ).mark_compact_shape_dynamic(
            mode=1
        ).mark_compact_shape_dynamic(
            mode=2, divisibility = 128 // 8
        )
        if int8 == True:
            cute_tensor.element_type = cutlass.Int8
        else:
            cute_tensor.element_type = cutlass.Float8E4M3FN
        return cute_tensor

    x_tensor = convert_from_dlpack(x.view(b, s, h*d))
    out_tensor = convert_from_dlpack_fp8(out.view(b, s_padding, h*d))
    out_scale_tensor = convert_from_dlpack(out_scale.view(b, h, s_padding))

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, int8, chunk_size)
    if compile_key not in preprocess_QK.compile_cache:
        preprocess_QK_op = PreprocessQK(dtype, int8, chunk_size)
        preprocess_QK.compile_cache[compile_key] = cute.compile(
            preprocess_QK_op,
            x_tensor,
            out_tensor,
            out_scale_tensor,
            current_stream,
        )
    preprocess_QK.compile_cache[compile_key](
        x_tensor, out_tensor, out_scale_tensor, current_stream, 
    )

    # print("...... out shape", out.shape, "stride", out.stride())
    out_final = out.view(b, s_padding, h, d)
    out_scale_final = out_scale.view(b, h, s_padding).permute(0, 2, 1).unsqueeze(-1)
    # print("out_final shape", out_final.shape, "stride", out_final.stride())

    return out_final, out_scale_final


preprocess_QK.compile_cache = {}



def test_preprocess_QK(b, s, h, d, chunk_size: int = 64, int8: bool = False, padding_size: int = 128):
    torch.manual_seed(0)

    x = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)

    # PyTorch native conversion
    x1 = torch.empty(b, s, h, d, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for i in range(40):
        with nvtx.annotate("torch fp16 copy"):
            x1.copy_(x)
            torch.cuda.synchronize()
        
    
    # padding only for sequence length dimension
    def torch_preprocess_qk(x: torch.Tensor, fill_one: bool, int8: bool, chunk_size: int, padding_size: int) -> tuple[torch.Tensor, torch.Tensor]:

        b, s, h, d = x.shape
        assert chunk_size == d, "for attention,chunk_size must be equal to d"

        s_padding = (s + padding_size - 1) // padding_size * padding_size

        range_max = 448.0
        if int8 == True:
            range_max = 127.0

        x_amax = x.abs().float().amax(dim=-1).clamp(1e-4)
        sf = x_amax / range_max
        if fill_one:
            sf.fill_(1.0)

        x_new = x.float() * (1.0 / sf.unsqueeze(-1))

        if int8 == True:
            x_new = torch.round(x_new).clamp(-127., 127.).to(torch.int8)
        else:
            x_new = x_new.to(torch.float8_e4m3fn)
        x_new_dtype = x_new.dtype

        # b s h -> b h s -> b s h 1
        sf = sf.permute(0, 2, 1).contiguous()
        # print("sf shape", sf.shape, " stride", sf.stride())
        sf = torch.cat([sf, torch.zeros(b, h, s_padding - s, device=x.device, dtype=sf.dtype)], dim=2)
        # print("sf shape", sf.shape, " stride", sf.stride())
        sf = sf.permute(0, 2, 1).unsqueeze(-1)

        # add padding for sequence length dimension

        # print("x_new shape", x_new.shape, " stride", x_new.stride())
        x_new = torch.cat([x_new.to(torch.float32), torch.zeros(b, s_padding - s, h, d, device=x.device, dtype=torch.float32)], dim=1).to(x_new_dtype)
        # print("x_new shape", x_new.shape, " stride", x_new.stride())

        return x_new, sf

    out_ref, out_ref_scale = torch_preprocess_qk(x, fill_one=False, int8=int8, chunk_size=chunk_size, padding_size=padding_size)

    torch.cuda.synchronize()
    for i in range(40):
        with nvtx.annotate("cute preprocess V"):
            out, out_scale = preprocess_QK(x, int8=int8, chunk_size=chunk_size)
            torch.cuda.synchronize()
    out, out_scale = preprocess_QK(x, int8=int8, chunk_size=chunk_size)

    print("out shape", out.shape, " stride", out.stride())
    print("out_scale shape", out_scale.shape, " stride", out_scale.stride())

    def compare_out(out_ref, out, flag=True):
        assert out_ref.shape == out.shape, "out_ref and out shapes must match"
        assert out_ref.stride() == out.stride(), "out_ref and out strides must match"

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
    print("---------out scale--------------")
    compare_out(out_ref_scale, out_scale, flag=True)
    print("--------------------------------")

    print("pass")


if __name__ == "__main__":
    b = 2
    s = 16384
    # s = 1370
    h = 24
    d = 128

    test_preprocess_QK(b, s, h, d, chunk_size=d, padding_size=128)
    exit(0)

    for b in [1, 2, 4]:
        for s in [16384, 32768, 1370, 1730]:
            for d in [32, 64, 128]:
                test_preprocess_QK(b, s, h, d, chunk_size=d)
