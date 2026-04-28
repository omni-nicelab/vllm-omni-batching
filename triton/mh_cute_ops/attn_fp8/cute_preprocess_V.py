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
from quack.utils import predicate_k

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


class PreprocessV(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, int8: bool = False):
        super().__init__(dtype, N=128, stage=1) 
        assert dtype.width == 16, "Input should be fp16/bf16"
        self.cluster_n = 1
        self.int8 = const_expr(True) if int8 == True else const_expr(False)

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
        _b, _hd, s_padding = mO.shape

        assert mX.element_type == self.dtype

        tiler_mn, tv_layout_input = self._get_tv_layout2()

        num_threads = cute.size(tv_layout_input, mode=[0])
        # print("num_threads", num_threads)
        num_warps = num_threads // cute.arch.WARP_SIZE

        grid = [
            cute.ceil_div(s_padding, tiler_mn[0]),
            cute.ceil_div(hd, tiler_mn[1]),
            b]
        self.kernel(self.int8, mX, mO, mScale, tv_layout_input, tiler_mn).launch(
            grid=grid,
            block=[num_threads, 1, 1],
            cluster=None,
            # smem=self._smem_size_in_bytes((tiler_mn[0], tiler_mn[1]+2), num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        int8: Constexpr[Boolean],
        mX: cute.Tensor,
        mO: cute.Tensor,
        mScale: cute.Tensor,
        tv_layout_input: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, batch = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sX0 = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout((tiler_mn[0], tiler_mn[1]+2), order=(1, 0)), byte_alignment=2
        )
        sX = cute.local_tile(
            sX0, tiler_mn, (0, 0)
        )

        shape = (mX.shape[1], mX.shape[2])
        idX = cute.make_identity_tensor(shape)
        cX = cute.local_tile(idX, tiler_mn, (bidx, bidy))
        shape_t = (mX.shape[2], mX.shape[1])
        idX_t = cute.make_identity_tensor(shape_t)
        cX_t = cute.local_tile(idX_t, tiler_mn, (bidy, bidx))

        gX = cute.local_tile(mX[(batch, None, None)], tiler_mn, (bidx, bidy))
        gO = cute.local_tile(mO[(batch, None, None)], tiler_mn, (bidy, bidx))
        gScale = cute.local_tile(mScale[(batch, None, None)], (1, 128), (bidx, bidy))

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

        if (bidx+1) * 128 > shape[0]:
            tXrX.fill(cute.Float16(0.0))
            cute.autovec_copy(tXrX, tXsX)
            cute.arch.sync_threads()

        row = tXcX[0][0]
        if row < shape[0]:
            cute.copy(copy_atom_load_X_async, tXgX, tXsX)
        cute.arch.sync_threads()
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.arch.sync_threads()


        tid_logical_x = tidx % 8
        tid_logical_y = tidx // 8
        for j in range(2):
            for i in range(8):
                tlx = tid_logical_x * 8 + i + j * 64
                tXrX[i+j*8] = sX[tlx, tid_logical_y]

        tXrX_fp32_SSA = tXrX.load().to(cutlass.Float32)
        tXrX_fp32_SSA_abs = cute.where(tXrX_fp32_SSA > 0., tXrX_fp32_SSA, -tXrX_fp32_SSA)

        # print("tXrX_fp32_SSA_abs shape", tXrX_fp32_SSA_abs.shape)
        max_SSA = tXrX_fp32_SSA_abs.reduce(cute.ReductionOp.MAX, init_val=1e-4, reduction_profile=((1,1),None,None))
        max_SSA = warp_reduce(max_SSA, cute.arch.fmax, width=8)

        if const_expr(int8):
            one_over_range_max = cute.Float32(1.0 / 127.0)
        else:
            one_over_range_max = cute.Float32(1.0 / 448.0)
        scale_SSA = max_SSA * one_over_range_max

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

        # tXrO.store(tXrO_fp16.load().to(cutlass.Float32).to(cutlass.Float8E4M3FN))

        scale = cute.make_fragment(scale_SSA.shape, scale_SSA.dtype)
        scale.store(scale_SSA)

        # tOpO = predicate_k(thr_copy_O.partition_S(cX_t), limit=shape[0])
        # print("tOpO shape", tOpO.shape)
        cute.copy(copy_atom_store_O, tXrO, tXgO)
        if tidx % 8 == 0:
            gScale[0, tidx // 8] = scale[0]

@torch.compiler.disable
def preprocess_V(
    x: torch.Tensor,
    int8: bool = True,
) -> torch.Tensor:
    assert x.dim() == 4, "Input must be 4D"
    assert x.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16 ], "Unsupported dtype"

    b, s, h, d = x.shape
    # assert d == 128, "Input D dim must be 128"
    # assert s % 128 == 0, "Input S dim must be divisible by 128"
    assert (h * d) % 128 == 0, "Input h * d dim must be divisible by 128"

    chunk_size = 128
    padding_size = chunk_size
    s_padding = (s + padding_size - 1) // padding_size * padding_size

    device = x.device

    out = torch.empty(
        b*s_padding*h*d, device=device, dtype=torch.int8 if int8 == True else torch.float8_e4m3fn
    ) 
    out_scale = torch.empty(
        b*s_padding*h*d//128, device=device, dtype=torch.float32
    )

    dtype = torch2cute_dtype_map[x.dtype]

    def convert_from_dlpack(x):
        return from_dlpack(x.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        ).mark_compact_shape_dynamic(
            mode=1
        ).mark_compact_shape_dynamic(
            mode=2, divisibility = 128 // 16
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

    # out2 = out.view(b, h*d, s)
    # # print("out2 shape", out2.shape, "stride", out2.stride())
    out_tensor = convert_from_dlpack_fp8(out.view(b, h*d, s_padding))
    out_scale_tensor = convert_from_dlpack(out_scale.view(b, s_padding//128, h*d))

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, int8)
    if compile_key not in preprocess_V.compile_cache:
        preprocess_V_op = PreprocessV(dtype, int8)
        preprocess_V.compile_cache[compile_key] = cute.compile(
            preprocess_V_op,
            x_tensor,
            out_tensor,
            out_scale_tensor,
            current_stream,
        )
    preprocess_V.compile_cache[compile_key](
        x_tensor, out_tensor, out_scale_tensor, current_stream, 
    )

    # print("...... out shape", out.shape, "stride", out.stride())
    out_final = out.view(b, h, d, s_padding).permute(0, 3, 1, 2)
    # print("out_final shape", out_final.shape, "stride", out_final.stride())
    # out_scale_final = out_scale.view(b, h, d, s//128).permute(0, 1, 3, 2)
    out_scale_final = out_scale.view(b, s_padding//128, h, d)

    return out_final, out_scale_final


preprocess_V.compile_cache = {}



def test_preprocess_V(b, s, h, d, int8: bool):
    torch.manual_seed(0)

    x = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)

    # PyTorch native conversion
    x1 = torch.empty(b, s, h, d, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for i in range(40):
        with nvtx.annotate("torch fp16 copy"):
            x1.copy_(x)
            torch.cuda.synchronize()

    def torch_preprocess_v(x, int8: bool):

        # v = v0.to(torch.float8_e4m3fn).permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)

        chunk_size = 128
        padding_size = chunk_size
        # assert chunk_size == padding_size, "chunk_size must be equal to padding_size"
        # as seq length is the k dimension in mnk for pv

        b, s, h, d = x.shape
        s_padding = (s + padding_size - 1) // padding_size * padding_size

        range_max = 448.0
        if int8 == True:
            range_max = 127.0

        x2 = x.permute(0, 2, 3, 1).contiguous() # b h d s
        x2 = torch.cat([x2, torch.zeros(b, h, d, s_padding - s, device=x.device, dtype=x2.dtype)], dim=3)

        x2 = x2.view(b, h, d, s_padding//chunk_size, chunk_size)

        x_amax = x2.abs().float().amax(dim=-1).clamp(1e-4)
        sf = x_amax / range_max
        # sf.fill_(1.0)

        x_new = x2.float() * (1.0 / sf.unsqueeze(-1))
        x_new = x_new.view(b, h, d, s_padding)

        if int8 == True:
            x_new = torch.round(x_new).clamp(-range_max, range_max).to(torch.int8)
        else:
            x_new = x_new.to(torch.float8_e4m3fn)

        # b h d s -> b s h d
        x_new = x_new.permute(0, 3, 1, 2)

        # b h d s//128 -> b s//128 h d
        sf = sf.permute(0, 3, 1, 2)
        sf = sf.contiguous()

        return x_new, sf

    out_ref, out_scale_ref = torch_preprocess_v(x, int8=int8)
    print("x shape", x.shape, " stride", x.stride())
    print("out_ref shape", out_ref.shape, " stride", out_ref.stride())
    print("out_scale_ref shape", out_scale_ref.shape, " stride", out_scale_ref.stride())

    torch.cuda.synchronize()
    for i in range(40):
        with nvtx.annotate("cute preprocess V"):
            out = preprocess_V(x, int8=int8)
            torch.cuda.synchronize()
    out, out_scale = preprocess_V(x, int8=int8)

    def compare_out(out_ref, out, flag=False):
        # for i in range(10):
        #     print(f"out_ref[{i}] = {out_ref.flatten()[i]}, out[{i}] = {out.flatten()[i]}")

        out_ref_fp32 = out_ref.to(torch.float32)
        out_fp32 = out.to(torch.float32)
        # find top 10 maximum absolute difference and their values respectively
        max_diff = (out_ref_fp32 - out_fp32).abs().flatten().topk(10)
        # print("top 10 maximum absolute difference: ", max_diff.values)
        # print("top 10 maximum absolute difference indices: ", max_diff.indices)
        for i in range(10):
            print(f"out_ref[{max_diff.indices[i].item()}] = {out_ref_fp32.flatten()[max_diff.indices[i].item()]}, out[{max_diff.indices[i].item()}] = {out_fp32.flatten()[max_diff.indices[i].item()]}")


        print("0st element: ref ", out_ref_fp32.flatten()[0].item(), " vs out ", out_fp32.flatten()[0].item())
        print("max error: ", (out_ref_fp32 - out_fp32).abs().max())
        print("mean error: ", (out_ref_fp32 - out_fp32).abs().mean())

        if flag == True:
            torch.testing.assert_close(out_fp32, out_ref_fp32, atol=1e-3, rtol=1e-3)
    
    print("---------out_scale--------------------")
    compare_out(out_scale_ref, out_scale, flag=True)
    print("---------out--------------------")
    compare_out(out_ref, out, flag=True)
    print("--------------------------------")
    print("pass")


if __name__ == "__main__":
    b = 2
    s = 16384
    s = 1370
    # s = 127
    h = 24
    d = 128

    test_preprocess_V(b, s, h, d, int8=True)
    exit(0)

    for int8 in [True, False]:
        for b in [4, 2]:
            for s in [16384, 16384 * 2, 1370, 1730]:
                test_preprocess_V(b, s, h, d, int8)
        print("--------------------------------")

# # Self-reference for `from .cute_preprocess_V import preprocess_V` style imports
# import sys
# preprocess_V = sys.modules[__name__]
