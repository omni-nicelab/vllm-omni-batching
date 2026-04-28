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

import operator

import nvtx

# PyTorch has its own NVRTC, which may have a lower version than the system
# So try to disable PyTorch's NVRTC, or import NVRTC before PyTorch
import cuda.bindings.nvrtc as nvrtc
# print(f'NVRTC version: {nvrtc.nvrtcVersion()[1:]}')

# import sys
# import os
# # Add parent directory to path to import modules from hopper-bench root
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # Import functions for shift calculation
# from AF8G.fp8_gemm_scale_only import fp8_roundtrip_e4m3nv



class PerTokenCastToFp8(ReductionBase):

    def __init__(
        self, dtype: cutlass.Numeric, N: int, preprocess: int = 0, bias: int = 0, is_chunk32: bool = False
        , compute_shift: bool = False
    ):
        super().__init__(dtype, N, stage=1)
        assert dtype.width == 16, "Input should be fp16/bf16"
        self.preprocess = cutlass.const_expr(0)
        self.bias = cutlass.const_expr(0)
        self.chunk_size = cutlass.const_expr(128)
        self.compute_shift = cutlass.const_expr(0)
        if preprocess == 1:
            self.preprocess = cutlass.const_expr(1)
        if bias == 1:
            self.bias = cutlass.const_expr(1)
        if is_chunk32 == 1:
            self.chunk_size = cutlass.const_expr(32)
        if compute_shift == 1:
            self.compute_shift = cutlass.const_expr(1)

    def _calculate_threads_per_row(self):
        N = self.N
        tmp = (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (32 if N <= 3072 else (64 if N <= 6144 else (128 if N <= 16384 else 256)))
            )
        )
        # tmp = max(tmp, 16) # as we need at least 128 per row
        return tmp

    def _set_cluster_n(self):
        N = self.N
        # cluster_n = 4 is faster and cluster_n = 2 for N=64k for some reason
        # Similarly cluster_n = 8 is faster for N=128k
        if cutlass.const_expr(self.dtype.width == 16):
            cluster_n = (
                1
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        else:  # fp32
            cluster_n = (
                1
                if N <= 32 * 1024
                else (
                    2
                    if N <= 64 * 1024
                    else (4 if N <= 128 * 1024 else (8 if N <= 256 * 1024 else 16))
                )
            )
        self.cluster_n = cluster_n
        self.cluster_n = 1

    def _get_tv_layout2(self):
        copy_bits = 128
        vecsize = copy_bits // self.dtype.width
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"

        num_threads = self._get_num_threads()
        if self.compute_shift == 1:
            num_threads = num_threads * 2
        assert num_threads % cute.arch.WARP_SIZE == 0

        threads_per_row = self._calculate_threads_per_row()
        if self.compute_shift == 1:
            threads_per_row = num_threads // 1

        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)

        cols_per_block = num_threads // threads_per_row

        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)

        tv_layout = cute.make_layout(
            ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_row),
            ),
        )

        # tiler_mn = (128, 512)

        # tv_layout = cute.make_layout(
        #     ((8,128), (8,8)),
        #     stride=(
        #         (128*8,1),
        #         (128+0, (128+0)*8*8)
        #     ),
        # )

        tiler_mn_scale = (tiler_mn[0], min(self.N // 32,tiler_mn[1] // 32))

        if cutlass.const_expr(self.bias == 1):
            num_threads = threads_per_row * cols_per_block
            num_blocks_N = self.N // vecsize // num_threads
            tv_layout_bias = cute.make_layout(
                ((num_threads, 1), (vecsize, num_blocks_N)),
                stride=(
                    (vecsize, 1),
                    (1, num_threads * vecsize),
                ),
            )
        else:
            tv_layout_bias = None

        # print("tv layout ", tv_layout)
        return tiler_mn, tv_layout, tiler_mn_scale, tv_layout_bias

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mB: cute.Tensor,
        mO: cute.Tensor,
        mScale: cute.Tensor,
        mShift: cute.Tensor,
        mSum: cute.Tensor,
        coord: int,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == cutlass.Float8E4M3FN
        assert mScale.element_type == cutlass.Float32
        # assert mShift.element_type == cutlass.Float32
        # assert mSum.element_type == cutlass.Float32

        self._set_cluster_n()
        tiler_mn, tv_layout, tiler_mn_scale, tv_layout_bias = self._get_tv_layout2()
        # print("tiler_mn", tiler_mn)
        # if coord % tiler_mn[1] != 0:
        #     print(" bug ---------------------------------")
        #     print("coord is ", coord, " tiler_n ",tiler_mn[1])
        #     print(" bug ---------------------------------")
        #     print(" bug ---------------------------------")
        #     print(" bug ---------------------------------")
        #     print(" bug ---------------------------------")

        num_threads = cute.size(tv_layout, mode=[0])

        if cutlass.const_expr(self.bias == 1):
            assert mB.shape[1] % (num_threads * 8) == 0  # TODO handle boundary

        num_warps = num_threads // cute.arch.WARP_SIZE

        smem_bias = 0
        if cutlass.const_expr(self.bias == 1):
            smem_bias = cute.size_in_bytes(self.dtype, cute.make_layout(mX.shape[1]))

        grid = [cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1]
        self.kernel(mX, mB, mO, mScale, mShift, mSum, coord, tv_layout, tiler_mn, tiler_mn_scale, tv_layout_bias, self.preprocess, self.bias, self.chunk_size, self.compute_shift).launch(
            grid=grid,
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if cutlass.const_expr(self.cluster_n > 1) else None,
            # smem=self._smem_size_in_bytes(tiler_mn, num_warps) + smem_bias,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mB: cute.Tensor, # bias
        mO: cute.Tensor,
        mScale: cute.Tensor,
        mShift: cute.Tensor,
        mSum: cute.Tensor,
        coord: cute.Int32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        tiler_mn_scale: cute.Shape,
        tv_layout_bias: cute.Layout,
        preprocess: cutlass.Constexpr[int] = 0,
        bias: cutlass.Constexpr[int] = 0,
        chunk_size: cutlass.Constexpr[int] = 128,
        compute_shift: cutlass.Constexpr[int] = 0
    ):

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )

        if cutlass.const_expr(bias == 1):
            sB = smem.allocate_tensor(
                mB.element_type, cute.make_layout((1,tiler_mn[1])), byte_alignment=16
            )
            gB = cute.local_tile(mB, (1,tiler_mn[1]), (0,0) )

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        # mO = [utils.domain_offset_i64((bidx * tiler_mn[0], coord), mT) for mT in (mO)]
        gX = cute.local_tile(mX, tiler_mn, (bidx, bidy))
        # if bidx == 0 and tidx == 0:
        #     cute.printf("in kernel, coord %d\n", coord)
        # gO = cute.local_tile(mO, tiler_mn, (bidx, bidy + coord // tiler_mn[1]))


        mO_new = cute.make_tensor((mO.iterator + coord).align(16), cute.make_layout((mO.shape[0], mO.shape[1]-coord), stride=(mO.shape[1], 1)))
        gO = cute.local_tile(mO_new, tiler_mn, (bidx, bidy))


        # gO = cute.local_tile(mO, tiler_mn, (bidx, bidy ))
        gScale = cute.local_tile(mScale, tiler_mn_scale, (bidx, bidy + coord//32 // tiler_mn_scale[1]))

        if cutlass.const_expr(compute_shift == 1):
            gShift = cute.local_tile(mShift, tiler_mn_scale, (bidx, bidy))
            gSum = cute.local_tile(mSum, tiler_mn_scale, (bidx, bidy))


        cX = cute.local_tile(idX, tiler_mn, (bidx, bidy))

        # declare the atoms which will be used later for memory copy
        copy_atom_load_B = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_load_X_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mO.element_type, num_bits_per_copy=64
        )

        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X_async, tv_layout, tiler_mn).get_slice(
            tidx
        )
        thr_copy_O = cute.make_tiled_copy(copy_atom_store_O, tv_layout, tiler_mn).get_slice(tidx)

        if cutlass.const_expr(bias == 1):
            thr_copy_B = cute.make_tiled_copy(copy_atom_load_B, tv_layout_bias, (1,tiler_mn[1])).get_slice(
                tidx
            )
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]
        tXrX_fp32 = cute.make_fragment_like(tXrX, cute.Float32)

        tXpX = utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
        row = tXcX[0][0]
        if row < shape[0]:
            cute.copy(copy_atom_load_X_async, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        if cutlass.const_expr(bias == 1):
            cute.copy(copy_atom_load_B, tBgB, tBsB)
            cute.arch.sync_threads()

        cute.autovec_copy(tXsX, tXrX)

        threads_per_row = tv_layout.shape[0][0]
        if cutlass.const_expr(bias == 1):
            for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                j = i % 8
                k = i // 8
                tXrX[i] += sB[k * 8 * threads_per_row + (tidx % threads_per_row) * 8 + j]

        if cutlass.const_expr(preprocess == 1):
            tXrX_fp32_SSA0 = tXrX.load()
            tmp = cute.tanh(cute.Float16(0.7978845608028654) * tXrX_fp32_SSA0 * (cute.Float16(1.0) + cute.Float16(0.044715) * tXrX_fp32_SSA0 * tXrX_fp32_SSA0))
            tXrX_fp32_SSA = cute.Float16(0.5)  * tXrX_fp32_SSA0 * (cute.Float16(1.0) + tmp)
            tXrX_fp32_SSA = tXrX_fp32_SSA.to(cute.Float32)
        else:
            tXrX_fp32_SSA = tXrX.load().to(cute.Float32)

        num_threads_per_chunk = chunk_size >> 3
        if cutlass.const_expr(compute_shift == 1):
            sum_SSA = tXrX_fp32_SSA.reduce(cute.ReductionOp.ADD, init_val=0., reduction_profile=((1,None),None,None))
            sum_SSA = warp_reduce(sum_SSA, operator.add, width=num_threads_per_chunk)
            sum_SSA = sum_SSA.to(cute.Float16)
            sum = cute.make_fragment(sum_SSA.shape, sum_SSA.dtype)
            sum.store(sum_SSA)

        # tXrX_fp32.store(tXrX_fp32_SSA)
        # for i in cutlass.range(cute.size(tXrX), unroll_full=True):
        #     tXrX_fp32[i] = max(tXrX_fp32[i], -tXrX_fp32[i])
        # tXrX_fp32_SSA_abs = tXrX_fp32.load()
        tXrX_fp32_SSA_abs = cute.where(tXrX_fp32_SSA > 0., tXrX_fp32_SSA, -tXrX_fp32_SSA)

        val = tXrX_fp32_SSA_abs.reduce(cute.ReductionOp.MAX, init_val=0., reduction_profile=((1,None),None,None))

        val = warp_reduce(val, cute.arch.fmax, width=num_threads_per_chunk)

        scale = cute.make_fragment(val.shape, val.dtype)
        scale.store(val)

        fp32_1_over_448 = cute.Float32(1.0) / 448.0
        for i in cutlass.range_constexpr(cute.size(scale)):
            # scale[i] = max(scale[i], 1e-4) / 448.0
            scale[i] = max(scale[i], 1e-4) * fp32_1_over_448
            # scale[i] = scale[i] * fp32_1_over_448
        val = scale.load()

        # print("val shape", val.shape, val.shape[0])
        b_val = cute.TensorSSA(val.ir_value(), ((1,val.shape[0]),1,1), val.dtype).broadcast_to(tXrX_fp32_SSA.shape)
        tXrX_fp32_SSA_tmp = tXrX_fp32_SSA * (cute.Float32(1.0) / b_val)
        tXrX_fp32.store(tXrX_fp32_SSA_tmp)


        if cutlass.const_expr(compute_shift == 0):
            tXrO.store(tXrX_fp32_SSA_tmp.to(tXrO.element_type))
            tOpO = utils.predicate_k(thr_copy_O.partition_S(cX), limit=shape[1])
            if row < shape[0]:
                cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)

        if row < shape[0]:
            # threads_per_row_valid = min(threads_per_row, shape[1] // 8) # TODO 8 for fp16 only!!!
            # if tidx % threads_per_row < threads_per_row_valid:
            if tidx % num_threads_per_chunk == 0:
                for i in cutlass.range_constexpr(cute.size(scale)):
                    index_in_row = i * (threads_per_row // num_threads_per_chunk) + (tidx % threads_per_row) // num_threads_per_chunk
                    if index_in_row < tiler_mn_scale[1]:
                        gScale[tidx // threads_per_row, index_in_row] = scale[i]
                        if cutlass.const_expr(compute_shift == 1):
                            gSum[tidx // threads_per_row, index_in_row] = sum[i]
        
        if cutlass.const_expr(compute_shift == 1):
            shifted_x_SSA = tXrX_fp32_SSA_tmp
            q_SSA_fp8 = shifted_x_SSA.to(cute.Float8E4M3FN)
            q_SSA = q_SSA_fp8.to(cute.Float32)
            q_SSA = tXrX_fp32_SSA_tmp - q_SSA
            init_loss_item = q_SSA * q_SSA
            init_loss = init_loss_item.reduce(cute.ReductionOp.ADD, init_val=0., reduction_profile=((1,None),None,None))
            init_loss = warp_reduce(init_loss, operator.add, width=num_threads_per_chunk)
            # init_loss = cute.TensorSSA(init_loss.ir_value(), ((1,val.shape[0]),1,1), init_loss.dtype).broadcast_to(tXrX_fp32_SSA.shape)

            q_SSA = q_SSA / 32.
            q_SSA = q_SSA.reduce(cute.ReductionOp.ADD, init_val=0., reduction_profile=((1,None),None,None))
            shift_SSA = warp_reduce(q_SSA, operator.add, width=num_threads_per_chunk)

            b_shift_SSA = cute.TensorSSA(shift_SSA.ir_value(), ((1,val.shape[0]),1,1), shift_SSA.dtype).broadcast_to(tXrX_fp32_SSA.shape)

            shifted_x_SSA  = tXrX_fp32_SSA_tmp - b_shift_SSA
            for iter in cutlass.range_constexpr(4):
                q_SSA_fp8 = shifted_x_SSA.to(cute.Float8E4M3FN)
                q_SSA = q_SSA_fp8.to(cute.Float32)
                q_SSA = tXrX_fp32_SSA_tmp - q_SSA
                q_SSA = q_SSA / 32.
                q_SSA = q_SSA.reduce(cute.ReductionOp.ADD, init_val=0., reduction_profile=((1,None),None,None))
                shift_SSA = warp_reduce(q_SSA, operator.add, width=num_threads_per_chunk)

                b_shift_SSA = cute.TensorSSA(shift_SSA.ir_value(), ((1,val.shape[0]),1,1), shift_SSA.dtype).broadcast_to(tXrX_fp32_SSA.shape)

                shifted_x_SSA  = tXrX_fp32_SSA_tmp - b_shift_SSA
            
            shifted_x_SSA_projected = shifted_x_SSA.to(cute.Float8E4M3FN).to(cute.Float32) - shifted_x_SSA
            final_loss_item = shifted_x_SSA_projected * shifted_x_SSA_projected
            final_loss = final_loss_item.reduce(cute.ReductionOp.ADD, init_val=0., reduction_profile=((1,None),None,None))
            final_loss = warp_reduce(final_loss, operator.add, width=num_threads_per_chunk)
            # final_loss = cute.TensorSSA(final_loss.ir_value(), ((1,val.shape[0]),1,1), final_loss.dtype).broadcast_to(tXrX_fp32_SSA.shape)

            shift_SSA_zero = cute.zeros_like(shift_SSA)
            shift_SSA = cute.where(final_loss > init_loss, shift_SSA_zero, shift_SSA)

            shift_SSA = shift_SSA * val
            shift_SSA = shift_SSA.to(cute.Float16)
            shift = cute.make_fragment(shift_SSA.shape, shift_SSA.dtype)
            shift.store(shift_SSA)


            if row < shape[0]:
                if tidx % num_threads_per_chunk == 0:
                    for i in cutlass.range_constexpr(cute.size(shift)):
                        index_in_row = i * (threads_per_row // num_threads_per_chunk) + (tidx % threads_per_row) // num_threads_per_chunk
                        if index_in_row < tiler_mn_scale[1]:
                            gShift[tidx // threads_per_row, index_in_row] = shift[i]

            tXrO.store(shifted_x_SSA.to(tXrO.element_type))
            tOpO = utils.predicate_k(thr_copy_O.partition_S(cX), limit=shape[1])
            if row < shape[0]:
                cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)

def per_token_cast_to_fp8_cute_shift(
    x: torch.Tensor,
    preprocess: int = 0,
    bias: torch.Tensor = None,
    is_chunk32: bool = False,
    compute_shift: bool = False,
    fp8_cat: torch.Tensor = None,
    scale_cat: torch.Tensor = None,
    is_shift_cat: bool = True, #
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.float32], "Unsupported dtype"

    M, N = x.shape

    coord = 0 # for each row
    if compute_shift == True:
        assert fp8_cat is None and scale_cat is None, "fp8_cat and scale_cat must be None when compute_shift is True"
    else:
        assert fp8_cat is not None and scale_cat is not None, "fp8_cat and scale_cat must be not None when compute_shift is False"
        assert fp8_cat.dtype == torch.float8_e4m3fn and scale_cat.dtype == torch.float32, "fp8_cat and scale_cat must be float8_e4m3fn and float32"
        if is_shift_cat == True:
            coord = N * 32
        else:
            coord = N * 33
    
    # print("coord", coord)
    assert coord % 32 == 0, "coord must be divisible by 32"

    if cutlass.const_expr(is_chunk32 == 0):
        assert N % 128 == 0, "Number of columns must be divisible by 128"
    else:
        assert N % 32 == 0, "Number of columns must be divisible by 32"

    device = x.device

    # output is fp8, shape is (M, N)

    assert is_chunk32 == True, "is_chunk32 must be True"
    if compute_shift == True:
        out = torch.empty(
            M, N + N // 16, device=device, dtype=torch.float8_e4m3fn
        )
        scale = torch.empty(M, out.shape[1] // 32, device=device, dtype=torch.float32)
        shift = torch.empty(M, N // 32, device=device, dtype=torch.float16)
        sum = torch.empty(M, N // 32, device=device, dtype=torch.float16)
    else:
        out = fp8_cat
        scale = scale_cat
        shift = None
        sum = None
        # return out, scale, shift, sum

    dtype = torch2cute_dtype_map[x.dtype]

    convert_from_dlpack = lambda x: (
        from_dlpack(x.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    convert_from_dlpack_1d = lambda x: (
        from_dlpack(x.detach(), assumed_align=16)
    )
    def convert_from_dlpack_fp8(x):
        torch_tensor_view = x.view(torch.uint8)
        cute_tensor = from_dlpack(torch_tensor_view, assumed_align=16)
        cute_tensor.element_type = cutlass.Float8E4M3FN
        return cute_tensor

    x_tensor = convert_from_dlpack(x)
    bias_tensor = convert_from_dlpack_1d(bias) if bias is not None else None
    scale_tensor = convert_from_dlpack(scale)
    shift_tensor = convert_from_dlpack(shift) if shift is not None else None
    sum_tensor = convert_from_dlpack(sum) if sum is not None else None
    out_tensor = convert_from_dlpack_fp8(out)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    bias_flag = 0 if bias is None else 1
    compile_key = (dtype, N, preprocess, bias_flag, compute_shift)
    if compile_key not in per_token_cast_to_fp8_cute_shift.compile_cache:
        per_token_cast_to_fp8_cute_op = PerTokenCastToFp8(dtype, N, preprocess, bias_flag, is_chunk32, compute_shift)
        per_token_cast_to_fp8_cute_shift.compile_cache[compile_key] = cute.compile(
            per_token_cast_to_fp8_cute_op,
            x_tensor,
            bias_tensor,
            out_tensor,
            scale_tensor,
            shift_tensor,
            sum_tensor,
            coord,
            current_stream,
        )
    per_token_cast_to_fp8_cute_shift.compile_cache[compile_key](
        x_tensor, bias_tensor, out_tensor, scale_tensor, shift_tensor, sum_tensor, coord, current_stream
    )
    return out, scale, shift, sum


per_token_cast_to_fp8_cute_shift.compile_cache = {}

def ref_helper(x: torch.Tensor, chunk: int=128, clear_shift: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    assert n % chunk == 0
    padded_n = n
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, chunk)
    
    # Original scale calculation (keep unchanged)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    
    # Calculate scaled_X for shift and sum computation
    scaled_X = x_view * (1.0 / sf.unsqueeze(2))  # [m, num_chunks, chunk]
    
    # Optimize shift for each chunk independently
    num_chunks = scaled_X.shape[1]
    scaled_X_flat = scaled_X.reshape(m * num_chunks, chunk)
    shift_flat, _ = optimize_fp8_shift_only_rowwise(scaled_X_flat, max_iter=5)  # [m*num_chunks]
    shift_448 = shift_flat.view(m, num_chunks)  # [m, num_chunks] in 448 space
    
    # Calculate sum before quantization (sum of scaled_X, not shifted)
    sum_448 = scaled_X.sum(dim=2)  # [m, num_chunks] in 448 space
    if clear_shift == True:
        shift_448 = torch.zeros_like(shift_448)
    # Apply shift before quantization: fp8_x = scaled_X - shift
    fp8_X = (scaled_X - shift_448.unsqueeze(2)).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
    
    # Convert shift and sum back to normal space by multiplying with scale
    shift = shift_448 * sf  # [m, num_chunks] in normal space
    sum_before_quant = sum_448 * sf  # [m, num_chunks] in normal space
    
    return fp8_X , sf, shift, sum_before_quant

def per_token_cast_to_fp8_ref(x: torch.Tensor, chunk: int=128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation with shift and sum calculation
    
    Args:
        x: Input tensor [m, n]
        chunk: Chunk size (default 128)
        
    Returns:
        fp8_x: Quantized tensor [m, n] in fp8 format
        scale: Scale per chunk [m, num_chunks]
        shift: Optimized shift per chunk [m, num_chunks]
        sum_before_quant: Sum of scaled_X per chunk [m, num_chunks]
    """

    
    fp8_x, scale, shift, sum = ref_helper(x, chunk)
    shift_fp8, shift_scale, _, _ = ref_helper(shift.to(torch.float16), chunk, clear_shift=True)
    sum_fp8, sum_scale, _, _ = ref_helper(sum.to(torch.float16), chunk, clear_shift=True)

    final_fp8 = torch.cat([fp8_x, shift_fp8, sum_fp8], dim=1)
    final_scale = torch.cat([scale, shift_scale, sum_scale], dim=1)
    return final_fp8, final_scale, shift, sum

def fp8_roundtrip_e4m3nv(y: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3fn 量化-反量化（输入已在"448空间"）"""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("需要 PyTorch>=2.1 且支持 torch.float8_e4m3fn。")
    return y.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float16)

@torch.no_grad()
def optimize_fp8_shift_only_rowwise(X_scaled0: torch.Tensor, max_iter=5, tol=1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在固定scale下，只优化shift（固定点法，向量化）
    
    Args:
        X_scaled: [R, C]，每行已被各自的 scale 除过，位于"448空间"
        max_iter: 最大迭代次数
        tol: 收敛容差
        
    Returns:
        shift: [R] 行偏移（在"448空间"）
        loss_row: [R] 每行最终的 MSE（对列平均）
    
    使用固定点迭代：shift = mean(x - Q(x - shift))
    """
    X_scaled = X_scaled0.to(torch.float32)

    R, C = X_scaled.shape
    shift = torch.zeros(R, device=X_scaled.device, dtype=torch.float32)  # zero 初始化
    
    # 计算初始 loss
    q_init = fp8_roundtrip_e4m3nv(X_scaled - shift[:, None])
    initial_loss = ((q_init - (X_scaled - shift[:, None]))**2).mean(dim=1)  # [R]

    # 固定跑 max_iter 次迭代
    for _ in range(max_iter):
        q = fp8_roundtrip_e4m3nv(X_scaled - shift[:, None])   # Q(x - shift) [R, C]
        shift = (X_scaled - q).mean(dim=1)                     # 固定点更新 [R]

    # 计算最终 loss
    final_loss = ((fp8_roundtrip_e4m3nv(X_scaled - shift[:, None]) - (X_scaled - shift[:, None]))**2).mean(dim=1)
    
    # 如果最终 loss 大于初始 loss，把 shift 设为 0
    mask = final_loss > initial_loss  # [R]
    # shift[mask] = 0.0
    final_loss[mask] = initial_loss[mask]
    
    return shift, final_loss

def test_fp8_preprocess(M, N):
    torch.manual_seed(0)

    # M, N = 32768, 3072
    # assert N % 128 == 0

    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(1, N, device="cuda", dtype=torch.float16) 

    # PyTorch native conversion
    x0 = torch.empty(M, N, device="cuda", dtype=torch.float8_e4m3fn)
    x1 = torch.empty(M, N, device="cuda", dtype=torch.float16)

    torch.cuda.synchronize()
    for i in range(40):
        with nvtx.annotate("torch fp16 to fp8 conversion"):
            x0 = x.to(torch.float8_e4m3fn)
            torch.cuda.synchronize()
    for i in range(40):
        with nvtx.annotate("torch f16 copy"):
            x1.copy_(x)
            torch.cuda.synchronize()
        
    # for j in range(M):
    #     for i in range(1024):
    #         x[j,i] = i + j * 1024

    x_ref = x.detach().clone().requires_grad_()

    # out_ref = per_token_cast_to_fp8_ref(x_ref)
    bias = torch.randn(1, N, device="cuda", dtype=torch.float16) 
    # out_ref, scale_ref = per_token_cast_to_fp8_ref2(x_ref, bias)
    out_ref, scale_ref, shift_ref, sum_ref = per_token_cast_to_fp8_ref(x_ref+bias, chunk=32)

    for i in range(40):
        with nvtx.annotate("cute per token cast to fp8"):
            out, scale, shift, sum = per_token_cast_to_fp8_cute_shift(x, 0, bias, is_chunk32=True, compute_shift=True, fp8_cat=None, scale_cat=None, is_shift_cat=False)
            out, scale, _, _ = per_token_cast_to_fp8_cute_shift(shift.to(torch.float16), preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=out, scale_cat=scale, is_shift_cat=True)
            out, scale, _, _ = per_token_cast_to_fp8_cute_shift(sum.to(torch.float16), preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=out, scale_cat=scale, is_shift_cat=False)
            torch.cuda.synchronize()

    # exit(0)
    assert out_ref.shape == out.shape
    print("x shape", x.shape)
    print("out shape", out.shape)
    print("scale shape", scale.shape)
    assert scale_ref.shape == scale.shape
    assert out_ref.dtype == out.dtype and scale_ref.dtype == scale.dtype

    def compare_out(out_ref, out, flag=False):
        for i in range(10):
            print(f"out_ref[{i}] = {out_ref.flatten()[i]}, out[{i}] = {out.flatten()[i]}")


        out_ref_fp32 = out_ref.to(torch.float32)
        out_fp32 = out.to(torch.float32)

        # print("11 16 fp32", out_ref_fp32[11,16].item(), " vs ", out_fp32[11,16].item())
        # print("11 16 fp8", out_ref[11,16].item(), " vs ", out[11,16].item())

        print("0st element: ref ", out_ref_fp32.flatten()[0].item(), " vs out ", out_fp32.flatten()[0].item())
        print("max error: ", (out_ref_fp32 - out_fp32).abs().max())
        print("mean error: ", (out_ref_fp32 - out_fp32).abs().mean())

        if flag == True:
            torch.testing.assert_close(out_fp32, out_ref_fp32, atol=1e-3, rtol=1e-3)
        print("pass")
    
    assert out_ref.shape[1] == N + N // 32 + N // 32
    out_ref0 = out_ref[:, :N]
    out_ref1 = out_ref[:, N:N+N//32]
    out_ref2 = out_ref[:, N+N//32:]
    out0 = out[:, :N]
    out1 = out[:, N:N+N//32]
    out2 = out[:, N+N//32:]

    scale_ref0 = scale_ref[:, :N//32]
    scale_ref1 = scale_ref[:, N//32:N//32+N//32//32]
    scale_ref2 = scale_ref[:, N//32+N//32//32:]
    scale0 = scale[:, :N//32]
    scale1 = scale[:, N//32:N//32+N//32//32]
    scale2 = scale[:, N//32+N//32//32:]
    
    print("---------shift--------------------")
    compare_out(shift_ref, shift, flag=False)
    
    print("---------sum--------------------")
    compare_out(sum_ref, sum, flag=False)
    print("--------------------------------")

    print("---------out0--------------------")
    compare_out(out_ref0, out0, flag=False)
    print("---------out1--------------------")
    compare_out(out_ref1, out1, flag=False)
    print("---------out2--------------------")
    compare_out(out_ref2, out2, flag=False)
    print("--------------------------------")

    print("---------scale0--------------------")
    compare_out(scale_ref0, scale0, flag=False)
    print("---------scale1--------------------")
    compare_out(scale_ref1, scale1, flag=False)
    print("---------scale2--------------------")
    compare_out(scale_ref2, scale2, flag=False)
    print("--------------------------------")

if __name__ == "__main__":

    # test_fp8_preprocess(32768, 3072)
    test_fp8_preprocess(32768, 3072 * 4)