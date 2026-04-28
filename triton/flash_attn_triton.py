"""
*Experimental* implementation of FlashAttention in Triton.
Tested with triton==2.0.0.dev20221202.
Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
other than 64:
https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
We'll update this implementation with the new Triton backend once this is fixed.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""

import math

import torch
import triton
import triton.language as tl


# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
# )
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        # BUG: have to store and immediately load
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, k, trans_b=True)
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        if BIAS_TYPE != "none":
            tl.debug_barrier()  # Race condition otherwise
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk * softmax_scale + bias
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == "none":
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])
        # compute dv
        # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
        # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
        # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
        # the output is correct.
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        # if EVEN_M:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs)
        #     else:
        #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # else:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        #     else:
        #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
        #                                    & (offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(p.to(do.dtype), do, trans_a=True)
        # compute dp = dot(v, do)
        # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
        # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
        # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        dp = tl.dot(do, v, trans_b=True)
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(ds, q, trans_a=True)
        # compute dq
        if not (
            EVEN_M & EVEN_HEADDIM
        ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:  # If we're parallelizing across the seqlen_k dimension
            dq = tl.dot(ds, k)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
        # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Bias,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


def _flash_attn_backward(
    do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    # dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        # SEQUENCE_PARALLEL=False,
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq.copy_(dq_accum)


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, bias=None, causal=False, softmax_scale=None):
        """
        qkv: (batch, seqlen, 3, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen, seqlen).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen, seqlen)
        """
        # Make sure that the last dimension is contiguous
        if qkv.stride(-1) != 1:
            qkv = qkv.contiguous()
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            bias=bias,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        ctx.save_for_backward(qkv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        qkv, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[1], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dqkv = torch.empty_like(qkv)
            _flash_attn_backward(
                do,
                qkv[:, :, 0],
                qkv[:, :, 1],
                qkv[:, :, 2],
                o,
                lse,
                dqkv[:, :, 0],
                dqkv[:, :, 1],
                dqkv[:, :, 2],
                bias=bias,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dqkv, None, None, None


flash_attn_qkvpacked_func = FlashAttnQKVPackedFunc.apply


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, bias=None, causal=False, softmax_scale=None):
        """
        q: (batch, seqlen_q, nheads, headdim)
        kv: (batch, seqlen_k, 2, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, kv = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, kv]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, kv[:, :, 0], kv[:, :, 1], bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, kv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, o, lse, bias = ctx.saved_tensors
        if len(ctx.needs_input_grad) >= 3:
            assert not ctx.needs_input_grad[2], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dkv = torch.empty_like(kv)
            _flash_attn_backward(
                do,
                q,
                kv[:, :, 0],
                kv[:, :, 1],
                o,
                lse,
                dq,
                dkv[:, :, 0],
                dkv[:, :, 1],
                bias=bias,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dq, dkv, None, None, None


flash_attn_kvpacked_func = FlashAttnKVPackedFunc.apply


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, k, v, bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[3], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            _flash_attn_backward(
                do,
                q,
                k,
                v,
                o,
                lse,
                dq,
                dk,
                dv,
                bias=bias,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dq, dk, dv, None, None, None


flash_attn_func = FlashAttnFunc.apply


_BITDECODE96_TRITON_WORKSPACES = {}
_BITDECODE96_TRITON_GRAPHS = {}


@triton.jit
def _fp4_e2m1_to_float(x):
    x = x.to(tl.int32)
    abs_x = x & 0x7
    sign = x & 0x8
    exp = abs_x >> 1
    mant = abs_x & 0x1
    pow2 = tl.where(exp == 1, 0.5, tl.where(exp == 2, 1.0, 2.0))
    normal = (2.0 + mant.to(tl.float32)) * pow2
    y = tl.where(abs_x == 0, 0.0, tl.where(abs_x == 1, 0.5, normal))
    return tl.where(sign == 0, y, -y)


@triton.jit
def _load_dequant_nvfp4_block(
    FP4Data,
    BlockScales,
    GlobalScale,
    off_b,
    off_h,
    start_n,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    stride_db,
    stride_dh,
    stride_dn,
    stride_dd,
    stride_sb,
    stride_sh,
    stride_sn,
    stride_sd,
    BLOCK_HEADDIM: tl.constexpr,
):
    fp4_byte_offsets = offs_d // 2
    scale_offsets = offs_d // 16
    data_ptrs = (
        FP4Data
        + off_b * stride_db
        + off_h * stride_dh
        + (start_n + offs_n)[:, None] * stride_dn
        + fp4_byte_offsets[None, :] * stride_dd
    )
    scale_ptrs = (
        BlockScales
        + off_b * stride_sb
        + off_h * stride_sh
        + (start_n + offs_n)[:, None] * stride_sn
        + scale_offsets[None, :] * stride_sd
    )
    mask = ((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim)
    packed = tl.load(data_ptrs, mask=mask, other=0).to(tl.int32)
    shift = (offs_d & 1) * 4
    fp4 = (packed >> shift[None, :]) & 0xF
    block_scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
    global_scale = tl.load(GlobalScale).to(tl.float32)
    x = _fp4_e2m1_to_float(fp4) * block_scale * global_scale
    return tl.where(mask, x, 0.0)


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_fp4_kvcache_kernel(
    Q,
    KFP4,
    VFP4,
    KScales,
    VScales,
    KGlobalScale,
    VGlobalScale,
    Out,
    Lse,
    TMP,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ksb,
    stride_ksh,
    stride_ksn,
    stride_ksd,
    stride_vsb,
    stride_vsh,
    stride_vsn,
    stride_vsd,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    elif EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = _load_dequant_nvfp4_block(
            KFP4,
            KScales,
            KGlobalScale,
            off_b,
            off_h,
            start_n,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            stride_ksb,
            stride_ksh,
            stride_ksn,
            stride_ksd,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
        )

        qk = tl.dot(q, tl.trans(k.to(q.dtype)))
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]

        v = _load_dequant_nvfp4_block(
            VFP4,
            VScales,
            VGlobalScale,
            off_b,
            off_h,
            start_n,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            stride_vb,
            stride_vh,
            stride_vn,
            stride_vd,
            stride_vsb,
            stride_vsh,
            stride_vsn,
            stride_vsd,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
        )
        v = v.to(q.dtype)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i, mask=offs_m < seqlen_q)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    if EVEN_M & EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o)
    elif EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
    else:
        tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))


@triton.heuristics(
    {
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_fp4_kvcache_split_k_stage1_packed_kernel(
    Q,
    KFP4,
    VFP4,
    KScales,
    VScales,
    KGlobalScale,
    VGlobalScale,
    PartialOut,
    PartialM,
    PartialL,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ksb,
    stride_ksh,
    stride_ksn,
    stride_ksd,
    stride_vsb,
    stride_vsh,
    stride_vsn,
    stride_vsd,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_mb,
    stride_mh,
    stride_lb,
    stride_lh,
    nheads,
    seqlen_k,
    headdim,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DPACK: tl.constexpr,
):
    split_id = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    start_n = split_id * BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    offs_p = tl.arange(0, BLOCK_DPACK)
    offs_even = offs_p * 2
    offs_odd = offs_even + 1

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    q_even = tl.load(q_base + offs_even, mask=offs_even < headdim, other=0.0).to(tl.float32)
    q_odd = tl.load(q_base + offs_odd, mask=offs_odd < headdim, other=0.0).to(tl.float32)

    k_data_ptrs = (
        KFP4
        + off_b * stride_kb
        + off_h * stride_kh
        + (start_n + offs_n)[:, None] * stride_kn
        + offs_p[None, :] * stride_kd
    )
    k_scale_ptrs = (
        KScales
        + off_b * stride_ksb
        + off_h * stride_ksh
        + (start_n + offs_n)[:, None] * stride_ksn
        + (offs_p[None, :] // 8) * stride_ksd
    )
    pair_mask = ((start_n + offs_n)[:, None] < seqlen_k) & (offs_even[None, :] < headdim)
    k_packed = tl.load(k_data_ptrs, mask=pair_mask, other=0).to(tl.int32)
    k_scale = tl.load(k_scale_ptrs, mask=pair_mask, other=0.0).to(tl.float32) * tl.load(KGlobalScale).to(tl.float32)
    k_lo = _fp4_e2m1_to_float(k_packed & 0xF) * k_scale
    k_hi = _fp4_e2m1_to_float((k_packed >> 4) & 0xF) * k_scale

    qk = tl.sum(k_lo * q_even[None, :] + k_hi * q_odd[None, :], axis=1) * softmax_scale
    qk = tl.where((start_n + offs_n) < seqlen_k, qk, -float("inf"))
    m_i = tl.max(qk, axis=0)
    p = tl.exp(qk - m_i)
    l_i = tl.sum(p, axis=0)

    v_data_ptrs = (
        VFP4
        + off_b * stride_vb
        + off_h * stride_vh
        + (start_n + offs_n)[:, None] * stride_vn
        + offs_p[None, :] * stride_vd
    )
    v_scale_ptrs = (
        VScales
        + off_b * stride_vsb
        + off_h * stride_vsh
        + (start_n + offs_n)[:, None] * stride_vsn
        + (offs_p[None, :] // 8) * stride_vsd
    )
    v_packed = tl.load(v_data_ptrs, mask=pair_mask, other=0).to(tl.int32)
    v_scale = tl.load(v_scale_ptrs, mask=pair_mask, other=0.0).to(tl.float32) * tl.load(VGlobalScale).to(tl.float32)
    v_lo = _fp4_e2m1_to_float(v_packed & 0xF) * v_scale
    v_hi = _fp4_e2m1_to_float((v_packed >> 4) & 0xF) * v_scale

    acc_even = tl.sum(p[:, None] * v_lo, axis=0)
    acc_odd = tl.sum(p[:, None] * v_hi, axis=0)
    partial_base = PartialOut + off_b * stride_pob + off_h * stride_poh + split_id * stride_pos
    tl.store(partial_base + offs_even, acc_even, mask=offs_even < headdim)
    tl.store(partial_base + offs_odd, acc_odd, mask=offs_odd < headdim)
    tl.store(PartialM + off_b * stride_mb + off_h * stride_mh + split_id, m_i)
    tl.store(PartialL + off_b * stride_lb + off_h * stride_lh + split_id, l_i)


@triton.heuristics(
    {
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_fp4_kvcache_split_k_stage1_tc_kernel(
    Q,
    KFP4,
    VFP4,
    KScales,
    VScales,
    KGlobalScale,
    VGlobalScale,
    PartialOut,
    PartialLse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ksb,
    stride_ksh,
    stride_ksn,
    stride_ksd,
    stride_vsb,
    stride_vsh,
    stride_vsn,
    stride_vsd,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_lseb,
    stride_lseh,
    nheads,
    seqlen_k,
    headdim,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    split_id = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    start_n = split_id * BLOCK_N
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_vec = tl.load(
        Q + off_b * stride_qb + off_h * stride_qh + offs_d,
        mask=offs_d < headdim,
        other=0.0,
    )
    q = q_vec[None, :] + tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    k = _load_dequant_nvfp4_block(
        KFP4,
        KScales,
        KGlobalScale,
        off_b,
        off_h,
        start_n,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_ksb,
        stride_ksh,
        stride_ksn,
        stride_ksd,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )
    qk = tl.dot(q.to(q_vec.dtype), tl.trans(k.to(q_vec.dtype)))
    qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, -float("inf"))
    qk = qk * softmax_scale

    m_i = tl.max(qk, axis=1)
    p = tl.exp(qk - m_i[:, None])
    l_i = tl.sum(p, axis=1)

    v = _load_dequant_nvfp4_block(
        VFP4,
        VScales,
        VGlobalScale,
        off_b,
        off_h,
        start_n,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_vsb,
        stride_vsh,
        stride_vsn,
        stride_vsd,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )
    acc_o = tl.dot(p.to(q_vec.dtype), v.to(q_vec.dtype)) / l_i[:, None]
    row0 = offs_m == 0
    acc_row0 = tl.sum(tl.where(row0[:, None], acc_o, 0.0), axis=0)
    lse_row0 = tl.sum(tl.where(row0, m_i + tl.log(l_i), 0.0), axis=0)
    partial_ptrs = (
        PartialOut
        + off_b * stride_pob
        + off_h * stride_poh
        + split_id * stride_pos
        + offs_d
    )
    tl.store(partial_ptrs, acc_row0, mask=offs_d < headdim)
    tl.store(PartialLse + off_b * stride_lseb + off_h * stride_lseh + split_id, lse_row0)


@triton.heuristics(
    {
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_fp4_kvcache_split_k_stage1_kernel(
    Q,
    KFP4,
    VFP4,
    KScales,
    VScales,
    KGlobalScale,
    VGlobalScale,
    PartialOut,
    PartialLse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ksb,
    stride_ksh,
    stride_ksn,
    stride_ksd,
    stride_vsb,
    stride_vsh,
    stride_vsn,
    stride_vsd,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_lseb,
    stride_lseh,
    nheads,
    seqlen_k,
    headdim,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    split_id = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    start_n = split_id * BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_d
    q = tl.load(q_ptrs, mask=offs_d < headdim, other=0.0).to(tl.float32)

    k = _load_dequant_nvfp4_block(
        KFP4,
        KScales,
        KGlobalScale,
        off_b,
        off_h,
        start_n,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_ksb,
        stride_ksh,
        stride_ksn,
        stride_ksd,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )
    qk = tl.sum(k * q[None, :], axis=1) * softmax_scale
    qk = tl.where((start_n + offs_n) < seqlen_k, qk, -float("inf"))
    m_i = tl.max(qk, axis=0)
    p = tl.exp(qk - m_i)
    l_i = tl.sum(p, axis=0)

    v = _load_dequant_nvfp4_block(
        VFP4,
        VScales,
        VGlobalScale,
        off_b,
        off_h,
        start_n,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_vsb,
        stride_vsh,
        stride_vsn,
        stride_vsd,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )
    acc_o = tl.sum(p[:, None] * v, axis=0) / l_i
    partial_ptrs = (
        PartialOut
        + off_b * stride_pob
        + off_h * stride_poh
        + split_id * stride_pos
        + offs_d
    )
    tl.store(partial_ptrs, acc_o, mask=offs_d < headdim)
    tl.store(PartialLse + off_b * stride_lseb + off_h * stride_lseh + split_id, m_i + tl.log(l_i))


@triton.jit
def _fwd_fp4_kvcache_split_k_stage2_kernel(
    PartialOut,
    PartialLse,
    Out,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_lseb,
    stride_lseh,
    stride_ob,
    stride_oh,
    nheads,
    num_splits,
    headdim,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    off_hb = tl.program_id(0)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_s = tl.arange(0, BLOCK_SPLITS)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    split_mask = offs_s < num_splits

    lse = tl.load(
        PartialLse + off_b * stride_lseb + off_h * stride_lseh + offs_s,
        mask=split_mask,
        other=-float("inf"),
    )
    m = tl.max(lse, axis=0)
    weights = tl.exp(lse - m)
    denom = tl.sum(weights, axis=0)
    partial_ptrs = (
        PartialOut
        + off_b * stride_pob
        + off_h * stride_poh
        + offs_s[:, None] * stride_pos
        + offs_d[None, :]
    )
    partial = tl.load(
        partial_ptrs,
        mask=split_mask[:, None] & (offs_d[None, :] < headdim),
        other=0.0,
    )
    out = tl.sum(weights[:, None] * partial, axis=0) / denom
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + offs_d
    tl.store(out_ptrs, out, mask=offs_d < headdim)


@triton.jit
def _fwd_fp4_kvcache_split_k_stage2_raw_kernel(
    PartialOut,
    PartialM,
    PartialL,
    Out,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_mb,
    stride_mh,
    stride_lb,
    stride_lh,
    stride_ob,
    stride_oh,
    nheads,
    num_splits,
    headdim,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    off_hb = tl.program_id(0)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_s = tl.arange(0, BLOCK_SPLITS)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    split_mask = offs_s < num_splits

    m_i = tl.load(
        PartialM + off_b * stride_mb + off_h * stride_mh + offs_s,
        mask=split_mask,
        other=-float("inf"),
    )
    l_i = tl.load(
        PartialL + off_b * stride_lb + off_h * stride_lh + offs_s,
        mask=split_mask,
        other=0.0,
    )
    m = tl.max(m_i, axis=0)
    weights = tl.exp(m_i - m)
    denom = tl.sum(weights * l_i, axis=0)
    partial_ptrs = (
        PartialOut
        + off_b * stride_pob
        + off_h * stride_poh
        + offs_s[:, None] * stride_pos
        + offs_d[None, :]
    )
    partial = tl.load(
        partial_ptrs,
        mask=split_mask[:, None] & (offs_d[None, :] < headdim),
        other=0.0,
    )
    out = tl.sum(weights[:, None] * partial, axis=0) / denom
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + offs_d
    tl.store(out_ptrs, out, mask=offs_d < headdim)


@triton.jit
def _fwd_bitdecode_split_k_stage2_raw_exp2_kernel(
    PartialOut,
    PartialM,
    PartialL,
    Out,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_mb,
    stride_mh,
    stride_lb,
    stride_lh,
    stride_ob,
    stride_oh,
    nheads,
    num_splits,
    headdim,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    off_hb = tl.program_id(0)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_s = tl.arange(0, BLOCK_SPLITS)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    split_mask = offs_s < num_splits

    m_i = tl.load(
        PartialM + off_b * stride_mb + off_h * stride_mh + offs_s,
        mask=split_mask,
        other=-float("inf"),
    )
    l_i = tl.load(
        PartialL + off_b * stride_lb + off_h * stride_lh + offs_s,
        mask=split_mask,
        other=0.0,
    )
    m = tl.max(m_i, axis=0)
    weights = tl.exp2(m_i - m)
    denom = tl.sum(weights * l_i, axis=0)
    partial_ptrs = (
        PartialOut
        + off_b * stride_pob
        + off_h * stride_poh
        + offs_s[:, None] * stride_pos
        + offs_d[None, :]
    )
    partial = tl.load(
        partial_ptrs,
        mask=split_mask[:, None] & (offs_d[None, :] < headdim),
        other=0.0,
    )
    out = tl.sum(weights[:, None] * partial, axis=0) / denom
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + offs_d
    tl.store(out_ptrs, out, mask=offs_d < headdim)


@triton.jit
def _fwd_bitdecode96_i4_stage1_group_kernel(
    Q,
    KPack,
    KParamsH,
    VPack,
    VParamsH,
    PartialOut,
    PartialM,
    PartialL,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_kpb,
    stride_kpg,
    stride_kph,
    stride_kpd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_vpb,
    stride_vph,
    stride_vpt,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_mb,
    stride_mh,
    stride_lb,
    stride_lh,
    nheads,
    seqlen_k,
    BLOCK_D: tl.constexpr,
    BLOCK_TPACK: tl.constexpr,
    BLOCK_VPACK: tl.constexpr,
):
    group_id = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_t = tl.arange(0, BLOCK_TPACK)
    offs_d = tl.arange(0, BLOCK_D)
    offs_vp = tl.arange(0, BLOCK_VPACK)
    start_n = group_id * 128

    q = tl.load(
        Q + off_b * stride_qb + off_h * stride_qh + offs_d,
        mask=offs_d < 96,
        other=0.0,
    ).to(tl.float32)
    k_scale = tl.load(
        KParamsH + off_b * stride_kpb + group_id * stride_kpg + off_h * stride_kph + offs_d * stride_kpd,
        mask=offs_d < 96,
        other=0.0,
    ).to(tl.float32)
    k_zero = tl.load(
        KParamsH
        + off_b * stride_kpb
        + group_id * stride_kpg
        + off_h * stride_kph
        + (128 + offs_d) * stride_kpd,
        mask=offs_d < 96,
        other=0.0,
    ).to(tl.float32)
    q_scaled = q * k_scale
    q_zero = tl.sum(q * k_zero, axis=0)

    k_ptrs = (
        KPack
        + off_b * stride_kb
        + off_h * stride_kh
        + (group_id * 32 + offs_t)[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    k_raw = tl.load(k_ptrs, mask=offs_d[None, :] < 96, other=0).to(tl.int32)
    qk0 = (tl.sum(((k_raw >> 0) & 0xF).to(tl.float32) * q_scaled[None, :], axis=1) + q_zero) * softmax_scale
    qk1 = (tl.sum(((k_raw >> 4) & 0xF).to(tl.float32) * q_scaled[None, :], axis=1) + q_zero) * softmax_scale
    qk2 = (tl.sum(((k_raw >> 8) & 0xF).to(tl.float32) * q_scaled[None, :], axis=1) + q_zero) * softmax_scale
    qk3 = (tl.sum(((k_raw >> 12) & 0xF).to(tl.float32) * q_scaled[None, :], axis=1) + q_zero) * softmax_scale

    valid0 = start_n + offs_t < seqlen_k
    valid1 = start_n + 32 + offs_t < seqlen_k
    valid2 = start_n + 64 + offs_t < seqlen_k
    valid3 = start_n + 96 + offs_t < seqlen_k
    qk0 = tl.where(valid0, qk0, -float("inf"))
    qk1 = tl.where(valid1, qk1, -float("inf"))
    qk2 = tl.where(valid2, qk2, -float("inf"))
    qk3 = tl.where(valid3, qk3, -float("inf"))

    m_i = tl.maximum(tl.maximum(tl.max(qk0, axis=0), tl.max(qk1, axis=0)), tl.maximum(tl.max(qk2, axis=0), tl.max(qk3, axis=0)))
    p0 = tl.where(valid0, tl.exp(qk0 - m_i), 0.0)
    p1 = tl.where(valid1, tl.exp(qk1 - m_i), 0.0)
    p2 = tl.where(valid2, tl.exp(qk2 - m_i), 0.0)
    p3 = tl.where(valid3, tl.exp(qk3 - m_i), 0.0)
    l_i = tl.sum(p0, axis=0) + tl.sum(p1, axis=0) + tl.sum(p2, axis=0) + tl.sum(p3, axis=0)

    v_ptrs0 = (
        VPack
        + off_b * stride_vb
        + off_h * stride_vh
        + (start_n + offs_t)[:, None] * stride_vn
        + offs_vp[None, :] * stride_vd
    )
    v_ptrs1 = v_ptrs0 + 32 * stride_vn
    v_ptrs2 = v_ptrs0 + 64 * stride_vn
    v_ptrs3 = v_ptrs0 + 96 * stride_vn
    mask_vp = offs_vp[None, :] < 32
    v_raw0 = tl.load(v_ptrs0, mask=valid0[:, None] & mask_vp, other=0).to(tl.int32)
    v_raw1 = tl.load(v_ptrs1, mask=valid1[:, None] & mask_vp, other=0).to(tl.int32)
    v_raw2 = tl.load(v_ptrs2, mask=valid2[:, None] & mask_vp, other=0).to(tl.int32)
    v_raw3 = tl.load(v_ptrs3, mask=valid3[:, None] & mask_vp, other=0).to(tl.int32)

    v_param_base = VParamsH + off_b * stride_vpb + off_h * stride_vph + group_id * 256 * stride_vpt
    v_scale0 = tl.load(v_param_base + offs_t * stride_vpt, mask=valid0, other=0.0).to(tl.float32)
    v_scale1 = tl.load(v_param_base + (32 + offs_t) * stride_vpt, mask=valid1, other=0.0).to(tl.float32)
    v_scale2 = tl.load(v_param_base + (64 + offs_t) * stride_vpt, mask=valid2, other=0.0).to(tl.float32)
    v_scale3 = tl.load(v_param_base + (96 + offs_t) * stride_vpt, mask=valid3, other=0.0).to(tl.float32)
    v_zero0 = tl.load(v_param_base + (128 + offs_t) * stride_vpt, mask=valid0, other=0.0).to(tl.float32)
    v_zero1 = tl.load(v_param_base + (128 + 32 + offs_t) * stride_vpt, mask=valid1, other=0.0).to(tl.float32)
    v_zero2 = tl.load(v_param_base + (128 + 64 + offs_t) * stride_vpt, mask=valid2, other=0.0).to(tl.float32)
    v_zero3 = tl.load(v_param_base + (128 + 96 + offs_t) * stride_vpt, mask=valid3, other=0.0).to(tl.float32)

    ps0 = p0 * v_scale0
    ps1 = p1 * v_scale1
    ps2 = p2 * v_scale2
    ps3 = p3 * v_scale3
    zero_acc = (
        tl.sum(p0 * v_zero0, axis=0)
        + tl.sum(p1 * v_zero1, axis=0)
        + tl.sum(p2 * v_zero2, axis=0)
        + tl.sum(p3 * v_zero3, axis=0)
    )

    acc0 = (
        tl.sum(ps0[:, None] * ((v_raw0 >> 0) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps1[:, None] * ((v_raw1 >> 0) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps2[:, None] * ((v_raw2 >> 0) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps3[:, None] * ((v_raw3 >> 0) & 0xF).to(tl.float32), axis=0)
        + zero_acc
    )
    acc1 = (
        tl.sum(ps0[:, None] * ((v_raw0 >> 4) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps1[:, None] * ((v_raw1 >> 4) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps2[:, None] * ((v_raw2 >> 4) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps3[:, None] * ((v_raw3 >> 4) & 0xF).to(tl.float32), axis=0)
        + zero_acc
    )
    acc2 = (
        tl.sum(ps0[:, None] * ((v_raw0 >> 8) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps1[:, None] * ((v_raw1 >> 8) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps2[:, None] * ((v_raw2 >> 8) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps3[:, None] * ((v_raw3 >> 8) & 0xF).to(tl.float32), axis=0)
        + zero_acc
    )

    partial_base = PartialOut + off_b * stride_pob + off_h * stride_poh + group_id * stride_pos
    tl.store(partial_base + offs_vp, acc0, mask=offs_vp < 32)
    tl.store(partial_base + 32 + offs_vp, acc1, mask=offs_vp < 32)
    tl.store(partial_base + 64 + offs_vp, acc2, mask=offs_vp < 32)
    tl.store(PartialM + off_b * stride_mb + off_h * stride_mh + group_id, m_i)
    tl.store(PartialL + off_b * stride_lb + off_h * stride_lh + group_id, l_i)


@triton.jit
def _fwd_bitdecode96_i4_stage1_group96_kernel(
    Q,
    KPack,
    KParamsH,
    VPack,
    VParamsH,
    PartialOut,
    PartialM,
    PartialL,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_kpb,
    stride_kpg,
    stride_kph,
    stride_kpd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_vpb,
    stride_vph,
    stride_vpt,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_mb,
    stride_mh,
    stride_lb,
    stride_lh,
    nheads,
    seqlen_k,
    BLOCK_D0: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_TPACK: tl.constexpr,
    BLOCK_VPACK: tl.constexpr,
):
    group_id = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_t = tl.arange(0, BLOCK_TPACK)
    offs_d0 = tl.arange(0, BLOCK_D0)
    offs_d1 = tl.arange(0, BLOCK_D1)
    offs_vp = tl.arange(0, BLOCK_VPACK)
    start_n = group_id * 128

    q0 = tl.load(Q + off_b * stride_qb + off_h * stride_qh + offs_d0).to(tl.float32)
    q1 = tl.load(Q + off_b * stride_qb + off_h * stride_qh + 64 + offs_d1).to(tl.float32)
    k_scale0 = tl.load(
        KParamsH + off_b * stride_kpb + group_id * stride_kpg + off_h * stride_kph + offs_d0 * stride_kpd
    ).to(tl.float32)
    k_scale1 = tl.load(
        KParamsH + off_b * stride_kpb + group_id * stride_kpg + off_h * stride_kph + (64 + offs_d1) * stride_kpd
    ).to(tl.float32)
    k_zero0 = tl.load(
        KParamsH + off_b * stride_kpb + group_id * stride_kpg + off_h * stride_kph + (128 + offs_d0) * stride_kpd
    ).to(tl.float32)
    k_zero1 = tl.load(
        KParamsH
        + off_b * stride_kpb
        + group_id * stride_kpg
        + off_h * stride_kph
        + (128 + 64 + offs_d1) * stride_kpd
    ).to(tl.float32)
    q_scaled0 = q0 * k_scale0
    q_scaled1 = q1 * k_scale1
    q_zero = tl.sum(q0 * k_zero0, axis=0) + tl.sum(q1 * k_zero1, axis=0)

    k_ptrs0 = (
        KPack
        + off_b * stride_kb
        + off_h * stride_kh
        + (group_id * 32 + offs_t)[:, None] * stride_kn
        + offs_d0[None, :] * stride_kd
    )
    k_ptrs1 = (
        KPack
        + off_b * stride_kb
        + off_h * stride_kh
        + (group_id * 32 + offs_t)[:, None] * stride_kn
        + (64 + offs_d1)[None, :] * stride_kd
    )
    k_raw0 = tl.load(k_ptrs0).to(tl.int32)
    k_raw1 = tl.load(k_ptrs1).to(tl.int32)

    softmax_scale_log2 = softmax_scale * 1.4426950408889634
    qk0 = (
        tl.sum(((k_raw0 >> 0) & 0xF).to(tl.float32) * q_scaled0[None, :], axis=1)
        + tl.sum(((k_raw1 >> 0) & 0xF).to(tl.float32) * q_scaled1[None, :], axis=1)
        + q_zero
    ) * softmax_scale_log2
    qk1 = (
        tl.sum(((k_raw0 >> 4) & 0xF).to(tl.float32) * q_scaled0[None, :], axis=1)
        + tl.sum(((k_raw1 >> 4) & 0xF).to(tl.float32) * q_scaled1[None, :], axis=1)
        + q_zero
    ) * softmax_scale_log2
    qk2 = (
        tl.sum(((k_raw0 >> 8) & 0xF).to(tl.float32) * q_scaled0[None, :], axis=1)
        + tl.sum(((k_raw1 >> 8) & 0xF).to(tl.float32) * q_scaled1[None, :], axis=1)
        + q_zero
    ) * softmax_scale_log2
    qk3 = (
        tl.sum(((k_raw0 >> 12) & 0xF).to(tl.float32) * q_scaled0[None, :], axis=1)
        + tl.sum(((k_raw1 >> 12) & 0xF).to(tl.float32) * q_scaled1[None, :], axis=1)
        + q_zero
    ) * softmax_scale_log2

    valid0 = start_n + offs_t < seqlen_k
    valid1 = start_n + 32 + offs_t < seqlen_k
    valid2 = start_n + 64 + offs_t < seqlen_k
    valid3 = start_n + 96 + offs_t < seqlen_k
    qk0 = tl.where(valid0, qk0, -float("inf"))
    qk1 = tl.where(valid1, qk1, -float("inf"))
    qk2 = tl.where(valid2, qk2, -float("inf"))
    qk3 = tl.where(valid3, qk3, -float("inf"))

    m_i = tl.maximum(tl.maximum(tl.max(qk0, axis=0), tl.max(qk1, axis=0)), tl.maximum(tl.max(qk2, axis=0), tl.max(qk3, axis=0)))
    p0 = tl.where(valid0, tl.exp2(qk0 - m_i), 0.0)
    p1 = tl.where(valid1, tl.exp2(qk1 - m_i), 0.0)
    p2 = tl.where(valid2, tl.exp2(qk2 - m_i), 0.0)
    p3 = tl.where(valid3, tl.exp2(qk3 - m_i), 0.0)
    l_i = tl.sum(p0, axis=0) + tl.sum(p1, axis=0) + tl.sum(p2, axis=0) + tl.sum(p3, axis=0)

    v_ptrs0 = (
        VPack
        + off_b * stride_vb
        + off_h * stride_vh
        + (start_n + offs_t)[:, None] * stride_vn
        + offs_vp[None, :] * stride_vd
    )
    v_ptrs1 = v_ptrs0 + 32 * stride_vn
    v_ptrs2 = v_ptrs0 + 64 * stride_vn
    v_ptrs3 = v_ptrs0 + 96 * stride_vn
    mask_vp = offs_vp[None, :] < 32
    v_raw0 = tl.load(v_ptrs0, mask=valid0[:, None] & mask_vp, other=0).to(tl.int32)
    v_raw1 = tl.load(v_ptrs1, mask=valid1[:, None] & mask_vp, other=0).to(tl.int32)
    v_raw2 = tl.load(v_ptrs2, mask=valid2[:, None] & mask_vp, other=0).to(tl.int32)
    v_raw3 = tl.load(v_ptrs3, mask=valid3[:, None] & mask_vp, other=0).to(tl.int32)

    v_param_base = VParamsH + off_b * stride_vpb + off_h * stride_vph + group_id * 256 * stride_vpt
    v_scale0 = tl.load(v_param_base + offs_t * stride_vpt, mask=valid0, other=0.0).to(tl.float32)
    v_scale1 = tl.load(v_param_base + (32 + offs_t) * stride_vpt, mask=valid1, other=0.0).to(tl.float32)
    v_scale2 = tl.load(v_param_base + (64 + offs_t) * stride_vpt, mask=valid2, other=0.0).to(tl.float32)
    v_scale3 = tl.load(v_param_base + (96 + offs_t) * stride_vpt, mask=valid3, other=0.0).to(tl.float32)
    v_zero0 = tl.load(v_param_base + (128 + offs_t) * stride_vpt, mask=valid0, other=0.0).to(tl.float32)
    v_zero1 = tl.load(v_param_base + (128 + 32 + offs_t) * stride_vpt, mask=valid1, other=0.0).to(tl.float32)
    v_zero2 = tl.load(v_param_base + (128 + 64 + offs_t) * stride_vpt, mask=valid2, other=0.0).to(tl.float32)
    v_zero3 = tl.load(v_param_base + (128 + 96 + offs_t) * stride_vpt, mask=valid3, other=0.0).to(tl.float32)

    ps0 = p0 * v_scale0
    ps1 = p1 * v_scale1
    ps2 = p2 * v_scale2
    ps3 = p3 * v_scale3
    zero_acc = (
        tl.sum(p0 * v_zero0, axis=0)
        + tl.sum(p1 * v_zero1, axis=0)
        + tl.sum(p2 * v_zero2, axis=0)
        + tl.sum(p3 * v_zero3, axis=0)
    )
    acc0 = (
        tl.sum(ps0[:, None] * ((v_raw0 >> 0) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps1[:, None] * ((v_raw1 >> 0) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps2[:, None] * ((v_raw2 >> 0) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps3[:, None] * ((v_raw3 >> 0) & 0xF).to(tl.float32), axis=0)
        + zero_acc
    )
    acc1 = (
        tl.sum(ps0[:, None] * ((v_raw0 >> 4) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps1[:, None] * ((v_raw1 >> 4) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps2[:, None] * ((v_raw2 >> 4) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps3[:, None] * ((v_raw3 >> 4) & 0xF).to(tl.float32), axis=0)
        + zero_acc
    )
    acc2 = (
        tl.sum(ps0[:, None] * ((v_raw0 >> 8) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps1[:, None] * ((v_raw1 >> 8) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps2[:, None] * ((v_raw2 >> 8) & 0xF).to(tl.float32), axis=0)
        + tl.sum(ps3[:, None] * ((v_raw3 >> 8) & 0xF).to(tl.float32), axis=0)
        + zero_acc
    )

    partial_base = PartialOut + off_b * stride_pob + off_h * stride_poh + group_id * stride_pos
    tl.store(partial_base + offs_vp, acc0, mask=offs_vp < 32)
    tl.store(partial_base + 32 + offs_vp, acc1, mask=offs_vp < 32)
    tl.store(partial_base + 64 + offs_vp, acc2, mask=offs_vp < 32)
    tl.store(PartialM + off_b * stride_mb + off_h * stride_mh + group_id, m_i)
    tl.store(PartialL + off_b * stride_lb + off_h * stride_lh + group_id, l_i)


@triton.jit
def _fwd_bitdecode96_i4_stage1_tc_kernel(
    Q,
    KPack,
    KParamsH,
    VPack,
    VParamsH,
    PartialOut,
    PartialM,
    PartialL,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_kpb,
    stride_kpg,
    stride_kph,
    stride_kpd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_vpb,
    stride_vph,
    stride_vpt,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_mb,
    stride_mh,
    stride_lb,
    stride_lh,
    nheads,
    seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    group_id = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    start_n = group_id * 128

    q_vec = tl.load(
        Q + off_b * stride_qb + off_h * stride_qh + offs_d,
        mask=offs_d < 96,
        other=0.0,
    )
    q = q_vec[None, :] + tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    k_scale = tl.load(
        KParamsH + off_b * stride_kpb + group_id * stride_kpg + off_h * stride_kph + offs_d * stride_kpd,
        mask=offs_d < 96,
        other=0.0,
    ).to(tl.float32)
    k_zero = tl.load(
        KParamsH
        + off_b * stride_kpb
        + group_id * stride_kpg
        + off_h * stride_kph
        + (128 + offs_d) * stride_kpd,
        mask=offs_d < 96,
        other=0.0,
    ).to(tl.float32)
    packed_t = group_id * 32 + (offs_n & 31)
    k_shift = ((offs_n >> 5) * 4).to(tl.int32)
    k_ptrs = (
        KPack
        + off_b * stride_kb
        + off_h * stride_kh
        + packed_t[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    k_raw = tl.load(k_ptrs, mask=offs_d[None, :] < 96, other=0).to(tl.int32)
    k_q = ((k_raw >> k_shift[:, None]) & 0xF).to(tl.float32)
    k = tl.where(offs_d[None, :] < 96, k_q * k_scale[None, :] + k_zero[None, :], 0.0)

    qk = tl.dot(q.to(q_vec.dtype), tl.trans(k.to(q_vec.dtype))) * softmax_scale
    valid = start_n + offs_n < seqlen_k
    qk = tl.where(valid[None, :], qk, -float("inf"))
    m_i = tl.max(qk, axis=1)
    p = tl.exp(qk - m_i[:, None])
    p = tl.where(valid[None, :], p, 0.0)
    l_i = tl.sum(p, axis=1)

    v_pack_col = offs_d & 31
    v_shift = ((offs_d >> 5) * 4).to(tl.int32)
    v_ptrs = (
        VPack
        + off_b * stride_vb
        + off_h * stride_vh
        + (start_n + offs_n)[:, None] * stride_vn
        + v_pack_col[None, :] * stride_vd
    )
    v_raw = tl.load(v_ptrs, mask=valid[:, None] & (offs_d[None, :] < 96), other=0).to(tl.int32)
    v_q = ((v_raw >> v_shift[None, :]) & 0xF).to(tl.float32)
    v_param_base = VParamsH + off_b * stride_vpb + off_h * stride_vph + group_id * 256 * stride_vpt
    v_scale = tl.load(v_param_base + offs_n * stride_vpt, mask=valid, other=0.0).to(tl.float32)
    v_zero = tl.load(v_param_base + (128 + offs_n) * stride_vpt, mask=valid, other=0.0).to(tl.float32)
    v = tl.where(offs_d[None, :] < 96, v_q * v_scale[:, None] + v_zero[:, None], 0.0)
    acc = tl.dot(p.to(q_vec.dtype), v.to(q_vec.dtype))

    row0 = offs_m == 0
    acc_row0 = tl.sum(tl.where(row0[:, None], acc, 0.0), axis=0)
    m_row0 = tl.sum(tl.where(row0, m_i, 0.0), axis=0)
    l_row0 = tl.sum(tl.where(row0, l_i, 0.0), axis=0)
    partial_base = PartialOut + off_b * stride_pob + off_h * stride_poh + group_id * stride_pos
    tl.store(partial_base + offs_d, acc_row0, mask=offs_d < 96)
    tl.store(PartialM + off_b * stride_mb + off_h * stride_mh + group_id, m_row0)
    tl.store(PartialL + off_b * stride_lb + off_h * stride_lh + group_id, l_row0)


@triton.jit
def _fwd_bitdecode96_i4_stage1_groupx_kernel(
    Q,
    KPack,
    KParamsH,
    VPack,
    VParamsH,
    PartialOut,
    PartialM,
    PartialL,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_kpb,
    stride_kpg,
    stride_kph,
    stride_kpd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_vpb,
    stride_vph,
    stride_vpt,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_mb,
    stride_mh,
    stride_lb,
    stride_lh,
    nheads,
    seqlen_k,
    GROUPS_PER_SPLIT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_TPACK: tl.constexpr,
    BLOCK_VPACK: tl.constexpr,
):
    split_id = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_t = tl.arange(0, BLOCK_TPACK)
    offs_d = tl.arange(0, BLOCK_D)
    offs_vp = tl.arange(0, BLOCK_VPACK)

    q = tl.load(
        Q + off_b * stride_qb + off_h * stride_qh + offs_d,
        mask=offs_d < 96,
        other=0.0,
    ).to(tl.float32)

    m_total = tl.full((), -float("inf"), dtype=tl.float32)
    l_total = tl.full((), 0.0, dtype=tl.float32)
    acc0_total = tl.zeros([BLOCK_VPACK], dtype=tl.float32)
    acc1_total = tl.zeros([BLOCK_VPACK], dtype=tl.float32)
    acc2_total = tl.zeros([BLOCK_VPACK], dtype=tl.float32)

    for go in range(0, GROUPS_PER_SPLIT):
        group_id = split_id * GROUPS_PER_SPLIT + go
        start_n = group_id * 128
        group_active = start_n < seqlen_k

        k_scale = tl.load(
            KParamsH + off_b * stride_kpb + group_id * stride_kpg + off_h * stride_kph + offs_d * stride_kpd,
            mask=group_active & (offs_d < 96),
            other=0.0,
        ).to(tl.float32)
        k_zero = tl.load(
            KParamsH
            + off_b * stride_kpb
            + group_id * stride_kpg
            + off_h * stride_kph
            + (128 + offs_d) * stride_kpd,
            mask=group_active & (offs_d < 96),
            other=0.0,
        ).to(tl.float32)
        q_scaled = q * k_scale
        q_zero = tl.sum(q * k_zero, axis=0)

        k_ptrs = (
            KPack
            + off_b * stride_kb
            + off_h * stride_kh
            + (group_id * 32 + offs_t)[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        k_raw = tl.load(k_ptrs, mask=group_active & (offs_d[None, :] < 96), other=0).to(tl.int32)
        qk0 = (tl.sum(((k_raw >> 0) & 0xF).to(tl.float32) * q_scaled[None, :], axis=1) + q_zero) * softmax_scale
        qk1 = (tl.sum(((k_raw >> 4) & 0xF).to(tl.float32) * q_scaled[None, :], axis=1) + q_zero) * softmax_scale
        qk2 = (tl.sum(((k_raw >> 8) & 0xF).to(tl.float32) * q_scaled[None, :], axis=1) + q_zero) * softmax_scale
        qk3 = (tl.sum(((k_raw >> 12) & 0xF).to(tl.float32) * q_scaled[None, :], axis=1) + q_zero) * softmax_scale

        valid0 = start_n + offs_t < seqlen_k
        valid1 = start_n + 32 + offs_t < seqlen_k
        valid2 = start_n + 64 + offs_t < seqlen_k
        valid3 = start_n + 96 + offs_t < seqlen_k
        qk0 = tl.where(valid0, qk0, -float("inf"))
        qk1 = tl.where(valid1, qk1, -float("inf"))
        qk2 = tl.where(valid2, qk2, -float("inf"))
        qk3 = tl.where(valid3, qk3, -float("inf"))

        m_i = tl.maximum(
            tl.maximum(tl.max(qk0, axis=0), tl.max(qk1, axis=0)),
            tl.maximum(tl.max(qk2, axis=0), tl.max(qk3, axis=0)),
        )
        p0 = tl.where(valid0, tl.exp(qk0 - m_i), 0.0)
        p1 = tl.where(valid1, tl.exp(qk1 - m_i), 0.0)
        p2 = tl.where(valid2, tl.exp(qk2 - m_i), 0.0)
        p3 = tl.where(valid3, tl.exp(qk3 - m_i), 0.0)
        l_i = tl.sum(p0, axis=0) + tl.sum(p1, axis=0) + tl.sum(p2, axis=0) + tl.sum(p3, axis=0)

        v_ptrs0 = (
            VPack
            + off_b * stride_vb
            + off_h * stride_vh
            + (start_n + offs_t)[:, None] * stride_vn
            + offs_vp[None, :] * stride_vd
        )
        v_ptrs1 = v_ptrs0 + 32 * stride_vn
        v_ptrs2 = v_ptrs0 + 64 * stride_vn
        v_ptrs3 = v_ptrs0 + 96 * stride_vn
        mask_vp = offs_vp[None, :] < 32
        v_raw0 = tl.load(v_ptrs0, mask=valid0[:, None] & mask_vp, other=0).to(tl.int32)
        v_raw1 = tl.load(v_ptrs1, mask=valid1[:, None] & mask_vp, other=0).to(tl.int32)
        v_raw2 = tl.load(v_ptrs2, mask=valid2[:, None] & mask_vp, other=0).to(tl.int32)
        v_raw3 = tl.load(v_ptrs3, mask=valid3[:, None] & mask_vp, other=0).to(tl.int32)

        v_param_base = VParamsH + off_b * stride_vpb + off_h * stride_vph + group_id * 256 * stride_vpt
        v_scale0 = tl.load(v_param_base + offs_t * stride_vpt, mask=valid0, other=0.0).to(tl.float32)
        v_scale1 = tl.load(v_param_base + (32 + offs_t) * stride_vpt, mask=valid1, other=0.0).to(tl.float32)
        v_scale2 = tl.load(v_param_base + (64 + offs_t) * stride_vpt, mask=valid2, other=0.0).to(tl.float32)
        v_scale3 = tl.load(v_param_base + (96 + offs_t) * stride_vpt, mask=valid3, other=0.0).to(tl.float32)
        v_zero0 = tl.load(v_param_base + (128 + offs_t) * stride_vpt, mask=valid0, other=0.0).to(tl.float32)
        v_zero1 = tl.load(v_param_base + (128 + 32 + offs_t) * stride_vpt, mask=valid1, other=0.0).to(tl.float32)
        v_zero2 = tl.load(v_param_base + (128 + 64 + offs_t) * stride_vpt, mask=valid2, other=0.0).to(tl.float32)
        v_zero3 = tl.load(v_param_base + (128 + 96 + offs_t) * stride_vpt, mask=valid3, other=0.0).to(tl.float32)

        ps0 = p0 * v_scale0
        ps1 = p1 * v_scale1
        ps2 = p2 * v_scale2
        ps3 = p3 * v_scale3
        zero_acc = (
            tl.sum(p0 * v_zero0, axis=0)
            + tl.sum(p1 * v_zero1, axis=0)
            + tl.sum(p2 * v_zero2, axis=0)
            + tl.sum(p3 * v_zero3, axis=0)
        )
        acc0 = (
            tl.sum(ps0[:, None] * ((v_raw0 >> 0) & 0xF).to(tl.float32), axis=0)
            + tl.sum(ps1[:, None] * ((v_raw1 >> 0) & 0xF).to(tl.float32), axis=0)
            + tl.sum(ps2[:, None] * ((v_raw2 >> 0) & 0xF).to(tl.float32), axis=0)
            + tl.sum(ps3[:, None] * ((v_raw3 >> 0) & 0xF).to(tl.float32), axis=0)
            + zero_acc
        )
        acc1 = (
            tl.sum(ps0[:, None] * ((v_raw0 >> 4) & 0xF).to(tl.float32), axis=0)
            + tl.sum(ps1[:, None] * ((v_raw1 >> 4) & 0xF).to(tl.float32), axis=0)
            + tl.sum(ps2[:, None] * ((v_raw2 >> 4) & 0xF).to(tl.float32), axis=0)
            + tl.sum(ps3[:, None] * ((v_raw3 >> 4) & 0xF).to(tl.float32), axis=0)
            + zero_acc
        )
        acc2 = (
            tl.sum(ps0[:, None] * ((v_raw0 >> 8) & 0xF).to(tl.float32), axis=0)
            + tl.sum(ps1[:, None] * ((v_raw1 >> 8) & 0xF).to(tl.float32), axis=0)
            + tl.sum(ps2[:, None] * ((v_raw2 >> 8) & 0xF).to(tl.float32), axis=0)
            + tl.sum(ps3[:, None] * ((v_raw3 >> 8) & 0xF).to(tl.float32), axis=0)
            + zero_acc
        )

        m_next = tl.maximum(m_total, m_i)
        old_scale = tl.exp(m_total - m_next)
        new_scale = tl.where(group_active, tl.exp(m_i - m_next), 0.0)
        acc0_total = acc0_total * old_scale + acc0 * new_scale
        acc1_total = acc1_total * old_scale + acc1 * new_scale
        acc2_total = acc2_total * old_scale + acc2 * new_scale
        l_total = l_total * old_scale + l_i * new_scale
        m_total = m_next

    partial_base = PartialOut + off_b * stride_pob + off_h * stride_poh + split_id * stride_pos
    tl.store(partial_base + offs_vp, acc0_total, mask=offs_vp < 32)
    tl.store(partial_base + 32 + offs_vp, acc1_total, mask=offs_vp < 32)
    tl.store(partial_base + 64 + offs_vp, acc2_total, mask=offs_vp < 32)
    tl.store(PartialM + off_b * stride_mb + off_h * stride_mh + split_id, m_total)
    tl.store(PartialL + off_b * stride_lb + off_h * stride_lh + split_id, l_total)


@triton.heuristics(
    {
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_fp4_kvcache_decode_kernel(
    Q,
    KFP4,
    VFP4,
    KScales,
    VScales,
    KGlobalScale,
    VGlobalScale,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ksb,
    stride_ksh,
    stride_ksn,
    stride_ksd,
    stride_vsb,
    stride_vsh,
    stride_vsn,
    stride_vsd,
    stride_ob,
    stride_oh,
    nheads,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(0)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_d
    q = tl.load(q_ptrs, mask=offs_d < headdim, other=0.0).to(tl.float32)

    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_HEADDIM], dtype=tl.float32)

    for start_n in range(0, seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = _load_dequant_nvfp4_block(
            KFP4,
            KScales,
            KGlobalScale,
            off_b,
            off_h,
            start_n,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            stride_ksb,
            stride_ksh,
            stride_ksn,
            stride_ksd,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
        )
        qk = tl.sum(k * q[None, :], axis=1)
        if not EVEN_N:
            qk = tl.where((start_n + offs_n) < seqlen_k, qk, -float("inf"))
        qk = qk * softmax_scale

        m_ij = tl.maximum(tl.max(qk, axis=0), m_i)
        p = tl.exp(qk - m_ij)
        alpha = tl.exp(m_i - m_ij)
        acc_o = acc_o * alpha
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v = _load_dequant_nvfp4_block(
            VFP4,
            VScales,
            VGlobalScale,
            off_b,
            off_h,
            start_n,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            stride_vb,
            stride_vh,
            stride_vn,
            stride_vd,
            stride_vsb,
            stride_vsh,
            stride_vsn,
            stride_vsd,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
        )
        acc_o += tl.sum(p[:, None] * v, axis=0)
        m_i = m_ij

    acc_o = acc_o / l_i
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + offs_d
    tl.store(out_ptrs, acc_o, mask=offs_d < headdim)
    tl.store(Lse + off_hb * seqlen_q_rounded, m_i + tl.log(l_i))


def _normalize_fp4_scale_tensor(scale, name):
    if scale.dtype == torch.uint8:
        return scale.view(torch.float8_e4m3fn)
    if scale.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{name} must have dtype torch.float8_e4m3fn or torch.uint8, got {scale.dtype}")
    return scale


def _normalize_global_scale(scale, device, name):
    if torch.is_tensor(scale):
        if scale.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor")
        return scale.to(device=device, dtype=torch.float32).reshape(1)
    return torch.tensor([float(scale)], dtype=torch.float32, device=device)


def _flash_attn_fp4_kvcache_forward(
    q,
    k_fp4,
    v_fp4,
    k_scales,
    v_scales,
    k_global_scale,
    v_global_scale,
    causal=False,
    softmax_scale=None,
    use_split_k=True,
    split_k_block_size=128,
    use_split_k_packed=True,
    use_split_k_tensor_cores=True,
    use_decode_kernel=False,
):
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, d_pack = k_fp4.shape
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16 query"
    assert q.is_cuda and k_fp4.is_cuda and v_fp4.is_cuda
    assert k_fp4.dtype == torch.uint8 and v_fp4.dtype == torch.uint8
    assert d <= 128 and d % 16 == 0, "NVFP4 attention requires headdim <= 128 and divisible by 16"
    assert k_fp4.shape == (batch, seqlen_k, nheads, d // 2)
    assert v_fp4.shape == (batch, seqlen_k, nheads, d // 2)
    assert d_pack == d // 2
    assert k_scales.shape == (batch, seqlen_k, nheads, d // 16)
    assert v_scales.shape == (batch, seqlen_k, nheads, d // 16)
    q = q if q.stride(-1) == 1 else q.contiguous()
    k_fp4 = k_fp4 if k_fp4.stride(-1) == 1 else k_fp4.contiguous()
    v_fp4 = v_fp4 if v_fp4.stride(-1) == 1 else v_fp4.contiguous()
    k_scales = _normalize_fp4_scale_tensor(k_scales, "k_scales")
    v_scales = _normalize_fp4_scale_tensor(v_scales, "v_scales")
    k_scales = k_scales if k_scales.stride(-1) == 1 else k_scales.contiguous()
    v_scales = v_scales if v_scales.stride(-1) == 1 else v_scales.contiguous()
    k_global_scale = _normalize_global_scale(k_global_scale, q.device, "k_global_scale")
    v_global_scale = _normalize_global_scale(v_global_scale, q.device, "v_global_scale")
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    if use_split_k and seqlen_q == 1 and not causal:
        num_splits = triton.cdiv(seqlen_k, split_k_block_size)
        BLOCK_SPLITS = triton.next_power_of_2(num_splits)
        stage1_grid = (num_splits, batch * nheads)
        if use_split_k_packed:
            partial_o = torch.empty((batch, nheads, num_splits, d), device=q.device, dtype=q.dtype)
            partial_m = torch.empty((batch, nheads, num_splits), device=q.device, dtype=torch.float32)
            partial_l = torch.empty((batch, nheads, num_splits), device=q.device, dtype=torch.float32)
            _fwd_fp4_kvcache_split_k_stage1_packed_kernel[stage1_grid](
                q,
                k_fp4,
                v_fp4,
                k_scales,
                v_scales,
                k_global_scale,
                v_global_scale,
                partial_o,
                partial_m,
                partial_l,
                softmax_scale,
                q.stride(0),
                q.stride(2),
                k_fp4.stride(0),
                k_fp4.stride(2),
                k_fp4.stride(1),
                k_fp4.stride(3),
                v_fp4.stride(0),
                v_fp4.stride(2),
                v_fp4.stride(1),
                v_fp4.stride(3),
                k_scales.stride(0),
                k_scales.stride(2),
                k_scales.stride(1),
                k_scales.stride(3),
                v_scales.stride(0),
                v_scales.stride(2),
                v_scales.stride(1),
                v_scales.stride(3),
                partial_o.stride(0),
                partial_o.stride(1),
                partial_o.stride(2),
                partial_m.stride(0),
                partial_m.stride(1),
                partial_l.stride(0),
                partial_l.stride(1),
                nheads,
                seqlen_k,
                d,
                seqlen_k // 32,
                BLOCK_HEADDIM,
                BLOCK_N=split_k_block_size,
                BLOCK_DPACK=BLOCK_HEADDIM // 2,
                num_warps=8 if d > 64 else 4,
                num_stages=1,
            )
            _fwd_fp4_kvcache_split_k_stage2_raw_kernel[(batch * nheads,)](
                partial_o,
                partial_m,
                partial_l,
                o,
                partial_o.stride(0),
                partial_o.stride(1),
                partial_o.stride(2),
                partial_m.stride(0),
                partial_m.stride(1),
                partial_l.stride(0),
                partial_l.stride(1),
                o.stride(0),
                o.stride(2),
                nheads,
                num_splits,
                d,
                BLOCK_SPLITS=BLOCK_SPLITS,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                num_warps=8 if d > 64 else 4,
                num_stages=1,
            )
            return o, lse, softmax_scale

        partial_o = torch.empty((batch, nheads, num_splits, d), device=q.device, dtype=torch.float32)
        partial_lse = torch.empty((batch, nheads, num_splits), device=q.device, dtype=torch.float32)
        if use_split_k_tensor_cores:
            stage1_kernel = _fwd_fp4_kvcache_split_k_stage1_tc_kernel
        else:
            stage1_kernel = _fwd_fp4_kvcache_split_k_stage1_kernel
        stage1_kwargs = {
            "BLOCK_N": split_k_block_size,
            "num_warps": 8 if d > 64 else 4,
            "num_stages": 1,
        }
        if use_split_k_tensor_cores:
            stage1_kwargs["BLOCK_M"] = 16
        stage1_kernel[stage1_grid](
            q,
            k_fp4,
            v_fp4,
            k_scales,
            v_scales,
            k_global_scale,
            v_global_scale,
            partial_o,
            partial_lse,
            softmax_scale,
            q.stride(0),
            q.stride(2),
            k_fp4.stride(0),
            k_fp4.stride(2),
            k_fp4.stride(1),
            k_fp4.stride(3),
            v_fp4.stride(0),
            v_fp4.stride(2),
            v_fp4.stride(1),
            v_fp4.stride(3),
            k_scales.stride(0),
            k_scales.stride(2),
            k_scales.stride(1),
            k_scales.stride(3),
            v_scales.stride(0),
            v_scales.stride(2),
            v_scales.stride(1),
            v_scales.stride(3),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            nheads,
            seqlen_k,
            d,
            seqlen_k // 32,
            BLOCK_HEADDIM,
            **stage1_kwargs,
        )
        _fwd_fp4_kvcache_split_k_stage2_kernel[(batch * nheads,)](
            partial_o,
            partial_lse,
            o,
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            o.stride(0),
            o.stride(2),
            nheads,
            num_splits,
            d,
            BLOCK_SPLITS=BLOCK_SPLITS,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            num_warps=8 if d > 64 else 4,
            num_stages=1,
        )
        return o, lse, softmax_scale

    if use_decode_kernel and seqlen_q == 1 and not causal:
        BLOCK_N = 128
        grid = (batch * nheads,)
        _fwd_fp4_kvcache_decode_kernel[grid](
            q,
            k_fp4,
            v_fp4,
            k_scales,
            v_scales,
            k_global_scale,
            v_global_scale,
            o,
            lse,
            softmax_scale,
            q.stride(0),
            q.stride(2),
            k_fp4.stride(0),
            k_fp4.stride(2),
            k_fp4.stride(1),
            k_fp4.stride(3),
            v_fp4.stride(0),
            v_fp4.stride(2),
            v_fp4.stride(1),
            v_fp4.stride(3),
            k_scales.stride(0),
            k_scales.stride(2),
            k_scales.stride(1),
            k_scales.stride(3),
            v_scales.stride(0),
            v_scales.stride(2),
            v_scales.stride(1),
            v_scales.stride(3),
            o.stride(0),
            o.stride(2),
            nheads,
            seqlen_k,
            seqlen_q_rounded,
            d,
            seqlen_k // 32,
            BLOCK_HEADDIM,
            BLOCK_N=BLOCK_N,
            num_warps=8 if d > 64 else 4,
            num_stages=1,
        )
        return o, lse, softmax_scale

    BLOCK_M = 16 if seqlen_q <= 16 else 32
    BLOCK_N = 64
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_fp4_kvcache_kernel[grid](
        q,
        k_fp4,
        v_fp4,
        k_scales,
        v_scales,
        k_global_scale,
        v_global_scale,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k_fp4.stride(0),
        k_fp4.stride(2),
        k_fp4.stride(1),
        k_fp4.stride(3),
        v_fp4.stride(0),
        v_fp4.stride(2),
        v_fp4.stride(1),
        v_fp4.stride(3),
        k_scales.stride(0),
        k_scales.stride(2),
        k_scales.stride(1),
        k_scales.stride(3),
        v_scales.stride(0),
        v_scales.stride(2),
        v_scales.stride(1),
        v_scales.stride(3),
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale


def flash_attn_fp4_kvcache_func(
    q,
    k_fp4,
    v_fp4,
    k_scales,
    v_scales,
    k_global_scale,
    v_global_scale,
    causal=False,
    softmax_scale=None,
    use_split_k=True,
    split_k_block_size=128,
    use_split_k_packed=True,
    use_split_k_tensor_cores=True,
    use_decode_kernel=False,
):
    """Forward-only fused attention over NVFP4 KV cache.

    q: [batch, seqlen_q, nheads, headdim], fp16/bf16.
    k_fp4/v_fp4: [batch, seqlen_k, nheads, headdim // 2], uint8, two E2M1 FP4 values per byte.
    k_scales/v_scales: [batch, seqlen_k, nheads, headdim // 16], FP8 E4M3 or its uint8 view.
    global scales are scalar tensors/floats. Dequantization is
    fp4_e2m1 * block_scale * global_scale, matching FlashInfer's NVFP4 KV dequant path.
    """
    o, _, _ = _flash_attn_fp4_kvcache_forward(
        q,
        k_fp4,
        v_fp4,
        k_scales,
        v_scales,
        k_global_scale,
        v_global_scale,
        causal=causal,
        softmax_scale=softmax_scale,
        use_split_k=use_split_k,
        split_k_block_size=split_k_block_size,
        use_split_k_packed=use_split_k_packed,
        use_split_k_tensor_cores=use_split_k_tensor_cores,
        use_decode_kernel=use_decode_kernel,
    )
    return o


def _get_bitdecode96_triton_workspace(q, batch, nheads, num_splits, d):
    key = (q.device.index, batch, nheads, num_splits, d, q.dtype)
    workspace = _BITDECODE96_TRITON_WORKSPACES.get(key)
    if workspace is None:
        partial_o = torch.empty((batch, nheads, num_splits, d), device=q.device, dtype=q.dtype)
        partial_m = torch.empty((batch, nheads, num_splits), device=q.device, dtype=torch.float32)
        partial_l = torch.empty((batch, nheads, num_splits), device=q.device, dtype=torch.float32)
        workspace = (partial_o, partial_m, partial_l)
        _BITDECODE96_TRITON_WORKSPACES[key] = workspace
    return workspace


def _flash_attn_bitdecode96_i4_triton_forward(
    q,
    k_pack,
    k_params,
    v_pack,
    v_params,
    softmax_scale=None,
    effective_seqlen_k=None,
    use_tensor_cores=False,
    groups_per_split=1,
    cache_workspace=True,
    use_cuda_graph=False,
):
    batch, seqlen_q, nheads, d = q.shape
    if seqlen_q != 1 or d != 96:
        raise ValueError(f"Triton bitdecode96 path requires q shape [B, 1, H, 96], got {tuple(q.shape)}")
    if k_pack.dtype != torch.uint16 or v_pack.dtype != torch.uint16:
        raise TypeError("Triton bitdecode96 path expects uint16 k_pack/v_pack")
    if k_params.dtype != torch.float32 or v_params.dtype != torch.float32:
        raise TypeError("Triton bitdecode96 path expects float32 half2-storage params")
    if k_pack.shape[0] != batch or k_pack.shape[2] != nheads or k_pack.shape[3] != 96:
        raise ValueError(f"k_pack must be [B, L/4, H, 96], got {tuple(k_pack.shape)}")
    if v_pack.shape[0] != batch or v_pack.shape[2] != nheads or v_pack.shape[3] != 32:
        raise ValueError(f"v_pack must be [B, L, H, 32], got {tuple(v_pack.shape)}")
    physical_seqlen = int(v_pack.shape[1])
    seqlen_k = physical_seqlen if effective_seqlen_k is None or int(effective_seqlen_k) <= 0 else int(effective_seqlen_k)
    if seqlen_k <= 0 or seqlen_k > physical_seqlen:
        raise ValueError(f"effective_seqlen_k must be in [1, {physical_seqlen}], got {seqlen_k}")
    if k_params.shape[1] * 128 < seqlen_k:
        raise ValueError("k_params does not cover effective_seqlen_k groups")

    q = q if q.is_contiguous() else q.contiguous()
    k_pack = k_pack if k_pack.is_contiguous() else k_pack.contiguous()
    k_params = k_params if k_params.is_contiguous() else k_params.contiguous()
    v_pack = v_pack if v_pack.is_contiguous() else v_pack.contiguous()
    v_params = v_params if v_params.is_contiguous() else v_params.contiguous()
    k_params_h = k_params.view(torch.float16)
    v_params_h = v_params.view(torch.float16)
    k_pack_stage1 = k_pack
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    if groups_per_split < 1:
        raise ValueError(f"groups_per_split must be >= 1, got {groups_per_split}")
    token_groups = triton.cdiv(seqlen_k, 128)
    num_splits = triton.cdiv(token_groups, int(groups_per_split))
    block_splits = triton.next_power_of_2(num_splits)
    if cache_workspace:
        partial_o, partial_m, partial_l = _get_bitdecode96_triton_workspace(q, batch, nheads, num_splits, d)
    else:
        partial_o = torch.empty((batch, nheads, num_splits, d), device=q.device, dtype=q.dtype)
        partial_m = torch.empty((batch, nheads, num_splits), device=q.device, dtype=torch.float32)
        partial_l = torch.empty((batch, nheads, num_splits), device=q.device, dtype=torch.float32)

    if use_tensor_cores:
        stage1_kernel = _fwd_bitdecode96_i4_stage1_tc_kernel
    elif groups_per_split == 1:
        stage1_kernel = _fwd_bitdecode96_i4_stage1_group96_kernel
    else:
        stage1_kernel = _fwd_bitdecode96_i4_stage1_groupx_kernel
    stage1_kwargs = {
        "BLOCK_D": 128,
        "num_warps": 4,
        "num_stages": 1,
    }
    if use_tensor_cores:
        stage1_kwargs.update({"BLOCK_M": 16, "BLOCK_N": 128})
    elif groups_per_split == 1:
        stage1_kwargs.pop("BLOCK_D")
        stage1_kwargs.update(
            {
                "BLOCK_D0": 64,
                "BLOCK_D1": 32,
                "BLOCK_TPACK": 32,
                "BLOCK_VPACK": 32,
                "num_warps": 1,
                "num_stages": 4,
            }
        )
    else:
        stage1_kwargs.update({"GROUPS_PER_SPLIT": int(groups_per_split), "BLOCK_TPACK": 32, "BLOCK_VPACK": 32})
    stage2_kernel = (
        _fwd_bitdecode_split_k_stage2_raw_exp2_kernel
        if (not use_tensor_cores and groups_per_split == 1)
        else _fwd_fp4_kvcache_split_k_stage2_raw_kernel
    )
    stage2_num_warps = 8 if (not use_tensor_cores and groups_per_split == 1) else 4

    def launch(o):
        stage1_kernel[(num_splits, batch * nheads)](
            q,
            k_pack_stage1,
            k_params_h,
            v_pack,
            v_params_h,
            partial_o,
            partial_m,
            partial_l,
            float(softmax_scale),
            q.stride(0),
            q.stride(2),
            k_pack_stage1.stride(0),
            k_pack_stage1.stride(2),
            k_pack_stage1.stride(1),
            k_pack_stage1.stride(3),
            k_params_h.stride(0),
            k_params_h.stride(1),
            k_params_h.stride(2),
            k_params_h.stride(3),
            v_pack.stride(0),
            v_pack.stride(2),
            v_pack.stride(1),
            v_pack.stride(3),
            v_params_h.stride(0),
            v_params_h.stride(2),
            v_params_h.stride(3),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_m.stride(0),
            partial_m.stride(1),
            partial_l.stride(0),
            partial_l.stride(1),
            nheads,
            seqlen_k,
            **stage1_kwargs,
        )
        stage2_kernel[(batch * nheads,)](
            partial_o,
            partial_m,
            partial_l,
            o,
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_m.stride(0),
            partial_m.stride(1),
            partial_l.stride(0),
            partial_l.stride(1),
            o.stride(0),
            o.stride(2),
            nheads,
            num_splits,
            d,
            BLOCK_SPLITS=block_splits,
            BLOCK_HEADDIM=128,
            num_warps=stage2_num_warps,
            num_stages=1,
        )

    if use_cuda_graph:
        if use_tensor_cores or groups_per_split != 1:
            raise ValueError("use_cuda_graph is only enabled for the default hdim96 int4 Triton path")
        graph_key = (
            q.device.index,
            int(q.data_ptr()),
            int(k_pack.data_ptr()),
            int(k_params.data_ptr()),
            int(v_pack.data_ptr()),
            int(v_params.data_ptr()),
            batch,
            nheads,
            seqlen_k,
            num_splits,
            d,
            q.dtype,
            float(softmax_scale),
        )
        graph_entry = _BITDECODE96_TRITON_GRAPHS.get(graph_key)
        if graph_entry is None:
            graph_o = torch.empty_like(q)
            launch(graph_o)
            torch.cuda.synchronize(q.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(graph_o)
            graph_entry = (graph, graph_o)
            _BITDECODE96_TRITON_GRAPHS[graph_key] = graph_entry
        graph, graph_o = graph_entry
        graph.replay()
        return graph_o

    o = torch.empty_like(q)
    launch(o)
    return o


def flash_attn_bitdecode96_i4_triton_kvcache_func(
    q,
    k_pack,
    k_params,
    v_pack,
    v_params,
    softmax_scale=None,
    effective_seqlen_k=None,
    use_tensor_cores=False,
    groups_per_split=1,
    cache_workspace=True,
    use_cuda_graph=False,
):
    """Forward-only Triton decode attention for BitDecoding's hdim96 kv4 layout."""
    return _flash_attn_bitdecode96_i4_triton_forward(
        q,
        k_pack,
        k_params,
        v_pack,
        v_params,
        softmax_scale=softmax_scale,
        effective_seqlen_k=effective_seqlen_k,
        use_tensor_cores=use_tensor_cores,
        groups_per_split=groups_per_split,
        cache_workspace=cache_workspace,
        use_cuda_graph=use_cuda_graph,
    )


def flash_attn_bitdecode_kvcache_func(
    q,
    k_pack,
    k_params,
    v_pack,
    v_params,
    block_table=None,
    softmax_scale=None,
    quant_mode="k-channel",
    group_size=128,
    num_bits=4,
    effective_seqlen_k=None,
    num_splits=0,
    use_triton_96=False,
    use_triton_tensor_cores=False,
    triton_groups_per_split=1,
    cache_triton_workspace=True,
    use_triton_cuda_graph=False,
):
    """Forward-only decode attention over BitDecoding packed int KV cache.

    This is the fast hdim96/128 path for the BitDecoding layout:
    k_pack/k_params/v_pack/v_params are the tensors produced by
    BitDecoding's kvcache_pack_int / allocate_packed_kv helpers.
    """
    if num_bits not in (2, 4):
        raise ValueError(f"num_bits must be 2 or 4, got {num_bits}")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if use_triton_96:
        if block_table is not None:
            raise ValueError("use_triton_96 does not support block_table")
        if quant_mode != "k-channel" or group_size != 128 or num_bits != 4:
            raise ValueError("use_triton_96 requires quant_mode='k-channel', group_size=128, num_bits=4")
        return _flash_attn_bitdecode96_i4_triton_forward(
            q,
            k_pack,
            k_params,
            v_pack,
            v_params,
            softmax_scale=softmax_scale,
            effective_seqlen_k=effective_seqlen_k,
            use_tensor_cores=use_triton_tensor_cores,
            groups_per_split=triton_groups_per_split,
            cache_workspace=cache_triton_workspace,
            use_cuda_graph=use_triton_cuda_graph,
        )
    try:
        import bit_decode_cuda
    except ImportError as exc:
        import sys
        from pathlib import Path

        bitdecode_repo = Path("/mnt/upfs/jiazhen.wu/wjz/quant/BitDecoding")
        if bitdecode_repo.exists() and str(bitdecode_repo) not in sys.path:
            sys.path.insert(0, str(bitdecode_repo))
            try:
                import bit_decode_cuda
            except ImportError:
                raise ImportError(
                    "flash_attn_bitdecode_kvcache_func requires BitDecoding's bit_decode_cuda extension "
                    "to be importable. Add /mnt/upfs/jiazhen.wu/wjz/quant/BitDecoding to PYTHONPATH "
                    "or run from that repository."
                ) from exc
        else:
            raise ImportError(
                "flash_attn_bitdecode_kvcache_func requires BitDecoding's bit_decode_cuda extension "
                "to be importable."
            ) from exc

    block_table_arg = block_table
    seqlen_arg = -1 if effective_seqlen_k is None else int(effective_seqlen_k)
    run = bit_decode_cuda.fwd_kvcache_i4 if num_bits == 4 else bit_decode_cuda.fwd_kvcache_i2
    q_arg = q if q.is_contiguous() else q.contiguous()
    k_pack_arg = k_pack if k_pack.is_contiguous() else k_pack.contiguous()
    k_params_arg = k_params if k_params.is_contiguous() else k_params.contiguous()
    v_pack_arg = v_pack if v_pack.is_contiguous() else v_pack.contiguous()
    v_params_arg = v_params if v_params.is_contiguous() else v_params.contiguous()
    return run(
        q_arg,
        k_pack_arg,
        k_params_arg,
        v_pack_arg,
        v_params_arg,
        block_table_arg,
        float(softmax_scale),
        quant_mode,
        int(group_size),
        False,
        -1,
        -1,
        0.0,
        True,
        int(num_splits),
        seqlen_arg,
    )
