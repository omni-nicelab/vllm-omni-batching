import math

import torch
import triton
import triton.language as tl


GROUP_SIZE = 128
HEAD_DIM = 96
K_PARAM_HALF_DIM = 256
V_PACK_DIM = 32


@triton.jit
def _round_i4(x):
    x = tl.floor(x + 0.5).to(tl.int32)
    x = tl.minimum(tl.maximum(x, 0), 15)
    return x


@triton.jit
def _pack_k_prefill_kernel(
    K,
    KPack,
    KParamsH,
    stride_ks,
    stride_kh,
    stride_kp_n,
    stride_kp_h,
    stride_kp_d,
    stride_kpar_g,
    stride_kpar_h,
    stride_kpar_d,
    BLOCK_T: tl.constexpr,
):
    group_id = tl.program_id(0)
    head_id = tl.program_id(1)
    dim_id = tl.program_id(2)
    offs_t = tl.arange(0, BLOCK_T)
    src = tl.load(K + (group_id * 128 + offs_t) * stride_ks + head_id * stride_kh + dim_id)
    src_f = src.to(tl.float32)
    mn = tl.min(src_f, axis=0)
    mx = tl.max(src_f, axis=0)
    scale = (mx - mn) * 0.06666666666666667
    inv_scale = tl.where(scale > 0.0, 1.0 / scale, 0.0)
    tl.store(KParamsH + group_id * stride_kpar_g + head_id * stride_kpar_h + dim_id * stride_kpar_d, scale)
    tl.store(KParamsH + group_id * stride_kpar_g + head_id * stride_kpar_h + (128 + dim_id) * stride_kpar_d, mn)

    offs_w = tl.arange(0, 32)
    k0 = tl.load(K + (group_id * 128 + offs_w) * stride_ks + head_id * stride_kh + dim_id).to(tl.float32)
    k1 = tl.load(K + (group_id * 128 + 32 + offs_w) * stride_ks + head_id * stride_kh + dim_id).to(tl.float32)
    k2 = tl.load(K + (group_id * 128 + 64 + offs_w) * stride_ks + head_id * stride_kh + dim_id).to(tl.float32)
    k3 = tl.load(K + (group_id * 128 + 96 + offs_w) * stride_ks + head_id * stride_kh + dim_id).to(tl.float32)
    q0 = _round_i4((k0 - mn) * inv_scale)
    q1 = _round_i4((k1 - mn) * inv_scale)
    q2 = _round_i4((k2 - mn) * inv_scale)
    q3 = _round_i4((k3 - mn) * inv_scale)
    packed = (q0 | (q1 << 4) | (q2 << 8) | (q3 << 12)).to(tl.uint32)
    tl.store(
        KPack + (group_id * 32 + offs_w) * stride_kp_n + head_id * stride_kp_h + dim_id * stride_kp_d,
        packed,
    )


@triton.jit
def _pack_v_prefill_kernel(
    V,
    VPack,
    VParamsH,
    stride_vs,
    stride_vh,
    stride_vp_n,
    stride_vp_h,
    stride_vp_d,
    stride_vpar_h,
    stride_vpar_t,
    BLOCK_D: tl.constexpr,
):
    group_id = tl.program_id(0)
    head_id = tl.program_id(1)
    token_off = tl.program_id(2)
    token_idx = group_id * 128 + token_off
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < 96
    vals = tl.load(V + token_idx * stride_vs + head_id * stride_vh + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    vals_for_reduce = tl.where(mask_d, vals, float("inf"))
    mn = tl.min(vals_for_reduce, axis=0)
    vals_for_reduce = tl.where(mask_d, vals, -float("inf"))
    mx = tl.max(vals_for_reduce, axis=0)
    scale = (mx - mn) * 0.06666666666666667
    inv_scale = tl.where(scale > 0.0, 1.0 / scale, 0.0)
    param_base = head_id * stride_vpar_h + group_id * 256 * stride_vpar_t
    tl.store(VParamsH + param_base + token_off * stride_vpar_t, scale)
    tl.store(VParamsH + param_base + (128 + token_off) * stride_vpar_t, mn)

    offs_vp = tl.arange(0, 32)
    v0 = tl.load(V + token_idx * stride_vs + head_id * stride_vh + offs_vp).to(tl.float32)
    v1 = tl.load(V + token_idx * stride_vs + head_id * stride_vh + 32 + offs_vp).to(tl.float32)
    v2 = tl.load(V + token_idx * stride_vs + head_id * stride_vh + 64 + offs_vp).to(tl.float32)
    q0 = _round_i4((v0 - mn) * inv_scale)
    q1 = _round_i4((v1 - mn) * inv_scale)
    q2 = _round_i4((v2 - mn) * inv_scale)
    packed = (q0 | (q1 << 4) | (q2 << 8)).to(tl.uint32)
    tl.store(VPack + token_idx * stride_vp_n + head_id * stride_vp_h + offs_vp * stride_vp_d, packed)


@triton.jit
def _store_decode_residual_kernel(
    K,
    V,
    ContextLens,
    KResidual,
    VResidual,
    stride_kh,
    stride_vh,
    stride_r_t,
    stride_r_h,
    BLOCK_D: tl.constexpr,
):
    head_id = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    n = tl.load(ContextLens)
    if n <= 0:
        return
    token_off = (n - 1) % 128
    mask = offs_d < 96
    k = tl.load(K + head_id * stride_kh + offs_d, mask=mask, other=0.0)
    v = tl.load(V + head_id * stride_vh + offs_d, mask=mask, other=0.0)
    dst = token_off * stride_r_t + head_id * stride_r_h + offs_d
    tl.store(KResidual + dst, k, mask=mask)
    tl.store(VResidual + dst, v, mask=mask)


@triton.jit
def _pack_k_decode_if_full_kernel(
    ContextLens,
    KResidual,
    KPack,
    KParamsH,
    stride_r_t,
    stride_r_h,
    stride_kp_n,
    stride_kp_h,
    stride_kp_d,
    stride_kpar_g,
    stride_kpar_h,
    stride_kpar_d,
    BLOCK_T: tl.constexpr,
):
    head_id = tl.program_id(0)
    dim_id = tl.program_id(1)
    n = tl.load(ContextLens)
    if n <= 0 or n % 128 != 0:
        return
    group_id = n // 128 - 1
    offs_t = tl.arange(0, BLOCK_T)
    src = tl.load(KResidual + offs_t * stride_r_t + head_id * stride_r_h + dim_id).to(tl.float32)
    mn = tl.min(src, axis=0)
    mx = tl.max(src, axis=0)
    scale = (mx - mn) * 0.06666666666666667
    inv_scale = tl.where(scale > 0.0, 1.0 / scale, 0.0)
    tl.store(KParamsH + group_id * stride_kpar_g + head_id * stride_kpar_h + dim_id * stride_kpar_d, scale)
    tl.store(KParamsH + group_id * stride_kpar_g + head_id * stride_kpar_h + (128 + dim_id) * stride_kpar_d, mn)

    offs_w = tl.arange(0, 32)
    k0 = tl.load(KResidual + offs_w * stride_r_t + head_id * stride_r_h + dim_id).to(tl.float32)
    k1 = tl.load(KResidual + (32 + offs_w) * stride_r_t + head_id * stride_r_h + dim_id).to(tl.float32)
    k2 = tl.load(KResidual + (64 + offs_w) * stride_r_t + head_id * stride_r_h + dim_id).to(tl.float32)
    k3 = tl.load(KResidual + (96 + offs_w) * stride_r_t + head_id * stride_r_h + dim_id).to(tl.float32)
    q0 = _round_i4((k0 - mn) * inv_scale)
    q1 = _round_i4((k1 - mn) * inv_scale)
    q2 = _round_i4((k2 - mn) * inv_scale)
    q3 = _round_i4((k3 - mn) * inv_scale)
    packed = (q0 | (q1 << 4) | (q2 << 8) | (q3 << 12)).to(tl.uint32)
    tl.store(
        KPack + (group_id * 32 + offs_w) * stride_kp_n + head_id * stride_kp_h + dim_id * stride_kp_d,
        packed,
    )


@triton.jit
def _pack_v_decode_if_full_kernel(
    ContextLens,
    VResidual,
    VPack,
    VParamsH,
    stride_r_t,
    stride_r_h,
    stride_vp_n,
    stride_vp_h,
    stride_vp_d,
    stride_vpar_h,
    stride_vpar_t,
    BLOCK_D: tl.constexpr,
):
    head_id = tl.program_id(0)
    token_off = tl.program_id(1)
    n = tl.load(ContextLens)
    if n <= 0 or n % 128 != 0:
        return
    group_id = n // 128 - 1
    token_idx = group_id * 128 + token_off
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < 96
    vals = tl.load(VResidual + token_off * stride_r_t + head_id * stride_r_h + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    vals_for_reduce = tl.where(mask_d, vals, float("inf"))
    mn = tl.min(vals_for_reduce, axis=0)
    vals_for_reduce = tl.where(mask_d, vals, -float("inf"))
    mx = tl.max(vals_for_reduce, axis=0)
    scale = (mx - mn) * 0.06666666666666667
    inv_scale = tl.where(scale > 0.0, 1.0 / scale, 0.0)
    param_base = head_id * stride_vpar_h + group_id * 256 * stride_vpar_t
    tl.store(VParamsH + param_base + token_off * stride_vpar_t, scale)
    tl.store(VParamsH + param_base + (128 + token_off) * stride_vpar_t, mn)

    offs_vp = tl.arange(0, 32)
    v0 = tl.load(VResidual + token_off * stride_r_t + head_id * stride_r_h + offs_vp).to(tl.float32)
    v1 = tl.load(VResidual + token_off * stride_r_t + head_id * stride_r_h + 32 + offs_vp).to(tl.float32)
    v2 = tl.load(VResidual + token_off * stride_r_t + head_id * stride_r_h + 64 + offs_vp).to(tl.float32)
    q0 = _round_i4((v0 - mn) * inv_scale)
    q1 = _round_i4((v1 - mn) * inv_scale)
    q2 = _round_i4((v2 - mn) * inv_scale)
    packed = (q0 | (q1 << 4) | (q2 << 8)).to(tl.uint32)
    tl.store(VPack + token_idx * stride_vp_n + head_id * stride_vp_h + offs_vp * stride_vp_d, packed)


@triton.jit
def _bitdecode96_stage1_kernel(
    Q,
    KPack,
    KParamsH,
    VPack,
    VParamsH,
    ContextLens,
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
    stride_kpg,
    stride_kph,
    stride_kpd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_vph,
    stride_vpt,
    stride_poh,
    stride_pos,
    stride_mh,
    stride_lh,
    nheads,
    BLOCK_D0: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_TPACK: tl.constexpr,
    BLOCK_VPACK: tl.constexpr,
):
    group_id = tl.program_id(0)
    off_h = tl.program_id(1)
    offs_t = tl.arange(0, BLOCK_TPACK)
    offs_d0 = tl.arange(0, BLOCK_D0)
    offs_d1 = tl.arange(0, BLOCK_D1)
    offs_vp = tl.arange(0, BLOCK_VPACK)
    context_len = tl.load(ContextLens)
    seqlen_k = (context_len // 128) * 128
    start_n = group_id * 128

    q0 = tl.load(Q + off_h * stride_qh + offs_d0).to(tl.float32)
    q1 = tl.load(Q + off_h * stride_qh + 64 + offs_d1).to(tl.float32)
    k_scale0 = tl.load(KParamsH + group_id * stride_kpg + off_h * stride_kph + offs_d0 * stride_kpd).to(tl.float32)
    k_scale1 = tl.load(KParamsH + group_id * stride_kpg + off_h * stride_kph + (64 + offs_d1) * stride_kpd).to(tl.float32)
    k_zero0 = tl.load(KParamsH + group_id * stride_kpg + off_h * stride_kph + (128 + offs_d0) * stride_kpd).to(tl.float32)
    k_zero1 = tl.load(KParamsH + group_id * stride_kpg + off_h * stride_kph + (128 + 64 + offs_d1) * stride_kpd).to(tl.float32)
    q_scaled0 = q0 * k_scale0
    q_scaled1 = q1 * k_scale1
    q_zero = tl.sum(q0 * k_zero0, axis=0) + tl.sum(q1 * k_zero1, axis=0)

    k_ptrs0 = KPack + off_h * stride_kh + (group_id * 32 + offs_t)[:, None] * stride_kn + offs_d0[None, :] * stride_kd
    k_ptrs1 = KPack + off_h * stride_kh + (group_id * 32 + offs_t)[:, None] * stride_kn + (64 + offs_d1)[None, :] * stride_kd
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

    v_ptrs0 = VPack + off_h * stride_vh + (start_n + offs_t)[:, None] * stride_vn + offs_vp[None, :] * stride_vd
    v_ptrs1 = v_ptrs0 + 32 * stride_vn
    v_ptrs2 = v_ptrs0 + 64 * stride_vn
    v_ptrs3 = v_ptrs0 + 96 * stride_vn
    v_raw0 = tl.load(v_ptrs0, mask=valid0[:, None], other=0).to(tl.int32)
    v_raw1 = tl.load(v_ptrs1, mask=valid1[:, None], other=0).to(tl.int32)
    v_raw2 = tl.load(v_ptrs2, mask=valid2[:, None], other=0).to(tl.int32)
    v_raw3 = tl.load(v_ptrs3, mask=valid3[:, None], other=0).to(tl.int32)

    v_param_base = off_h * stride_vph + group_id * 256 * stride_vpt
    v_scale0 = tl.load(VParamsH + v_param_base + offs_t * stride_vpt, mask=valid0, other=0.0).to(tl.float32)
    v_scale1 = tl.load(VParamsH + v_param_base + (32 + offs_t) * stride_vpt, mask=valid1, other=0.0).to(tl.float32)
    v_scale2 = tl.load(VParamsH + v_param_base + (64 + offs_t) * stride_vpt, mask=valid2, other=0.0).to(tl.float32)
    v_scale3 = tl.load(VParamsH + v_param_base + (96 + offs_t) * stride_vpt, mask=valid3, other=0.0).to(tl.float32)
    v_zero0 = tl.load(VParamsH + v_param_base + (128 + offs_t) * stride_vpt, mask=valid0, other=0.0).to(tl.float32)
    v_zero1 = tl.load(VParamsH + v_param_base + (128 + 32 + offs_t) * stride_vpt, mask=valid1, other=0.0).to(tl.float32)
    v_zero2 = tl.load(VParamsH + v_param_base + (128 + 64 + offs_t) * stride_vpt, mask=valid2, other=0.0).to(tl.float32)
    v_zero3 = tl.load(VParamsH + v_param_base + (128 + 96 + offs_t) * stride_vpt, mask=valid3, other=0.0).to(tl.float32)

    ps0 = p0 * v_scale0
    ps1 = p1 * v_scale1
    ps2 = p2 * v_scale2
    ps3 = p3 * v_scale3
    zero_acc = tl.sum(p0 * v_zero0, axis=0) + tl.sum(p1 * v_zero1, axis=0) + tl.sum(p2 * v_zero2, axis=0) + tl.sum(p3 * v_zero3, axis=0)
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

    partial_base = PartialOut + off_h * stride_poh + group_id * stride_pos
    tl.store(partial_base + offs_vp, acc0)
    tl.store(partial_base + 32 + offs_vp, acc1)
    tl.store(partial_base + 64 + offs_vp, acc2)
    tl.store(PartialM + off_h * stride_mh + group_id, m_i)
    tl.store(PartialL + off_h * stride_lh + group_id, l_i)


@triton.jit
def _bitdecode96_stage2_kernel(
    PartialOut,
    PartialM,
    PartialL,
    ContextLens,
    PrefixOut,
    PrefixLse,
    stride_poh,
    stride_pos,
    stride_mh,
    stride_lh,
    stride_oh,
    nheads,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    off_h = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_SPLITS)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    context_len = tl.load(ContextLens)
    num_splits = (context_len // 128)
    split_mask = offs_s < num_splits
    m_i = tl.load(PartialM + off_h * stride_mh + offs_s, mask=split_mask, other=-float("inf"))
    l_i = tl.load(PartialL + off_h * stride_lh + offs_s, mask=split_mask, other=0.0)
    m = tl.max(m_i, axis=0)
    weights = tl.where(split_mask & (l_i > 0.0), tl.exp2(m_i - m), 0.0)
    denom = tl.sum(weights * l_i, axis=0)
    partial_ptrs = PartialOut + off_h * stride_poh + offs_s[:, None] * stride_pos + offs_d[None, :]
    partial = tl.load(partial_ptrs, mask=split_mask[:, None] & (offs_d[None, :] < 96), other=0.0)
    out = tl.sum(weights[:, None] * partial, axis=0)
    out = tl.where(denom > 0.0, out / denom, 0.0)
    tl.store(PrefixOut + off_h * stride_oh + offs_d, out, mask=offs_d < 96)
    lse = tl.where(denom > 0.0, m + tl.log2(denom), -float("inf"))
    tl.store(PrefixLse + off_h, lse)


@triton.jit
def _residual_attention_kernel(
    Q,
    KResidual,
    VResidual,
    ContextLens,
    ResidualOut,
    ResidualLse,
    softmax_scale,
    stride_qh,
    stride_r_t,
    stride_r_h,
    stride_oh,
    BLOCK_T: tl.constexpr,
    BLOCK_D0: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_VPACK: tl.constexpr,
):
    off_h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    offs_d0 = tl.arange(0, BLOCK_D0)
    offs_d1 = tl.arange(0, BLOCK_D1)
    offs_vp = tl.arange(0, BLOCK_VPACK)
    context_len = tl.load(ContextLens)
    residual_len = context_len - (context_len // 128) * 128
    q0 = tl.load(Q + off_h * stride_qh + offs_d0).to(tl.float32)
    q1 = tl.load(Q + off_h * stride_qh + 64 + offs_d1).to(tl.float32)
    k0 = tl.load(KResidual + offs_t[:, None] * stride_r_t + off_h * stride_r_h + offs_d0[None, :], mask=offs_t[:, None] < residual_len, other=0.0).to(tl.float32)
    k1 = tl.load(KResidual + offs_t[:, None] * stride_r_t + off_h * stride_r_h + (64 + offs_d1)[None, :], mask=offs_t[:, None] < residual_len, other=0.0).to(tl.float32)
    softmax_scale_log2 = softmax_scale * 1.4426950408889634
    qk = (tl.sum(k0 * q0[None, :], axis=1) + tl.sum(k1 * q1[None, :], axis=1)) * softmax_scale_log2
    valid = offs_t < residual_len
    qk = tl.where(valid, qk, -float("inf"))
    m = tl.max(qk, axis=0)
    p = tl.where(valid, tl.exp2(qk - m), 0.0)
    denom = tl.sum(p, axis=0)
    v0 = tl.load(VResidual + offs_t[:, None] * stride_r_t + off_h * stride_r_h + offs_vp[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    v1 = tl.load(VResidual + offs_t[:, None] * stride_r_t + off_h * stride_r_h + (32 + offs_vp)[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    v2 = tl.load(VResidual + offs_t[:, None] * stride_r_t + off_h * stride_r_h + (64 + offs_vp)[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    out0 = tl.sum(p[:, None] * v0, axis=0)
    out1 = tl.sum(p[:, None] * v1, axis=0)
    out2 = tl.sum(p[:, None] * v2, axis=0)
    out0 = tl.where(denom > 0.0, out0 / denom, 0.0)
    out1 = tl.where(denom > 0.0, out1 / denom, 0.0)
    out2 = tl.where(denom > 0.0, out2 / denom, 0.0)
    out_base = ResidualOut + off_h * stride_oh
    tl.store(out_base + offs_vp, out0)
    tl.store(out_base + 32 + offs_vp, out1)
    tl.store(out_base + 64 + offs_vp, out2)
    lse = tl.where(denom > 0.0, m + tl.log2(denom), -float("inf"))
    tl.store(ResidualLse + off_h, lse)


@triton.jit
def _merge_prefix_residual_kernel(
    PrefixOut,
    PrefixLse,
    ResidualOut,
    ResidualLse,
    Out,
    stride_poh,
    stride_roh,
    stride_oh,
    BLOCK_D: tl.constexpr,
):
    off_h = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    lp = tl.load(PrefixLse + off_h)
    lr = tl.load(ResidualLse + off_h)
    m = tl.maximum(lp, lr)
    wp = tl.where(lp != -float("inf"), tl.exp2(lp - m), 0.0)
    wr = tl.where(lr != -float("inf"), tl.exp2(lr - m), 0.0)
    denom = wp + wr
    po = tl.load(PrefixOut + off_h * stride_poh + offs_d, mask=offs_d < 96, other=0.0).to(tl.float32)
    ro = tl.load(ResidualOut + off_h * stride_roh + offs_d, mask=offs_d < 96, other=0.0).to(tl.float32)
    out = tl.where(denom > 0.0, (po * wp + ro * wr) / denom, 0.0)
    tl.store(Out + off_h * stride_oh + offs_d, out, mask=offs_d < 96)


class BitdecodeKVCache96:
    def __init__(
        self,
        max_model_len: int,
        num_heads: int,
        dtype: torch.dtype,
        device: torch.device | str,
        block_size: int = 256,
    ):
        self.max_model_len = int(max_model_len)
        self.capacity = ((self.max_model_len + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
        self.max_groups = self.capacity // GROUP_SIZE
        self.block_size = int(block_size)
        self.num_heads = int(num_heads)
        self.dtype = dtype
        self.device = torch.device(device)
        h = self.num_heads
        g = self.max_groups
        self.k_pack = torch.empty((1, g * 32, h, HEAD_DIM), dtype=torch.uint16, device=self.device)
        self.k_params_h = torch.empty((1, g, h, K_PARAM_HALF_DIM), dtype=torch.float16, device=self.device)
        self.v_pack = torch.empty((1, g * GROUP_SIZE, h, V_PACK_DIM), dtype=torch.uint16, device=self.device)
        self.v_params_h = torch.empty((1, 1, h, g * 256), dtype=torch.float16, device=self.device)
        self.k_residual = torch.empty((GROUP_SIZE, h, HEAD_DIM), dtype=dtype, device=self.device)
        self.v_residual = torch.empty_like(self.k_residual)
        self.partial_o = torch.empty((1, h, g, HEAD_DIM), dtype=dtype, device=self.device)
        self.partial_m = torch.empty((1, h, g), dtype=torch.float32, device=self.device)
        self.partial_l = torch.empty((1, h, g), dtype=torch.float32, device=self.device)
        self.prefix_o = torch.empty((1, h, HEAD_DIM), dtype=dtype, device=self.device)
        self.residual_o = torch.empty((1, h, HEAD_DIM), dtype=dtype, device=self.device)
        self.out = torch.empty((1, 1, h, HEAD_DIM), dtype=dtype, device=self.device)
        self.prefix_lse = torch.empty((1, h), dtype=torch.float32, device=self.device)
        self.residual_lse = torch.empty((1, h), dtype=torch.float32, device=self.device)
        self._zero_i32 = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self.block_splits = triton.next_power_of_2(self.max_groups)

    def max_groups_for_blocks(self, num_blocks: int) -> int:
        block_capacity = int(num_blocks) * self.block_size
        groups = (block_capacity + GROUP_SIZE - 1) // GROUP_SIZE
        return max(1, min(self.max_groups, groups))

    def reset_from_prefill(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape[-2:] != (self.num_heads, HEAD_DIM) or v.shape[-2:] != (self.num_heads, HEAD_DIM):
            raise ValueError(f"BitdecodeKVCache96 expects [S,{self.num_heads},{HEAD_DIM}] K/V, got {tuple(k.shape)}")
        seqlen = int(k.shape[0])
        if seqlen > self.capacity:
            raise ValueError(f"prefill length {seqlen} exceeds bitdecode capacity {self.capacity}")
        full_groups = seqlen // GROUP_SIZE
        if full_groups > 0:
            _pack_k_prefill_kernel[(full_groups, self.num_heads, HEAD_DIM)](
                k,
                self.k_pack,
                self.k_params_h,
                k.stride(0),
                k.stride(1),
                self.k_pack.stride(1),
                self.k_pack.stride(2),
                self.k_pack.stride(3),
                self.k_params_h.stride(1),
                self.k_params_h.stride(2),
                self.k_params_h.stride(3),
                BLOCK_T=GROUP_SIZE,
                num_warps=4,
                num_stages=4,
            )
            _pack_v_prefill_kernel[(full_groups, self.num_heads, GROUP_SIZE)](
                v,
                self.v_pack,
                self.v_params_h,
                v.stride(0),
                v.stride(1),
                self.v_pack.stride(1),
                self.v_pack.stride(2),
                self.v_pack.stride(3),
                self.v_params_h.stride(2),
                self.v_params_h.stride(3),
                BLOCK_D=128,
                num_warps=4,
                num_stages=4,
            )
        tail = seqlen - full_groups * GROUP_SIZE
        if tail > 0:
            self.k_residual[:tail].copy_(k[full_groups * GROUP_SIZE : seqlen])
            self.v_residual[:tail].copy_(v[full_groups * GROUP_SIZE : seqlen])

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context_lens: torch.Tensor,
        softmax_scale: float,
        max_groups: int | None = None,
    ) -> torch.Tensor:
        if q.shape != (1, self.num_heads, HEAD_DIM):
            raise ValueError(f"decode q must be [1,{self.num_heads},{HEAD_DIM}], got {tuple(q.shape)}")
        if context_lens.numel() != 1:
            raise ValueError("BitdecodeKVCache96 only supports batch size 1")
        active_groups = self.max_groups if max_groups is None else max(1, min(self.max_groups, int(max_groups)))
        block_splits = triton.next_power_of_2(active_groups)
        _store_decode_residual_kernel[(self.num_heads,)](
            k,
            v,
            context_lens,
            self.k_residual,
            self.v_residual,
            k.stride(1),
            v.stride(1),
            self.k_residual.stride(0),
            self.k_residual.stride(1),
            BLOCK_D=128,
            num_warps=4,
            num_stages=4,
        )
        _pack_k_decode_if_full_kernel[(self.num_heads, HEAD_DIM)](
            context_lens,
            self.k_residual,
            self.k_pack,
            self.k_params_h,
            self.k_residual.stride(0),
            self.k_residual.stride(1),
            self.k_pack.stride(1),
            self.k_pack.stride(2),
            self.k_pack.stride(3),
            self.k_params_h.stride(1),
            self.k_params_h.stride(2),
            self.k_params_h.stride(3),
            BLOCK_T=GROUP_SIZE,
            num_warps=4,
            num_stages=4,
        )
        _pack_v_decode_if_full_kernel[(self.num_heads, GROUP_SIZE)](
            context_lens,
            self.v_residual,
            self.v_pack,
            self.v_params_h,
            self.v_residual.stride(0),
            self.v_residual.stride(1),
            self.v_pack.stride(1),
            self.v_pack.stride(2),
            self.v_pack.stride(3),
            self.v_params_h.stride(2),
            self.v_params_h.stride(3),
            BLOCK_D=128,
            num_warps=4,
            num_stages=4,
        )
        q4 = q.view(1, 1, self.num_heads, HEAD_DIM)
        _bitdecode96_stage1_kernel[(active_groups, self.num_heads)](
            q4,
            self.k_pack,
            self.k_params_h,
            self.v_pack,
            self.v_params_h,
            context_lens,
            self.partial_o,
            self.partial_m,
            self.partial_l,
            float(softmax_scale),
            q4.stride(0),
            q4.stride(2),
            self.k_pack.stride(0),
            self.k_pack.stride(2),
            self.k_pack.stride(1),
            self.k_pack.stride(3),
            self.k_params_h.stride(1),
            self.k_params_h.stride(2),
            self.k_params_h.stride(3),
            self.v_pack.stride(0),
            self.v_pack.stride(2),
            self.v_pack.stride(1),
            self.v_pack.stride(3),
            self.v_params_h.stride(2),
            self.v_params_h.stride(3),
            self.partial_o.stride(1),
            self.partial_o.stride(2),
            self.partial_m.stride(1),
            self.partial_l.stride(1),
            self.num_heads,
            BLOCK_D0=64,
            BLOCK_D1=32,
            BLOCK_TPACK=32,
            BLOCK_VPACK=32,
            num_warps=1,
            num_stages=4,
        )
        _bitdecode96_stage2_kernel[(self.num_heads,)](
            self.partial_o,
            self.partial_m,
            self.partial_l,
            context_lens,
            self.prefix_o,
            self.prefix_lse,
            self.partial_o.stride(1),
            self.partial_o.stride(2),
            self.partial_m.stride(1),
            self.partial_l.stride(1),
            self.prefix_o.stride(1),
            self.num_heads,
            BLOCK_SPLITS=block_splits,
            BLOCK_HEADDIM=128,
            num_warps=8,
            num_stages=1,
        )
        _residual_attention_kernel[(self.num_heads,)](
            q4,
            self.k_residual,
            self.v_residual,
            context_lens,
            self.residual_o,
            self.residual_lse,
            float(softmax_scale),
            q4.stride(2),
            self.k_residual.stride(0),
            self.k_residual.stride(1),
            self.residual_o.stride(1),
            BLOCK_T=GROUP_SIZE,
            BLOCK_D0=64,
            BLOCK_D1=32,
            BLOCK_VPACK=32,
            num_warps=4,
            num_stages=4,
        )
        _merge_prefix_residual_kernel[(self.num_heads,)](
            self.prefix_o,
            self.prefix_lse,
            self.residual_o,
            self.residual_lse,
            self.out,
            self.prefix_o.stride(1),
            self.residual_o.stride(1),
            self.out.stride(2),
            BLOCK_D=128,
            num_warps=4,
            num_stages=1,
        )
        return self.out
