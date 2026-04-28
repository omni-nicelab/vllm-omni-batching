# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""
FP8量化方法 - 方案3：同时优化scale和bias（指派步+闭式回归）

提供简单的外部接口，输入两个矩阵A和B，返回FP8量化后的GEMM结果。
这是三种方案中最精确的方法，通过联合优化scale和bias来最小化量化误差。
"""

import torch
from typing import Tuple

try:
    from .cute_per_token_cast_shift import ref_helper
except:
    from cute_per_token_cast_shift import ref_helper

def fp8_roundtrip_e4m3nv(y: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3fn 量化-反量化（输入已在"448空间"）"""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("需要 PyTorch>=2.1 且支持 torch.float8_e4m3fn。")
    return y.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)


@torch.no_grad()
def optimize_fp8_scale_and_bias_rowwise(
    X: torch.Tensor, 
    max_iter: int = 200, 
    tol: float = 1e-6, 
    lam: float = 1e-12, 
    s0=None, 
    b0: float = 0.0,
    use_absmax_init: bool = True, 
    use_minmax_init: bool = False, 
    use_adaptive_minmax_init: bool = False, 
    debug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    同时优化每行的 scale 和 bias，使得 FP8 量化后的重构误差最小
    
    Args:
        X: [R, C] 原始数据
        max_iter: 最大迭代次数
        tol: 收敛容差
        lam: 正则化参数（防止除零）
        s0: 手动指定的初始scale
        b0: 手动指定的初始bias
        use_absmax_init: 是否使用absmax初始化（默认True）
        use_minmax_init: 是否使用min-max初始化
        use_adaptive_minmax_init: 是否使用自适应min-max初始化
        debug: 是否输出调试信息
        
    Returns:
        s: [R] 每行的 scale
        b: [R] 每行的 bias（在"448空间"）
        loss_row: [R] 每行最终的 MSE（对列平均）
    
    使用指派步 + 闭式回归步同时优化 s 和 b（向量化实现，所有行并行处理）
    """
    R, C = X.shape
    device = X.device
    dtype = torch.float32
    n = C
    
    # 初始化 scale 和 bias
    if s0 is None and not use_minmax_init and not use_adaptive_minmax_init:
        # 原有的初始化方式
        if use_absmax_init:
            # 使用 absmax 初始化（与方案1相同）
            s = (X.abs().amax(dim=1) / 448.0).clamp_min(1e-12).to(dtype)
        else:
            # 使用 99.5% 分位数初始化
            quantile_vals = torch.quantile(X.abs(), 0.995, dim=1, keepdim=False).clamp_min(1e-8)
            s = (quantile_vals / 448.0).to(dtype)
        # 初始化 bias
        b = torch.full((R,), float(b0), device=device, dtype=dtype)
    elif use_adaptive_minmax_init:
        # 自适应min-max初始化：检测outlier，有则用absmax，无则用min-max
        k_iqr = 3.0  # 使用更保守的阈值
        eps = 1e-8
        
        if C >= 10:
            # 对每行排序
            X_sorted = torch.sort(X, dim=1)[0]  # [R, C]
            
            # 计算四分位数索引
            q1_idx = C // 4
            q3_idx = C * 3 // 4
            q1 = X_sorted[:, q1_idx]  # [R]
            q3 = X_sorted[:, q3_idx]  # [R]
            iqr = (q3 - q1).clamp_min(eps)  # [R]
            
            # 检查最小值和最大值是否在合理范围内
            x_min = X_sorted[:, 0]
            x_max = X_sorted[:, -1]
            max_outlier = (x_max - q3) > (k_iqr * iqr)
            min_outlier = (q1 - x_min) > (k_iqr * iqr)
            has_outlier = max_outlier | min_outlier
        else:
            # 数据点太少，不检测outlier
            has_outlier = torch.zeros(R, dtype=torch.bool, device=device)
        
        # 计算min-max的scale和bias
        x_min = X.min(dim=1, keepdim=False)[0]
        x_max = X.max(dim=1, keepdim=False)[0]
        s_minmax = ((x_max - x_min) / 896.0).clamp_min(1e-12).to(dtype)
        b_minmax = (448.0 * (x_max + x_min) / (x_max - x_min).clamp_min(1e-12)).to(dtype)
        
        # 计算absmax的scale和bias
        x_absmax = X.abs().amax(dim=1)
        s_absmax = (x_absmax / 448.0).clamp_min(1e-12).to(dtype)
        b_absmax = torch.zeros(R, device=device, dtype=dtype)
        
        # 根据outlier检测结果选择
        s = torch.where(has_outlier, s_absmax, s_minmax)
        b = torch.where(has_outlier, b_absmax, b_minmax)
        
        if debug:
            outlier_pct = has_outlier.float().mean().item() * 100
            print(f"  自适应初始化 (IQR方法, k={k_iqr}):")
            print(f"    - {outlier_pct:.1f}%的行检测到离群值，使用absmax")
            print(f"    - {100-outlier_pct:.1f}%的行无离群值，使用min-max")
    elif use_minmax_init:
        # 使用 min-max 初始化：让 min 映射到 -448，max 映射到 448
        x_min = X.min(dim=1, keepdim=False)[0]  # [R]
        x_max = X.max(dim=1, keepdim=False)[0]  # [R]
        x_range = (x_max - x_min).clamp_min(1e-12)
        s = (x_range / 896.0).to(dtype)
        b = (448.0 * (x_max + x_min) / x_range).to(dtype)
        # 处理常数行
        const_mask = (x_max - x_min) < 1e-12
        b = torch.where(const_mask, torch.zeros_like(b), b)
    else:
        # 手动指定 s0
        s = torch.full((R,), float(s0), device=device, dtype=dtype)
        b = torch.full((R,), float(b0), device=device, dtype=dtype)
    
    # 保存初始 scale 用于调试
    s_init = s.clone()
    
    # 预计算 X 的和（每行）
    X_sum = X.sum(dim=1)  # [R]
    
    # 迭代优化（所有行并行）
    converged_iter = max_iter
    prev_obj = float('inf')
    obj_increased_count = 0
    
    for iter_idx in range(max_iter):
        # 指派步：y = x / s - b，[R, C]
        y = X / s[:, None] - b[:, None]
        q = fp8_roundtrip_e4m3nv(y)  # [R, C]
        
        # 计算当前目标函数值
        x_recon_current = s[:, None] * (q + b[:, None])
        obj_current = ((X - x_recon_current)**2).mean().item()
        
        # 统计量（每行）
        A = (q * q).sum(dim=1)      # [R]
        C_stat = q.sum(dim=1)       # [R]
        D = (q * X).sum(dim=1)      # [R]
        
        # 闭式更新
        denom = (n * A - C_stat * C_stat) + lam  # [R]
        s_new = (n * D - C_stat * X_sum) / denom  # [R]
        t_new = (A * X_sum - C_stat * D) / denom  # [R]
        
        # 保障：检查 s_new 是否有效
        valid_mask = torch.isfinite(s_new) & (s_new > 0)  # [R]
        
        # 对于无效的 s_new，退化为只更新 b（固定 s）
        b_fallback = (X_sum - s * C_stat) / (n * s)  # [R]
        s_new = torch.where(valid_mask, s_new, s)
        b_new = torch.where(valid_mask, t_new / s_new, b_fallback)
        
        # 检查目标函数是否增加
        if obj_current > prev_obj + 1e-10:
            obj_increased_count += 1
            if debug and obj_increased_count <= 3:
                print(f"警告: 第{iter_idx}次迭代目标函数增加: {prev_obj:.6e} -> {obj_current:.6e}")
        
        prev_obj = obj_current
        
        # 收敛判据
        s_diff = (s_new - s).abs().max().item()
        b_diff = (b_new - b).abs().max().item()
        
        s, b = s_new, b_new
        
        if s_diff < tol and b_diff < tol:
            converged_iter = iter_idx + 1
            break
    
    # 计算最终损失（向量化）
    y_final = X / s[:, None] - b[:, None]
    q_final = fp8_roundtrip_e4m3nv(y_final)
    x_recon = s[:, None] * (q_final + b[:, None])
    final_loss = ((X - x_recon)**2).mean(dim=1)  # [R]
    
    # 计算 absmax + bias=0 的基线损失（用于安全检查）
    absmax_scale = (X.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
    y_baseline = X / absmax_scale[:, None]
    q_baseline = fp8_roundtrip_e4m3nv(y_baseline)
    x_recon_baseline = absmax_scale[:, None] * q_baseline
    baseline_loss = ((X - x_recon_baseline)**2).mean(dim=1)
    
    # 安全保障：如果优化结果比基线更差，回退到基线
    worse_mask = final_loss > baseline_loss * 1.001
    if worse_mask.any():
        rollback_count = worse_mask.sum().item()
        if debug:
            print(f"回退 {rollback_count}/{R} 行到 absmax 基线")
        s = torch.where(worse_mask, absmax_scale, s)
        b = torch.where(worse_mask, torch.zeros_like(b), b)
        # 重新计算这些行的损失
        y_final = X / s[:, None] - b[:, None]
        q_final = fp8_roundtrip_e4m3nv(y_final)
        x_recon = s[:, None] * (q_final + b[:, None])
        final_loss = ((X - x_recon)**2).mean(dim=1)
    
    # 调试输出
    if debug:
        print(f"\n=== Scale+Bias 优化调试信息 ===")
        print(f"收敛迭代次数: {converged_iter}/{max_iter}")
        print(f"目标函数增加次数: {obj_increased_count}")
        print(f"Scale 变化: init={s_init.mean().item():.6f}, final={s.mean().item():.6f}")
        print(f"Scale vs AbsMax: min={((s/absmax_scale).min().item()):.4f}, "
              f"mean={((s/absmax_scale).mean().item()):.4f}, "
              f"max={((s/absmax_scale).max().item()):.4f}")
        print(f"Bias 统计: min={b.min().item():.4f}, mean={b.mean().item():.4f}, max={b.max().item():.4f}")
        print(f"Final loss vs Baseline: {final_loss.mean().item():.6e} vs {baseline_loss.mean().item():.6e}")
        
        final_worse_count = (final_loss > baseline_loss).sum().item()
        if final_worse_count > 0:
            print(f"异常: 回退后仍有 {final_worse_count}/{R} 行比基线更差（数值误差）")
    
    return s, b, final_loss


def codes_with_scale_and_bias_chunked(
    X: torch.Tensor, 
    chunk_size: int, 
    use_absmax_init: bool = True, 
    use_minmax_init: bool = False, 
    use_adaptive_minmax_init: bool = False, 
    debug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    同时优化scale和bias对矩阵进行FP8量化（分块版本）
    
    Args:
        X: 输入张量 [R, K]
        chunk_size: 每个chunk的大小
        use_absmax_init: 是否使用absmax初始化（默认True）
        use_minmax_init: 是否使用min-max初始化
        use_adaptive_minmax_init: 是否使用自适应min-max初始化
        debug: 是否输出调试信息
        
    Returns:
        Qb: 量化后的张量 [R, K]（码域重构值）
        s: 优化后的scale张量 [R, num_chunks]
        b: 优化后的bias张量 [R, num_chunks]
    """
    R, K = X.shape
    num_chunks = K // chunk_size
    
    # 重塑
    X_chunked = X.view(R, num_chunks, chunk_size)  # [R, num_chunks, chunk_size]
    
    # 对每个 chunk 独立优化 scale 和 bias
    # 将其重塑为 [R*num_chunks, chunk_size] 来批量处理
    X_flat = X_chunked.reshape(R * num_chunks, chunk_size)
    s_flat, b_flat, _ = optimize_fp8_scale_and_bias_rowwise(
        X_flat, use_absmax_init=use_absmax_init, use_minmax_init=use_minmax_init, 
        use_adaptive_minmax_init=use_adaptive_minmax_init, debug=debug
    )  # [R*num_chunks]
    s = s_flat.view(R, num_chunks)  # [R, num_chunks]
    b = b_flat.view(R, num_chunks)  # [R, num_chunks]
    
    # 量化：y = x / s - b，然后 Q(y)（向量化）
    y_chunked = X_chunked / s[:, :, None] - b[:, :, None]
    Qb_chunked = fp8_roundtrip_e4m3nv(y_chunked)  # [R, num_chunks, chunk_size]
    
    Qb = Qb_chunked.reshape(R, K)  # [R, K]
    
    return Qb, s, b


@torch.no_grad()
def fp8_gemm_opt_scale_opt_bias(
    A: torch.Tensor, 
    B: torch.Tensor, 
    chunk_size: int = 32,
    use_absmax_init: bool = True,
    use_minmax_init: bool = False,
    use_adaptive_minmax_init: bool = False,
    debug: bool = False
) -> torch.Tensor:
    """
    使用同时优化scale和bias的FP8量化方法进行矩阵乘法：C = A @ B^T
    
    这是方案3的外部接口函数，通过联合优化scale和bias来最小化量化误差。
    这是三种方案中最精确的方法，但计算开销也最大。
    
    Args:
        A: 第一个输入矩阵 [m, k]
        B: 第二个输入矩阵 [n, k]
        chunk_size: 每个chunk的大小（默认32，必须能整除k）
        use_absmax_init: 是否使用absmax初始化（默认True）
        use_minmax_init: 是否使用min-max初始化
        use_adaptive_minmax_init: 是否使用自适应min-max初始化（推荐）
        debug: 是否输出调试信息
        
    Returns:
        C: 矩阵乘法结果 [m, n]，C = A @ B^T
        
    示例:
        >>> A = torch.randn(128, 1024, device='cuda')
        >>> B = torch.randn(128, 1024, device='cuda')
        >>> C = fp8_gemm_opt_scale_opt_bias(A, B, chunk_size=32)
        >>> print(C.shape)  # torch.Size([128, 128])
        
        # 使用自适应min-max初始化（推荐）
        >>> C = fp8_gemm_opt_scale_opt_bias(A, B, chunk_size=32, 
        ...                                  use_absmax_init=False,
        ...                                  use_adaptive_minmax_init=True)
    """
    m, k = A.shape
    n, k_B = B.shape
    assert k == k_B, f"A和B的列数必须相同，但得到A: {A.shape}, B: {B.shape}"
    assert k % chunk_size == 0, f"k={k} 必须是 chunk_size={chunk_size} 的倍数"
    
    device = A.device
    num_chunks = k // chunk_size
    
    # 量化A和B（同时优化scale和bias）
    if debug:
        print("\n=== 优化 A 矩阵 ===")
    QA_sb, sA_opt, bA_opt = codes_with_scale_and_bias_chunked(
        A, chunk_size, use_absmax_init=use_absmax_init, use_minmax_init=use_minmax_init, 
        use_adaptive_minmax_init=use_adaptive_minmax_init, debug=debug
    )  # [m, k], [m, num_chunks], [m, num_chunks]
    
    if debug:
        print("\n=== 优化 B 矩阵 ===")
    QB_sb, sB_opt, bB_opt = codes_with_scale_and_bias_chunked(
        B, chunk_size, use_absmax_init=use_absmax_init, use_minmax_init=use_minmax_init, 
        use_adaptive_minmax_init=use_adaptive_minmax_init, debug=debug
    )  # [n, k], [n, num_chunks], [n, num_chunks]
    
    # 按 chunk 计算修复后的结果（使用优化的 scale 和 bias）
    C = torch.zeros(m, n, device=device, dtype=torch.float32)
    ones_chunk = torch.ones(chunk_size, device=device, dtype=torch.float32)
    
    for c in range(num_chunks):
        chunk_start = c * chunk_size
        chunk_end = (c + 1) * chunk_size
        
        QA_chunk = QA_sb[:, chunk_start:chunk_end]  # [m, chunk_size]
        QB_chunk = QB_sb[:, chunk_start:chunk_end]  # [n, chunk_size]
        bA_c = bA_opt[:, c]  # [m]
        bB_c = bB_opt[:, c]  # [n]
        sA_c = sA_opt[:, c]  # [m] 使用优化后的 scale
        sB_c = sB_opt[:, c]  # [n] 使用优化后的 scale
        
        # 码域乘积
        G0_chunk = QA_chunk @ QB_chunk.T  # [m, n]
        
        # 修复项
        rA_chunk = QA_chunk @ ones_chunk  # [m]
        rB_chunk = QB_chunk @ ones_chunk  # [n]
        
        # G_chunk = QA @ QB^T + (QA @ 1) @ bB^T + bA @ (QB @ 1)^T + chunk_size * bA @ bB^T
        G_chunk_corr = (
            G0_chunk
            + rA_chunk[:, None] @ bB_c[None, :]
            + bA_c[:, None] @ rB_chunk[None, :]
            + (chunk_size * (bA_c[:, None] @ bB_c[None, :]))
        )
        
        # 应用优化后的 scale
        C += (sA_c[:, None] * G_chunk_corr) * sB_c[None, :]
    
    return C

def per_token_cast_to_fp8_complex_shift(x: torch.Tensor, option_init: int = 0) -> tuple[torch.Tensor, torch.Tensor, ]:

    if option_init == 0:
        use_absmax_init = True
        use_minmax_init = False
        use_adaptive_minmax_init = False
    elif option_init == 1:
        use_absmax_init = False
        use_minmax_init = True
        use_adaptive_minmax_init = False
    else:
        use_absmax_init = False
        use_minmax_init = False
        use_adaptive_minmax_init = True

    fp16_x, scale, shift  = codes_with_scale_and_bias_chunked(x, chunk_size=32, use_absmax_init=use_absmax_init, use_minmax_init=use_minmax_init, use_adaptive_minmax_init=use_adaptive_minmax_init)
    shift = shift * scale
    # fp16_x, scale, shift, sum = ref_helper(x.to(torch.float16), chunk=32, clear_shift=True)
    x_reshaped = x.reshape(x.shape[0], x.shape[1] // 32, 32)
    sum = x_reshaped.sum(dim=2)

    shift_fp8, shift_scale, _, _ = ref_helper(shift.to(torch.float16), chunk=32, clear_shift=True)
    sum_fp8, sum_scale, _, _ = ref_helper(sum.to(torch.float16), chunk=32, clear_shift=True)

    final_fp8 = torch.cat([fp16_x.to(torch.float8_e4m3fn), sum_fp8, shift_fp8], dim=1)
    final_scale = torch.cat([scale, sum_scale, shift_scale], dim=1)
    return final_fp8, final_scale

if __name__ == "__main__":
    # 对比不同初始化方法的测试
    print("=" * 60)
    print("测试 1024x1024 矩阵 - 对比不同的初始化方法")
    print("=" * 60)
    
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    
    # 生成测试矩阵
    A = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    B = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)

    # A[:, 1] = 1000
    # B[:, 10] = 1000
    C_ref = (A @ B.T).to(torch.float32)
    
    chunk_size = 32
    
    # 测试1: absmax 初始化
    print("\n1. AbsMax 初始化:")
    C_absmax = fp8_gemm_opt_scale_opt_bias(
        A, B, 
        chunk_size=chunk_size,
        use_absmax_init=True,
        use_minmax_init=False,
        use_adaptive_minmax_init=False
    )
    rel_err_absmax = (torch.linalg.norm(C_absmax - C_ref) / torch.linalg.norm(C_ref)).item()
    print(f"   相对误差: {rel_err_absmax:.6e}")
    
    # 测试2: min-max 初始化
    print("\n2. Min-Max 初始化:")
    C_minmax = fp8_gemm_opt_scale_opt_bias(
        A, B, 
        chunk_size=chunk_size,
        use_absmax_init=False,
        use_minmax_init=True,
        use_adaptive_minmax_init=False
    )
    rel_err_minmax = (torch.linalg.norm(C_minmax - C_ref) / torch.linalg.norm(C_ref)).item()
    print(f"   相对误差: {rel_err_minmax:.6e}")
    
    # 测试3: 自适应 min-max 初始化
    print("\n3. 自适应 Min-Max 初始化:")
    C_adaptive = fp8_gemm_opt_scale_opt_bias(
        A, B, 
        chunk_size=chunk_size,
        use_absmax_init=False,
        use_minmax_init=False,
        use_adaptive_minmax_init=True
    )
    rel_err_adaptive = (torch.linalg.norm(C_adaptive - C_ref) / torch.linalg.norm(C_ref)).item()
    print(f"   相对误差: {rel_err_adaptive:.6e}")
    
    # 汇总对比
    print("\n" + "=" * 60)
    print("汇总对比:")
    print("=" * 60)
    print(f"AbsMax 初始化:           {rel_err_absmax:.6e}")
    print(f"Min-Max 初始化:          {rel_err_minmax:.6e}")
    print(f"自适应 Min-Max 初始化:   {rel_err_adaptive:.6e}")
    
    # 找出最佳方法
    errors = {
        'AbsMax': rel_err_absmax,
        'Min-Max': rel_err_minmax,
        'Adaptive Min-Max': rel_err_adaptive
    }
    best_method = min(errors, key=errors.get)
    print(f"\n最佳方法: {best_method} (相对误差: {errors[best_method]:.6e})")
    print("=" * 60)