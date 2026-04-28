# Copyright (c) 2025, Ming
# Test and benchmark file for cute Flash Attention 3 implementation

import os
import sys
import torch
import math
import time
import nvtx

# Add parent directories to path to allow importing cute modules directly
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)  # flash_attn/cute
flash_attn_dir = os.path.dirname(current_dir)  # flash_attn
flash_attention_dir = os.path.dirname(flash_attn_dir)  # flash-attention root

# Add flash-attention root to path
if flash_attention_dir not in sys.path:
    sys.path.insert(0, flash_attention_dir)

# Create a fake flash_attn module to prevent importing flash_attn/__init__.py
import types

try:
    from .cute_preprocess_V import preprocess_V
    from .cute_preprocess_QK import preprocess_QK
except:
    from cute_preprocess_V import preprocess_V
    from cute_preprocess_QK import preprocess_QK


class FakeFlashAttnModule(types.ModuleType):
    """Fake flash_attn module to avoid importing flash_attn_2_cuda"""
    def __init__(self):
        super().__init__("flash_attn")
        self.__path__ = [flash_attn_dir]
        self.__file__ = None

# Set up fake flash_attn module before any imports
if "flash_attn" not in sys.modules:
    sys.modules["flash_attn"] = FakeFlashAttnModule()

# Import cute modules
try:
    from .interface import flash_attn_func as cute_flash_attn_func
    from .testing import attention_ref
except:
    from .interface import flash_attn_func as cute_flash_attn_func
    from .testing import attention_ref

# Try to import hopper fa3
# First try to import from installed package, then from hopper directory
HOPPER_AVAILABLE = False
hopper_flash_attn_func = None

# Method 1: Try importing from installed package (if hopper was installed via setup.py)
try:
    import flash_attn_interface
    hopper_flash_attn_func = flash_attn_interface.flash_attn_func
    HOPPER_AVAILABLE = True
    print("✓ Hopper FA3 imported successfully from installed package")
except (ImportError, AttributeError):
    # Method 2: Try importing from hopper directory
    try:
        hopper_dir = os.path.join(flash_attention_dir, "hopper")
        print(f"Attempting to import hopper fa3 from: {hopper_dir}")
        if hopper_dir not in sys.path:
            sys.path.insert(0, hopper_dir)
        # Check if flash_attn_3._C is available (required by flash_attn_interface)
        try:
            import flash_attn_3._C
            print("✓ flash_attn_3._C module found")
        except ImportError:
            print("⚠ flash_attn_3._C module not found. Hopper FA3 needs to be installed:")
            print("  cd hopper && python setup.py install")
            raise ImportError("flash_attn_3._C not found. Please install hopper package first.")
        
        # Now import flash_attn_interface
        import flash_attn_interface
        hopper_flash_attn_func = flash_attn_interface.flash_attn_func
        HOPPER_AVAILABLE = True
        print("✓ Hopper FA3 imported successfully from hopper directory")
    except ImportError as e:
        print(f"⚠ Warning: hopper fa3 ImportError ({e}), skipping hopper comparison")
        HOPPER_AVAILABLE = False
        hopper_flash_attn_func = None
    except Exception as e:
        print(f"⚠ Warning: hopper fa3 import failed with exception ({e}), skipping hopper comparison")
        import traceback
        traceback.print_exc()
        HOPPER_AVAILABLE = False
        hopper_flash_attn_func = None


def benchmark_function(func, *args, warmup=10, repeats=100, **kwargs):
    """Benchmark a function"""
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        _ = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / repeats * 1000  # Convert to ms
    return avg_time

def torch_preprocess_v(x, int8: bool = True):

    # v = v0.to(torch.float8_e4m3fn).permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)

    chunk_size = 128

    b, s, h, d = x.shape

    range_max = 448.0
    if int8 == True:
        range_max = 127.0

    x2 = x.permute(0, 2, 3, 1).contiguous() # b h d s
    x2 = x2.view(b, h, d, s//chunk_size, chunk_size)

    x_amax = x2.abs().float().amax(dim=-1).clamp(1e-4)
    sf = x_amax / range_max
    # sf.fill_(1.0)

    x_new = x2.float() * (1.0 / sf.unsqueeze(-1))
    x_new = x_new.view(b, h, d, s)

    if int8 == True:
        x_new = torch.round(x_new).clamp(-range_max, range_max).to(torch.int8)
    else:
        x_new = x_new.to(torch.float8_e4m3fn)

    x_new = x_new.permute(0, 3, 1, 2)

    # b n//128 h d i.e. col major in ds gemm
    sf = sf.permute(0, 3, 1, 2).contiguous()

    return x_new, sf
    
    return v
   
def per_token_cast_to_fp8(x: torch.Tensor, fill_one: bool = False, int8: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    b, s, h, d = x.shape

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

    # b s h -> b h s -> b s h 1
    sf = sf.permute(0, 2, 1).contiguous().permute(0, 2, 1).unsqueeze(-1)

    return x_new, sf

def test_basic_attention(turn_on_int8=True):
    """Basic test for cute flash attention 3 - non-causal only (using small size for numerical validation)"""
    device = "cuda"
    dtype = torch.float16
    
    # Set random seed for reproducibility
    torch.random.manual_seed(42)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Test parameters: Use small size for numerical validation
    batch_size = 2
    seqlen = 16384 // 2
    nheads = 24
    head_dim = 128

    # seqlen_kv = (1370 + 127)//128 * 128
    seqlen_kv = 1370
    
    # print(f"Test configuration (small size for numerical validation):")
    # print(f"  Batch size: {batch_size}")
    # print(f"  Sequence length: {seqlen}")
    # print(f"  Number of heads: {nheads}")
    # print(f"  Head dimension: {head_dim}")
    # print(f"  Input shape: [{batch_size}, {seqlen}, {nheads * head_dim}] = [{batch_size}, {seqlen}, 3072]")
    
    # Create random Q, K, V tensors with shape [bsz, seqlen, nheads * head_dim]
    # Then reshape to [bsz, seqlen, nheads, head_dim] for flash attention
    q_flat = torch.randn(batch_size, seqlen, nheads * head_dim, device=device, dtype=dtype)
    k_flat = torch.randn(batch_size, seqlen_kv, nheads * head_dim, device=device, dtype=dtype)
    v_flat = torch.randn(batch_size, seqlen_kv, nheads * head_dim, device=device, dtype=dtype)
    # v_flat = torch.ones(batch_size, seqlen, nheads * head_dim, device=device, dtype=dtype) 

    def add_outlier(X, scale=50):
        mask = torch.rand_like(X) < 0.01  # 1% 的离群值
        X[mask] *= scale  # 制造 Outliers
        return X
    
    # q_flat = add_outlier(q_flat, scale=50)
    # k_flat = add_outlier(k_flat, scale=50)
    # v_flat = add_outlier(v_flat, scale=50)
    
    # Reshape to [bsz, seqlen, nheads, head_dim]
    q = q_flat.view(batch_size, seqlen, nheads, head_dim)
    k = k_flat.view(batch_size, seqlen_kv, nheads, head_dim)
    v = v_flat.view(batch_size, seqlen_kv, nheads, head_dim)
    
    # vt_fp8, v_scale = torch_preprocess_v(v, int8=turn_on_int8)
    # q_fp8, q_scale = per_token_cast_to_fp8(q, fill_one=False, int8=turn_on_int8)
    # k_fp8, k_scale = per_token_cast_to_fp8(k, fill_one=False, int8=turn_on_int8)

    # vt_fp8, v_scale = preprocess_V(v, int8=turn_on_int8)
    # q_fp8, q_scale = preprocess_QK(q, int8=turn_on_int8, chunk_size=128)
    # k_fp8, k_scale = preprocess_QK(k, int8=turn_on_int8, chunk_size=128)

    # for i in range(128):
    #     print(f"v_scale[{i}] = {v_scale[0,0,0,i].item()}")
    # exit(0)

    out_cute, lse = cute_flash_attn_func(
        q, k, v,
        turn_on_int8=turn_on_int8,
        causal=False,
        softmax_scale=None,
    )
    # exit(0)
    
    # Reference implementation
    # out_ref0, attn_ref = attention_ref(
    #     q, k, v,
    #     query_padding_mask=None,
    #     key_padding_mask=None,
    #     causal=False,
    # )

    out_ref = torch.nn.functional.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=False).permute(0, 2, 1, 3)
    
    # Check output shape
    expected_shape = (batch_size, seqlen, nheads, head_dim)
    assert out_cute.shape == expected_shape, \
        f"Shape mismatch: expected {expected_shape}, got {out_cute.shape}"
    assert out_cute.shape == out_ref.shape, \
        f"Shape mismatch with reference: {out_cute.shape} vs {out_ref.shape}"
    
    # Check numerical accuracy
    max_diff = (out_cute - out_ref).abs().max().item()
    mean_diff = (out_cute - out_ref).abs().mean().item()

    # find maximum 10 difference elements
    max_diff_elements = (out_cute.flatten() - out_ref.flatten()).abs().topk(5).indices
    print("max 5 difference elements:")
    print("--------------------------------")
    for i in max_diff_elements:
        print("cute ", out_cute.flatten()[i].item(), "ref ", out_ref.flatten()[i].item())
    print("--------------------------------")

    print("out_cute: ", out_cute.shape, "out_ref: ", out_ref.shape)
    # from deep_gemm.testing import calc_diff
    # diff = calc_diff(out_cute.to(torch.float32).reshape(batch_size * seqlen, nheads*head_dim), out_ref.to(torch.float32).reshape(batch_size * seqlen, nheads*head_dim))
    # print("diff: ", diff)
    
    # for i in range(100):
    #     print("cute ", out_cute.flatten()[i].item(), "ref ", out_ref.flatten()[i].item())

    def calc_qsnr(x, x_ref):
        qsnr = -10 * torch.log10(torch.mean((x.float() - x_ref.float())**2) / (torch.mean(x_ref.float()**2)))
        return qsnr

    def calc_diff(x, y):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        sim = 2 * (x * y).sum() / denominator
        return 1 - sim

    calc_diff = calc_diff(out_cute, out_ref)
    calc_qsnr = calc_qsnr(out_cute, out_ref)
    print("Error metrics:")
    print("--------------------------------")
    print("Calc diff: ", calc_diff.item())
    print("Calc qsnr: ", calc_qsnr.item())
    print("Max difference: ", max_diff)
    print("Mean difference: ", mean_diff)
    print("--------------------------------")


def benchmark_comparison(turn_on_int8=True):
    """Compare performance between hopper fa3 and cute implementation"""
    # Clear compile cache to ensure grid_dim is recalculated for new tensor sizes
    try:
        from .interface import interface
    except:
        from .interface import interface
    if hasattr(interface, '_flash_attn_fwd'):
        interface._flash_attn_fwd.compile_cache.clear()
        print("Cleared compile cache for benchmark")
    
    # print(f"\nHOPPER_AVAILABLE = {HOPPER_AVAILABLE}")
    # print(f"hopper_flash_attn_func = {hopper_flash_attn_func}")
    if not HOPPER_AVAILABLE:
        print("\n⚠ Skipping benchmark comparison (hopper fa3 not available)")
        print("  This could be due to:")
        print("  1. flash_attn_interface module not found")
        print("  2. flash_attn_func not found in flash_attn_interface")
        print("  3. Import error during hopper module loading")
        return
    
    device = "cuda"
    dtype = torch.float16
    
    print("\n" + "=" * 60)
    print("Performance Comparison: Hopper FA3 vs Cute FA3")
    print("=" * 60)
    
    # Test configuration: [bsz, 16384, 3072] where 3072 = 24 * 128
    configs = [
        {"batch": 2, "seqlen": 16384, "nheads": 24, "head_dim": 128},
    ]
    
    results = []
    
    for config in configs:
        batch_size = config["batch"]
        seqlen = config["seqlen"]
        nheads = config["nheads"]
        head_dim = config["head_dim"]
        
        print(f"\nConfig: batch={batch_size}, seqlen={seqlen}, nheads={nheads}, head_dim={head_dim}")
        print(f"  Input shape: [{batch_size}, {seqlen}, {nheads * head_dim}] = [{batch_size}, {seqlen}, 3072]")

        seqlen_kv = seqlen + 1370
        
        # Create tensors: [bsz, seqlen, 3072] then reshape to [bsz, seqlen, nheads, head_dim]
        q_flat = torch.randn(batch_size, seqlen, nheads * head_dim, device=device, dtype=dtype)
        k_flat = torch.randn(batch_size, seqlen_kv, nheads * head_dim, device=device, dtype=dtype)
        v_flat = torch.randn(batch_size, seqlen_kv, nheads * head_dim, device=device, dtype=dtype)
        
        # Reshape to [bsz, seqlen, nheads, head_dim] for flash attention
        q = q_flat.view(batch_size, seqlen, nheads, head_dim)
        k = k_flat.view(batch_size, seqlen_kv, nheads, head_dim)
        v = v_flat.view(batch_size, seqlen_kv, nheads, head_dim)

        # vt_fp8, v_scale = torch_preprocess_v(v, int8=turn_on_int8)
        # q_fp8, q_scale = per_token_cast_to_fp8(q, fill_one=False, int8=turn_on_int8)
        # k_fp8, k_scale = per_token_cast_to_fp8(k, fill_one=False, int8=turn_on_int8)

        vt_fp8, v_scale = preprocess_V(v, int8=turn_on_int8)
        q_fp8, q_scale = preprocess_QK(q, int8=turn_on_int8, chunk_size=128)
        k_fp8, k_scale = preprocess_QK(k, int8=turn_on_int8, chunk_size=128)
        
        # Benchmark cute (non-causal only)
        try:
            with nvtx.annotate("cute_flash_attn_func"):
                cute_time = benchmark_function(
                    cute_flash_attn_func,
                    q, k, v,
                    turn_on_int8=turn_on_int8,
                    causal=False,
                    softmax_scale=None,
                )
                print(f"  Cute FA3:     {cute_time:.3f} ms")
        except Exception as e:
            print(f"  Cute FA3:     Failed ({e})")
            cute_time = None
        
        # Benchmark hopper (non-causal only)
        try:
            hopper_time = benchmark_function(
                hopper_flash_attn_func,
                q, k, v,
                softmax_scale=None,
                causal=False,
            )
            print(f"  Hopper FA3:   {hopper_time:.3f} ms")
        except Exception as e:
            print(f"  Hopper FA3:   Failed ({e})")
            hopper_time = None
        
        # Calculate speedup
        if cute_time is not None and hopper_time is not None:
            speedup = hopper_time / cute_time
            print(f"  Speedup:      {speedup:.2f}x ({'Cute faster' if speedup > 1 else 'Hopper faster'})")
            results.append({
                "config": config,
                "cute_time": cute_time,
                "hopper_time": hopper_time,
                "speedup": speedup,
            })
        elif cute_time is not None:
            print(f"  Speedup:      N/A (Hopper failed)")
        elif hopper_time is not None:
            print(f"  Speedup:      N/A (Cute failed)")
    
    # Summary
    # if results:
    #     print("\n" + "=" * 60)
    #     print("Summary:")
    #     print("=" * 60)
    #     avg_speedup = sum(r["speedup"] for r in results) / len(results)
    #     print(f"Average speedup: {avg_speedup:.2f}x")
    #     if avg_speedup > 1:
    #         print("Cute FA3 is faster on average")
    #     else:
    #         print("Hopper FA3 is faster on average")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing and Benchmarking cute Flash Attention 3")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8", type=int, default=1)
    args = parser.parse_args()
    turn_on_int8 = args.int8 != 0

    try:
        # Run correctness test
        test_basic_attention(turn_on_int8=turn_on_int8)
        # exit(0)
        
        # Run performance comparison
        benchmark_comparison(turn_on_int8=turn_on_int8)
        
        # print("\n" + "=" * 60)
        # print("All tests completed!")
        # print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
