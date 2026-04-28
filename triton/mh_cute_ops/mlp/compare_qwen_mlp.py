from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F


THIS_FILE = pathlib.Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mh_cute_ops.mlp.cute_infer_mlp import CuteInferMLP


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


def tensor_stats(x: torch.Tensor) -> str:
    x_float = x.detach().float()
    finite_ratio = torch.isfinite(x_float).float().mean().item()
    return (
        f"shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={x_float.min().item():.6g} max={x_float.max().item():.6g} "
        f"mean={x_float.mean().item():.6g} std={x_float.std(unbiased=False).item():.6g} "
        f"finite={finite_ratio:.4f}"
    )


@dataclass(slots=True)
class ErrorMetrics:
    max_abs: float
    mean_abs: float
    rmse: float
    rel_l2: float
    qsnr_db: float
    p50_abs: float
    p90_abs: float
    p99_abs: float
    p999_abs: float


def compute_error_metrics(
    pred: torch.Tensor,
    ref: torch.Tensor,
) -> ErrorMetrics:
    pred_float = pred.detach().float()
    ref_float = ref.detach().float()
    diff = (pred_float - ref_float).abs()
    diff_flat = diff.reshape(-1)

    mse = torch.mean((pred_float - ref_float) ** 2).item()
    ref_power = torch.mean(ref_float ** 2).item()
    rel_l2 = (torch.linalg.vector_norm(pred_float - ref_float) / torch.linalg.vector_norm(ref_float)).item()
    if mse == 0.0:
        qsnr_db = math.inf
    elif ref_power == 0.0:
        qsnr_db = -math.inf
    else:
        qsnr_db = -10.0 * math.log10(mse / ref_power)

    quantiles = torch.quantile(
        diff_flat,
        torch.tensor([0.5, 0.9, 0.99, 0.999], device=diff_flat.device),
    )
    return ErrorMetrics(
        max_abs=diff.max().item(),
        mean_abs=diff.mean().item(),
        rmse=math.sqrt(mse),
        rel_l2=rel_l2,
        qsnr_db=qsnr_db,
        p50_abs=quantiles[0].item(),
        p90_abs=quantiles[1].item(),
        p99_abs=quantiles[2].item(),
        p999_abs=quantiles[3].item(),
    )


def print_error_metrics(name: str, pred: torch.Tensor, ref: torch.Tensor) -> None:
    metrics = compute_error_metrics(pred, ref)
    print(
        f"[diff] {name:<26} "
        f"max={metrics.max_abs:.6g} mean={metrics.mean_abs:.6g} rmse={metrics.rmse:.6g} "
        f"rel_l2={metrics.rel_l2:.6g} qsnr={metrics.qsnr_db:.4f}dB "
        f"p50={metrics.p50_abs:.6g} p90={metrics.p90_abs:.6g} "
        f"p99={metrics.p99_abs:.6g} p99.9={metrics.p999_abs:.6g}"
    )


def topk_error_report(
    name: str,
    pred: torch.Tensor,
    ref: torch.Tensor,
    topk: int,
) -> None:
    if topk <= 0:
        return
    diff = (pred.detach().float() - ref.detach().float()).abs().reshape(-1)
    pred_flat = pred.detach().float().reshape(-1)
    ref_flat = ref.detach().float().reshape(-1)
    count = min(topk, diff.numel())
    values, indices = torch.topk(diff, k=count)
    print(f"[topk] {name} worst {count} elements:")
    for rank in range(count):
        flat_idx = indices[rank].item()
        print(
            f"  idx={flat_idx:<10d} "
            f"abs_diff={values[rank].item():.6g} "
            f"pred={pred_flat[flat_idx].item():.6g} "
            f"ref={ref_flat[flat_idx].item():.6g}"
        )


def run_torch_mlp(
    x: torch.Tensor,
    w0: torch.Tensor,
    bias0: torch.Tensor,
    w1: torch.Tensor,
    bias1: torch.Tensor,
    compute_dtype: torch.dtype,
    activation: str,
) -> torch.Tensor:
    x_compute = x.to(dtype=compute_dtype)
    w0_compute = w0.to(dtype=compute_dtype)
    bias0_compute = bias0.to(dtype=compute_dtype)
    w1_compute = w1.to(dtype=compute_dtype)
    bias1_compute = bias1.to(dtype=compute_dtype)

    hidden = F.linear(x_compute, w0_compute, bias0_compute)
    if activation == "gelu":
        hidden = F.gelu(hidden)
    elif activation == "quick_gelu":
        hidden = quick_gelu(hidden)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    out = F.linear(hidden, w1_compute, bias1_compute)
    return out.float()


def build_cute_infer_mlp(
    hidden_size: int,
    intermediate_dim: int,
    w0: torch.Tensor,
    bias0: torch.Tensor,
    w1: torch.Tensor,
    bias1: torch.Tensor,
    int8: bool,
    chunk_size: int,
) -> CuteInferMLP:
    return CuteInferMLP(
        hidden_size=hidden_size,
        intermediate_dim=intermediate_dim,
        W0=w0.to(dtype=torch.float16).contiguous(),
        bias0=bias0.to(dtype=torch.float16).contiguous(),
        W1=w1.to(dtype=torch.float16).contiguous(),
        bias1=bias1.to(dtype=torch.float16).contiguous(),
        int8=int8,
        chunk_size=chunk_size,
    )


def maybe_import_qwen_feed_forward(pbr_edit_root: str | None):
    if pbr_edit_root is None:
        return None
    root = pathlib.Path(pbr_edit_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from diffsynth.models.qwen_image_dit import QwenFeedForward

    return QwenFeedForward


def run_qwen_module(
    qwen_cls,
    x: torch.Tensor,
    w0: torch.Tensor,
    bias0: torch.Tensor,
    w1: torch.Tensor,
    bias1: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    module = qwen_cls(dim=w0.shape[1], dim_out=w1.shape[0], dropout=0.0).to(device=x.device, dtype=compute_dtype)
    module.eval()
    with torch.no_grad():
        module.net[0].proj.weight.copy_(w0.to(dtype=compute_dtype))
        module.net[0].proj.bias.copy_(bias0.to(dtype=compute_dtype))
        module.net[2].weight.copy_(w1.to(dtype=compute_dtype))
        module.net[2].bias.copy_(bias1.to(dtype=compute_dtype))
        return module(x.to(dtype=compute_dtype)).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare mh_cute_ops CuteInferMLP against torch MLP references used by Qwen-style DiT blocks."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--bias-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--int8", action="store_true", help="Use int8 preprocess_XW path instead of fp8.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--pbr-edit-root",
        type=str,
        default=None,
        help="Optional path to pbr-edit root. If set, also instantiates diffsynth.models.qwen_image_dit.QwenFeedForward.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA because CuteInferMLP depends on deep_gemm.")

    device = torch.device("cuda")
    hidden_size = args.hidden_size
    intermediate_dim = args.intermediate_size or hidden_size * 4

    if hidden_size % args.chunk_size != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by chunk_size={args.chunk_size}.")
    if intermediate_dim % args.chunk_size != 0:
        raise ValueError(
            f"intermediate_size={intermediate_dim} must be divisible by chunk_size={args.chunk_size}."
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    x = torch.randn(
        args.batch_size,
        args.seq_len,
        hidden_size,
        device=device,
        dtype=torch.float32,
    )
    x.mul_(args.input_scale)

    w0 = torch.empty(intermediate_dim, hidden_size, device=device, dtype=torch.float32)
    w1 = torch.empty(hidden_size, intermediate_dim, device=device, dtype=torch.float32)
    torch.nn.init.kaiming_normal_(w0, nonlinearity="relu")
    torch.nn.init.kaiming_normal_(w1, nonlinearity="relu")
    w0.mul_(args.weight_scale)
    w1.mul_(args.weight_scale)

    bias0 = torch.randn(intermediate_dim, device=device, dtype=torch.float32)
    bias1 = torch.randn(hidden_size, device=device, dtype=torch.float32)
    bias0.mul_(args.bias_scale)
    bias1.mul_(args.bias_scale)

    cute_mlp = build_cute_infer_mlp(
        hidden_size=hidden_size,
        intermediate_dim=intermediate_dim,
        w0=w0,
        bias0=bias0,
        w1=w1,
        bias1=bias1,
        int8=args.int8,
        chunk_size=args.chunk_size,
    ).to(device=device)
    cute_mlp.eval()

    qwen_cls = maybe_import_qwen_feed_forward(args.pbr_edit_root)

    with torch.no_grad():
        torch_fp32_gelu = run_torch_mlp(x, w0, bias0, w1, bias1, torch.float32, "gelu")
        torch_fp16_gelu = run_torch_mlp(x, w0, bias0, w1, bias1, torch.float16, "gelu")
        torch_fp32_quick = run_torch_mlp(x, w0, bias0, w1, bias1, torch.float32, "quick_gelu")
        torch_fp16_quick = run_torch_mlp(x, w0, bias0, w1, bias1, torch.float16, "quick_gelu")
        cute_out = cute_mlp(x).float()
        if qwen_cls is not None:
            qwen_fp16 = run_qwen_module(qwen_cls, x, w0, bias0, w1, bias1, torch.float16)
            qwen_fp32 = run_qwen_module(qwen_cls, x, w0, bias0, w1, bias1, torch.float32)
        else:
            qwen_fp16 = None
            qwen_fp32 = None
    torch.cuda.synchronize(device)

    print("[config]")
    print(
        f"  batch_size={args.batch_size} seq_len={args.seq_len} hidden_size={hidden_size} "
        f"intermediate_size={intermediate_dim}"
    )
    print(
        f"  chunk_size={args.chunk_size} int8={args.int8} seed={args.seed} "
        f"input_scale={args.input_scale} weight_scale={args.weight_scale} bias_scale={args.bias_scale}"
    )
    if args.pbr_edit_root is not None:
        print(f"  pbr_edit_root={pathlib.Path(args.pbr_edit_root).resolve()}")

    print("[tensor]")
    print(f"  input              {tensor_stats(x)}")
    print(f"  torch_fp32_gelu    {tensor_stats(torch_fp32_gelu)}")
    print(f"  torch_fp16_gelu    {tensor_stats(torch_fp16_gelu)}")
    print(f"  torch_fp32_quick   {tensor_stats(torch_fp32_quick)}")
    print(f"  torch_fp16_quick   {tensor_stats(torch_fp16_quick)}")
    print(f"  cute_out           {tensor_stats(cute_out)}")
    if qwen_fp32 is not None and qwen_fp16 is not None:
        print(f"  qwen_fp32          {tensor_stats(qwen_fp32)}")
        print(f"  qwen_fp16          {tensor_stats(qwen_fp16)}")

    print("[diff]")
    print_error_metrics("torch_fp16_gelu vs fp32_gelu", torch_fp16_gelu, torch_fp32_gelu)
    print_error_metrics("torch_fp16_quick vs fp32_quick", torch_fp16_quick, torch_fp32_quick)
    print_error_metrics("torch_fp16_gelu vs fp16_quick", torch_fp16_gelu, torch_fp16_quick)
    print_error_metrics("cute vs torch_fp32_gelu", cute_out, torch_fp32_gelu)
    print_error_metrics("cute vs torch_fp16_gelu", cute_out, torch_fp16_gelu)
    print_error_metrics("cute vs torch_fp32_quick", cute_out, torch_fp32_quick)
    print_error_metrics("cute vs torch_fp16_quick", cute_out, torch_fp16_quick)
    if qwen_fp32 is not None and qwen_fp16 is not None:
        print_error_metrics("qwen_fp32 vs torch_fp32_quick", qwen_fp32, torch_fp32_quick)
        print_error_metrics("qwen_fp16 vs torch_fp16_quick", qwen_fp16, torch_fp16_quick)
        print_error_metrics("cute vs qwen_fp32", cute_out, qwen_fp32)
        print_error_metrics("cute vs qwen_fp16", cute_out, qwen_fp16)

    topk_error_report("cute vs torch_fp16_gelu", cute_out, torch_fp16_gelu, args.topk)
    if qwen_fp16 is not None:
        topk_error_report("cute vs qwen_fp16", cute_out, qwen_fp16, args.topk)


if __name__ == "__main__":
    main()
