import torch
import torch.nn as nn

import nvtx

import deep_gemm

from mh_cute_ops.mlp.cute_per_token_cast import preprocess_XW

class AG_CuteInferMLP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X0, W0, bias0, W1, bias1, int8, chunk_size, compute_dtype):

        B, L, H = X0.shape
        if X0.dim() != 2:
            X0 = X0.reshape(-1, X0.shape[-1])

        assert X0.dtype == torch.float32
        X = X0.to(compute_dtype)

        M, K = X.shape
        N, K_ = W0[0].shape
        # assert K == K_

        D = torch.empty((M, N), device=X.device, dtype=compute_dtype)
        X_fp8, X_scale = preprocess_XW(X, int8=int8, chunk_size=chunk_size)
        deep_gemm.fp8_gemm_nt((X_fp8, X_scale), W0, D, c=D, bias=bias0, disable_ue8m0_cast=True, recipe=(1, 1, 128), is_add=False, apply_gelu=True)

        E = torch.empty_like(X, dtype=compute_dtype)
        D_fp8, D_scale = preprocess_XW(D, int8=int8, chunk_size=chunk_size)
        deep_gemm.fp8_gemm_nt((D_fp8, D_scale), W1, E, c=E, bias=bias1, disable_ue8m0_cast=True, recipe=(1, 1, 128), is_add=False, apply_gelu=False)

        return E.reshape(B, L, H)

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("no backward of cute infer mlp")
        return None, None, None, None, None

class CuteInferMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_dim: int,
        W0: torch.tensor,
        bias0: torch.tensor,
        W1: torch.tensor,
        bias1: torch.tensor,
        int8: bool,
        chunk_size: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_dim = intermediate_dim
        self.int8 = int8
        self.chunk_size = chunk_size
        self.compute_dtype = W0.dtype

        assert self.compute_dtype in (torch.float16, torch.bfloat16)
        assert W0.dtype == self.compute_dtype
        assert bias0.dtype == self.compute_dtype
        assert W1.dtype == self.compute_dtype
        assert bias1.dtype == self.compute_dtype

        W0_fp8, W0_scale = self._preprocess_XW(W0)
        self.W0_fp8 = nn.Parameter(W0_fp8, requires_grad=False)
        self.W0_scale = nn.Parameter(W0_scale, requires_grad=False)

        W1_fp8, W1_scale = self._preprocess_XW(W1)
        self.W1_fp8 = nn.Parameter(W1_fp8, requires_grad=False)
        self.W1_scale = nn.Parameter(W1_scale, requires_grad=False)

        assert bias0.dim() == 1 and bias1.dim() == 1, "bias must be 1D"

        self.bias0 = nn.Parameter(bias0, requires_grad=False)
        self.bias1 = nn.Parameter(bias1, requires_grad=False)

    def _preprocess_XW(self, X: torch.Tensor):
        return preprocess_XW(X, int8=self.int8, chunk_size=self.chunk_size)

    def forward(
        self,
        x: torch.Tensor,
    ):
        return AG_CuteInferMLP.apply(
            x,
            (self.W0_fp8, self.W0_scale),
            self.bias0,
            (self.W1_fp8, self.W1_scale),
            self.bias1,
            self.int8,
            self.chunk_size,
            self.compute_dtype,
        )

class MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_dim = intermediate_dim

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_dim, bias=True, device=device, dtype=torch.float16
        )
        self.down_proj = nn.Linear(
            self.intermediate_dim, self.hidden_size, bias=True, device=device, dtype=torch.float16
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        up = self.up_proj(x.to(torch.float16))
        hidden = torch.nn.functional.gelu(up)
        down_proj = self.down_proj(hidden)

        return down_proj


def test_mlp(seq_len, hidden_size, int8, chunk_size):
    from deep_gemm.testing import calc_diff

    intermediate_dim = 4 * hidden_size
    W0 = torch.empty(intermediate_dim, hidden_size, device='cuda', dtype=torch.float16)
    bias0 = torch.randn(intermediate_dim, device='cuda', dtype=torch.float16)
    W1 = torch.empty( hidden_size, intermediate_dim, device='cuda', dtype=torch.float16)
    bias1 = torch.randn( hidden_size, device='cuda', dtype=torch.float16)

    torch.nn.init.kaiming_normal_(W0, nonlinearity='relu')
    torch.nn.init.kaiming_normal_(W1, nonlinearity='relu')

    layer_cute = CuteInferMLP(hidden_size, intermediate_dim, W0, bias0, W1, bias1, int8, chunk_size)
    layer_torch = MLP(hidden_size, intermediate_dim)

    # copy to cute parameters
    layer_torch.up_proj.weight.data = W0.clone()
    layer_torch.up_proj.bias.data = bias0.clone()
    layer_torch.down_proj.weight.data = W1.clone()
    layer_torch.down_proj.bias.data = bias1.clone()

    batch_size = 2
    X = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)
    X.requires_grad = True

    num_iters = 40
    warmup_iters = 10
    x16 = X.to(torch.float16)

    # Warmup for torch
    torch.cuda.synchronize()
    for i in range(warmup_iters):
        Y_torch = layer_torch(x16)
    torch.cuda.synchronize()
    
    # Time torch forward
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(num_iters):
        Y_torch = layer_torch(x16)
    end_event.record()
    torch.cuda.synchronize()
    torch_time_ms = start_event.elapsed_time(end_event) / num_iters

    # Warmup for cute
    torch.cuda.synchronize()
    for i in range(warmup_iters):
        Y_cute = layer_cute(X)
    torch.cuda.synchronize()

    # Time cute forward
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(num_iters):
        Y_cute = layer_cute(X)
    end_event.record()
    torch.cuda.synchronize()
    cute_time_ms = start_event.elapsed_time(end_event) / num_iters

    Y_cute = Y_cute.to(torch.float32)
    Y_torch = Y_torch.to(torch.float32)

    assert Y_cute.shape == Y_torch.shape
    assert Y_cute.dtype == Y_torch.dtype
    assert Y_cute.device == Y_torch.device

    # verify correctness
    print("--------------------------------forward--------------------------------")
    for i in range(10):
        print("element 0 of Y_cute and Y_torch", Y_cute.flatten()[i].item(), Y_torch.flatten()[i].item())

    def calc_qsnr(x, x_ref):
        qsnr = -10 * torch.log10(torch.mean((x.float() - x_ref.float())**2) / (torch.mean(x_ref.float()**2)))
        return qsnr

    print(f"calc diff: {calc_diff(Y_cute, Y_torch).item():.6e}")
    print(f"calc qsnr: {calc_qsnr(Y_cute, Y_torch).item():.6e}")
    print(f"max error: {((Y_cute - Y_torch).abs().max().item()):.6e}")
    print(f"mean error: {((Y_cute - Y_torch).abs().mean().item()):.6e}")
    
    print("--------------------------------timing--------------------------------")
    print(f"Torch forward avg time: {torch_time_ms:.4f} ms")
    print(f"Cute forward avg time:  {cute_time_ms:.4f} ms")
    print(f"Speedup (Torch/Cute):   {torch_time_ms/cute_time_ms:.2f}x")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    torch.manual_seed(0)

    seq_lens = [4096, 8192, 16384, 32768]
    hidden_sizes = [1536, 2048, 3072]

    seq_lens = [32000]
    hidden_sizes = [3072]

    int8s = [True, False]
    chunk_sizes = [32, 64, 128]

    for int8 in int8s:
        for seq_len in seq_lens:
            for hidden_size in hidden_sizes:
                for chunk_size in chunk_sizes:
                    test_mlp(seq_len, hidden_size, int8, chunk_size)
