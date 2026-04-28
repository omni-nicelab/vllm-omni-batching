import torch
import torch.nn as nn

import nvtx

import deep_gemm

from mh_cute_ops.mlp.cute_per_token_cast import preprocess_XW

class AG_CuteInferLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X0, weight, bias, int8, chunk_size, compute_dtype):
        output_shape = (*X0.shape[:-1], weight[0].shape[0])
        if X0.dim() != 2:
            X0 = X0.reshape(-1, X0.shape[-1])

        assert X0.dtype in (torch.float32, torch.float16, torch.bfloat16)
        X = X0.to(compute_dtype)

        M, K = X.shape
        N, K_ = weight[0].shape
        # assert K == K_

        D = torch.empty((M, N), device=X.device, dtype=compute_dtype)

        X_fp8, X_scale = preprocess_XW(X, int8=int8, chunk_size=chunk_size)

        deep_gemm.fp8_gemm_nt((X_fp8, X_scale), (weight[0], weight[1]), D, c=D, bias=bias, disable_ue8m0_cast=True, recipe=(1, 1, 128), is_add=False, apply_gelu=False)

        return D.reshape(output_shape)

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("no backward of cute infer mlp")
        return None, None, None, None, None, None

class CuteInferLinear(nn.Module):

    def __init__(
        self,
        W: torch.tensor,
        bias: torch.tensor,
        int8: bool,
        chunk_size: int
    ):
        super().__init__()

        self.int8 = int8
        self.chunk_size = chunk_size
        self.compute_dtype = W.dtype

        assert self.compute_dtype in (torch.float16, torch.bfloat16)
        assert W.dtype == self.compute_dtype
        if bias is not None:
            assert bias.dtype == self.compute_dtype

        weight_fp8, weight_scale = self._preprocess_XW(W)

        self.weight_fp8 = nn.Parameter(weight_fp8, requires_grad=False)
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)

        if bias is not None:
            assert bias.dim() == 1
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    def _preprocess_XW(self, X: torch.Tensor):
        return preprocess_XW(X, int8=self.int8, chunk_size=self.chunk_size)

    def forward(
        self,
        x: torch.Tensor,
    ):
        return AG_CuteInferLinear.apply(
            x,
            (self.weight_fp8, self.weight_scale),
            self.bias,
            self.int8,
            self.chunk_size,
            self.compute_dtype,
        )



def test_linear(seq_len, hidden_size, int8, chunk_size):
    from deep_gemm.testing import calc_diff

    intermediate_dim = 3 * hidden_size

    W = torch.randn(intermediate_dim, hidden_size, device='cuda', dtype=torch.float16)
    bias = torch.randn(intermediate_dim, device='cuda', dtype=torch.float16)
    W.requires_grad = False
    bias.requires_grad = False

    layer_cute = CuteInferLinear(W, bias, int8=int8, chunk_size=chunk_size)
    layer_torch = nn.Linear(hidden_size, intermediate_dim, bias=True)

    # copy to cute parameters
    layer_torch.weight.data = W.to(torch.float16).clone()
    layer_torch.bias.data = bias.to(torch.float16).clone()

    batch_size = 2
    X = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)
    X.requires_grad = False

    # X[:, :, 1] = 1000
    # W0[:, 1] = 1000
    # W1[:, 1] = 1000

    num_iters = 40
    warmup_iters = 10
    x16 = X.to(torch.float16)
    
    # Warmup for torch
    torch.cuda.synchronize()
    for i in range(warmup_iters):
        Y_torch = layer_torch(x16)
    torch.cuda.synchronize()
    
    # Benchmark torch forward with nvtx
    for i in range(num_iters):
        with nvtx.annotate("torch forward: seq_len=" + str(seq_len) + " hidden_size=" + str(hidden_size)):
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

    # Benchmark cute forward with nvtx
    for i in range(num_iters):
        with nvtx.annotate("cute forward: seq_len=" + str(seq_len) + " hidden_size=" + str(hidden_size)):
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
    # assert Y_cute.dtype == Y_torch.dtype
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
                        test_linear(seq_len, hidden_size, int8, chunk_size)
