import torch
import torch.nn as nn

import nvtx

import deep_gemm

try:
    from .cute_per_token_cast_shift import per_token_cast_to_fp8_cute_shift
    from .helper_shift import per_token_cast_to_fp8_complex_shift
except:
    from cute_per_token_cast_shift import per_token_cast_to_fp8_cute_shift
    from helper_shift import per_token_cast_to_fp8_complex_shift

class AG_CuteInferLinearShift(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X0, weight, bias):

        B, L, H = X0.shape
        if X0.dim() != 2:
            X0 = X0.reshape(-1, X0.shape[-1])
        
        assert X0.dtype == torch.float32 or X0.dtype == torch.float16
        X = X0.to(torch.float16)

        M, K = X.shape
        N, K_ = weight[0].shape
        # assert K == K_

        D = torch.empty((M, N), device=X.device, dtype=torch.float16)

        X_fp8, X_scale, X_shift, X_sum = per_token_cast_to_fp8_cute_shift(X, 0, bias=None, is_chunk32=True, compute_shift=True, fp8_cat=None, scale_cat=None, is_shift_cat=False)
        X_fp8, X_scale, _, _ = per_token_cast_to_fp8_cute_shift(X_shift, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=X_fp8, scale_cat=X_scale, is_shift_cat=True)
        X_fp8, X_scale, _, _ = per_token_cast_to_fp8_cute_shift(X_sum, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=X_fp8, scale_cat=X_scale, is_shift_cat=False)

        # 拼接3个fp8矩阵: mxk, mxk/32, mxk/32 -> mx(k+k/16)
        # X_combined_fp8 = torch.cat([X_fp8, X_shift_fp8, X_sum_fp8], dim=1)
        # X_combined_scale = torch.cat([X_scale, X_shift_scale, X_sum_scale], dim=1)
        

        deep_gemm.fp8_gemm_nt((X_fp8, X_scale), (weight[0], weight[1]), D, c=D, bias=bias, disable_ue8m0_cast=True, recipe=(1, 1, 128), is_add=False, apply_gelu=False)


        return D.reshape(B, L, N)

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("no backward of cute infer mlp")
        return None, None, None, None, None

class CuteInferLinearShift(nn.Module):

    def __init__(
        self,
        W: torch.tensor,
        bias: torch.tensor,
    ):
        super().__init__()

        assert W.dtype == torch.float16
        if bias is not None:
            assert bias.dtype == torch.float16

        weight_fp8, weight_scale, shift, sum = per_token_cast_to_fp8_cute_shift(W, 0, bias=None, is_chunk32=True, compute_shift=True, fp8_cat=None, scale_cat=None, is_shift_cat=False)
        weight_fp8, weight_scale, _, _ = per_token_cast_to_fp8_cute_shift(shift, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=weight_fp8, scale_cat=weight_scale, is_shift_cat=False)
        weight_fp8, weight_scale, _, _ = per_token_cast_to_fp8_cute_shift(sum, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=weight_fp8, scale_cat=weight_scale, is_shift_cat=True)

        self.weight_fp8 = nn.Parameter(weight_fp8)
        self.weight_scale = nn.Parameter(weight_scale)

        if bias is not None:
            assert bias.dim() == 1 
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None


    def forward(
        self,
        x: torch.Tensor,
    ):
        return AG_CuteInferLinear.apply(x, (self.weight_fp8, self.weight_scale), self.bias)



def test_linear(seq_len, hidden_size):
    from deep_gemm.testing import calc_diff

    intermediate_dim = 3 * hidden_size

    W = torch.randn(intermediate_dim, hidden_size, device='cuda', dtype=torch.float16)
    bias = torch.randn(intermediate_dim, device='cuda', dtype=torch.float16)

    layer_cute = CuteInferLinearShift(W, bias)
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
    x16 = X.to(torch.float16)
    torch.cuda.synchronize()
    for i in range(num_iters):
        with nvtx.annotate("torch forward: seq_len=" + str(seq_len) + " hidden_size=" + str(hidden_size)):
            Y_torch = layer_torch(x16)
            torch.cuda.synchronize()

    torch.cuda.synchronize()
    for i in range(num_iters):
        with nvtx.annotate("cute forward: seq_len=" + str(seq_len) + " hidden_size=" + str(hidden_size)):
            Y_cute = layer_cute(X)
            torch.cuda.synchronize()

    Y_cute = Y_cute.to(torch.float32)
    Y_torch = Y_torch.to(torch.float32)

    assert Y_cute.shape == Y_torch.shape
    # assert Y_cute.dtype == Y_torch.dtype
    assert Y_cute.device == Y_torch.device

    # verify correctness
    print("--------------------------------forward--------------------------------")
    for i in range(10):
        print("element 0 of Y_cute and Y_torch", Y_cute.flatten()[i].item(), Y_torch.flatten()[i].item())

    print("max error of Y: ", (Y_cute - Y_torch).abs().max().item())
    print("mean error of Y: ", (Y_cute - Y_torch).abs().mean().item())
    print("calc diff: ", calc_diff(Y_cute, Y_torch).item())
    print("-----------------------------------------------------")

if __name__ == "__main__":
    torch.manual_seed(0)

    seq_lens = [4096, 8192, 16384, 32768]
    hidden_sizes = [1536, 2048, 3072]

    seq_lens = [32000]
    hidden_sizes = [3072]

    for seq_len in seq_lens:
        for hidden_size in hidden_sizes:
            test_linear(seq_len, hidden_size)