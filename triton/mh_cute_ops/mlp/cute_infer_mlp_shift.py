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

class AG_CuteInferMLP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X0, weight0, bias0, weight1, bias1):

        B, L, H = X0.shape
        if X0.dim() != 2:
            X0 = X0.reshape(-1, X0.shape[-1])
        
        assert X0.dtype == torch.float32
        X = X0.to(torch.float16)

        M, K = X.shape
        N, K_ = weight0[0].shape
        # assert K == K_

        D = torch.empty((M, N), device=X.device, dtype=torch.float16)

        X_fp8, X_scale, X_shift, X_sum = per_token_cast_to_fp8_cute_shift(X, 0, bias=None, is_chunk32=True, compute_shift=True, fp8_cat=None, scale_cat=None, is_shift_cat=False)
        X_fp8, X_scale, _, _ = per_token_cast_to_fp8_cute_shift(X_shift, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=X_fp8, scale_cat=X_scale, is_shift_cat=True)
        X_fp8, X_scale, _, _ = per_token_cast_to_fp8_cute_shift(X_sum, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=X_fp8, scale_cat=X_scale, is_shift_cat=False)

        # 拼接3个fp8矩阵: mxk, mxk/32, mxk/32 -> mx(k+k/16)
        # X_combined_fp8 = torch.cat([X_fp8, X_shift_fp8, X_sum_fp8], dim=1)
        # X_combined_scale = torch.cat([X_scale, X_shift_scale, X_sum_scale], dim=1)
        
        deep_gemm.fp8_gemm_nt((X_fp8, X_scale), (weight0[0], weight0[1]), D, c=D, bias=bias0, disable_ue8m0_cast=True, recipe=(1, 1, 128), is_add=False, apply_gelu=True)

        # D = torch.nn.functional.gelu(D)

        D_fp8, D_scale, D_shift, D_sum = per_token_cast_to_fp8_cute_shift(D, 0, bias=None, is_chunk32=True, compute_shift=True, fp8_cat=None, scale_cat=None, is_shift_cat=False)
        D_fp8, D_scale, _, _ = per_token_cast_to_fp8_cute_shift(D_shift, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=D_fp8, scale_cat=D_scale, is_shift_cat=True)
        D_fp8, D_scale, _, _ = per_token_cast_to_fp8_cute_shift(D_sum, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=D_fp8, scale_cat=D_scale, is_shift_cat=False)

        E = torch.empty_like(X, dtype=torch.float16)
        deep_gemm.fp8_gemm_nt((D_fp8, D_scale), (weight1[0], weight1[1]), E, c=E, bias=bias1, disable_ue8m0_cast=True, recipe=(1, 1, 128), is_add=False)

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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_dim = intermediate_dim

        assert W0.dtype == torch.float16
        assert bias0.dtype == torch.float16
        assert W1.dtype == torch.float16
        assert bias1.dtype == torch.float16

        option_init = 0
        use_complex = True

        if use_complex == False:
            weight0_fp8, weight0_scale, shift0, sum0 = per_token_cast_to_fp8_cute_shift(W0, 0, bias=None, is_chunk32=True, compute_shift=True, fp8_cat=None, scale_cat=None, is_shift_cat=False)
            weight0_fp8, weight0_scale, _, _ = per_token_cast_to_fp8_cute_shift(shift0, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=weight0_fp8, scale_cat=weight0_scale, is_shift_cat=False)
            weight0_fp8, weight0_scale, _, _ = per_token_cast_to_fp8_cute_shift(sum0, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=weight0_fp8, scale_cat=weight0_scale, is_shift_cat=True)
        else:
            weight0_fp8, weight0_scale = per_token_cast_to_fp8_complex_shift(W0)

        self.weight0_fp8 = nn.Parameter(weight0_fp8)
        self.weight0_scale = nn.Parameter(weight0_scale)

        if use_complex == False:
            weight1_fp8, weight1_scale, shift1, sum1 = per_token_cast_to_fp8_cute_shift(W1, 0, bias=None, is_chunk32=True, compute_shift=True, fp8_cat=None, scale_cat=None, is_shift_cat=False)
            weight1_fp8, weight1_scale, _, _ = per_token_cast_to_fp8_cute_shift(shift1, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=weight1_fp8, scale_cat=weight1_scale, is_shift_cat=False)
            weight1_fp8, weight1_scale, _, _ = per_token_cast_to_fp8_cute_shift(sum1, preprocess=0, bias=None, is_chunk32=True, compute_shift=False, fp8_cat=weight1_fp8, scale_cat=weight1_scale, is_shift_cat=True)
        else:
            weight1_fp8, weight1_scale = per_token_cast_to_fp8_complex_shift(W1)

        self.weight1_fp8 = nn.Parameter(weight1_fp8)
        self.weight1_scale = nn.Parameter(weight1_scale)

        assert bias0.dim() == 1 and bias1.dim() == 1, "bias must be 1D"

        self.bias0 = nn.Parameter(bias0)
        self.bias1 = nn.Parameter(bias1)


    def forward(
        self,
        x: torch.Tensor,
    ):
        return AG_CuteInferMLP.apply(x, (self.weight0_fp8, self.weight0_scale), self.bias0, (self.weight1_fp8, self.weight1_scale), self.bias1)

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


def test_mlp(seq_len, hidden_size):
    from deep_gemm.testing import calc_diff

    intermediate_dim = 4 * hidden_size
    W0 = torch.empty(intermediate_dim, hidden_size, device='cuda', dtype=torch.float16)
    bias0 = torch.randn(intermediate_dim, device='cuda', dtype=torch.float16)
    W1 = torch.empty( hidden_size, intermediate_dim, device='cuda', dtype=torch.float16)
    bias1 = torch.randn( hidden_size, device='cuda', dtype=torch.float16)

    torch.nn.init.kaiming_normal_(W0, nonlinearity='relu')
    torch.nn.init.kaiming_normal_(W1, nonlinearity='relu')

    layer_cute = CuteInferMLP(hidden_size, intermediate_dim, W0, bias0, W1, bias1)
    layer_torch = MLP(hidden_size, intermediate_dim)

    # copy to cute parameters
    layer_torch.up_proj.weight.data = W0.clone()
    layer_torch.up_proj.bias.data = bias0.clone()
    layer_torch.down_proj.weight.data = W1.clone()
    layer_torch.down_proj.bias.data = bias1.clone()

    batch_size = 2
    X = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)
    X.requires_grad = False

    # X[:, :, 1] = 1000
    # W0[:, 1] = 1000
    # W1[:, 1] = 1000

    num_iters = 40
    # forward
    torch.cuda.synchronize()
    for i in range(num_iters):
        with nvtx.annotate("cute forward: seq_len=" + str(seq_len) + " hidden_size=" + str(hidden_size)):
            Y_cute = layer_cute(X)
            torch.cuda.synchronize()

    torch.cuda.synchronize()
    for i in range(num_iters):
        with nvtx.annotate("torch forward: seq_len=" + str(seq_len) + " hidden_size=" + str(hidden_size)):
            Y_torch = layer_torch(X)
            torch.cuda.synchronize()


    assert Y_cute.shape == Y_torch.shape
    assert Y_cute.dtype == Y_torch.dtype
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
            test_mlp(seq_len, hidden_size)