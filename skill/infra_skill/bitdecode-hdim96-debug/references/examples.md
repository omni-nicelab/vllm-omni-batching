# BitDecoding hdim96 Examples

## Minimal e2e Repro

Run 256 tokens first. This reaches the first 128-token packed prefix and catches most packed-prefix LSE/layout failures quickly.

```bash
CUDA_VISIBLE_DEVICES=0 \
PERF_MODE=1 \
MODE=bitdecode \
MAX_SEQ_LEN=256 \
TRUNCATE_SIZE=256 \
OVERLAP_SIZE=128 \
MAX_SAMPLES=1 \
SKIP_MESH=1 \
IGNORE_EOS=1 \
PROGRESS_INTERVAL=128 \
OUT_DIR=outputs/e2e_shapeopt_bitdecode_debug256 \
bash run.sh
```

Run 1024 next:

```bash
CUDA_VISIBLE_DEVICES=0 \
PERF_MODE=1 \
MODE=bitdecode \
MAX_SEQ_LEN=1024 \
TRUNCATE_SIZE=1024 \
OVERLAP_SIZE=512 \
MAX_SAMPLES=1 \
SKIP_MESH=1 \
IGNORE_EOS=1 \
PROGRESS_INTERVAL=512 \
OUT_DIR=outputs/e2e_shapeopt_bitdecode_smoke1024 \
bash run.sh
```

Then run 10k with mesh:

```bash
CUDA_VISIBLE_DEVICES=0 \
PERF_MODE=1 \
MODE=bitdecode \
MAX_SEQ_LEN=10000 \
TRUNCATE_SIZE=10000 \
OVERLAP_SIZE=5000 \
MAX_SAMPLES=1 \
SKIP_MESH=0 \
IGNORE_EOS=1 \
PROGRESS_INTERVAL=512 \
OUT_DIR=outputs/e2e_shapeopt_bitdecode_10k \
bash run.sh
```

## Dump Replay Pattern

Use this when a dump such as `outputs/bitdecode_lse_failure.pt` exists.

```bash
CUDA_VISIBLE_DEVICES=0 PERF_MODE=1 .venv/bin/python - <<'PY'
import torch
import bit_decode_cuda

p = "outputs/bitdecode_lse_failure.pt"
d = torch.load(p, map_location="cpu")
q = d["q"].cuda().contiguous()
kp = d["k_pack"].cuda().contiguous()
kpar = d["k_params"].cuda().contiguous()
vp = d["v_pack"].cuda().contiguous()
vpar = d["v_params"].cuda().contiguous()
scale = float(d["softmax_scale"])
L = int(d["logical_seqlen_k"])

out, lse = bit_decode_cuda.fwd_kvcache_lse_i4(
    q, kp, kpar, vp, vpar,
    None, scale, "k-channel", 128,
    False, -1, -1, 0.0, True, 0, L,
)
native = bit_decode_cuda.native96_fwd_kvcache_i4(q, kp, kpar, vp, vpar, scale, L)
torch.cuda.synchronize()

print("out finite", bool(torch.isfinite(out.float()).all()))
print("lse finite", bool(torch.isfinite(lse.float()).all()))
diff = (out.float() - native.float()).abs()
print("diff max", float(torch.nan_to_num(diff).max()))
print("diff mean", float(torch.nan_to_num(diff).mean()))
print("lse", [round(float(x), 3) for x in lse.cpu().flatten()])
PY
```

Expected after the float reduction-buffer fix:

```text
out finite True
lse finite True
diff max around 1e-3 or lower
```

## Inspect Dump Magnitudes

Use this to distinguish model-state blowup from kernel LSE overflow.

```bash
.venv/bin/python - <<'PY'
import torch

d = torch.load("outputs/bitdecode_lse_failure.pt", map_location="cpu")
for k, v in d.items():
    if not torch.is_tensor(v):
        print(k, v)
        continue
    if v.is_floating_point():
        vf = v.float()
        print(
            k, tuple(v.shape), v.dtype,
            "finite", bool(torch.isfinite(vf).all()),
            "nan", int(torch.isnan(vf).sum()),
            "inf", int(torch.isinf(vf).sum()),
            "absmax", float(torch.nan_to_num(vf).abs().max()) if vf.numel() else None,
        )
    else:
        print(k, tuple(v.shape), v.dtype)
PY
```

Interpretation:

- `out finite=True` with `lse finite=False` points at softmax/LSE reduction or combine.
- `full` mode passing while `bitdecode` fails points at BitDecoding kernel/cache handling, not the model itself.
- Large q values are not by themselves a reason to clamp; full cache may handle them correctly.

## V Pack One-Hot Check

Use this to verify hdim96 4bit V pack physical width and dim mapping.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python - <<'PY'
import torch
from bit_decode import allocate_packed_kv, kvcache_pack_int

B, H, D, L = 1, 1, 96, 128
cu = torch.arange(0, (B + 1) * L, L, dtype=torch.int32, device="cuda")
k = torch.zeros(B, L, H, D, device="cuda", dtype=torch.float16)

for d in [0, 23, 24, 31, 32, 56, 63, 64, 88, 95]:
    v = torch.zeros(B, L, H, D, device="cuda", dtype=torch.float16)
    v[..., d] = 1
    kp, kpar, vp, vpar = allocate_packed_kv(
        B, L, H, D,
        physical_head_dim=D,
        device="cuda",
        group_size=128,
        num_bits=4,
    )
    kvcache_pack_int(k, kp, kpar, v, vp, vpar, None, cu, L, "k-channel", 128, 4)
    torch.cuda.synchronize()
    row = vp[0, 0, 0].cpu().to(torch.int32)
    nz = (row != 0).nonzero().flatten().tolist()
    print("d", d, "vpack_shape", tuple(vp.shape), "nz", [(i, int(row[i])) for i in nz[:5]])
PY
```

Expected hdim96 4bit V pack shape:

```text
(1, 128, 1, 32)
```

## Build

After changing CUDA headers or generated hdim96 files, rebuild:

```bash
MAX_JOBS=8 .venv/bin/python setup.py build_ext --inplace
```

If stale objects are suspected, remove hdim96 objects before rebuilding:

```bash
rm -f \
  build/temp.linux-x86_64-cpython-312/csrc/bit_decode/native96_decode.o \
  build/temp.linux-x86_64-cpython-312/csrc/bit_decode/src/genfile/flash_qpack_hdim96_fp16_sm80_4bit.o \
  build/temp.linux-x86_64-cpython-312/csrc/bit_decode/src/genfile/flash_fwd_split_hdim96_fp16_sm80_4bit.o
MAX_JOBS=8 .venv/bin/python setup.py build_ext --inplace
```
