---
name: bitdecode-hdim96-debug
description: Debug and stabilize BitDecoding hdim96 Flash/CUTLASS tensor-core decode paths, especially ShapeOPT e2e failures, NaN/inf attention outputs, qpack/dequant layout bugs, packed-prefix LSE issues, split-kv hdim96 template failures, and single-operator correctness/performance regressions versus FA2. Use when working on BitDecoding hdim96 4bit KV-cache kernels, ShapeOPT bitdecode e2e runs, Flash split-kv templates, qpack layout, V pack layout, K/V dequant params, or CUDA numerical failures in packed-prefix attention.
---

# BitDecoding hdim96 Debug

## Core Rules

Use the Flash/CUTLASS BitDecoding tensor-core path as the production path.

Do not fix failures by:
- Switching production decode to `native96_decode.cu`
- Adding fallback paths
- Applying `nan_to_num`, clamping, or masking non-finite output
- Padding hdim96 to hdim128 to bypass broken hdim96 templates
- Declaring success from random single-op tests without ShapeOPT e2e

Use native96 only as a correctness oracle or debug comparison.

## Production Target

The target path is:

- qpack: `flash_qpack_hdim96...`
- decode: `flash_fwd_split_hdim96...`
- Python entry: `fwd_kvcache_i4` / `fwd_kvcache_lse_i4`
- ShapeOPT mode: `MODE=bitdecode`

## Debug Workflow

Start with the smallest e2e that reproduces packed-prefix behavior. Use 256 tokens because the first 128-token packed prefix enters the Flash/CUTLASS path there.

Then run 1024 to cover multiple packed blocks. Only run 10k with mesh after 256 and 1024 pass.

For concrete commands, dump replay snippets, and validation examples, read [references/examples.md](references/examples.md).

## Failure Triage

When e2e reports non-finite attention output, inspect whether failure is in:

1. qpack layout
2. K params layout
3. V pack/dequant layout
4. packed-prefix LSE
5. split-kv combine
6. downstream model state after earlier numeric drift

Prefer dump replay over guessing. Save or inspect:

- `q`
- `k_pack`
- `k_params`
- `v_pack`
- `v_params`
- `out`
- `lse`
- `logical_seqlen_k`
- `num_splits`

Replay both Flash and native oracle when available.

## hdim96 Layout Requirements

For hdim96 + 4bit:

- K pack uses BitDecoding time layout:
  ```cpp
  packed_t = (t & ~127) / 4 + (t & 31);
  nibble = (t >> 5) & 3;
  ```

- V pack physical width is 32, not `96 / 4 = 24`.

- K-channel params for hdim96 must use 128 half2 slots for physical alignment.

- V dequant/GEMM should use V-specific tensor-core traits:
  - `kHeadDim_v_pack`
  - `TiledMmaV`
  - `TiledMmaV_i4`
  - V-specific copy path

Do not reuse K i4 copy/MMA assumptions blindly for V.

## Numerical Trap

If packed-prefix `out` is finite but `lse` contains `inf`, inspect the softmax reduction temporary shared memory.

A common hdim96 failure is using fp16 shared memory for row max / row sum reduction. Large raw QK scores can exceed fp16 range before final LSE is computed.

Correct pattern:

- Store softmax reduction temporary values in `ElementAccum` / float shared memory.
- Do not reuse fp16 `smem_acc` for reduction temp.
- Wire forward kernels to use `shared_storage.smem_reduce_tmp`.

Expected fix shape:

```cpp
array_aligned<ElementAccum, cosize_v<SmemLayoutReduce_tmp>> smem_reduce_tmp;
```

and:

```cpp
Tensor sReduce_tmp = make_tensor(
    make_smem_ptr(shared_storage.smem_reduce_tmp.data()),
    typename Kernel_traits::SmemLayoutReduce_tmp{}
);
```

## Oracle Use

Use native96 only to validate Flash/CUTLASS output.

Expected single-op checks:

- Flash output finite
- LSE finite
- Flash vs native max error around `1e-3` or lower for difficult hdim96 cases

Do not route production ShapeOPT decode through native96.

## Validation Checklist

Run focused tests first:

```bash
CUDA_VISIBLE_DEVICES=0 PERF_MODE=0 .venv/bin/python -m pytest -q \
  tests/test_hdim96_template_correctness.py \
  tests/test_bitdecode_qpack_mask.py \
  tests/test_shapeopt_bitdecode_adapter.py
```

Then run:

1. 256 e2e
2. 1024 e2e
3. 10k e2e with `SKIP_MESH=0`

A successful 10k run must show:

- `MODE=bitdecode`
- `mesh_saved=yes`
- no non-finite attention output
- `.token.txt` written
- `.bitdecode.ply` written

## Performance Follow-Up

Only start NCU/FA2 performance work after correctness passes e2e.

Use `cuda-optimize` / `ncu-rep-analyze` on the actual hdim96 Flash/CUTLASS kernels, not native96. Profile the production path after verifying:

- qpack layout correctness
- packed-prefix LSE correctness
- split-kv output correctness
- ShapeOPT 10k e2e success
