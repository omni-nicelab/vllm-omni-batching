# ShapeOPT hdim96 KV4 Attention Case Study

## Problem

Target workload:

- ShapeOPT-style decoder self-attention.
- `hidden_size=1536`, `num_attention_heads=16`, `num_key_value_heads=16`, `head_dim=96`.
- Decode with an existing long KV cache, typically 10k to 50k tokens.
- Baseline is corresponding FA2 decode.
- Goal is fused 4-bit KV cache dequant + attention, not standalone dequant followed by FA2.

## Initial Triton NVFP4 Prototype

The first working Triton direction implemented:

- packed FP4 K/V cache input,
- blockwise scale load,
- FP4 dequant inside the attention kernel,
- QK,
- online softmax,
- V dequant,
- PV accumulation.

It proved the correct architecture: dequant must happen inside the streaming attention loop. However, on SM90/H20, software NVFP4 E2M1 decode plus scalar q_len=1 reductions were too expensive.

## Split-K Pivot

The next direction split decode into:

- **stage1**: one program per `(head, KV block/group)`, producing local numerator, max, and sum.
- **stage2**: one program per head, merging split results with online-softmax math.

This fixed the major parallelism problem. A single program per head streaming 10k tokens was too serial.

Useful details:

- Store raw numerator, `m`, and `l`; do not normalize in stage1.
- Merge with `weight_i = exp2(m_i - M)` and denominator `sum(weight_i * l_i)`.
- Benchmark stage1 and stage2 separately. In this case, stage1 dominated.

## Micro-Optimizations Tried

These improved the prototype but did not make NVFP4 Triton beat FA2:

- Decode packed bytes once instead of reloading per logical dim.
- Try scale-load deduplication; sometimes saved memory but added broadcast/register pressure.
- Replace LUT-like E2M1 decode with bitfield arithmetic.
- Use fp16 partial output workspace instead of fp32 after checking accuracy.
- Avoid stage1 `log` and division.

The important lesson: if the instruction mix is dominated by software dequant and scalar reductions, small memory tweaks will not be enough.

## Direction Change to Int4 Affine

The successful direction used bitdecoding-style int4 affine KV cache:

- K quantization: per 128-token group, per head, per dim.
- K pack: four token values for one dim into one `uint16`.
- V quantization: per token, per head over the 96 dims.
- V pack: 96 dims into 32 `uint16` words, three int4 values per word.
- Decode:
  - write current token to residual,
  - pack residual when the 128-token group fills,
  - run quantized prefix attention,
  - run fp16 residual attention,
  - merge prefix/residual outputs via LSE.

This matches the attention memory objective: reduce KV cache bandwidth while paying dequant cost only on data consumed by attention.

## Important Performance Fixes

### Capacity-vs-active-grid bug

Launching stage1 over `max_model_len` capacity made short current lengths slow. For example, a 10k decode with `max_model_len=50000` launched many empty groups and lost to FA2.

Fix:

- derive active group count from the current block-table or graph bucket,
- compile/capture bucket-specific grids,
- keep graph replay stable for each bucket.

### Hidden sync

When testing the CUDA extension path, a forced `cudaStreamSynchronize(stream)` after hdim96 forward erased performance. Kernel math was not the only issue; host synchronization can dominate decode loops.

### CUDA graph

For sub-0.1ms decode attention, eager Python launch overhead can hide the real operator speed. Always test both:

- eager CUDA events,
- CUDA graph replay CUDA events.

## Example Results

Representative operator result after active-grid fix:

```text
B=1, H=16, D=96, seqlen=10001, max_model_len=50000, CUDA graph:
FA2:       ~0.088 ms
bitdecode: ~0.040 ms
speedup:   ~2.2x
MAE:       ~0.0018
max_abs:   ~0.0083
```

Representative e2e result:

```text
ShapeOPT 10k decode, max_requests=1, CUDA graph:
FA2 baseline:  ~218 tok/s
bitdecode4:    ~257 tok/s
```

Operator speedups do not transfer 1:1 to e2e because non-attention model work remains.

## E2E Integration Lessons

- Keep prefill FA2 and pack cache after prefill.
- Quantize only decoder self-attention unless cross-attention is explicitly targeted.
- Require `max_requests=1` until batching semantics are implemented.
- Pair baseline and bitdecode per input file for long mesh runs.
- Skip existing output files to allow resuming large 40-file generation runs.
- Align generation settings before comparing outputs:
  - `MAX_SEQ_LEN`,
  - `TRUNCATE_SIZE`,
  - overlap,
  - `IGNORE_EOS`,
  - sampler parameters,
  - seed,
  - model/data path,
  - graph/eager mode.

## Precision Interpretation

With `temperature=1.0`, top-k/top-p sampling, token streams may diverge quickly even if attention errors are small. For precision work:

- first verify FA2-vs-FA2 output parity between runners,
- use greedy/argmax mode for token-level stability,
- use hidden/logit MAE and argmax match for quantized attention validation,
- inspect mesh only after numerical and generation settings are controlled.
