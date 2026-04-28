---
name: triton-fa-kv4-attention-opt
description: Optimize Triton or CUDA FlashAttention-style decode kernels with fused 4-bit KV cache dequantization. Use when working on FA2/FA3 baselines, q_len=1 decode, KV4/int4/NVFP4 cache layouts, ShapeOPT head_dim=96, split-K/online-softmax attention, CUDA graph capture, vLLM/nano-vLLM e2e mesh generation, or performance/accuracy comparisons against FlashAttention.
---

# Triton FA KV4 Attention Optimization

Use this skill to turn a fused KV4 attention idea into a measured, e2e-safe implementation. The default target is decode attention with pre-existing long KV cache, especially `B=1`, `q_len=1`, `H=16`, `D=96`, 4-bit KV cache, and FA2 as baseline.

## Core Rule

Do not claim success from a standalone kernel until all three layers pass:

1. **Operator correctness**: compare against the matching FA2 path with the same shape, causal setting, cache length, dtype, and layout.
2. **Operator performance**: use CUDA events or CUDA graph replay; isolate kernel time from Python launch overhead.
3. **E2E behavior**: run the real serving path with the same generation settings, CUDA graph policy, truncation/windowing, EOS behavior, and output checks.

## Workflow

1. **Freeze the baseline**
   - Record exact shape: batch, heads, KV heads, head_dim, q_len, kv_len, dtype, group size, residual length.
   - Record exact baseline API: `flash_attn_with_kvcache`, varlen FA2/FA3, paged cache, block table, or dense cache.
   - For generation, record `max_seq_len`, `truncate_size`, overlap, `ignore_eos`, sampler mode, temperature, top-k/top-p, seed, model path, data path, and CUDA graph/eager mode.

2. **Start with the simplest fused kernel**
   - Keep dequant, QK, online softmax, and PV inside one logical operator.
   - Avoid `dequant -> write full fp16 KV -> FA2`; this adds full KV write/read traffic and normally cannot beat FA2.
   - Use a direct PyTorch unpack/dequant reference for early correctness.

3. **Add parallelism before micro-optimizing**
   - A q_len=1 kernel that lets one program stream the entire 10k KV cache is usually under-parallelized.
   - Prefer split-K: stage1 per `(head, KV block)` produces local numerator/max/sum, stage2 merges with online-softmax math.
   - Time stage1 and stage2 separately. If stage2 is tiny, keep optimizing stage1.

4. **Optimize memory and instruction shape**
   - Decode packed bytes/words once, not once per logical element.
   - Load scales at their natural granularity, but benchmark deduplication; broadcasts can cost more than saved memory.
   - Store raw numerator, `m`, and `l` from stage1; defer division/log to stage2.
   - Use `exp2` with log2-scaled softmax scale when it matches the local style.
   - Reduce partial workspace precision only after measuring accuracy impact.

5. **Make graph/capacity explicit**
   - CUDA graph can flip the result when Python launch overhead is comparable to kernel time.
   - Do not launch over max cache capacity when the current graph bucket is smaller. Tie stage1 grid to active block bucket or active group count.
   - Capture separate graph buckets when sequence length varies widely.

6. **Integrate cautiously**
   - Prefill should usually remain FA2/FA3, then pack K/V into quantized cache.
   - Decode writes the current token to residual, packs residual when a group fills, runs quantized prefix attention, runs residual attention, then merges prefix/residual LSE.
   - Keep cross-attention on the original dense FA path unless explicitly optimizing it.

7. **Validate e2e without sampling noise**
   - First align FA2 baseline between old and new runners.
   - Only compare bitdecode-vs-FA2 token streams after generation settings match.
   - For stochastic top-k/top-p sampling, token/mesh divergence after a few tokens is expected; use hidden/logit MAE, argmax match, or greedy runs for precision diagnosis.

## Profiling Signals

Use CUDA events first. Use NCU when available, but be ready for `ERR_NVGPUCTRPERM` on restricted systems.

Prefer these measurements:

- FA2 baseline latency for the exact decode shape.
- Fused kernel eager latency.
- Fused kernel CUDA graph replay latency.
- Stage1-only and stage2-only latency for split-K kernels.
- Accuracy: finite check, MAE/max_abs against FA2 or unpacked PyTorch reference.
- E2E throughput and output count under the real serving script.

Interpret common outcomes:

- **Graph speedup huge, eager not faster**: Python launch overhead dominates; integrate with CUDA graph.
- **Stage1 dominates**: dequant/QK/PV work is the bottleneck; stage2 merge is not the right target.
- **Large `max_model_len` slows short decode**: launch grid likely follows capacity instead of active blocks.
- **Operator fast, e2e modest**: attention is no longer the dominant end-to-end component.

## Known Pitfalls

- Triton q_len=1 often cannot use `tl.dot` tensor-core paths directly because dot dimensions may need to be at least 16. Repeating query rows can compile but may not win.
- Software NVFP4/E2M1 decode in Triton can be instruction-heavy on SM90; int4 affine layouts may be faster for this workload.
- `.venv` Triton versions may reject global `tl.constexpr` annotations. Use literals or pass constants as constexpr arguments.
- A hidden `cudaStreamSynchronize` inside an extension forward path can destroy decode-loop performance.
- Comparing mesh output is invalid if truncation, EOS, random sampling, or input ordering differ.
- Existing outputs should be skipped in long e2e runs; paired baseline/bitdecode should run per file to keep comparisons local.

## References

- For the concrete ShapeOPT hdim96 path, implementation pivots, and measured numbers, read [references/shapeopt-hdim96-case-study.md](references/shapeopt-hdim96-case-study.md).
- For a compact benchmark/profiling checklist, read [references/profiling-checklist.md](references/profiling-checklist.md).
