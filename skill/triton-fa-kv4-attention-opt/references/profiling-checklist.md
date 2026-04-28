# Profiling Checklist

## Operator Benchmark

Use one script that reports all of these:

- exact shape and dtype,
- FA2 baseline latency,
- fused attention latency,
- speedup,
- finite check,
- MAE and max_abs against reference,
- eager CUDA-event timing,
- CUDA graph replay timing.

Use graph replay when the kernel is sub-millisecond:

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = fn()
for _ in range(warmup):
    graph.replay()
start.record()
for _ in range(iters):
    graph.replay()
end.record()
```

## Stage Timing

If the fused path is split-K:

- time stage1 alone,
- time stage2 alone,
- time residual attention,
- time final merge.

Act on the largest component only. In the ShapeOPT hdim96 work, stage1 dominated; stage2 optimization was not the main path.

## NCU When Available

Run NCU after correctness passes. If `ERR_NVGPUCTRPERM` blocks metrics, fall back to CUDA events and kernel decomposition.

Prioritize:

- DRAM bytes and bandwidth,
- instruction mix,
- warp stall reasons,
- occupancy,
- register pressure,
- memory load efficiency,
- branch/predicate overhead.

## Accuracy Checks

For operator-level checks:

- compare to FA2 using dequantized dense K/V or to a PyTorch unpack/dequant reference,
- use same `causal`, scale, shape, and cache length,
- report finite, mean absolute error, and max absolute error.

For e2e checks:

- first compare FA2 baseline against old runner FA2 baseline,
- keep truncation and EOS identical,
- avoid stochastic token-stream comparisons as the only metric.

## Common Regression Tests

Run these before claiming a result:

- `seqlen=129` or another non-multiple of group size,
- exactly full group boundary, e.g. `128`, `256`,
- `seqlen=10001`,
- `max_model_len` much larger than `seqlen`,
- CUDA graph replay,
- e2e generation with resume/skip behavior.

## Decision Table

| Observation | Likely cause | Next action |
|---|---|---|
| Eager slow, graph fast | launch overhead | integrate CUDA graph before judging |
| Long `max_model_len` slows short decode | grid uses capacity | bucket active groups |
| Stage2 dominates | too many splits or large partials | reduce split count or compress partials |
| Stage1 dominates | dequant/QK/PV instruction pressure | optimize pack granularity, dequant math, group shape |
| FA2 baseline differs across repos | generation/input mismatch | align truncation, EOS, model/data, sampler |
| Mesh diverges after few tokens | stochastic sampling amplified error | compare logits/hidden or use greedy |
