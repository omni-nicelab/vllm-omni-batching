---
name: kernel-benchmark
description: Compiles, validates, and benchmarks a CUDA kernel (.cu file) against a Python reference (*_ref.py). Auto-detects GPU arch and infers dimension args from the `extern "C" void solve(...)` signature. Runs benchmark.py to: compile with nvcc, optionally validate outputs (exits on failure), benchmark both kernel and reference with CUDA-event timing, and print a latency/bandwidth/speedup summary. Use when the user wants to validate or benchmark a .cu file.
---

# Kernel Benchmark

对 CUDA kernel 执行正确性验证 + 压测，最后汇总结果。

## 执行流程

所有命令均在项目**根目录**中执行。

### 进度追踪

复制如下 checklist 并实时更新：

```
Task Progress:
- [ ] Step 1: 正确性验证 + 性能压测 (benchmark.py)
- [ ] Step 2: 汇总输出结果
```

---

### Step 1: 正确性验证 + 性能压测

`benchmark.py` 在有 `--ref` 时会先做正确性校验，**不通过则直接退出（exit code 非 0）**，通过后再分别对 ref 和 kernel 压测并打印汇总。

```bash
python3 skills/kernel-benchmark/scripts/benchmark.py <cu_file> \
    --ref=<ref_file> [--PARAM=VALUE ...] --repeat=20
```

**示例**（矩阵转置）：

```bash
python3 skills/kernel-benchmark/scripts/benchmark.py kernel/MatrixTranspose/solution.cu \
    --ref=kernel/MatrixTranspose/transpose_ref.py --M=10000 --N=1000 --repeat=20
```

- 若校验**失败**（exit code 非 0 或输出含 `FAIL`），停止后续步骤，将错误信息反馈给用户。
- 若校验**通过**（`ALL PASS ✓`），继续 Step 2。

---

## 参数推断规则

| 参数 | 推断方式 |
|------|---------|
| `<cu_file>` | 用户提供的 .cu 文件路径 |
| `<ref_file>` | 用户提供；若未指定，在 `.cu` 文件同级目录下查找 `*_ref.py`（如 `matmul_ref.py`、`vector_add_ref.py`、`transpose_ref.py`） |
| 维度参数 (`--M`, `--N`, `--K` 等) | 从 `extern "C" void solve(...)` 签名推断参数名；未指定时用合理默认值（矩阵乘法 M=K=N=4096，向量加法 N=1000000） |
| `--repeat` | 默认 20 |

---

## Step 2: 汇总输出

```markdown
## Kernel 验证报告

### 基本信息
- **Kernel 文件**: `<cu_file>`
- **参考实现**: `<ref_file>`
- **维度参数**: M=..., N=... (等)
- **GPU**: <GPU name>

### 1. 正确性验证
- **结果**: ✅ ALL PASS / ❌ FAILED
- （如失败，附上错误信息）

### 2. 性能压测
| 指标 | Kernel | Reference |
|------|--------|-----------|
| Average | X.XXXX ms | X.XXXX ms |
| Median  | X.XXXX ms | X.XXXX ms |
| Min     | X.XXXX ms | X.XXXX ms |
| Max     | X.XXXX ms | X.XXXX ms |
| ~Bandwidth | XX.XX GB/s | XX.XX GB/s |
| Speedup | X.XXx vs ref | — |
```
