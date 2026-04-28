---
name: ncu-rep-analyze
description: Profiles a CUDA kernel with NCU and analyzes the resulting .ncu-rep report to diagnose performance bottlenecks and generate optimization recommendations. Use when the user provides a .cu file or a .ncu-rep file and asks for performance analysis, NCU profiling, or bottleneck diagnosis. If given a .cu file, runs NCU via benchmark.py to produce a .ncu-rep, then imports it with `ncu --import` to extract metrics (SM throughput, DRAM/L1 bandwidth, occupancy), classifies the bottleneck (DRAM-bound, compute-bound, latency-bound, etc.), and saves a structured *_analysis.md report alongside the .ncu-rep file.
---

# NCU Profiling 与性能分析

对 CUDA kernel 执行 NCU profiling，并分析 `.ncu-rep` 报告，诊断性能瓶颈并给出优化建议。

---

## 工作流程

### 进度追踪

复制如下 checklist 并实时更新：

```
Task Progress:
- [ ] Step 1: 定位文件
- [ ] Step 2: NCU Profiling（生成 .ncu-rep）
- [ ] Step 3: 读取报告摘要
- [ ] Step 4: 自动诊断瓶颈
- [ ] Step 5: 生成分析报告
```

---

### Step 1: 定位文件

**若用户提供 `.ncu-rep` 文件**：直接进入 Step 3，跳过 Step 2。

**若用户提供 `.cu` 文件**（或需重新 profiling）：确认以下信息后执行 Step 2。

| 信息 | 推断方式 |
|------|---------|
| `<cu_file>` | 用户提供的 `.cu` 文件路径 |
| 维度参数 (`--M`, `--N`, `--K` 等) | 从 `extern "C" void solve(...)` 签名推断参数名；未指定时用合理默认值（矩阵乘法 M=K=N=4096，向量加法 N=1000000） |
| `{kernel_stem}` | `.cu` 文件不含扩展名的文件名 |
| `{kernel_dir}` | `.cu` 文件所在目录 |

若用户未指定 `.ncu-rep` 且也未指定 `.cu`，在当前目录及常见子目录搜索：

```bash
find . -name "*.ncu-rep" 2>/dev/null
```

---

### Step 2: NCU Profiling

输出文件名直接使用 `{kernel_stem}`，保存到 `.cu` 文件所在的 `{kernel_dir}` 目录下，不追加时间戳。

```bash
ncu --target-processes all \
    --profile-from-start on \
    --launch-skip 20 \
    --launch-count 1 \
    --set full \
    -o {kernel_dir}/{kernel_stem} -f \
    python3 skills/kernel-benchmark/scripts/benchmark.py <cu_file> \
    [--PARAM=VALUE ...] --repeat=22
```

**示例**（文件名为 `solution.cu`）：

```bash
ncu --target-processes all \
    --profile-from-start on \
    --launch-skip 20 \
    --launch-count 1 \
    --set full \
    -o kernel/MatrixTranspose/solution -f \
    python3 skills/kernel-benchmark/scripts/benchmark.py kernel/MatrixTranspose/solution.cu \
    --M=10000 --N=1000 --repeat=22
```

> `--launch-skip 20` 跳过 warmup，`--launch-count 1` 只采集第 1 个正式迭代。
> `--repeat` 需 ≥ `launch-skip + launch-count + 1`，推荐用 22。

#### NCU 执行失败的处理

若 `ncu` 失败（权限拒绝、命令不存在、沙箱拦截等），**必须**：

1. 输出明确失败原因
2. 在最终报告中标注：`NCU Profiling 状态: ❌ FAILED: <原因>`
3. 不得静默跳过，不得用其他算法的 `.ncu-rep` 文件替代
4. 输出手动修复建议：

```bash
# 修复权限问题：
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

# 重新执行 NCU Profiling（命令同上）
```

---

### Step 3: 读取报告摘要

```bash
ncu --import <file.ncu-rep> --print-summary per-kernel
```

---

### Step 4: 自动诊断瓶颈

根据提取到的指标，按以下逻辑判断主要瓶颈：

```
roofline  = sm__throughput %
dram      = gpu__dram_throughput %
l1tex     = l1tex__throughput %
sm_busy   = sm__cycles_active %
occupancy = sm__warps_active %

IF sm_throughput < 30:
    IF dram > 70:       → DRAM_MEMORY_BOUND
    ELIF l1tex > 80 AND dram < 30: → L1_PRESSURE_BOUND
    ELSE:               → LATENCY_BOUND
ELIF sm_throughput > 60:
    IF sm_busy > 80:    → COMPUTE_BOUND
    ELSE:               → OCCUPANCY_BOUND
ELSE:                   → MIXED_BOUND
```

---

### Step 5: 生成分析报告

按照下方模板输出分析结果，自动保存分析报告到与 `.ncu-rep` **相同的 kernel 目录**下：

```
project_root/
├── kernel/
│   └── <AlgoName>/
│       ├── solution.cu
│       ├── <kernel_stem>.ncu-rep         # NCU 报告
│       └── <kernel_stem>_analysis.md    # AI 分析报告
```

---

## 输出模板

```markdown
# NCU 性能分析报告

## 报告信息
- **文件**: {file.ncu-rep}
- **Kernel**: {kernel_name}
- **分析时间**: {timestamp}

## 执行摘要

| 项目 | 数值 |
|------|------|
| **主要瓶颈** | {bottleneck_type} |
| **置信度** | {confidence} |
| **优化潜力** | {potential}x |

## 关键指标

### 性能指标
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| sm_throughput | {sm_throughput}% | > 60% | {status} |
| SM Busy | {sm_busy}% | > 70% | {status} |
| Occupancy | {occupancy}% | > 50% | {status} |

### 内存指标
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| DRAM Throughput | {dram}% | < 50% | {status} |
| L1/TEX Throughput | {l1tex}% | < 80% | {status} |
| L2 Throughput | {l2}% | < 80% | {status} |

## 诊断详情

**瓶颈类型**: {bottleneck_type}

**判断依据**:
- {reason_1}
- {reason_2}

## 优化建议

### 高优先级
{high_priority_suggestions}

### 验证清单
- [ ] 实施优化建议
- [ ] 重新运行 NCU 采集（Step 2）
- [ ] 对比优化前后指标
```

---

## 瓶颈诊断与优化策略

### DRAM_MEMORY_BOUND

```
IF dram_throughput > 70% AND sm_throughput < 30%:
    诊断: DRAM_MEMORY_BOUND (置信度: HIGH)
    
    优化策略:
    1. Block Tiling (共享内存缓存)
    2. Vectorized Load (float4)
    3. Prefetching (数据预取)
```

### L1_PRESSURE_BOUND

```
IF l1tex_throughput > 80% AND dram_throughput < 30%:
    诊断: L1_PRESSURE_BOUND (置信度: HIGH)
    
    优化策略:
    1. Shared Memory Padding
    2. Data Transpose
    3. Fragment Caching
```

### LATENCY_BOUND

```
IF sm_busy < 50% AND occupancy > 60%:
    诊断: LATENCY_BOUND (置信度: HIGH)
    
    优化策略:
    1. Double Buffering
    2. Instruction-level Parallelism
    3. Loop Unrolling
```

### COMPUTE_BOUND

```
IF sm_throughput > 60% AND sm_busy > 80%:
    诊断: COMPUTE_BOUND (置信度: HIGH)
    
    优化策略:
    1. Use FMA instructions
    2. Reduce precision (FP32 -> FP16/TF32)
    3. Tensor Cores
```

### OCCUPANCY_BOUND

```
IF occupancy < 30% AND sm_busy > 70%:
    诊断: OCCUPANCY_BOUND (置信度: HIGH)
    
    优化策略:
    1. Reduce register usage
    2. Adjust block size
    3. Use __launch_bounds__
```

---

## 常见误区

1. **高 Throughput ≠ 高效率** — Compute + Memory Throughput 都高但 sm_throughput 低，说明 GPU 在"忙碌地等待"
2. **DRAM Throughput 低可能是好事** — 说明数据在缓存中复用，这是优化成功的信号
3. **Occupancy 不是越高越好** — 目标是最小足够 occupancy 来隐藏延迟
