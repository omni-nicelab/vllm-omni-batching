---
name: cuda-code-gen
description: Generates optimized CUDA kernel code based on performance analysis reports or algorithm type. Reads NCU analysis reports (e.g. *_analysis.md) and optionally existing kernel code, then produces high-quality compilable .cu files with applied optimizations. Use when the user provides an NCU analysis report or requests CUDA kernel generation, optimization, or implementation of techniques like Shared Memory Tiling, vectorized loads, bank conflict elimination, or double buffering. Does not handle compilation, execution, or profiling.
---

# CUDA Code Gen

读取性能分析报告，生成优化后的 CUDA kernel 代码（`.cu` 文件）。**不负责编译、运行、profiling。**

## 执行流程

### 进度追踪

```
Task Progress:
- [ ] Step 1: 需求解析
- [ ] Step 2: 读取输入文件
- [ ] Step 3: 确定优化策略
- [ ] Step 4: 生成代码并写入文件
```

---

### Step 1: 需求解析

确认以下信息（可从报告或用户输入获取）：

| 项目 | 来源 |
|------|------|
| 算法类型（MatMul / Reduction / Convolution / 自定义） | 报告 Kernel 名称 |
| 数据规模（M, N, K 或其他维度） | 用户提供 |
| 精度（FP32 / FP16 / INT8） | 报告或用户 |
| GPU 架构（sm_XX） | 报告 Device CC |

---

### Step 2: 读取输入文件

1. **读取 NCU 分析报告**（`*_analysis.md`，必须）
   - 提取瓶颈类型（DRAM_MEMORY_BOUND / L1_PRESSURE_BOUND / LATENCY_BOUND / COMPUTE_BOUND / OCCUPANCY_BOUND）
   - 提取优化优先级列表（P0~Pn）
   - 提取关键指标（SM Busy、DRAM Throughput、Warp Cycles 等）

2. **读取现有 kernel 代码**（若用户提供路径）
   - 识别当前线程块配置、内存访问模式、计算逻辑
   - 保留 host 端 API 接口不变（参数列表、函数名）

若**无分析报告**，按算法类型选默认策略，见 [cuda-optimization-strategies.md](cuda-optimization-strategies.md)。

---

### Step 3: 确定优化策略

按报告中**瓶颈类型**从 [cuda-optimization-strategies.md](cuda-optimization-strategies.md) 中查找对应策略组合，再按 P0 → Pn 优先级依次实施。

各瓶颈类型对应策略速查（详细说明和代码模板见参考文档）：

| 瓶颈类型 | 条件特征 | 优先策略 |
|---------|---------|---------|
| DRAM_MEMORY_BOUND | DRAM > 70%，SM < 30% | Block Tiling → Vectorized Load → Prefetching |
| L1_PRESSURE_BOUND | L1 > 80%，DRAM < 30% | Shared Memory Tiling → Padding → Data Transpose |
| LATENCY_BOUND | SM Busy < 50%，Occupancy > 60% | Double Buffering → ILP → Loop Unrolling |
| COMPUTE_BOUND | SM > 60%，SM Busy > 80% | FMA → FP16/TF32 → Tensor Core |
| OCCUPANCY_BOUND | Occupancy < 30%，SM Busy > 70% | 调整 Block Size → `__launch_bounds__` → 减少 smem |

---

### Step 4: 生成代码并写入文件

#### 输出文件路径

1. 执行 `date +%Y%m%d_%H%M%S` 获取时间戳（如 `20260316_153045`）
2. 基于原 `.cu` 文件路径拼接时间戳后缀，**始终创建新文件，不覆盖原文件**：
   - `kernel/MatrixMultiplication/solution.cu` → `kernel/MatrixMultiplication/solution_opt_20260316_153045.cu`
   - 若用户未提供原路径，默认在 kernel 同目录下命名为 `solution_opt_<timestamp>.cu`

#### 文件头注释模板

```cuda
/*
 * Optimized CUDA Kernel - <算法名称>
 *
 * 生成时间：<YYYY-MM-DD HH:MM:SS>
 *
 * 优化措施（来自 NCU 分析报告）：
 *   [P0] Shared Memory Tiling (TILE_SIZE=16)
 *   [P1] Shared Memory Padding (+1 消除 Bank Conflict)
 *   [P2] Vectorized Load (float4)
 *
 * 编译命令：
 *   nvcc -O3 -arch=sm_89 -o kernel solution_opt_<timestamp>.cu
 *
 * 目标设备：Ada Lovelace (sm_89)
 * 精度：FP32
 */
```

#### 代码质量要求

- **边界检查**：K 非 TILE_SIZE 整数倍时必须处理尾 Tile 越界
- **架构兼容性**：`cp.async` / `__pipeline` 仅支持 sm_80+（Ampere）及以上
- **函数签名不变**：host 调用接口保持与原 kernel 一致

---

## 参考资料

- 优化策略详细说明与代码模板 → [cuda-optimization-strategies.md](cuda-optimization-strategies.md)
