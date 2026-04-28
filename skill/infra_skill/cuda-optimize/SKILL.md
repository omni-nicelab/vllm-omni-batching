---
name: cuda-optimize
description: Orchestrates a full profiling-driven CUDA kernel optimization loop (write → validate → profile → analyze → optimize) until performance converges or no further gains are possible. Capabilities include generating reference implementations, writing initial kernels via cuda-code-gen, running correctness validation and benchmarks via kernel-benchmark, profiling and analyzing NCU reports via ncu-rep-analyze, and applying targeted optimizations via cuda-code-gen. Use when the user wants to optimize a .cu file or CUDA kernel, improve GPU kernel performance, run a CUDA optimization workflow.
---

# CUDA Kernel Optimize

驱动 **生成 → 验证压测 → 评估 → NCU 分析 → 再优化** 的完整循环，直到 kernel 无优化空间或性能收敛。

> ⚠️ **核心约束（必读）**
>
> 本技能是**编排者（Orchestrator）**，负责驱动整个循环直至满足退出条件。
> **每次子技能（kernel-benchmark / ncu-rep-analyze / cuda-code-gen）调用返回后，
> 必须立刻回到主流程继续执行下一步，绝对不能在子技能返回后停止。**
> 子技能的输出是下一步的输入，不是终点。

---

## 执行流程

### 进度追踪

每次循环开始前**在对话中输出**并实时更新如下 checklist：

```
Optimization Loop #N:
- [ ] Step 1: 正确性验证 + 性能压测 (kernel-benchmark)
- [ ] Step 2: 评估退出条件
- [ ] Step 3: NCU Profiling + 瓶颈分析 (ncu-rep-analyze)  ← 仅在 Step 2 判定"继续"时执行
- [ ] Step 4: 实施优化 (cuda-code-gen)  ← 仅在 Step 2 判定"继续"时执行
```

---

### 准备阶段：生成 Reference

**如果用户已提供 reference 文件**：直接使用，跳到"第一版 Kernel"。

**如果用户没有提供 reference**：
1. 根据算法类型在 `kernel/<AlgoName>/` 目录下创建 `<algo_name>_ref.py`
2. Reference 格式（以向量加法为例）：

```python
"""Reference for: solve(const float* A, const float* B, float* C, int N)"""
import torch


def reference(*, A, B, C, N, **kwargs):
    C[:N] = A[:N] + B[:N]
```

规则：
- 文件顶部 docstring 必须写出 `solve(...)` 的完整签名
- 函数名固定为 `reference`，所有参数均为 keyword-only（`*` 后）
- 用 PyTorch tensor 操作实现算法逻辑，结果写入输出 tensor
- 接受 `**kwargs` 以忽略额外参数

---

### 第一版 Kernel

**如果用户已提供 `.cu` 文件**（如 `solution.cu`）：直接使用，立即进入优化循环 Loop #1。

**如果用户没有提供 `.cu` 文件**：
1. 读取 reference 文件，提取算法语义（输入/输出 tensor、维度参数）
2. 调用 **cuda-code-gen 技能**生成第一版 kernel（朴素实现，无复杂优化），保存为：
   - `kernel/<AlgoName>/solution.cu`
3. Kernel 函数签名必须为 `extern "C" void solve(...)` 且与 reference docstring 一致
4. **cuda-code-gen 返回后**，立即进入优化循环 Loop #1。

---

### 优化循环（重复执行，直到退出条件满足）

> **编排说明**：Step 1 ~ Step 4 是一个完整的循环体。每个步骤完成后
> **必须立即执行下一步**，不得因为子技能输出了报告就停在当前步骤。
> 整个循环由本技能（cuda-optimize）全程驱动，子技能仅提供信息输入。

---

#### Step 1：正确性验证 + 性能压测

调用 **kernel-benchmark 技能**，对当前 kernel 文件执行：
- 正确性验证（与 reference 对比）
- 性能压测（记录 Average / Median / Min / Max / Bandwidth，**含与 reference 的 Speedup**）

**kernel-benchmark 返回后**，立即处理结果并决定下一步：

| 结果 | 处理方式 |
|------|---------|
| 正确性验证失败 | 调用 **cuda-code-gen** 修复 bug → 修复完成后**重新从 Step 1 开始** |
| 验证通过 | **立即进入 Step 2** |

---

#### Step 2：评估退出条件

基于 Step 1 的 benchmark 结果，评估以下退出条件：

| 条件 | 说明 |
|------|------|
| ① 性能收敛 | 已连续 **2 轮**优化后性能提升 < 2% |

**如果满足退出条件**：退出循环，输出[最终报告](#最终报告)。

**如果不满足退出条件**：在对话中明确说明"未达到退出条件，继续 Step 3"，**立即进入 Step 3**。

---

#### Step 3：NCU Profiling + 瓶颈分析

调用 **ncu-rep-analyze 技能**，对当前 kernel 文件执行：
- NCU Profiling（生成 `.ncu-rep` 文件）
- 读取并分析报告，输出：
  - 瓶颈类型（DRAM_BOUND / L1_BOUND / LATENCY_BOUND / COMPUTE_BOUND）
  - 优化优先级列表（P0 ~ Pn）

**ncu-rep-analyze 返回后**，立即处理结果：

| 结果 | 处理方式 |
|------|---------|
| NCU Profiling 失败 | **立即停止循环**，输出失败原因，等待用户修复环境后重新触发 |
| NCU 成功 + 分析完成（无 P0/P1 建议） | 退出循环，输出[最终报告](#最终报告)（无优化空间） |
| NCU 成功 + 分析完成（有优化建议） | 提取优化建议，**立即进入 Step 4** |

> ⚠️ NCU 失败时：不得静默跳过，不得复用其他算法或轮次的 `.ncu-rep` 文件。
> ⚠️ **严格约束**：必须使用**当前 kernel 对应的 `.ncu-rep`**，禁止复用其他算法或其他轮次的 NCU 报告。

---

#### Step 4：实施优化

调用 **cuda-code-gen 技能**，基于 Step 3 输出的优化建议（P0 优先），生成新版 kernel：
- 新文件命名带时间戳，**不覆盖原文件**（遵循 cuda-code-gen 技能的命名规则）
- 在文件头注释中说明本轮实施的优化项

**cuda-code-gen 返回后**，将新生成的 `.cu` 文件路径作为下一轮的当前 kernel，
**立即回到 Step 1**，开始新一轮循环（Loop #N+1）。

---

### 最终报告

满足退出条件后，输出完整优化报告：

```markdown
## CUDA Kernel 优化报告

### 算法
<算法名称>

### Reference 文件
`kernel/<AlgoName>/<algo_name>_ref.py`

### 优化历程

| 轮次 | Kernel 文件 | Average (ms) | Speedup vs Ref | 主要优化项 |
|------|------------|-------------|----------------|-----------|
| 初版 | solution.cu | X.XX | 0.XXx | 朴素实现 |
| #1   | solution_opt_<ts1>.cu | X.XX | X.XXx | P0: Shared Memory Tiling |
| #2   | solution_opt_<ts2>.cu | X.XX | X.XXx | P1: Bank Conflict Padding |
| ...  | ... | ... | ... | ... |

### 结论
<退出原因：无优化空间 / 性能收敛>

### 最优 Kernel
`kernel/<AlgoName>/solution_opt_<timestamp>.cu`
- Average: X.XX ms
- Speedup: X.XXx vs reference
```

---

## 参数推断规则

| 参数 | 推断方式 |
|------|---------|
| 算法名称 | 从用户描述或 reference 文件名推断 |
| 维度参数默认值 | 矩阵乘法: M=K=N=4096；向量加法: N=1,000,000；卷积: N=1,000,000 |
| reference 路径 | 用户提供；否则创建 `kernel/<AlgoName>/<algo_name>_ref.py` |
| kernel 输出路径 | `kernel/<AlgoName>/solution.cu`（初版）；后续版本自动带时间戳 |

---

## 技能依赖与调用规范

本技能编排以下三个子技能。**每次调用子技能后，必须立即处理其返回结果并继续主流程，不得停止。**

| 子技能 | 用途 | 调用完成后的动作 |
|--------|------|----------------|
| `cuda-code-gen` | 生成 / 修复 / 优化 kernel 代码 | 记录新文件路径，进入下一步 |
| `kernel-benchmark` | 正确性验证 + 性能压测 | 记录 benchmark 数据，进入 Step 2 |
| `ncu-rep-analyze` | NCU Profiling + 解读报告 + 输出优化建议 | 提取优化建议，**立即进入 Step 4** |

> **常见中断陷阱**：`ncu-rep-analyze` 会输出格式化的分析报告。
> 这个报告是给 Step 4 (cuda-code-gen) 使用的输入，**不是优化循环的终点**。
> 收到分析报告后必须立即执行 Step 4 实施优化，不得等待用户指令。
