# CUDA 优化策略参考

当无 NCU 分析报告时，按算法类型选择默认优化策略。

## 按算法类型的默认策略

### MatMul（矩阵乘法）

**默认优化组合**：P0 + P1 + P2

| 参数 | 推荐值 |
|------|--------|
| TILE_SIZE | 16（保守）或 32（激进，需 sm_80+） |
| Block | (TILE_SIZE, TILE_SIZE, 1) |
| Grid | (ceil(N/TILE), ceil(M/TILE), 1) |
| 共享内存 | 2 × TILE × (TILE+1) × sizeof(T) |

**瓶颈特征**：L1_PRESSURE_BOUND → 首选 Shared Memory Tiling

---

### Reduction（规约）

**默认优化组合**：Warp Shuffle + 多轮规约

| 参数 | 推荐值 |
|------|--------|
| Block | (256, 1, 1) |
| Grid | (ceil(N/Block.x/2), 1, 1) |

**核心模式**：
```cuda
// Warp 级规约（无需 __syncthreads）
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);

// Block 级规约（通过共享内存）
__shared__ float smem[32];  // 每个 warp 写一个值
if (lane == 0) smem[wid] = val;
__syncthreads();
```

**瓶颈特征**：MEMORY_BOUND → 减少 GMEM 访问轮次

---

### 1D Convolution（一维卷积）

**默认优化组合**：P0（Shared Memory Halo）+ P2（Vectorized Load）

| 参数 | 推荐值 |
|------|--------|
| Block | (256, 1, 1) |
| 共享内存 | (BLOCK + 2*RADIUS) × sizeof(float) |

**核心模式**：
```cuda
#define RADIUS 3  // 卷积半径（= (FILTER_SIZE-1)/2）

__shared__ float smem[BLOCK_SIZE + 2 * RADIUS];

// 加载 halo 区域
int gx = blockIdx.x * BLOCK_SIZE + threadIdx.x - RADIUS;
smem[threadIdx.x] = (gx >= 0 && gx < N) ? input[gx] : 0.0f;
// 加载右侧 halo（线程尾部处理）
if (threadIdx.x < 2 * RADIUS) {
    int gx2 = blockIdx.x * BLOCK_SIZE + BLOCK_SIZE + threadIdx.x - RADIUS;
    smem[BLOCK_SIZE + threadIdx.x] = (gx2 < N) ? input[gx2] : 0.0f;
}
__syncthreads();

float result = 0.0f;
for (int r = -RADIUS; r <= RADIUS; r++)
    result += smem[threadIdx.x + RADIUS + r] * filter[r + RADIUS];
```

---

### 通用 Element-wise Kernel

**默认优化组合**：Grid-Stride Loop + Vectorized Load

```cuda
__global__ void elementwise(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-Stride Loop：一个 thread 处理多个元素
    for (int i = idx; i < N / 4; i += stride) {
        float4 a4 = reinterpret_cast<const float4*>(A)[i];
        float4 b4 = reinterpret_cast<const float4*>(B)[i];
        float4 c4 = {a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w};
        reinterpret_cast<float4*>(C)[i] = c4;
    }
    // 处理尾部（N 非 4 整除时）
    for (int i = (N / 4) * 4 + idx; i < N; i += stride)
        C[i] = A[i] + B[i];
}
```

---

## 优化措施详细说明

### 线程块配置参考

| 场景 | blockDim | 说明 |
|------|---------|------|
| 2D 矩阵算法 | `(16, 16, 1)` 或 `(32, 32, 1)` | TILE 对齐，sm_80+ 可用 32 |
| Reduction / 1D | `(256, 1, 1)` | 通用，利于 Warp Shuffle |
| 自定义 | `(128, 1, 1)` | 保守选择 |

> blockDim.x 必须为 32 的倍数（warp 对齐）。

---

### Shared Memory Tiling

**目的**：减少 L1/GMEM 冗余加载，每个数据块仅从全局内存加载一次。

**适用条件**：
- `L1/TEX Cache Throughput > 85%`
- `DRAM Throughput < 30%`（数据集中在 L1 层面反复访问）
- 算法存在数据复用（矩阵乘、卷积等）

**TILE_SIZE 选择**：
| sm_XX | 推荐 TILE_SIZE | 共享内存消耗 |
|-------|---------------|-------------|
| sm_70~sm_75 | 16 | 2KB per tile |
| sm_80~sm_89 | 16 或 32 | 2KB 或 8KB |
| sm_90 | 32 | 8KB per tile |

**完整 MatMul Tiling 模板**：
```cuda
#define TILE_SIZE 16  // 可调：16 或 32

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 消除 Bank Conflict
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 边界安全加载（尾 Tile 越界补 0）
        As[threadIdx.y][threadIdx.x] = (row < M && t * TILE_SIZE + threadIdx.x < K)
            ? A[row * K + t * TILE_SIZE + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (t * TILE_SIZE + threadIdx.y < K && col < N)
            ? B[(t * TILE_SIZE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Host 端启动
dim3 blockDim(TILE_SIZE, TILE_SIZE);
dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
matmul_tiled<<<gridDim, blockDim>>>(A, B, C, M, N, K);
```

---

### Bank Conflict 消除（Padding）

**原理**：32 个 bank，步长为 4 字节。当 TILE_SIZE 为 32 时，每列元素落在同一 bank → 32-way conflict。

**解决**：在列维度加 1 元素 padding：
```cuda
__shared__ float As[TILE][TILE + 1];  // TILE+1 打破对齐
```

**不需要 padding 的情况**：TILE_SIZE = 16 时 bank conflict 较轻（最多 2-way），可酌情省略。

---

### Vectorized Load（float4）

**要求**：
1. 全局内存地址 **16 字节对齐**（`cudaMalloc` 默认满足）
2. 数据长度为 4 的倍数（尾部单独处理）

**收益**：将 4 条 Load 指令合并为 1 条，降低指令发射压力，提升 L1/L2 带宽利用率。

**Half precision 版本**：
```cuda
half2 h2 = reinterpret_cast<const half2*>(A)[idx / 2];
```

---

### Double Buffering / cp.async（sm_80+）

**目的**：计算当前 Tile 的同时异步预取下一个 Tile，隐藏 GMEM 延迟。

**适用条件**：
- GPU 架构 >= sm_80（Ampere）
- `Warp Cycles Per Issued > 20`（延迟未被隐藏）
- 已完成 Tiling 优化后仍存在 LATENCY_BOUND

**模板**（需 `#include <cuda/pipeline>`）：
```cuda
__shared__ float As[2][TILE][TILE + 1];
__shared__ float Bs[2][TILE][TILE + 1];

cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

// 预加载第 0 个 Tile
cuda::memcpy_async(As[0][threadIdx.y] + threadIdx.x,
                   A + row * K + 0 * TILE + threadIdx.x,
                   sizeof(float), pipe);
pipe.producer_commit();

for (int t = 0; t < numTiles; t++) {
    int next = (t + 1) % 2;
    // 异步加载下一个 Tile
    if (t + 1 < numTiles) {
        cuda::memcpy_async(As[next][threadIdx.y] + threadIdx.x,
                           A + row * K + (t+1) * TILE + threadIdx.x,
                           sizeof(float), pipe);
        pipe.producer_commit();
    }
    // 等待当前 Tile 就绪
    pipe.consumer_wait();
    __syncthreads();

    // 使用当前 Tile 计算
    #pragma unroll
    for (int k = 0; k < TILE; k++)
        sum += As[t % 2][threadIdx.y][k] * Bs[t % 2][k][threadIdx.x];

    pipe.consumer_release();
    __syncthreads();
}
```

---

### `__launch_bounds__` 调优

**目的**：提示编译器控制寄存器分配，避免 spilling 或 occupancy 下降。

```cuda
// maxThreadsPerBlock: 与启动 blockDim 一致
// minBlocksPerMultiprocessor: 目标 occupancy 对应的最小 block 数
__global__ __launch_bounds__(256, 4)
void kernel(...) { ... }
```

**参考对照表**（sm_89，每 SM 最大 1536 threads）：
| maxThreads | minBlocks | 最大寄存器/线程 |
|-----------|-----------|----------------|
| 256 | 6 | 42 |
| 256 | 4 | 64 |
| 256 | 2 | 128 |

---

### Prefetching（数据预取，sm_80+）

**目的**：DRAM_MEMORY_BOUND 场景下，利用 `__builtin_assume_aligned` 和 `prefetch` 提前发出加载请求，隐藏 DRAM 延迟。

**适用条件**：
- `DRAM Throughput > 70%`
- GPU 架构 >= sm_80（Ampere）支持 `cp.async`

**模板**：
```cuda
// 方式一：使用 __ldg 只读缓存（sm_35+）
float val = __ldg(&A[idx]);  // 走 texture/read-only cache，减少 L1 压力

// 方式二：cp.async 异步预取（sm_80+，配合 Double Buffering 使用）
// 见 Double Buffering 章节

// 方式三：prefetch 指令（适合 stride 访问）
asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr + prefetch_offset));
```

---

### Data Transpose（访问模式调整）

**目的**：L1_PRESSURE_BOUND + 列访问场景下，将非合并的列访问转换为合并的行访问，消除 L1 带宽浪费。

**适用条件**：
- Kernel 存在 `A[col * M + row]` 形式的列优先访问
- `L1/TEX Throughput > 80%`，且 L2 Hit Rate 低

**模板**（Shared Memory 中转实现转置）：
```cuda
// 将 A 的一个 Tile 先合并读入共享内存，再转置后写出
__shared__ float tile[TILE][TILE + 1];  // +1 消除 Bank Conflict

// 合并读（行访问）
tile[threadIdx.y][threadIdx.x] = A[(blockIdx.y * TILE + threadIdx.y) * N
                                   + blockIdx.x * TILE + threadIdx.x];
__syncthreads();

// 转置写（转为列写，对 B 矩阵行访问合并）
B[(blockIdx.x * TILE + threadIdx.y) * M
  + blockIdx.y * TILE + threadIdx.x] = tile[threadIdx.x][threadIdx.y];
```

---

### Fragment Caching（寄存器级缓存）

**目的**：L1_PRESSURE_BOUND + 数据规模小时，将频繁访问的小数组（如卷积权重、LUT 表）放入寄存器数组，彻底绕开 L1。

**适用条件**：
- 数据大小 <= 每线程可用寄存器（通常 < 256 字节）
- 同一线程在循环中多次读取固定索引的数据

**模板**：
```cuda
// 将权重 / 小型 LUT 加载进寄存器数组（不走 L1）
float reg_filter[FILTER_SIZE];
#pragma unroll
for (int i = 0; i < FILTER_SIZE; i++)
    reg_filter[i] = filter[i];  // 只加载一次

// 循环内直接使用寄存器，不再访问全局/共享内存
for (int i = 0; i < N; i++)
    output[i] += input[i] * reg_filter[i % FILTER_SIZE];
```

---

### ILP（指令级并行）

**目的**：LATENCY_BOUND 场景下，每个线程同时处理多个独立元素，使调度器在等待一组内存操作时可以发射另一组的计算指令。

**适用条件**：
- `Warp Cycles Per Issued > 15`，但 Occupancy 无法再提升
- 寄存器充足（不会导致 spilling）

**模板**（每线程处理 4 个元素，ILP=4）：
```cuda
// 每线程负责连续 ILP 个元素
int base = (blockIdx.x * blockDim.x + threadIdx.x) * ILP;

float a0 = A[base + 0], a1 = A[base + 1];
float a2 = A[base + 2], a3 = A[base + 3];
float b0 = B[base + 0], b1 = B[base + 1];
float b2 = B[base + 2], b3 = B[base + 3];

// 独立计算，调度器可流水线执行
C[base + 0] = a0 + b0;
C[base + 1] = a1 + b1;
C[base + 2] = a2 + b2;
C[base + 3] = a3 + b3;
```

---

### Loop Unrolling（循环展开）

**目的**：LATENCY_BOUND 场景下，减少循环控制开销，为编译器和调度器提供更多指令级并行机会。

**模板**：
```cuda
// 静态展开（循环次数编译期已知）
#pragma unroll
for (int k = 0; k < TILE_SIZE; k++)
    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

// 部分展开（循环次数未知，展开 N 次）
#pragma unroll 4
for (int i = 0; i < N; i++)
    sum += A[i] * B[i];
```

**注意**：过度展开（unroll factor 过大）会增加寄存器压力，需结合 `__launch_bounds__` 控制。

---

### FP16 / TF32 / Tensor Core（COMPUTE_BOUND）

**目的**：COMPUTE_BOUND 场景下，通过降低精度或使用专用矩阵乘法硬件单元（Tensor Core）大幅提升算术吞吐。

**适用条件**：
- `SM Busy > 80%`，计算密集
- sm_70+（Volta）支持 FP16 Tensor Core；sm_80+（Ampere）支持 TF32

**FP16 Tensor Core 模板（使用 WMMA API）**：
```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// Fragment 大小：16×16×16
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

fill_fragment(c_frag, 0.0f);
load_matrix_sync(a_frag, A + warp_row * 16 * K, K);
load_matrix_sync(b_frag, B + warp_col * 16,     N);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(C + warp_row * 16 * N + warp_col * 16, c_frag, N, mem_row_major);
```

**TF32（Ampere，无需修改数据类型，仅启用编译选项）**：
```bash
nvcc -O3 -arch=sm_80 --use_fast_math -o kernel kernel.cu
# 或在代码中使用 cublasMath_t CUBLAS_TF32_TENSOR_OP_MATH
```

---

### Block Size 调优（OCCUPANCY_BOUND）

**目的**：OCCUPANCY_BOUND 场景下，通过调整 `blockDim` 在寄存器、共享内存、occupancy 三者之间取得平衡。

**诊断步骤**：
1. 查看 NCU 报告中 `Block Limit Registers` / `Block Limit Shared Mem` / `Block Limit Warps` 的限制因素
2. 若 `Block Limit Registers` 最小 → 增大 Block 或用 `__launch_bounds__` 降低寄存器/线程
3. 若 `Block Limit Shared Mem` 最小 → 减少共享内存分配或使用动态共享内存
4. 若 `Block Limit Warps` 最小 → 增大 Block（但不超过 1024）

**常用配置对比**（sm_89，每 SM 1536 threads）：

| blockDim | 每 SM block 数 | Occupancy | 适用场景 |
|---------|--------------|-----------|---------|
| (64, 1) | 24 | 100% | 简单 element-wise |
| (128, 1) | 12 | 100% | 通用 1D kernel |
| (256, 1) | 6 | 100% | Reduction |
| (16, 16) | 6 | 100% | 2D 矩阵 Tiling |
| (32, 32) | 1~2 | 低（受寄存器/smem 限制） | 大 Tile，需 `__launch_bounds__` |

**Grid-Stride Loop（Block 过小时的替代方案）**：
```cuda
// 不改变 gridDim，通过循环让每个线程处理多行
for (int row = blockIdx.y * blockDim.y + threadIdx.y;
         row < M;
         row += gridDim.y * blockDim.y) {
    // ... 处理逻辑
}
```

---

## 常见瓶颈 → 优化映射

| NCU 特征 | 瓶颈类型 | 优先优化 |
|---------|---------|---------|
| L1 Throughput > 90%，DRAM < 30% | L1_PRESSURE_BOUND | P0: Shared Memory Tiling → P1: Padding → P2: Data Transpose |
| DRAM Throughput > 70%，SM Busy < 30% | DRAM_MEMORY_BOUND | P0: Block Tiling → P1: Vectorized Load → P2: Prefetching |
| Warp Cycles > 20，Eligible < 40% | LATENCY_BOUND | P0: Double Buffering → P1: ILP → P2: Loop Unrolling |
| SM Busy > 80%，Compute Throughput > 80% | COMPUTE_BOUND | P0: FMA → P1: FP16/TF32 → P2: Tensor Core |
| Occupancy < 30%，SM Busy > 70% | OCCUPANCY_BOUND | P0: 调整 Block Size → P1: `__launch_bounds__` → P2: 减少 smem |
