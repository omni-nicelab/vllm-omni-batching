# Qwen-Image Disaggregated VAE 设计

本文档描述当前 Qwen-Image disaggregated VAE 的实现方式，重点覆盖多 stage 控制面、数据面传输、模型加载、现有引擎兼容性，以及本次新增的组件。

## 背景与目标

Qwen-Image 原始执行路径是一个耦合的 diffusion pipeline：prompt encode、DiT denoise loop、VAE decode 都在同一个 pipeline 实例和同一个 stage 内完成。disaggregated VAE 的目标是把它拆成三个可独立部署的 stage：

```text
Encode stage -> Denoise stage -> Decode stage
GPU 0          GPU 1            GPU 0
```

目标如下：

- 将 text encoder/VAE 相关逻辑和 DiT denoise loop 解耦，使模型权重可以按 stage 分开加载。
- 保持 Qwen-Image 现有单 stage 执行路径可用，不要求已有用户改配置。
- 保持 runner 和 pipeline 的职责清晰：runner 管执行、进程、传输、batch loop；pipeline 只做模型计算。
- 复用 vLLM-Omni 现有多 stage 控制面和 `custom_process_input_func` 风格，不新增 Qwen 专用的通用 IO schema。
- 以 offline Qwen-Image baseline/3-stage 脚本作为性能 testbed，目标是 3-stage 平均生成耗时不超过 baseline 的 1.1x。

当前实现不是一个通用的任意子模块拓扑系统，而是先为 Qwen-Image 建立一个简洁的 encode/denoise/decode 分离路径。后续 text encoder、image/audio tower、talker aggregator 等模块可以沿用同一类 stage 抽象继续演进。

## Motivation：LightX2V/Mooncake 的参考意义

LightX2V 在 2026-04-14 的博客 [Breaking the Memory and Throughput Bottlenecks of Diffusion Model Inference](https://light-ai.top/LightX2V-BLOG/posts/Disaggregation/) 中给出了一个和本 feature 高度相关的生产化参考：把 diffusion pipeline 拆成 Encoder、Transformer/DiT、Decoder 三个服务，并用 Mooncake RDMA 传输阶段间 tensor。

这篇博客对当前 feature 的 motivation 有三点直接帮助：

| 观察 | 对本 feature 的意义 |
| --- | --- |
| Qwen-Image-2512 的 DiT、text encoder、VAE 同时加载时显存压力很高，单体 BF16 pipeline 会超过 40GB 级别 GPU 的承载范围。 | 证明“把 text encoder/VAE 从 DiT stage 拆出去”不是代码结构洁癖，而是为了解决真实的 memory wall。当前 `_STAGE_COMPONENTS` 按 `encode`、`denoise`、`decode` 拆权重加载，方向是对的。 |
| 在内存受限 GPU 上，baseline 往往依赖 CPU offload；LightX2V 的数据表明 encoder 单独部署后可以避免 offload，从而显著降低 text encoder latency。 | 当前三 stage 设计的第一目标应该是减少 offload 和显存竞争，而不只是把 VAE decode 移出 diffusion stage。encode stage 应该明确承载 text encoder、image/VAE encoder 等前处理模块。 |
| LightX2V 报告 Qwen T2I 的 Phase1 prompt embedding 约几十 MB，Phase2 latent 远小于像素空间输出；Mooncake RDMA 下网络占比很低。 | 支持“阶段间传 latent/feature tensor 是可行的”这个假设。它也说明真正的瓶颈通常仍在 DiT loop，传输优化的目标是不要让通信成为新的瓶颈。 |

需要注意的是，LightX2V 的结论不能直接等价到当前实现：

- LightX2V 的低通信开销依赖 Mooncake RDMA；当前实现仍使用 ZMQ/msgpack，tensor 会经过 CPU bytes，因此是 D2H2D。
- LightX2V 的吞吐收益还来自去中心化队列和多个 Transformer worker；当前 `qwen_image_3stage.yaml` 是静态线性 1E:1T:1D 拓扑。
- LightX2V 的数据主要用于论证方向和设定优化目标；当前 feature 的真实性能仍应以本仓库 offline baseline/3-stage 脚本为准。

## LightX2V/Mooncake 参考与演进路线

结合本地 `lightx2v/disagg` 实现，当前 feature 可以借鉴的不是整套服务化架构，而是几条可以逐步迁移的工程模式：

- 语义上继续保持 Encoder/Transformer/Decoder 的分工，其中当前配置里的 `encode` 对应 Encoder，`denoise` 对应 only DiT/Transformer，`decode` 对应 Decoder；`diffusion` 保留为完整耦合 pipeline。
- 部署上允许 Encoder 和 Decoder 共卡，因为 decoder 权重和耗时相对较小，但这应该是配置选择，不应成为硬编码约束。
- 性能上优先优化 DiT/denoise worker 的利用率。LightX2V 的吞吐模型说明 Transformer 通常是瓶颈，encoder 通常不是瓶颈，因此多卡生产形态应优先考虑 1 个 encode/decode stage 对多个 denoise workers。
- 数据面上把小 metadata/control plane 和大 tensor data plane 分离。LightX2V 用 RDMA ring 传 dispatch metadata，用 Mooncake buffer 传 tensor；这正是 vLLM-Omni 当前 ZMQ/msgpack inline tensor 路径最需要改进的地方。

下面的改进建议按对 vLLM-Omni 当前 disaggregated VAE 的收益和侵入性排序。

| 优先级 | LightX2V 实现模式 | vLLM-Omni 当前状态 | 建议改进 |
| --- | --- | --- | --- |
| P0 | metadata/control plane 和 tensor data plane 分离 | 当前 `OmniMsgpackEncoder` 会把 tensor `.cpu().numpy().tobytes()` 放进 ZMQ 消息 | 保留 `OmniRequestOutput.multimodal_output` 作为小 metadata 通道，但大 tensor 改为 connector handle。encode stage 输出 `{context_ref, latents_ref, shape, dtype, key}`，denoise/decode runner 根据 handle 拉取 tensor。 |
| P0 | 按 request 建立 transfer lifecycle | 当前 stage payload 随 output 生命周期走，缺少独立 transfer key、cleanup 和 abort cleanup | 引入 `StageTensorTransferManager`，按 `request_id/from_stage/to_stage/payload_name` 生成 key，统一 `put/get/cleanup`。完成、失败、abort 时由 orchestrator 或 runner 清理 sender/receiver 状态。 |
| P0 | 显式记录 shape/dtype/hash metadata | 当前 payload 里有 shape，但缺少统一 dtype、nbytes、hash 或完整性检查 | encode/denoise 输出 metadata 增加 `shape`、`dtype`、`nbytes`、可选 `hash`。默认不开 hash；debug/perf mode 打开 hash，定位跨进程 tensor 污染或 stale payload。 |
| P1 | receiver 预分配 buffer，再由 sender 写入 | 当前下游直接从 msgpack 反序列化 CPU tensor，再 `.to(device)` | connector 模式下先用 shape/dtype 估算 buffer，receiver 准备目标 buffer 或 connector managed buffer；sender 只传输到目标 buffer，减少额外 copy。短期可先用 `SharedMemoryConnector` 验证接口，再接 Mooncake。 |
| P1 | 异步 transfer 状态管理 | 当前 direct stage 转发基本是同步 payload 传递，缺少 transfer queue/ready 状态 | 在 runner 或 stage proc 内把 transfer 拆成 `enqueue_send`、`poll_send_done`、`poll_recv_ready`。Orchestrator 只看到 request 是否 ready，不直接等待 tensor copy 完成。 |
| P1 | per-stage metrics 拆 compute/scheduling/communication | 当前 offline compare 主要看总耗时，stage metrics 粒度不足 | 对 encode、denoise、decode 记录 `request_received_ts`、`compute_start_ts`、`compute_end_ts`、`output_enqueued_ts`、`transfer_done_ts`、payload bytes。这样才能判断 10% gap 是来自调度、序列化、D2H/H2D 还是模型计算。 |
| P1 | transformer/denoise worker pull queue | 当前 `qwen_image_3stage.yaml` 是固定线性 1E:1T:1D | 先在 orchestrator 内支持多个同类型 denoise stage 的 round-robin/least-queue 路由；之后再考虑像 LightX2V 一样用 phase1/phase2 queue，让 denoise/decode worker 主动 pull。 |
| P1 | transformer output room/sidecar 保持 transfer 资源生命周期 | 当前 submodule/diffusion stage 进程各自持有 output，缺少独立 owner | 如果接入 async connector，需要一个轻量 owner 管理 pending output，避免发送方 request 已 cleanup 但接收方还没拉完。可以先做进程内 manager，不必马上引入独立 sidecar。 |
| P2 | buffer size 估算函数 | 当前 Qwen stage payload 依赖实际 tensor 序列化，不预估容量 | 为 Qwen-Image 增加 encode payload 和 denoise payload 的 `estimate_*_payload_bytes()`。offline compare report 打印理论 bytes 和实际 bytes，辅助选择 ZMQ/SHM/Mooncake。 |
| P2 | strict/non-strict integrity mode | 当前没有 payload hash 校验开关 | 增加环境变量或 config，如 `VLLM_OMNI_STAGE_TRANSFER_STRICT_HASH=1`，用于压测和 debug；默认关闭，避免 hash 带来的 D2H 开销。 |
| P2 | image/VAE encode 路径 | 当前 encode stage 已加载 VAE，但 text-to-image 路径主要做 text encode、latent init、timestep prep | image edit/I2I/I2V 场景应把 image encoder/VAE encoder 输入也放在 encode stage 输出 payload 中，继续保持 denoise stage 只处理 DiT 输入。 |

推荐落地顺序如下：

1. 先不要改 stage 拓扑，保留当前 encode -> denoise -> decode 线性路径。
2. 把 Qwen-Image stage tensor 从 msgpack inline tensor 改成 `metadata + connector handle`，先用 `SharedMemoryConnector` 做单机验证。
3. 在 `VAEModelRunner` 和 `DiffusionModelRunner` 中引入通用 `StageTensorTransferManager`，runner 负责 transfer，pipeline 继续只负责模型计算。
4. offline compare report 加入 per-stage compute、transfer bytes、serialization/connector time。
5. 在性能和生命周期稳定后，把 connector 从 shared memory 切到 `MooncakeTransferEngineConnector`。
6. 最后再做多个 denoise stage 的调度；这一步收益依赖并发和多卡，不应该和 P0 的数据面改造混在一起。

这个演进路线和当前代码职责边界是一致的：pipeline 不感知 RDMA/Mooncake；stage input processor 不搬大 tensor，只搬 metadata；runner/stage proc 管 connector put/get；orchestrator 管 request routing 和生命周期。

## 设计点回应与补齐状态

本节逐条回应设计阶段提出的问题，并说明当前实现已经覆盖到什么程度。

| 设计点 | 当前状态 | 当前实现与判断 |
| --- | --- | --- |
| 新的阶段类型和配置 | 已补齐 | 当前代码落地为 `StageType.SUBMODULE` 和 `stage_type: submodule`。如果文档中称为 substage module，需要注意这只是概念名称；配置和代码里的稳定名称是 `submodule`。 |
| Encode -> Denoise -> Decode 三 stage | 已补齐 | `qwen_image_3stage.yaml` 定义 encode、denoise、decode 三段线性拓扑。encode/decode 是 `submodule` stage，denoise 是 `diffusion` stage。 |
| encode 同时支持 text encoder/VAE encoder 在一张卡上 | 部分补齐 | 当前 encode stage 加载 `text_encoder`、`tokenizer`、`vae`、`scheduler`。text-to-image 路径已经把 text encode、latent init、timestep prep 放在 encode stage；真正依赖输入图像的 VAE encode/image conditioning 仍需要后续在 `execute_encode()` 中补齐。 |
| 是否要把 text encoder 单独拆成另一个 stage | 当前不拆 | 当前先把 text encoder 和 VAE 归到同一个 encode submodule，避免过早增加 stage 拓扑复杂度。后续如果模型/任务需要复用 text encoder、跨请求缓存 text embedding，或 text encoder 成为瓶颈，可以把 text encoder 做成独立 `submodule` stage。 |
| 阶段间数据传输 | 部分补齐 | encode->denoise 和 denoise->decode 的 payload 已经通过 `multimodal_output`、`custom_process_input_func`、`OmniTokensPrompt.additional_information` 串起来。当前传输仍是 ZMQ/msgpack 的 D2H2D，不是 connector-backed D2D。 |
| 潜在张量比像素空间张量小，传输应较高效 | 当前假设成立但需要性能数据持续验证 | 三 stage 路径主要传 text embedding、latent、timestep metadata 和 final latents，不传 decode 后图片作为中间结果。相比像素空间张量，这个选择更适合跨 stage 传输；offline compare 使用 1.1x 阈值验证端到端性能。 |
| 拆分 diffusion stage 当前处理逻辑 | 已补齐到 Qwen-Image 级别 | prompt/text encode、latent/timestep 准备被移动到 encode stage；denoise stage 的 `prepare_encode()` 只从上游 payload 还原 `DiffusionRequestState`；decode 被移动到 decode stage。 |
| `custom_process_input_func` 承担跨 stage 处理 | 已补齐 | `encode_to_denoise()` 和 `denoise_to_decode()` 负责读取上游 stage output、校验字段、构造下游 prompt。orchestrator 和 runner 不直接理解 Qwen-Image payload schema。 |
| pipeline 拆成 `encode()`、`denoise()`、`decode()` 而不是整体 `call()` | 已补齐 | 当前仍保留一个 `QwenImagePipeline` 类，但增加了 `execute_encode()`、`prepare_encode()`/`denoise_step()`/`step_scheduler()`、`post_intermediate_output()`、`execute_decode()` 等分段入口。没有拆成多个 pipeline 类。 |
| pipeline 是否尽量不划分 | 已按这个方向实现 | 权重和执行函数按 stage 分离，但 pipeline 类保持统一。这样可以复用原有 Qwen-Image helper，减少 full path 和 three-stage path 的行为漂移。 |
| 模型权重分开加载 | 已补齐 | `_STAGE_COMPONENTS` 控制不同 `model_stage` 加载不同组件：`diffusion` 加载完整组件，`encode` 加载 text encoder/tokenizer/VAE/scheduler，`denoise` 加载 transformer/scheduler，`decode` 只加载 VAE。 |
| 向后兼容 | 已补齐 | `model_stage=None` 会被 normalize 为 `diffusion`，仍然走完整耦合 pipeline。没有配置单独 VAE stage 时，VAE 仍在 diffusion stage 内加载和执行。 |
| VAE runner 类似 `DiffusionModelRunner` | 已补齐 | 新增 `VAEModelRunner`，用于 encode/decode submodule 的单次执行。它比 `DiffusionModelRunner` 简单，不包含 diffusion scheduler loop。 |
| runner 和 pipeline 职责边界 | 已补齐 | runner 负责加载入口、forward context、inference mode、step loop 和 intermediate output 选择；pipeline 负责模型计算；stage input processor 负责模型相关跨 stage payload 转换。 |
| 是否需要一个支持单 module/op 的通用 stage | 部分补齐 | `submodule` stage 已经提供控制面和进程形态，但当前 runner 仍是 Qwen-Image encode/decode 导向的 `VAEModelRunner`。后续可以把它泛化为 module/op runner，用于 text encoder、image/audio tower、talker aggregator。 |
| LLM/Diffusion/Sub-Module 三类层次映射 | 已补齐控制面，执行层仍偏 Qwen-Image | 当前映射是 `StageEngineCoreClient`/`StageDiffusionClient`/`StageSubModuleClient`，`StageEngineCoreProc`/`StageDiffusionProc`/`StageSubModuleProc`。Sub-Module 执行层目前是 `DiffusionWorker + VAEModelRunner + pipeline.execute_*`，还不是独立的通用 module engine。 |

总体判断是：当前实现已经补齐 Qwen-Image disaggregated VAE 所需的最小闭环，包括 stage 类型、配置、控制面、payload 转换、分段 pipeline 入口、分 stage 权重加载和向后兼容。尚未完全补齐的是更通用的 submodule/op stage 抽象、真正的 connector-backed tensor transfer，以及 image conditioning 场景下的 VAE encode 输入路径。

## Stage 语义

当前 Qwen-Image 明确区分 `diffusion` 和 `denoise`：

| `model_stage` | `stage_type` | 语义 | 主要 runner |
| --- | --- | --- | --- |
| `None` 或 `diffusion` | `diffusion` | 完整耦合路径，包含 text encode、denoise loop、VAE decode | `DiffusionModelRunner` |
| `encode` | `submodule` | 上游准备阶段，负责 prompt/text encode、latent/timestep/scheduler 输入准备，并作为未来 VAE encode 的归属位置 | `VAEModelRunner` |
| `denoise` | `diffusion` | 只跑 DiT denoise loop 和 scheduler step，不做 VAE decode | `DiffusionModelRunner` |
| `decode` | `submodule` | 只做 VAE decode 和 image postprocess | `VAEModelRunner` |

关键约束是：`diffusion` 表示完整三段耦合执行，`denoise` 表示 only DiT/scheduler loop。二者不能混用。

三 stage 配置位于 [`qwen_image_3stage.yaml`](gh-file:vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml)：

| Stage | 配置 | 设备 | 输入来源 | 输出 |
| --- | --- | --- | --- | --- |
| 0 | `stage_type: submodule`, `model_stage: encode` | GPU 0 | 原始 prompt | encode payload |
| 1 | `stage_type: diffusion`, `model_stage: denoise`, `step_execution: true` | GPU 1 | stage 0 | denoised latents |
| 2 | `stage_type: submodule`, `model_stage: decode` | GPU 0 | stage 1 | final image |

stage 之间的数据转换由配置中的 `custom_process_input_func` 指定：

- `vllm_omni.model_executor.stage_input_processors.qwen_image.encode_to_denoise`
- `vllm_omni.model_executor.stage_input_processors.qwen_image.denoise_to_decode`

## 控制面

控制面沿用 vLLM-Omni 的多 stage orchestrator 模型，新增 `submodule` 作为轻量 stage 类型。

### Stage 初始化

stage config 解析后会生成每个 stage 的 metadata，包括 `stage_id`、`stage_type`、`model_stage`、`engine_input_source`、`custom_process_input_func`、`final_output` 等字段。

初始化入口在 [`stage_init_utils.py`](gh-file:vllm_omni/engine/stage_init_utils.py)：

- `stage_type == "diffusion"` 时初始化 `StageDiffusionClient`。
- `stage_type == "submodule"` 时初始化 `StageSubModuleClient`。
- `submodule` stage 使用 `OmniDiffusionConfig.from_kwargs(...)` 构造 diffusion config，但进程内挂载的是轻量 `VAEModelRunner`。

### Orchestrator 路由

[`Orchestrator`](gh-file:vllm_omni/engine/orchestrator.py) 把 `diffusion` 和 `submodule` 都视为 direct stage：

```python
DIRECT_STAGE_TYPES = {"diffusion", "submodule"}
```

direct stage 的输出不会经过 LLM EngineCore 的 token output 处理，而是由 orchestrator 直接轮询：

- `submodule` stage 调 `get_submodule_output_nowait()`。
- `diffusion` stage 调 `get_diffusion_output_nowait()`。

当一个 stage 完成后，orchestrator 做三件事：

1. 将当前 stage 输出通过 `set_engine_outputs([output])` 缓存在对应 stage client 上。
2. 如果下游 stage 配置了 `custom_process_input_func`，调用该函数把上游输出转换成下游 prompt。
3. 调用下游 direct stage 的 `add_request_async()` 或 `add_batch_request_async()` 提交请求。

因此，orchestrator 只管理 stage 拓扑、请求生命周期和转发时机，不理解 Qwen-Image 的 tensor schema。

### Submodule 进程

`StageSubModuleClient` 和 `StageSubModuleProc` 是新增的轻量 stage client/proc：

- client 侧通过 ZMQ PUSH/PULL 给子进程发送请求、接收结果。
- proc 侧创建 `DiffusionWorker(skip_load_model=True)`，再替换为 `VAEModelRunner`。
- proc 侧执行 `worker.execute_model(...)`，最终把 `DiffusionOutput` 包装成 `OmniRequestOutput` 返回 orchestrator。

这个结构和 diffusion stage 的 client/proc 形态保持一致，但 submodule stage 不启动完整 diffusion step scheduler。

## 数据面传输

当前 Qwen-Image 三 stage 数据面使用已有的 stage output 和 prompt 结构传输中间结果：

```text
Pipeline output dataclass
  -> DiffusionOutput.multimodal_output
  -> OmniRequestOutput.multimodal_output
  -> ZMQ/msgpack
  -> stage_input_processor
  -> OmniTokensPrompt.additional_information
  -> downstream pipeline
```

### Encode 到 Denoise

encode stage 由 `VAEModelRunner.execute_model()` 调用 `pipeline.execute_encode()`。pipeline 返回 `EncodeOutput`，runner 将 dataclass 字段放入 `DiffusionOutput.multimodal_output`。

`encode_to_denoise()` 从上游 `OmniRequestOutput.multimodal_output` 读取并校验以下核心字段，然后包装成 `OmniTokensPrompt.additional_information`：

- `context`、`context_mask`
- `context_null`、`context_null_mask`
- `latents`、`latent_shape`
- `timesteps`、`sigmas`、`num_inference_steps`
- `height`、`width`、`img_shapes`
- `guidance`、`guidance_scale`、`true_cfg_scale`、`do_true_cfg`
- `txt_seq_lens`、`negative_txt_seq_lens`
- 可选 `image_latents`

denoise stage 的 `QwenImagePipeline.prepare_encode()` 在 `model_stage == "denoise"` 时不会重新做 text encode，而是调用 `_prepare_denoise_stage_state()` 从 `additional_information` 还原 `DiffusionRequestState`。

### Denoise 到 Decode

denoise stage 在 step loop 结束后不调用 `post_decode()`。`DiffusionModelRunner` 会检查：

- `pipeline.produces_intermediate_stage_output`
- `pipeline.post_intermediate_output(...)`

对于 Qwen-Image `model_stage == "denoise"`，`post_intermediate_output()` 输出 denoised latents 和必要的 shape metadata：

- `latents`
- `latent_shape`
- `height`
- `width`
- `output_type`

`denoise_to_decode()` 将这些字段包装进 decode stage 的 `OmniTokensPrompt.additional_information`。decode stage 调 `pipeline.execute_decode()`，读取 latents 后执行 VAE decode 和后处理。

### 当前传输特性

当前数据面没有新增 connector-backed tensor transfer。direct stage 之间仍通过 `OmniMsgpackEncoder`/`OmniMsgpackDecoder` 序列化，tensor 会被转成 CPU bytes 再通过 ZMQ 传输。因此当前路径本质是 D2H2D：

```text
GPU tensor -> CPU bytes -> ZMQ/msgpack -> CPU tensor -> target GPU tensor
```

这个选择是为了保持实现简单，并复用已有 direct stage 通道。Qwen-Image 三 stage 的主要计算量在 40 次 DiT denoise loop，stage 间只传一次 encode payload 和一次 final latents，因此当前目标是先把性能 gap 控制在 10% 以内。如果后续更大 batch、更大 context 或跨机部署导致传输成为瓶颈，再把这些中间 tensor 接入 Mooncake/UCX 等 connector 做更接近 D2D 的传输。

## 模型加载

Qwen-Image pipeline 内部通过 `_STAGE_COMPONENTS` 决定当前 stage 拥有哪些组件：

| `model_stage` | 加载组件 |
| --- | --- |
| `diffusion` | `scheduler`, `text_encoder`, `tokenizer`, `vae`, `transformer` |
| `encode` | `scheduler`, `text_encoder`, `tokenizer`, `vae` |
| `denoise` | `scheduler`, `transformer` |
| `decode` | `vae` |

加载策略在 [`pipeline_qwen_image.py`](gh-file:vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py) 中完成：

- `transformer` 只在拥有 transformer 的 stage 构造，并通过 `weights_sources` 交给 `DiffusersPipelineLoader`/vLLM 权重加载路径。
- `text_encoder`、`tokenizer`、`vae` 使用 diffusers/HF `from_pretrained(...)` 按组件加载。
- `encode` stage 当前 text-to-image 路径主要执行 prompt encode、latent 初始化和 timestep 准备；VAE 也归属 encode stage，便于未来 image conditioning 或 VAE encode 逻辑留在同一张卡上。
- `decode` stage 只加载 VAE，避免加载 DiT transformer 和 text encoder。
- `denoise` stage 不加载 VAE 和 text encoder，只加载 scheduler 和 transformer。

submodule 进程初始化时会补齐必要 config：

- 从 `model_index.json` 推断 `model_class_name`。
- 从 `transformer/config.json` 构造 `TransformerConfig`。
- 调用 `update_multimodal_support()` 保持 diffusion config 语义一致。

## 执行流程

### 单 stage 兼容路径

当 `model_stage` 未设置或设置为 `diffusion` 时，Qwen-Image 仍走原来的完整 pipeline：

```text
DiffusionModelRunner
  -> QwenImagePipeline.prepare_encode()
  -> denoise_step() * N
  -> step_scheduler() * N
  -> post_decode()
```

此时 `prepare_encode()` 调 `_prepare_full_diffusion_state()`，会完整执行 prompt encode、latent/timestep 准备，并在结束后本地 VAE decode。

### 三 stage 分离路径

三 stage 分离路径如下：

1. Encode stage 收到原始 prompt，`VAEModelRunner` 调 `QwenImagePipeline.execute_encode()`。
2. `execute_encode()` 生成 text embedding、latents、timesteps、CFG metadata，并输出 `EncodeOutput`。
3. Orchestrator 调 `encode_to_denoise()`，把 encode payload 转为 denoise stage prompt。
4. Denoise stage 使用 `DiffusionModelRunner.execute_stepwise()`。新请求第一次进入时，`prepare_encode()` 根据 `model_stage == "denoise"` 从 payload 还原 request state。
5. 每个 denoise tick 调 `InputBatch.make_batch(...)` 重新构造当前 batch，然后执行 `denoise_step()` 和 `step_scheduler()`。
6. 请求完成后，runner 调 `post_intermediate_output()` 输出 denoised latents，而不是 decode image。
7. Orchestrator 调 `denoise_to_decode()`，把 latents 转为 decode stage prompt。
8. Decode stage 的 `VAEModelRunner` 调 `QwenImagePipeline.execute_decode()`，执行 VAE decode 和 image postprocess，返回最终图片。

这个设计保留了 denoise loop 中的多次 make batch 操作，同时把 encode/decode 从 DiT stage 中移走。

## 职责边界

当前实现刻意避免新增 `DiffusionStageIO` 或 `QwenImageStageIO` 这类 Qwen 专用 IO wrapper。原因是现有 GLM-Image、Bagel 以及 omni 多 stage 模型已经采用 stage input processor 做跨 stage 数据适配，新增 wrapper 会让 runner 和模型 schema 耦合更重。

当前职责划分如下：

| 层 | 职责 |
| --- | --- |
| Orchestrator | stage 拓扑、请求生命周期、轮询输出、转发下游 |
| Stage client/proc | 进程边界、ZMQ 消息、sampling params 序列化、子进程生命周期 |
| Stage input processor | 模型相关 payload 校验、字段选择、上游输出到下游 prompt 的转换 |
| Runner | 权重加载入口、forward context、inference mode、step loop、是否输出 intermediate |
| Pipeline | 模型相关计算：encode、denoise、scheduler、decode、postprocess |

这样做的结果是：

- `DiffusionModelRunner` 同时兼容完整 `diffusion` 和 only `denoise`。它不硬编码 Qwen-Image stage schema，只根据 pipeline 暴露的 `produces_intermediate_stage_output` 决定是否输出中间结果。
- `VAEModelRunner` 只负责轻量 submodule 的单次执行，具体 encode/decode payload 由 pipeline 定义。
- Qwen-Image 的字段命名和校验集中在 `stage_input_processors/qwen_image.py` 和 `pipeline_qwen_image.py`。

## 兼容现有引擎

兼容性主要体现在四点：

- 默认 `model_stage=None` 会被 normalize 为 `diffusion`，已有不分离配置仍然表示完整耦合 pipeline。
- `QwenImagePipeline.forward()` 和 stepwise `post_decode()` 保持完整路径语义，不要求用户启用三 stage。
- `DiffusionModelRunner` 的 step execution contract 仍然是 `prepare_encode()`、`denoise_step()`、`step_scheduler()`、`post_decode()`，只额外支持可选的 `post_intermediate_output()`。
- 新增 `submodule` stage 不替换 LLM stage 和 diffusion stage，只作为 direct stage 被 orchestrator 识别。

因此，现有单 stage Qwen-Image、其他 diffusion pipeline、LLM stage 的控制面不需要感知 Qwen-Image disaggregated VAE 的内部 payload。

## 新增组件

本次设计相关的新增或扩展点如下：

| 组件 | 作用 |
| --- | --- |
| `StageType.SUBMODULE` | 新增轻量 stage 类型 |
| [`stage_submodule_client.py`](gh-file:vllm_omni/diffusion/stage_submodule_client.py) | submodule stage 的主进程 client |
| [`stage_submodule_proc.py`](gh-file:vllm_omni/diffusion/stage_submodule_proc.py) | submodule stage 的子进程入口 |
| [`vae_model_runner.py`](gh-file:vllm_omni/diffusion/worker/vae_model_runner.py) | encode/decode submodule runner |
| [`stage_data.py`](gh-file:vllm_omni/diffusion/models/qwen_image/stage_data.py) | Qwen-Image stage payload dataclass |
| [`stage_input_processors/qwen_image.py`](gh-file:vllm_omni/model_executor/stage_input_processors/qwen_image.py) | encode->denoise、denoise->decode payload 转换 |
| [`qwen_image_3stage.yaml`](gh-file:vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml) | Qwen-Image 三 stage 示例配置 |
| offline baseline/3-stage scripts | baseline 和 3-stage 性能/一致性 testbed |

## 测试与性能

单元测试重点覆盖：

- `diffusion` 和 `denoise` stage 语义区分。
- `prepare_encode()` 根据 `model_stage` 选择 full diffusion 或 denoise-only 准备逻辑。
- denoise stage 结束后输出 intermediate latents，而不是直接 decode。
- encode->denoise、denoise->decode payload 字段校验。
- `DiffusionModelRunner` 对 `post_intermediate_output()` 的通用处理。

推荐基础验证命令：

```bash
python -m pytest -q \
  tests/diffusion/test_qwen_image_disagg_vae.py \
  tests/diffusion/test_diffusion_model_runner.py \
  tests/diffusion/test_diffusion_step_pipeline.py \
  tests/diffusion/test_diffusion_batching.py
```

性能 testbed 是 offline Qwen-Image baseline 和 3-stage 对比脚本：

```bash
python examples/offline_inference/text_to_image/compare_qwen_image_stage_vs_baseline.py \
  --model <qwen-image-model-path> \
  --stage-yaml vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml \
  --max-stage-ratio 1.1
```

`--max-stage-ratio 1.1` 表示 3-stage 平均生成耗时不能超过 baseline 的 1.1x。由于当前数据面还不是 D2D，如果这个阈值在更大 batch、跨 NUMA 或跨机环境下失败，优先检查 tensor payload 大小、CPU 序列化耗时、ZMQ copy 耗时，再考虑把 encode/denoise/decode 中间 tensor 接到 connector 传输路径。

## 当前限制与后续方向

当前实现还有以下限制：

- stage 间 tensor 传输仍是 ZMQ/msgpack 的 D2H2D 路径，不是 Mooncake/UCX 等 connector-backed D2D 路径。
- `submodule` stage 当前主要服务 Qwen-Image encode/decode，尚未抽象成所有 encoder/tower/aggregator 的统一 module runner。
- `qwen_image_3stage.yaml` 仍是静态线性拓扑，不包含复杂 DAG、并行 encode/decode 或自动设备放置。
- encode 和 decode 示例都放在 GPU 0，但当前没有跨 stage 显存复用调度器，只依赖进程和配置隔离。
- 当前 text-to-image encode 路径主要做 text encode、latent init、timestep prep；更复杂的 image conditioning/VAE encode 需要在 `execute_encode()` 中继续补齐。

后续建议优先级：

1. 在性能数据证明 D2H2D 成为瓶颈后，把中间 tensor payload 接入已有 omni connector。
2. 将 `submodule` stage 的语义扩展为更通用的 module/op stage，用于 text encoder、image/audio tower、talker aggregator 等模块。
3. 将 stage payload schema 进一步收敛到模型本地 pipeline 和 stage input processor，避免 runner 引入模型专用字段。
4. 为 offline compare 增加稳定的 report 字段，包括每段 stage time、payload bytes、serialization time 和 end-to-end latency。
