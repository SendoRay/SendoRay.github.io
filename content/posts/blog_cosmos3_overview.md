# 一文吃透 Cosmos-Framework：从一次训练 run 看 NVIDIA Cosmos 3 全模态世界模型

> 本文基于开源仓库 `NVIDIA/cosmos-framework` 的源码，目标是回答两个问题——
> **① 项目从启动到完成一次完整 run 究竟跑了什么？**
> **② Cosmos 3 这个"全模态世界模型"到底长什么样？**

---

## 0. 引子：Cosmos-Framework 是什么

README 写得很清楚：

> Part of the NVIDIA Cosmos project family — the **training and serving framework** repository.
>
> **Cosmos 3** is a suite of **omnimodal world models** designed to jointly process and generate **language, images, video, audio, and action sequences** within a unified **Mixture-of-Transformers** architecture.

把这句话拆开看：
- 这是一个**训练 + 推理一体**的框架（不是仅推理库）
- 它要训出来的模型 **Cosmos 3** 一次吃下五种模态：文本、图像、视频、音频、动作
- 架构关键词：**Mixture-of-Transformers (MoT)**

整个仓库的主体只有一个 Python 包 `cosmos_framework/`，里面拆成两块：训练基础设施（trainer / data / model / checkpoint / callbacks ...）和推理基础设施（`inference/` 子包）。

---

## 第一部分：一次完整训练 run 的全流程

### 0.1 入口在哪

启动一次训练永远是这条命令：

```bash
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.train \
      --sft-toml=examples/toml/sft_config/vision_sft_nano.toml
```

入口文件 = `cosmos_framework/scripts/train.py:227`。它做的事情是经典的"读 TOML → 转 Hydra config → 启动 Trainer"三件套。

```python
# scripts/train.py:227-298
if __name__ == "__main__":
    parser = argparse.ArgumentParser(...)              # L228
    args = parser.parse_args()                         # L276
    if args.deterministic:
        _setup_deterministic_env_and_backends()        # L59-76
    config = load_experiment_from_toml(                # L281
        args.sft_toml, extra_overrides=args.opts)
    if args.dryrun:
        save_yaml_and_exit()                           # L287-296
    else:
        launch(config, args)                           # L298
```

---

### 0.2 Stage 1：TOML → Pydantic → Hydra 三层配置

Cosmos 用了一个有意思的设计：**用户面**是结构化 TOML，**内核面**是 Hydra LazyDict。中间通过 Pydantic 做强校验。

```python
# configs/toml_config/sft_config.py:675-741
def load_experiment_from_toml(toml_path, extra_overrides):
    raw = tomllib.load(open(toml_path, "rb"))                # L707  解析
    SFTExperimentConfig.model_validate(raw)                  # L712  Pydantic 严校验(未知键报错)
    task = raw["job"]["task"]                                # vfm | vlm
    base_config_path = TASK_TO_BASE_CONFIG[task]             # L716  路由到 base config
    overrides = build_hydra_overrides(raw)                   # L723  TOML 树 → Hydra dotted-path
    overrides += extra_overrides                             # L728  CLI 覆盖优先级最高
    return load_config(base_config_path, overrides)          # L741
```

走完这步之后内存里就有一棵 `Config` 树，包含：`model / optimizer / scheduler / dataloader_train / dataloader_val / job / trainer / checkpoint / model_parallel`。

> 设计哲学：**TOML 是配方，Hydra 是装配厂**。配方里写 `model.precision=bfloat16`，装配厂帮你 lazy-instantiate 出真实的 `OmniMoTModel`。

---

### 0.3 Stage 2：分布式 + Trainer 实例化

```python
# scripts/train.py:181-224
def launch(config, args):
    distributed_init()                              # NCCL / RANK / LOCAL_RANK
    config.validate(); config.freeze()              # 锁定配置
    trainer = config.trainer.type(config)           # → ImaginaireTrainer.__init__
    with model_init():
        model = instantiate(config.model)           # 真正建出 OmniMoTModel
    with data_loader_init():
        dataloader_train = instantiate(config.dataloader_train)
        dataloader_val   = instantiate(config.dataloader_val)
    trainer.train(model, dataloader_train, dataloader_val)
```

`ImaginaireTrainer.__init__` 的关键动作：

```python
# trainer/__init__.py:45-138
class ImaginaireTrainer:
    def __init__(self, config):
        with distributed_init():
            parallel_state.initialize_model_parallel(   # TP / PP / CP 一次性切好
                pipeline_model_parallel_size=...,
                tensor_model_parallel_size=...,
                context_parallel_size=...)
        misc.set_random_seed(config.trainer.seed, by_rank=True)  # 不同 rank 不同 seed
        self.callbacks    = CallBackGroup(config, self)          # 注册全部 hook
        self.checkpointer = instantiate(config.checkpoint.type, ...)  # DCP checkpointer
        self.straggler_detector = StragglerDetectorV2(...)       # 慢 rank 检测
```

---

### 0.4 Stage 3：主训练循环

`trainer/__init__.py:192-334` 是整个项目的心脏：

```python
def train(self, model, dataloader_train, dataloader_val):
    model = model.to("cuda")                                    # L206
    model.on_train_start(...)                                   # L207
    optimizer, scheduler = model.init_optimizer_scheduler(...)  # L211
    grad_scaler = torch.amp.GradScaler("cuda", ...)             # L212
    iteration = self.checkpointer.load(...)                     # L215  自动续训
    self.callbacks.on_train_start(model, iteration=iteration)
        # ↑ LoadPretrained 在此触发：
        #   - 无 latest_checkpoint.txt → 加载 HF 权重
        #   - 有 latest_checkpoint.txt → 跳过 HF（DCP 已恢复）

    while True:                                                 # 外层：重建迭代器
        dataloader_train_iter = iter(dataloader_train)
        while True:                                             # 内层：batch 循环
            data_batch, stop = self._fetch_and_broadcast_data(...)
            if iteration >= max_iter: break
            data_batch = misc.to(data_batch, device="cuda")

            output_batch, loss, grad_accum_iter = self.training_step(...)

            if grad_accum_iter != 0: continue                   # 还在累积梯度
            iteration += 1
            if iteration % save_iter == 0:
                self.checkpointer.save(...)                     # 异步落盘
            self.callbacks.on_training_step_end(...)            # WandbCallback 写日志
            if iteration % validation_iter == 0:
                self.validate(...)
```

---

### 0.5 Stage 4：单步 `training_step` 内部

```python
# trainer/__init__.py:336-395
def training_step(self, model_ddp, optimizer, scheduler, grad_scaler,
                  data, iteration, grad_accum_iter):
    with distributed.ddp_sync_grad(model_ddp, is_last_micro_batch):
        self.callbacks.on_before_forward(iteration)             # L364
        output_batch, loss = model_ddp.training_step(data, iteration)  # ← 模型 forward+loss
        self.callbacks.on_after_forward(iteration)

        self.callbacks.on_before_backward(model, loss, iteration)
        loss_scaled = grad_scaler.scale(loss / grad_accum_iter)
        loss_scaled.backward()                                  # L378
        model.on_after_backward()
        self.callbacks.on_after_backward(model, iteration)

    grad_accum_iter += 1
    if grad_accum_iter == config.trainer.grad_accum_iter:       # 累积满了
        self.callbacks.on_before_optimizer_step(...)            # ← GradClip 全局裁剪
        self._optimizer_step(...)                               # grad_scaler.step + scheduler.step
        self.callbacks.on_before_zero_grad(...)
        self._zero_grad(...)
        grad_accum_iter = 0
    return output_batch, loss, grad_accum_iter
```

> **关键点**：所有 callback 是嵌在循环各个时机点的"钩子"。这就是 cosmos-framework 不直接把日志、裁剪、HF 加载、HF 导出写死在循环里，而是塞进 `callbacks/` 的原因——任意能力都可以在 TOML 里开关。

---

### 0.6 Stage 5：Checkpoint 异步落盘

`checkpoint/dcp.py` 的设计很值得抄作业：**主进程把 state_dict 丢进队列就返回，后台进程跑 `torch.distributed.checkpoint.save`**，这样 GPU 不会被 IO 阻塞。

```python
class DistributedCheckpointer(AbstractCheckpointer):
    def save(model, optimizer, scheduler, grad_scaler, iteration):
        state_dict = build_state_dict(...)               # 含 ModelWrapper
        if async_mode:
            receiver_queue.put((state_dict, path))       # 丢进后台进程
        # 后台进程：dcp.py:171-254 save_checkpoint_in_background
        #   ├── save_state_dict_worker(state_dict, path)     使用 CustomSavePlanner
        #   │     - plan caching：~60% 提速 (235B Qwen3-VL on 64×GB200 实测)
        #   │     - EMA 转换
        #   └── sender_queue.put(SaveDone(iteration))
        _write_latest_checkpoint_file(iteration)
```

---

### 0.7 关键调用栈（一图流）

```
torchrun -m cosmos_framework.scripts.train --sft-toml=<recipe>.toml
└─ scripts/train.py:227                          [argparse]
   └─ scripts/train.py:281  load_experiment_from_toml()
      ├─ tomllib.load + Pydantic 校验
      ├─ build_hydra_overrides()
      └─ utils/config.py:521  load_config()
         └─ configs/base/config.py:49  make_config() / register_*
   └─ scripts/train.py:298  launch(config, args)
      ├─ distributed_init() / config.freeze()
      ├─ trainer/__init__.py:45      ImaginaireTrainer.__init__
      │  ├─ parallel_state.initialize_model_parallel() (TP/PP/CP)
      │  ├─ CallBackGroup(config, self)
      │  └─ DistributedCheckpointer 实例化
      ├─ instantiate(config.model)              → OmniMoTModel
      ├─ instantiate(config.dataloader_train)   → JointDataLoader / PackingDataLoader
      └─ trainer.train()
         ├─ checkpointer.load()                 [DCP 续训]
         ├─ callbacks.on_train_start()
         │  └─ callbacks/load_pretrained.py:22  [HF 权重决策]
         └─ while iteration < max_iter:
            ├─ next(dataloader_train_iter)
            ├─ training_step()
            │  ├─ model_ddp.training_step(data) [模型 forward+loss]
            │  ├─ loss.backward()
            │  └─ on_before_optimizer_step
            │     └─ callbacks/grad_clip.py:245  per-mesh 全局裁剪
            ├─ checkpointer.save()              [异步落盘]
            └─ on_training_step_end
               └─ callbacks/wandb_log.py:88     wandb.log(loss_dict)
```

---

### 0.8 数据流：从 TOML 到 wandb

```python
# 配置侧
[TOML 文件]                         examples/toml/sft_config/<recipe>.toml
   │ tomllib.load
[raw dict]                          sft_config.py:707
   │ Pydantic 校验
[SFTExperimentConfig]               sft_config.py:712
   │ build_hydra_overrides
[list[str] Hydra overrides]
   │ load_config + Hydra compose
[Config 树 (LazyDict)]              utils/config.py:465
   │ instantiate
[Trainer / Model / DataLoader 实例]

# 数据侧
[JSONL / WebDataset]                inputs/...
   │ get_sft_dataset
[Dataset]
   │ PackingDataLoader (token-budget 打包)
[batch dict {text_token_ids, images, video, ...}]
   │ custom_collate_fn (列表型 key 不堆叠)
[data_batch] → misc.to("cuda") → model_ddp.training_step(data, iteration)
   │
[output_batch, loss] → backward → grad_clip → optimizer.step

# 输出侧
$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/
├── checkpoints/iter_*              DCP checkpoint
├── config.{yaml,pkl}               配置快照
├── stdout.log                      训练日志
└── (wandb cloud)                   loss / grad_norm / iter_speed
```

---

## 第二部分：Cosmos 3 模型架构深度剖析

讲完了"怎么跑"，再来看"在跑什么"——也就是 `OmniMoTModel`（`omni_mot_model.py:62`）这条 4403 行的鲸鱼。

### 1.1 它要干什么

按 README 的说法，Cosmos 3 用**一个网络**同时干完了下面四件事：

```
VLM (视觉-语言理解)
+ Video Generator (视频生成)
+ World Simulator (世界模拟)
+ World-Action Model (机器人动作生成)
```

实现路径：把"理解（understanding）"和"生成（generation）"两套权重塞进同一个 Transformer 主干，但**走同一条注意力**。这就是 **Mixture-of-Transformers** 的字面意思——区别于把专家放在 MLP 里的 Mixture-of-Experts。

---

### 1.2 顶层架构图

```
══════════════════════════════════════════════════════════════════════════════
                  Cosmos3 OmniMoTModel  (omni_mot_model.py:62)
══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────  输入端：任意模态子集  ──────────────────────────────┐
│                                                                            │
│  Text         Image/Video         Audio (16kHz)         Action (机器人)     │
│  "describe..."   .mp4/.png         .wav                 [a_t, a_t+1, ...]  │
│     │              │                  │                    │               │
│     ▼              ▼                  ▼                    ▼               │
│  ┌────────┐   ┌───────────┐    ┌────────────┐       ┌────────────┐         │
│  │ VLM    │   │  Vision   │    │   AVAE     │       │ raw  action│         │
│  │Processor│  │ Tokenizer │    │  (audio    │       │ tensor     │         │
│  │+special│   │ (VAE)     │    │   VAE)     │       │ R^action_dim│        │
│  │ tokens │   │ Wan2.2 /  │    │            │       │            │         │
│  └────┬───┘   │ DC-AE /   │    └─────┬──────┘       └────┬───────┘         │
│       │       │ FluxVAE / │          │                   │                 │
│       │       │ UniAE     │          │                   │                 │
│       │       └─────┬─────┘          │                   │                 │
│       ▼             ▼                ▼                   ▼                 │
│   token_ids   latent[C,T,H,W]   latent[D,T_a]      action[D_a,T]           │
│                     │                │                   │                 │
│                  patchify          ──┘                ──┘                  │
│                  (p×p patches)                                             │
└─────────────────────┼──────────────────────────────────────────────────────┘
                      ▼
                ┌─────────────────────────────────────────┐
                │   Sequence Packing  (sequence_packing)  │
                │                                         │
                │  把不同模态的 token 拼到一条序列上,         │
                │  每个 token 带 SequencePlan 描述:         │
                │   • modality (text/vision/sound/action) │
                │   • shape (T,H,W) for vision            │
                │   • clean / noised flag                 │
                └────────────────┬────────────────────────┘
                                 ▼
══════════════════════════════════════════════════════════════════════════════
        Cosmos3VFMNetwork  (cosmos3_vfm_network.py:113) ── self.net
══════════════════════════════════════════════════════════════════════════════

┌─────────────────  各模态 latent → hidden_size 的入投影头  ─────────────────┐
│                                                                            │
│  text_token_ids ──► language_model.embed_tokens                            │
│                                                                            │
│  vision latent  ──► vae2llm   (Linear: patch_latent_dim → hidden)          │
│                  + latent_pos_embed (3D RoPE / sin-cos / unified-3d-mrope) │
│                  + time_embedder(t)         ← 扩散时间步嵌入               │
│                                                                            │
│  sound latent   ──► sound2llm (Linear: sound_dim → hidden)                 │
│                  + sound_modality_embed                                    │
│                                                                            │
│  action         ──► action2llm (DomainAwareLinear, 多 embodiment)          │
│                  + action_pos_embed + action_modality_embed                │
│                                                                            │
│  ※ vision/action/sound 三类生成 token 还会注入 timestep_embedding,         │
│     文本 token 不加噪也不加 timestep。                                     │
└─────────────────────────────┬──────────────────────────────────────────────┘
                              ▼
              ╔═══════════════════════════════════════════╗
              ║  language_model = Unified MoT backbone    ║
              ║  (unified_mot.py)                         ║
              ║                                           ║
              ║  3 种 variant 二选一:                     ║
              ║   • Qwen3VLTextForCausalLM    (Dense)     ║
              ║   • Qwen3VLMoeTextForCausalLM (MoE)       ║
              ║   • Nemotron3DenseVLTextForCausalLM       ║
              ║                                           ║
              ║  + visual tower (Qwen3VLVisionModel /     ║
              ║    Nemotron3 Vision) ← 理解通路用 ViT     ║
              ╚═══════════════════════════════════════════╝
                              │
                              ▼
┌─────────────────  各模态 hidden → latent 的出投影头  ──────────────────────┐
│                                                                            │
│  hidden ──► lm_head           ──► next-token logits   (text)               │
│  hidden ──► llm2vae           ──► patch_latent_dim    (vision velocity)    │
│                              ──► unpatchify ──► VAE.decode ──► video/img   │
│  hidden ──► llm2sound         ──► sound_dim           ──► AVAE.decode      │
│  hidden ──► llm2action        ──► action_dim          (DomainAwareLinear)  │
└────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
══════════════════════════════════════════════════════════════════════════════
        训练目标 (algorithm/loss/) ── 加权求和
══════════════════════════════════════════════════════════════════════════════

   Flow Matching loss   (vision/sound/action)   ← rectified_flow.py
   Cross-Entropy loss   (text tokens)            ← cross_entropy.py
   Load-Balancing loss  (仅 MoE variant)         ← load_balancing.py

══════════════════════════════════════════════════════════════════════════════
        推理采样 (diffusion/samplers/)
══════════════════════════════════════════════════════════════════════════════

   EDMSampler  /  FixedStepSampler  /  UniPCSampler
   text 部分用 AR 采样 (generate_reasoner_text)
```

---

### 1.3 MoT 的灵魂：双权重单注意力

每一个 `MoTDecoderLayer`（`unified_mot.py:868`）内部跑两条独立权重的通路（**und** = understanding，**gen** = generation），但共享同一个注意力矩阵：

```
══════════════════════════════════════════════════════════════════════════════
            一个 MoTDecoderLayer 的内部结构 (unified_mot.py:868)
══════════════════════════════════════════════════════════════════════════════

  packed_hidden_states  (text + vision_und + vision_gen + sound + action)
            │
            ▼
   ┌────────────────────────────────────────────────────────────────┐
   │            PackedAttentionMoT  (unified_mot.py:430)            │
   │                                                                │
   │   ┌─ und tokens (text/vision_understanding) ──┐                │
   │   │   q_proj / k_proj / v_proj / o_proj       │  ← 原 Qwen3-VL │
   │   │   q_norm / k_norm                         │     权重       │
   │   └────────────────────────────────────────────┘                │
   │                                                                │
   │   ┌─ gen tokens (vision_gen / sound / action) ─────┐           │
   │   │   q_proj_moe_gen / k_proj_moe_gen /            │           │
   │   │   v_proj_moe_gen / o_proj_moe_gen              │ ← 新增的  │
   │   │   q_norm_moe_gen / k_norm_moe_gen              │   gen 权重│
   │   └─────────────────────────────────────────────────┘           │
   │                                                                │
   │            Joint Attention (按 token 模态拼 Q/K/V)              │
   │            joint_attn_implementation:                          │
   │              two_way   = und ↔ gen 双向                        │
   │              three_way = und / vision_gen / 其他生成 三组       │
   │              + 可选 NATTEN sparse mask  (video 局部窗口)        │
   │              + 可选 video_temporal_causal (因果时序)            │
   └─────────────────────────┬──────────────────────────────────────┘
                             ▼
   ┌────────────────────────────────────────────────────────────────┐
   │   RMSNorm                                                      │
   │     ↓                                                          │
   │   MLP / Sparse-MoE-Block      ← 看 variant：                   │
   │      • Dense:   Qwen3VLTextMLP / Nemotron3DenseVLMLP           │
   │      • MoE:     Qwen3VLMoeTextSparseMoeBlock                   │
   │                  (top-k 路由 + load balancing loss)            │
   └────────────────────────────────────────────────────────────────┘
            │
            ▼  (× num_hidden_layers 次堆叠)
```

**MoT 的本质**：所有模态的 token 走同一座注意力塔，但通过给 Q/K/V/O 投影**复制一份独立权重**（带 `_moe_gen` 后缀），让"理解"和"生成"两条专家路径既独立又能在 attention 里互相看见。这就是论文里所谓"Mixture **of Transformers**"（区别于 Mixture **of Experts**，后者是 MLP 内部专家）。

权重映射证据见 `inference/model.py:73-96`：
```
^add_q_proj   →  q_proj_moe_gen
^add_k_proj   →  k_proj_moe_gen
^add_v_proj   →  v_proj_moe_gen
^to_add_out   →  o_proj_moe_gen
^norm_added_q →  q_norm_moe_gen
^norm_added_k →  k_norm_moe_gen
```

也就是 diffusers 里熟悉的 "additional projections" 在这边被重新组织成"生成专家"。

---

### 1.4 文件结构对照

```
cosmos_framework/model/vfm/                   ← Cosmos3 视觉基础模型实现
│
├── omni_mot_model.py            (4403 行)    OmniMoTModel 顶层 ImaginaireModel
│                                             ├── set_up_tokenizers (L107)
│                                             ├── build_net  (L174)
│                                             ├── training_step
│                                             └── generate_samples
│
├── mot/                                      Mixture-of-Transformers 主干
│   ├── cosmos3_vfm_network.py   (1176 行)    Cosmos3VFMNetwork
│   │                                         ├── vae2llm / llm2vae
│   │                                         ├── action2llm / llm2action
│   │                                         ├── sound2llm / llm2sound
│   │                                         ├── time_embedder
│   │                                         ├── latent_pos_embed
│   │                                         └── language_model (= Unified MoT)
│   │
│   ├── unified_mot.py           (2201 行)    底层 MoT decoder
│   │                                         ├── PackedAttentionMoT      (L430)
│   │                                         ├── MoTDecoderLayer         (L868)
│   │                                         ├── Qwen3VLTextForCausalLM  (L1842)
│   │                                         ├── Qwen3VLMoeTextForCausalLM(L1987)
│   │                                         └── Nemotron3...ForCausalLM (L2116)
│   │
│   ├── attention.py / dot_product_attention.py / context_parallel_*.py
│   ├── domain_aware_linear.py                跨 embodiment 的动作投影
│   ├── modeling_utils.py                     TimestepEmbedder, VideoRopePosition3DEmb
│   └── parallelize_*.py                      FSDP / TP / CP 切分
│
├── tokenizers/                               各模态编解码器
│   ├── wan2pt2_vae_4x16x16.py  (1682)        Wan2.2 VAE (主推, 视频)
│   ├── wan2pt1_vae_4x8x8.py    (843)         Wan2.1 VAE
│   ├── dc_ae/dc_ae_v.py        (1052)        Deep-Compression Auto-Encoder
│   ├── flux_vae_8x8.py         (492)         FLUX VAE (图像)
│   ├── uniae/noncausal_4x16x16.py            UniAE 16x16 spatial
│   └── audio/avae.py           (601)         Audio VAE (sound 模态)
│
├── vlm/                                      文本/理解侧的 VLM 实现
│   ├── qwen3_vl/qwen3_vl.py                  Qwen3-VL Dense backbone + ViT
│   ├── qwen3_vl_moe/qwen3_vl_moe.py          Qwen3-VL MoE backbone
│   └── nemotron_3_dense_vl/                  Nemotron 3 VLM backbone
│
├── diffusion/                                扩散调度 / 采样
│   ├── rectified_flow.py                     Rectified Flow 训练目标
│   └── samplers/{edm, fixed_step, unipc, fm_solvers_unipc}.py
│
├── algorithm/loss/                           训练损失
│   ├── flow_matching.py                      视觉/动作/声音损失
│   ├── cross_entropy.py                      文本损失
│   └── load_balancing.py                     MoE 负载均衡损失
│
└── upsampler/prompts.py        (1137)        Prompt upsampler (用 reasoner tower)
```

---

## 第三部分：把所有线串起来

把训练流程和模型结构合起来看，一次 `torchrun` 实际上完成的工作是：

```python
# 配置层
TOML_recipe
  → Pydantic_validate → Hydra_overrides → Config_tree

# 装配层
Config.model            → instantiate  OmniMoTModel
                                 ├── tokenizer_vision_gen  (Wan2.2 VAE …)
                                 ├── tokenizer_sound_gen   (AVAE)
                                 ├── vlm_processor         (Qwen3-VL Processor)
                                 └── net = Cosmos3VFMNetwork
                                          └── language_model = Unified MoT
                                                              (Qwen3-VL / Nemotron3)
Config.dataloader_train → instantiate  PackingDataLoader (token-budget 打包)
Config.checkpoint.type  → instantiate  DistributedCheckpointer (DCP + 异步)
Config.trainer.callbacks→ instantiate  CallBackGroup (GradClip / Wandb / ...)

# 运行层
trainer.train():
  for batch in dataloader:                        # JSONL → Packed batch
      out, loss = model.training_step(batch)       # MoT forward
                  └── flow_matching + CE + LBL    # 三种 loss 加权
      loss.backward()                              # FSDP/TP/CP 反向
      grad_clip → optimizer.step → wandb.log       # callbacks 时机点
      checkpointer.save() (异步)                   # 不阻塞训练
```

---

---

## 进一步阅读

- 项目入口手册：`AGENTS.md`
- 训练全指南：`docs/training.md`
- 推理全指南：`docs/inference.md`
- 代码地图：`docs/code_structure.md`
- Cosmos 3 技术报告：https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
- HuggingFace 模型集合：https://huggingface.co/collections/nvidia/cosmos3

---



