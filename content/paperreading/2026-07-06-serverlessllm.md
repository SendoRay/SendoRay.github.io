---
title: "ServerlessLLM: 用于大语言模型的低延迟无服务器推理"
date: '2026-07-06'
tags:
- LLM
- serverless
- inference
- cold-start
- checkpoint-loading
- model-migration
- scheduling
draft: false
math: true
ShowToc: true
summary: "通过多层存储检查点加载、推理实时迁移和启动时间优化调度，将无服务器 LLM 推理延迟降低 10-200×。"
---

## 论文 PDF

{{< pdf src="pdf/2401.14351_zh_CN.pdf" >}}

## 0. 论文图表

> 图表提取命令（见 paperreading.skill §1.3）：
> ```
> python3 scripts/extract_pdf_figures.py assets/pdf/2401.14351_zh_CN.pdf \
>     --page 3 --bbox 20 40 590 390 \
>     --name fig1-architecture --slug 2401-serverlessllm --dpi 250
> ```

{{< figure src="/images/paperreading/2401-serverlessllm/fig1-architecture.png" title="图 1: ServerlessLLM 系统架构 — Controller 负责请求路由和模型加载调度，GPU Server 利用 DRAM/SSD 多层存储，三大核心优化：快速多层检查点加载、推理实时迁移、启动时间优化调度" width="95%" >}}

{{< figure src="/images/paperreading/2401-serverlessllm/fig2-checkpoint-loading.png" title="图 2: 快速多层检查点加载组件 — 左侧为加载优化检查点格式（模型执行文件 + 张量索引文件），右侧为基于分块的内存管理子系统，通过 Direct I/O 并行加载" width="70%" >}}

{{< figure src="/images/paperreading/2401-serverlessllm/fig3-policy-comparison.png" title="图 3: 三种调度策略对比 — (a) 可用性驱动：GPU 空闲即调度，忽略局部性；(b) 局部性驱动：优先本地加载，但抢占延迟高；(c) 支持实时迁移的局部性驱动：兼顾局部性与迁移，两者延迟均达标" width="70%" >}}

{{< figure src="/images/paperreading/2401-serverlessllm/fig4-migration-process.png" title="图 4: LLM 推理实时迁移过程 — 源服务器推理过程中将 KV cache 迁移到目标服务器，目标服务器恢复中间 token 后继续推理" width="95%" >}}

{{< figure src="/images/paperreading/2401-serverlessllm/fig6-loading-performance.png" title="图 6: 检查点加载性能 — (a) ServerlessLLM 在所有模型上加载延迟最低，LLaMA-2-70B 上比 Safetensors 快 8.2×；(b) 带宽利用率接近 1.0，远超 PyTorch 和 Safetensors" width="95%" >}}

{{< figure src="/images/paperreading/2401-serverlessllm/fig10-system-performance.png" title="图 10: 数据集和模型对整体系统的影响 — ServerlessLLM 在 GSM8K 和 ShareGPT 上均显著优于 Ray Serve 及其缓存版本，OPT-30B GSM8K 延迟从 213s 降至 7.5s" width="95%" >}}

{{< figure src="/images/paperreading/2401-serverlessllm/fig12-scalability.png" title="图 12: 系统可扩展性与资源效率 — (a) ServerlessLLM 用 1 GPU 即达 KServe 4 GPU 的延迟水平；(b) 随模型数量增加，ServerlessLLM 延迟基本持平，Ray Serve 急剧上升" width="95%" >}}

## 1. 问题定义

### 1.1 解决了什么问题

本文解决的是 **"无服务器环境中大语言模型推理的冷启动延迟过高"** 的问题。

- **问题类别**：系统架构设计 + 存储层次优化
- **问题性质**：旧问题的新解法——无服务器冷启动是经典问题，但 LLM 的模型检查点体积（数十到数百 GB）使得传统冷启动优化方案失效
- **输入**：用户 LLM 推理请求 + 分布式 GPU 集群中的模型检查点
- **输出**：低延迟的 LLM 推理服务，冷启动时间从分钟级降至秒级

### 1.2 为什么要解决这个问题

**动机来源**：工程瓶颈 + 实际需求。

- **现有方法的不足**——具体痛点：
  1. **模型体积爆炸**：Grok-1 约 600 GB，DBRX 约 250 GB，Mixtral-8x22B 约 280 GB（§2.3）。从远程存储下载这些检查点需要数十秒到数分钟
  2. **冷启动占比极高**：Azure 追踪数据显示超过 40% 的函数冷启动率 >25%，约 25% 的函数冷启动率 >60%（§2.3，引用 [64]）
  3. **传统方案失效**：Hot-standby GPU 预加载模型仍需完整冷启动；Hybrid caching 仍有高 I/O 开销（§2.4）
  4. **LLM 推理负载的不可预测性**：编程助手、搜索引擎、语音代理等应用负载动态且不规则，天然适合无服务器模式，但冷启动延迟使体验不可接受（§2.1）

- **不解决的后果**：无服务器 LLM 推理无法实用化，用户被迫维持大量常驻 GPU，成本居高不下
- **为什么是现在**：LLM 模型规模从 7B 快速增长到 70B+，冷启动问题从"可忍受"变为"不可忍受"；同时 GPU 服务器本地存储能力（NVMe SSD、大容量 DRAM）已足够支撑本地化部署

### 1.3 问题难点

**核心挑战**：

1. **检查点加载瓶颈**：LLM 检查点体积远超 GPU 显存，需要从远程存储→本地 SSD→DRAM→GPU 显存逐级加载。传统加载方式（PyTorch、Safetensors）带宽利用率极低——在 RAID0 NVMe 上仅 13%-27%（图 6b），大量时间浪费在系统调用和内存拷贝上

2. **根本矛盾：通用性 vs 局部性**：
   - 无服务器调度追求**通用性**——任意 GPU 可服务任意模型，导致模型检查点分布在远程存储
   - 低延迟推理需要**局部性**——模型数据尽可能靠近 GPU，减少网络传输
   - 传统方案选择通用性，牺牲延迟；Hot-standby 选择局部性，牺牲资源利用率

3. **为什么简单方法不行**：
   - 直接缓存所有模型？显存/DRAM 不够——一个 70B 模型约 130 GB，8-GPU 服务器 DRAM 通常 1 TB，同时服务多个模型时缓存命中率低
   - 预加载到所有 GPU？资源浪费——GPU 空闲时预加载仍占显存，且无法应对多模型场景
   - 远程存储直接流式加载？网络带宽瓶颈——10 Gbps 网络加载 130 GB 模型需要约 100 秒

## 2. 方法

### 2.1 核心思想

**一句话概括**：利用 GPU 服务器内部的**多层存储层次结构**（GPU 显存 → DRAM → NVMe SSD），通过优化的检查点格式和并行 I/O 实现快速本地加载，配合推理实时迁移和启动时间感知调度，将冷启动延迟从分钟级降至秒级。

**Key Insight**：GPU 服务器拥有巨大的本地存储带宽（NVMe SSD 可达 20+ GB/s，远超网络带宽），但传统框架（PyTorch、Safetensors）的加载流程未充分利用这一带宽。问题不在硬件，而在**软件数据路径**——通过重新设计检查点格式和加载流程，可以逼近硬件带宽极限。

### 2.2 技术细节

#### 2.2.1 多层存储设计（§3）

ServerlessLLM 的设计基于对现代 GPU 服务器存储层次的观察：

| 存储层级 | 容量 | 带宽 | 延迟 |
|---------|------|------|------|
| GPU 显存 | ~80-192 GB | ~2-3 TB/s | ~ns |
| DRAM | ~1 TB | ~100 GB/s | ~ns |
| NVMe SSD (RAID0) | ~4-8 TB | ~20 GB/s | ~μs |
| SATA SSD | ~192 TB | ~0.5 GB/s | ~μs |
| 远程存储 (S3/MinIO) | ∞ | ~1 GB/s (10Gbps) | ~ms |

三个设计关注点：
1. **支持复杂的多层存储层次结构**：不同层级的带宽差异达 100×，需要自适应数据放置
2. **强局部性驱动的推理**：频繁访问的模型层缓存在 GPU/DRAM，冷数据在 SSD
3. **模型检查点仅存一份**：避免多 GPU 重复存储，通过共享内存区域加载

#### 2.2.2 快速多层检查点加载（§4）

**优化检查点格式**（图 2）：

传统检查点（PyTorch `.pt`、Safetensors）将元数据和张量数据混合存储，加载时需要：
1. 读取元数据 → 2. 逐个分配 GPU 内存 → 3. 逐个拷贝张量

ServerlessLLM 重新设计检查点格式，将**模型执行文件**与**张量索引文件**分离：

```
Loading-optimised checkpoint:
├── Model execution files    # 模型结构定义
│   └── class Model(nn.Module)
│       └── <Name, GPU_id, offset, size>  # 张量布局预计算
├── Tensor index file        # 张量分区索引
│   ├── Partition 0
│   ├── Partition 1
│   └── ...
└── Tensor data (contiguous) # 张量数据连续存储
```

**加载流程优化**：

1. **预计算张量布局**：检查点中预先记录每个张量在 GPU 内存中的偏移量（offset）和大小（size），加载时直接计算 GPU 内存基地址 + 偏移量，无需逐张量分配
2. **Direct I/O 绕过页缓存**：使用 `O_DIRECT` 直接从 SSD 读取到用户空间，避免内核页缓存拷贝开销
3. **多线程并行加载**：将张量数据分为多个 chunk，每个线程负责一个 chunk，通过并行 PCIe 链路同时加载
4. **Pinned memory + CUDA streaming**：使用锁页内存（pinned memory）加速 CPU→GPU 的 DMA 传输

**性能分解**（图 7，§7.2）：

| 优化手段 | 吞吐提升 |
|---------|---------|
| ReadByTensor（基线） | 1× |
| + Bulk（批量读取） | 1.2× |
| + Direct（Direct I/O） | ~1.5× |
| + Thread（多线程） | ~2.3× |
| + Pinned（锁页内存） | ~3× |
| + Pipeline（流水线） | ~4.4× |

#### 2.2.3 推理实时迁移（§5）

**问题**：当局部性驱动的调度导致某个 GPU 被新请求抢占时，正在进行的推理需要迁移到其他 GPU。传统方案是终止推理、重新加载模型、重新计算 prefix——延迟极高。

**ServerlessLLM 的方案**：KV cache 级别的实时迁移（图 4）：

1. 源服务器继续推理，同时将已生成的 KV cache 迁移到目标服务器
2. 目标服务器在后台加载模型检查点
3. 模型加载完成后，源服务器将最新 token 的 KV cache 发送到目标服务器
4. 目标服务器恢复推理，源服务器停止

**三种调度策略对比**（图 3）：

| 策略 | 思路 | Model A 延迟 | Model B 延迟 | 问题 |
|------|------|-------------|-------------|------|
| (a) 可用性驱动 | GPU 空闲即调度 | ✅ | ❌ | 忽略局部性，B 需从远程加载 |
| (b) 局部性驱动 | 优先本地缓存 | ✅ | ❌ | 抢占 A 导致 A 延迟升高 |
| (c) 迁移支持局部性 | 局部性 + 实时迁移 | ✅ | ✅ | A 迁移走，B 本地加载 |

**迁移时间估算**（§6.2）：

$$T_{\text{migrate}} \approx \frac{d_{\text{in}} \cdot t_{\text{in}} + d_{\text{out}} \cdot t_{\text{out}}}{\text{bandwidth}}$$

其中 $d_{\text{in}}$、$d_{\text{out}}$ 为输入/输出 token 数，$t_{\text{in}}$、$t_{\text{out}}$ 为每 token 处理时间。迁移量通常仅 10-100 KB（KV cache 增量），远小于完整模型检查点。

#### 2.2.4 启动时间优化调度（§6）

调度器维护两个关键估算（图 5）：

1. **模型加载时间**：$T_{\text{load}} = q + \frac{n}{b}$，其中 $q$ 为排队时间，$n$ 为模型大小，$b$ 为存储带宽
2. **模型迁移时间**：基于 token 数和网络带宽估算

调度决策：选择 $T_{\text{load}}$ 或 $T_{\text{migrate}}$ 最小的服务器。如果目标服务器已有模型缓存（DRAM/SSD），则 $T_{\text{load}} \approx 0$；如果源服务器可以迁移，则 $T_{\text{startup}} \approx T_{\text{migrate}}$。

### 2.3 相关工作对比

| 技术路线 | 代表系统 | 核心思路 | 优势 | 局限 |
|---------|---------|---------|------|------|
| 远程存储 + 网络加载 | KServe [16], Ray Serve [73] | 模型存 S3，按需下载 | 通用性强 | 网络带宽瓶颈，冷启动 >100s |
| GPU 热备 | Hot-standby [13,84] | 空闲 GPU 预加载模型 | 冷启动快 | 资源浪费，不支持多模型 |
| 混合缓存 | Shepherd [91], AlpaServe [44] | 模型缓存 + 卸载 | 平衡缓存与容量 | I/O 开销仍高，带宽利用率低 |
| 快照/fork | FaaSnap [15], SOCK [53] | VM 快照恢复 | 启动快 | 不适配大模型检查点 |
| **ServerlessLLM** | **本文** | **本地多层存储 + 检查点格式优化 + 迁移** | **带宽利用率 ~100%，延迟 10-200× 降低** | **需要本地 SSD，模型仅存一份** |

**与最接近 Baseline 的核心差异**：

- vs **Shepherd**：Shepherd 使用混合缓存但仍用传统检查点格式，带宽利用率低。ServerlessLLM 重新设计检查点格式，带宽利用率从 ~22% 提升到 ~100%
- vs **Ray Serve w/ Cache**：Ray Serve 缓存模型到本地但仍用 PyTorch 加载流程，启动延迟 ~213s。ServerlessLLM 用优化的加载流程，启动延迟 ~4s

### 2.4 Tradeoff 分析

**优势**（量化）：

1. **检查点加载速度**：比 PyTorch 快 3.6-8.2×，比 Safetensors 快 4.4-8.2×（图 6a）
2. **带宽利用率**：RAID0 NVMe 上从 13%（PyTorch）提升到 100%（图 6b）
3. **端到端延迟**：OPT-30B GSM8K 从 213s（Ray Serve）降至 7.5s，降低 28×（图 10a）
4. **资源效率**：1 GPU 达到 KServe 4 GPU 的延迟水平（图 12a）
5. **可扩展性**：模型数量从 16 增至 64 时延迟基本持平，Ray Serve 上升 4×（图 12b）

**代价**：

1. **本地存储依赖**：需要 GPU 服务器配备高速 NVMe SSD，不适用于无本地存储的云实例
2. **模型仅存一份**：多 GPU 共享同一检查点副本，并发加载时可能竞争 I/O 带宽
3. **迁移开销**：实时迁移需要源和目标 GPU 同时占用，高峰期资源利用率可能下降
4. **检查点格式不兼容**：需要将标准格式（`.pt`、`.safetensors`）转换为 ServerlessLLM 格式
5. **调度器复杂性**：需要维护全局存储状态、估算加载/迁移时间，调度器成为单点

**适用边界**：
- **最优场景**：多模型、动态负载、GPU 服务器有本地 NVMe SSD
- **不如基线**：单模型常驻、无本地存储、模型极小（<2GB）时优化收益有限

## 3. 实验

### 3.1 实验设置

**硬件**：

| 平台 | GPU | DRAM | SSD | 网络 |
|------|-----|------|-----|------|
| Platform 1 | 8× NVIDIA A5000 | 1 TB DDR4 | 2× NVMe SSD (RAID 0) | - |
| Platform 2 | 4× NVIDIA A40 | 512 GB DDR4 | 1× PCIe 4.0 NVMe 2 TB | 10 Gbps (MinIO) |

**模型**：OPT-2.7B/6.7B/13B/30B/66B, LLaMA-2-7B/13B/70B, Falcon-7B/40B

**数据集**：
- **ShareGPT**：83M 条对话，随机采样 20,448 条，每模型约 4K 样本
- **GSM8K**：数学推理数据集

**Baseline**：PyTorch, Safetensors, Ray Serve, Ray Serve w/ Cache, KServe, Shepherd*

**指标**：检查点加载延迟、带宽利用率、端到端推理延迟 (Mean Latency)、请求率 (RPS)

### 3.2 关键结果

**结果 1：检查点加载性能（图 6）**

| 模型 | PyTorch | Safetensors | ServerlessLLM | 加速比 (vs Safetensors) |
|------|---------|-------------|---------------|------------------------|
| OPT-2.7B | 3.0s | 1.8s | 0.5s | 3.6× |
| OPT-13B | 15.0s | 9.0s | 2.0s | 4.5× |
| OPT-66B | 80.0s | 45.0s | 10.0s | 4.5× |
| LLaMA-2-70B | 84.0s | 48.0s | 10.3s | 4.7× |
| Falcon-40B | 50.0s | 28.0s | 6.0s | 4.7× |

带宽利用率（图 6b）更揭示根本差异——ServerlessLLM 在所有存储介质上均接近 1.0，而 PyTorch 在 RAID0 NVMe 上仅 0.13。

**结果 2：端到端系统性能（图 10）**

| 模型 | 数据集 | Ray Serve | Ray Serve w/ Cache | ServerlessLLM | 改善 |
|------|--------|-----------|-------------------|---------------|------|
| OPT-6.7B | GSM8K | 12.1s | 8.2s | 0.8s | 10× |
| OPT-13B | GSM8K | 142.8s | 140.1s | 0.9s | 156× |
| OPT-30B | GSM8K | 213.0s | 199.2s | 7.5s | 27× |
| OPT-6.7B | ShareGPT | 27.6s | 17.9s | 0.8s | 22× |
| OPT-13B | ShareGPT | 182.2s | 162.4s | 1.6s | 102× |
| OPT-30B | ShareGPT | 260.2s | 261.8s | 89.8s | 2.9× |

ServerlessLLM 在 OPT-30B GSM8K 上 300s 超时内满足 89% 请求，Ray Serve w/ Cache 仅 26%。

**结果 3：可扩展性与资源效率（图 12）**

- KServe 需要 4 GPU 达到 12s 延迟，ServerlessLLM 仅需 1 GPU
- 模型数量从 16 增至 64 时，ServerlessLLM 延迟从 ~25s 到 ~30s（基本持平），Ray Serve 从 ~15s 升至 ~70s

### 3.3 消融实验

图 7 展示了检查点加载的逐级优化效果：

| 优化阶段 | 吞吐提升 | 关键贡献 |
|---------|---------|---------|
| ReadByTensor（基线） | 1× | 逐张量读取 |
| + Bulk | 1.2× | 批量读取减少系统调用 |
| + Direct I/O | ~1.5× | 绕过页缓存，减少内存拷贝 |
| + Thread | ~2.3× | 并行 I/O 利用多 PCIe 通道 |
| + Pinned memory | ~3× | 锁页内存加速 DMA 传输 |
| + Pipeline | ~4.4× | 加载与计算流水线化 |

图 8 展示了不同 RPS 下调度器的效果——ServerlessLLM 在 GSM8K 和 ShareGPT 上均保持低延迟，而 Shepherd* 在高 RPS 时延迟急剧上升。

## 4. 评估

### 4.1 局限性

**方法局限**：
1. **单 GPU 服务器假设**：ServerlessLLM 依赖服务器内部存储层次，不适用于跨节点张量并行的大模型推理（如 405B 模型需要多节点）
2. **迁移仅限单轮**：§5.3 描述的迁移流程假设模型在目标服务器上可直接加载，多轮迁移的累积延迟未充分讨论
3. **检查点格式转换开销**：需要预先将标准格式转换为 ServerlessLLM 格式，转换时间和存储空间开销未量化

**实验局限**：
1. **模型规模有限**：最大测试 LLaMA-2-70B（~130 GB），未测试 100B+ MoE 模型（如 Mixtral-8x22B ~280 GB）
2. **工作负载单一**：仅用 ShareGPT 和 GSM8K，未测试多轮对话、长上下文等真实场景
3. **Baseline 不完整**：未与 vLLM、TensorRT-LLM 等主流推理引擎直接对比，也未与 ColdStart 优化方案（如 FaaSNet、Catalyzer）对比

**假设局限**：
1. **GPU 服务器配 NVMe SSD**：假设每台 GPU 服务器有高速本地 SSD，但许多云实例（如 AWS p3.2xlarge）仅配 EBS 网络存储
2. **模型检查点仅存一份**：多 GPU 并发加载同一检查点时 I/O 带宽竞争未充分分析
3. **迁移延迟可忽略**：假设 KV cache 迁移量仅 10-100 KB，但长上下文场景下 KV cache 可能达 GB 级

### 4.2 启示

**对领域的贡献**：
1. **揭示带宽利用率问题**：传统框架在高带宽存储上的利用率仅 13-27%，指出软件数据路径而非硬件是瓶颈——这一洞察对 LLM 推理系统设计具有普遍指导意义
2. **检查点格式分离设计**：将元数据与张量数据分离、预计算布局的思路可推广到其他大模型加载场景
3. **局部性 + 迁移的调度范式**：用实时迁移化解局部性与通用性矛盾，为无服务器 ML 推理调度提供新范式

**对工程实践的指导**：
1. **优先利用本地存储**：GPU 服务器内部 NVMe SSD 带宽远超网络，模型检查点应优先存本地
2. **Direct I/O + 多线程**：大文件加载应绕过页缓存并并行读取，避免内核开销
3. **KV cache 迁移 > 模型重载**：抢占时迁移 KV cache 比重新加载模型 + 重算 prefix 快几个数量级

**可扩展方向**：
1. **多节点张量并行**：将本地多层存储扩展到多节点，支持 100B+ 模型的无服务器推理
2. **MoE 模型适配**：MoE 模型的稀疏激活特性可进一步减少检查点加载量
3. **与 PagedAttention 结合**：将 KV cache 迁移与 vLLM 的 PagedAttention 结合，实现更细粒度的内存管理

### 4.3 一句话总结

> ServerlessLLM 用 **多层存储检查点格式优化 + Direct I/O 并行加载 + KV cache 实时迁移** 解决了无服务器 LLM 推理的冷启动延迟问题，核心创新是**将带宽利用率从 13% 提升到 100%**，代价是**依赖本地 NVMe SSD 且检查点格式不兼容标准**。

## 参考文献

- [Paper (arXiv)](https://arxiv.org/abs/2401.14351)
- [ServerlessLLM GitHub](https://github.com/ServerlessLLM/ServerlessLLM)
- AlpaServe (OSDI '23) — 统计多路复用 + 模型并行
- Shepherd — 混合缓存推理
- vLLM / PagedAttention (SOSP '23) — KV cache 内存管理
- KServe — 标准化无服务器推理框架
