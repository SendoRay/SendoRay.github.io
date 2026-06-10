---
title: "LLM 领域，你需要知道的数字"
date: '2026-06-11'
tags:
- LLM
- GPU
- Training
- Inference


draft: false
math: true
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

> 这篇文章不讲"什么是 Transformer"，只回答一个问题：
>
> **当我们说"跑一个 70B 模型"时，到底在跟多少 FLOPs、多少 GB、多少 GB/s 打交道？**


下面所有节都以 **Llama-3-70B（dense, GQA, 80 层, hidden=8192, KV heads=8, head_dim=128, FFN=28672）** 作为参考模型，必要时会拓展到 7B / 405B / DeepSeek-V3 / GPT-4 / moe 量级做对比。

---

## 一、推理：单 token 究竟值多少计算

### 1.1 一条公式打底：前向 ≈ $2\Theta$ FLOPs / token [理论]

对一个参数量为 $\Theta$ 的 dense Transformer，**忽略 attention 二次项**时：

$$
F_{\text{fwd}} \approx 2\,\Theta \quad \text{FLOPs/token}
$$

来源：每个权重在前向传播中参与一次乘加（MAC = 2 FLOPs）。这条公式不依赖 batch、不依赖序列长度，只依赖参数量。

| 模型 | $\Theta$ | $F_{\text{fwd}}$ / token | 直觉 |
|---|---|---|---|
| Llama-3-8B | 8 B | 16 GFLOPs | 一张消费卡 1 ms 内可解 |
| Llama-3-70B | 70 B | 140 GFLOPs | H100 BF16 算力 ~1000 TFLOPs/s，理论 0.14 ms |
| Llama-3-405B | 405 B | 810 GFLOPs | 单卡装不下，必须 TP/PP |
| GPT-4（推测 ~1.8T MoE，激活 ~280B [实测/估算]） | 280 B 激活 | 560 GFLOPs | MoE 让"算"的成本回落到 70B 量级 |

**认知误区一**：很多人把 "175B 比 7B 慢 25 倍" 直接套到 latency 上。错。decode 阶段是 memory-bound，慢的不是算力，是**读权重的带宽**——下面会展开。

### 1.2 Prefill vs Decode：同一张图，两条 roofline [理论]

| 阶段 | 计算形状 | 算术强度 (FLOPs / Byte) | bound | 典型 MFU  |
|---|---|---|---|---|
| Prefill | $[N, d] \times [d, d']$，矩阵-矩阵 | 几百 ~ 几千 | **compute-bound** | 0.35 ~ 0.55 |
| Decode | $[1, d] \times [d, d']$，矩阵-向量 | 1 ~ 2 | **memory-bound** | 0.05 ~ 0.15 |

H100 BF16 的 ridge point（拐点）：

$$
I^* = \frac{P_{\text{compute}}}{B_{\text{HBM}}} = \frac{989\ \text{TFLOPs/s}}{3.35\ \text{TB/s}} \approx 295\ \text{FLOPs/Byte}
$$

decode 的算术强度只有 1~2，离这个拐点差**两个数量级以上**——这就是为什么 decode 阶段 GPU 算力利用率永远上不去 10%。

**工程含义**：
- **想提升 prefill 吞吐**：堆 batch、堆 FlashAttention、用 FP8。
- **想提升 decode 吞吐**：堆 batch（让权重读取被多个 token 摊分）、量化权重（W4/W8）、PagedAttention 减少 KV 浪费。
- 不要混用药方。decode 上 FP8 收益远不如 W4A16，因为它瓶颈不在算力。

### 1.3 KV Cache：每 token 多少 KB [理论]

$$
s_{\text{kv}} = 2 \cdot L \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot \text{bytes}
$$

| 模型 | 配置 | $s_{\text{kv}}$ / token (BF16) | 32k 上下文 | 128k 上下文 |
|---|---|---|---|---|
| Llama-3-8B（GQA, 8 KV heads, 32 层） | $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2$ | **128 KB** | 4 GB | 16 GB |
| Llama-3-70B（GQA, 8 KV heads, 80 层） | $2 \cdot 80 \cdot 8 \cdot 128 \cdot 2$ | **320 KB** | 10 GB | 40 GB |
| GPT-3-175B（MHA, 96 KV heads, 96 层） | $2 \cdot 96 \cdot 96 \cdot 128 \cdot 2$ | **4.5 MB** | 144 GB | 576 GB |
| DeepSeek-V3（MLA） | 压缩到 ~70 KB / token  | 70 KB | 2.2 GB | 8.7 GB |

**认知误区二**："长上下文 = 算力问题"。错。**长上下文 90% 的痛苦是 KV Cache 装不下**：175B MHA 在 128k 上下文下，单个请求就要 576 GB KV，而一张 H100 只有 80 GB。这是 GQA / MQA / MLA 这类 attention 变体存在的全部理由。

### 1.4 Decode 单 token 时延的下限 [理论]

decode 受限于"把所有权重读一遍"，下限是：

$$
t_{\text{decode}}^{\min} = \frac{\Theta \cdot \text{bytes}}{B_{\text{HBM}}}
$$

| 模型/精度 | 权重大小 | H100 (3.35 TB/s) | H200 (4.8 TB/s) | B200 (8 TB/s)  |
|---|---|---|---|---|
| 70B BF16（单卡装不下，按 TP=8 后单卡 17.5 GB 算） | 17.5 GB / 卡 | **5.2 ms** | 3.6 ms | 2.2 ms |
| 70B INT8（TP=4，单卡 17.5 GB） | 17.5 GB / 卡 | **5.2 ms** | 3.6 ms | 2.2 ms |
| 70B INT4（单卡 35 GB） | 35 GB | **10.4 ms** | 7.3 ms | 4.4 ms |
| 8B BF16（单卡 16 GB） | 16 GB | **4.8 ms** | 3.3 ms | 2.0 ms |

> 注意：上面是**理论下限**，对应 batch=1。batch 越大，单 token 时延几乎不变（权重只读一次），但**吞吐**线性增长——这就是 decode 阶段疯狂凑 batch 的根本原因。
>
>  vLLM / SGLang 在 H100 上 70B BF16 TP=8、batch=1 的端到端 decode 时延约 **8~12 ms / token**（含 attention、调度、采样开销），和 5.2 ms 的理论下限差 ~2× 是合理工程损耗。

### 1.5 一张推理速查表

| 量 | 70B BF16 / H100 TP=8 | 备注 |
|---|---|---|
| 模型大小 | 140 GB | $\Theta \times 2$ |
| 单卡权重 | 17.5 GB | TP=8 摊分 |
| KV / token | 320 KB | GQA 救命 |
| Prefill 1 token | ~0.5 ms [理论] / ~1 ms  | compute-bound |
| Decode 1 token | ~5 ms [理论] / ~10 ms  | memory-bound |
| 1k 输入 + 1k 输出 端到端 | ~10s  | 长尾主要在 decode |
| 每 token 输出 throughput（batch=64） | ~30 tokens/s/user  | concurrency 越高越摊薄 |

---

## 二、训练：每一步 step 烧多少卡时

### 2.1 一条公式打底：训练 ≈ $6\Theta$ FLOPs / token [理论]

来源：
- 前向 $2\Theta$
- 反向需要算两次（对输入梯度 + 对权重梯度）：$4\Theta$
- 合计 $6\Theta$

这条经验公式（出自 Kaplan 2020 / Chinchilla 论文）适用于 dense Transformer，不算 activation recomputation。

**Chinchilla 比例 **：最优训练 token 数 $D \approx 20\,\Theta$，所以训练总 FLOPs：

$$
F_{\text{train}} \approx 6\,\Theta \cdot 20\,\Theta = 120\,\Theta^2
$$

| 模型 | 训练 token | 训练总 FLOPs [理论] | 等价 H100·days [理论, MFU=0.4] |
|---|---|---|---|
| Llama-3-8B | 15 T | $7.2 \times 10^{23}$ | ~21 H100·days |
| Llama-3-70B | 15 T | $6.3 \times 10^{24}$ | ~180 H100·days = **6k H100 训 1 个月**（实测 Llama 3 用了 ~6.4k H100 跑 ~50 天 ）|
| Llama-3-405B | 15 T | $3.6 \times 10^{25}$ | ~16k H100 训 60 天 [实测，对得上]|
| GPT-4 (估算) | ~13 T | ~$2 \times 10^{25}$ | ~25k A100 训 ~3 个月 [实测/估算]|

> **认知误区三**："训练比推理贵 1000 倍"。这种说法太粗。准确的说法是：**训练 1 个 token 比推理 1 个 token 贵 3 倍**（$6\Theta$ vs $2\Theta$）。但训练要吃 $20\Theta$ 个 token 跑一遍，所以**训练总成本 ≈ 推理 60× per-token 成本 × 20Θ token ≈ 60Θ 倍 推理单 token**。这才是 1000× 量级的来源。

### 2.2 预训练 vs 后训练：两个数量级的差距 [理论 + 实测]

上面 $6\Theta \cdot D \approx 120\Theta^2$ 的账只描述了**预训练**。一个完整的 LLM 生命周期，至少要再走 SFT、RLHF/DPO 等若干"后训练"阶段。它们用的是**同一个模型，但完全不同的算力 / 数据 / 系统画像**——不分清楚就会拿预训练 infra 直接套，杀鸡用牛刀且抓不到真正瓶颈。

| 阶段 | 数据量级 | 主要 FLOPs 公式 | 70B 总 FLOPs [理论] | 典型时长  | 主瓶颈 |
|---|---|---|---|---|---|
| **Pretrain（预训练）** | 10~15 T token | $6\Theta \cdot D$ | $\sim 6\times 10^{24}$ | 千卡 × 数十天 | 算力 + 通信 |
| **CPT（继续预训练）** | 100 B ~ 1 T token | $6\Theta \cdot D$ | $10^{22} \sim 10^{23}$ | 百卡 × 数天 | 算力 |
| **SFT（指令微调）** | 10 K ~ 10 M 条样本（~$10^7$~$10^{10}$ token） | $6\Theta \cdot D$ | $10^{18} \sim 10^{21}$ | 数十卡 × 小时~天 | 显存 |
| **DPO / RPO**（偏好对齐 offline） | $10^4 \sim 10^6$ pair | $\sim 12\Theta \cdot D$（双前向）| $10^{19} \sim 10^{21}$ | 数十卡 × 小时 | 显存 |
| **RLHF (PPO)** | $10^4 \sim 10^5$ prompt | 训练 $6\Theta \cdot D$ **+** 大量 rollout 推理 | $10^{20} \sim 10^{22}$ | 百卡 × 数天 | **推理 throughput** |
| **RLVR / GRPO**（推理任务 RL） | $10^4 \sim 10^5$ prompt | 同 RLHF，rollout 占主导 | $10^{20} \sim 10^{22}$ | 百卡 × 数天 | **推理 throughput** |

几个关键反差：

1. **算力差 2~4 个数量级**——70B 预训练 ~$6 \times 10^{24}$ FLOPs，SFT 通常 $10^{20}$ 量级，差 4 个 0。这是为什么 SFT 一夜跑完很正常，预训练要跑两个月。
2. **显存账完全没省**——参数量 $\Theta$ 没变，BF16+Adam 的 $16\Theta$ 静态显存照样要交。70B SFT 仍要 ~1.1 TB 显存，仍要 FSDP / ZeRO-3。"后训练便宜"指的是**算力和数据**便宜，不是**显存**便宜。
3. **RLHF / RLVR 的瓶颈是推理，不是训练**——一个 PPO step 里通常 60%~80% 时间花在 **actor rollout**（生成 response），而 rollout 是典型 decode 任务，**memory-bound**。所以 RL 集群的工程画像更像"推理服务 + 一点训练"。
4. **DPO ≈ 两次 SFT 前向，没有 rollout**——所以 DPO 的算力账跟 SFT 几乎在一个数量级，**便宜、稳定、可复现**，这是它在工业界压过 PPO 的根本原因。
5. **后训练数据 IO 不是问题**——10 M 样本 × 几 KB ≈ GB 级，单机 NVMe 装得下，根本用不上 GPFS / Weka。**预训练那一套数据 infra 在后训练阶段是负优化**（启动开销 > 收益）。
6. **后训练 ckpt 频率更高**——预训练一天 1~2 个 ckpt，SFT/RLHF 经常一天 10+ 个，因为 reward 容易跑飞要回滚。每个 ckpt 仍然 1.1 TB（70B），异步写是必选项。

**工程决策**：

- **RL 集群的典型形态不是"训练集群"**：常见配比 **1 份 train GPU : 4~8 份 inference GPU（actor rollout）**。把 rollout 单独拆出去（vLLM / SGLang），训练侧只做 policy update，是 OpenRLHF / veRL / Areal 这些框架的核心设计。
- **能用 DPO 就别上 PPO**：算力差一个数量级，工程复杂度差两个数量级。只有当 reward 信号难以离线收集（如代码 / 数学 RLVR）时才上 on-policy RL。
- **SFT 别用预训练那套 infra**：预训练为吞吐设计（大 batch、长 step、TB/s 数据流），SFT 数据小、step 短、需要快速实验迭代——更接近"大规模微调"而不是"小规模预训练"。

本节后面的"显存占用 / 并行通信 / 训练速查表"主要按**预训练**画像展开，因为它是规模最大、瓶颈最多的一档。后训练的数字按上表的算力比例缩放即可。

### 2.3 显存占用：参数只是冰山一角 [理论]

训练时 GPU 上每个参数对应的内存（以 BF16 训练 + Adam 为例）：

| 项 | 字节/参数 | 70B 总量 |
|---|---|---|
| 模型权重 (BF16) | 2 | 140 GB |
| 梯度 (BF16) | 2 | 140 GB |
| Adam m (FP32) | 4 | 280 GB |
| Adam v (FP32) | 4 | 280 GB |
| Master weights (FP32) | 4 | 280 GB |
| **小计（静态）** | **16** | **1120 GB** |

**没错——光是静态状态，70B 训练就要 1.1 TB 显存。** 一张 H100 80 GB 装不下零头。这是为什么训练必须用 ZeRO / FSDP 切分。

加上**激活值**（activation）：

$$
M_{\text{act}} \approx s \cdot b \cdot L \cdot (34 \cdot d + 5 \cdot s \cdot h) \cdot \text{bytes}
$$
（Megatron 论文 Table 2 [理论]）

| 配置 | 激活内存 [理论] |
|---|---|
| 70B, seq=4k, batch=1, 不重计算 | ~80 GB |
| 70B, seq=8k, batch=4, 不重计算 | ~640 GB |
| 70B, seq=8k, batch=4, 全重计算 | ~30 GB |

**工程决策**：activation recomputation（gradient checkpointing）几乎是必选项，能把 activation 砍 10×~20×，代价是 forward 多算一遍（训练总 FLOPs 从 $6\Theta$ 涨到 $\sim 8\Theta$，约 +33%）。**这笔 33% 的算力税，是显存窗口的最便宜门票**。

### 2.4 各种并行的通信量 [理论]

| 并行方式 | 切什么 | 每 step 通信量（70B 单层） | 通信类型 |
|---|---|---|---|
| **Data Parallel (DP)** | batch 切到不同卡，每卡完整模型 | 全模型梯度 AllReduce: $\sim 2\Theta$ = 280 GB / step | AllReduce |
| **ZeRO-3 / FSDP** | 参数 + 梯度 + 优化器都切 | 每层都要 AllGather 权重 + ReduceScatter 梯度，**总量 ≈ 3× DP** | AllGather + ReduceScatter |
| **Tensor Parallel (TP)** | 切矩阵列 / 行 | 每层 2 次 AllReduce，每次 $b \cdot s \cdot d$ | AllReduce |
| **Pipeline Parallel (PP)** | 切层 | 层间 send/recv：$b \cdot s \cdot d$ / micro-batch | P2P |
| **Sequence Parallel** | 切 sequence 维 | LayerNorm / Dropout 内部 AllGather | AllGather |
| **Expert Parallel (MoE)** | 切 expert | All2All：$b \cdot s \cdot d$ 量级，每 layer 2 次 | All2All |

具体数字（70B, batch=4M token, BF16, 1024 张 H100, MFU=0.4）：

- 1 step 计算时间 [理论]：$\frac{6 \cdot 70 \times 10^9 \cdot 4 \times 10^6}{1024 \cdot 989 \times 10^{12} \cdot 0.4} \approx 4.1\ \text{s}$
- DP 梯度 AllReduce：280 GB，需要 4.1s 内完成 → 至少 **70 GB/s 等效带宽**
- NVLink 4.0 单向 900 GB/s ，AllReduce 等效带宽 ~400 GB/s ，单机内绰绰有余
- 跨机走 IB NDR（400 Gb/s = 50 GB/s ），AllReduce 跨机会立刻成为瓶颈 → 必须 hierarchical AllReduce

**认知误区四**：很多人以为 "TP=8 比 DP=8 慢"。其实 TP 只在**单机内**跑（NVLink），AllReduce 在 1 ms 量级；DP **跨机**跑（IB），AllReduce 在百 ms 量级。**TP 是为了让模型装下，DP/ZeRO 是为了 scale 训练数据，二者目标完全不同**。

### 2.5 一张训练速查表（70B / 1024 H100）

| 量 | 数值 | 备注 |
|---|---|---|
| 静态显存（per param） | 16 B / param | BF16 + Adam |
| 70B 静态显存总量 | 1120 GB | TP=8 + ZeRO-1 后单卡 ~140 GB |
| Activation（recompute） | ~30 GB / 卡 | 必须开 |
| 单 step 计算 | ~4 s [理论] | MFU=0.4 |
| 单 step 梯度 AllReduce | 280 GB | 跨机 ~6 s 不开 hierarchical → 反而成瓶颈 |
| 训练 1.5T token | ~50 天  | Llama 3 70B 数量级 |
| 单卡日志/秒 throughput | ~3.5k token/s  | H100 BF16 |

---

## 三、GPU：四个关键数能不能背下来

每张训练/推理卡的命运，由四个数决定：**算力、显存容量、显存带宽、互联带宽**。下表全部 ，来源是各家 datasheet。

### 3.1 NVIDIA 主流训推卡

| GPU | BF16 算力 (TFLOPs) | FP8 算力 (TFLOPs) | HBM 容量 | HBM 带宽 (TB/s) | NVLink 单向 (GB/s) | 
|---|---|---|---|---|---|
| **A100 80GB SXM** | 312 | — | 80 GB HBM2e | 2.0 | 600 (NVLink 3) | 
| **H100 SXM** | 989 | 1979 | 80 GB HBM3 | 3.35 | 900 (NVLink 4) | 
| **H200 SXM** | 989 | 1979 | 141 GB HBM3e | 4.8 | 900 | 
| **B200 SXM** | 2250 | 4500 | 192 GB HBM3e | 8.0 | 1800 (NVLink 5) | 
| **GB200 (1×Grace+2×B200)** | 4500 | 9000 | 384 GB HBM3e | 16.0 | C2C: 900 | 
| **B100** | 1750 | 3500 | 192 GB HBM3e | 8.0 | 1800 |
| **L40S** | 91.6 | 366 | 48 GB GDDR6 | 0.864 | 无 NVLink | 

> **FP8 不等于 BF16 × 2**。FP8 有 E5M2 / E4M3 两种格式，训练的有效精度损失比例约 1~3%（Llama-3 全 FP8 ），但 attention softmax 经常被迫保留 BF16，所以**端到端 MFU 提升通常只有 1.5×~1.7×，而不是理论的 2×**。

### 3.2 算力 vs 带宽 vs 容量：三个量谁是瓶颈

| 工作负载 | 主瓶颈 | 关键参数 |
|---|---|---|
| **训练（compute-bound）** | 算力 | TFLOPs，FP8 加成 |
| **推理 prefill** | 算力 | TFLOPs |
| **推理 decode** | HBM 带宽 | TB/s |
| **长上下文 KV** | HBM 容量 | GB |
| **大模型部署装下** | HBM 容量 + NVLink | GB + GB/s |
| **MoE All2All** | NVLink + 网卡 | GB/s |

**工程决策**：
- 70B 推理生意，H200 比 H100 好用得不是一点点：**HBM 翻 1.4×（141 vs 80）+ 带宽翻 1.43×（4.8 vs 3.35）**，decode 时延几乎线性下降。
- 405B 训练，B200 的优势主要不是 BF16 算力，而是 **192 GB 显存让 TP 可以从 8 降到 4**，省一半通信。
- 如果你只跑 7B / 13B 推理 batch service，**L40S 性价比远胜 H100**（每 GB/s 带宽便宜 ~3×）。

### 3.3 互联：从卡到机房的带宽阶梯 

```text
HBM3       (片内)          : 3,350 GB/s
NVLink 4   (8 卡同机内)     :   900 GB/s   ← H100 NVSwitch
NVLink 5   (B200/GB200)     : 1,800 GB/s
PCIe Gen5 x16 (CPU↔GPU)    :    64 GB/s   (单向)
IB NDR 400G (机间)          :    50 GB/s   ← AI 训练集群标配
IB HDR 200G                 :    25 GB/s
100 GbE                     :    12.5 GB/s
NVMe Gen4 SSD               :     7 GB/s
SATA SSD                    :     0.5 GB/s
HDD                         :     0.2 GB/s
```

**注意一个数量级落差**：**HBM → NVLink → PCIe → IB → NVMe 每一跳都掉一个数量级**。这是所有 AI infra 决策的物理基础——**链路上最窄的那一段决定一切**（参见 [KV Cache 是算还是传？](/posts/2026-06-10-kvcache-compute-vs-transfer/)）。

**实测打折系数 **：
- HBM：可用 ~85% × 标称
- NVLink AllReduce：~50%~70% × 标称（NCCL）
- PCIe DMA：~80% × 标称
- IB AllReduce：~60%~80% × 标称（取决于消息大小）
- NVMe 顺序读：~70%~90% × 标称；随机 4k 读：**1%~5% ×** 标称（这是 IO 设计最大坑）

### 3.4 一台典型 H100 节点的"硬件配方"

```text
                  ┌──────────────────────┐
                  │  2× CPU (Sapphire R) │  ─── PCIe Gen5 ────┐
                  └──────────┬───────────┘                    │
                             │ 1.5 TB DRAM                    │
                             │                                │
            ┌────────────────┴────────────────┐               │
            │                                 │               │
       ┌────▼────┐ NVLink Switch 900 GB/s ┌──▼─────┐          │
       │ H100×8  ├──────────────────────► │ H100×8 │          │
       │ 8×80GB  │                        │ 8×80GB │          │
       └────┬────┘                        └────┬───┘          │
            │                                  │              │
            └──── 8× ConnectX-7 NIC (400G IB) ─┴──► 集群 fabric
                                                              │
                                                ┌─────────────▼──┐
                                                │ NVMe SSD ×8    │ 56 GB/s
                                                │ (本地数据缓存)  │
                                                └────────────────┘
```

**单节点关键数字**：
- 算力：8 × 989 = **7.9 PFLOPs BF16**
- 显存：8 × 80 = **640 GB**
- HBM 带宽：8 × 3.35 = **26.8 TB/s**（卡内）
- 节点内 NVLink：900 GB/s × 双向（NVSwitch 全互联）
- 出节点：8 × 50 = **400 GB/s**（IB）
- TDP：~10 kW（不算 CPU/网卡/PSU 损耗，整机 ~12 kW）

> **认知误区五**："多买点卡训练就快了"。错。**训练吞吐 = min(算力, 通信)**。如果你 IB 网络只有 100G，加再多卡也只是堆通信瓶颈。Llama 3 团队在 405B 训练里**专门重写了 NCCL 调度**，因为默认 ring AllReduce 在 16k 卡规模根本撑不住。

---

## 四、存储与 IO：被严重低估的训练瓶颈

GPU 工程师最爱算 FLOPs，但训练 run 里**最容易卡死的不是算力，是 IO**。

### 4.1 训练数据的 IO 带宽需求 

每 step 要读多少数据？

$$
B_{\text{IO}} = \frac{\text{batch\_tokens} \cdot \text{bytes/token}}{t_{\text{step}}}
$$

70B 训练为例：
- batch = 4M token，$t_{\text{step}}$ ≈ 4 s 
- bytes/token：纯 text token id（int32）= 4 B；但**实际训练读的是 raw text**，UTF-8 平均 ~4 B/char ≈ **15 B/token** （含 padding、metadata）
- $B_{\text{IO}}$ = 4M × 15 / 4 ≈ **15 GB/s 单节点**

**这只是 1 个节点**。1024 卡 = 128 节点，全集群读吞吐 = **128 × 15 = ~2 TB/s**。

| 存储方案 | 可用读带宽  | 70B 训练能否扛住？ |
|---|---|---|
| 单机 NVMe ×8 | ~50 GB/s | 单节点够，但每节点要存全量 → 不现实 |
| GPFS（IBM Spectrum Scale，100 节点级） | 100~400 GB/s | 中型集群够 |
| Lustre（百节点级） | 200~800 GB/s | 大型 HPC 集群标配 |
| **WekaFS / DAOS** | 1~5 TB/s | 顶级 LLM 集群（Meta、xAI）|
| S3 直读 | 单 client 1~5 GB/s | **必须本地缓存** |
| HDFS | 10~50 GB/s（小文件杀手）| 不适合 |

**工程决策**：
- 1k 卡以下，本地 NVMe + 数据预取就够；
- 1k~10k 卡，必须 GPFS / Lustre / Weka；
- 10k+ 卡，存储是**独立工程团队**而不是"运维一下"。Meta 在 Llama 3 用了 **240 PB Tectonic 存储 + 7000+ 客户端节点**。

**认知误区六**："SSD 7 GB/s 应该够吧？" 7 GB/s 是**单盘顺序读**。训练 dataloader 是 **多 worker 随机读小文件**，实测掉到 **0.1 ~ 1 GB/s**，瞬间就成瓶颈。所以训练数据**几乎一定要打包成大文件**（webdataset / mosaic streaming / Megatron 的 indexed dataset），不能直接放散文件。

### 4.2 Checkpoint 存储 

Checkpoint 大小：

$$
S_{\text{ckpt}} = \Theta \cdot (\text{weight bytes} + \text{grad bytes} + \text{optimizer bytes})
$$

| 模型 | 仅权重 (BF16) | 全 ckpt（含 Adam FP32） | 备注 |
|---|---|---|---|
| 70B | 140 GB | **~1.1 TB** | 训练必须存 full ckpt |
| 405B | 810 GB | **~6.5 TB** | 单 ckpt 已是磁带级 |
| 1.8T MoE | 3.6 TB | **~29 TB** | 一次 ckpt 写入要数小时 |

**实际训练里要存多少？**

$$
S_{\text{total}} = S_{\text{ckpt}} \times \underbrace{f_{\text{freq}}}_{\text{保留多少版本}} \times \underbrace{n_{\text{rep}}}_{\text{副本数}} \times \underbrace{n_{\text{exp}}}_{\text{实验数}}
$$

典型配置：
- $f_{\text{freq}}$ = 10（保留最近 10 个 ckpt）
- $n_{\text{rep}}$ = 3（异地 + 同城副本）
- $n_{\text{exp}}$ = 5（同时跑的训练实验）

70B：$1.1\text{ TB} \times 10 \times 3 \times 5 = $ **~165 TB**
405B：$6.5\text{ TB} \times 10 \times 3 \times 5 = $ **~975 TB ≈ 1 PB**

**写入带宽是另一个杀手** ：
- 70B ckpt = 1.1 TB，要在 60 s 内写完（不阻塞训练）→ 需要 **~18 GB/s 写带宽**
- 405B ckpt = 6.5 TB，写满 60 s 需要 **108 GB/s**
- 这意味着 ckpt 必须 **shard + 并行写**，不能单 rank 收集后串行写。Llama 3 用的是**异步 ckpt + 后台 flush**。

**认知误区七**："ckpt 只是备份，慢点没关系"。错。ckpt 阻塞训练每分钟都是百卡时损失。一次 1 小时 ckpt 在 16k H100 集群上 = **16000 H100·hours = $40,000+** 直接损失。

### 4.3 Tokenization 与数据 pipeline：一条容易忽略的红线

Tokenization 速度（CPU 端，BPE / SentencePiece）：

| 实现 | 单核 throughput  | 备注 |
|---|---|---|
| HuggingFace Tokenizers (Rust) | ~1~5 MB/s | 多线程可线性扩展 |
| sentencepiece C++ | ~3~10 MB/s | |
| tiktoken (Rust) | ~10~30 MB/s | OpenAI 出品，最快 |

训练消耗 token 速度（70B，1024 H100）：
- 4M token / 4 s = **1 M token/s** = **~3~5 MB/s（按 4 B/token 算 id）**
- **看起来 tokenizer 单核就够了？** 错。问题在于：
  1. 训练读的是 **raw text**（~15 B/token），单节点 15 GB/s，再 ×128 节点；
  2. 边训边 tokenize 等于把 GPU 等 CPU；
  3. 实际工程做法：**离线 tokenize 一次，存成 .bin/.idx 大文件**（Megatron-LM IndexedDataset），训练只做 mmap 读取。

**一条铁律**：
> **tokenization 必须是离线一次性工作。在线 tokenize = GPU 利用率掉 30%+。**

数据 pipeline 完整账本（per node, 70B 训练）：

```text
[远端存储 GPFS]  ──── 10 GB/s 读 ────►  [本地 NVMe 缓存]
                                              │
                                              │ 50 GB/s mmap
                                              ▼
                              [DataLoader workers × N]
                                              │
                                              │ shuffle + collate (CPU)
                                              ▼
                                  [Pinned host memory]
                                              │
                                              │ 25 GB/s PCIe Gen5
                                              ▼
                                      [GPU HBM]
                                              │
                                              ▼
                                          训练
```

**任何一段堵塞，整个 pipeline 失速**。最常见的两种事故：
1. **NVMe 缓存被驱逐**：epoch 切换时数据集被换出，下个 epoch 直接命中远端 → IO 掉到 1 GB/s 量级；
2. **DataLoader worker 不够**：默认 4 worker 在大 batch 下吃不饱，要按 `num_workers ≈ batch_size / (CPU 单核 tokenize 速度 × seq_len)` 估。

**认知误区八**："数据准备是 CSE 学生的活"。错。Llama 3 paper 里关于数据 pipeline 的内容（去重、质量过滤、curriculum、tokenize、pack）**比 attention kernel 的篇幅多 5 倍**。**真正决定大模型质量的是数据 infra，不是 GPU kernel**。

### 4.4 一张存储/IO 速查表

| 量 | 70B 训练（1024 H100） | 405B 训练（16k H100） |
|---|---|---|
| 训练数据读带宽 | ~2 TB/s 集群级 | ~30 TB/s |
| Ckpt 单次大小 | 1.1 TB | 6.5 TB |
| Ckpt 写带宽要求 | 18 GB/s | 108 GB/s |
| Ckpt 总存储 | 165 TB | ~1 PB |
| Tokenize 阶段 | 离线，一次性 | 离线，一次性 |
| 推荐文件系统 | GPFS / Lustre | WekaFS / DAOS |
| 单节点 NVMe 缓存 | 8× 7 GB/s = 56 GB/s | 同 |

---

## 五、MoE：当 dense 公式集体失效

前面四节有一个隐藏前提：**所有公式都是按 dense Transformer 推的**。一旦切到 MoE（Mixture of Experts），$2\Theta$ / $6\Theta$ / $16\Theta$ 这些常数会同时分裂成两个版本——**激活参数 $\Theta_a$** 和 **总参数 $\Theta_t$**。不分清楚，账就全错了。

### 5.1 MoE 的两个 $\Theta$ [理论]

$$
\Theta_t = \Theta_{\text{shared}} + N_{\text{exp}} \cdot \Theta_{\text{expert}}, \qquad
\Theta_a = \Theta_{\text{shared}} + k \cdot \Theta_{\text{expert}}
$$

- $N_{\text{exp}}$：总专家数
- $k$：每个 token 激活的专家数（top-k 路由）
- $\Theta_{\text{shared}}$：attention + 共享专家（如 DeepSeek 的 shared expert）

**一条铁律**：

| 量 | 看哪个 $\Theta$ |
|---|---|
| 推理算力 / token FLOPs | $\Theta_a$（只激活的专家在算） |
| 训练算力 / token FLOPs | $\Theta_a$ |
| **GPU 显存（权重）** | $\Theta_t$（所有 expert 都得装下） |
| **Ckpt 大小** | $\Theta_t$ |
| **梯度 / 优化器状态** | $\Theta_t$ |
| Activation 内存 | 跟 dense 接近，只看 hidden + seq |
| All2All 通信量 | 看 hidden × tokens，跟 $\Theta$ 无关 |

这就是 MoE 的根本张力：**算力账走 $\Theta_a$，显存账走 $\Theta_t$**。两者经常差 10× 以上。

### 5.2 主流 MoE 模型对照表 

| 模型 | $\Theta_t$ | $\Theta_a$ | 路由 | $\Theta_t/\Theta_a$ | BF16 总权重 |
|---|---|---|---|---|---|
| Mixtral 8×7B | 47 B | 13 B | top-2 of 8 | 3.6× | 94 GB |
| Mixtral 8×22B | 141 B | 39 B | top-2 of 8 | 3.6× | 282 GB |
| Qwen3-235B-A22B | 235 B | 22 B | top-8 of 128 | 10.7× | 470 GB |
| **DeepSeek-V3** | 671 B | 37 B | top-8 of 256 + 1 shared | **18.1×** | **1342 GB** |
| Kimi-K2（推测）| ~1 T | ~32 B | top-? | ~30× | ~2 TB |
| GPT-4（估算）[实测/估算] | ~1.8 T | ~280 B | top-2 of 16 | ~6× | ~3.6 TB |

**直觉**：现代 MoE 在朝 **$\Theta_t/\Theta_a \geq 15$** 走（DeepSeek-V3 是分水岭）。这意味着同等推理算力下，模型容量翻 10 倍以上——但显存和通信账要重新算。

### 5.3 MoE 推理：一笔典型的"算便宜，存贵"账

以 **DeepSeek-V3（671B 总 / 37B 激活）** vs **Llama-3-70B（dense）** 对比，BF16，H100：

| 维度 | DeepSeek-V3 | Llama-3-70B | 谁赢 |
|---|---|---|---|
| 单 token 前向 FLOPs | $2\Theta_a$ = 74 GFLOPs | 140 GFLOPs | **MoE 省 47%** |
| 模型权重总量 | 1342 GB | 140 GB | dense 省 9.6× |
| 单卡装下需要 | EP=16 (H100 80GB) 起步 | TP=2 起步 | dense 省卡 |
| Decode 单 token 时延下限 | $\frac{1342/16}{3.35\text{TB/s}}$ ≈ 25 ms | 5 ms | dense 快 5× |
| KV / token (MLA 压缩 ) | ~70 KB | 320 KB | **MoE 省 4.6×** |
| 端到端 decode（batch=1） | ~30~40 ms / token | ~10 ms / token | dense 快 3× |
| 端到端 decode（batch=128） | ~50 ms / token, 60+ tok/s/user | ~25 ms / token | dense 快 2× |

**结论**：
- **小 batch 推理 MoE 反而更慢**——因为每张卡虽然激活的 expert 算得少，但**仍然要把整张卡上的全部 expert 权重读一遍才能 routing**（部分实现），或者 expert 利用率极低，HBM 带宽被浪费；
- **大 batch 推理 MoE 才划算**——batch 大了之后 expert 普遍激活，单 expert 摊到的 token 数变多，权重读取被复用；
- DeepSeek-V3 自己的优化（MTP + MLA + FP8）能把 decode 拉到 60+ tok/s ，已经是 MoE 工程的天花板。

**认知误区九**："MoE 比 dense 推理便宜"。**只在 batch 足够大时成立**。online serving 长尾 batch 通常很小，MoE 的 MFU 实测会掉到 dense 的一半以下。

### 5.4 MoE 训练：All2All 是新的瓶颈 [理论]

MoE 每层多了 **2 次 All2All**：

```text
          input tokens [B*S, d]
                 │
                 ▼  router (gate)
         ┌──────────────────┐
         │  All2All dispatch │  把 token 按 expert 分发到对应 GPU
         └────────┬──────────┘
                  ▼
           ┌─────────────┐
           │  Expert FFN │  各 GPU 只算自己的 expert
           └──────┬──────┘
                  ▼
         ┌──────────────────┐
         │  All2All combine  │  把结果送回原 GPU
         └────────┬──────────┘
                  ▼
              output
```

**通信量公式 [理论]**：

$$
V_{\text{a2a}} = 2 \cdot k \cdot B \cdot S \cdot d \cdot \text{bytes}
$$

（dispatch + combine，每个 token 发出去 $k$ 份）

以 DeepSeek-V3 训练（$d=7168$, $k=8$, $B \cdot S$ = 4M token, BF16, 每层）为例：

$$
V_{\text{a2a}} = 2 \times 8 \times 4 \times 10^6 \times 7168 \times 2 \approx 920\ \text{GB / 层}
$$

61 层 × 920 GB = **~56 TB / step / 集群**。如果 step 时间 4 s，要求 **14 TB/s 集群级 All2All 带宽**——这个数字逼近现役顶级 IB fabric 的极限。

**对比 dense 70B 训练的 AllReduce**：280 GB / step，差了**两个数量级**。

**工程含义**：
- MoE 训练**对 NIC 带宽极度敏感**，IB NDR 400G（50 GB/s）是底线，HDR 200G 几乎不可用；
- DeepSeek 在 V3 论文里专门设计了 **device-limited routing**（每个 token 最多路由到 4 个节点），就是为了**把 All2All 压到节点内 NVLink 域**；
- expert 越多 / $k$ 越大，All2All 越贵——这就是为什么不是所有人都敢上 256 专家。

### 5.5 专家并行 (EP) 与并行组合

MoE 多了一个并行维度 **EP（Expert Parallel）**，把 expert 切到不同 GPU：

| 并行 | 切什么 | MoE 里的角色 |
|---|---|---|
| TP | hidden 维 | 切共享部分 + 单个 expert 内部 |
| PP | 层 | 同 dense |
| **EP** | expert（FFN 模块）| **MoE 独有**：每张卡放 $N_{\text{exp}}/EP$ 个 expert |
| DP / ZeRO | batch / 状态 | 同 dense |

**典型配置**（DeepSeek-V3 训练 ）：

```text
TP = 1     (MLA 让 attention 不需要 TP)
PP = 16    (沿 61 层切)
EP = 64    (256 expert / 64 = 4 expert/卡)
DP = ...   (剩余维度撑 batch)
```

**工程决策**：
- **EP × TP ≤ 节点内卡数**（避免跨节点 All2All）——这是几乎所有 MoE 训练的硬约束；
- 共享 expert + device-limited routing，本质都是把通信压回到 NVLink 域；
- 小集群跑 MoE 几乎不可行，**MoE 是天生的大集群技术**。

### 5.6 MoE 的显存与 ckpt 账

训练静态显存（BF16 + Adam）：

$$
M_{\text{static}} = 16 \cdot \Theta_t
$$

| 模型 | 静态显存 | 单 ckpt | 训练实验 5 份 × 副本 3 × 版本 10 |
|---|---|---|---|
| Mixtral 8×22B | 2.3 TB | 2.3 TB | ~340 TB |
| DeepSeek-V3 | **10.7 TB** | **10.7 TB** | **~1.6 PB** |
| GPT-4（估算） | ~28.8 TB | ~28.8 TB | ~4 PB |

**Ckpt 写入时间**：DeepSeek-V3 一次 ckpt 10.7 TB，要 60 s 内写完需要 **~180 GB/s 集群写带宽**——比 dense 70B 高一个数量级。**异步 ckpt + sharded write 是 MoE 训练的必选项**。

### 5.7 MoE 速查表

| 量 | 公式 | DeepSeek-V3 |
|---|---|---|
| 推理 FLOPs / token | $2\Theta_a$ | 74 GFLOPs |
| 训练 FLOPs / token | $6\Theta_a$ | 222 GFLOPs |
| 权重显存 | $\Theta_t \cdot \text{bytes}$ | 1.34 TB (BF16), 671 GB (FP8) |
| 静态训练显存 | $16\Theta_t$ | 10.7 TB |
| Ckpt 大小 | $16\Theta_t$ | 10.7 TB |
| All2All / step / 层 | $2 k \cdot B \cdot S \cdot d \cdot \text{bytes}$ | ~920 GB（4M token）|
| KV / token (MLA) | 压缩公式 | ~70 KB |
| EP × TP 上限 | ≤ 节点卡数 | ≤ 8（同机内）|

### 5.8 三条 MoE 反直觉

1. **MoE 不便宜**——算力便宜，但**显存 + 通信 + ckpt 都贵一个数量级**。MoE 不是穷人方案，是大厂方案。
2. **小 batch 上 MoE 是负优化**——expert 利用率不足，HBM 带宽白浪费。<2k QPS 的服务跑 dense 更划算。
3. **MoE 的真正价值是"模型容量"不是"推理速度"**——GPT-4 / DeepSeek-V3 之所以 MoE，是因为想塞进 1T+ 总参数同时控制 per-token 算力，**不是为了更快**。

---

## 六、把五张表叠在一起：决策视角

最后，把上面四节的关键数字叠成一张可贴墙的"工程决策卡"：

### 5.1 推理 vs 训练的物理本质对比

| 维度 | 推理 (decode) | 训练 |
|---|---|---|
| FLOPs / token | $2\Theta$ | $6\Theta$（无重计算）/ $\sim 8\Theta$（开重计算）|
| 主瓶颈 | HBM 带宽 | 算力 + 通信 |
| 关键 GPU 指标 | HBM TB/s, HBM GB | TFLOPs, NVLink, IB |
| MFU | 0.05~0.15 | 0.35~0.55 |
| 显存占用 / param | 2~4 B | 16~20 B |
| IO 压力 | 几乎为零 | 集群级 TB/s |
| 一次启动成本 | 秒级 | 周~月级 |

### 5.2 当你拿到一个新模型，先填这张表

| 量 | 公式 | 你的模型 |
|---|---|---|
| $\Theta$ | 参数量 | ?? |
| 单 token 前向 FLOPs | $2\Theta$ | ?? |
| 训练总 FLOPs | $\sim 6\Theta \cdot D$ | ?? |
| KV / token | $2 \cdot L \cdot H_{kv} \cdot d_h \cdot \text{bytes}$ | ?? |
| 推理 decode 下限 | $\Theta \cdot \text{bytes} / B_{\text{HBM}}$ | ?? |
| 训练静态显存 | $16 \cdot \Theta$（Adam BF16）| ?? |
| 单 ckpt 大小 | $16 \cdot \Theta$ | ?? |
| 集群 IO 需求 | $\text{batch\_tokens} \cdot 15 / t_{\text{step}}$ × 节点数 | ?? |

填完这张表，你就能在白板上跟同事 debate：
- 该买 H100 还是 H200？
- 推理是 PD 分离还是 colocate？
- 训练数据该不该上 Weka？
- 8 个实验同时跑，存储要不要扩容？

### 5.3 五条最值钱的"反直觉" 

1. **70B 推理的瓶颈不是算力是带宽**。FP8 不解决问题，W4 / 更大 HBM 才解决。
2. **训练时 activation 比权重还贵**——但 recomputation 用 33% 的算力税换 10× 显存，这笔交易稳赚。
3. **"加更多卡"的回报曲线在 1024 卡之后变 sublinear**，因为通信开始统治一切。
4. **存储是 LLM infra 的隐藏一等公民**。Meta 的 Tectonic、xAI 的 Memphis 都是**专门团队**。
5. **真正的 GPU MFU 实测值，比销售给你看的 PPT 低 2~3 倍**。永远在你算预算时打这个折。

---

## 七、参考与延伸

- Llama 3 技术报告（2024）：训练数据规模与系统工程细节，本文  数据主要来源
- Chinchilla（Hoffmann et al., 2022）：$D \approx 20\Theta$ 的来源
- Megatron-LM 论文系列：activation 显存公式
- DeepSeek-V3 技术报告（2024）：MLA / device-limited routing / FP8 训练  数据来源
- Mixtral of Experts（Jiang et al., 2024）：MoE 推理基线
- NVIDIA H100 / H200 / B200 datasheet： 算力与带宽
- 本站姊妹篇：
  - [深度学习训练全解：从零理解 GPU 上到底有哪些数据](/posts/2026-06-09-gpu-training-data-anatomy/)
  - [KV Cache 是算还是传？一份 Roofline 视角的全景账](/posts/2026-06-10-kvcache-compute-vs-transfer/)
  - [GPU 显存 Offload 技术：训练与推理中的工程实践](/posts/2026-05-28-gpu-memory-offload-techniques/)

---

> **最后一句**：所有 AI infra 的工程决策，本质上都是 **FLOPs / GB / GB·s⁻¹ 这三类数字之间的算术**。看到一个新方案，先把它落到这三个维度上量化，再谈"先进性"。
>
> 没有数字的架构讨论，都是文学。

---

## 附录 A：术语与单位速查

本文出现的所有单位 / 缩写。上面表格里几个特别容易踩坑的（FLOPs vs FLOPS、GB vs GiB、Gbps vs GB/s）请重点看。

### A.1 计算量

| 术语 | 全称 | 含义 | 容易踩的坑 |
|---|---|---|---|
| **FLOP** | Floating-Point Operation | 一次浮点运算 | 一次乘加 (FMA / MAC) = **2 FLOPs** |
| **FLOPs** | 复数，总计算量 | 一个模型跑一次需多少浮点运算 | 名词 |
| **FLOPS** | FLOP per Second | 算力速率 | 动词性质，**跟 FLOPs 差一个 /s** |
| GFLOPs | $10^9$ FLOPs | 10亿次运算 | 单 token 推理量级 |
| TFLOPs | $10^{12}$ FLOPs | 1万亿次 | 单卡算力量级（H100=989 TFLOPS BF16）|
| PFLOPs | $10^{15}$ FLOPs | 1千万亿次 | 单节点 / 单 step |
| EFLOPs | $10^{18}$ FLOPs | 100亿亿次 | 70B 训练 ·· · ~6 EFLOPs |
| ZFLOPs | $10^{21}$ FLOPs | — | 跨代训练总量（405B 量级）|
| **MFU** | Model FLOPs Utilization | 实际达到的有效算力 / 硬件峰值 | 训练 0.35~0.55，decode 0.05~0.15 |
| **HFU** | Hardware FLOPs Utilization | 含重计算的硬件使用率 | HFU > MFU，两者不要混 |

> **最常见误写**：拿 `FLOPS` 当计算量。**带不带 /s 完全两件事**。H100 "989 TFLOPS" 是个速率，训 70B 要 ~$6\times 10^{24}$ FLOPs 是个总量。

### A.2 存储容量：SI vs IEC

这是公司里哀叹最多的一类坑：

| 单位 | 全称 | 进制 | 三个数量级举例 |
|---|---|---|---|
| **KB** | kilobyte (SI) | $10^3$ B = 1000 B | 硬盘厂商、网络用 |
| **KiB** | kibibyte (IEC) | $2^{10}$ B = 1024 B | 操作系统、内存用 |
| MB / MiB | — | $10^6$ / $2^{20}$ | 差 4.86% |
| GB / GiB | — | $10^9$ / $2^{30}$ | 差 7.37%（购买杯具 SSD 的口口相传）|
| TB / TiB | — | $10^{12}$ / $2^{40}$ | 差 9.95% |

**本文约定**：
- 说到 GPU 显存、模型权重大小等，默认用 **GiB**，但为了可读性**写作"GB"**（NVIDIA datasheet 官方写法就是如此，如 H100 80GB 实际是 80 GiB）。
- 说到权重参数，"70B" 是 **70×10⁹ = 70 billion**个参数（SI），跟存储进制无关。

> **举例**："H100 显存 80 GB"，实际 = $80 \times 2^{30}$ B = 85.9 GB (SI)。如果你看到发货单写 "85 GB"，不是多送了，是 IEC 与 SI 的转换。

### A.3 带宽：bps vs Bps、Gbps vs GB/s

全文最容易被误读的一组：

| 单位 | 含义 | 资料里常见场合 |
|---|---|---|
| **bit/s** 或 **bps** | 每秒多少 bit | 网络 / NIC 发货单 |
| **byte/s** 或 **B/s** 或 **Bps** | 每秒多少 byte | GPU / 存储 datasheet |
| Gbps | $10^9$ bit/s | IB / 以太网口语 |
| GB/s | $10^9$ B/s | HBM / NVLink / PCIe |

**换算铁律**：

$$
1\ \text{Bps} = 8\ \text{bps}, \qquad 1\ \text{GB/s} = 8\ \text{Gbps}
$$

实例：
- **IB NDR 400G** = 400 **Gbps** ≈ **50 GB/s**（除以 8）
- **400 GbE** 以太网 = 400 **Gbps** ≈ **50 GB/s**
- **HBM3 3.35 TB/s** 已经是 **GB/s** 语义，不要再除 8！

> **最常见误会**："我们集群都上了 400G了"——这里的 400G 是 Gbps，折算出来只有 50 GB/s，比 NVLink 900 GB/s **差近 20×**。跨机不要拿机内带宽套。

### A.4 数值精度

| 名称 | 位宽 | 字节/参数 | 常见场合 |
|---|---|---|---|
| FP32 | 32 bit | 4 B | 优化器 master weight、梯度累加 |
| TF32 | 19 bit 有效 | 4 B 存储 | A100/H100 默认 GEMM |
| **BF16** | 16 bit | 2 B | 训练 / 推理主流，动态范围 ≈ FP32 |
| FP16 | 16 bit | 2 B | 老牌，范围窄容易溢出 |
| **FP8 E4M3** | 8 bit | 1 B | 前向、权重，精度高范围窄 |
| FP8 E5M2 | 8 bit | 1 B | 反向 / 梯度，范围宽精度低 |
| INT8 | 8 bit | 1 B | 推理量化（W8A8）|
| INT4 / NF4 | 4 bit | 0.5 B | 推理权重量化（W4A16 / GPTQ / AWQ）|

**一个极高频公式**：模型权重显存 = $\Theta \times \text{bytes}$。

### A.5 模型 / 训练符号表

本文中使用的数学符号：

| 符号 | 含义 |
|---|---|
| $\Theta$ | 参数量（dense） |
| $\Theta_t$ / $\Theta_a$ | MoE 总参数 / 激活参数 |
| $D$ | 训练 token 总数 |
| $L$ | Transformer 层数 |
| $d$ | hidden size |
| $h$ / $H_{kv}$ | query head 数 / KV head 数 |
| $d_h$ | head 维度 |
| $d_{ff}$ | FFN 中间维 |
| $N$ / $S$ | 序列长度 |
| $B$ | batch size |
| $k$ | MoE top-k |

### A.6 系统 / 硬件缩写

| 缩写 | 全称 | 一句话 |
|---|---|---|
| **HBM** | High-Bandwidth Memory | GPU 片上高带宽显存（3–8 TB/s）|
| GDDR | Graphics DDR | 消费卡 / L40S 用，带宽约 1 TB/s |
| **NVLink** | NVIDIA 点对点互联 | 同机卡间 900–1800 GB/s |
| NVSwitch | NVLink 全互联交换机 | 8八卡同机同带宽 |
| **PCIe** | Peripheral Component Interconnect Express | CPU↔GPU，Gen5 x16 = 64 GB/s |
| **IB** | InfiniBand | RDMA 高速网，NDR=400 Gbps |
| RoCE | RDMA over Converged Ethernet | 以太网上跑 RDMA |
| **NVMe** | NVM Express | 快 SSD 协议，顺读 7 GB/s |
| GPFS / Lustre / Weka / DAOS | — | 分布式并行文件系统 |

### A.7 并行与集体通信

| 缩写 | 全称 | 一句话 |
|---|---|---|
| **DP** | Data Parallel | 每卡完整模型，batch 切开 |
| **TP** | Tensor Parallel | 拆矩阵运算，全部在节点内 |
| **PP** | Pipeline Parallel | 拆层，micro-batch 流水线 |
| **EP** | Expert Parallel | MoE 专家切到不同卡 |
| SP | Sequence Parallel | 切 sequence 维 |
| **ZeRO / FSDP** | Zero Redundancy Optimizer / Fully Sharded DP | 参数+梯度+优化器分片 DP |
| **AllReduce** | — | 所有 rank 合出一个总和并广播回去 |
| **AllGather** | — | 各卡拼出完整数据 |
| **ReduceScatter** | — | AllReduce 的上半场，结果分片 |
| **All2All** | — | 任意两卡都交换一块，MoE 专用 |

### A.8 推理专有名词

| 缩写 | 全称 | 一句话 |
|---|---|---|
| **prefill** | — | 一次性吃完所有输入 token，compute-bound |
| **decode** | — | 逐 token 生成，memory-bound |
| **KV Cache** | — | 历史 token 的 K、V，避免重算 |
| **MHA** | Multi-Head Attention | 每 head 独立 K/V |
| **GQA** | Grouped-Query Attention | 多个 query head 共享一组 K/V |
| **MQA** | Multi-Query Attention | 所有 query head 共享一组 K/V |
| **MLA** | Multi-head Latent Attention | DeepSeek，在低维隐空间压缩 KV |
| **PD 分离** | Prefill / Decode 解耦 | 两阶段拆到不同机器跑 |
| **MTP** | Multi-Token Prediction | DeepSeek-V3 使用的预测多个未来 token 的技术 |
| **Roofline** | — | 算力 vs 带宽 二维模型，判断瓶颈 |
| **Arithmetic Intensity** | — | FLOPs / Byte，roofline 横坐标 |

### A.9 一个帮你随手换算的例子

看到 "H100 SXM、989 TFLOPS BF16、HBM3 3.35 TB/s、NVLink 900 GB/s、跨机 IB NDR 400G"，应该立刻在脑里出现：

```text
算力   : 989×10¹²  FLOP / s     ← 总量 = 带 /s
HBM    : 3.35×10¹² B / s        ← byte为单位
NVLink : 900×10⁹    B / s        ← byte为单位
IB     : 400×10⁹    bit / s      ← 除以 8 → 50 GB/s
```

反过来检查一下：H100 一张卡 BF16 跑满一秒需要多大 batch？

$$
\frac{P}{B_{\text{HBM}}} = \frac{989\,\text{TFLOPS}}{3.35\,\text{TB/s}} \approx 295\ \text{FLOPs/Byte}
$$

这个 295 就是本文被反复引用的 ridge point——你能在脑里一环环推出来，说明单位已经逻辑自洽了。

