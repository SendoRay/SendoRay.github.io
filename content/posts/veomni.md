---
title: VeRL-Omni 全面解析：多模态扩散模型的 RL 后训练框架
date: '2026-06-09'
tags:
- LLM
-  AI-Infra
- Diffusion
- GPU

draft: false
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

## 一、项目概述

VeRL-Omni 是字节跳动 Seed MLSys 团队开源的**多模态生成模型 RL 后训练框架**。它不像 Stable Diffusion 那样做图像生成，而是用强化学习（RL）来微调已经训好的生成模型，让它们生成更高质量的图像/视频/音频。

简单说：你有一个 Qwen-Image 模型能文生图，但生成质量不够好。VeRL-Omni 用 RL（比如 FlowGRPO）来继续训练它，根据 reward（OCR 准确率、人类偏好评分等）来优化模型，让它生成更好的图片。

---

## 二、为什么需要 VeRL-Omni

> Multimodal generative RL training differs from text-only LLM RL not only in model structure, but also in I/O patterns, compute characteristics, and runtime bottlenecks.

多模态生成模型的 RL 训练与纯文本 LLM 的 RL 存在根本性差异，不能直接复用 verl 或 vLLM：

| 差异维度 | LLM | 扩散模型 |
|---|---|---|
| 生成方式 | 自回归逐 token 生成 | 迭代去噪（逐 step 去噪） |
| 训练循环 | token-level log-probs，PPO 在 token 维度 | SDE 逆过程计算每步 transition 的 log-prob |
| 数据 I/O | token 序列 | 高维 latent tensor `(B,C,H,W)` 甚至 `(B,C,T,H,W)` |
| 计算瓶颈 | KV cache 和 attention | VAE 编解码、多步 UNet/DiT 推理 |
| log-prob 计算 | softmax 输出直接获取 | 需要 SDE 逆过程的高斯转移概率 |
| 推理引擎 | vLLM 为 LLM 优化 | 需要 vLLM-Omni 做扩散 rollout |

性能方面，VeRL-Omni 比原始 flow_grpo 实现快约 25%，得益于 vLLM-Omni rollout + FSDP 训练 + 异步 reward 计算的重叠。

---

## 三、扩散模型基础 —— 从 LLM 人的视角

### 3.1 核心直觉

**LLM 是自回归生成**：给定 `[我, 爱]`，预测下一个 token `[吃]`，再预测 `[苹果]`……逐 token 串行。

**扩散模型是迭代去噪生成**：先画一张纯噪声图，然后一步一步"擦掉"噪声，露出清晰的图像。

```
LLM 推理:
  [BOS] → [我] → [爱] → [吃] → [苹] → [果] → [EOS]
  每一步: 选下一个 token (discrete sampling from softmax)

扩散模型推理 (去噪):
  纯噪声 z_T → z_{T-1} → z_{T-2} → ... → z_0 (清晰 latent) → VAE decode → 图像
  每一步: 对 latent tensor 做连续数值的更新 (continuous denoising)
```

关键对应关系：

| LLM 概念 | 扩散模型概念 |
|---|---|
| token | latent pixel/tensor element |
| autoregressive sampling（逐个 token） | iterative denoising（逐步去噪，通常 20-50 步） |
| vocabulary size V | latent 是连续值 (R^d) |
| softmax → categorical sampling | 高斯分布 → continuous sampling |
| KV cache | 无（每步独立，但 latent 尺寸大） |
| next-token log-prob | SDE transition log-prob |

### 3.2 两大流派

**DDPM (Denoising Diffusion Probabilistic Models)**
- 2020 年 Ho et al. 提出
- 前向过程：逐步加高斯噪声 `x_t = sqrt(1-β_t) * x_{t-1} + sqrt(β_t) * ε`
- 反向过程：学习去预测噪声 `ε_θ(x_t, t)`
- 数学基础：离散时间步的马尔可夫链

**Flow Matching / Continuous-time Diffusion**
- 2023 年 Lipman et al. (Flow Matching) 和 Song et al. (Score-based SDE) 统一
- 关键思想：扩散过程是连续时间的，由 ODE/SDE 描述
- 训练目标：预测 velocity field `v_θ(x_t, t)` 或 score function `∇log p_t(x)`
- **VeRL-Omni 主要使用 Flow Matching 框架**

为什么选择 Flow Matching？DDPM 把时间离散化了，步数少则质量差。Flow Matching 是连续时间框架，可以用任意步数采样（ODE solver），且数学上更"干净"——利于 RL 训练中计算 log-prob。

### 3.3 Flow Matching 训练（预训练阶段）

直觉：你有一个清晰的图像 `x_0`（来自数据集），随机挑一个时间 `t ∈ [0,1]`，然后做线性插值。模型要学习的是 velocity（速度场）`v = x_0 - ε`，即"从噪声到清晰图像的方向"。

```python
# ============================================
# Flow Matching 预训练伪代码
# ============================================

# x_0: 清晰图像的 latent，shape (B, C, H, W)，来自 VAE encode
# text_embedding: 文本 prompt 的 embedding，shape (B, L, D)
# model: DiT (Diffusion Transformer)

for x_0, text_embedding in dataloader:
    B = x_0.shape[0]

    # 1. 随机采样时间 t ∈ [0, 1]
    t = torch.rand(B, device=device)

    # 2. 采样噪声
    epsilon = torch.randn_like(x_0)

    # 3. 线性插值：t=0 → 纯噪声, t=1 → 清晰图
    x_t = (1 - t) * epsilon + t * x_0

    # 4. 目标 velocity
    target_velocity = x_0 - epsilon

    # 5. 模型预测 velocity v_θ(x_t, t, text)
    v_pred = model(
        hidden_states=x_t,
        timestep=t,
        encoder_hidden_states=text_embedding,
    )

    # 6. 简单的 MSE Loss！
    loss = F.mse_loss(v_pred, target_velocity)
    loss.backward()
    optimizer.step()
```

**重点**：Flow Matching 的预训练 loss 就是 MSE！跟 LLM 的 cross-entropy loss 一样简单——模型只需要做回归。

### 3.4 Flow Matching 推理（ODE 采样）

训好模型后，用 ODE solver 从噪声一步步去噪：

```python
# ============================================
# Flow Matching 推理 (ODE sampling)
# ============================================

# 1. 从纯噪声开始
x_t = torch.randn(B, C, H, W)  # t=1 → 纯噪声

# 2. 时间步长列表 (由 scheduler 决定)
sigmas = scheduler.sigmas  # [1.0, 0.98, ..., 0.0]

# 3. 迭代去噪
for step in range(num_inference_steps):
    sigma = sigmas[step]
    sigma_next = sigmas[step + 1]

    v_pred = model(
        hidden_states=x_t,
        timestep=sigma,
        encoder_hidden_states=text_embedding,
    )

    # Euler ODE step
    x_t = x_t + (sigma_next - sigma) * v_pred

# 4. VAE decode latent → 像素空间
image = vae.decode(x_t)
```

**关键理解**：Flow Matching 的 ODE 采样是**确定性的**！给定相同初始噪声和 prompt，每次生成完全一样的图。这跟 LLM 采样（有 randomness from softmax）完全不同。

---

## 四、SDE 与 RL 训练 —— VeRL-Omni 的核心创新

### 4.1 问题：扩散模型没有 softmax

在 LLM RL 中，log-prob 很直接：

```python
logits = model(input_ids)                      # (B, seq_len, vocab_size)
log_probs = F.log_softmax(logits, dim=-1)
token_log_prob = log_probs[:, t, token_id]     # 很简单！
```

扩散模型每一步输出的是一个连续向量（velocity/noise prediction），没有 softmax。那 log-prob 怎么算？

**答案**：把去噪过程当作一个**随机微分方程（SDE）**的离散化，然后计算高斯转移概率。这就是 VeRL-Omni 的 `FlowMatchSDEDiscreteScheduler` 做的事情。

### 4.2 ODE vs SDE

```
ODE (确定性):        SDE (随机性):
  dx = v(x,t) dt       dx = [v(x,t) + 噪声项] dt + 扩散项 dw

  固定输入 → 固定输出   固定输入 → 随机输出（可以算概率）
```

FlowGRPO 的核心思想：
1. 预训练用 ODE（MSE loss，简单高效）
2. RL 训练用 SDE（需要随机性来定义 log-prob）
3. 只在部分步（SDE Window）注入噪声，其余步用 ODE

### 4.3 SDE 逆过程计算（核心数学）

这是 VeRL-Omni 最核心的计算。`FlowMatchSDEDiscreteScheduler` 实现了三种 SDE：
- **`"sde"`**（FlowGRPO）：`std_dev_t = sqrt(sigma / (1-sigma)) * noise_level`
- **`"cps"`**（Conservative Proposal Step）：`std_dev_t = sigma_prev * sin(noise_level * pi/2)`
- **`"dance_sde"`**（DanceGRPO）：基于 score-based SDE correction，数值更稳定

```python
def sde_reverse_step(x_t, sigma, sigma_next, model_output, noise_level):
    """
    单步 SDE 逆过程
    
    输入:
        x_t:          当前 noisy latent, shape (B, C, H, W)
        sigma:        当前噪声水平 (float)
        sigma_next:   下一步噪声水平 (float), sigma_next < sigma
        model_output: 模型预测的 velocity v_θ(x_t, sigma), shape (B, C, H, W)
        noise_level:  SDE 噪声强度 η ∈ [0, 1]
    
    输出:
        x_next:   下一步 latent x_{t-1}
        log_prob: 转移概率 log p(x_{t-1} | x_t), shape (B,)
        mean:     转移分布的均值
        std:      转移分布的标准差
    """

    dt = sigma_next - sigma  # 负数（sigma 减小）

    # ==========================================
    # Step 1: SDE 噪声方差 (这是关键!)
    # ==========================================
    # std_dev_t 决定了注入多少随机性
    sigma_max = scheduler.sigmas[1]  # 最大 sigma (~1.0)

    # 公式: std_dev_t = sqrt(sigma/(1-sigma)) * noise_level
    # 当 sigma→1: 分母→0，但 clamp 到 sigma_max
    std_dev_t = torch.sqrt(
        sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))
    ) * noise_level
    # shape: 标量, broadcast 到 (B, 1, 1, 1)

    # ==========================================
    # Step 2: 计算转移分布的均值
    # ==========================================
    # mean = x_t * (1 + std²/(2σ) * dt) + v_pred * (1 + std²*(1-σ)/(2σ)) * dt
    prev_sample_mean = (
        x_t * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    # ==========================================
    # Step 3: 采样下一步 latent
    # ==========================================
    # noise ~ N(0, I)
    noise = torch.randn_like(x_t)

    # 转移分布: p(x_{t-1} | x_t) = N(mean, (std * sqrt(-dt))²)
    # 注意 sqrt(-dt) 因为 dt 是负数
    x_next = prev_sample_mean + std_dev_t * torch.sqrt(-dt) * noise

    # ==========================================
    # Step 4: 计算 log-prob (高斯密度)
    # ==========================================
    # 这是 RL 训练需要的关键值！
    # p(x_{t-1} | x_t) 服从高斯分布 N(mean, variance)
    variance = (std_dev_t * torch.sqrt(-dt)) ** 2

    # 对数高斯密度
    log_prob = (
        - (x_next.detach() - prev_sample_mean) ** 2 / (2 * variance)  # 二次项
        - torch.log(std_dev_t * torch.sqrt(-dt))                       # log(std)
        - torch.log(torch.sqrt(2 * torch.pi))                          # 归一化常数
    )  # shape: (B, C, H, W)

    # 对空间维度取平均 → (B,)
    log_prob = log_prob.mean(dim=(1, 2, 3))

    return x_next, log_prob, prev_sample_mean, std_dev_t
```

每一步的 log-prob 计算本质就是高斯分布的对数密度：

```
log_prob = -(prev_sample - mean)² / (2 * std²) - log(std) - log(sqrt(2π))
```

### 4.4 SDE Window —— 为什么不全用 SDE

```
Timestep:   T ──────────────────────────────────────→ 0
            噪声                                      清晰

SDE Window:  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
            只有这 4 步      其余都用 ODE (确定性的)
            注入噪声+收集     (快, 不需要 log-prob)
            log-prob
```

原因：
1. **计算开销**：每步 SDE 需要存 latents + 额外计算
2. **方差控制**：只在噪声较大的步注入随机性（梯度信号好），接近清晰的步用 ODE
3. **经验发现**：`window_size=4-8` 就够了，性能和全 SDE 几乎一样

---

## 五、模型架构

### 5.1 支持的模型

| 模型 | 架构 | 模态 | 算法 |
|---|---|---|---|
| Qwen-Image | QwenImageTransformer2DModel（DiT） | Text→Image | FlowGRPO, MixGRPO, GRPO-Guard, DiffusionNFT, DPO |
| Wan2.2 | WanTransformer3DModel（3D DiT） | Text→Video | DanceGRPO |
| SD3.5 | MMDiT（双流 DiT） | Text→Image | DPO |
| BAGEL（WIP） | 统一理解+生成 | Text+Image | FlowGRPO |
| Qwen3-Omni（WIP） | 全模态统一 | Text/Image/Video/Audio | GSPO |
| HunyuanImage-3.0（计划） | 统一理解+生成 | Text+Image | MixGRPO, SRPO |
| LTX2.3（WIP） | 扩散生成器 | Text→Video+Audio | FlowGRPO |

### 5.2 Qwen-Image 全局架构

Qwen-Image 是一个 Flow Matching DiT（Diffusion Transformer），20B 参数的 MM-DiT：

1. **VAE Encoder**：图像 `(B,3,H,W)` → latent `(B,C,H/8,W/8)`
2. **Text Encoder**（Qwen2.5-VL，冻结）：文本 prompt → `prompt_embeds (B,L,D)`
3. **QwenImageTransformer2DModel**：核心 DiT，输入 noisy latent + timestep + text embeddings，输出 velocity prediction
4. **VAE Decoder**：去噪后的 latent → 像素空间

```
              ┌─────────────────────────────┐
              │      用户输入 Prompt          │
              └─────────────┬───────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
    ┌────────────────────┐   ┌────────────────────┐
    │ Qwen2.5-VL (冻结)  │   │ 纯噪声 ε ~ N(0,I)  │
    │ → text_emb         │   │ (B, 64, H/8, W/8)  │
    │   (B, L_text, 3584)│   └─────────┬──────────┘
    └─────────┬──────────┘             │
              │                        │
              └──────────┬─────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────┐
    │  Patch Embed: latent → 切 2×2 patch → Linear    │
    │  (B,64,H,W) → (B, N_img, 3584)                  │
    │  例如 H=W=128 → N_img = 4096 tokens              │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │  MSRoPE 位置编码                                  │
    │  图像: 2D RoPE (二维坐标)                         │
    │  文本: 沿对角线方向 1D RoPE                       │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │           MMDiT Backbone (60层, 20B)              │
    │                                                  │
    │  输入序列 = [image_tokens ‖ text_tokens]          │
    │            (B, 4096+512, 3584)                    │
    │                                                  │
    │  全部是 full bi-directional self-attention       │
    │  + Timestep Embedding 通过 AdaLN 注入每层        │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │  Final Layer: 只取 image tokens                  │
    │  (B, 4096, 3584) → Linear → (B, 64, H, W)       │
    │  输出: velocity prediction v_θ                    │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │  SDE Scheduler (FlowMatchSDEDiscreteScheduler)   │
    │  x_t + v_pred → x_{t-1} + log_prob              │
    │  迭代 50 步                                      │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │  VAE Decoder (冻结) → 最终图像 (B, 3, H, W)      │
    └──────────────────────────────────────────────────┘
```

### 5.3 MMDiT Block 内部结构

这是最核心的模块，也是和标准 LLM Transformer Block 差异最大的地方。下面是一层 MMDiT Block 的完整前向过程：

```
输入:
  image_tokens: (B, N_img, D)      例如 (1, 4096, 3584)
  text_tokens:  (B, N_text, D)     例如 (1, 512,  3584)
  c:            条件向量 (B, D)     来自 timestep embedding

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌───────────────────────────────────────────────────────────┐
│  Step 1: adaLN_modulation — 从条件向量回归出 6 组参数     │
│                                                           │
│  adaLN_modulation = SiLU → Linear(D → 6*D)                │
│                                                           │
│  [shift_msa, scale_msa, gate_msa,                         │
│   shift_mlp, scale_mlp, gate_mlp] =                       │
│       adaLN_modulation(c).chunk(6, dim=-1)                 │
│                                                           │
│  每组 shape: (B, D) → 每个通道独立的 scale/shift/gate     │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────┐
│  Step 2: 文本-图像联合 Self-Attention (双流)               │
│                                                           │
│  拼接: joint = [image_tokens ‖ text_tokens]                │
│         shape: (B, N_img+N_text, D)                       │
│                                                           │
│  先做 AdaLN:                                              │
│    joint_normed = LayerNorm(joint)  ← 无 affine 参数      │
│    joint_modulated = joint_normed * (1+scale_msa)         │
│                      + shift_msa                          │
│                                                           │
│  再做 Joint Self-Attention:                               │
│    Q = joint_modulated @ W_Q   (B, N_total, D)            │
│    K = joint_modulated @ W_K   (B, N_total, D)            │
│    V = joint_modulated @ W_V   (B, N_total, D)            │
│                                                           │
│    # 先加 MSRoPE 到 Q, K:                                 │
│    Q_img += RoPE_2D(Q_img, pos_x, pos_y)  ← 图像用2D坐标  │
│    K_img += RoPE_2D(K_img, pos_x, pos_y)                  │
│    Q_txt += RoPE_1D(Q_txt, pos_k)         ← 文本用1D坐标   │
│    K_txt += RoPE_1D(K_txt, pos_k)                         │
│                                                           │
│    attn_out = softmax(Q @ K^T / sqrt(d)) @ V              │
│                                                           │
│    完整 attention 矩阵包含 4 个区域:                       │
│    ┌──────────────┬──────────────┐                        │
│    │ img→img self │ img→txt crs  │  ← 图像 token 作为 query│
│    │  (4096,4096) │  (4096,512)  │                        │
│    ├──────────────┼──────────────┤                        │
│    │ txt→img crs  │ txt→txt self │  ← 文本 token 作为 query│
│    │  (512,4096)  │   (512,512)  │                        │
│    └──────────────┴──────────────┘                        │
│                                                           │
│  Gate 控制:                                               │
│    joint = joint + gate_msa * attn_out                    │
│                                                           │
│  拆分回双流:                                              │
│    image_tokens = joint[:, :N_img, :]                     │
│    text_tokens  = joint[:, N_img:, :]                     │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────┐
│  Step 3: Feed-Forward Network (双流独立 MLP)              │
│                                                           │
│  Image 侧:                                                │
│    img_normed = LayerNorm(image_tokens)   ← 无 affine     │
│    img_modulated = img_normed * (1+scale_mlp) + shift_mlp │
│    img_ffn = SiLU(img_modulated @ W_up) @ W_down          │
│    image_tokens = image_tokens + gate_mlp * img_ffn       │
│                                                           │
│  Text 侧:                                                 │
│    txt_normed = LayerNorm(text_tokens)                    │
│    txt_modulated = txt_normed * (1+scale_mlp) + shift_mlp │
│    txt_ffn = SiLU(txt_modulated @ W_up) @ W_down          │
│    text_tokens = text_tokens + gate_mlp * txt_ffn         │
│                                                           │
│  (image 和 text 共享同一套 shift/scale/gate 参数)          │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
输出: image_tokens (B, N_img, D) + text_tokens (B, N_text, D)
      ↓ 进入下一层 Block (共 60 层)
```

跟 LLM Transformer Block 比，有几个关键区别值得展开说：

- **AdaLN 代替 Pre-LN**：LN 的参数由 timestep embedding 动态生成 `(scale(t), shift(t))`，而不是固定参数。这意味着在噪声大的时候（t 接近 1），模型可以学到一套行为；在噪声小的时候（t 接近 0），学到另一套行为——同一个模型通过 AdaLN 实现了对不同去噪阶段的自适应。
- **没有 causal mask**：latent patch 之间是全 attention（双向），不需要 mask。因为图像不存在"因果"的概念——不像文本有前后顺序。
- **输入是连续 latent 不是离散 token**：不需要 embedding lookup。
- **输出维度 = 输入维度**：预测 velocity（同尺寸 tensor），不是下一个 token 的概率分布。
- **Gate 机制**：每个子层输出乘以一个 gate（初始化为 0），这样训练初期整个 Block 相当于恒等映射，稳定训练。

### 5.4 与 LLM Transformer Block 的核心差异

| 组件 | LLM Decoder Block | MMDiT Block |
|---|---|---|
| 归一化 | Pre-LN / RMSNorm（固定参数） | AdaLN（scale/shift 由 timestep 动态回归） |
| 子层输出控制 | 直接残差 `x = x + attn(x)` | Gated 残差 `x = x + gate * attn(x)` |
| Attention | Causal Self-Attention（单向 mask） | Full Bi-directional（无 mask） |
| 位置编码 | 1D RoPE | MSRoPE（图像 2D RoPE + 文本 1D RoPE） |
| 输入模态 | 纯文本 token | 图像 latent + 文本 token 拼接 |
| 条件注入 | 无（自回归） | Timestep embedding 注入每一层 |
| 初始化 | 标准初始化 | Zero 初始化（gate=0，训练初期为恒等映射） |

### 5.5 伪代码：完整模型定义

```python
class QwenImageMMDiT(nn.Module):
    """
    Qwen-Image 的完整 DiT 模型
    三层结构: Text Encoder (frozen) → MMDiT Backbone → VAE Decoder (frozen)
    """
    def __init__(self):
        # ── 第一层: 编码器 (冻结，不做 RL 训练) ──
        self.text_encoder = FrozenQwen25VL()    # 语义编码
        self.vae_encoder = FrozenVAE()           # latent 压缩
        self.vae_decoder = FrozenVAE()           # latent → 像素

        # ── 第二层: Patch Embedding ──
        self.patch_embed = nn.Linear(
            patch_size**2 * in_channels,  # 2*2*64 = 256
            hidden_dim                     # 3584
        )

        # ── 第三层: Timestep Embedding ──
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),   # 频率编码
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── 第四层: MMDiT Blocks × 60 ──
        self.blocks = nn.ModuleList([
            MMDiTBlock(hidden_dim=3584, num_heads=28)
            for _ in range(60)
        ])

        # ── 第五层: Final Layer (unpatch + output projection) ──
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)  # 只需 shift, scale
        )
        self.final_proj = nn.Linear(
            hidden_dim,
            patch_size**2 * in_channels  # 2*2*64 = 256
        )

    def forward(self, latent, timestep, text_emb, text_mask):
        """
        latent: (B, C, H, W)  noisy latent
        timestep: (B,)  float in [0, 1]
        text_emb: (B, L_text, D)  from frozen text encoder
        """
        B = latent.shape[0]

        # 1. 图像 patchify
        image_tokens = patchify(latent, patch_size=2)  # → (B, N_img, 256)
        image_tokens = self.patch_embed(image_tokens)   # → (B, N_img, 3584)

        # 2. 时间条件
        c = self.time_embed(timestep)  # → (B, 3584)

        # 3. 位置编码 (MSRoPE 在 attention 里面加)

        # 4. 60 层 MMDiT Block
        for block in self.blocks:
            image_tokens, text_tokens = block(
                image_tokens, text_emb, c
            )

        # 5. Final Layer: 只取 image tokens 输出
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(image_tokens)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        velocity = self.final_proj(x)  # → (B, N_img, 256)

        # 6. Unpatchify
        velocity = unpatchify(velocity, H, W)  # → (B, 64, H/2, W/2)
        return velocity  # model_output, 也叫 v_pred 或 noise_pred


class MMDiTBlock(nn.Module):
    """单个 MMDiT Block"""
    def __init__(self, hidden_dim=3584, num_heads=28):
        # 无参数的 LayerNorm (affine 参数由 adaLN 动态提供)
        self.norm_attn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = MultiHeadAttention(hidden_dim, num_heads)

        self.norm_ffn_img = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_ffn_txt = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn_img = SwiGLU_MLP(hidden_dim, 4 * hidden_dim)
        self.ffn_txt = SwiGLU_MLP(hidden_dim, 4 * hidden_dim)

        # 关键：从条件向量 c 回归 6 组参数
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)  # 初始化为 0
        )

    def forward(self, img_tokens, txt_tokens, c):
        # 1. 回归 AdaLN 参数
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = \
            self.adaLN(c).chunk(6, dim=-1)

        # 2. Joint Attention (文本+图像拼接)
        joint = torch.cat([img_tokens, txt_tokens], dim=1)
        joint = self.norm_attn(joint) * (1 + scale_a) + shift_a
        joint = joint + gate_a * self.attn(joint)
        img_tokens, txt_tokens = joint.split([N_img, N_txt], dim=1)

        # 3. FFN (双流独立)
        img_tokens = img_tokens + gate_f * self.ffn_img(
            self.norm_ffn_img(img_tokens) * (1 + scale_f) + shift_f
        )
        txt_tokens = txt_tokens + gate_f * self.ffn_txt(
            self.norm_ffn_txt(txt_tokens) * (1 + scale_f) + shift_f
        )

        return img_tokens, txt_tokens
```

---

## 六、完整训练流程

### 6.1 七个阶段

整个系统架构分 7 个阶段：

**① 数据加载** — `StatefulDataLoader` 从 parquet 文件读取 prompts（文本描述 + ground truth），支持断点续训。

**② Rollout（生成）** — vLLM-Omni Async Server 作为 Ray actor 部署。`DiffusionAgentLoopWorker` 把 prompt 发给 vLLM-Omni，自定义 Pipeline（如 `QwenImagePipelineWithLogProb`）跑完整 SDE 去噪循环。SDE Window 控制只在指定 timestep 窗口内注入噪声 + 收集 log-prob。返回：生成图片 + `all_latents` + `all_log_probs` + `all_timesteps` + prompt embeddings。

**③ Reward 计算** — `VisualRewardManager` / `MultiVisualRewardManager` 对生成图片打分。支持多种 reward：
- 规则型：OCR 准确率、JPEG 压缩率
- 模型型：HPSv3（人类偏好评分）、UnifiedReward 2.0
- HTTP 远程：外部 reward server
- 异步计算：reward 和 rollout 可以重叠执行

**④ Old Log-Prob 重算** — 用当前 actor 模型权重，在 rollout 产生的 latent 轨迹上重新做 forward + SDE 逆过程，得到 `old_log_probs`。这是 PPO 风格 proximal optimization 必需的一步。

**⑤ Advantage 估计（GRPO 风格）** — 对同一 prompt 生成的多个样本（如 n=8），以组为单位归一化 reward：`advantage = (reward - group_mean) / group_std`。不需要单独的 value function（critic）。

**⑥ Actor 更新（FSDP 训练引擎）** — `DiffusersFSDPEngine` 使用 PyTorch FSDP/FSDP2 做分布式训练，支持 Ulysses Sequence Parallelism 和 LoRA。三种子引擎：
- `PPODiffusersFSDPEngine`：policy gradient 方法（FlowGRPO 等）
- `DPODiffusersFSDPEngine`：DPO 方法
- `NFTDiffusersFSDPEngine`：DiffusionNFT 方法

**⑦ Weight Sync + Checkpointing** — 训练完的权重同步回 rollout replica，定期保存 checkpoint。

### 6.2 FlowGRPO 完整训练流程

```python
# ============================================
# FlowGRPO RL 训练的完整流程
# ============================================

# Phase 1: Rollout (vLLM-Omni)
# 输入: prompt → 输出: image + rollout_log_probs (SDE window 内收集)

# Phase 2: Reward
reward = compute_reward(image, ground_truth)  # OCR accuracy, HPSv3 etc.

# Phase 3: Old log-prob recompute (用当前模型权重)
for step in sde_window:
    v_pred = model(x_t, sigma, prompt)
    x_next, old_log_prob_step, _, _ = sde_reverse_step(...)
old_log_probs = sum_over_steps

# Phase 4: Advantage (GRPO)
group_rewards = rewards[group_mask]          # 同一 prompt 的多个样本
advantage[i] = (reward[i] - group_rewards.mean()) / group_rewards.std()

# Phase 5: Actor forward + Loss
for step in sde_window:
    v_pred = model(x_t, sigma, prompt)
    x_next, log_prob_step, _, _ = sde_reverse_step(...)
log_probs = sum_over_steps

# PPO-style clipped objective
ratio = exp(log_probs - old_log_probs)
loss = max(-adv * ratio, -adv * clip(ratio, 1-ε, 1+ε))
loss.backward()
```

### 6.3 三种 log-prob 的区分

这是最容易搞混的地方：

| | rollout_log_prob | old_log_prob | log_prob |
|---|---|---|---|
| 谁算的 | vLLM-Omni（推理 server） | Actor（训练 engine） | Actor（训练 engine） |
| 用的模型 | 当前 rollout 权重 | 当前 actor 权重（更新后） | 当前 actor 权重 |
| 用的轨迹 | 随机采样（SDE 有噪声） | 同一轨迹（latent 复用） | 同一轨迹（latent 复用） |
| 什么时候算 | Rollout 阶段 | 更新前 | 更新时（forward+backward） |
| 作用 | metadata，不用于 loss | 用于 ratio 分母 | 用于 ratio 分子 |

```python
ratio = exp(log_prob - old_log_prob)
# old_log_prob: 模型参数变了但 latent 轨迹不变，重新算一遍
# log_prob:     当前 forward 的结果，有 grad
# rollout_log_prob: 原始轨迹的 log-prob（只用于 ref 或 debug）
```

---

## 七、各算法原理

### FlowGRPO

最核心的算法。PPO-style clipped objective，但 log_prob 来自 SDE 逆过程而非 token 采样：

```
ratio = exp(log_prob_current - log_prob_old)
loss = max(-adv * ratio, -adv * clip(ratio, 1-eps, 1+eps))
```

### GRPO-Guard

在 FlowGRPO 基础上加一个 "ratio-mean bias" 项，惩罚当前策略的 reverse-SDE mean 偏离 rollout 策略的 mean：

```
ratio_mean_bias = ||prev_sample_mean - old_prev_sample_mean||² / (2 * scale²)
scale = sqrt_dt * std_dev_t  # 使不同 timestep 上梯度量级一致
```

### DPO（扩散版）

直接偏好优化，不需要显式 reward model。比较 chosen vs rejected 样本的 noise prediction 误差：

```
loss = -log_sigmoid(-0.5 * β * (
    model_err_chosen - ref_err_chosen
    - model_err_rejected + ref_err_rejected
))
```

### DiffusionNFT

前向过程的 DPO（Forward-process direct-preference）。在 forward noising 方向上做优化：`x_0 → x_t` 加噪后预测 `x_0`，用 positive/negative 预测的加权组合。

---

## 八、迭代去噪的复用机制 —— 与 LLM 推理的对比

### 8.1 核心结论：扩散模型去噪不复用 KV cache

```
LLM 自回归推理:
  step 1: [BOS]        → "我"   ── KV cache[t=1]
  step 2: [BOS, 我]    → "爱"   ── 复用 KV cache[t=1], 只算新增 token
  step 3: [BOS,我,爱]  → "吃"   ── 复用 KV cache[t=1,2]
  每一步输入变长，但前面的 token 不变！

扩散模型迭代去噪:
  step 1: latent x₁   (纯噪声)       → x₀.₉₈  ── 整个 latent 变了
  step 2: latent x₀.₉₈(少一点噪声)   → x₀.₉₆  ── 整个 latent 又变了
  step 3: latent x₀.₉₆               → x₀.₉₄  ── ...
  每一步输入是上一步的完整输出，尺寸相同，数值全变！
```

### 8.2 每一步之间：复用什么，不复用什么

```
Step t (sigma = 0.98)                    Step t-1 (sigma = 0.96)
┌──────────────────────┐                ┌──────────────────────┐
│ 输入:                 │                │ 输入:                 │
│  x_t (B,C,H,W) ───────── 变了！────→ │  x_{t-1} (B,C,H,W)   │
│  sigma_t = 0.98       │   ← 不同值    │  sigma_{t-1} = 0.96   │
│                      │                │                      │
│ 复用 (不变):          │                │ 复用 (不变):          │
│  ✓ prompt_embeds     │  ─── 相同 ──→  │  ✓ prompt_embeds     │
│  ✓ prompt_mask       │  ─── 相同 ──→  │  ✓ prompt_mask       │
│  ✓ img_shapes        │  ─── 相同 ──→  │  ✓ img_shapes        │
│  ✓ 模型权重 θ         │  ─── 相同 ──→  │  ✓ 模型权重 θ         │
│  ✓ guidance (CFG)    │  ─── 相同 ──→  │  ✓ guidance          │
│  ✓ negative_prompt   │  ─── 相同 ──→  │  ✓ negative_prompt   │
│                      │                │                      │
│ 不复用 (全变了):       │                │ 不复用:               │
│  ✗ latent x_t 的 KV  │   无法复用      │  需要重新算 attention │
│    因为输入 latent 的 │   ← 因为 →     │  整个 QKV 从头计算    │
│    所有数值都变了     │   ！！！！！    │                      │
└──────────────────────┘                └──────────────────────┘
```

### 8.3 完整去噪循环（标明每一步什么变了什么没变）

```python
def denoising_loop(noise_latent, text_embedding, text_mask, model, scheduler):
    """
    完整去噪循环，标明每一步什么变了什么没变
    """
    # ═══ 这些只算一次，所有步复用 ═══
    prompt_embeds = text_embedding        # (B, L, D) — 来自 frozen text encoder
    prompt_mask = text_mask               # (B, L)
    img_shapes = compute_img_shapes(H, W) # [(H_lat, W_lat), ...]
    negative_embeds = encode_negative_prompt()  # CFG 用的无条件 prompt
    guidance_emb = torch.tensor([4.0])    # CFG scale

    # ═══ 初始状态 ═══
    x_t = noise_latent                    # (B, C, H, W) 纯噪声
    sigmas = scheduler.sigmas             # [1.0, 0.98, 0.96, ..., 0.0]

    all_latents = []
    all_log_probs = []

    # ═══ 迭代开始：每一步只改变 x_t 和 sigma ═══
    for step_idx, sigma in enumerate(sigmas[:-1]):
        sigma_next = sigmas[step_idx + 1]

        # ── 模型 forward ──
        # 这个 forward 和上一步的 forward 完全没有因果关系
        # 无法复用任何 KV cache！
        # 因为 x_t 的所有元素都变了

        model_inputs = {
            "hidden_states": x_t,                           # ← 每步不同
            "timestep": sigma.expand(B),                    # ← 每步不同
            "encoder_hidden_states": prompt_embeds,         # ← 复用!
            "encoder_hidden_states_mask": prompt_mask,      # ← 复用!
            "img_shapes": img_shapes,                       # ← 复用!
            "guidance": guidance_emb,                       # ← 复用!
        }

        # 如果有 True-CFG（需要条件+无条件两个 forward）
        if do_cfg:
            neg_inputs = {**model_inputs,
                "encoder_hidden_states": negative_embeds}   # ← 复用!
            neg_pred = model(**neg_inputs)
            pos_pred = model(**model_inputs)
            velocity = pos_pred + cfg_scale * (pos_pred - neg_pred)
        else:
            velocity = model(**model_inputs)

        # ── SDE step ──
        # 计算均值、采样下一步、计算 log-prob
        x_next, log_prob, mean, std = scheduler.step(
            model_output=velocity,
            timestep=sigma,
            sample=x_t,
            noise_level=0.7,
            sde_type="sde",
        )

        # 存储（只在 SDE window 内）
        if in_sde_window(step_idx):
            all_latents.append(x_t)
            all_log_probs.append(log_prob)

        x_t = x_next  # 唯一传递到下一步的东西

    return x_t, all_latents, all_log_probs
```

### 8.4 对比图：LLM vs Diffusion 迭代复用

这个对比非常重要，直接决定了两种模型 infra 设计的根本差异：

```
LLM 自回归 (可以复用 KV cache):
══════════════════════════════════════════════════════
  Step 1: [BOS] ──→ attention(Q1,K1,V1) ──→ "我"
          │                                  │
          ├── KV cache[1] = (K1,V1) ────────┤
          │                                  │
  Step 2: [BOS, "我"] ──→ attention(Q2,[K1,K2],[V1,V2])
          │                  只算 Q2,K2,V2！ 复用 K1,V1
          ├── KV cache[2] = (K2,V2) ────────┤
          │                                  │
  Step 3: [BOS,"我","爱"] ──→ 只算新增的 KV
          ...
  KV cache 增长: O(T²) 内存, O(T²) 计算 → O(T²) 内存, O(T) 计算

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

扩散模型去噪 (无法复用 KV cache):
══════════════════════════════════════════════════════
  Step 1: x₁   ──→ attention(Q₁,K₁,V₁) wrt text ──→ x₀.₉₈
          完全独立的 forward pass

  Step 2: x₀.₉₈ ──→ attention(Q₂,K₂,V₂) wrt text ──→ x₀.₉₆
          完全独立的 forward pass
          所有 latent token 都变了，Q₂ ≠ Q₁ 因为输入变了
          text K,V 理论可以复用（text 不变）但...

  Step 3: x₀.₉₆ ──→ ...

  每次 forward 都是全量计算: O(T) 次独立 forward
  但是！每步 latent 尺寸相同 → 可以用 CUDA Graph 加速！

  text K,V 理论上可以缓存（text 不变），但实际上：
  - Qwen-Image 是 joint attention，text K,V 和 image K,V 在同一个
    attention op 里，框架不支持只缓存一半
  - text 只占 512/(4096+512) ≈ 11% 的序列长度，缓存收益有限
  - 反而增加了显存碎片和调度复杂度
```

### 8.5 扩散模型实际可做的优化（不同于 LLM 的 KV cache）

| 优化手段 | 说明 | 与 LLM 的对比 |
|---|---|---|
| Text embedding 一次编码 | text encoder 只跑一次，输出复用全部步 | 类似 prompt encoding 只做一次 |
| CUDA Graph capture | 因为每步计算图完全相同（只是输入数值不同），可以 capture 整个 denoising step 的 CUDA graph | LLM 不行因为每步 seq_len 变长 |
| VAE tiling | VAE decode 用 tiling 减少峰值显存 | 无类比 |
| Batch 并行去噪 | 多个 prompt 的 latent 一起处理，矩阵乘法的 batch 维度 | LLM continuous batching |
| LoRA | 只训练低秩适配器，全量模型权重冻结 | 跟 LLM LoRA 完全一样 |
| CFG batch 合并 | pos 和 neg 的 latent 拼成 2B 一起算 | 无类比 |

**关键 insight**：扩散模型的性能瓶颈不是 KV cache 和 decode 延迟，而是每一步都需要完整的 model forward。优化方向是减少步数（fewer denoising steps）、提高单步吞吐（larger batch、CUDA Graph）、异步流水线（overlap rollout 和 reward compute）。

---

## 九、可扩展性 —— 双注册机制

VeRL-Omni 的可扩展性来自两个注册表（`verl_omni/pipelines/model_base.py`）：

1. **`DiffusionModelBase`**：训练侧适配器，`(architecture, algorithm) → 具体训练逻辑`
2. **`VllmOmniPipelineBase`**：推理侧管道，`(architecture, algorithm) → vLLM-Omni 自定义 pipeline`

要支持一个新模型，只需：
1. 实现 `DiffusionModelBase` 子类（`build_scheduler`, `set_timesteps`, `prepare_model_inputs`, `forward_and_sample_previous_step`）
2. 实现 vLLM-Omni 自定义 pipeline（`diffuse` 方法）
3. 注册到对应 `(architecture, algorithm)` key 下

配置系统中只需指定 `external_lib=your.module.path` 即可触发自动注册。

---

## 十、术语表

| 术语 | 解释 | LLM 类比 |
|---|---|---|
| Latent | 图像在 VAE 压缩空间的表示 `(B,C,H,W)` | token embedding `(B,L,D)` |
| Timestep / Sigma | 扩散过程的时间参数，σ 越大越"噪声" | 生成步骤 t |
| Velocity (v) | Flow Matching 中的预测目标 `= x_0 - ε` | 类似预测下一个 token 的 logits |
| Score Function | `∇log p_t(x)`，指向高概率密度区域的方向 | 无直接类比 |
| ODE / SDE | 常微分方程 / 随机微分方程 | ODE=贪心解码, SDE=带温度采样 |
| CFG | 用 `pred + scale * (pred_pos - pred_neg)` 增强文本对齐 | contrastive decoding |
| DiT | 用 Transformer 架构的扩散模型 | Decoder-only Transformer |
| VAE | 图像 ↔ latent 的编解码器 | Tokenizer (BPE) |
| SDE Window | 只在这几步注入随机性 + 收集 log-prob | N/A |
| noise_level (η) | 控制 SDE 噪声强度 | temperature |
| num_inference_steps | 去噪步数（通常 20-50） | max_new_tokens |
| FlowGRPO | Flow Matching + Group Relative Policy Optimization | PPO / GRPO |
| DanceGRPO | Dance SDE 变体 + GRPO，数值更稳定 | 同上 |
| GRPO-Guard | FlowGRPO + reverse-SDE mean 漂移惩罚 | PPO + KL penalty |
| DPO (diffusion) | 比较 chosen vs rejected 的 noise pred 误差 | DPO for LLM |
| DiffusionNFT | 前向过程 DPO | N/A |
| Ulysses SP | 序列并行（latent spatial 维度切到不同 GPU） | Tensor Parallel 但切 seq 维度 |

---

## 十一、Infra 视角：Text LLM vs Diffusion/Omni 模型

如果你是做 LLM Infra 的，这一节是专门为你准备的。我们把推理和训练两个维度分开讲，系统地对比 text LLM 和扩散模型在 infra 层面的根本差异。

### 11.1 推理 Infra 对比

#### 请求生命周期

```
Text LLM 推理:
  请求进来 → prefill (一次性算完 prompt) → decode (逐 token 生成)
                                            ↑
                                      每步只算 1 个 token
                                      但需要读 KV cache
                                      延迟敏感，TTFT 和 TPS 是核心指标

扩散模型推理:
  请求进来 → text encode (一次) → N 步去噪 (每步全量 forward)
                                   ↑
                                 每步算整个图片
                                 但步数固定 (20-50)
                                 吞吐敏感，images/sec 是核心指标
```

#### 关键 Infra 差异

| 维度 | Text LLM | 扩散模型 |
|---|---|---|
| **核心瓶颈** | memory-bound（decode 阶段读 KV cache） | compute-bound（每步全量 attention + FFN） |
| **KV cache** | 必须管理，是显存的大头，决定了 batch size 上限 | 不存在。没有跨步的状态需要缓存 |
| **Batching 策略** | continuous batching：不同请求可以在不同 decode 阶段插入/退出 | 静态 batch：同一批请求必须跑完全部 N 步才能返回 |
| **请求调度** | 复杂——每个请求生成长度不同，随时可以结束，需要 iteration-level scheduling | 简单——所有请求步数相同，同时开始同时结束 |
| **显存管理** | PagedAttention 等虚拟显存管理 KV cache | 不需要——每步 latent 尺寸固定且可以原地覆盖 |
| **序列长度变化** | 每个 decode 步 seq_len+1，计算图形状变化 | 每步计算图完全相同——天然适合 CUDA Graph |
| **CUDA Graph** | 只能对 decode 阶段用，且需要 padding 到固定 bucket | 可以 capture 整个 denoising step，覆盖全部推理 |
| **延迟特征** | 两阶段：TTFT（prefill 延迟）+ TPS（decode 速度） | 单阶段：总时间 = N_steps × 单步延迟 + VAE decode |
| **Speculative decoding** | 核心优化方向——用小模型草稿加速大模型 | 不适用——没有自回归的概念 |
| **量化** | W8A8、W4A16 等，主要减少 KV cache 显存和 decode bandwidth | 同样有效，但目标不同：减少每步 forward 的计算量 |
| **Tensor Parallel** | 切 attention heads 和 FFN columns | 类似，但额外有 Ulysses SP（切 latent 的 spatial 维度） |

#### 为什么 vLLM 不能直接用于扩散模型

vLLM 的核心设计假设是自回归生成：

1. **PagedAttention**：专为管理不定长 KV cache 设计。扩散模型没有 KV cache，这整套机制用不上。
2. **Continuous batching**：依赖于不同请求在不同时间结束（有的生成 10 个 token，有的 1000 个）。扩散模型所有请求步数相同，continuous batching 没有意义。
3. **Tokenizer + Detokenizer**：vLLM 假设输入是 token ids，输出也是 token ids。扩散模型输入是 text embedding，输出是高维 latent tensor。
4. **Sampling 逻辑**：vLLM 的 sampling 是从 logits 里挑 token（top-k、top-p 等）。扩散模型的 "sampling" 是 SDE reverse step——完全不同的数学过程。

所以 VeRL-Omni 需要 **vLLM-Omni**：一个专门为扩散模型设计的推理引擎，保留了 vLLM 的 Ray 部署和 async serving 框架，但替换了核心的推理循环。

#### Serving 模式差异

```
Text LLM Serving (vLLM):
  ┌─────────────────────────────────────────────────────────┐
  │  Client → API Server → Scheduler (iteration-level)      │
  │                              │                           │
  │              ┌───────────────┴───────────────┐           │
  │              │         Worker                │           │
  │              │  ┌─────────────────────────┐  │           │
  │              │  │ Model Runner             │  │           │
  │              │  │  prefill → decode loop   │  │           │
  │              │  │  KV cache management     │  │           │
  │              │  │  PagedAttention          │  │           │
  │              │  └─────────────────────────┘  │           │
  │              └───────────────────────────────┘           │
  │                                                         │
  │  核心挑战: KV cache 显存管理、continuous batching、       │
  │           preemption、多请求调度                          │
  └─────────────────────────────────────────────────────────┘

Diffusion Serving (vLLM-Omni):
  ┌─────────────────────────────────────────────────────────┐
  │  Client → API Server → Batch Scheduler (request-level)  │
  │                              │                           │
  │              ┌───────────────┴───────────────┐           │
  │              │         Worker                │           │
  │              │  ┌─────────────────────────┐  │           │
  │              │  │ Pipeline Runner          │  │           │
  │              │  │  text encode (一次)      │  │           │
  │              │  │  N-step denoising loop   │  │           │
  │              │  │  VAE decode              │  │           │
  │              │  │  (无 KV cache)           │  │           │
  │              │  └─────────────────────────┘  │           │
  │              └───────────────────────────────┘           │
  │                                                         │
  │  核心挑战: 大 batch 吞吐、CUDA Graph 优化、              │
  │           VAE 显存峰值、CFG 双倍计算                     │
  └─────────────────────────────────────────────────────────┘
```

### 11.2 训练 Infra 对比

#### 预训练

| 维度 | Text LLM 预训练 | 扩散模型预训练 |
|---|---|---|
| **Loss** | Cross-entropy（next-token prediction） | MSE（velocity/noise prediction） |
| **数据格式** | Token 序列，打包成固定 seq_len | 图像 latent + 文本 embedding，尺寸可能不一致 |
| **数据加载** | 简单：tokenize → pack → shuffle | 复杂：需要先 VAE encode 图像到 latent（通常离线预处理），动态分辨率需要 aspect ratio bucketing |
| **前向计算** | 一次 forward 得到所有 token 的 logits | 一次 forward 只处理一个 `(x_t, t)` 对 |
| **激活显存** | 主要是 attention 矩阵和中间激活 | 类似，但 latent token 数量可能很大（4096+ patches） |
| **数据并行** | FSDP / ZeRO | FSDP，但需要额外处理 VAE 和 text encoder（冻结，不分片） |
| **Pipeline 并行** | 常见（GPipe / 1F1B） | 少见——扩散模型层数虽多但没有 LLM 那么深（60 层 vs 100+ 层） |
| **混合精度** | BF16 为主 | BF16，但 VAE 通常需要 FP32（数值敏感） |
| **Gradient checkpointing** | 常用于减少激活显存 | 同样常用 |

#### RL 后训练（VeRL-Omni 的核心场景）

这是差异最大的地方。LLM 的 RLHF/GRPO 和扩散模型的 FlowGRPO 在 infra 层面有深刻区别：

```
LLM RL 训练循环:
═══════════════════════════════════════════════════════════
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ Rollout  │ ──→ │  Reward  │ ──→ │  Update  │
  │ (vLLM)   │     │ (RM)     │     │ (FSDP)   │
  └──────────┘     └──────────┘     └──────────┘
       │                                  │
       │  输出: token 序列                │  输入: token 序列
       │  大小: (B, seq_len) int64        │  + log_probs: (B, seq_len) float
       │  ~几十 KB/sample                 │  每个 token 独立计算 loss
       │                                  │
       │  瓶颈: KV cache 显存             │  瓶颈: 标准 Transformer forward
       └──────────────────────────────────┘
              weight sync: 传模型权重

Diffusion RL 训练循环:
═══════════════════════════════════════════════════════════
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ Rollout  │ ──→ │  Reward  │ ──→ │  Update  │
  │(vLLM-Omni)    │ (HPSv3)  │     │ (FSDP)   │
  └──────────┘     └──────────┘     └──────────┘
       │                                  │
       │  输出: 高维 latent 轨迹           │  输入: latent 轨迹
       │  all_latents: N步×(B,C,H,W)     │  + log_probs: N步×(B,) float
       │  ~几百 MB/sample !!!             │  需要遍历 SDE window 每步
       │                                  │  重新做 forward + SDE step
       │  瓶颈: 带宽和显存               │
       │  (传输大量 latent tensor)        │  瓶颈: 多次 full forward
       └──────────────────────────────────┘
              weight sync: 传模型权重
```

| 维度 | LLM RL (GRPO/PPO) | Diffusion RL (FlowGRPO) |
|---|---|---|
| **Rollout 产出大小** | `(B, seq_len)` int64 — 几十 KB/sample | `N_steps × (B, C, H, W)` float — **几百 MB/sample** |
| **Rollout → Train 传输** | 轻量级——几个 tensor | 重量级——需要传完整 latent 轨迹（SDE window 内每步的 latent） |
| **Old log-prob 重算** | 一次 forward：整个序列的 logits 都拿到了 | **N 次 forward**：SDE window 内每一步都需要单独 forward + SDE step |
| **Actor forward 次数** | 1 次（整个序列） | N 次（每个 denoising step） |
| **Reward 模型** | 文本 reward model 较轻量 | 视觉 reward model 较重（HPSv3、UnifiedReward 需要图像编码） |
| **Reward 计算依赖** | 需要完整生成文本 | 需要 VAE decode 出图像再评分 |
| **异步 overlap** | Rollout 和 reward 难以 overlap（reward 需要完整文本） | **可以 overlap**：不同 prompt 的 rollout 和 reward 可以流水线化 |
| **Actor-Rollout 共置** | 常见——同一 GPU 交替做 rollout 和 train | VeRL-Omni 也支持 colocated worker |
| **Reference model** | 需要维护一份 frozen ref model（KL penalty） | FlowGRPO 不需要 ref model（GRPO 不用 KL），DPO/NFT 需要 |

#### Rollout 数据传输是 Diffusion RL 独有的难题

这一点值得展开说。在 LLM RL 中，rollout 产生的数据就是一组 token ids 和对应的 log-probs，数据量很小。但在 Diffusion RL 中：

```python
# LLM rollout 数据量
tokens = (batch_size, seq_len)          # int64
log_probs = (batch_size, seq_len)       # float32
# 假设 B=8, seq=2048:
# 数据量 = 8 * 2048 * (8 + 4) = 196 KB

# Diffusion rollout 数据量
all_latents = (sde_window, batch_size, 64, 128, 128)  # float32
all_log_probs = (sde_window, batch_size)               # float32
prompt_embeds = (batch_size, 512, 3584)                # float32
# 假设 B=8, window=4:
# latents = 4 * 8 * 64 * 128 * 128 * 4 = 1.07 GB !!!
# prompt_embeds = 8 * 512 * 3584 * 4 = 56 MB
```

这意味着 Diffusion RL 的 rollout→train 数据通道必须精心设计：
- **Actor-Rollout colocated**（同 GPU）：避免跨机传输，VeRL-Omni 的默认模式
- **SDE Window 剪裁**：只保留 window 内的 latent，大幅减少存储（从 50 步减到 4-8 步）
- **异步 weight sync**：通过 `CheckpointEngine` 异步同步权重，不阻塞训练

### 11.3 GPU 显存布局对比

```
Text LLM 推理显存 (以 70B 模型为例):
══════════════════════════════════════════
  模型权重 (FP16):     ~140 GB
  KV cache:            ~30-60 GB  ← 这是变量，决定了 batch size
  激活 (per batch):    ~2-5 GB
  ──────────────
  总计:                ~170-200 GB (需要多卡 TP)

  关键约束: KV cache 和 batch size 互相争夺显存
  → PagedAttention: 用虚拟页面管理 KV cache
  → 更大 KV cache = 更大 batch = 更高吞吐

═══════════════════════════════════════════

Diffusion 推理显存 (以 Qwen-Image 20B 为例):
══════════════════════════════════════════
  DiT 权重 (FP16):     ~40 GB
  Text Encoder (冻结):  ~15 GB
  VAE (冻结):           ~1 GB
  Latent (per batch):  ~0.5 GB  ← 非常小！
  激活 (per batch):    ~5-10 GB
  ──────────────
  总计:                ~60-65 GB

  关键约束: 没有 KV cache 争夺！
  → batch size 主要受限于激活显存
  → CFG 需要 2× 计算（pos + neg），相当于 batch 翻倍
  → VAE decode 有瞬时峰值（解码 latent 到全分辨率）

═══════════════════════════════════════════

Diffusion RL 训练显存 (VeRL-Omni):
══════════════════════════════════════════
  DiT 权重 (FP16):     ~40 GB
  LoRA 适配器:          ~0.5 GB   ← 如果用 LoRA
  优化器状态 (Adam):   ~80 GB    ← 如果 full finetune
  Text Encoder (冻结):  ~15 GB
  VAE (冻结):           ~1 GB
  Latent 轨迹:         ~1-2 GB  ← SDE window 内的所有 latent
  梯度:                ~40 GB
  激活 (gradient ckpt): ~10-20 GB
  ──────────────
  总计:                ~180+ GB (需要 FSDP 分布式)

  关键约束:
  → 训练需要存 SDE window 内所有 latent（用于 old_log_prob 重算）
  → FSDP 把权重和优化器状态分片到多 GPU
  → LoRA 大幅减少可训练参数（40GB → 0.5GB 优化器状态）
```

### 11.4 Infra 总结：该关注什么

如果你从 LLM Infra 转到 Diffusion/Omni Infra，最需要调整认知的几点：

1. **忘掉 KV cache**：它不存在。扩散模型的显存管理比 LLM 简单得多，不需要 PagedAttention。
2. **Batching 变简单了**：不需要 continuous batching、iteration-level scheduling。所有请求步数相同，静态 batch 就行。
3. **CUDA Graph 变重要了**：LLM 只能在 decode 阶段用 CUDA Graph（且需要 padding），扩散模型每步计算图完全相同，天然适合 CUDA Graph capture。
4. **数据传输变重要了**：LLM 产出 token 序列（KB 级），扩散模型产出 latent 轨迹（GB 级）。Actor-Rollout colocate 是必须的。
5. **多次 forward 代替单次**：LLM 的 loss 在一次 forward 里就算完了，扩散 RL 需要在 SDE window 的每一步分别做 forward，训练循环的 complexity 乘以了步数。
6. **VAE 是额外开销**：LLM 没有类似的编解码瓶颈。扩散模型的 VAE encode/decode 虽然不大，但有显存峰值需要注意（tiling 可以缓解）。
7. **CFG 翻倍计算**：LLM 没有类比。Classifier-Free Guidance 需要同时跑 conditional 和 unconditional 两个 forward——计算量直接翻倍。
8. **Reward 模型更重**：LLM 的 reward model 通常跟 policy model 同构（都是 Transformer），可以复用 infra。扩散模型的 reward（HPSv3、UnifiedReward）是独立的视觉模型，需要额外的 GPU 资源。

---

## 十二、总结

扩散模型 RL 和 LLM RL 的本质是一样的——都是 **PPO 的 ratio × advantage**。唯一不同的是 log-prob 的来源：LLM 来自 softmax 输出，扩散模型来自 SDE 高斯转移概率。理解了这个，其他都是工程细节。

**学习路线建议**：

1. **理解扩散模型基础** → Lilian Weng 博客 + DDPM 论文
2. **理解 Flow Matching** → Yang Song 博客 + Flow Matching 论文 (Lipman 2023)
3. **理解 SDE 和 log-prob** → Score-based SDE 论文 (Song 2021) → VeRL-Omni 的 `flow_match_sde.py`
4. **理解 RL 训练流程** → FlowGRPO 论文 → `flow_grpo` GitHub → VeRL-Omni 的 `diffusion_algos.py`
5. **理解工程架构** → `ray_diffusion_trainer.py` → `engine_workers.py` → `vllm_omni_async_server.py`

---

## 十三、推荐阅读

**必读论文**

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al., 2023，Flow Matching 框架奠基论文
- [FlowGRPO: Training Diffusion Models with RL](https://arxiv.org/abs/2505.05470) — VeRL-Omni 核心算法论文
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al., 2020，DDPM 原始论文
- [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — Song et al., 2021，SDE 框架的数学基础
- [Scaling Rectified Flow Transformers (SD3)](https://arxiv.org/abs/2403.03206) — Stability AI, 2024，实战视角

**博客 / 教程**

- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — Lilian Weng，可能是最好的扩散模型入门博客
- [Understanding Diffusion Models: A Unified Perspective](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html) — Calvin Luo，从 VAE 出发推导
- [Flow Matching Explained](https://yang-song.net/blog/2023/flow-matching/) — Yang Song（Flow Matching 作者博客）
- [Diffusion Models from Scratch](https://huggingface.co/docs/diffusers/tutorials/basic_training) — HuggingFace Diffusers 实战教程

**关键仓库**

- [flow_grpo](https://github.com/yifan123/flow_grpo) — 原始 FlowGRPO 实现，单文件 SD3 训练代码
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) — 扩散模型标准库
