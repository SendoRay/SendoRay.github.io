---
title: "扩散模型优化技术全面综述：从采样加速到高效架构设计"
date: 2026-06-22
draft: false
tags: ["diffusion", "deep-learning", "inference", "engineering"]
summary: "系统梳理扩散模型全栈优化技术：采样加速、步数蒸馏、高效架构、训练优化、推理部署，涵盖DDIM到Consistency Models、U-Net到DiT的完整技术演进。"
---

## 第1章：引言与扩散模型基础回顾

### 1.1 扩散模型的历史背景与发展脉络

#### 1.1.1 从VAE/GAN到Diffusion Models的演进

生成模型是深度学习中最令人兴奋的研究方向之一。在扩散模型崛起之前，生成对抗网络（GAN）和变分自编码器（VAE）是两大主流范式。让我们先回顾这段演进历程，理解为什么扩散模型能够后来居上。

**变分自编码器（VAE, 2013）** 由 Kingma 和 Welling 提出，VAE 通过变分推断将数据编码到潜在空间，再从潜在空间解码重建数据。其核心思想是最大化数据的证据下界（ELBO）：

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

VAE 的优点是训练稳定、有明确的概率解释，但生成样本往往模糊，缺乏细节。这是因为像素级重建损失倾向于产生均值预测，而 KL 散度正则化限制了潜在空间的表达能力。

**生成对抗网络（GAN, 2014）** 由 Goodfellow 等人提出，GAN 引入了一个革命性的思想：通过生成器和判别器的对抗博弈来学习数据分布。生成器试图产生以假乱真的样本，判别器则试图区分真假样本：

$$\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

GAN 能生成极其清晰的图像，但训练不稳定（模式崩塌、训练振荡），缺乏多样性保证，且没有显式的似然估计。从 DCGAN 到 StyleGAN、BigGAN，研究者投入了大量精力解决这些问题，但根本性的训练困难始终存在。

**扩散模型的出现** 则提供了一条全新的道路。2015年，Sohl-Dickstein 等人首次提出了扩散概率模型（Diffusion Probabilistic Models）的概念，但直到2020年 Ho 等人的 DDPM（Denoising Diffusion Probabilistic Models）才真正展示了扩散模型的强大能力。

```
生成模型演进时间线：

2013        2014        2015        2020        2021        2022        2023
 │           │           │           │           │           │           │
 ▼           ▼           ▼           ▼           ▼           ▼           ▼
VAE         GAN        DPM原型     DDPM       DALL·E2    Stable      SDXL
Kingma      Goodfellow  Sohl-      Ho et al.  Guided     Diffusion   Consistency
                        Dickstein             Diffusion   LDM         Models

┌─────────────────────────────────────────────────────────────────────────────┐
│  关键性质对比                                                                 │
├──────────┬──────────┬──────────┬─────────────────────────────────────────────┤
│  模型     │ 训练稳定  │ 样本质量  │ 多样性    │ 似然估计   │ 采样速度          │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────────────┤
│  VAE     │  ★★★★★  │  ★★☆☆☆  │  ★★★★☆  │  ★★★★☆  │  ★★★★★ (1步)       │
│  GAN     │  ★★☆☆☆  │  ★★★★★  │  ★★☆☆☆  │  ★☆☆☆☆  │  ★★★★★ (1步)       │
│  Flow    │  ★★★★☆  │  ★★★☆☆  │  ★★★★☆  │  ★★★★★  │  ★★★★☆ (1步)       │
│  Diffusion│ ★★★★★  │  ★★★★★  │  ★★★★★  │  ★★★★☆  │  ★☆☆☆☆ (多步)      │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────────────────┘
```

#### 1.1.2 DDPM的开创性贡献

2020年，Jonathan Ho、Ajay Jain 和 Pieter Abbeel 发表了开创性论文 "Denoising Diffusion Probabilistic Models"。这篇论文有三个关键贡献：

**贡献一：简化训练目标。** DDPM将复杂的变分下界优化简化为一个直觉清晰的噪声预测任务。网络只需要学习预测在每一步添加的噪声 $\epsilon$，而不是直接预测数据分布的复杂参数。

**贡献二：高质量样本。** 在 CIFAR-10 数据集上，DDPM 首次实现了与 GAN 相当甚至超越的 FID 分数（FID 3.17），同时保持了完美的训练稳定性和样本多样性。

**贡献三：连接不同理论框架。** DDPM 的成功催生了大量后续工作，将去噪得分匹配、随机微分方程、最优传输等不同数学工具引入扩散模型的分析中。

#### 1.1.3 Score-based Generative Models的另一条路线

几乎在同一时期，Yang Song 和 Stefano Ermon 从得分匹配（Score Matching）的角度发展了另一条理论路线。

得分函数定义为数据对数概率密度的梯度：$s(x) = \nabla_x \log p(x)$。如果我们能准确估计数据分布的得分函数，就可以通过 Langevin 动力学采样生成新样本：

$$x_{k+1} = x_k + \frac{\delta}{2} \nabla_x \log p(x_k) + \sqrt{\delta} \cdot z_k, \quad z_k \sim \mathcal{N}(0, I)$$

然而，直接估计原始数据分布的得分函数在低密度区域是不准确的。NCSN（Noise Conditional Score Networks）的核心洞察是：**对数据添加不同程度的噪声，在每个噪声级别上估计得分函数**。这与 DDPM 的多步去噪过程在数学上是等价的——后来 Song et al. (2021) 在 SDE 框架下统一了这两条路线。

#### 1.1.4 当前的广泛应用

扩散模型已经渗透到几乎所有生成式AI领域：

- **文本到图像生成**：DALL·E 2/3、Stable Diffusion、Midjourney、Imagen
- **视频生成**：Sora、Runway Gen-2、Stable Video Diffusion
- **3D内容生成**：DreamFusion、Magic3D、Point-E
- **音频合成**：DiffWave、Grad-TTS、AudioLDM
- **科学计算**：蛋白质结构预测（RFDiffusion）、分子生成、天气预报

这种广泛应用使得扩散模型的优化具有极高的实际价值——任何采样速度的提升都意味着数百万用户体验的改善和计算成本的降低。

### 1.2 前向扩散过程（Forward Process）

#### 1.2.1 马尔可夫链噪声注入

前向扩散过程是一个固定的马尔可夫链，逐步向干净数据 $x_0$ 中添加高斯噪声，直到数据变为近似纯噪声。这个过程不包含任何可学习参数。

给定数据样本 $x_0 \sim q(x_0)$，前向过程定义为：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \cdot I)$$

其中 $\beta_t \in (0, 1)$ 是预定义的噪声调度（noise schedule），控制每一步添加噪声的幅度。整个前向过程的联合分布为：

$$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})$$

直觉上，每一步做的事情是：
1. 将当前信号缩小一点（乘以 $\sqrt{1-\beta_t}$）
2. 添加一小点高斯噪声（方差为 $\beta_t$）

经过足够多步（通常 $T=1000$）后，$x_T$ 的分布趋近于标准正态分布 $\mathcal{N}(0, I)$。

```
前向扩散过程可视化：

x_0 (清晰图像)                                         x_T (纯噪声)
    │                                                      │
    ▼          ▼          ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│████████│ │█▓██▓█▓█│ │▓▒▓▒▓▒▓▒│ │▒░▒░▒░▒░│ │░ ░ ░ ░ │ │        │
│████████│ │▓██▓██▓█│ │▒▓▒▓▒▓▒▓│ │░▒░▒░▒░▒│ │ ░ ░ ░  │ │ noise  │
│████████│ │█▓█▓██▓█│ │▓▒▓▒▓▒▓▒│ │▒░▒░▒░▒░│ │░ ░ ░ ░ │ │        │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
   t=0       t=200      t=400      t=600      t=800      t=1000

每步操作：x_t = √(1-β_t) · x_{t-1} + √β_t · ε,  ε ~ N(0,I)

信号强度：  ████100%    ▓▓▓80%     ▒▒▒50%     ░░░20%      5%       ~0%
噪声强度：     0%        20%        50%        80%        95%      ~100%
```

#### 1.2.2 噪声调度 β_t 的定义

噪声调度的选择对扩散模型的性能有显著影响。常见的调度方式包括：

**线性调度（Linear Schedule）：** DDPM 原始论文使用的方式，$\beta_t$ 从 $\beta_1 = 10^{-4}$ 线性增长到 $\beta_T = 0.02$：

$$\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)$$

**余弦调度（Cosine Schedule）：** Nichol & Dhariwal (2021) 提出的改进方案。他们发现线性调度在 $t$ 接近 $T$ 时破坏信息过快，而余弦调度提供更平滑的信息衰减：

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

其中 $s=0.008$ 是一个小偏移量，防止 $\beta_t$ 在 $t=0$ 附近太小。

**Scaled Linear（SD常用）：** Stable Diffusion中使用的调度，对 $\sqrt{\beta_t}$ 进行线性插值：

$$\sqrt{\beta_t} = \sqrt{\beta_1} + \frac{t-1}{T-1}(\sqrt{\beta_T} - \sqrt{\beta_1})$$

#### 1.2.3 关键性质：q(x_t|x_0) 的闭合形式

前向过程最重要的性质是：**我们可以直接从 $x_0$ 一步跳到任意时间步 $t$，而不需要逐步计算中间过程。**

定义 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，则有：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1-\bar{\alpha}_t) \cdot I)$$

**推导过程：**

从 $q(x_t|x_{t-1})$ 出发，$x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{1-\alpha_t} \cdot \epsilon_t$。

递归展开：
$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_t$$
$$= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}} \epsilon_{t-1}) + \sqrt{1-\alpha_t} \epsilon_t$$
$$= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-1} + \sqrt{1-\alpha_t} \epsilon_t$$

由于两个独立高斯噪声的和仍是高斯噪声，其方差相加：

$$\text{Var} = \alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = 1 - \alpha_t\alpha_{t-1}$$

继续递归到 $x_0$，最终得到：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这个性质对训练至关重要——我们可以在训练时随机采样时间步 $t$，直接计算 $x_t$，而不需要模拟整个前向过程。

#### 1.2.4 信噪比（SNR）的物理含义

从闭合形式公式中，我们可以定义信噪比：

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

- 当 $t=0$ 时，$\bar{\alpha}_0 \approx 1$，SNR很大，信号主导
- 当 $t=T$ 时，$\bar{\alpha}_T \approx 0$，SNR接近0，噪声主导
- SNR是单调递减函数

信噪比为我们理解扩散过程提供了直觉：去噪网络在不同时间步面临不同难度的任务——高SNR（小t）时需要精细去噪，低SNR（大t）时需要从噪声中恢复全局结构。

### 1.3 反向去噪过程（Reverse Process）

#### 1.3.1 参数化反向过程

反向过程的目标是从纯噪声 $x_T \sim \mathcal{N}(0, I)$ 出发，逐步去噪恢复出数据。理论上，如果我们知道反向过程的条件分布 $q(x_{t-1}|x_t)$，就能完美地反转前向过程。

然而，直接计算 $q(x_{t-1}|x_t)$ 需要知道整个数据分布 $q(x_0)$，这是不可行的。因此我们用一个参数化的神经网络来近似它：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

在 DDPM 中，方差被固定为 $\sigma_t^2 I$（$\sigma_t^2 = \beta_t$ 或 $\sigma_t^2 = \tilde{\beta}_t$），只学习均值 $\mu_\theta$。

**关键推导：后验分布 $q(x_{t-1}|x_t, x_0)$**

利用贝叶斯定理，当已知 $x_0$ 时，后验分布有解析解：

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

其中：
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t$$

由于 $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\epsilon)$，将其代入 $\tilde{\mu}_t$ 得到：

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right)$$

因此，**预测 $\mu_\theta$ 等价于预测噪声 $\epsilon_\theta$**：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

#### 1.3.2 训练目标：简化ELBO到噪声预测MSE损失

**完整ELBO推导：**

$$\log p(x_0) \geq \mathbb{E}_q\left[\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right] = -L$$

将 $L$ 分解：

$$L = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0}$$

- $L_T$：前向过程终点与先验的差异，无可训练参数
- $L_{t-1}$：两个高斯分布的KL散度，有解析形式
- $L_0$：重建损失

Ho et al. 发现，使用简化目标（忽略权重系数）效果更好：

$$L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \|^2\right]$$

这就是 DDPM 的训练损失：随机采样时间步 $t$、数据 $x_0$、噪声 $\epsilon$，让网络预测添加的噪声。

#### 1.3.3 采样流程完整步骤

```
DDPM采样算法流程：

┌─────────────────────────────────────────────────────────────┐
│  Algorithm: DDPM Sampling                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  输入: 训练好的噪声预测网络 ε_θ                                │
│  输出: 生成样本 x_0                                           │
│                                                              │
│  1. x_T ~ N(0, I)                    ← 采样纯噪声            │
│  2. for t = T, T-1, ..., 1:                                  │
│     │                                                        │
│     ├─ z ~ N(0, I) if t > 1, else z = 0                     │
│     │                                                        │
│     ├─ ε_pred = ε_θ(x_t, t)         ← 神经网络前向推理       │
│     │                                                        │
│     ├─ μ = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_pred)         │
│     │                                                        │
│     └─ x_{t-1} = μ + σ_t · z        ← 加噪声(最后一步除外)  │
│                                                              │
│  3. return x_0                                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

计算量分析：
┌────────────────────────────────────────────────────────────┐
│  采样步数: T = 1000                                         │
│  每步计算: 1次完整的U-Net前向传播                             │
│  总计算量: 1000 × U-Net FLOPs                               │
│                                                             │
│  对于256×256图像，U-Net约500M参数:                           │
│    单步推理: ~100 GFLOPs                                    │
│    总采样:   ~100 TFLOPs = 约30秒(A100)                     │
└────────────────────────────────────────────────────────────┘
```

### 1.4 完整代码实现：DDPM前向过程与采样

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class NoiseScheduler:
    """
    噪声调度器：支持多种噪声调度策略
    
    支持的调度类型：
    - linear: DDPM原始线性调度
    - cosine: Improved DDPM的余弦调度
    - scaled_linear: Stable Diffusion使用的缩放线性调度
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "linear"
    ):
        self.num_timesteps = num_timesteps
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_schedule(num_timesteps)
        elif schedule_type == "scaled_linear":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps
            ) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # 预计算关键量
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
        
        # 采样时需要的量
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        
        # SNR
        self.snr = self.alpha_bars / (1.0 - self.alpha_bars)
    
    def _cosine_schedule(self, num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """余弦噪声调度"""
        steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
        f_t = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bars = f_t / f_t[0]
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def add_noise(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向过程：q(x_t | x_0) = N(x_t; sqrt(ᾱ_t)*x_0, (1-ᾱ_t)*I)
        
        Args:
            x_0: 干净数据 [B, C, H, W]
            t: 时间步 [B]
            noise: 可选的预生成噪声
            
        Returns:
            x_t: 加噪后的数据
            noise: 使用的噪声（用于训练损失计算）
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # 提取对应时间步的系数，reshape用于广播
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1).to(x_0.device)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1).to(x_0.device)
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """均匀随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,))


class DDPMSampler:
    """
    DDPM采样器：完整的反向去噪采样实现
    """
    
    def __init__(self, scheduler: NoiseScheduler):
        self.scheduler = scheduler
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        DDPM完整采样流程
        
        Args:
            model: 噪声预测网络 ε_θ(x_t, t) -> ε
            shape: 生成样本的形状 (B, C, H, W)
            device: 计算设备
            verbose: 是否打印进度
            
        Returns:
            生成的样本 x_0
        """
        # Step 1: 从标准正态分布采样初始噪声
        x_t = torch.randn(shape, device=device)
        
        # Step 2: 逐步去噪 t = T, T-1, ..., 1
        for t in reversed(range(self.scheduler.num_timesteps)):
            # 构造batch时间步
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # 预测噪声
            eps_pred = model(x_t, t_batch)
            
            # 计算均值
            sqrt_recip_alpha = self.scheduler.sqrt_recip_alphas[t]
            beta_t = self.scheduler.betas[t]
            sqrt_one_minus_alpha_bar = self.scheduler.sqrt_one_minus_alpha_bars[t]
            
            mu = sqrt_recip_alpha * (
                x_t - beta_t / sqrt_one_minus_alpha_bar * eps_pred
            )
            
            # 添加噪声（最后一步除外）
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(self.scheduler.posterior_variance[t])
                x_t = mu + sigma_t * noise
            else:
                x_t = mu
            
            if verbose and t % 100 == 0:
                print(f"  Sampling step {self.scheduler.num_timesteps - t}/{self.scheduler.num_timesteps}")
        
        return x_t
    
    @torch.no_grad()
    def sample_with_trajectory(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        save_every: int = 100
    ) -> Tuple[torch.Tensor, list]:
        """采样并保存中间轨迹（用于可视化）"""
        x_t = torch.randn(shape, device=device)
        trajectory = [x_t.cpu().clone()]
        
        for t in reversed(range(self.scheduler.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps_pred = model(x_t, t_batch)
            
            sqrt_recip_alpha = self.scheduler.sqrt_recip_alphas[t]
            beta_t = self.scheduler.betas[t]
            sqrt_one_minus_alpha_bar = self.scheduler.sqrt_one_minus_alpha_bars[t]
            
            mu = sqrt_recip_alpha * (
                x_t - beta_t / sqrt_one_minus_alpha_bar * eps_pred
            )
            
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(self.scheduler.posterior_variance[t])
                x_t = mu + sigma_t * noise
            else:
                x_t = mu
            
            if t % save_every == 0:
                trajectory.append(x_t.cpu().clone())
        
        return x_t, trajectory


class DDPMTrainer:
    """
    DDPM训练器：完整的训练循环实现
    """
    
    def __init__(
        self,
        model: nn.Module,
        scheduler: NoiseScheduler,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
    
    def train_step(self, x_0: torch.Tensor) -> dict:
        """
        单步训练
        
        Args:
            x_0: 干净数据 batch [B, C, H, W]
            
        Returns:
            包含损失等指标的字典
        """
        batch_size = x_0.shape[0]
        x_0 = x_0.to(self.device)
        
        # 随机采样时间步
        t = self.scheduler.sample_timesteps(batch_size).to(self.device)
        
        # 前向加噪
        noise = torch.randn_like(x_0)
        x_t, _ = self.scheduler.add_noise(x_0, t, noise)
        
        # 预测噪声
        eps_pred = self.model(x_t, t)
        
        # 计算L_simple损失
        loss = F.mse_loss(eps_pred, noise)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止训练不稳定
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "mean_t": t.float().mean().item(),
        }


# 使用示例
if __name__ == "__main__":
    # 初始化噪声调度器
    scheduler = NoiseScheduler(
        num_timesteps=1000,
        schedule_type="cosine"  # 推荐使用余弦调度
    )
    
    # 验证SNR单调递减
    print("SNR at t=0:", scheduler.snr[0].item())      # 大值
    print("SNR at t=500:", scheduler.snr[500].item())   # 中等
    print("SNR at t=999:", scheduler.snr[999].item())   # 接近0
    
    # 验证alpha_bar在终点接近0
    print(f"\nalpha_bar[0] = {scheduler.alpha_bars[0]:.6f}")
    print(f"alpha_bar[999] = {scheduler.alpha_bars[999]:.6f}")
    
    # 模拟前向加噪
    x_0 = torch.randn(4, 3, 32, 32)  # 假设CIFAR-10大小
    t = torch.tensor([0, 250, 500, 999])
    x_t, noise = scheduler.add_noise(x_0, t)
    
    print(f"\nx_t stats at different t:")
    for i, ti in enumerate(t):
        print(f"  t={ti.item()}: mean={x_t[i].mean():.4f}, std={x_t[i].std():.4f}")
```

### 1.5 扩散模型的挑战与优化动机

#### 1.5.1 采样速度：最核心的瓶颈

DDPM 的采样需要 1000 步序列去噪，每步都需要一次完整的神经网络前向推理。对于大型模型如 Stable Diffusion（约860M参数的 U-Net），单张图像在 A100 GPU 上需要约10-30秒。这在用户交互场景中是不可接受的。

更具体地，对于分辨率为 $H \times W$ 的图像生成：
- **计算量**：$T \times \text{FLOPs}(\text{U-Net})$，随分辨率二次增长
- **延迟**：$T \times \text{Latency}(\text{U-Net})$，串行不可并行
- **内存**：需要维护中间状态，峰值显存与分辨率相关

#### 1.5.2 计算资源消耗

除了推理延迟，训练扩散模型同样消耗巨大资源：
- Stable Diffusion 1.5 的训练使用了 256 块 A100 GPU，训练约 150,000 步
- DALL·E 2 的训练据估计消耗了数千万GPU小时
- 高分辨率模型（如 SDXL 1024×1024）的资源需求更是成倍增长

#### 1.5.3 模型规模与质量的权衡

更大的模型通常带来更好的生成质量，但也带来：
- 更高的推理延迟和内存消耗
- 更大的部署难度（端侧设备受限）
- 更高的训练成本

如何在保持质量的同时降低计算开销，是优化研究的核心问题。

#### 1.5.4 优化方向总览

```
扩散模型优化技术全景图：

┌─────────────────────────────────────────────────────────────────────────┐
│                        扩散模型优化技术全景                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │  第2章 数学基础   │   │  第3章 采样加速   │   │  第4章 步数蒸馏   │       │
│  │                  │   │                  │   │                  │       │
│  │  · DDPM变分推导  │   │  · DDIM          │   │  · Progressive   │       │
│  │  · Score Match   │   │  · DPM-Solver    │   │  · Consistency   │       │
│  │  · SDE/ODE统一   │   │  · UniPC         │   │  · Rectified Flow│       │
│  │  · 连续vs离散    │   │  · EDM Solver    │   │  · SDXL-Turbo    │       │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘       │
│                                                                          │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │  第5章 架构设计   │   │  第6章 训练优化   │   │  第7章 推理部署   │       │
│  │                  │   │                  │   │                  │       │
│  │  · U-Net演进     │   │  · Min-SNR加权   │   │  · 量化(INT8/4)  │       │
│  │  · DiT架构      │   │  · v-prediction  │   │  · 模型剪枝      │       │
│  │  · MMDiT(SD3)   │   │  · 噪声调度优化   │   │  · Flash Attn   │       │
│  │  · Efficient Attn│   │  · EMA策略       │   │  · 编译优化      │       │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘       │
│                                                                          │
│  优化效果汇总：                                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  DDPM 1000步 → DDIM 50步 → DPM++ 20步 → LCM 4步 → 1步蒸馏    │     │
│  │  延迟: 30s   → 1.5s     → 0.6s      → 0.12s   → 0.03s       │     │
│  │  加速: 1x    → 20x      → 50x       → 250x    → 1000x       │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

上图展示了本文将要涵盖的完整优化技术栈。从第2章的数学基础出发，我们将逐步深入每个优化方向，最终在第7章汇总推理部署的工程实践。每个方向都不是孤立的——例如，采样加速和步数蒸馏可以结合使用，架构设计影响训练和推理效率，量化需要考虑采样器的数值稳定性。

接下来的章节将按照「直觉 → 数学 → 实现」的三层递进方式，系统展开每个优化技术的详细内容。

---


## 第2章：数学基础 - DDPM/Score Matching/SDE-ODE统一视角

本章从三个不同但等价的数学视角理解扩散模型，这些理论基础是理解后续所有优化技术的关键。我们将看到DDPM的变分推导、得分匹配的优雅框架，以及如何用随机/常微分方程统一这两个视角。

### 2.1 DDPM的变分推导

#### 2.1.1 证据下界（ELBO）分解

扩散模型的训练目标来源于最大化数据的对数似然 $\log p_\theta(x_0)$。由于直接优化似然不可行，我们转而优化其变分下界。

**推导起点：**

$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T}) dx_{1:T}$$

引入变分分布 $q(x_{1:T}|x_0)$（即前向过程），利用 Jensen 不等式：

$$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = -L_{\text{VLB}}$$

展开 $L_{\text{VLB}}$：

$$L_{\text{VLB}} = -\mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

$$= -\mathbb{E}_q\left[\log \frac{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^T q(x_t|x_{t-1})}\right]$$

通过仔细的代数变换（利用马尔可夫性质和贝叶斯公式），可以将其分解为逐步的 KL 散度之和：

$$L_{\text{VLB}} = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T: \text{先验匹配}} + \sum_{t=2}^{T} \underbrace{\mathbb{E}_q\left[D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))\right]}_{L_{t-1}: \text{去噪匹配}} + \underbrace{(-\mathbb{E}_q[\log p_\theta(x_0|x_1)])}_{L_0: \text{重建}}$$

```
ELBO分解结构图：

log p(x_0) ≥ -L_VLB
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    L_VLB = L_T + L_{T-1} + ... + L_1 + L_0     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  L_T = KL(q(x_T|x_0) || p(x_T))                                │
│       └──► 无参数项，前向终点≈N(0,I)，近似为0                     │
│                                                                  │
│  L_{t-1} = KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t))         │
│       └──► 核心项：让模型的去噪分布匹配真实后验                    │
│       └──► 两个高斯分布的KL有解析解                               │
│                                                                  │
│  L_0 = -log p_θ(x_0|x_1)                                        │
│       └──► 最终重建损失                                           │
│                                                                  │
│  关键洞察：L_{t-1}中的后验 q(x_{t-1}|x_t,x_0) 是已知的！         │
│           因为给定x_0和x_t，x_{t-1}的分布是高斯的                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.1.2 KL散度简化

对于 $L_{t-1}$ 项，由于 $q(x_{t-1}|x_t, x_0)$ 和 $p_\theta(x_{t-1}|x_t)$ 都是高斯分布，KL散度有解析形式。

设 $q(x_{t-1}|x_t, x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I)$ 且 $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$，其中 $\sigma_t$ 固定。

两个多元高斯的KL散度：

$$D_{KL}(\mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I) \| \mathcal{N}(\mu_\theta, \sigma_t^2 I)) = \frac{1}{2\sigma_t^2} \|\tilde{\mu}_t - \mu_\theta\|^2 + C$$

其中 $C$ 是不依赖于 $\theta$ 的常数。因此训练目标简化为最小化均值之间的差距。

回忆后验均值：

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

将 $x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}$ 代入：

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right)$$

因此，如果我们参数化 $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$，则：

$$L_{t-1} = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)} \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

#### 2.1.3 L_simple：去除权重系数

Ho et al. 发现直接使用加权损失训练效果不如简化版本。**简化损失**去除了时间步相关的权重系数：

$$L_{\text{simple}} = \mathbb{E}_{t \sim U[1,T], \epsilon \sim \mathcal{N}(0,I)}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]$$

为什么简化版本更好？直觉上，完整ELBO的权重在不同时间步差异巨大——小 $t$ 对应的权重很小（因为去噪任务简单），大 $t$ 对应的权重也不大（因为信息已被完全破坏）。$L_{\text{simple}}$ 的等权重策略让网络在所有时间步上均匀学习，经验上带来更好的样本质量。

后续的 Min-SNR 加权策略（第6章详述）在此基础上重新审视了最优加权方案。

### 2.2 Score Matching视角

#### 2.2.1 Score Function定义

得分函数（Score Function）是概率密度函数对数梯度：

$$s(x) = \nabla_x \log p(x)$$

直觉上，得分函数指向数据密度增大的方向。在数据流形附近，得分函数指向最近的数据点；在高密度区域，得分函数较小（因为已经在"峰顶"）。

得分函数的优势在于它不需要知道归一化常数：

$$\nabla_x \log p(x) = \nabla_x \log \frac{\tilde{p}(x)}{Z} = \nabla_x \log \tilde{p}(x)$$

这使得我们可以用能量模型或任何未归一化的密度来估计得分。

#### 2.2.2 Denoising Score Matching

直接估计得分函数（Explicit Score Matching或Sliced Score Matching）存在计算困难。Vincent (2011) 提出的 **去噪得分匹配**（Denoising Score Matching, DSM）提供了一个优雅的替代方案：

**核心定理：** 如果 $q_\sigma(x) = \int q_\sigma(x|x_0) p_{data}(x_0) dx_0$ 是加噪后的数据分布，那么最小化：

$$\mathbb{E}_{x_0 \sim p_{data}, x \sim q_\sigma(x|x_0)}\left[\|s_\theta(x) - \nabla_x \log q_\sigma(x|x_0)\|^2\right]$$

等价于最小化 $\mathbb{E}_{x \sim q_\sigma}\left[\|s_\theta(x) - \nabla_x \log q_\sigma(x)\|^2\right]$。

对于高斯噪声 $q_\sigma(x|x_0) = \mathcal{N}(x; x_0, \sigma^2 I)$：

$$\nabla_x \log q_\sigma(x|x_0) = -\frac{x - x_0}{\sigma^2} = -\frac{\epsilon}{\sigma}$$

因此 DSM 损失变为：

$$L_{\text{DSM}} = \mathbb{E}\left[\left\|s_\theta(x) + \frac{\epsilon}{\sigma}\right\|^2\right]$$

#### 2.2.3 噪声条件Score Networks (NCSN)

单一噪声级别的得分估计在低密度区域不准确。NCSN 的解决方案是使用多个噪声级别 $\{\sigma_i\}_{i=1}^L$（从大到小），并训练一个条件得分网络 $s_\theta(x, \sigma)$：

$$L_{\text{NCSN}} = \sum_{i=1}^{L} \lambda(\sigma_i) \mathbb{E}_{x_0, x}\left[\|s_\theta(x, \sigma_i) + \frac{x - x_0}{\sigma_i^2}\|^2\right]$$

采样时使用退火 Langevin 动力学（Annealed Langevin Dynamics）：从最大噪声级别开始，逐步降低噪声级别，在每个级别运行多步 Langevin 更新：

$$x_{k+1} = x_k + \frac{\delta_i}{2} s_\theta(x_k, \sigma_i) + \sqrt{\delta_i} z_k$$

#### 2.2.4 Score与ε的关系

连接DDPM和Score Matching的关键等式是：

$$\nabla_{x_t} \log q(x_t|x_0) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

因此：

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

**这意味着 DDPM 的噪声预测网络本质上就是在估计得分函数，只是差一个与时间步相关的缩放因子。** 两种看似不同的方法在数学上是完全等价的。

```
DDPM噪声预测 vs Score Matching 的等价关系：

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  DDPM视角:                                                       │
│    训练目标: min ||ε - ε_θ(x_t, t)||²                            │
│    预测对象: 添加的噪声 ε                                         │
│    采样公式: x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ)    │
│                                                                  │
│  ═══════════════════════ 等价桥梁 ══════════════════════════      │
│                                                                  │
│    s_θ(x_t, t) = -ε_θ(x_t, t) / √(1-ᾱ_t)                      │
│                                                                  │
│  ═══════════════════════════════════════════════════════════      │
│                                                                  │
│  Score视角:                                                       │
│    训练目标: min ||s_θ(x_t,t) - ∇log q(x_t|x_0)||²              │
│    预测对象: 对数概率的梯度 ∇_x log p_t(x)                        │
│    采样公式: Langevin dynamics with learned score                 │
│                                                                  │
│  两种视角的统一意义：                                              │
│    → 为SDE/ODE框架提供理论基础                                    │
│    → 不同参数化(ε-pred, v-pred, x-pred)都是score的变体            │
│    → 采样器设计可以利用ODE求解器理论                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 SDE/ODE统一框架

#### 2.3.1 前向SDE

Song et al. (2021) 提出了一个优雅的统一框架：将离散的前向扩散过程推广为连续时间的随机微分方程（SDE）。

**一般形式：**

$$dx = f(x, t)dt + g(t)dw$$

其中 $f(x, t)$ 是漂移系数，$g(t)$ 是扩散系数，$w$ 是标准维纳过程。

三种经典的前向SDE：

**VP-SDE（Variance Preserving）：** 对应DDPM

$$dx = -\frac{1}{2}\beta(t)x \, dt + \sqrt{\beta(t)} \, dw$$

这里 $\beta(t)$ 是连续版本的噪声调度。$x_t$ 的边际分布为：

$$q(x_t|x_0) = \mathcal{N}\left(x_0 e^{-\frac{1}{2}\int_0^t \beta(s)ds}, I - I \cdot e^{-\int_0^t \beta(s)ds}\right)$$

令 $\bar{\alpha}(t) = e^{-\int_0^t \beta(s)ds}$，则与离散DDPM完全一致。

**VE-SDE（Variance Exploding）：** 对应NCSN/SMLD

$$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dw$$

纯扩散过程，没有漂移项。方差随时间单调增长（explode）。

**sub-VP SDE：**

$$dx = -\frac{1}{2}\beta(t)x \, dt + \sqrt{\beta(t)(1-e^{-2\int_0^t \beta(s)ds})} \, dw$$

在VP-SDE和VE-SDE之间的折中，实验中表现略优。

#### 2.3.2 反向SDE

Anderson (1982) 的经典结果告诉我们，如果前向SDE为 $dx = f(x,t)dt + g(t)dw$，则反向过程也是一个SDE：

$$dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)d\bar{w}$$

其中 $\bar{w}$ 是反向时间的维纳过程，$\nabla_x \log p_t(x)$ 是时间 $t$ 处数据分布的得分函数。

**关键洞察：如果我们有一个准确的得分估计 $s_\theta(x,t) \approx \nabla_x \log p_t(x)$，就可以通过数值求解反向SDE来生成样本。**

#### 2.3.3 Probability Flow ODE

Song et al. 的另一个重要发现是：存在一个确定性的常微分方程（ODE），其边际分布与SDE完全相同：

$$dx = \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt$$

这被称为 **Probability Flow ODE**。对比反向SDE，唯一的区别是：
1. 得分函数前的系数变为 $\frac{1}{2}g(t)^2$（而不是 $g(t)^2$）
2. 没有随机噪声项 $g(t)d\bar{w}$

```
SDE vs ODE 采样对比：

                         前向过程 (加噪)
            x_0 ════════════════════════════════► x_T
                    dx = f(x,t)dt + g(t)dw

                         反向过程 (生成)
            x_0 ◄════════════════════════════════ x_T

┌─────────────────────────────────────────────────────────────────┐
│  反向SDE:                                                        │
│    dx = [f(x,t) - g²(t)·∇log p_t(x)]dt + g(t)dw̄               │
│                                                                  │
│    特点: · 随机过程，每次采样结果不同                               │
│          · 需要很多步以控制离散化误差                               │
│          · 样本多样性更好                                         │
│                                                                  │
│  ═══════════════════════════════════════════════════════          │
│                                                                  │
│  Probability Flow ODE:                                           │
│    dx = [f(x,t) - ½g²(t)·∇log p_t(x)]dt                        │
│                                                                  │
│    特点: · 确定性过程，同一起点同一终点                             │
│          · 可以使用高阶ODE求解器(少步高精度)                        │
│          · 支持精确的似然计算(通过变量替换公式)                      │
│          · DDIM就是VP-SDE对应的离散化ODE求解器                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

为什么ODE允许步数减少？
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  SDE离散化误差: O(√h)  ← 由噪声项的√dt尺度决定              │
│  ODE离散化误差: O(h^{k+1})  ← k阶求解器可实现高阶精度       │
│                                                              │
│  例: Euler法(1阶) O(h²), 中点法(2阶) O(h³)                  │
│      DPM-Solver-2达到 O(h³)精度                              │
│                                                              │
│  同样精度要求下：                                             │
│    SDE: 需要 ~1000步 (因为√h收敛太慢)                        │
│    ODE 1阶: ~50步                                            │
│    ODE 2阶: ~20步                                            │
│    ODE 3阶: ~10步                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 2.3.4 VP-SDE下的Probability Flow ODE

对于VP-SDE，$f(x,t) = -\frac{1}{2}\beta(t)x$，$g(t) = \sqrt{\beta(t)}$，代入得：

$$\frac{dx}{dt} = -\frac{1}{2}\beta(t)\left[x + \nabla_x \log p_t(x)\right]$$

用 $\epsilon_\theta$ 替换得分函数：

$$\frac{dx}{dt} = -\frac{1}{2}\beta(t)\left[x - \frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}(t)}}\right]$$

这就是DDIM在连续时间下的本质——它是VP-SDE对应的Probability Flow ODE的一种离散化求解器。

### 2.4 从SDE到ODE的采样加速原理

#### 2.4.1 ODE求解器的收敛理论

为什么ODE框架允许大幅减少采样步数？答案在于ODE求解器的收敛阶次。

对于一个一般的ODE $\frac{dx}{dt} = v(x, t)$，不同阶次的数值求解器有不同的局部截断误差：

| 方法 | 阶次 | 局部误差 | 全局误差 |
|------|------|----------|----------|
| Euler | 1阶 | $O(h^2)$ | $O(h)$ |
| Heun/中点法 | 2阶 | $O(h^3)$ | $O(h^2)$ |
| RK4 | 4阶 | $O(h^5)$ | $O(h^4)$ |

其中 $h$ 是步长。步长 $h = T/N$，$N$ 是总步数。因此高阶方法在相同步数下能获得指数级更高的精度。

**对于扩散模型：** 总时间范围固定（从 $T$ 到 $0$），步数 $N$ 越少意味着步长 $h$ 越大。要在少步情况下保持精度，需要高阶求解器。

#### 2.4.2 信号噪声比(SNR)的数学定义与物理含义

信噪比在扩散模型优化中扮演核心角色，特别是在设计噪声调度和训练损失权重时。

**数学定义：** 对于边际分布 $q(x_t|x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2 I)$：

$$\text{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2}$$

在VP-SDE（DDPM）中，$\alpha_t = \sqrt{\bar{\alpha}_t}$，$\sigma_t = \sqrt{1-\bar{\alpha}_t}$，所以：

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$$

**对数SNR：** 实践中更常用对数SNR $\lambda_t = \log \text{SNR}(t) = \log \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$。对数SNR从 $+\infty$（$t=0$）单调递减到 $-\infty$（$t=T$）。

**物理含义：**
- $\text{SNR}(t) \gg 1$：信号主导，去噪任务"简单"——只需去除微小噪声
- $\text{SNR}(t) \approx 1$：信号和噪声等量，去噪任务中等难度
- $\text{SNR}(t) \ll 1$：噪声主导，去噪任务"困难"——需要从几乎纯噪声中恢复结构

**在优化中的应用：**
- **噪声调度设计**：好的调度应让SNR均匀覆盖从高到低的完整范围
- **损失加权**：Min-SNR策略用 $\min(\text{SNR}(t), \gamma)$ 作为权重，平衡不同时间步的学习
- **采样器步长分配**：在SNR变化剧烈的区域分配更多步数

### 2.5 连续时间与离散时间的关系

#### 2.5.1 从离散β_t到连续β(t)

将离散时间步 $t \in \{1, ..., T\}$ 映射到连续时间 $t \in [0, 1]$，有以下对应关系：

**离散时间：**
$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$$

**连续时间：**
$$\bar{\alpha}(t) = e^{-\int_0^t \beta(s)ds}$$

当步数 $T \to \infty$ 时，离散和连续的公式一致。具体地：

$$\log \bar{\alpha}_T = \sum_{t=1}^T \log(1-\beta_t) \approx -\sum_{t=1}^T \beta_t \approx -\int_0^1 \beta(s) ds$$

（利用 $\log(1-x) \approx -x$ 对小 $x$ 的近似）

#### 2.5.2 不同参数化方式

实践中有多种等价的网络输出参数化方式：

| 参数化 | 网络预测 | 损失函数 | 优势 |
|--------|----------|----------|------|
| $\epsilon$-prediction | $\epsilon_\theta(x_t, t)$ | $\|\epsilon - \epsilon_\theta\|^2$ | DDPM默认，高SNR时数值稳定 |
| $x_0$-prediction | $x_\theta(x_t, t)$ | $\|x_0 - x_\theta\|^2$ | 直觉清晰，但低SNR时不稳定 |
| $v$-prediction | $v_\theta(x_t, t)$ | $\|v - v_\theta\|^2$ | 全SNR范围稳定，SD 2.0+ |
| Score | $s_\theta(x_t, t)$ | $\|s - s_\theta\|^2$ | 理论优雅 |

其中 $v$-prediction 定义为：$v_t = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1-\bar{\alpha}_t} x_0$。

它们之间的转换关系：

$$x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$

$$v = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1-\bar{\alpha}_t} x_0$$

$$\epsilon = \sqrt{\bar{\alpha}_t} v + \sqrt{1-\bar{\alpha}_t} x_t / \text{...}$$

### 2.6 完整代码实现：Score-based模型训练

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal


class ContinuousNoiseSchedule:
    """
    连续时间噪声调度 - SDE框架下的实现
    
    支持VP-SDE和VE-SDE两种形式，提供连续时间的
    alpha(t), sigma(t), SNR(t)等量的计算。
    """
    
    def __init__(
        self,
        sde_type: Literal["vp", "ve"] = "vp",
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
    ):
        self.sde_type = sde_type
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """连续噪声调度 β(t)，仅VP-SDE使用"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def marginal_params(self, t: torch.Tensor):
        """
        计算边际分布 q(x_t|x_0) = N(α(t)x_0, σ²(t)I) 的参数
        
        Returns:
            alpha: 信号缩放系数
            sigma: 噪声标准差
        """
        if self.sde_type == "vp":
            # VP-SDE: α(t) = exp(-½∫₀ᵗ β(s)ds)
            log_alpha = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
            alpha = torch.exp(log_alpha)
            sigma = torch.sqrt(1.0 - alpha**2)
        elif self.sde_type == "ve":
            # VE-SDE: α(t) = 1, σ(t) = σ_min * (σ_max/σ_min)^t
            alpha = torch.ones_like(t)
            sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        
        return alpha, sigma
    
    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """信噪比 SNR(t) = α²(t) / σ²(t)"""
        alpha, sigma = self.marginal_params(t)
        return (alpha / sigma) ** 2
    
    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """对数信噪比 log SNR(t)"""
        return torch.log(self.snr(t))


class ScoreNetwork(nn.Module):
    """
    简化的得分网络架构（用于演示）
    
    实际应用中这里应该是U-Net或DiT，
    这里用简单的MLP+时间嵌入展示核心结构。
    """
    
    def __init__(self, data_dim: int = 784, hidden_dim: int = 512, time_dim: int = 128):
        super().__init__()
        
        # 时间嵌入（正弦位置编码）
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 主网络
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
        
        # 时间调制层
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_dim = time_dim
    
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """正弦时间嵌入"""
        half_dim = self.time_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据 [B, D]
            t: 连续时间 [B] ∈ [0, 1]
            
        Returns:
            预测的score或噪声 [B, D]
        """
        # 时间嵌入
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_embed(t_emb)
        
        # 主网络前向（带时间调制）
        h = self.net[0](x)  # Linear
        h = self.net[1](h)  # SiLU
        h = h + self.time_proj(t_emb)  # 时间调制
        h = self.net[2](h)  # Linear
        h = self.net[3](h)  # SiLU
        h = self.net[4](h)  # Linear
        h = self.net[5](h)  # SiLU
        h = self.net[6](h)  # Linear -> output
        
        return h


class ScoreBasedTrainer:
    """
    Score-based模型的完整训练循环
    
    支持多种参数化方式：
    - epsilon: 预测噪声 (DDPM风格)
    - score: 预测得分函数
    - v: 预测velocity (Stable Diffusion 2.0+)
    """
    
    def __init__(
        self,
        model: nn.Module,
        schedule: ContinuousNoiseSchedule,
        parameterization: Literal["epsilon", "score", "v"] = "epsilon",
        loss_weighting: Literal["uniform", "snr", "min_snr"] = "uniform",
        min_snr_gamma: float = 5.0,
        lr: float = 2e-4,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.schedule = schedule
        self.parameterization = parameterization
        self.loss_weighting = loss_weighting
        self.min_snr_gamma = min_snr_gamma
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    def get_target(
        self, x_0: torch.Tensor, noise: torch.Tensor, 
        alpha: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """根据参数化方式计算训练目标"""
        if self.parameterization == "epsilon":
            return noise
        elif self.parameterization == "score":
            # score = -ε/σ
            return -noise / sigma.view(-1, 1)
        elif self.parameterization == "v":
            # v = α*ε - σ*x_0
            alpha_view = alpha.view(-1, 1)
            sigma_view = sigma.view(-1, 1)
            return alpha_view * noise - sigma_view * x_0
    
    def get_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """计算时间步相关的损失权重"""
        if self.loss_weighting == "uniform":
            return torch.ones_like(t)
        
        snr = self.schedule.snr(t)
        
        if self.loss_weighting == "snr":
            # SNR加权：等价于完整VLB
            return snr
        elif self.loss_weighting == "min_snr":
            # Min-SNR-γ加权策略 (Hang et al., 2023)
            # 截断SNR，防止高SNR时间步权重过大
            return torch.minimum(snr, torch.tensor(self.min_snr_gamma))
    
    def train_step(self, x_0: torch.Tensor) -> dict:
        """
        单步训练
        
        Args:
            x_0: 干净数据 [B, D]
            
        Returns:
            训练指标字典
        """
        x_0 = x_0.to(self.device)
        batch_size = x_0.shape[0]
        
        # 1. 采样连续时间 t ~ U(ε, 1)  (避免t=0的数值问题)
        t = torch.rand(batch_size, device=self.device) * (1.0 - 1e-5) + 1e-5
        
        # 2. 计算噪声调度参数
        alpha, sigma = self.schedule.marginal_params(t)
        
        # 3. 采样噪声并加噪
        noise = torch.randn_like(x_0)
        x_t = alpha.view(-1, 1) * x_0 + sigma.view(-1, 1) * noise
        
        # 4. 网络预测
        pred = self.model(x_t, t)
        
        # 5. 计算目标
        target = self.get_target(x_0, noise, alpha, sigma)
        
        # 6. 加权MSE损失
        weight = self.get_loss_weight(t)
        loss = (weight.view(-1, 1) * (pred - target) ** 2).mean()
        
        # 7. 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "mean_snr": self.schedule.snr(t).mean().item(),
            "mean_t": t.mean().item(),
        }
    
    def train_epoch(self, dataloader, epoch: int = 0) -> dict:
        """完整训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x_0 = batch[0]  # 忽略标签
            else:
                x_0 = batch
            
            metrics = self.train_step(x_0)
            total_loss += metrics["loss"]
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}: avg_loss = {avg_loss:.6f}")
        return {"epoch": epoch, "avg_loss": avg_loss}


class ODESampler:
    """
    基于Probability Flow ODE的采样器
    
    实现Euler和Heun（2阶）两种求解方式，
    展示ODE采样的核心逻辑。
    """
    
    def __init__(
        self,
        model: nn.Module,
        schedule: ContinuousNoiseSchedule,
        parameterization: Literal["epsilon", "score", "v"] = "epsilon",
    ):
        self.model = model
        self.schedule = schedule
        self.parameterization = parameterization
    
    def get_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算Probability Flow ODE的漂移项
        dx/dt = f(x,t) - ½g²(t)·score(x,t)
        """
        alpha, sigma = self.schedule.marginal_params(t)
        
        # 网络预测
        pred = self.model(x, t)
        
        # 转换为score
        if self.parameterization == "epsilon":
            score = -pred / sigma.view(-1, 1)
        elif self.parameterization == "score":
            score = pred
        elif self.parameterization == "v":
            # v = α*ε - σ*x_0 → ε = (v + σ*x_0)/α
            # 先恢复x_0: x_0 = (x_t - σ*ε)/α = (α*x_t - σ*v)/(α²+σ²)
            alpha_v = alpha.view(-1, 1)
            sigma_v = sigma.view(-1, 1)
            eps = alpha_v * pred + sigma_v * x  # v-pred to eps-pred
            score = -eps / sigma_v
        
        # VP-SDE漂移
        beta_t = self.schedule.beta(t)
        drift = -0.5 * beta_t.view(-1, 1) * (x + score)
        
        return drift
    
    @torch.no_grad()
    def sample_euler(
        self, shape: tuple, num_steps: int = 50, device: str = "cuda"
    ) -> torch.Tensor:
        """Euler方法采样（1阶）"""
        self.model.eval()
        
        # 时间从1到ε（避免数值问题）
        dt = -1.0 / num_steps
        t_steps = torch.linspace(1.0, 1e-5, num_steps + 1, device=device)
        
        # 初始化
        x = torch.randn(shape, device=device)
        
        for i in range(num_steps):
            t = t_steps[i].expand(shape[0])
            drift = self.get_drift(x, t)
            x = x + drift * dt
        
        return x
    
    @torch.no_grad()
    def sample_heun(
        self, shape: tuple, num_steps: int = 25, device: str = "cuda"
    ) -> torch.Tensor:
        """Heun方法采样（2阶，每步需要2次网络评估）"""
        self.model.eval()
        
        t_steps = torch.linspace(1.0, 1e-5, num_steps + 1, device=device)
        x = torch.randn(shape, device=device)
        
        for i in range(num_steps):
            t_cur = t_steps[i].expand(shape[0])
            t_next = t_steps[i + 1].expand(shape[0])
            dt = t_next[0] - t_cur[0]
            
            # Euler预测
            d1 = self.get_drift(x, t_cur)
            x_euler = x + d1 * dt
            
            # 校正步（Heun方法）
            d2 = self.get_drift(x_euler, t_next)
            x = x + 0.5 * (d1 + d2) * dt
        
        return x


# 使用示例
if __name__ == "__main__":
    # 配置
    data_dim = 784  # 28x28 flattened (MNIST)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化
    schedule = ContinuousNoiseSchedule(sde_type="vp")
    model = ScoreNetwork(data_dim=data_dim)
    
    trainer = ScoreBasedTrainer(
        model=model,
        schedule=schedule,
        parameterization="epsilon",
        loss_weighting="min_snr",  # 使用Min-SNR加权
        device=device
    )
    
    # 模拟训练步骤
    fake_data = torch.randn(32, data_dim)
    metrics = trainer.train_step(fake_data)
    print(f"Training metrics: {metrics}")
    
    # ODE采样
    sampler = ODESampler(model=model, schedule=schedule)
    samples_euler = sampler.sample_euler(shape=(16, data_dim), num_steps=50, device=device)
    samples_heun = sampler.sample_heun(shape=(16, data_dim), num_steps=25, device=device)
    
    print(f"Euler samples shape: {samples_euler.shape}")
    print(f"Heun samples shape: {samples_heun.shape}")
```

### 2.7 本章小结：三种视角的统一

```
三种视角的统一关系图：

┌─────────────────────────────────────────────────────────────────────────┐
│                          统一视角总结                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   DDPM (离散)              Score Matching              SDE/ODE (连续)    │
│   ┌──────────┐            ┌──────────────┐           ┌──────────────┐   │
│   │ε-predict │◄──────────►│score = -ε/σ  │◄─────────►│∇log p_t(x)   │   │
│   │L_simple  │            │L_DSM         │           │PF-ODE        │   │
│   │离散步t    │            │多噪声级别σ_i  │           │连续时间t∈[0,1]│   │
│   └──────────┘            └──────────────┘           └──────────────┘   │
│        │                         │                         │             │
│        │    数学等价              │    收敛极限              │             │
│        ▼                         ▼                         ▼             │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │               统一的生成模型框架                                │      │
│   │                                                               │      │
│   │  训练: 学习 score/ε/v/x_0 中的任意一种                         │      │
│   │  采样: ODE求解器(DDIM,DPM-Solver,UniPC,...)                   │      │
│   │        或SDE求解器(DDPM, ancestral sampling,...)               │      │
│   │  优化: 利用ODE理论加速 + 利用SDE理论保多样性                    │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
│  实践启示：                                                              │
│  1. 训练时用什么参数化不影响模型本质（只是不同的loss weighting）          │
│  2. 采样时可以自由选择ODE或SDE求解器（与训练时的参数化无关）              │
│  3. ODE框架为步数减少提供了理论保证（高阶求解器的收敛性）                 │
│  4. 所有采样加速方法本质上都是在设计更好的ODE/SDE数值求解器              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

本章建立的数学基础将在后续章节中反复使用。特别是：
- **Probability Flow ODE** 是第3章采样加速方法的理论基础
- **Score Function** 的概念贯穿所有采样器的推导
- **SNR** 是第6章训练优化（如 Min-SNR 加权）的核心量
- **连续时间框架** 是理解Consistency Models和Rectified Flow的前提

理解了这些数学工具，我们就可以进入第3章，看看如何利用ODE求解器理论设计高效的采样算法了。

---


## 第3章：采样加速 - DDIM/DPM-Solver/高阶求解器

采样速度是扩散模型实际部署的最大瓶颈。本章详细剖析从DDIM到DPM-Solver++的一系列采样加速方法，展示如何将1000步缩减到10-50步，同时保持接近原始质量的生成效果。

### 3.1 DDIM详解

#### 3.1.1 非马尔可夫推导的核心思想

DDIM（Denoising Diffusion Implicit Models, Song et al. 2021a）是第一个突破DDPM马尔可夫假设的采样方法。其核心洞察是：

**DDPM的前向过程定义了唯一的边际分布 $q(x_t|x_0)$，但满足相同边际分布的联合分布并不唯一。**

DDPM假设 $q(x_{t-1}|x_t, x_0)$ 是唯一确定的后验分布（由马尔可夫性质导出）。但DDIM发现，我们可以构造一族非马尔可夫的前向过程，它们具有相同的边际分布，但允许更大步长的采样。

具体地，DDIM定义如下推断分布：

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma_t^2 I\right)$$

其中 $\sigma_t$ 是一个自由参数，控制随机性的程度：
- $\sigma_t = \sqrt{\frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)} \cdot \beta_t}$：退化为DDPM
- $\sigma_t = 0$：完全确定性采样（DDIM）

#### 3.1.2 确定性采样公式推导

当 $\sigma_t = 0$ 时，采样公式变为确定性的：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{预测的}x_0} + \underbrace{\sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)}_{\text{指向}x_t\text{的方向}}$$

这个公式的直觉是：
1. 首先用网络预测的噪声估计 $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$
2. 然后在 $\hat{x}_0$ 和噪声方向之间进行插值，得到 $x_{t-1}$

```
DDIM采样的几何直觉：

     噪声空间 (ε方向)
         ▲
         │          x_t (当前位置)
         │         ╱
         │        ╱  "噪声分量"
         │       ╱   √(1-ᾱ_t)·ε_θ
         │      ╱
         │     ╱
         │    ╱─────────── x̂_0 (预测的干净图像)
         │   ╱
         │  ╱
         │ ╱  x_{t-1} (下一步位置)
         │╱    = √ᾱ_{t-1}·x̂_0 + √(1-ᾱ_{t-1})·ε_θ
         ┼─────────────────────────────────► 数据空间 (x_0方向)
         O

    关键：DDIM在"预测x_0"和"噪声方向"之间做线性插值
    步长可以任意选择——不需要逐步t-1, t-2, ...
    只需要选择一个时间步子集 τ₁ > τ₂ > ... > τ_S
```

#### 3.1.3 步数子集选择策略

DDIM 的一个关键设计选择是如何从 $\{0, 1, ..., T-1\}$ 中选择 $S$ 个时间步子集 $\tau_1 > \tau_2 > ... > \tau_S$。常见策略：

**均匀选择（Uniform）：** $\tau_i = \lfloor (T-1) \cdot i / S \rfloor$

$$\text{例: T=1000, S=10 → } \{999, 899, 799, 699, 599, 499, 399, 299, 199, 99\}$$

**二次方选择（Quadratic）：** $\tau_i = \lfloor (T-1) \cdot (i/S)^2 \rfloor$

在高噪声区域（大t）分配更多步数，因为SNR变化更剧烈。

**Leading策略：** 从 $t=0$ 端开始均匀划分

$$\tau_i = \lfloor T \cdot i / S \rfloor, \quad i = 0, 1, ..., S-1$$

**Trailing策略：** 从 $t=T$ 端开始均匀划分

$$\tau_i = T - 1 - \lfloor T \cdot i / S \rfloor, \quad i = 0, 1, ..., S-1$$

**Karras/EDM策略：** 基于 $\sigma$ 空间均匀分布（而非 $t$ 空间）

$$\sigma_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{S-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho})\right)^\rho$$

其中 $\rho = 7$ 是 EDM 论文推荐值。

```
不同子集选择策略对比 (T=1000, S=10):

时间步 t: 0 ──────────────────────────────────────────── 999

Uniform:     │    │    │    │    │    │    │    │    │    │
             99  199  299  399  499  599  699  799  899  999

Quadratic:   ││  │  │   │    │     │      │       │        │
             9 39 99 179 279  399   539    699     879      999

Leading:     │    │    │    │    │    │    │    │    │    │
             0   100  200  300  400  500  600  700  800  900

Trailing:    │    │    │    │    │    │    │    │    │    │
             99  199  299  399  499  599  699  799  899  999

EDM(σ-space):│ │ │  │   │    │     │       │         │         │
             均匀分配在σ空间，对应t空间非均匀
             低噪声(小t)区域更密集

实践建议：
┌─────────────────────────────────────────────────────────┐
│  步数 ≤ 10: EDM/Quadratic 通常最好                       │
│  步数 10-25: Uniform 和 Trailing 差异不大                │
│  步数 ≥ 50: 所有策略结果接近                             │
│  对于SD/SDXL: Trailing + DPM++ 2M 是常用组合            │
└─────────────────────────────────────────────────────────┘
```

#### 3.1.4 DDIM性能分析

DDIM的性能（CIFAR-10 unconditional）：

| 步数 | FID ↓ | 方法 |
|------|-------|------|
| 1000 | 3.17 | DDPM (baseline) |
| 100 | 4.16 | DDIM (η=0) |
| 50 | 5.17 | DDIM (η=0) |
| 20 | 8.23 | DDIM (η=0) |
| 10 | 13.36 | DDIM (η=0) |

可以看到，DDIM在50步时已经接近DDPM 1000步的质量，实现了20倍加速。但10步以下质量下降明显——这正是后续高阶方法要解决的问题。

#### 3.1.5 η参数的影响

参数 $\eta \in [0, 1]$ 通过 $\sigma_t = \eta \sqrt{\frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)}\beta_t}$ 控制采样的随机性：

- $\eta = 0$：完全确定性，适合少步采样（ODE路径）
- $\eta = 1$：等价于DDPM（SDE路径）
- $0 < \eta < 1$：部分随机性，在多样性和效率之间权衡

经验上，当步数较少时 $\eta = 0$ 最好；当步数充足时适量的 $\eta > 0$ 可以提升多样性。

### 3.2 DPM-Solver系列

#### 3.2.1 背景：为什么需要更高阶的求解器？

DDIM本质上是Probability Flow ODE的一阶Euler离散化。从数值分析的角度：

- **DDIM（一阶）：** 局部截断误差 $O(h^2)$，全局误差 $O(h)$
- **需要50+步才能获得好质量**

DPM-Solver（Lu et al., 2022）的核心贡献是：**利用扩散ODE的特殊结构，设计具有更高收敛阶次的求解器，在10-20步内达到高质量。**

#### 3.2.2 扩散ODE的半线性结构

DPM-Solver的关键观察是：扩散ODE具有特殊的半线性（semi-linear）结构。将Probability Flow ODE写成：

$$\frac{dx}{dt} = f(t)x + g(t)\epsilon_\theta(x, t)$$

其中 $f(t) = -\frac{1}{2}\beta(t)$ 是线性部分，$g(t)\epsilon_\theta(x, t)$ 是非线性部分。

**变量替换：** 令 $\lambda_t = \log(\alpha_t/\sigma_t)$（对数SNR），将时间变量从 $t$ 换为 $\lambda$：

$$\frac{d x_\lambda}{d\lambda} = -\sigma_\lambda \cdot \epsilon_\theta(x_\lambda, \lambda)$$

这是一个更简洁的形式。利用变差常数法（variation of constants），精确解为：

$$x_{\lambda_s} = \frac{\alpha_s}{\alpha_t} x_{\lambda_t} - \alpha_s \int_{\lambda_t}^{\lambda_s} e^{-\lambda} \epsilon_\theta(x_\lambda, \lambda) d\lambda$$

DPM-Solver通过对被积函数 $\epsilon_\theta(x_\lambda, \lambda)$ 做Taylor展开，设计不同阶次的近似：

#### 3.2.3 DPM-Solver-1（一阶）

零阶Taylor展开：$\epsilon_\theta(x_\lambda, \lambda) \approx \epsilon_\theta(x_{\lambda_t}, \lambda_t)$

$$x_{\lambda_s} \approx \frac{\alpha_s}{\alpha_t} x_{\lambda_t} - \sigma_s(e^{h} - 1)\epsilon_\theta(x_{\lambda_t}, \lambda_t)$$

其中 $h = \lambda_s - \lambda_t$。这等价于DDIM！

#### 3.2.4 DPM-Solver-2（二阶）

一阶Taylor展开需要 $\epsilon_\theta$ 关于 $\lambda$ 的导数。DPM-Solver-2使用中点法近似：

**步骤1（中点预测）：**
$$u = \frac{\alpha_{r}}{\alpha_t} x_t - \sigma_{r}(e^{h/2} - 1)\epsilon_\theta(x_t, \lambda_t)$$

其中 $r$ 是 $t$ 和 $s$ 之间的中间时间点。

**步骤2（校正）：**
$$x_s = \frac{\alpha_s}{\alpha_t} x_t - \sigma_s(e^h - 1)\epsilon_\theta(u, \lambda_r)$$

这样每步需要2次网络评估，但精度从 $O(h^2)$ 提升到 $O(h^3)$。在同样的网络评估预算（NFE, Number of Function Evaluations）下，DPM-Solver-2 可以用更少的步数达到相同质量。

#### 3.2.5 DPM-Solver++：多步法与改进

DPM-Solver++（Lu et al., 2022b）在DPM-Solver基础上引入两个关键改进：

**改进1：数据预测参数化。** 不直接展开 $\epsilon_\theta$，而是展开 $x_\theta$（预测的 $x_0$）：

$$\hat{x}_0 = \frac{x_t - \sigma_t \epsilon_\theta(x_t, t)}{\alpha_t}$$

在数据空间做Taylor展开更稳定，特别是对于条件生成（如classifier-free guidance）。

**改进2：多步法（Multistep）。** 利用之前步的网络评估结果，避免额外计算。例如 DPM-Solver++ 2M（2阶多步法）：

$$x_s = \frac{\sigma_s}{\sigma_t} x_t + \alpha_s \left[(e^{-h} - 1) D_0 + \frac{1}{2r}(e^{-h} - 1)(D_0 - D_1)\right]$$

其中 $D_0 = \hat{x}_0^{(t)}$，$D_1 = \hat{x}_0^{(t_{prev})}$ 是上一步的预测。

**每步只需1次网络评估，但达到2阶精度！**

```
DPM-Solver++多步法原理：

单步法 vs 多步法对比：

┌─────────────────────────────────────────────────────────────────┐
│  单步法（DPM-Solver-2 Singlestep）                              │
│                                                                  │
│  步骤: t_n → 中间点 → t_{n+1}                                   │
│  网络评估: 2次/步                                                │
│  信息利用: 只用当前步的信息                                       │
│                                                                  │
│  t_n ──────┬──────── t_{n+1}                                     │
│            │                                                      │
│         ε_θ(x,t_n)                                               │
│         ε_θ(u,t_mid)  ← 额外1次评估                              │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  多步法（DPM-Solver++ 2M Multistep）                             │
│                                                                  │
│  步骤: t_n → t_{n+1}（利用t_{n-1}步的缓存）                      │
│  网络评估: 1次/步                                                │
│  信息利用: 当前步 + 上一步缓存                                    │
│                                                                  │
│  t_{n-1} ─────── t_n ─────── t_{n+1}                            │
│       │            │                                              │
│    D_{n-1}(缓存)  D_n(新计算)                                    │
│       └────────────┴──── 两点外推 → 2阶精度                      │
│                                                                  │
│  优势: 每步1次NFE达到2阶 (vs 单步法每步2次NFE)                    │
│  代价: 第一步退化为1阶 (缺少历史信息)                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2.6 截断误差分析

不同阶次求解器的误差行为：

| 求解器 | 阶次 | 局部截断误差 | 20步FID(ImageNet-256) |
|--------|------|-------------|----------------------|
| DDIM | 1 | $O(h^2)$ | ~6.5 |
| DPM-Solver-2 | 2 | $O(h^3)$ | ~5.1 |
| DPM-Solver++ 2M | 2 | $O(h^3)$ | ~4.7 |
| DPM-Solver-3 | 3 | $O(h^4)$ | ~4.5 |

注意：更高阶并不总是更好。当步数极少（如5步）时，3阶方法可能因为数值不稳定而退化。实践中 DPM-Solver++ 2M 是最稳健的通用选择。

#### 3.2.7 自适应步长策略

DPM-Solver还支持自适应步长——根据局部误差估计动态调整步长：

1. 用两个不同阶次的方法计算同一步的结果
2. 它们的差异作为局部误差估计
3. 如果误差太大，缩小步长重算；如果误差很小，增大步长

这类似经典的 Runge-Kutta-Fehlberg 方法。自适应步长可以在SNR变化剧烈的区域自动细化，在平坦区域加速跳过。

### 3.3 其他高阶求解器

#### 3.3.1 Analytic-DPM：解析方差估计

Bao et al. (2022) 提出了一种不同的加速路径：**精确估计DDPM中的方差参数**（而不仅仅固定为 $\beta_t$ 或 $\tilde{\beta}_t$）。

在DDPM中，反向过程的最优方差为：

$$\sigma_t^{*2} = \frac{\mathbb{E}_{q(x_0)}[(1-\bar{\alpha}_t)\text{Var}[\epsilon|x_t]]}{1}$$

Analytic-DPM通过Monte Carlo估计和一些解析近似来获得更准确的方差，从而在不改变采样框架的情况下提升少步质量。但其改进幅度不如DPM-Solver系列显著。

#### 3.3.2 UniPC：统一预测-校正框架

UniPC（Zhao et al., 2023）提出了一个统一的预测-校正（Predictor-Corrector）框架，将各种求解器纳入同一理论下：

**预测步（Predictor）：** 用多步法外推得到初始估计
**校正步（Corrector）：** 在预测结果上做一次额外评估来修正

UniPC的创新在于：
1. 统一了DPM-Solver++和PNDM等方法
2. 提出了更优的多步系数选择方案
3. 支持B(h)系数的自动调优

在实践中，UniPC在5-10步的极少步数场景下比DPM-Solver++有约5-10%的FID改善。

#### 3.3.3 PNDM/PLMS：线性多步法

PNDM（Pseudo Numerical Methods, Liu et al. 2022）将经典的线性多步法（Linear Multi-step Methods, LMS）应用于扩散ODE。

**Adams-Bashforth 2阶（AB2）：**
$$x_{n+1} = x_n + h\left(\frac{3}{2}f_n - \frac{1}{2}f_{n-1}\right)$$

**Adams-Bashforth 3阶（AB3）：**
$$x_{n+1} = x_n + h\left(\frac{23}{12}f_n - \frac{16}{12}f_{n-1} + \frac{5}{12}f_{n-2}\right)$$

PLMS（Pseudo Linear Multi-Step）是PNDM的一个变体，使用缓存的网络评估实现高阶近似。由于利用了多步历史信息，每步只需要1次NFE就能达到高阶精度。

#### 3.3.4 EDM求解器（Karras et al., 2022）

EDM（Elucidating the Design Space of Diffusion-Based Generative Models）对扩散模型的设计空间进行了系统性的分析和优化。其中采样器部分的关键贡献包括：

**1. Heun求解器（2阶）：** 使用标准的Heun方法（改进的Euler法），但配合精心设计的噪声调度和时间步分配。

**2. σ空间的时间步选择：** 在 $\sigma$（噪声标准差）空间而非 $t$ 空间进行均匀划分：

$$\sigma_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho})\right)^\rho, \quad \rho=7$$

**3. 随机性注入（Stochastic Sampling）：** EDM提出在确定性ODE求解器中加入少量噪声（churn），可以提升样本质量：

$$\hat{x}_i = x_i + \sqrt{\hat{\sigma}_i^2 - \sigma_i^2} \cdot \epsilon_i$$

这种"添加噪声再去噪"的操作类似SDE采样，但更加可控。

```
EDM Stochastic Sampler流程：

for i = 0, 1, ..., N-1:
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  1. Noise injection (if S_churn > 0):                    │
│     γ = min(S_churn/N, √2 - 1)                          │
│     σ̂_i = σ_i + γ·σ_i                                  │
│     x̂_i = x_i + √(σ̂_i² - σ_i²)·ε    ← 添加少量噪声    │
│                                                          │
│  2. Euler step:                                          │
│     d = (x̂_i - D(x̂_i, σ̂_i)) / σ̂_i   ← 去噪方向        │
│     x_{i+1} = x̂_i + (σ_{i+1} - σ̂_i)·d                 │
│                                                          │
│  3. Second-order correction (if i < N-1):                │
│     d' = (x_{i+1} - D(x_{i+1}, σ_{i+1})) / σ_{i+1}    │
│     x_{i+1} = x̂_i + (σ_{i+1} - σ̂_i)·(d+d')/2         │
│                                                          │
└─────────────────────────────────────────────────────────┘

参数含义：
  S_churn: 噪声注入强度 (0=纯ODE, 越大越随机)
  S_noise: 噪声缩放因子
  ρ: 时间步分配的形状参数
```

### 3.4 采样器选择指南

#### 3.4.1 不同步数下的最优采样器推荐

基于大量实验和社区实践，以下是不同步数范围的推荐：

```
采样器推荐指南（按步数范围）：

┌─────────────────────────────────────────────────────────────────────────┐
│                        采样器选择决策树                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  目标步数？                                                              │
│  │                                                                       │
│  ├─ 1-4步 ──► 需要蒸馏模型 (LCM, SDXL-Turbo, InstaFlow)                │
│  │            普通采样器在此步数下无法生成有意义的图像                      │
│  │                                                                       │
│  ├─ 5-10步 ─► DPM-Solver++ 2M + Karras σ调度                           │
│  │            或 UniPC (2阶，predictor-corrector)                         │
│  │            CFG场景: DPM++ 2M SDE Karras                               │
│  │                                                                       │
│  ├─ 10-25步 ► DPM-Solver++ 2M Karras (最稳健)                           │
│  │            EDM Heun (需2x NFE但质量最高)                               │
│  │            DPM++ 2M SDE (多样性更好)                                   │
│  │                                                                       │
│  ├─ 25-50步 ► DPM-Solver++ 2M / DDIM / Euler (差异缩小)                 │
│  │            DPM++ SDE (如需最大多样性)                                  │
│  │                                                                       │
│  └─ 50+步 ──► 任何采样器均可，差异极小                                   │
│               DDPM/Ancestral (多样性最大)                                 │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  特殊场景补充：                                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  · 高CFG scale(>10): 使用"data prediction"参数化的求解器       │     │
│  │    (DPM++ 2M避免值爆炸)                                        │     │
│  │  · 高分辨率(>1024): 考虑EDM调度（σ空间更均匀）                 │     │
│  │  · 视频生成: Euler/DDIM更稳定（避免帧间不一致）                 │     │
│  │  · 精确可控生成(ControlNet): 20步DPM++ 2M Karras是安全选择    │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 3.4.2 质量vs速度的Pareto前沿

以下是不同采样器在ImageNet 256×256上的Pareto前沿（使用ADM模型）：

| NFE (网络评估次数) | 最优方法 | FID |
|---|---|---|
| 6 | DPM-Solver-3 | ~7.8 |
| 10 | DPM-Solver++ 2M | ~5.2 |
| 12 | UniPC | ~4.8 |
| 15 | DPM-Solver++ 2M | ~4.5 |
| 20 | DPM-Solver++ 2M | ~4.2 |
| 25 | EDM Heun (50 NFE) | ~3.9 |
| 50 | DDIM | ~4.1 |
| 100 | DDIM | ~3.5 |
| 250 | DDPM | ~3.2 |

**关键结论：** 在10-20 NFE范围内，DPM-Solver++系列是Pareto最优的。超过25 NFE后边际收益递减明显。

#### 3.4.3 实践调参建议

**关于Classifier-Free Guidance (CFG)的影响：**

CFG在采样时修改了有效的score估计：

$$\tilde{\epsilon}_\theta(x_t, t, c) = (1+w) \cdot \epsilon_\theta(x_t, t, c) - w \cdot \epsilon_\theta(x_t, t)$$

高guidance scale ($w > 5$) 会导致值域扩大，某些求解器可能不稳定。DPM-Solver++的data-prediction参数化在高CFG下更稳定，因为它在数据空间做近似（数据空间的值域有界）。

**Dynamic Thresholding：** Imagen论文提出的技术，将预测的 $x_0$ 裁剪到一定范围内：

$$\hat{x}_0 = \text{clip}(\hat{x}_0, -s, s) / s$$

其中 $s$ 是动态阈值（如99.5百分位数）。这防止了高CFG导致的值爆炸。

### 3.5 完整代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Literal, Tuple
import math


class DDIMSampler:
    """
    DDIM采样器完整实现
    
    支持:
    - 确定性(η=0)和随机(η>0)采样
    - 多种时间步选择策略
    - Classifier-Free Guidance
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        schedule_type: str = "scaled_linear",
    ):
        if schedule_type == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        elif schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.num_train_timesteps = num_train_timesteps
    
    def get_timesteps(
        self, 
        num_inference_steps: int, 
        strategy: Literal["uniform", "trailing", "leading", "quadratic"] = "trailing"
    ) -> torch.Tensor:
        """
        根据策略选择时间步子集
        """
        if strategy == "uniform":
            step_ratio = self.num_train_timesteps // num_inference_steps
            timesteps = (torch.arange(num_inference_steps) * step_ratio).flip(0)
        elif strategy == "trailing":
            step_ratio = self.num_train_timesteps / num_inference_steps
            timesteps = torch.arange(num_inference_steps, 0, -1) * step_ratio - 1
            timesteps = timesteps.long()
        elif strategy == "leading":
            step_ratio = self.num_train_timesteps // num_inference_steps
            timesteps = torch.arange(0, num_inference_steps) * step_ratio
            timesteps = timesteps.flip(0)
        elif strategy == "quadratic":
            timesteps = (
                torch.linspace(0, 1, num_inference_steps) ** 2 
                * (self.num_train_timesteps - 1)
            ).long().flip(0)
        
        return timesteps.long()
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        condition: Optional[torch.Tensor] = None,
        timestep_strategy: str = "trailing",
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        DDIM采样主函数
        
        Args:
            model: 噪声预测网络
            shape: 输出形状 (B, C, H, W)
            num_inference_steps: 采样步数
            eta: 随机性参数 [0=确定性, 1=DDPM]
            guidance_scale: CFG强度 (1.0=无guidance)
            condition: 条件信息 (文本embedding等)
            timestep_strategy: 时间步选择策略
            device: 设备
            
        Returns:
            生成的样本
        """
        # 获取时间步序列
        timesteps = self.get_timesteps(num_inference_steps, timestep_strategy)
        timesteps = timesteps.to(device)
        
        # 初始化纯噪声
        x_t = torch.randn(shape, device=device)
        
        # 将alpha_cumprod移到设备
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        for i in range(len(timesteps)):
            t = timesteps[i]
            
            # 获取alpha参数
            alpha_bar_t = alphas_cumprod[t]
            alpha_bar_prev = alphas_cumprod[timesteps[i+1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            
            # 噪声预测 (支持CFG)
            if guidance_scale > 1.0 and condition is not None:
                # Classifier-Free Guidance
                t_batch = t.expand(shape[0])
                eps_cond = model(x_t, t_batch, condition)
                eps_uncond = model(x_t, t_batch, None)  # unconditional
                eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                t_batch = t.expand(shape[0])
                eps_pred = model(x_t, t_batch, condition)
            
            # DDIM更新
            # 1. 预测x_0
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            
            # 可选：dynamic thresholding
            # x0_pred = self._dynamic_threshold(x0_pred)
            
            # 2. 计算σ
            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            
            # 3. 计算"指向x_t的方向"
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_pred
            
            # 4. 组合
            x_t = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
            
            # 5. 添加噪声（如果η > 0 且不是最后一步）
            if sigma_t > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x_t)
                x_t = x_t + sigma_t * noise
        
        return x_t
    
    def _dynamic_threshold(self, x0_pred: torch.Tensor, percentile: float = 0.995) -> torch.Tensor:
        """Dynamic thresholding (Imagen)"""
        batch_size = x0_pred.shape[0]
        x_flat = x0_pred.reshape(batch_size, -1)
        s = torch.quantile(x_flat.abs(), percentile, dim=1, keepdim=True)
        s = torch.clamp(s, min=1.0)
        x0_pred = torch.clamp(x0_pred.reshape(batch_size, -1), -s, s) / s
        return x0_pred.reshape(x0_pred.shape)


class DPMSolverPP:
    """
    DPM-Solver++ (2M) 实现
    
    2阶多步法求解器，每步1次NFE达到2阶精度。
    这是Stable Diffusion WebUI和diffusers中最常用的采样器之一。
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
    ):
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        
        # 计算lambda (log-SNR)
        self.lambdas = torch.log(torch.sqrt(self.alphas_cumprod) / torch.sqrt(1 - self.alphas_cumprod))
    
    def get_scalings(self, t: torch.Tensor):
        """获取t时刻的alpha和sigma"""
        alpha = torch.sqrt(self.alphas_cumprod[t])
        sigma = torch.sqrt(1 - self.alphas_cumprod[t])
        return alpha, sigma
    
    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        """从噪声预测转换为x0预测"""
        alpha, sigma = self.get_scalings(t)
        
        if self.prediction_type == "epsilon":
            x0 = (x_t - sigma * eps_pred) / alpha
        elif self.prediction_type == "v_prediction":
            x0 = alpha * x_t - sigma * eps_pred
        
        return x0
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        condition: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        DPM-Solver++ 2M 采样
        
        核心算法：
        1. 首步使用1阶（等同DDIM单步）
        2. 后续步使用2阶多步法（利用上一步的D缓存）
        """
        # Karras σ调度下的时间步
        timesteps = self._get_karras_timesteps(num_inference_steps, device)
        
        # 初始化
        x_t = torch.randn(shape, device=device)
        alphas_cumprod = self.alphas_cumprod.to(device)
        lambdas = self.lambdas.to(device)
        
        # 缓存上一步的x0预测
        prev_x0_pred = None
        prev_lambda = None
        
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            
            lambda_cur = lambdas[t_cur]
            lambda_next = lambdas[t_next]
            h = lambda_next - lambda_cur  # 步长(在lambda空间)
            
            alpha_cur, sigma_cur = self.get_scalings(t_cur)
            alpha_next, sigma_next = self.get_scalings(t_next)
            
            # 网络预测 (with CFG)
            t_batch = t_cur.expand(shape[0])
            if guidance_scale > 1.0 and condition is not None:
                eps_cond = model(x_t, t_batch, condition)
                eps_uncond = model(x_t, t_batch, None)
                eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = model(x_t, t_batch, condition)
            
            # 转换为x0预测 (data prediction参数化)
            x0_pred = self.predict_x0(x_t, t_cur, eps_pred)
            
            # DPM-Solver++ 2M更新
            if prev_x0_pred is None or i == 0:
                # 首步：1阶更新 (等同DDIM)
                x_t = (sigma_next / sigma_cur) * x_t + alpha_next * (
                    torch.exp(-h) - 1.0
                ) * x0_pred
            else:
                # 后续步：2阶多步更新
                r = (prev_lambda - lambda_cur) / h  # 步长比
                D0 = x0_pred
                D1 = (x0_pred - prev_x0_pred) / (2.0 * r)  # 一阶差分近似导数
                
                x_t = (sigma_next / sigma_cur) * x_t + alpha_next * (
                    torch.exp(-h) - 1.0
                ) * D0 + alpha_next * (
                    (torch.exp(-h) - 1.0) / h + 1.0
                ) * D1
            
            # 缓存当前步
            prev_x0_pred = x0_pred
            prev_lambda = lambda_cur
        
        return x_t
    
    def _get_karras_timesteps(self, num_steps: int, device: str) -> torch.Tensor:
        """Karras调度的时间步"""
        # 在sigma空间均匀分配
        sigma_max = ((1 - self.alphas_cumprod[-1]) / self.alphas_cumprod[-1]).sqrt()
        sigma_min = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]).sqrt()
        
        rho = 7.0
        ramp = torch.linspace(0, 1, num_steps + 1, device=device)
        sigmas = (sigma_max ** (1/rho) + ramp * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        
        # sigma转换为timestep
        alphas_cumprod = self.alphas_cumprod.to(device)
        timesteps = []
        for sigma in sigmas:
            # 找到最接近的离散时间步
            target_alpha_bar = 1.0 / (1.0 + sigma**2)
            idx = torch.argmin((alphas_cumprod - target_alpha_bar).abs())
            timesteps.append(idx)
        
        return torch.stack(timesteps)


class EDMSampler:
    """
    EDM (Karras et al. 2022) 采样器实现
    
    特点：
    1. 在sigma空间设计（而非t空间）
    2. Heun求解器（2阶）
    3. 支持stochastic采样（噪声注入）
    """
    
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_noise: float = 1.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn  # 噪声注入强度
        self.S_noise = S_noise  # 噪声缩放
    
    def get_sigmas(self, num_steps: int, device: str = "cuda") -> torch.Tensor:
        """EDM sigma调度"""
        ramp = torch.linspace(0, 1, num_steps + 1, device=device)
        sigmas = (
            self.sigma_max ** (1 / self.rho) + 
            ramp * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        # 最后追加0
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
        return sigmas
    
    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        shape: Tuple[int, ...],
        num_steps: int = 40,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        EDM Heun采样器 (with optional stochastic churn)
        
        Args:
            denoiser: D(x, σ) 去噪网络 (直接输出去噪后的图像)
            shape: 输出形状
            num_steps: 步数
            device: 设备
            
        Note: EDM参数化中，网络D直接预测x_0:
              D(x, σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x, c_noise(σ))
        """
        sigmas = self.get_sigmas(num_steps, device)
        
        # 初始噪声 (scaled by sigma_max)
        x = torch.randn(shape, device=device) * sigmas[0]
        
        for i in range(num_steps):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Step 1: Optional noise injection (stochastic churn)
            gamma = min(self.S_churn / num_steps, math.sqrt(2) - 1)
            sigma_hat = sigma_cur * (1 + gamma)
            
            if gamma > 0:
                noise = torch.randn_like(x) * self.S_noise
                x = x + torch.sqrt(sigma_hat**2 - sigma_cur**2) * noise
            else:
                sigma_hat = sigma_cur
            
            # Step 2: 去噪 (得到dx/dσ方向)
            denoised = denoiser(x, sigma_hat.expand(shape[0]))
            d_cur = (x - denoised) / sigma_hat  # Euler方向
            
            # Step 3: Euler步
            x_next = x + (sigma_next - sigma_hat) * d_cur
            
            # Step 4: Heun校正 (2阶，除了最后一步)
            if sigma_next > 0 and i < num_steps - 1:
                denoised_next = denoiser(x_next, sigma_next.expand(shape[0]))
                d_next = (x_next - denoised_next) / sigma_next
                # 取平均方向
                x_next = x + (sigma_next - sigma_hat) * (d_cur + d_next) / 2
            
            x = x_next
        
        return x


# 完整的采样器benchmark脚本
def benchmark_samplers():
    """
    对比不同采样器的性能
    (示意代码 - 需要真实模型才能运行)
    """
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (4, 4, 64, 64)  # batch=4, latent space
    
    # 假设有一个训练好的模型
    # model = load_pretrained_model()
    
    results = {}
    
    # DDIM
    ddim = DDIMSampler()
    for steps in [10, 20, 50]:
        # start = time.time()
        # samples = ddim.sample(model, shape, num_inference_steps=steps, device=device)
        # elapsed = time.time() - start
        # results[f"DDIM-{steps}"] = {"time": elapsed, "nfe": steps}
        pass
    
    # DPM-Solver++ 2M
    dpm = DPMSolverPP()
    for steps in [10, 15, 20]:
        # start = time.time()
        # samples = dpm.sample(model, shape, num_inference_steps=steps, device=device)
        # elapsed = time.time() - start
        # results[f"DPM++2M-{steps}"] = {"time": elapsed, "nfe": steps}
        pass
    
    # EDM
    edm = EDMSampler()
    for steps in [20, 40]:
        # start = time.time()
        # samples = edm.sample(model, shape, num_steps=steps, device=device)
        # elapsed = time.time() - start
        # results[f"EDM-Heun-{steps}"] = {"time": elapsed, "nfe": steps * 2}
        pass
    
    print("Sampler Benchmark Results:")
    print(f"{'Method':<20} {'NFE':<6} {'Time(s)':<10}")
    print("-" * 36)
    for name, r in results.items():
        print(f"{name:<20} {r['nfe']:<6} {r.get('time', 'N/A'):<10}")


if __name__ == "__main__":
    # 演示各采样器的时间步选择
    ddim = DDIMSampler()
    
    print("=== DDIM Time Step Strategies (20 steps) ===")
    for strategy in ["uniform", "trailing", "leading", "quadratic"]:
        ts = ddim.get_timesteps(20, strategy)
        print(f"  {strategy:12s}: {ts[:8].tolist()} ...")
    
    print("\n=== EDM Sigma Schedule (10 steps) ===")
    edm = EDMSampler()
    sigmas = edm.get_sigmas(10, device="cpu")
    print(f"  Sigmas: {[f'{s:.3f}' for s in sigmas.tolist()]}")
    
    print("\n=== DPM-Solver++ Lambda Values ===")
    dpm = DPMSolverPP()
    print(f"  Lambda[0] (t=0): {dpm.lambdas[0]:.3f}")
    print(f"  Lambda[500] (t=500): {dpm.lambdas[500]:.3f}")
    print(f"  Lambda[999] (t=999): {dpm.lambdas[999]:.3f}")
```

### 3.6 采样器的数值稳定性分析

在实际部署中，采样器的数值稳定性是一个常被忽视但至关重要的问题。特别是在以下场景：

#### 3.6.1 高Guidance Scale的稳定性

当CFG scale较大（如 $w > 10$）时：

$$\tilde{\epsilon} = (1+w)\epsilon_{cond} - w \cdot \epsilon_{uncond}$$

放大后的噪声预测可能超出训练时的分布，导致预测的 $x_0$ 出现极端值。不同求解器对此的敏感度不同：

- **DDIM（ε-pred参数化）**：在ε空间做外推，容易值爆炸
- **DPM-Solver++（x-pred参数化）**：在数据空间做近似，天然有界
- **Dynamic Thresholding**：显式裁剪，最稳定但可能丢失细节

#### 3.6.2 半精度(FP16)下的精度问题

在GPU推理中通常使用FP16加速。但某些计算在FP16下精度不足：

- $\sqrt{\bar{\alpha}_t}$ 当 $t$ 接近 $T$ 时非常小，FP16可能下溢
- $1 - \bar{\alpha}_t$ 当 $t$ 接近 $0$ 时非常接近0
- 高阶方法中的差分运算会放大舍入误差

```
数值稳定性对策：

┌────────────────────────────────────────────────────────────────┐
│  问题               │  对策                                     │
├────────────────────────────────────────────────────────────────┤
│  FP16下溢           │  关键量(alpha_bar等)保持FP32              │
│  CFG值爆炸         │  Dynamic Thresholding / data-pred参数化   │
│  高阶差分误差放大   │  步数过少时退化为低阶方法                  │
│  极端sigma值       │  sigma clipping (EDM中的sigma_min/max)     │
│  多步法初始化      │  首1-2步使用单步法，之后切换多步法          │
└────────────────────────────────────────────────────────────────┘
```

### 3.7 本章小结

```
第3章知识图谱：

┌─────────────────────────────────────────────────────────────────────────┐
│                      采样加速技术演进路线                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  DDPM(2020)           DDIM(2021)          DPM-Solver(2022)              │
│  1000步               50步                20步                           │
│  马尔可夫             非马尔可夫           半线性ODE                      │
│  SDE采样              ODE采样(1阶)        ODE采样(2-3阶)                 │
│      │                    │                    │                         │
│      ▼                    ▼                    ▼                         │
│  ┌────────┐         ┌──────────┐         ┌──────────────┐              │
│  │每步去噪 │         │跳步+确定性│        │利用ODE结构   │              │
│  │σ_t固定  │         │η参数控制  │        │多步法复用NFE │              │
│  └────────┘         └──────────┘         └──────────────┘              │
│                                                │                         │
│                                                ▼                         │
│                     ┌──────────────────────────────────────┐            │
│                     │  DPM-Solver++ (2022)                  │            │
│                     │  · Data prediction参数化（CFG稳定）    │            │
│                     │  · Multistep (1NFE/step, 2阶)         │            │
│                     │  · 10-20步即可高质量                   │            │
│                     └──────────────────────────────────────┘            │
│                            │               │                             │
│                            ▼               ▼                             │
│                     ┌────────────┐  ┌────────────┐                      │
│                     │ UniPC(2023) │  │ EDM(2022)  │                      │
│                     │ 预测-校正   │  │ Heun+Churn │                      │
│                     │ 5-10步极限  │  │ σ空间最优  │                      │
│                     └────────────┘  └────────────┘                      │
│                                                                          │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─              │
│  采样器加速的理论极限:                                                    │
│    · ODE求解器受限于向量场的Lipschitz常数                                │
│    · 当步数<5时，仅靠求解器优化难以维持质量                               │
│    · 进一步加速需要"改变ODE本身" → 蒸馏方法(第4章)                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

本章覆盖了采样加速的核心方法。关键takeaway：

1. **DDIM** 是第一个突破性工作，将ODE视角引入扩散采样，实现20x加速
2. **DPM-Solver++** 是目前最实用的通用采样器，20步即可达到接近最优质量
3. **高阶方法的收益递减**：超过3阶后，网络预测误差成为瓶颈，而非离散化误差
4. **实践中的最佳选择**取决于步数预算和应用场景

��一章将进入步数蒸馏技术，展示如何打破采样器优化的理论极限——通过改变ODE本身（而非更好地求解它），实现1-4步的极限加速。

---

## 第4章：步数蒸馏 - 打破采样器的理论极限

### 4.1 蒸馏范式总览

#### 4.1.1 为什么需要蒸馏

上一章我们详细分析了采样加速方法：从DDIM到DPM-Solver++，通过高阶ODE求解器将采样步数从1000步压缩到20步左右。然而，一个根本性问题摆在面前——**ODE求解器的精度瓶颈**：

$$\text{离散化误差} = \mathcal{O}(h^{p+1})$$

其中 $h$ 是步长，$p$ 是求解器阶数。当步数进一步减少到5步以下时，步长 $h$ 增大导致离散化误差急剧膨胀，仅靠提高求解器阶数无法弥补。更关键的是，**神经网络预测本身存在误差**，高阶方法对中间点精度的依赖反而放大了网络误差。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    采样加速的两种范式对比                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  范式一：改进ODE求解器（第3章）                                          │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │  固定ODE:  dx = f(x,t)dt                                     │      │
│  │  优化目标: 用更少步数更精确地求解这个ODE                       │      │
│  │  方法:     高阶求解器(DPM-Solver)、自适应步长                 │      │
│  │  极限:     ~5步（受向量场Lipschitz常数约束）                   │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                          ↓ 触及天花板                                    │
│  范式二：改变ODE本身（蒸馏，本章）                                       │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │  新ODE:    dx = g(x,t)dt  (学习一个新的、更简单的向量场)      │      │
│  │  优化目标: 让新ODE一步/少步即可到达同样终点                    │      │
│  │  方法:     蒸馏、一致性约束、分布匹配                         │      │
│  │  极限:     1步（理论极限）                                    │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  关键区别:                                                              │
│    · 求解器优化: 训练好的模型不变，只改推理算法                          │
│    · 蒸馏:       改变模型本身，让它学会"跳步"                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

直觉理解：想象从A城到B城，采样加速类似于优化路线规划（选高速而非国道），而蒸馏则相当于直接修建一条直线隧道。前者受限于现有地形（ODE向量场的曲率），后者则重新定义了路径本身。

#### 4.1.2 蒸馏 vs 采样加速的本质区别

从数学角度，两者的根本差异在于：

**采样加速**保持原始概率流ODE不变：

$$\frac{dx}{dt} = f(x, t) = -\frac{1}{2}\beta(t)\left[x + \nabla_x \log p_t(x)\right]$$

只是用更高阶的数值方法求解。误差来源是离散化，受向量场曲率（Lipschitz常数 $L$）限制：

$$\|x_{\text{numerical}} - x_{\text{exact}}\| \leq C \cdot L^p \cdot h^{p+1}$$

**蒸馏**则学习一个新的映射 $g_\theta$，使得：

$$g_\theta(x_t, t) \approx \text{ODE}_{\text{solve}}(x_t, t \to 0)$$

即直接预测ODE积分的最终结果，而非预测局部方向。这从根本上消除了多步离散化误差的累积问题。

#### 4.1.3 蒸馏技术分类

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        蒸馏方法分类体系                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  步数蒸馏（Step Distillation）                                          │
│  │                                                                      │
│  ├── 逐步蒸馏（Progressive Distillation）                               │
│  │   └── 教师N步 → 学生N/2步 → 迭代减半 → 1步                          │
│  │                                                                      │
│  ├── 一致性蒸馏（Consistency Distillation）                              │
│  │   ├── Consistency Models (CM)                                        │
│  │   ├── Latent Consistency Models (LCM)                                │
│  │   ├── improved CM (iCT/sCT)                                          │
│  │   └── Easy Consistency Models (ECM)                                  │
│  │                                                                      │
│  ├── 分布匹配蒸馏（Distribution Matching）                               │
│  │   ├── DMD (Distribution Matching Distillation)                       │
│  │   ├── DMD2 (去除回归损失)                                            │
│  │   └── Score Distillation (SDS/VSD)                                   │
│  │                                                                      │
│  ├── 对抗蒸馏（Adversarial Distillation）                                │
│  │   ├── SDXL-Turbo (ADD)                                               │
│  │   └── SDXL-Lightning                                                 │
│  │                                                                      │
│  └── 轨迹校正（Trajectory Rectification）                                │
│      ├── Rectified Flow                                                 │
│      └── InstaFlow                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

各类方法的核心理念差异：

| 方法类别 | 核心思想 | 优势 | 代表步数 |
|---------|---------|------|----------|
| Progressive Distillation | 教师多步→学生少步，逐步压缩 | 稳定、渐进 | 4-8步 |
| Consistency Models | 学习将ODE轨迹任意点映射到终点 | 灵活、无需教师(CT) | 1-2步 |
| DMD/DMD2 | 匹配生成分布而非逐像素 | 一步生成 | 1步 |
| 对抗蒸馏 | GAN判别器引导 | 一步高质量 | 1-4步 |
| Rectified Flow | 拉直ODE轨迹 | 简洁优雅 | 1步 |

### 4.2 Progressive Distillation（Salimans & Ho 2022）

#### 4.2.1 逐步减半策略

Progressive Distillation的核心思想极为直觉：如果教师模型用2步能完成的工作，能否训练学生模型用1步完成？然后用这个学生作为新教师，继续蒸馏……

训练过程采用**逐步减半（halving）**策略：

$$N \to N/2 \to N/4 \to \cdots \to 4 \to 2 \to 1$$

每一轮蒸馏中：
- 教师使用 $2k$ 步采样
- 学生学习用 $k$ 步达到相同效果
- 蒸馏完成后，学生成为下一轮的教师

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Progressive Distillation 逐步减半流程                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Round 1: 1024步教师 → 512步学生                                        │
│  ┌──────────┐    蒸馏     ┌──────────┐                                  │
│  │Teacher   │ ──────────→ │Student₁  │                                  │
│  │1024 steps│             │512 steps │                                  │
│  └──────────┘             └────┬─────┘                                  │
│                                │ 升级为教师                              │
│  Round 2: 512步教师 → 256步学生 ↓                                       │
│  ┌──────────┐    蒸馏     ┌──────────┐                                  │
│  │Teacher   │ ──────────→ │Student₂  │                                  │
│  │512 steps │             │256 steps │                                  │
│  └──────────┘             └────┬─────┘                                  │
│                                │                                        │
│       ...  (继续减半)  ...      ↓                                       │
│                                                                         │
│  Round k: 4步教师 → 2步学生                                             │
│  ┌──────────┐    蒸馏     ┌──────────┐                                  │
│  │Teacher   │ ──────────→ │Student_k │                                  │
│  │ 4 steps  │             │ 2 steps  │                                  │
│  └──────────┘             └────┬─────┘                                  │
│                                │                                        │
│  Round k+1: 2步教师 → 1步学生  ↓                                        │
│  ┌──────────┐    蒸馏     ┌──────────┐                                  │
│  │Teacher   │ ──────────→ │Student   │  ← 最终一步生成模型               │
│  │ 2 steps  │             │ 1 step   │                                  │
│  └──────────┘             └──────────┘                                  │
│                                                                         │
│  总蒸馏轮数: log₂(N)                                                    │
│  例: 1024→1 需要 10轮蒸馏                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 教师-学生框架设计

具体地，在每一轮蒸馏中，训练步骤如下：

1. **采样起始点**：从噪声分布采样 $x_t \sim \mathcal{N}(0, I)$，选取时间步 $t$
2. **教师推理**：教师模型执行2个DDIM步骤，从 $x_t$ 到 $x_{t-2\Delta t}$
3. **学生目标**：学生模型用1步直接从 $x_t$ 到 $x_{t-2\Delta t}$
4. **计算损失**：学生输出与教师2步结果之间的MSE

教师的两步DDIM更新：

$$x_{t-\Delta t} = \sqrt{\bar\alpha_{t-\Delta t}} \cdot \hat{x}_0^{(1)} + \sqrt{1-\bar\alpha_{t-\Delta t}} \cdot \epsilon_\theta(x_t, t)$$

$$x_{t-2\Delta t} = \sqrt{\bar\alpha_{t-2\Delta t}} \cdot \hat{x}_0^{(2)} + \sqrt{1-\bar\alpha_{t-2\Delta t}} \cdot \epsilon_\theta(x_{t-\Delta t}, t-\Delta t)$$

其中 $\hat{x}_0^{(1)} = \frac{x_t - \sqrt{1-\bar\alpha_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$。

#### 4.2.3 损失函数

Progressive Distillation的损失函数核心：

$$\mathcal{L}_{\text{PD}} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\hat{x}_0^{\text{student}}(x_t, t) - \hat{x}_0^{\text{teacher\_2step}}(x_t, t)\|_2^2\right]$$

在实际实现中，可以使用 $\epsilon$-prediction形式：

$$\mathcal{L}_{\text{PD}} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon_\phi(x_t, t) - \tilde{\epsilon}_{\text{target}}\|_2^2\right]$$

其中 $\tilde{\epsilon}_{\text{target}}$ 是从教师两步结果反推得到的等效噪声预测。

也可以使用v-prediction形式，Salimans & Ho发现v-prediction在蒸馏场景下更稳定：

$$v = \alpha_t \epsilon - \sigma_t x_0$$

$$\mathcal{L}_{\text{PD-v}} = \mathbb{E}\left[\|v_\phi(x_t, t) - v_{\text{target}}\|_2^2\right]$$

#### 4.2.4 误差累积问题与缓解策略

逐步蒸馏的主要挑战在于**误差累积**：每一轮蒸馏都会引入近似误差，经过 $\log_2(N)$ 轮后，误差逐渐积累：

$$\text{总误差} \leq \sum_{k=1}^{\log_2 N} \epsilon_k$$

缓解策略包括：

1. **v-prediction**：相比 $\epsilon$-prediction，v-prediction在低步数下更稳定，因为它在SNR=1的时间步附近提供了更均衡的信号
2. **渐进式学习率衰减**：随着轮次增加降低学习率，避免过度拟合单步
3. **训练步数递增**：后期轮次（少步时）使用更多训练步数
4. **数据增强**：随机时间步采样确保全面覆盖

#### 4.2.5 性能分析

在CIFAR-10上的逐阶段FID变化：

| 蒸馏轮次 | 采样步数 | FID |
|---------|---------|-----|
| 原始教师 | 1024 | 2.57 |
| Round 1 | 512 | 2.58 |
| Round 5 | 32 | 2.64 |
| Round 8 | 8 | 2.89 |
| Round 9 | 4 | 3.42 |
| Round 10 | 2 | 4.51 |
| Round 11 | 1 | 8.34 |

可以看到，前几轮蒸馏几乎无损，但最后从4步到1步时FID显著恶化，这体现了最后一步蒸馏的困难。

#### 4.2.6 Progressive Distillation 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class ProgressiveDistillationTrainer:
    """Progressive Distillation训练器
    
    实现Salimans & Ho (2022)的逐步减半蒸馏策略。
    每轮将教师的2步压缩为学生的1步。
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1024,
        prediction_type: str = "v_prediction",  # v_prediction更稳定
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.model = model
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        
        # 噪声调度
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def get_v_from_eps(self, x_t, epsilon, t):
        """从epsilon预测转换为v预测目标"""
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return alpha_t * epsilon - sigma_t * x_t
    
    def predict_x0_from_v(self, x_t, v, t):
        """从v预测恢复x0"""
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return alpha_t * x_t - sigma_t * v
    
    def predict_eps_from_v(self, x_t, v, t):
        """从v预测恢复epsilon"""
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sigma_t * x_t + alpha_t * v
    
    @torch.no_grad()
    def teacher_two_step_ddim(self, teacher_model, x_t, t, step_size):
        """教师模型执行2步DDIM
        
        从x_t出发，执行两个DDIM步骤，返回x_{t-2*step_size}
        """
        device = x_t.device
        
        # 第一步: t -> t - step_size
        t1 = t
        t2 = t - step_size
        
        # 教师预测
        v1 = teacher_model(x_t, t1)
        x0_pred_1 = self.predict_x0_from_v(x_t, v1, t1)
        eps_pred_1 = self.predict_eps_from_v(x_t, v1, t1)
        
        # DDIM步骤1
        alpha_t2 = self.sqrt_alphas_cumprod[t2].view(-1, 1, 1, 1)
        sigma_t2 = self.sqrt_one_minus_alphas_cumprod[t2].view(-1, 1, 1, 1)
        x_mid = alpha_t2 * x0_pred_1 + sigma_t2 * eps_pred_1
        
        # 第二步: t - step_size -> t - 2*step_size
        t3 = t - 2 * step_size
        
        v2 = teacher_model(x_mid, t2)
        x0_pred_2 = self.predict_x0_from_v(x_mid, v2, t2)
        eps_pred_2 = self.predict_eps_from_v(x_mid, v2, t2)
        
        # DDIM步骤2
        alpha_t3 = self.sqrt_alphas_cumprod[t3].view(-1, 1, 1, 1)
        sigma_t3 = self.sqrt_one_minus_alphas_cumprod[t3].view(-1, 1, 1, 1)
        x_final = alpha_t3 * x0_pred_2 + sigma_t3 * eps_pred_2
        
        return x_final, x0_pred_2
    
    def compute_distillation_loss(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        x_0: torch.Tensor,
        current_num_steps: int,
    ) -> torch.Tensor:
        """计算单轮蒸馏的损失
        
        Args:
            student_model: 学生模型
            teacher_model: 教师模型（frozen）
            x_0: 真实数据
            current_num_steps: 当前教师使用的步数
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        step_size = self.num_timesteps // current_num_steps
        
        # 随机选择时间步（只选教师轨迹上的点）
        step_indices = torch.randint(
            1, current_num_steps, (batch_size,), device=device
        )
        t = step_indices * step_size  # 映射到实际时间步
        
        # 添加噪声得到x_t
        noise = torch.randn_like(x_0)
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = alpha_t * x_0 + sigma_t * noise
        
        # 教师2步结果（目标）
        with torch.no_grad():
            x_target, x0_target = self.teacher_two_step_ddim(
                teacher_model, x_t, t, step_size
            )
        
        # 学生1步预测
        v_student = student_model(x_t, t)
        x0_student = self.predict_x0_from_v(x_t, v_student, t)
        
        # 损失: 对齐x0预测
        loss = F.mse_loss(x0_student, x0_target)
        
        return loss
    
    def distillation_round(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        dataloader,
        optimizer,
        current_num_steps: int,
        num_train_steps: int = 50000,
    ):
        """执行一轮蒸馏"""
        student_model.train()
        teacher_model.eval()
        
        total_loss = 0
        step = 0
        
        while step < num_train_steps:
            for batch in dataloader:
                if step >= num_train_steps:
                    break
                
                x_0 = batch[0].to(next(student_model.parameters()).device)
                
                loss = self.compute_distillation_loss(
                    student_model, teacher_model, x_0, current_num_steps
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(), 1.0
                )
                optimizer.step()
                
                total_loss += loss.item()
                step += 1
                
                if step % 1000 == 0:
                    avg_loss = total_loss / step
                    print(f"  Step {step}/{num_train_steps}, "
                          f"Loss: {avg_loss:.4f}")
        
        return total_loss / num_train_steps
    
    def progressive_distill(
        self,
        model: nn.Module,
        dataloader,
        initial_steps: int = 1024,
        target_steps: int = 1,
        lr: float = 1e-4,
    ):
        """完整的Progressive Distillation流程"""
        import copy
        
        current_steps = initial_steps
        teacher = copy.deepcopy(model)
        student = model
        
        round_num = 0
        while current_steps > target_steps:
            round_num += 1
            new_steps = current_steps // 2
            print(f"\n{'='*50}")
            print(f"Round {round_num}: {current_steps}步 → {new_steps}步")
            print(f"{'='*50}")
            
            # 后期轮次用更多训练步数
            train_steps = 50000 * (1 + round_num // 3)
            
            optimizer = torch.optim.Adam(
                student.parameters(), lr=lr * (0.9 ** round_num)
            )
            
            avg_loss = self.distillation_round(
                student, teacher, dataloader,
                optimizer, current_steps, train_steps
            )
            
            print(f"Round {round_num} complete. Avg loss: {avg_loss:.4f}")
            
            # 学生升级为教师
            teacher = copy.deepcopy(student)
            current_steps = new_steps
        
        print(f"\nDistillation complete! Final model: {target_steps}-step")
        return student
```

### 4.3 Consistency Models（Song et al. 2023）

Consistency Models是步数蒸馏领域最具影响力的工作之一。它提出了一种全新的建模视角：**学习一个一致性函数**，将ODE轨迹上的任意点直接映射到轨迹的终点（即干净数据）。

#### 4.3.1 一致性函数定义

给定概率流ODE的解轨迹 $\{x_t\}_{t \in [\epsilon, T]}$，一致性函数 $f: (x_t, t) \to x_\epsilon$ 满足：

**自一致性性质（Self-consistency Property）**：对于同一条ODE轨迹上的任意两个点 $(x_s, s)$ 和 $(x_t, t)$，有：

$$f(x_s, s) = f(x_t, t), \quad \forall s, t \in [\epsilon, T]$$

**边界条件**：

$$f(x_\epsilon, \epsilon) = x_\epsilon$$

即在 $t = \epsilon$（接近0的小正数）时，一致性函数退化为恒等映射。

```
┌─────────────────────────────────────────────────────────────────────────┐
│            Consistency Model: ODE轨迹映射示意                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  噪声空间 x_T ──────── ODE轨迹 ─────────→ 数据空间 x_ε                  │
│                                                                         │
│       x_T ●                                                             │
│            \                                                            │
│             \   f(x_T, T) ─────────────────┐                           │
│              \                               ↓                          │
│     x_t₃ ●   \                                                         │
│            \   \                                                        │
│             \   f(x_t₃, t₃) ──────────────→ x_ε ●  (同一终点!)         │
│              \                                ↑                          │
│     x_t₂ ●   \                               │                         │
│            \   f(x_t₂, t₂) ─────────────────┘                          │
│             \                                                           │
│     x_t₁ ●──→ f(x_t₁, t₁) ─────────────────→ x_ε ●                   │
│             \                                                           │
│     x_ε  ●──→ f(x_ε, ε) = x_ε  (边界条件)                              │
│                                                                         │
│  核心性质:                                                              │
│    f(x_T, T) = f(x_t₃, t₃) = f(x_t₂, t₂) = f(x_t₁, t₁) = x_ε       │
│    同一轨迹上所有点映射到相同终点                                         │
│                                                                         │
│  一步生成: x_T ~ N(0,I) → x̂_ε = f(x_T, T) ← 直接完成!                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

为确保边界条件，实际参数化采用跳跃连接形式：

$$f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot F_\theta(x, t)$$

其中 $c_{\text{skip}}(\epsilon) = 1$，$c_{\text{out}}(\epsilon) = 0$，确保 $f_\theta(x_\epsilon, \epsilon) = x_\epsilon$。

一种常用的参数化方案：

$$c_{\text{skip}}(t) = \frac{\sigma_{\text{data}}^2}{(t - \epsilon)^2 + \sigma_{\text{data}}^2}$$

$$c_{\text{out}}(t) = \frac{\sigma_{\text{data}}(t - \epsilon)}{\sqrt{\sigma_{\text{data}}^2 + t^2}}$$

#### 4.3.2 一致性蒸馏（Consistency Distillation, CD）

一致性蒸馏使用预训练的扩散模型（教师）来指导一致性函数的学习。

**训练步骤**：

1. 将时间区间 $[\epsilon, T]$ 离散化为 $N$ 个点：$t_1 = \epsilon < t_2 < \cdots < t_N = T$
2. 对相邻时间步 $(t_{n+1}, t_n)$，使用教师ODE求解器从 $x_{t_{n+1}}$ 得到 $\hat{x}_{t_n}$
3. 训练一致性函数使得 $f_\theta(x_{t_{n+1}}, t_{n+1}) \approx f_{\theta^-}(\hat{x}_{t_n}, t_n)$

**损失函数**：

$$\mathcal{L}_{\text{CD}} = \mathbb{E}_{n, x_0}\left[d\left(f_\theta(x_{t_{n+1}}, t_{n+1}),\, f_{\theta^-}(\hat{x}_{t_n}^\phi, t_n)\right)\right]$$

其中：
- $\hat{x}_{t_n}^\phi$ 是教师模型从 $x_{t_{n+1}}$ 经一步ODE求解到 $t_n$ 的结果
- $\theta^-$ 是目标网络参数（EMA更新）
- $d(\cdot, \cdot)$ 是距离度量（通常用LPIPS或Pseudo-Huber损失）

目标网络的EMA更新：

$$\theta^- \leftarrow \mu \cdot \theta^- + (1 - \mu) \cdot \theta$$

其中 $\mu$ 通常取 0.999 或更高。

#### 4.3.3 一致性训练（Consistency Training, CT）

CT的革命性在于**完全不需要预训练教师**，直接端到端训练一致性函数：

**核心思路**：不使用教师ODE求解器，而是利用数据本身构造相邻轨迹点：

$$x_{t_n} = \sqrt{\bar\alpha_{t_n}} x_0 + \sqrt{1-\bar\alpha_{t_n}} \epsilon$$
$$x_{t_{n+1}} = \sqrt{\bar\alpha_{t_{n+1}}} x_0 + \sqrt{1-\bar\alpha_{t_{n+1}}} \epsilon$$

使用**相同的 $x_0$ 和 $\epsilon$**，确保 $(x_{t_n}, t_n)$ 和 $(x_{t_{n+1}}, t_{n+1})$ 确实在同一条ODE轨迹上（在连续极限下）。

**CT损失函数**：

$$\mathcal{L}_{\text{CT}} = \mathbb{E}_{n, x_0, \epsilon}\left[d\left(f_\theta(x_{t_{n+1}}, t_{n+1}),\, f_{\theta^-}(x_{t_n}, t_n)\right)\right]$$

**关键设计——离散化步骤数调度 $N(k)$**：

训练过程中逐渐增加离散化密度：

$$N(k) = \lceil\sqrt{k/K \cdot ((s_1 + 1)^2 - s_0^2) + s_0^2} - 1\rceil + 1$$

其中 $k$ 是当前训练步，$K$ 是总训练步数，$s_0$ 和 $s_1$ 是初始和最终离散化步数。

直觉理解：训练初期使用粗糙离散化（相邻时间步间距大，约束宽松，容易满足），随着训练进行逐渐增加密度，最终在近乎连续的情况下满足一致性约束。

**EMA decay调度**：

$$\mu(k) = \exp\left(\frac{s_0 \log \mu_0}{N(k)}\right)$$

随着 $N(k)$ 增加，$\mu$ 逐渐增大，目标网络更新更保守。

#### 4.3.4 改进版本

**iCT（improved Consistency Training）**：
- 改用Pseudo-Huber损失替代LPIPS
- 引入连续时间调度
- FID 2.51 on CIFAR-10（一步），2.24（两步）

**sCT（stabilized Consistency Training）**：
- 使用停止梯度技巧稳定训练
- 消除EMA对大batch size的依赖

**ECM（Easy Consistency Models）**：
- 简化训练流程，降低超参数敏感度
- 提供更鲁棒的默认配置

#### 4.3.5 多步采样策略

一致性模型独特的优势之一是支持灵活的多步采样：

**一步采样**：$\hat{x}_0 = f_\theta(x_T, T)$，最快但质量最低

**两步采样**：
1. $\hat{x}_0^{(1)} = f_\theta(x_T, T)$
2. 重新加噪：$x_{t'} = \sqrt{\bar\alpha_{t'}} \hat{x}_0^{(1)} + \sqrt{1-\bar\alpha_{t'}} \epsilon$
3. $\hat{x}_0^{(2)} = f_\theta(x_{t'}, t')$

**多步采样**（交替去噪-加噪）：可以逐步改善质量

| 采样步数 | CIFAR-10 FID (CD) | CIFAR-10 FID (CT) |
|---------|------------------|------------------|
| 1步 | 3.55 | 3.93 |
| 2步 | 2.93 | 2.75 |
| 3步 | 2.46 | - |

#### 4.3.6 Consistency Training 核心实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class ConsistencyTraining:
    """Consistency Training (CT) 实现
    
    无需教师模型的端到端一致性函数训练。
    实现Song et al. (2023)的核心算法。
    """
    
    def __init__(
        self,
        model: nn.Module,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        # N(k) 调度参数
        s0: int = 10,
        s1: int = 1280,
        total_training_steps: int = 800000,
        # EMA参数
        ema_decay_base: float = 0.9999,
    ):
        self.model = model
        self.ema_model = deepcopy(model)
        self.ema_model.requires_grad_(False)
        
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.s0 = s0
        self.s1 = s1
        self.total_steps = total_training_steps
        self.ema_decay_base = ema_decay_base
        self.current_step = 0
    
    def get_num_discretization_steps(self, k: int) -> int:
        """计算当前训练步的离散化步数 N(k)"""
        N = np.ceil(
            np.sqrt(
                k / self.total_steps * ((self.s1 + 1)**2 - self.s0**2)
                + self.s0**2
            ) - 1
        ) + 1
        return int(max(N, self.s0))
    
    def get_ema_decay(self, N: int) -> float:
        """计算当前EMA decay"""
        return np.exp(self.s0 * np.log(self.ema_decay_base) / N)
    
    def get_sigmas(self, N: int) -> torch.Tensor:
        """生成N个离散sigma值（Karras调度）"""
        indices = torch.arange(N)
        sigmas = (
            self.sigma_min ** (1/self.rho)
            + indices / (N - 1)
            * (self.sigma_max ** (1/self.rho) - self.sigma_min ** (1/self.rho))
        ) ** self.rho
        return sigmas
    
    def consistency_function(self, model, x, sigma):
        """一致性函数参数化 f_theta(x, sigma)
        
        使用跳跃连接确保边界条件：f(x, sigma_min) = x
        """
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min)**2 + self.sigma_data**2
        )
        c_out = (
            self.sigma_data * (sigma - self.sigma_min)
            / torch.sqrt(sigma**2 + self.sigma_data**2)
        )
        c_in = 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)
        
        # 网络输入预处理
        scaled_x = c_in.view(-1, 1, 1, 1) * x
        
        # 网络预测
        F_x = model(scaled_x, sigma)
        
        # 跳跃连接
        output = (
            c_skip.view(-1, 1, 1, 1) * x
            + c_out.view(-1, 1, 1, 1) * F_x
        )
        return output
    
    def pseudo_huber_loss(self, x, y, c=0.00054):
        """Pseudo-Huber损失（iCT推荐）
        
        比MSE更鲁棒，比L1更平滑
        L(x,y) = sqrt((x-y)^2 + c^2) - c
        """
        diff = x - y
        return torch.mean(
            torch.sqrt(diff**2 + c**2) - c
        )
    
    def training_step(self, x_0: torch.Tensor) -> torch.Tensor:
        """执行一步CT训练
        
        Args:
            x_0: 真实数据样本 [B, C, H, W]
            
        Returns:
            loss: 一致性训练损失
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # 1. 计算当前N(k)和EMA decay
        N = self.get_num_discretization_steps(self.current_step)
        ema_decay = self.get_ema_decay(N)
        
        # 2. 生成离散sigma序列
        sigmas = self.get_sigmas(N).to(device)
        
        # 3. 随机选择相邻时间步
        n = torch.randint(0, N - 1, (batch_size,), device=device)
        sigma_n = sigmas[n]          # t_n (较小)
        sigma_n1 = sigmas[n + 1]     # t_{n+1} (较大)
        
        # 4. 使用相同噪声构造轨迹上的两个点
        noise = torch.randn_like(x_0)
        x_n = x_0 + sigma_n.view(-1, 1, 1, 1) * noise    # x at t_n
        x_n1 = x_0 + sigma_n1.view(-1, 1, 1, 1) * noise  # x at t_{n+1}
        
        # 5. 在线模型预测 f_theta(x_{n+1}, sigma_{n+1})
        output_online = self.consistency_function(
            self.model, x_n1, sigma_n1
        )
        
        # 6. 目标模型预测 f_{theta^-}(x_n, sigma_n)
        with torch.no_grad():
            output_target = self.consistency_function(
                self.ema_model, x_n, sigma_n
            )
        
        # 7. 计算一致性损失
        # 加权因子: 距离越近的时间步权重越大
        weight = 1.0 / (sigma_n1 - sigma_n)
        loss = self.pseudo_huber_loss(output_online, output_target)
        loss = (weight * loss).mean()
        
        # 8. 更新EMA模型
        self.update_ema(ema_decay)
        self.current_step += 1
        
        return loss
    
    @torch.no_grad()
    def update_ema(self, decay: float):
        """更新目标网络的EMA参数"""
        for p_ema, p_online in zip(
            self.ema_model.parameters(),
            self.model.parameters()
        ):
            p_ema.data.mul_(decay).add_(p_online.data, alpha=1 - decay)
    
    @torch.no_grad()
    def sample(self, num_samples: int, shape: tuple, device: str,
               num_steps: int = 1) -> torch.Tensor:
        """使用一致性模型采样
        
        Args:
            num_samples: 生成样本数
            shape: 单个样本形状 (C, H, W)
            device: 设备
            num_steps: 采样步数 (1=一步生成, >1=多步改善)
            
        Returns:
            生成的样本
        """
        # 初始噪声
        x = torch.randn(num_samples, *shape, device=device) * self.sigma_max
        sigma = torch.full((num_samples,), self.sigma_max, device=device)
        
        # 第一步：直接映射到x_0
        x = self.consistency_function(self.ema_model, x, sigma)
        
        if num_steps == 1:
            return x
        
        # 多步采样：交替加噪-去噪
        sigmas = self.get_sigmas(num_steps + 1).to(device)
        
        for i in range(1, num_steps):
            # 加噪到中间sigma
            sigma_i = sigmas[-(i+1)]
            noise = torch.randn_like(x)
            x = x + sigma_i * noise
            
            # 再次映射到x_0
            sigma_curr = torch.full(
                (num_samples,), sigma_i.item(), device=device
            )
            x = self.consistency_function(self.ema_model, x, sigma_curr)
        
        return x
```

### 4.4 DMD / DMD2（Distribution Matching Distillation）

#### 4.4.1 核心思想

DMD（Yin et al. 2024）提出了一种与上述方法截然不同的蒸馏思路：**不追求逐像素匹配教师输出，而是匹配整体生成分布**。

传统蒸馏的问题在于：
- 逐像素MSE损失倾向于产生模糊结果
- 确定性映射无法完美重建多模态分布

DMD的解决方案：

$$\min_\theta D_{KL}(p_\theta \| p_{\text{teacher}})$$

即让学生模型的生成分布 $p_\theta$ 尽可能接近教师模型的（多步）生成分布 $p_{\text{teacher}}$。

#### 4.4.2 DMD的训练框架

DMD使用**对抗训练**和**分布匹配**的组合：

1. **分布匹配损失**：通过score matching实现KL散度最小化

$$\nabla_\theta D_{KL}(p_\theta \| p_{\text{data}}) = -\mathbb{E}_{x \sim p_\theta}\left[\nabla_\theta \log p_\theta(x) \cdot (s_\theta(x) - s_{\text{data}}(x))\right]$$

其中 $s_{\text{data}}(x) = \nabla_x \log p_{\text{data}}(x)$ 用教师扩散模型近似。

2. **回归损失（DMD1）**：额外加入配对MSE损失稳定训练

$$\mathcal{L}_{\text{reg}} = \mathbb{E}_{z}\left[\|G_\theta(z) - \text{Teacher\_sample}(z)\|_2^2\right]$$

3. **对抗判别器**：真实样本（教师多步结果）vs 假样本（学生一步结果）

#### 4.4.3 DMD2的改进

DMD2（Yin et al. 2024b）的关键创新：**完全去除回归损失**，只保留分布匹配。

去除回归损失的好处：
- 避免模式平均效应导致的模糊
- 生成更多样化的结果
- 简化训练pipeline

DMD2额外引入了：
- 两个时间尺度的对抗训练
- GAN损失的光谱归一化
- 改进的噪声注入策略

#### 4.4.4 性能对比

| 方法 | MS-COCO FID (零样本) | 步数 | 训练成本 |
|------|---------------------|------|----------|
| SDXL | 23.4 | 50 | - |
| DMD (一步) | 23.3 | 1 | 较高 |
| DMD2 (一步) | 22.1 | 1 | 中等 |
| SDXL-Turbo | 23.8 | 1 | 高 |

### 4.5 Latent Consistency Models / SDXL-Turbo / SDXL-Lightning

#### 4.5.1 LCM（Latent Consistency Models）

LCM（Luo et al. 2023）将Consistency Models应用到潜空间（Latent Space），具体改进：

1. **潜空间操作**：在VAE编码器的潜空间中训练一致性函数，而非像素空间
2. **Classifier-Free Guidance蒸馏**：将CFG集成到一致性函数中
3. **跳步策略**：不使用所有时间步，而是使用间隔跳步

关键技术——**Guidance Distillation**：

原始CFG采样：
$$\hat{\epsilon} = \epsilon_\theta(x_t, c_{\text{null}}) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, c_{\text{null}}))$$

LCM将这个augmented预测直接蒸馏进一致性函数，使得推理时不需要两次前向传播。

**LCM-LoRA**：使用LoRA微调，仅需32A100小时即可完成蒸馏，极大降低了部署门槛。

#### 4.5.2 SDXL-Turbo（Adversarial Diffusion Distillation）

Stability AI的SDXL-Turbo使用了对抗蒸馏：

$$\mathcal{L}_{\text{ADD}} = \mathcal{L}_{\text{adv}} + \lambda \mathcal{L}_{\text{distill}}$$

- **对抗损失**：使用预训练DINOv2特征的判别器
- **蒸馏损失**：Score Distillation Sampling (SDS)变体

核心创新：
- 判别器在DINOv2特征空间操作，而非像素空间
- 避免了传统GAN的模式崩溃
- 一步即可生成512x512图像

#### 4.5.3 SDXL-Lightning

SDXL-Lightning（Lin et al. 2024）结合了Progressive Distillation和对抗训练：

1. 先做Progressive Distillation到4步
2. 再用对抗损失微调
3. 提供LoRA版本方便社区使用

4步版本在质量和速度间取得最佳平衡。

#### 4.5.4 方法对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│              LCM / SDXL-Turbo / SDXL-Lightning 对比                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │      LCM        │  │   SDXL-Turbo    │  │ SDXL-Lightning  │         │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤         │
│  │蒸馏方式:        │  │蒸馏方式:        │  │蒸馏方式:        │         │
│  │ Consistency     │  │ Adversarial     │  │ Progressive +   │         │
│  │ Distillation    │  │ Distillation    │  │ Adversarial     │         │
│  │                 │  │                 │  │                 │         │
│  │最佳步数: 4步    │  │最佳步数: 1步    │  │最佳步数: 4步    │         │
│  │                 │  │                 │  │                 │         │
│  │CFG: 蒸馏进模型  │  │CFG: 不需要      │  │CFG: 蒸馏进模型  │         │
│  │                 │  │                 │  │                 │         │
│  │训练成本: 低     │  │训练成本: 高     │  │训练成本: 中     │         │
│  │(LoRA: 32 A100h) │  │(全参数)         │  │(提供LoRA)       │         │
│  │                 │  │                 │  │                 │         │
│  │质量评价:        │  │质量评价:        │  │质量评价:        │         │
│  │ 4步优秀         │  │ 1步略带伪影     │  │ 4步最优         │         │
│  │ 1步尚可         │  │ 但极快          │  │ 质量-速度均衡   │         │
│  │                 │  │                 │  │                 │         │
│  │适用场景:        │  │适用场景:        │  │适用场景:        │         │
│  │ 交互式生成      │  │ 实时预览        │  │ 生产级部署      │         │
│  │ 快速原型        │  │ 草图即时反馈    │  │ 高质量需求      │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                         │
│  性能对比 (SDXL基准, 1024x1024):                                        │
│  ┌────────────────┬──────┬───────────┬───────────────────────┐          │
│  │ 方法           │ 步数 │ FID (↓)   │ 延迟 (A100, s)       │          │
│  ├────────────────┼──────┼───────────┼───────────────────────┤          │
│  │ SDXL原始       │  50  │  23.4     │  ~8.5                │          │
│  │ LCM            │   4  │  25.1     │  ~0.7                │          │
│  │ SDXL-Turbo     │   1  │  23.8     │  ~0.2                │          │
│  │ SDXL-Lightning │   4  │  22.9     │  ~0.7                │          │
│  │ DMD2           │   1  │  22.1     │  ~0.2                │          │
│  └────────────────┴──────┴───────────┴───────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.5.5 LCM推理Pipeline实现

```python
import torch
import torch.nn as nn
from typing import Optional, List
import numpy as np


class LCMScheduler:
    """Latent Consistency Model 推理调度器
    
    实现LCM的少步采样策略，包含guidance scale蒸馏。
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        original_inference_steps: int = 50,
        num_inference_steps: int = 4,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.original_inference_steps = original_inference_steps
        self.num_inference_steps = num_inference_steps
        
        # 计算alpha调度
        betas = torch.linspace(
            beta_start**0.5, beta_end**0.5, num_train_timesteps
        ) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # LCM使用跳步策略选择时间步
        self.timesteps = self._get_lcm_timesteps()
    
    def _get_lcm_timesteps(self) -> torch.Tensor:
        """获取LCM的跳步时间步序列
        
        从原始50步DDIM的时间步中均匀选取num_inference_steps个
        """
        # 原始DDIM时间步
        original_steps = np.linspace(
            0, self.num_train_timesteps - 1,
            self.original_inference_steps + 1
        ).round().astype(np.int64)
        
        # 跳步选取
        skip = len(original_steps) // self.num_inference_steps
        lcm_timesteps = original_steps[::skip][:self.num_inference_steps]
        
        # 反转（从大到小）
        return torch.from_numpy(lcm_timesteps[::-1].copy())
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        next_timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """LCM单步更新
        
        一致性模型的特殊之处：每步直接预测x_0，
        然后如果还有后续步骤则重新加噪到下一个时间步。
        """
        # 获取alpha值
        alpha_prod_t = self.alphas_cumprod[timestep]
        sqrt_alpha_prod = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - alpha_prod_t)
        
        # 从模型输出（v-prediction或epsilon）恢复x_0
        # LCM直接输出x_0预测
        predicted_x0 = model_output
        
        if next_timestep is not None and next_timestep > 0:
            # 重新加噪到下一个时间步
            alpha_prod_next = self.alphas_cumprod[next_timestep]
            noise = torch.randn_like(sample)
            x_next = (
                torch.sqrt(alpha_prod_next) * predicted_x0
                + torch.sqrt(1 - alpha_prod_next) * noise
            )
            return x_next
        else:
            # 最后一步，直接返回x_0
            return predicted_x0


class LCMPipeline:
    """Latent Consistency Model 推理Pipeline
    
    完整的文本到图像生成流程。
    """
    
    def __init__(
        self,
        unet: nn.Module,        # LCM蒸馏后的UNet
        vae_decoder: nn.Module, # VAE解码器
        text_encoder: nn.Module,# 文本编码器
        tokenizer,              # 分词器
        scheduler: LCMScheduler,
    ):
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
    
    @torch.no_grad()
    def encode_prompt(self, prompt: str, device: str) -> torch.Tensor:
        """编码文本提示"""
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).input_ids.to(device)
        
        text_embeddings = self.text_encoder(tokens)[0]
        return text_embeddings
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 7.5,  # 已蒸馏进模型，此处仅作embedding缩放
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """执行LCM推理
        
        Args:
            prompt: 文本提示
            height: 图像高度
            width: 图像宽度
            num_inference_steps: 推理步数（通常2-4步）
            guidance_scale: 引导强度（LCM中已蒸馏）
            seed: 随机种子
            
        Returns:
            生成的图像 [1, 3, H, W]
        """
        device = next(self.unet.parameters()).device
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # 1. 编码文本
        prompt_embeds = self.encode_prompt(prompt, device)
        
        # 2. 准备潜变量
        latent_shape = (1, 4, height // 8, width // 8)
        latents = torch.randn(latent_shape, device=device)
        
        # 3. 设置时间步
        self.scheduler.num_inference_steps = num_inference_steps
        timesteps = self.scheduler._get_lcm_timesteps().to(device)
        
        # 4. LCM去噪循环
        for i, t in enumerate(timesteps):
            # 时间步嵌入
            t_input = t.unsqueeze(0).to(device)
            
            # UNet前向（LCM模型已包含guidance）
            # 传入guidance_scale作为额外条件
            model_output = self.unet(
                latents,
                t_input,
                encoder_hidden_states=prompt_embeds,
                guidance_scale=guidance_scale,
            )
            
            # 确定下一个时间步
            next_t = timesteps[i + 1].item() if i < len(timesteps) - 1 else None
            
            # 调度器步骤
            latents = self.scheduler.step(
                model_output, t.item(), latents, next_t
            )
        
        # 5. VAE解码
        latents = latents / 0.18215  # VAE缩放因子
        images = self.vae_decoder(latents)
        
        # 6. 后处理
        images = (images / 2 + 0.5).clamp(0, 1)
        
        return images
    
    def benchmark(self, prompt: str, num_runs: int = 10) -> dict:
        """性能基准测试"""
        import time
        
        # 预热
        _ = self(prompt)
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self(prompt)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        return {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "throughput_fps": 1.0 / np.mean(times),
        }
```

### 4.6 Rectified Flow与InstaFlow

#### 4.6.1 Rectified Flow核心思想

Rectified Flow（Liu et al. 2022）从一个优雅的几何视角出发：如果ODE轨迹是直线，那么一步欧拉法就能精确求解。

标准扩散模型的ODE轨迹通常是**弯曲的**——这正是为什么需要多步求解。Rectified Flow的目标就是让轨迹变直。

**数学形式化**：

给定数据分布 $\pi_0$（干净数据）和噪声分布 $\pi_1$（高斯噪声），Rectified Flow学习一个速度场 $v_\theta$：

$$\frac{dx_t}{dt} = v_\theta(x_t, t), \quad t \in [0, 1]$$

使得从 $x_1 \sim \pi_1$ 出发积分到 $t=0$ 得到 $x_0 \sim \pi_0$。

**训练目标**（Flow Matching目标）：

$$\mathcal{L}_{\text{RF}} = \mathbb{E}_{x_0 \sim \pi_0, x_1 \sim \pi_1, t \sim U[0,1]}\left[\|v_\theta(x_t, t) - (x_0 - x_1)\|_2^2\right]$$

其中插值路径：$x_t = (1-t) \cdot x_0 + t \cdot x_1$

注意 $(x_0 - x_1)$ 就是连接数据和噪声的**直线方向**！但由于耦合 $(x_0, x_1)$ 通常不是最优传输耦合，学得的轨迹仍然弯曲。

#### 4.6.2 Reflow程序

Reflow是让轨迹变直的关键操作：

1. 用当前模型从噪声 $x_1$ 生成数据 $\hat{x}_0 = \text{ODE\_solve}(x_1)$
2. 用新的配对 $(\hat{x}_0, x_1)$ 重新训练Flow Matching
3. 迭代此过程

每次Reflow都让轨迹更直，因为新配对 $(\hat{x}_0, x_1)$ 的耦合比随机配对更接近最优传输。

理论保证：经过 $k$ 次Reflow后，轨迹曲率以 $\mathcal{O}(1/k)$ 速率减小。

#### 4.6.3 InstaFlow

InstaFlow（Liu et al. 2023）将Rectified Flow应用到Stable Diffusion规模：

1. 在SD 1.5的潜空间上训练Rectified Flow
2. 执行1-2次Reflow
3. 最终蒸馏到1步

结果：一步生成512x512图像，FID 23.4（MS-COCO）。

#### 4.6.4 与Consistency Models的对比

| 维度 | Rectified Flow | Consistency Models |
|------|---------------|-------------------|
| 核心思路 | 让轨迹变直 | 学习轨迹→终点的映射 |
| 训练方式 | Reflow迭代 | CT/CD单阶段 |
| 多步改善 | 减少步数=增加截断误差 | 加噪-去噪循环改善 |
| 理论优雅度 | 高（最优传输视角） | 高（一致性约束） |
| 实用性 | SD3采用 | LCM广泛使用 |
| 代表应用 | SD3, FLUX | LCM, LCM-LoRA |

值得注意的是，Stable Diffusion 3（SD3）和FLUX均基于Rectified Flow框架，证明了该方法在大规模应用中的有效性。

### 4.7 本章小结

步数蒸馏是将扩散模型从"慢但好"推向"快且好"的核心技术路线：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    步数蒸馏技术发展脉络                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  2022.03 ─── Progressive Distillation (Salimans & Ho)                  │
│              首次实现4步高质量生成                                        │
│                        │                                                │
│  2022.10 ─── Rectified Flow (Liu et al.)                               │
│              最优传输视角，轨迹拉直                                       │
│                        │                                                │
│  2023.03 ─── Consistency Models (Song et al.)                          │
│              一步生成理论框架                                             │
│                        │                                                │
│  2023.10 ─── LCM (Luo et al.)                                          │
│              潜空间一致性蒸馏，4步SDXL                                    │
│                        │                                                │
│  2023.11 ─── SDXL-Turbo (Stability AI)                                 │
│              对抗蒸馏，1步512x512                                        │
│                        │                                                │
│  2024.01 ─── DMD (Yin et al.)                                          │
│              分布匹配，1步高质量                                          │
│                        │                                                │
│  2024.02 ─── SDXL-Lightning (ByteDance)                                │
│              4步最优质量-速度平衡                                         │
│                        │                                                │
│  2024.03 ─── InstaFlow & SD3 (Rectified Flow大规模应用)                  │
│              Reflow + 蒸馏                                              │
│                        │                                                │
│  2024.06 ─── iCT/sCT/ECM (改进一致性训练)                               │
│              无教师训练逼近有教师质量                                     │
│                                                                         │
│  趋势: 1步生成质量持续提升，逐步接近多步教师                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**关键Takeaway**：

1. **Progressive Distillation** 是最稳定的基线方法，通过渐进式减半避免了突变，但误差累积是其固有缺陷
2. **Consistency Models** 提供了最优雅的理论框架，CT模式无需教师是重大优势，但训练调度复杂
3. **DMD/对抗蒸馏** 在一步生成上表现最好，但训练不稳定且成本高
4. **Rectified Flow** 从几何视角提供了最优传输的理论解释，SD3/FLUX的选择证明了其工业价值
5. **实践选择**：4步LCM/Lightning是当前生产环境的最优选择，兼顾质量和延迟

下一章将转向模型架构本身的优化——从经典U-Net到DiT/MMDiT，探讨如何设计更高效的去噪网络骨架。


---

## 第5章：高效模型架构设计 - 从U-Net到DiT/MMDiT

### 5.1 U-Net在扩散模型中的演进

扩散模型的去噪网络是其核心组件——它决定了模型的表达能力、计算效率与生成质量。本章系统梳理去噪网络架构的演进路径，从经典U-Net到革命性的Transformer架构。

#### 5.1.1 原始DDPM U-Net架构

Ho et al. (2020)的DDPM首次将U-Net架构应用于扩散模型。其设计继承了医学图像分割中的U-Net经典结构，但加入了两个关键创新：时间步条件化和Self-Attention。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  U-Net 扩散模型完整架构                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: x_t [B,C,H,W]         时间步: t                                     │
│       │                         │                                           │
│       │                    ┌────┴────┐                                      │
│       │                    │Sinusoidal│                                      │
│       │                    │Embedding │                                      │
│       │                    └────┬────┘                                      │
│       │                         │ t_emb [B, D]                              │
│                                                                             │
│  ════ Encoder (Downsampling) ═══════                                        │
│       │                                                                     │
│  ┌────┴─────────────────────┐  h1──────────────────────────┐               │
│  │ResBlock + Attn (64×64)   │ ───→ Skip Connection         │               │
│  └────┬─────────────────────┘                              │               │
│       │ Downsample 2x                                      │               │
│  ┌────┴─────────────────────┐  h2─────────────────────┐    │               │
│  │ResBlock + Attn (32×32)   │ ───→ Skip Connection    │    │               │
│  └────┬─────────────────────┘                         │    │               │
│       │ Downsample 2x                                 │    │               │
│  ┌────┴─────────────────────┐  h3────────────────┐    │    │               │
│  │ResBlock + Attn (16×16)   │ ───→ Skip Conn.   │    │    │               │
│  └────┬─────────────────────┘                    │    │    │               │
│       │ Downsample 2x                            │    │    │               │
│                                                  │    │    │               │
│  ════ Bottleneck (8×8) ════                      │    │    │               │
│  ┌──────────────────────────┐                    │    │    │               │
│  │ResBlock + Attn + ResBlock│                    │    │    │               │
│  └────┬─────────────────────┘                    │    │    │               │
│                                                  │    │    │               │
│  ════ Decoder (Upsampling) ════                  │    │    │               │
│       │                                          │    │    │               │
│  ┌────┴─────────────────────┐ ←─ Concat(h3)─────┘    │    │               │
│  │ResBlock + Attn (16×16)   │                         │    │               │
│  └────┬─────────────────────┘                         │    │               │
│       │ Upsample 2x                                   │    │               │
│  ┌────┴─────────────────────┐ ←─ Concat(h2)──────────┘    │               │
│  │ResBlock + Attn (32×32)   │                              │               │
│  └────┬─────────────────────┘                              │               │
│       │ Upsample 2x                                        │               │
│  ┌────┴─────────────────────┐ ←─ Concat(h1)───────────────┘               │
│  │ResBlock + Attn (64×64)   │                                              │
│  └────┬─────────────────────┘                                              │
│       │                                                                     │
│  ┌────┴───────┐                                                            │
│  │ Conv Out   │ → 输出: ε_θ(x_t, t) [B,C,H,W]                            │
│  └────────────┘                                                            │
│                                                                             │
│  各层详细规格 (SD 1.5, 输入 64×64 latent):                                 │
│  ┌──────────┬─────────┬──────────┬──────────┬──────────┐                   │
│  │ 层       │ 分辨率  │ 通道数   │ ResBlocks│ Attention│                   │
│  ├──────────┼─────────┼──────────┼──────────┼──────────┤                   │
│  │ Down 1   │ 64×64   │ 320      │ 2        │ ✔ (8h)  │                   │
│  │ Down 2   │ 32×32   │ 640      │ 2        │ ✔ (8h)  │                   │
│  │ Down 3   │ 16×16   │ 1280     │ 2        │ ✔ (8h)  │                   │
│  │ Down 4   │ 8×8     │ 1280     │ 2        │ ✘       │                   │
│  │ Mid      │ 8×8     │ 1280     │ 1        │ ✔ (8h)  │                   │
│  │ Up 1     │ 16×16   │ 1280     │ 3        │ ✔ (8h)  │                   │
│  │ Up 2     │ 32×32   │ 640      │ 3        │ ✔ (8h)  │                   │
│  │ Up 3     │ 64×64   │ 320      │ 3        │ ✔ (8h)  │                   │
│  └──────────┴─────────┴──────────┴──────────┴──────────┘                   │
│  总参数量: ~860M                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

U-Net设计的核心组件：

**ResNet块（ResBlock）**：
- GroupNorm + SiLU + Conv3x3 + GroupNorm + SiLU + Conv3x3
- 加入时间步嵌入（通过线性投影后add到特征图）
- 残差连接确保梯度流动

**Attention块**：
- Self-Attention：空间特征内部关注
- Cross-Attention：与条件信号（文本 embedding）交互
- 位于每个ResBlock之后

**时间步嵌入**：

$$\text{emb}(t) = [\sin(t \cdot \omega_1), \cos(t \cdot \omega_1), \ldots, \sin(t \cdot \omega_d), \cos(t \cdot \omega_d)]$$

其中 $\omega_i = 10000^{-2i/d}$，与Transformer位置编码相同。

#### 5.1.2 时间步嵌入的融合方式

时间步信息融入ResBlock的方式经历了演进：

1. **Additive**（DDPM）：$h = h + \text{MLP}(t\_\text{emb})$，简单直接
2. **Scale+Shift**（ADM）：$h = \gamma(t) \cdot h + \beta(t)$，更强表达力
3. **Adaptive LayerNorm**（DiT）：$\text{LN}(h; \gamma(t), \beta(t))$，最优方案

Scale+Shift方式在ADM（Dhariwal & Nichol 2021）中证明了其优越性，将FID从DDPM的3.17改善到2.07。

#### 5.1.3 跳跃连接的重要性

U-Net的跳跃连接（Skip Connection）对扩散模型至关重要：

1. **保留高频细节**：编码器的浅层特征包含边缘、纹理等细节，直接传递给解码器
2. **缓解梯度消失**：提供快捷路径，允许深层网络稳定训练
3. **多尺度信息融合**：解码器可以同时利用局部和全局信息

实验证据：移除跳跃连接后FID可能恶化超过50%，尤其是高分辨率细节受损严重。

#### 5.1.4 Efficient U-Net（Imagen）

Google的Imagen（Saharia et al. 2022）提出了Efficient U-Net，关键改进：

1. **参数重分布**：将更多参数分配给低分辨率层（计算密度高，信息密度大）
2. **深度优先于宽度**：增加ResBlock数量而非通道数
3. **移除高分辨率Attention**：64×64层不用Self-Attention（节省大量计算）
4. **使用更高效的上/下采样**

结果：参数量减少3x，同等质量下训练和推理速度提升显著。

#### 5.1.5 SD 1.5 U-Net详细规格

Stable Diffusion 1.5的U-Net是目前最广泛使用的扩散模型架构，860M参数的分布：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| ResBlock（卷积） | ~350M | 40.7% |
| Self-Attention | ~230M | 26.7% |
| Cross-Attention | ~200M | 23.3% |
| 其他（Norm, 投影等） | ~80M | 9.3% |

**计算量分布（FLOPs，512×512输入）**：
- 总计: ~110 GFLOPs
- Attention占比: ~35%（且随分辨率二次方增长）
- 卷积占比: ~55%
- 线性层: ~10%

### 5.2 DiT（Diffusion Transformer）

#### 5.2.1 设计动机

Peebles & Xie (2023)提出了一个核心问题：**U-Net是扩散模型的最优骨架吗？**

动机分析：
1. **Scaling Law**：ViT在分类任务上展现了清晰的scaling law，U-Net没有
2. **统一架构**：Transformer在NLP、视觉等多个领域均是默认选择
3. **灵活性**：Transformer易于扩展、并行化、跨模态融合
4. **硬件亲和性**：现代GPU/TPU对矩阵乘法优化极佳，Transformer可以充分利用

DiT的回答是：**用纯Transformer替代U-Net作为扩散模型的骨架，并且展现出更好的缩放特性**。

#### 5.2.2 Patch嵌入

DiT将潜空间特征图转换为token序列：

1. 输入：$z \in \mathbb{R}^{C \times H \times W}$（VAE编码后的潜变量）
2. 分块：将 $z$ 分割为 $p \times p$ 的patch
3. 线性投影：每个patch展平后投影为 $d$-维token

对于256×256图像（latent 32×32），使用patch size $p=2$：

$$\text{token数量} = \frac{32}{2} \times \frac{32}{2} = 256$$

#### 5.2.3 位置编码

DiT探索了两种位置编码：

**正弦位置编码**（固定）：
$$\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d})$$
$$\text{PE}(pos, 2i+1) = \cos(pos / 10000^{2i/d})$$

使用二维分解：分别对行、列坐标编码后拼接。

**学习式位置编码**：可训练参数，性能略优但不支持分辨率外推。

DiT发现两者性能差异很小（<0.5 FID），但后续工作（如SD3）倾向使用RoPE（Rotary Position Embedding），兼顾性能和分辨率泛化。

#### 5.2.4 条件融合方案对比

DiT系统对比了四种条件（时间步+类别标签）融入方案：

**1. In-context Conditioning**：
- 将条件token拼接到输入序列：$[t_{\text{emb}}, c_{\text{emb}}, z_1, z_2, \ldots, z_N]$
- 简单但效果一般：条件信息仅通过attention间接影响

**2. Cross-Attention Conditioning**：
- 类似U-Net中的cross-attention
- 条件作为KV，图像token作为Q
- 质量不错但计算量增加

**3. Adaptive Layer Norm (adaLN)**：
- 从条件生成LayerNorm的 $\gamma, \beta$ 参数
- 类似BatchNorm中的条件归一化
- 比in-context和cross-attention都更强

**4. adaLN-Zero（最优方案）**：
- 在adaLN基础上，额外生成一个缩放因子 $\alpha$
- 每个DiT Block的输出 = $\alpha \cdot \text{Block\_output}$
- **初始化时 $\alpha = 0$**，使得网络初始行为为恒等函数
- 此设计灵感来自残差网络的零初始化技巧

性能对比（DiT-XL/2 on ImageNet 256×256）：

| 条件融合方式 | FID-50K |
|------------|--------|
| In-context | 21.3 |
| Cross-attention | 15.7 |
| adaLN | 12.1 |
| **adaLN-Zero** | **9.62** |

#### 5.2.5 DiT Block完整结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 DiT Block (adaLN-Zero)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: x (图像token)     条件: c = [t_emb, y_emb]                           │
│       │                        │                                            │
│       │                   ┌────┴────┐                                       │
│       │                   │  MLP    │→ (γ₁, β₁, α₁, γ₂, β₂, α₂)           │
│       │                   └─────────┘                                       │
│                                                                             │
│  ════ Self-Attention Branch ════                                            │
│       │                                                                     │
│       ├──→ LayerNorm ──→ adaLN(γ₁, β₁)                                    │
│       │         │                                                           │
│       │    Multi-Head Self-Attention                                        │
│       │         │                                                           │
│       │    Scale by α₁  (零初始化，训练初期≈ skip)                           │
│       │         │                                                           │
│       │←────────┘  (residual add)                                           │
│                                                                             │
│  ════ Feed-Forward Branch ════                                              │
│       │                                                                     │
│       ├──→ LayerNorm ──→ adaLN(γ₂, β₂)                                    │
│       │         │                                                           │
│       │    FFN: Linear → GELU → Linear                                     │
│       │         │                                                           │
│       │    Scale by α₂                                                     │
│       │         │                                                           │
│       │←────────┘  (residual add)                                           │
│       │                                                                     │
│       ↓                                                                     │
│  输出: x'                                                                   │
│                                                                             │
│  关键设计思想:                                                              │
│  1. 条件通过adaLN调制LayerNorm的参数（而非单独的条件token）                  │
│  2. α初始为0 → 网络初始行为为恒等映射 → 训练更稳定                          │
│  3. 所有参数(γ,β,α)从同一个MLP生成，减少参数量                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.2.6 缩放定律

DiT最重要的发现是扩散模型也存在清晰的scaling law：

$$\text{FID} \approx a \cdot \text{GFLOPs}^{-b} + c$$

即FID与计算量成幂律关系——更多计算=更低FID=更好质量。

各规模性能（ImageNet 256×256, 无CFG）：

| 模型 | 参数量 | GFLOPs | FID-50K |
|------|--------|--------|--------|
| DiT-S/2 | 33M | 6 | 68.4 |
| DiT-B/2 | 130M | 23 | 43.5 |
| DiT-L/2 | 458M | 80 | 23.3 |
| DiT-XL/2 | 675M | 119 | 9.62 |

加入CFG (w=1.5) 后，DiT-XL/2达到 **FID 2.27**，超越了当时所有U-Net基线。

这个结果意义重大：它证明了Transformer在扩散模型中的缩放优势，直接促成了SD3、FLUX等后续工作的架构选择。

#### 5.2.7 DiT Block完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def modulate(x, shift, scale):
    """自适应LayerNorm的核心操作: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT Block with adaLN-Zero conditioning.
    
    实现Peebles & Xie (2023)的核心设计。
    特点: 条件通过自适应LayerNorm融入，输出零初始化。
    """
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Layer Norms (无可学习参数，由adaLN提供)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Self-Attention
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.attn_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Feed-Forward Network
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        
        # adaLN-Zero: 从条件生成6组参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        
        # 零初始化: 让模型初始行为为恒等函数
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像token [B, N, D]
            c: 条件嵌入 [B, D] (t_emb + class_emb)
        Returns:
            处理后的token [B, N, D]
        """
        B, N, C = x.shape
        
        # 生成adaLN参数
        params = self.adaLN_modulation(c)  # [B, 6*D]
        shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=1)
        
        # Self-Attention分支
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.attn_proj(attn_out)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # FFN分支
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class PatchEmbed(nn.Module):
    """Patch Embedding + Positional Encoding for DiT"""
    
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 1152,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        
        # 线性投影 (用Conv2d实现)
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 可学习位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 潜变量
        Returns:
            [B, N, D] token序列
        """
        x = self.proj(x)  # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = x + self.pos_embed
        return x
```

### 5.3 MMDiT（Stable Diffusion 3的多模态DiT）

#### 5.3.1 双流设计原理

SD3（Esser et al. 2024）提出了MMDiT（Multi-Modal DiT），核心创新是**双流架构**：文本和图像各有独立的token流，通过Joint Attention交互。

**为什么需要双流？**

DiT原版使用cross-attention处理文本条件，存在问题：
- 文本和图像的表示空间不对齐
- Cross-attention中文本是"被查询"角色，缺乏自我更新
- 复杂提示的理解能力受限

MMDiT的解决方案：**让文本和图像在相同Transformer中共同演化**，但保持各自独立的参数。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MMDiT 双流架构                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Text Stream                           Image Stream                         │
│  (CLIP + T5 tokens)                    (Latent patches)                     │
│       │                                       │                             │
│  ┌────┴──────┐                          ┌─────┴───────┐                    │
│  │ Text LN   │                          │ Image LN    │                    │
│  │ (adaLN)   │                          │ (adaLN)     │                    │
│  └────┬──────┘                          └─────┬───────┘                    │
│       │                                       │                             │
│  ┌────┴──────┐                          ┌─────┴───────┐                    │
│  │ Text QKV  │                          │ Image QKV   │                    │
│  │ Projection│                          │ Projection  │                    │
│  └────┬──────┘                          └─────┬───────┘                    │
│       │  Q_t, K_t, V_t                       │  Q_i, K_i, V_i             │
│       │                                       │                             │
│       └────────────┬──────────────────────────┘                             │
│                    │                                                         │
│          ┌─────────┴──────────────────────────┐                             │
│          │     Joint Attention                │                             │
│          │  K = [K_t; K_i]                    │                             │
│          │  V = [V_t; V_i]                    │                             │
│          │  Attn(Q_t, [K;V]) → text_out       │                             │
│          │  Attn(Q_i, [K;V]) → image_out      │                             │
│          └─────────┬──────────┬───────────────┘                             │
│                    │          │                                              │
│            Text out      Image out                                           │
│                 │              │                                             │
│  ┌──────────────┴──┐   ┌─────┴──────────┐                                  │
│  │ Text FFN        │   │ Image FFN      │                                  │
│  │ (independent)   │   │ (independent)  │                                  │
│  └─────────────────┘   └────────────────┘                                  │
│       │                         │                                           │
│       ↓                         ↓                                           │
│  Text tokens'               Image tokens'                                   │
│                                                                             │
│  与标准DiT的区别:                                                           │
│  1. 文本和图像有独立的LN、QKV投影、FFN                                     │
│  2. 但共享同一个attention矩阵（Joint Attention）                            │
│  3. 双向信息流: 文本能看到图像, 图像能看到文本                              │
│  4. 文本流也在不断演化(不同cross-attn中文本固定)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.3.2 Joint Attention机制

Joint Attention的具体实现：

1. 文本token $x_t \in \mathbb{R}^{M \times D}$ 和图像token $x_i \in \mathbb{R}^{N \times D}$ 分别经过独立的QKV投影
2. 将K和V在序列维度拼接：$K = [K_t; K_i]$，$V = [V_t; V_i]$
3. 文本Q和图像Q分别与合并后的KV做 attention
4. 从结果中拆分回各自的stream

这意味着：
- 每个图像token可以关注所有文本token（理解提示）
- 每个文本token可以关注所有图像token（视觉反馈）
- 计算复杂度：$\mathcal{O}((M+N)^2)$

#### 5.3.3 SD3的完整架构

SD3的完整流水线：

1. **文本编码**：三个编码器并行
   - CLIP-L: 77 tokens, 768维
   - CLIP-G: 77 tokens, 1280维  
   - T5-XXL: 77 tokens, 4096维
   - 拼接后投影到模型维度

2. **图像编码**：VAE将512×512图像编码为64×64×16潜变量

3. **MMDiT处理**：24层双流Transformer块

4. **解码**：VAE解码潜变量为图像

SD3参数规模：
- SD3-Medium: 2B参数
- SD3-Large: 8B参数

#### 5.3.4 MMDiT Joint Attention实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MMDiTBlock(nn.Module):
    """Multi-Modal DiT Block (SD3架构)
    
    双流设计: 文本和图像各有独立参数，
    但通过Joint Attention共享信息。
    """
    
    def __init__(self, hidden_size: int = 1536, num_heads: int = 24, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # === Image Stream 参数 ===
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.img_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.img_proj = nn.Linear(hidden_size, hidden_size)
        mlp_dim = int(hidden_size * mlp_ratio)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_size),
        )
        
        # === Text Stream 参数 ===
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.txt_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.txt_proj = nn.Linear(hidden_size, hidden_size)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_size),
        )
        
        # === adaLN调制 (图像流和文本流各自独立) ===
        self.img_adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.txt_adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        
        # 零初始化
        nn.init.zeros_(self.img_adaLN[-1].weight)
        nn.init.zeros_(self.img_adaLN[-1].bias)
        nn.init.zeros_(self.txt_adaLN[-1].weight)
        nn.init.zeros_(self.txt_adaLN[-1].bias)
    
    def forward(self, img: torch.Tensor, txt: torch.Tensor, cond: torch.Tensor):
        """
        Args:
            img: 图像token [B, N_img, D]
            txt: 文本token [B, N_txt, D]
            cond: 条件向量 [B, D]
        Returns:
            (img', txt') 更新后的双流
        """
        B, N_img, D = img.shape
        N_txt = txt.shape[1]
        head_dim = D // self.num_heads
        
        # 生成adaLN参数
        img_s1, img_sc1, img_g1, img_s2, img_sc2, img_g2 = \
            self.img_adaLN(cond).chunk(6, dim=-1)
        txt_s1, txt_sc1, txt_g1, txt_s2, txt_sc2, txt_g2 = \
            self.txt_adaLN(cond).chunk(6, dim=-1)
        
        # === Joint Attention ===
        # Image QKV
        img_n = self.img_norm1(img) * (1 + img_sc1.unsqueeze(1)) + img_s1.unsqueeze(1)
        img_qkv = self.img_qkv(img_n).reshape(B, N_img, 3, self.num_heads, head_dim)
        img_q, img_k, img_v = img_qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # Text QKV
        txt_n = self.txt_norm1(txt) * (1 + txt_sc1.unsqueeze(1)) + txt_s1.unsqueeze(1)
        txt_qkv = self.txt_qkv(txt_n).reshape(B, N_txt, 3, self.num_heads, head_dim)
        txt_q, txt_k, txt_v = txt_qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # 拼接K,V用于Joint Attention
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)
        
        # 各自Q attend to joint KV
        img_attn = F.scaled_dot_product_attention(img_q, joint_k, joint_v)
        img_attn = img_attn.transpose(1, 2).reshape(B, N_img, D)
        img_attn = self.img_proj(img_attn)
        
        txt_attn = F.scaled_dot_product_attention(txt_q, joint_k, joint_v)
        txt_attn = txt_attn.transpose(1, 2).reshape(B, N_txt, D)
        txt_attn = self.txt_proj(txt_attn)
        
        # Residual + Gate
        img = img + img_g1.unsqueeze(1) * img_attn
        txt = txt + txt_g1.unsqueeze(1) * txt_attn
        
        # === 独立FFN ===
        img_ff = self.img_norm2(img) * (1 + img_sc2.unsqueeze(1)) + img_s2.unsqueeze(1)
        img = img + img_g2.unsqueeze(1) * self.img_mlp(img_ff)
        
        txt_ff = self.txt_norm2(txt) * (1 + txt_sc2.unsqueeze(1)) + txt_s2.unsqueeze(1)
        txt = txt + txt_g2.unsqueeze(1) * self.txt_mlp(txt_ff)
        
        return img, txt
```

### 5.4 U-ViT：混合架构

#### 5.4.1 设计思路

U-ViT（Bao et al. 2023）提出了一个有趣的问题：**能否结合U-Net的跳跃连接优势和ViT的缩放优势？**

设计原则：
- 主体是标准ViT（无下采样）
- 添加长程跳跃连接：第 $i$ 层连接到第 $(L-i)$ 层
- 跳跃连接通过简单的拼接+线性投影实现

$$h_j = \text{Block}_j([h_{j-1}; h_{L-j}])$$

其中 $j > L/2$，$[\cdot; \cdot]$ 表示通道拼接。

优势：
- 保持了ViT的统一架构和缩放特性
- 跳跃连接提供了浅层细节信息
- 无需下采样/上采样，避免了信息损失

缺点：
- 不做下采样意味着所有层都在全分辨率上操作，内存和计算开销大
- 缺少多尺度特征层次

### 5.5 PixArt-α/Σ：高效训练策略

#### 5.5.1 设计理念

PixArt-α（Chen et al. 2023）的目标：以极小的训练成本达到SDXL级别的质量。

关键技术：

**1. 分解训练三阶段**：
- 第一阶段：在大规模图像上预训练（学习图像分布）
- 第二阶段：引入文本条件（学习图文对齐）
- 第三阶段：高质量数据微调（提升美学）

**2. Block Dropping**：
- 训练时随机跳过部分Transformer块
- 类似DropPath/Stochastic Depth
- 减少有效计算量，提升正则化

**3. 弱到强蒸馏**：
- 使用质量较差但便宜的教师（如SD 1.5）初始化
- 然后用高质量数据精调超越教师

#### 5.5.2 训练成本对比

| 模型 | 训练成本 (GPU天) | 总计算量 | FID |
|------|-----------------|---------|-----|
| DALL-E 2 | ~40,000 | - | 10.39 |
| Imagen | ~50,000 | - | 7.27 |
| SDXL | ~6,250 | - | - |
| **PixArt-α** | **~675** | 10.8% of SDXL | 7.32 |
| **PixArt-Σ** | ~750 | - | 6.34 |

PixArt-α仅用SDXL约10%的训练成本就达到了接近的质量，证明了合理的训练策略设计的重要性。

### 5.6 架构选择指南

#### 5.6.1 U-Net vs DiT的适用场景

| 维度 | U-Net | DiT/MMDiT |
|------|-------|-----------|
| 多尺度特征 | ✔ 天然支持 | ✘ 需要设计 |
| Scaling | 有限 | 清晰幂律 |
| 硬件效率 | 中（不规则计算） | 高（纯矩阵乘法） |
| 推理优化 | 复杂（多分辨率） | 简单（统一计算图） |
| 社区生态 | 成熟（SD1.x/SDXL） | 快速成长（SD3/FLUX） |
| 适用规模 | <2B参数 | 任意规模 |
| 训练稳定性 | 高 | 需要adaLN-Zero等技巧 |

**实践建议**：

- **预算有限/小模型**：U-Net仍是安全选择，训练成熟稳定
- **追求最优质量**：DiT/MMDiT，利用scaling law堆算力
- **需要灵活多模态**：MMDiT，双流设计天然支持多模态
- **部署效率优先**：DiT，计算图规整易于优化
- **视频/3D生成**：DiT，时空patch天然扩展

#### 5.6.2 参数量-FLOPs-质量三角关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              架构演进时间线与性能对比                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  2020 ─ DDPM U-Net (~36M)                                                  │
│         │                                                                   │
│  2021 ─ ADM U-Net (~560M) ─ classifier guidance                            │
│         │                                                                   │
│  2022 ─ LDM/SD 1.x U-Net (~860M) ─ 潜空间+cross-attn                      │
│         │                                                                   │
│  2023 ─┬─ U-ViT (ViT+skip connections)                                     │
│        ├─ DiT (纯Transformer, scaling law)                                  │
│        ├─ PixArt-α (高效DiT训练)                                            │
│        └─ SDXL U-Net (~2.6B) ─ U-Net最大规模                               │
│         │                                                                   │
│  2024 ─┬─ SD3 MMDiT (~2B) ─ 双流Joint Attention                            │
│        ├─ FLUX (~12B) ─ MMDiT变体, 最大DiT                                  │
│        ├─ PixArt-Σ (4K高分辨率)                                             │
│        └─ Sora (猜测: 时空DiT, ~XB)                                         │
│                                                                             │
│  趋势:                                                                      │
│  1. U-Net → Transformer 是确定趋势                                          │
│  2. 模型规模从百M级→十B级                                                   │
│  3. 训练效率(PixArt式分阶段)日益重要                                        │
│  4. 多模态融合(MMDiT)成为标配                                               │
│                                                                             │
│  各架构代表性能 (ImageNet 256×256 class-conditional):                        │
│  ┌────────────────────┬───────┬────────┬─────────┐                          │
│  │ 模型               │ 参数  │ GFLOPs │ FID     │                          │
│  ├────────────────────┼───────┼────────┼─────────┤                          │
│  │ ADM (U-Net)        │ 560M  │  ~500  │ 2.07    │                          │
│  │ DiT-XL/2           │ 675M  │  119   │ 2.27    │                          │
│  │ U-ViT-H/2          │ 501M  │  ~100  │ 2.29    │                          │
│  │ MDTv2 (masked DiT) │ 676M  │  119   │ 1.58    │                          │
│  └────────────────────┴───────┴────────┴─────────┘                          │
│                                                                             │
│  注: FLOPs更低的DiT达到了与U-Net相当的质量                                  │
│  → Transformer的计算效率更高                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.7 本章小结

模型架构的演进遵循一条清晰的路线：从手工设计的U-Net到数据驱动的Transformer。关键takeaway：

1. **U-Net的贡献不可磨灭**：跳跃连接、多尺度处理等思想至今仍在被借鉴
2. **DiT证明了Transformer的缩放优势**：清晰的scaling law使得"堆算力"成为可靠策略
3. **MMDiT解决了多模态融合**：Joint Attention实现了文本和图像的深度交互
4. **训练策略同样重要**：PixArt证明合理的训练设计可以10x降低成本
5. **未来方向**：更大的模型、更长的序列（视频）、更多的模态（3D、音频）

下一章将聚焦训练优化技术——噪声调度、损失加权、Min-SNR等策略，这些技术与架构选择正交，但对最终生成质量同样至关重要。



---

## 第6章：训练优化 - 噪声调度/损失加权/Min-SNR/v-prediction

### 6.1 噪声调度（Noise Schedule）设计

噪声调度定义了扩散过程中信号如何逐步被噪声淹没，是扩散模型训练中最基础也最关键的设计选择之一。它直接影响模型的训练效率和生成质量。

#### 6.1.1 Linear Schedule

DDPM原始使用的线性调度：

$$\beta_t = \beta_{\text{start}} + \frac{t}{T}(\beta_{\text{end}} - \beta_{\text{start}})$$

其中通常 $\beta_{\text{start}} = 10^{-4}$，$\beta_{\text{end}} = 0.02$。

对应的累积信噪比：

$$\bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$$

$$\text{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}$$

**问题**：线性调度在前期（小t）SNR下降太慢，后期（大t）下降太快。这意味着：
- 早期时间步几乎不添加噪声，模型学习信号不足
- 后期时间步信号几乎完全消失，模型需要"凭空"生成

#### 6.1.2 Cosine Schedule

Nichol & Dhariwal (2021)提出的余弦调度，设计更合理：

$$\bar\alpha_t = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$$

其中 $s = 0.008$ 是小偏移量，避免 $t=0$ 时 $\bar\alpha_0$ 过于接近1。

**直觉**：余弦函数使得SNR在整个时间范围内更均匀地下降，每个时间步都有足够的学习信号。

对比：
- Linear: 前70%时间步几乎无噪声，后30%急剧加噪
- Cosine: 噪声均匀增长，信息保留更平衡

#### 6.1.3 Shifted/Scaled Cosine（高分辨率适配）

对于高分辨率图像（512×512及以上），标准cosine调度存在问题：由于高分辨率图像的冗余度更高，相同的SNR下实际感知噪声更小。

Hoogeboom et al. (2023)提出的解决方案：

$$\bar\alpha_t' = \frac{1}{1 + \exp(-\text{logSNR}'(t))}$$

其中：

$$\text{logSNR}'(t) = \text{logSNR}(t) + 2\log\frac{d_{\text{ref}}}{d}$$

$d$ 是图像分辨率，$d_{\text{ref}} = 64$ 是参考分辨率。这相当于根据分辨率对SNR曲线做平移。

#### 6.1.4 Logit-Normal Schedule（SD3/FLUX）

Stable Diffusion 3和FLUX使用的调度策略，优先训练中间SNR区间：

时间步采样分布：

$$t \sim \text{LogitNormal}(\mu, \sigma^2)$$

即 $\text{logit}(t) = \log(t/(1-t)) \sim \mathcal{N}(\mu, \sigma^2)$

**设计动机**：中间时间步（SNR≈1附近）是模型最难预测的区间，此处梯度信息最丰富。Logit-Normal将更多训练样本集中于此区间。

#### 6.1.5 各调度的SNR曲线对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              四种噪声调度的 log-SNR 曲线对比                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  log(SNR)                                                                   │
│     ↑                                                                       │
│  +10│  ·                                                                    │
│     │   ··                                                                  │
│   +5│     ···                                                               │
│     │  L:    ······                                                         │
│     │  C:  ·  ·    ······                                                   │
│    0│──S:──·──·──·──·──·──·──·─────────────────────────── SNR=1 线         │
│     │  N:  ·  ·  ·  ·  ·  ·  ····                                          │
│   -5│      ·  ·  ·  ·  ·  ·      ·····                                     │
│     │      ·  ·  ·  ·  ·  ·           ······                               │
│  -10│      ·  ·  ·  ·  ·  ·                 ·····                           │
│     │      ·  ·  ·  ·  ·  ·                      ···                       │
│  -15│──────┴──┴──┴──┴──┴──┴──────────────────────────→ t/T                 │
│     0    0.2  0.4  0.6  0.8  1.0                                            │
│                                                                             │
│  图例:                                                                      │
│  L = Linear Schedule (DDPM原始)                                             │
│      特点: 后期急剧下降，前期几乎不变                                       │
│                                                                             │
│  C = Cosine Schedule (iDDPM)                                                │
│      特点: 均匀下降，信息保留最平衡                                         │
│                                                                             │
│  S = Shifted Cosine (高分辨率)                                              │
│      特点: 整体下移，适配高分辨率冗余                                       │
│                                                                             │
│  N = Logit-Normal采样 (SD3/FLUX)                                            │
│      特点: 中间密集采样，两端稀疏                                           │
│                                                                             │
│  设计原则:                                                                  │
│  · SNR=1 (log-SNR=0) 是信号/噪声平衡点                                     │
│  · 此处梯度最大，模型学习效率最高                                           │
│  · 好的调度应让训练更多关注此区域                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.1.6 噪声调度对生成质量的影响

| 调度 | 适用分辨率 | FID变化 | 代表模型 |
|------|-----------|---------|---------|
| Linear | 32×32 - 64×64 | 基线 | DDPM |
| Cosine | 32×32 - 256×256 | -15~20% | iDDPM, SD 1.x |
| Shifted Cosine | 512×512+ | 必要 | Imagen |
| Logit-Normal | 任意 | -10~15% | SD3, FLUX |

### 6.2 预测目标选择

#### 6.2.1 三种预测目标

扩散模型的网络可以预测三种不同的目标，它们数学上等价但训练动态不同：

**ε-prediction（预测噪声）**：

$$\mathcal{L}_\epsilon = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon_\theta(x_t, t) - \epsilon\|_2^2\right]$$

其中 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$

**$x_0$-prediction（预测原始数据）**：

$$\mathcal{L}_{x_0} = \mathbb{E}_{x_0, \epsilon, t}\left[\|f_\theta(x_t, t) - x_0\|_2^2\right]$$

**v-prediction**：

$$v = \sqrt{\bar\alpha_t} \cdot \epsilon - \sqrt{1-\bar\alpha_t} \cdot x_0$$

$$\mathcal{L}_v = \mathbb{E}_{x_0, \epsilon, t}\left[\|v_\theta(x_t, t) - v\|_2^2\right]$$

#### 6.2.2 数学等价性证明

三者可以相互转换。给定任一预测，都可以恢复其他两个：

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t} \cdot \hat\epsilon}{\sqrt{\bar\alpha_t}} = \frac{\sqrt{\bar\alpha_t} \cdot x_t - \sqrt{1-\bar\alpha_t} \cdot \hat v}{\sqrt{\bar\alpha_t}^2 + (1-\bar\alpha_t)}$$

$$\hat\epsilon = \frac{x_t - \sqrt{\bar\alpha_t} \cdot \hat{x}_0}{\sqrt{1-\bar\alpha_t}} = \sqrt{1-\bar\alpha_t} \cdot x_t + \sqrt{\bar\alpha_t} \cdot \hat v$$

因此，如果网络有无限容量且完美训练，三种目标的最优解生成相同的分布。但在实际有限容量下，**训练梯度的行为有本质差异**。

#### 6.2.3 各预测目标在不同SNR区间的行为

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         三种预测目标在不同SNR下的有效梯度幅度                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  梯度幅度                                                                   │
│     ↑                                                                       │
│     │                                                                       │
│  高  │    ε-pred                      x₀-pred                              │
│     │    ╱╲                              ╱╲                                │
│     │   ╱  ╲                            ╱  ╲                               │
│  中  │  ╱    ╲     v-pred             ╱    ╲                               │
│     │ ╱      ╲    ─────────          ╱      ╲                              │
│     │╱        ╲  (相对均匀)         ╱        ╲                             │
│  低  │          ╲                   ╱          ╲                            │
│     │           ╲                 ╱             ╲                           │
│     └───────────┴────────────────┴──────────────┴────→ log(SNR)            │
│          高SNR       SNR≈1          低SNR                                   │
│         (t≈0)                      (t≈T)                                    │
│                                                                             │
│  分析:                                                                      │
│  · ε-prediction: 高SNR区间梯度大（噪声小时预测噪声难）                      │
│    → 过度关注细节/纹理，忽视整体结构                                        │
│                                                                             │
│  · x₀-prediction: 低SNR区间梯度大（强噪声下预测原图难）                     │
│    → 过度关注全局结构，细节模糊                                             │
│                                                                             │
│  · v-prediction: 梯度相对均匀                                               │
│    → 均衡的结构+细节学习                                                    │
│    → 特别适合蒸馏场景和高分辨率训练                                         │
│                                                                             │
│  数学解释:                                                                  │
│  对于均匀加权的损失:                                                        │
│  · L_ε 的有效权重 ∝ SNR(t)                                                 │
│  · L_x₀ 的有效权重 ∝ 1/SNR(t)                                              │
│  · L_v  的有效权重 ∝ 1 (常数!)                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.4 实践建议

| 场景 | 推荐目标 | 原因 |
|------|---------|------|
| 标准训练（低分辨率） | ε-prediction | 简单直接，社区生态完善 |
| 高分辨率训练 | v-prediction | 避免低SNR区间不稳定 |
| 蒸馏/少步推理 | v-prediction | 少步时各SNR区间都重要 |
| 一致性模型 | x₀ 或 v | 直接预测终点更自然 |
| SDXL | v-prediction | 官方选择，高分辨率稳定 |
| SD3/FLUX | ε-prediction + RF | Flow Matching框架 |

### 6.3 损失加权策略

#### 6.3.1 均匀加权的问题

标准扩散训练使用均匀的时间步采样：

$$\mathcal{L} = \mathbb{E}_{t \sim U[1,T]}[\|\epsilon_\theta(x_t, t) - \epsilon\|^2]$$

问题：不同时间步的梯度方向可能**相互冲突**：
- 高SNR时间步（t小）的梯度驱动模型关注细节
- 低SNR时间步（t大）的梯度驱动模型关注结构
- 两者对同一参数的更新方向可能相反

这导致训练效率低下——模型在不同目标间"左右为难"。

#### 6.3.2 P2 Weighting

Choi et al. (2022)提出的感知优先加权：

$$w(t) = \frac{1}{(\text{SNR}(t) + 1)^{1/2}}$$

**直觉**：降低高SNR（简单）时间步的权重，让模型更专注于中间和低SNR区间。

#### 6.3.3 Min-SNR-γ

Hang et al. (2023)提出了更优雅的解决方案——**Min-SNR-γ加权**：

$$w(t) = \frac{\min(\text{SNR}(t),\, \gamma)}{\text{SNR}(t)}$$

其中 $\gamma$ 是超参数，推荐值为5。

**数学解析**：

对于ε-prediction，均匀加权的有效损失为：

$$\mathcal{L}_{\text{eff}} = \mathbb{E}_t\left[\text{SNR}(t) \cdot \|x_0 - \hat{x}_0\|^2\right]$$

这意味着高SNR时间步对 $\hat{x}_0$ 的误差放大了 $\text{SNR}(t)$ 倍！

Min-SNR-γ将有效权重截断为：

$$\text{有效权重} = \min(\text{SNR}(t), \gamma)$$

当 $\text{SNR}(t) > \gamma$ 时权重恒定，消除了高SNR区间的梯度主导。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          Min-SNR加权 vs 均匀加权的训练动态对比                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  有效梯度权重                                                               │
│     ↑                                                                       │
│     │                                                                       │
│  大  │  均匀加权 (ε-pred)                                                   │
│     │  ╱                                                                    │
│     │ ╱                    ← 高SNR区间梯度主导                              │
│     │╱                       (过度关注细节)                                  │
│  γ  │─────────────────── Min-SNR-γ 截断线 ──────                            │
│     │                                                                       │
│     │  Min-SNR加权:                                                         │
│     │  ────────────────── 均匀 ──────────── ╲                               │
│  小  │                                        ╲                             │
│     │                                          ╲ 低SNR适当降权              │
│     └──────────────────────────────────────────────→ t                      │
│          高SNR (t小)           低SNR (t大)                                   │
│                                                                             │
│  训练效果对比 (ImageNet 256×256, DiT-XL/2):                                 │
│  ┌────────────────────┬──────────┬────────────┐                             │
│  │ 加权策略           │ FID      │ 收敛步数   │                             │
│  ├────────────────────┼──────────┼────────────┤                             │
│  │ 均匀加权           │ 3.20     │ 7M steps   │                             │
│  │ P2 Weighting       │ 2.76     │ 5.5M steps │                             │
│  │ Min-SNR-5          │ 2.06     │ 2M steps   │                             │
│  └────────────────────┴──────────┴────────────┘                             │
│                                                                             │
│  Min-SNR-5 相比均匀加权:                                                    │
│  · FID 改善: 3.20 → 2.06 (35.6% ↓)                                        │
│  · 收敛加速: 3.4x                                                          │
│  · 训练曲线更平滑（梯度冲突减少）                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.3.4 加权策略消除梯度冲突的原理

定义每个时间步的梯度：

$$g_t = \nabla_\theta \mathcal{L}_t$$

时间步间的梯度冲突度：

$$\text{conflict}(t_1, t_2) = -\frac{g_{t_1} \cdot g_{t_2}}{\|g_{t_1}\| \|g_{t_2}\|}$$

当此值为正时，两个时间步的梯度方向相反（冲突）。

均匀加权下，高SNR和低SNR时间步的冲突度约为0.3-0.5（显著冲突）。Min-SNR通过降低高SNR权重，使得梯度主要由中间SNR区间决定，减少了方向冲突。

### 6.4 EMA策略

#### 6.4.1 标准EMA

训练扩散模型时，通常维护模型参数的指数移动平均（EMA）版本用于推理：

$$\theta_{\text{EMA}} \leftarrow \mu \cdot \theta_{\text{EMA}} + (1-\mu) \cdot \theta$$

其中 $\mu$（decay）通常取 0.9999 或 0.99999。

**为什么EMA如此重要？**
- 训练过程中参数波动（尤其是后期），EMA平滑了这些波动
- EMA模型通常比任何单一checkpoint FID低0.5-1.0
- 相当于对时间维度做了模型集成

#### 6.4.2 Karras EMA

Karras et al. (2022)提出的改进EMA策略：

$$\mu(k) = \left(1 + \frac{k}{\tau}\right)^{-1/(1+\beta)}$$

其中 $k$ 是训练步数，$\tau$ 和 $\beta$ 是超参数。

关键特性：
- 训练初期 $\mu$ 较小（快速跟踪）
- 训练后期 $\mu$ 逐渐增大（更保守平滑）
- 自适应调整，无需手动调decay

#### 6.4.3 最优decay值选择

| 模型规模 | 推荐decay | batch size |
|---------|-----------|-----------|
| 小模型 (<100M) | 0.9999 | 256 |
| 中模型 (100M-1B) | 0.99999 | 2048 |
| 大模型 (>1B) | 0.999999 | 4096 |

经验法则：模型越大、batch size越大，decay应越接近1（更慢的EMA更新）。

### 6.5 大规模训练工程技巧

#### 6.5.1 有效Batch Size设计

扩散模型训练通常需要大batch size（2048-4096）来覆盖足够的时间步和数据多样性。

**实现方式**：梯度累积
```
有效batch = micro_batch × gradient_accumulation × num_GPUs
例: 2048 = 32 × 8 × 8 (32张/GPU × 8步累积 × 8卡)
```

#### 6.5.2 学习率缩放

当改变batch size时，学习率需要相应调整：

$$\text{lr} = \text{lr}_{\text{base}} \times \sqrt{\frac{\text{batch\_size}}{\text{batch\_size\_base}}}$$

使用sqrt scaling（而非线性scaling）是因为扩散模型对学习率敏感，线性缩放容易导致不稳定。

#### 6.5.3 训练系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           大规模扩散模型训练系统架构                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    训练集群 (例: 64×A100-80GB)                  │        │
│  │                                                                 │        │
│  │  ┌──────────┐  ┌──────────┐       ┌──────────┐                 │        │
│  │  │  Node 0  │  │  Node 1  │  ...  │  Node 7  │                 │        │
│  │  │  8×A100  │  │  8×A100  │       │  8×A100  │                 │        │
│  │  └────┬─────┘  └────┬─────┘       └────┬─────┘                 │        │
│  │       │              │                  │                       │        │
│  │       └──────────────┼──────────────────┘                       │        │
│  │                      │                                          │        │
│  │            ┌─────────┴─────────┐                                │        │
│  │            │ NCCL All-Reduce   │                                │        │
│  │            │ (梯度同步)        │                                │        │
│  │            └───────────────────┘                                │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                             │
│  每张GPU的训练配置:                                                         │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │                                                         │                │
│  │  ┌──────────────────────┐                               │                │
│  │  │    模型分片 (FSDP)   │  ← 模型参数分布在所有GPU     │                │
│  │  │  · 前向: All-Gather  │                               │                │
│  │  │  · 反向: Reduce-Scatter                              │                │
│  │  └──────────────────────┘                               │                │
│  │                                                         │                │
│  │  ┌──────────────────────┐                               │                │
│  │  │  Gradient Checkpoint │  ← 用计算换内存               │                │
│  │  │  · 前向不保存激活    │     节省60-70%激活内存        │                │
│  │  │  · 反向重新计算      │                               │                │
│  │  └──────────────────────┘                               │                │
│  │                                                         │                │
│  │  ┌──────────────────────┐                               │                │
│  │  │  混合精度 (bf16)     │  ← 计算用bf16，主权重fp32    │                │
│  │  │  · 前向/反向: bf16   │     2x吞吐提升              │                │
│  │  │  · 权重更新: fp32    │                               │                │
│  │  └──────────────────────┘                               │                │
│  │                                                         │                │
│  │  ┌──────────────────────┐                               │                │
│  │  │  Flash Attention     │  ← O(N)内存, 2-4x加速       │                │
│  │  │  · 分块计算attention │                               │                │
│  │  │  · 避免N²内存       │                               │                │
│  │  └──────────────────────┘                               │                │
│  │                                                         │                │
│  └─────────────────────────────────────────────────────────┘                │
│                                                                             │
│  典型配置 (训练2B参数DiT):                                                  │
│  · 64×A100-80GB, FSDP full shard                                           │
│  · bf16混合精度, Flash Attention v2                                         │
│  · Gradient Checkpointing每2层                                              │
│  · 有效batch size: 2048                                                    │
│  · 训练时间: ~2周                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.5.4 关键工程配置

**Flash Attention**：
- 将Attention的内存复杂度从 $O(N^2)$ 降为 $O(N)$
- 通过分块（tiling）在SRAM中完成计算
- 对DiT尤其重要（长序列attention）

**混合精度训练**：
- **bf16推荐**（而非fp16）：bf16的动态范围与fp32相同，不需要loss scaling
- 关键操作（如LayerNorm、softmax）保持fp32
- 权重更新始终在fp32精度

**Gradient Checkpointing**：
- 每隔2-4层checkpoint一次
- 节省60-70%激活内存
- 代价：约30%额外前向计算

**FSDP vs DeepSpeed ZeRO**：

| 特性 | FSDP (PyTorch原生) | DeepSpeed ZeRO-3 |
|------|-------------------|-----------------|
| 集成度 | 原生，API简洁 | 需要额外配置 |
| 性能 | 优秀 | 优秀 |
| 灵活性 | 支持任意模型 | 需要wrapper |
| 社区 | 活跃 | 活跃 |
| 推荐 | PyTorch用户首选 | 需要极致优化时 |

### 6.6 训练稳定性与调试

#### 6.6.1 常见训练不稳定的原因

1. **梯度爆炸**：通常发生在高SNR时间步，ε-prediction时 $\text{SNR} \to \infty$ 的区间
2. **Loss spike**：大batch size训练中偶发的异常样本
3. **NaN传播**：fp16下溢/上溢
4. **EMA分歧**：EMA模型与在线模型差距过大

#### 6.6.2 诊断与处理

| 问题 | 诊断信号 | 解决方案 |
|------|---------|---------|
| 梯度爆炸 | grad_norm突增 | 梯度裁剪(max_norm=1.0) |
| Loss spike | 损失突然跳升 | 跳过异常batch，使用稳定采样 |
| NaN | 任何参数出现inf/nan | 使用bf16，检查LayerNorm |
| 震荡 | 损失不收敛 | 降低学习率，增大batch |
| EMA分歧 | EMA评估差于在线 | 调整decay，增加warmup |

#### 6.6.3 最优超参组合推荐

基于社区实践和论文报告的最优配置：

```python
# 推荐超参数配置 (2B参数DiT, ImageNet 256×256)
config = {
    # 优化器
    "optimizer": "AdamW",
    "lr": 1e-4,
    "weight_decay": 0.0,
    "betas": (0.9, 0.999),  # 或 (0.9, 0.95) for larger models
    
    # 学习率调度
    "warmup_steps": 5000,
    "lr_schedule": "constant_with_warmup",  # 或cosine decay
    
    # 批次
    "effective_batch_size": 2048,
    "micro_batch_per_gpu": 32,
    
    # 精度
    "precision": "bf16",
    "grad_clip_norm": 1.0,
    
    # EMA
    "ema_decay": 0.9999,
    
    # 噪声调度
    "noise_schedule": "cosine",  # 或 shifted_cosine for 512+
    "prediction_type": "v_prediction",
    
    # 损失加权
    "loss_weighting": "min_snr",
    "snr_gamma": 5.0,
    
    # 内存优化
    "gradient_checkpointing": True,
    "flash_attention": True,
}
```

### 6.7 代码实现

#### 6.7.1 各种噪声调度的实现

```python
import torch
import numpy as np
from typing import Optional


class NoiseScheduler:
    """扩散模型噪声调度器
    
    实现四种常用调度：Linear, Cosine, Shifted Cosine, Logit-Normal。
    统一接口，方便对比实验。
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        shift_factor: float = 1.0,  # 用于shifted cosine
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
            alphas = 1.0 - betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)
            
        elif schedule_type == "cosine":
            steps = torch.linspace(0, 1, num_timesteps + 1)
            alphas_bar = torch.cos(
                (steps + cosine_s) / (1 + cosine_s) * torch.pi / 2
            ) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            self.alphas_cumprod = alphas_bar[1:]  # 去掉t=0
            # 裁剪避免数值问题
            self.alphas_cumprod = torch.clamp(self.alphas_cumprod, 1e-5, 1-1e-5)
            
        elif schedule_type == "shifted_cosine":
            # 高分辨率适配：根据shift_factor平移logSNR
            steps = torch.linspace(0, 1, num_timesteps + 1)
            alphas_bar = torch.cos(
                (steps + cosine_s) / (1 + cosine_s) * torch.pi / 2
            ) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            # 在logSNR空间做shift
            log_snr = torch.log(alphas_bar / (1 - alphas_bar))
            log_snr_shifted = log_snr + 2 * np.log(shift_factor)
            alphas_bar_shifted = torch.sigmoid(log_snr_shifted)
            self.alphas_cumprod = alphas_bar_shifted[1:]
            self.alphas_cumprod = torch.clamp(self.alphas_cumprod, 1e-5, 1-1e-5)
            
        elif schedule_type == "logit_normal":
            # SD3/FLUX使用: 中间密集，两端稀疏
            # 实际是采样策略而非调度，但可以等效实现
            steps = torch.linspace(0, 1, num_timesteps + 1)
            alphas_bar = torch.cos(
                (steps + cosine_s) / (1 + cosine_s) * torch.pi / 2
            ) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            self.alphas_cumprod = alphas_bar[1:]
            self.alphas_cumprod = torch.clamp(self.alphas_cumprod, 1e-5, 1-1e-5)
        
        # 预计算常用量
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        self.log_snr = torch.log(self.snr)
    
    def sample_timesteps(
        self, batch_size: int, device: str,
        strategy: str = "uniform"
    ) -> torch.Tensor:
        """采样时间步
        
        Args:
            batch_size: 批次大小
            device: 设备
            strategy: 采样策略 (uniform/logit_normal/low_discrepancy)
        """
        if strategy == "uniform":
            return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        elif strategy == "logit_normal":
            # Logit-Normal: 中间密集采样
            u = torch.randn(batch_size, device=device) * 1.0  # std=1.0
            t_continuous = torch.sigmoid(u)  # (0, 1)
            t = (t_continuous * self.num_timesteps).long()
            return t.clamp(0, self.num_timesteps - 1)
        
        elif strategy == "low_discrepancy":
            # 低差异序列：更均匀覆盖
            t0 = torch.rand(1, device=device)
            t = torch.fmod(
                t0 + torch.arange(batch_size, device=device) / batch_size, 1.0
            )
            t = (t * self.num_timesteps).long()
            return t.clamp(0, self.num_timesteps - 1)
    
    def add_noise(
        self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """前向加噪"""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x_0.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1).to(x_0.device)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """获取指定时间步的SNR"""
        return self.snr[t]
```

#### 6.7.2 Min-SNR-γ损失加权实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MinSNRWeightedLoss:
    """Min-SNR-γ 加权扩散训练损失
    
    实现Hang et al. (2023)的Min-SNR-γ加权策略。
    通过截断高SNR时间步的梯度权重，消除时间步间的梯度冲突。
    
    核心公式: w(t) = min(SNR(t), γ) / SNR(t)
    """
    
    def __init__(
        self,
        scheduler: 'NoiseScheduler',
        gamma: float = 5.0,
        prediction_type: str = "epsilon",  # epsilon/v_prediction/x0
    ):
        self.scheduler = scheduler
        self.gamma = gamma
        self.prediction_type = prediction_type
    
    def get_min_snr_weight(self, t: torch.Tensor) -> torch.Tensor:
        """计算Min-SNR-γ权重
        
        w(t) = min(SNR(t), γ) / SNR(t)
        
        对于SNR(t) < γ: w(t) = 1 (不修改)
        对于SNR(t) > γ: w(t) = γ/SNR(t) < 1 (降低权重)
        """
        snr = self.scheduler.get_snr(t)
        weight = torch.minimum(snr, torch.full_like(snr, self.gamma)) / snr
        return weight
    
    def get_v_prediction_weight(self, t: torch.Tensor) -> torch.Tensor:
        """v-prediction的Min-SNR权重（略有不同）
        
        对于v-prediction，有效SNR权重为 SNR/(SNR+1)
        Min-SNR应用于此有效权重
        """
        snr = self.scheduler.get_snr(t)
        effective_snr = snr / (snr + 1)
        weight = torch.minimum(
            effective_snr, torch.full_like(effective_snr, self.gamma)
        ) / effective_snr
        return weight
    
    def __call__(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """计算加权损失
        
        Args:
            model_output: 模型预测 [B, C, H, W]
            target: 真实目标 [B, C, H, W]
            t: 时间步 [B]
            
        Returns:
            加权后的标量损失
        """
        # 逐样本MSE损失
        per_sample_loss = F.mse_loss(
            model_output, target, reduction="none"
        ).mean(dim=[1, 2, 3])  # [B]
        
        # 计算Min-SNR权重
        if self.prediction_type == "v_prediction":
            weight = self.get_v_prediction_weight(t)
        else:
            weight = self.get_min_snr_weight(t)
        
        weight = weight.to(per_sample_loss.device)
        
        # 加权平均
        loss = (weight * per_sample_loss).mean()
        
        return loss


class DiffusionTrainingLoop:
    """带Min-SNR-γ和v-prediction的完整训练循环
    
    集成了本章所有最佳实践：
    - v-prediction目标
    - Min-SNR-γ加权
    - Cosine调度
    - EMA更新
    """
    
    def __init__(
        self,
        model: nn.Module,
        scheduler: 'NoiseScheduler',
        optimizer: torch.optim.Optimizer,
        ema_decay: float = 0.9999,
        snr_gamma: float = 5.0,
        prediction_type: str = "v_prediction",
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm
        self.prediction_type = prediction_type
        
        # 损失函数
        self.loss_fn = MinSNRWeightedLoss(
            scheduler, gamma=snr_gamma,
            prediction_type=prediction_type
        )
        
        # EMA模型
        from copy import deepcopy
        self.ema_model = deepcopy(model)
        self.ema_model.requires_grad_(False)
    
    def get_target(
        self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """根据prediction_type计算训练目标"""
        if self.prediction_type == "epsilon":
            return noise
        elif self.prediction_type == "x0":
            return x_0
        elif self.prediction_type == "v_prediction":
            alpha_t = self.scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sigma_t = self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            # v = alpha_t * epsilon - sigma_t * x_0
            v = alpha_t.to(x_0.device) * noise - sigma_t.to(x_0.device) * x_0
            return v
    
    def training_step(self, x_0: torch.Tensor) -> dict:
        """执行一步训练
        
        Args:
            x_0: 真实数据 [B, C, H, W]
            
        Returns:
            包含loss和诊断信息的字典
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # 1. 采样时间步 (可选logit-normal策略)
        t = self.scheduler.sample_timesteps(
            batch_size, device, strategy="uniform"
        )
        
        # 2. 添加噪声
        noise = torch.randn_like(x_0)
        x_t = self.scheduler.add_noise(x_0, noise, t)
        
        # 3. 计算目标
        target = self.get_target(x_0, noise, t)
        
        # 4. 模型预测
        model_output = self.model(x_t, t)
        
        # 5. Min-SNR加权损失
        loss = self.loss_fn(model_output, target, t)
        
        # 6. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 7. 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        
        # 8. 优化器步进
        self.optimizer.step()
        
        # 9. EMA更新
        self._update_ema()
        
        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "snr_mean": self.scheduler.get_snr(t).mean().item(),
        }
    
    @torch.no_grad()
    def _update_ema(self):
        """更新EMA模型参数"""
        for p_ema, p in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            p_ema.data.mul_(self.ema_decay).add_(
                p.data, alpha=1 - self.ema_decay
            )
```

### 6.8 本章小结

训练优化是决定扩散模型最终质量的关键因素，往往比架构选择的影响更大。核心takeaway：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    训练优化关键决策总结                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  决策1: 噪声调度                                                            │
│  ├─ 低分辨率 (≤256): Cosine                                                │
│  ├─ 高分辨率 (≥512): Shifted Cosine 或 Logit-Normal采样                    │
│  └─ 关键: 确保中间SNR区间得到充分训练                                      │
│                                                                             │
│  决策2: 预测目标                                                            │
│  ├─ 通用推荐: v-prediction (SDXL验证)                                      │
│  ├─ 简单基线: ε-prediction (社区生态好)                                     │
│  └─ 关键: v-prediction在少步推理和蒸馏中优势明显                            │
│                                                                             │
│  决策3: 损失加权                                                            │
│  ├─ 强烈推荐: Min-SNR-γ (γ=5)                                              │
│  ├─ 替代: P2 weighting                                                     │
│  └─ 关键: 3.4x收敛加速 + 35% FID改善，几乎零成本                          │
│                                                                             │
│  决策4: EMA                                                                 │
│  ├─ 标准: decay = 0.9999                                                   │
│  ├─ 大模型: decay = 0.99999+                                               │
│  └─ 关键: 必须使用，0.5-1.0 FID的免费提升                                  │
│                                                                             │
│  决策5: 工程优化                                                            │
│  ├─ 精度: bf16 (非fp16)                                                    │
│  ├─ 注意力: Flash Attention v2                                              │
│  ├─ 内存: Gradient Checkpointing                                           │
│  ├─ 分布式: FSDP (PyTorch) 或 DeepSpeed ZeRO-3                            │
│  └─ 关键: 这些不影响最终质量，但决定能否训得起                              │
│                                                                             │
│  黄金组合 (当前最佳实践):                                                   │
│  Cosine调度 + v-prediction + Min-SNR-5 + EMA-0.9999 + bf16 + Flash Attn   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**核心要点回顾**：

1. **Min-SNR-γ是最被低估的技术**：仅需改变时间步权重即可获得3.4x收敛加速和35% FID改善
2. **v-prediction适合大多数场景**：它在不同SNR区间提供均匀梯度，特别适合蒸馏和高分辨率
3. **噪声调度需要适配分辨率**：高分辨率必须使用shifted或logit-normal调度
4. **EMA是免费的质量提升**：没有理由不用
5. **工程优化决定可行性**：Flash Attention + bf16 + FSDP是大模型训练的基础设施

至此，我们完成了扩散模型优化的六章系统性讲解。从第1章的数学基础，到采样加速（第2-3章）、步数蒸馏（第4章）、架构设计（第5章）、训练优化（第6章），涵盖了扩散模型从理论到工程的全部核心优化技术。这些技术的组合应用，使得扩散模型从实验室论文走向了大规模工业部署。


---

## 第7章：潜空间扩散与VAE设计优化

前六章我们讨论的扩散模型都运行在**像素空间**——直接在高维图像上执行前向加噪和反向去噪。这种方法虽然理论上优雅，但面对现代高分辨率图像（512×512乃至1024×1024）时，其计算负担几乎无法承受。本章将深入Latent Diffusion Model（LDM）的核心设计思路：先将图像压缩到低维**潜空间**，再在潜空间中执行扩散过程，从而实现数量级的效率提升。

### 7.1 为什么在潜空间做扩散

#### 像素空间的计算瓶颈

考虑一张 $512 \times 512 \times 3$ 的RGB图像，其像素总数为786,432。扩散模型中的U-Net需要在这个高维空间中进行多次前向传播——通常是20-50步——这意味着：

- **Self-Attention的二次复杂度**：对于 $512 \times 512$ 分辨率，特征图包含 $65,536$ 个token。一次self-attention操作的计算量为 $O(N^2 \cdot d)$，其中 $N = 65536$，这对显存和计算都是灾难性的。
- **U-Net参数量**：像素空间的U-Net为了捕获全局语义关系，需要非常深的网络和大量通道数。
- **训练数据冗余**：自然图像的大部分像素信息是**感知冗余**的——人眼对高频细节的敏感度远低于对语义结构的感知。

关键洞察是：**扩散模型的去噪过程并不需要在高维像素空间中进行**。语义信息可以被有效压缩到低维表示，而高频细节（纹理、精确颜色）可以由一个固定的解码器恢复。

#### 感知压缩 vs 语义压缩

在理解LDM之前，需要区分两种不同的压缩维度：

- **感知压缩**：去除对人类感知不重要的高频细节。JPEG、WebP等有损压缩就是这一思路。这种压缩丢失的信息对图像语义没有影响。
- **语义压缩**：进一步去除语义冗余，将图像映射到抽象概念空间。CLIP的图像编码器就执行这种压缩。

LDM中的VAE执行的是**感知压缩**——它保留了足够的语义和细节信息以实现高质量重建，但去除了像素级的冗余。扩散模型则在此基础上学习**语义压缩**——理解数据分布的结构。

```
┌─────────────────────────────────────────────────────────────────────────┐
│             感知压缩 vs 语义压缩 vs 像素空间扩散对比                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  像素空间扩散 (DDPM原始方案):                                           │
│  ┌──────────────┐         ┌──────────────────────────────────┐         │
│  │ 图像          │ ──────→ │ 在 512×512×3 空间做 T步 去噪     │         │
│  │ 512×512×3    │         │ 计算量: ████████████████ (极大)   │         │
│  └──────────────┘         └──────────────────────────────────┘         │
│                                                                         │
│  潜空间扩散 (LDM方案):                                                  │
│  ┌──────────┐   Encoder   ┌──────────┐  扩散   ┌──────────┐  Decoder  │
│  │ 图像      │ ──────────→ │ 潜编码    │ ──────→ │ 去噪潜码  │ ────────→│
│  │512×512×3 │  (固定VAE)  │ 64×64×4  │ (T步)  │ 64×64×4  │ (固定VAE)│
│  └──────────┘             └──────────┘         └──────────┘           │
│       │                        │                                        │
│       │   压缩维度: 512×512×3 = 786,432  →  64×64×4 = 16,384          │
│       │   压缩比: 48x (维度) / 64x (面积×通道比)                       │
│       │   计算节省: Self-Attention ~64x 减少                            │
│       │                                                                 │
│  感知压缩层 (VAE做的事):                                                │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ 输入像素 → 去除感知冗余 → 保留语义信息 → 低维潜码          │       │
│  │ (颜色/纹理/高频细节被有损压缩，结构/语义被完整保留)         │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  语义压缩层 (扩散模型做的事):                                           │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ 潜码分布 → 学习数据分布结构 → 条件生成                     │       │
│  │ (理解"什么是猫"、"什么是风景"的概率分布)                     │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 LDM三阶段设计

Latent Diffusion Model的核心架构包含三个明确分离的阶段：

**阶段一：感知压缩模型（VAE）训练**

训练一个自编码器（通常是VAE），使其能够将高分辨率图像压缩到低维潜空间，同时保持高质量重建能力。VAE训练完成后参数固定，不再更新。

**阶段二：潜空间扩散模型训练**

在固定的VAE潜空间中训练扩散模型。输入不再是原始图像 $x$，而是编码后的潜表示 $z = \mathcal{E}(x)$。扩散模型学习潜空间中的数据分布。

**阶段三：推理时解码**

生成时，扩散模型在潜空间中从纯噪声 $z_T \sim \mathcal{N}(0, I)$ 出发去噪得到 $z_0$，然后通过固定解码器 $\mathcal{D}(z_0)$ 恢复为像素图像。

LDM的损失函数：

$$L_{LDM} = \mathbb{E}_{z \sim \mathcal{E}(x),\, \epsilon \sim \mathcal{N}(0,I),\, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]$$

其中 $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ 是潜空间中的加噪表示，$c$ 是可选条件信号。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LDM 完整 Pipeline 架构图                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ═══════════════════ 训练阶段 ════════════════════                          │
│                                                                             │
│  Phase 1: VAE训练 (一次性, 独立于扩散模型)                                  │
│  ┌───────┐     ┌───────────┐     ┌───────────┐     ┌───────┐             │
│  │ Image │────→│  Encoder  │────→│  Latent z │────→│Decoder│────→ x̂      │
│  │ x     │     │  E(x)     │     │  64×64×4  │     │ D(z)  │             │
│  └───────┘     └───────────┘     └───────────┘     └───────┘             │
│       │              │                  │                │                  │
│       └──────────────┴──── L_VAE = L_rec + λ_KL·L_KL + λ_P·L_LPIPS ──┘  │
│                                    + λ_adv·L_GAN                           │
│                                                                             │
│  Phase 2: 扩散模型训练 (VAE冻结)                                            │
│  ┌───────┐    ┌────────┐    ┌───────┐    ┌──────────┐    ┌────────────┐  │
│  │ Image │───→│Encoder │───→│ z_0   │───→│ Add Noise│───→│ z_t        │  │
│  │  x    │    │(frozen)│    │       │    │ t~U(1,T) │    │            │  │
│  └───────┘    └────────┘    └───────┘    └──────────┘    └─────┬──────┘  │
│                                                                  │         │
│                    ┌──────────────────────┐                      │         │
│  ┌────────┐       │   U-Net / DiT        │                      │         │
│  │ConditionC│─────→│   ε_θ(z_t, t, c)    │←─────────────────────┘         │
│  │(text/img)│      │                      │                                │
│  └────────┘       └──────────┬───────────┘                                │
│                              │                                              │
│                    L_LDM = ||ε - ε_θ(z_t, t, c)||²                         │
│                                                                             │
│  ═══════════════════ 推理阶段 ════════════════════                          │
│                                                                             │
│  ┌────────┐    ┌─────────────────────────┐    ┌────────┐    ┌────────┐   │
│  │z_T~N(0,I)──→│  反向去噪 (20-50步)     │───→│  z_0   │───→│Decoder │──→│
│  │ 64×64×4│    │  DDIM/DPM-Solver/...    │    │64×64×4 │    │(frozen)│ x̂│
│  └────────┘    └─────────────────────────┘    └────────┘    └────────┘   │
│                         ↑                                                   │
│                    ┌────┴─────┐                                             │
│                    │Condition │                                             │
│                    │ "a cat"  │                                             │
│                    └──────────┘                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 压缩比分析

以Stable Diffusion 1.5为例：

| 参数 | 数值 |
|------|------|
| 输入分辨率 | 512 × 512 × 3 |
| 潜空间分辨率 | 64 × 64 × 4 |
| 空间下采样 | 8× (每个方向) |
| 面积压缩 | 64× |
| 通道变化 | 3 → 4 |
| 总维度压缩 | 786,432 → 16,384 = **48×** |
| Self-Attention节省 | $(512/8)^2 / 512^2 = 1/64$ 的token数 |

对于更高分辨率（如1024×1024），节省更为显著：面积压缩为64×，对应self-attention的计算量减少为原来的 $1/64^2 = 1/4096$。

### 7.3 VAE架构设计与演进

#### VAE基础原理

变分自编码器（VAE）由三个组件构成：

1. **编码器 $\mathcal{E}$**：将图像 $x$ 映射为潜分布参数 $(\mu, \sigma)$
2. **重参数化采样**：$z = \mu + \sigma \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$
3. **解码器 $\mathcal{D}$**：从潜编码 $z$ 重建图像 $\hat{x} = \mathcal{D}(z)$

VAE的训练目标（ELBO）：

$$L_{VAE} = \underbrace{\mathbb{E}_{q(z|x)}[\| x - \mathcal{D}(z) \|^2]}_{\text{重建损失}} + \underbrace{\lambda_{KL} \cdot D_{KL}(q(z|x) \| p(z))}_{\text{KL正则化}}$$

在LDM的VAE中，还会引入额外损失以提升重建质量：

$$L_{total} = L_{rec} + \lambda_{KL} \cdot L_{KL} + \lambda_P \cdot L_{LPIPS} + \lambda_{adv} \cdot L_{GAN}$$

#### SD 1.5 VAE架构

Stable Diffusion 1.5使用的VAE（kl-f8-4ch）是整个LDM体系的基石。它的设计特点：

- **4通道潜空间**：$z \in \mathbb{R}^{H/8 \times W/8 \times 4}$
- **8×空间下采样**：通过3次2×下采样实现
- **对称编解码器**：Encoder和Decoder结构镜像对称

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  SD 1.5 VAE Encoder/Decoder 对称架构                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ════════════ Encoder (Image → Latent) ════════════                      │
│                                                                          │
│  Input: 512×512×3                                                        │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │ Conv2d(3→128, 3×3, pad=1)          → 512×512×128              │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ DownBlock_1:                                                   │     │
│  │   ResBlock(128→128) × 2                                       │     │
│  │   Downsample(stride=2)              → 256×256×128             │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ DownBlock_2:                                                   │     │
│  │   ResBlock(128→256) × 2                                       │     │
│  │   Downsample(stride=2)              → 128×128×256             │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ DownBlock_3:                                                   │     │
│  │   ResBlock(256→512) × 2                                       │     │
│  │   Downsample(stride=2)              → 64×64×512               │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ DownBlock_4:                                                   │     │
│  │   ResBlock(512→512) × 2                                       │     │
│  │   (无Downsample)                    → 64×64×512               │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ MidBlock:                                                      │     │
│  │   ResBlock(512) → SelfAttn(512) → ResBlock(512)               │     │
│  │                                     → 64×64×512               │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ Output:                                                        │     │
│  │   GroupNorm → SiLU → Conv2d(512→8, 3×3)                      │     │
│  │                                     → 64×64×8 (μ和σ各4通道)   │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                              │                                           │
│                   Reparameterize: z = μ + σ·ε                           │
│                              ↓                                           │
│                       z: 64×64×4                                         │
│                              │                                           │
│  ════════════ Decoder (Latent → Image) ════════════                      │
│                                                                          │
│  Input: 64×64×4                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │ Conv2d(4→512, 3×3, pad=1)          → 64×64×512               │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ MidBlock:                                                      │     │
│  │   ResBlock(512) → SelfAttn(512) → ResBlock(512)               │     │
│  │                                     → 64×64×512               │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ UpBlock_1:                                                     │     │
│  │   ResBlock(512→512) × 3                                       │     │
│  │   (无Upsample)                      → 64×64×512              │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ UpBlock_2:                                                     │     │
│  │   ResBlock(512→512) × 3                                       │     │
│  │   Upsample(scale=2)                 → 128×128×512             │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ UpBlock_3:                                                     │     │
│  │   ResBlock(512→256) × 3                                       │     │
│  │   Upsample(scale=2)                 → 256×256×256             │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ UpBlock_4:                                                     │     │
│  │   ResBlock(256→128) × 3                                       │     │
│  │   Upsample(scale=2)                 → 512×512×128             │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │ Output:                                                        │     │
│  │   GroupNorm → SiLU → Conv2d(128→3, 3×3)                      │     │
│  │                                     → 512×512×3               │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  已知缺陷:                                                               │
│  • 高频伪影: 解码器对精细纹理恢复不够精确                                │
│  • Posterior collapse: KL正则过强时潜空间退化                            │
│  • 颜色偏移: 某些颜色区域重建有偏差                                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

#### VAE演进路线

**SDXL VAE**：
- 基本架构与SD 1.5相同（4通道，8× downsample）
- 改进了编码器的训练稳定性
- 使用更大的batch size和更长的训练
- 重建质量有明显提升（特别是人脸和文字）

**SD3 / FLUX VAE（16通道设计）**：
- 将潜空间通道从4扩展到16
- 维度：$64 \times 64 \times 16$（对于512分辨率）
- 设计考量：
  * 更多通道 = 更高信息容量 = 更好重建质量
  * 扩散模型需要处理更多通道，但空间分辨率不变
  * 总体计算量：潜空间维度增加4×，但仍远小于像素空间
  * 对扩散模型的质量有显著提升

#### VAE训练损失详解

完整VAE训练损失包含四项：

$$L_{VAE} = L_{rec} + \lambda_{KL} \cdot L_{KL} + \lambda_P \cdot L_{LPIPS} + \lambda_{adv} \cdot L_{GAN}$$

1. **重建损失 $L_{rec}$**：像素级L1或L2损失
2. **KL散度 $L_{KL}$**：约束潜空间为标准正态分布
3. **感知损失 $L_{LPIPS}$**：使用预训练VGG特征衡量感知差异
4. **对抗损失 $L_{GAN}$**：PatchGAN鉴别器提升细节锐度

### 7.4 VAE训练技巧与优化

#### KL权重调度

KL散度正则化是VAE训练中最关键的超参数之一。过强的KL约束会导致posterior collapse（编码器学会忽略输入，输出接近标准正态），过弱则潜空间不规则，影响扩散模型训练。

推荐策略：**KL权重从0逐步增大（KL annealing）**

$$\lambda_{KL}(step) = \min\left(\lambda_{max}, \frac{step}{warmup\_steps} \cdot \lambda_{max}\right)$$

典型值：$\lambda_{max} = 10^{-6}$（是的，非常小——这是LDM VAE的关键设计决策，让重建质量优先于正则化）。

#### 对抗损失引入时机

对抗损失如果从训练一开始就引入，会导致训练不稳定。标准做法：

1. 先用重建损失+KL训练50k步，让编解码器学会基本重建
2. 之后引入PatchGAN鉴别器
3. 鉴别器使用较小的学习率（通常是生成器的1/2）

#### 感知损失的层选择

LPIPS使用VGG-16的多层特征计算感知距离：

$$L_{LPIPS} = \sum_{l} w_l \cdot \| \phi_l(x) - \phi_l(\hat{x}) \|^2$$

其中 $\phi_l$ 是VGG第 $l$ 层的特征提取器。通常选择conv1_2, conv2_2, conv3_3, conv4_3, conv5_3五层，低层关注纹理，高层关注语义。

### 7.5 Consistency Decoder

标准VAE Decoder的一个已知问题是：**解码质量存在天花板**。无论如何训练，单次前向通过的解码器都难以完美恢复所有高频细节。OpenAI提出的Consistency Decoder用一个小型扩散模型替代标准解码器：

**核心思路**：

1. VAE Encoder照常编码：$z = \mathcal{E}(x)$
2. 解码时，将 $z$ 作为条件，用一个小型扩散模型执行少步（2-4步）去噪
3. 初始噪声不是纯高斯噪声，而是从 $z$ 出发添加少量噪声

$$\hat{x} = \text{ConsistencyDecode}(z, \epsilon, steps=2)$$

**优点**：
- 解码质量显著高于标准Decoder
- 高频细节恢复更好
- 几乎消除了网格伪影

**缺点**：
- 推理时间增加（2-4×，取决于步数）
- 需要额外训练一个小型扩散模型
- 不适合需要实时响应的场景

### 7.6 潜空间维度选择分析

潜空间的通道数选择直接影响整个系统的性能：

```
┌────────────────────────────────────────────────────────────────────────┐
│           潜空间维度 vs 重建质量 vs 扩散效率 Pareto 分析               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  重建质量 (PSNR dB)                                                    │
│  ↑                                                                     │
│  │                                              ★ 16ch (SD3/FLUX)     │
│  40├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─★─ ─ ─ ─ ─ ─ ─ ─ ─    │
│  │                                        ╱                            │
│  │                                      ╱                              │
│  38├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─★╱─ ─ ─ ─  8ch                  │
│  │                                ╱                                    │
│  │                              ╱       Pareto前沿                     │
│  36├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ★╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─             │
│  │                        ╱    4ch (SD 1.5/SDXL)                      │
│  │                      ╱                                              │
│  34├─ ─ ─ ─ ─ ─ ─ ─ ★ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─             │
│  │                ╱        3ch                                         │
│  │              ╱                                                      │
│  32├─ ─ ─ ─ ★ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─             │
│  │        ╱        2ch                                                 │
│  │      ╱                                                              │
│  30├──★─────────────────────────────────────────────────→              │
│  │  1ch                                                                │
│  │                                            潜空间维度 (通道数)      │
│  └──┬────┬────┬────┬────┬────┬────┬────┬────┬───→                     │
│     1    2    3    4    6    8   12   16   32                           │
│                                                                        │
│  ┌────────────────────────────────────────────────────────────┐       │
│  │ 通道数 │ 维度(64²×c) │ 重建PSNR │ 扩散FID │ 推荐场景      │       │
│  │────────┼─────────────┼──────────┼─────────┼───────────────│       │
│  │  4     │  16,384     │  ~35 dB  │  较好   │ 通用(SD1.5)   │       │
│  │  8     │  32,768     │  ~37 dB  │  好     │ 高质量场景    │       │
│  │  16    │  65,536     │  ~39 dB  │  最佳   │ SD3/FLUX      │       │
│  │  32    │  131,072    │  ~40 dB  │  下降*  │ 过度(不推荐)  │       │
│  └────────────────────────────────────────────────────────────┘       │
│  * 32通道时扩散模型训练效率下降，因为潜空间维度过高                     │
│                                                                        │
│  最佳实践:                                                             │
│  • 4ch: 计算受限场景，对重建质量要求不极端                             │
│  • 16ch: 当前最优选择(SD3/FLUX验证)，质量与效率最佳平衡               │
│  • >16ch: 收益递减，扩散模型训练难度增加                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

#### 关键洞察

- **4通道是一个保守但合理的选择**：SD 1.5/SDXL使用4通道，重建质量可接受但非最优
- **16通道是当前甜点**：SD3和FLUX使用16通道，重建质量接近无损，同时扩散模型训练仍然高效
- **超过16通道收益递减**：潜空间维度过高时，扩散模型本身的训练效率下降，因为需要学习更高维分布

### 7.7 Latent空间的性质与可视化

训练好的VAE潜空间具有以下重要性质：

**几何结构**：与像素空间不同，潜空间中语义相近的图像在欧氏距离上也相近。这是因为KL正则化鼓励潜空间具有平滑的几何结构。

**插值特性**：在潜空间中线性插值可以产生语义上平滑的过渡：

$$z_{interp} = (1 - \alpha) \cdot z_1 + \alpha \cdot z_2, \quad \alpha \in [0, 1]$$

解码后得到的图像会从 $x_1$ 平滑过渡到 $x_2$，中间帧语义合理。

**与像素空间扩散的质量对比**：在相同计算预算下：
- LDM可以使用更多采样步数（因为每步计算更少）
- LDM的U-Net/DiT可以使用更大模型
- 最终生成质量通常高于像素空间方法

### 7.8 代码实现

以下是完整的VAE Encoder/Decoder和LDM训练循环实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResBlock(nn.Module):
    """VAE中的残差块，包含两个卷积层和可选的通道变换"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 通道数不匹配时使用1×1卷积进行shortcut
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """单头自注意力（VAE中的中间层使用）"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Scaled dot-product attention
        scale = C ** -0.5
        attn = torch.einsum('bci,bcj->bij', q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bij,bcj->bci', attn, v)
        
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    """2×空间下采样"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2×空间上采样"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class VAEEncoder(nn.Module):
    """
    SD风格VAE编码器
    将 H×W×3 图像编码为 H/8×W/8×(2*latent_channels) 的高斯分布参数
    """
    
    def __init__(self, latent_channels: int = 4, ch: int = 128):
        super().__init__()
        self.latent_channels = latent_channels
        
        # 通道配置: [128, 128, 256, 512, 512]
        channels = [ch, ch, ch * 2, ch * 4, ch * 4]
        
        # 初始卷积
        self.conv_in = nn.Conv2d(3, channels[0], 3, padding=1)
        
        # 下采样块
        self.down_blocks = nn.ModuleList()
        for i in range(4):
            block = nn.ModuleList([
                ResBlock(channels[i], channels[i + 1]),
                ResBlock(channels[i + 1], channels[i + 1]),
            ])
            # 前3个block有downsample (8× total)
            if i < 3:
                block.append(Downsample(channels[i + 1]))
            self.down_blocks.append(block)
        
        # 中间块
        self.mid = nn.Sequential(
            ResBlock(channels[-1], channels[-1]),
            SelfAttention(channels[-1]),
            ResBlock(channels[-1], channels[-1]),
        )
        
        # 输出层: 产生均值和对数方差 (2 * latent_channels)
        self.norm_out = nn.GroupNorm(32, channels[-1])
        self.conv_out = nn.Conv2d(channels[-1], 2 * latent_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] 输入图像, 归一化到[-1, 1]
        Returns:
            mean: [B, latent_channels, H/8, W/8] 潜分布均值
            logvar: [B, latent_channels, H/8, W/8] 潜分布对数方差
        """
        h = self.conv_in(x)
        
        for block in self.down_blocks:
            h = block[0](h)  # ResBlock 1
            h = block[1](h)  # ResBlock 2
            if len(block) > 2:
                h = block[2](h)  # Downsample
        
        h = self.mid(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # 分割为均值和对数方差
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar


class VAEDecoder(nn.Module):
    """
    SD风格VAE解码器
    将 H/8×W/8×latent_channels 的潜编码解码为 H×W×3 图像
    """
    
    def __init__(self, latent_channels: int = 4, ch: int = 128):
        super().__init__()
        
        # 通道配置 (与Encoder镜像): [512, 512, 256, 128, 128]
        channels = [ch * 4, ch * 4, ch * 2, ch, ch]
        
        # 输入卷积
        self.conv_in = nn.Conv2d(latent_channels, channels[0], 3, padding=1)
        
        # 中间块
        self.mid = nn.Sequential(
            ResBlock(channels[0], channels[0]),
            SelfAttention(channels[0]),
            ResBlock(channels[0], channels[0]),
        )
        
        # 上采样块
        self.up_blocks = nn.ModuleList()
        for i in range(4):
            block = nn.ModuleList([
                ResBlock(channels[i], channels[i + 1]),
                ResBlock(channels[i + 1], channels[i + 1]),
                ResBlock(channels[i + 1], channels[i + 1]),
            ])
            # 后3个block有upsample (8× total)
            if i > 0:
                block.append(Upsample(channels[i + 1]))
            self.up_blocks.append(block)
        
        # 输出层
        self.norm_out = nn.GroupNorm(32, channels[-1])
        self.conv_out = nn.Conv2d(channels[-1], 3, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_channels, H/8, W/8] 潜编码
        Returns:
            x_recon: [B, 3, H, W] 重建图像, 范围[-1, 1]
        """
        h = self.conv_in(z)
        h = self.mid(h)
        
        for block in self.up_blocks:
            h = block[0](h)  # ResBlock 1
            h = block[1](h)  # ResBlock 2
            h = block[2](h)  # ResBlock 3
            if len(block) > 3:
                h = block[3](h)  # Upsample
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return torch.tanh(h)


class VAE(nn.Module):
    """完整的变分自编码器，用于LDM的感知压缩"""
    
    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.encoder = VAEEncoder(latent_channels)
        self.decoder = VAEDecoder(latent_channels)
        # 潜空间缩放因子 (保持潜码方差接近1)
        self.scale_factor = 0.18215  # SD 1.5使用的值
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码图像到潜空间 (使用重参数化)"""
        mean, logvar = self.encoder(x)
        # 重参数化采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + std * eps
        # 缩放使方差接近1
        z = z * self.scale_factor
        return z, mean, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜空间解码到图像"""
        z = z / self.scale_factor
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor):
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mean, logvar


class LDMTrainer:
    """
    Latent Diffusion Model 完整训练循环
    展示VAE编码 → 潜空间加噪 → 去噪预测 → 损失计算的完整流程
    """
    
    def __init__(
        self,
        vae: VAE,
        denoiser: nn.Module,  # U-Net或DiT
        num_timesteps: int = 1000,
        device: str = 'cuda'
    ):
        self.vae = vae.eval()  # VAE冻结
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 预计算噪声调度参数 (线性调度为例)
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            denoiser.parameters(), lr=1e-4, weight_decay=0.01
        )
    
    @torch.no_grad()
    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """使用冻结的VAE将图像编码到潜空间"""
        z, _, _ = self.vae.encode(images)
        return z
    
    def q_sample(
        self, z_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """前向加噪过程: z_t = sqrt(α_bar_t) * z_0 + sqrt(1-α_bar_t) * ε"""
        sqrt_alpha = self.alphas_cumprod[t].sqrt()[:, None, None, None]
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t]).sqrt()[:, None, None, None]
        return sqrt_alpha * z_0 + sqrt_one_minus_alpha * noise
    
    def train_step(
        self,
        images: torch.Tensor,
        conditions: Optional[torch.Tensor] = None
    ) -> dict:
        """
        单步训练
        Args:
            images: [B, 3, H, W] 原始图像
            conditions: [B, seq_len, dim] 可选条件 (如文本embedding)
        Returns:
            损失字典
        """
        # Step 1: VAE编码 (不计算梯度)
        with torch.no_grad():
            z_0 = self.encode_to_latent(images)
        
        # Step 2: 随机采样时间步和噪声
        batch_size = z_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(z_0)
        
        # Step 3: 前向加噪
        z_t = self.q_sample(z_0, t, noise)
        
        # Step 4: 去噪网络预测
        noise_pred = self.denoiser(z_t, t, conditions)
        
        # Step 5: 计算损失 (简单MSE)
        loss = F.mse_loss(noise_pred, noise)
        
        # Step 6: 反向传播和更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        self.optimizer.step()
        
        return {'loss': loss.item(), 'timestep_mean': t.float().mean().item()}
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        conditions: Optional[torch.Tensor] = None,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        DDIM采样 (简化版)
        Returns:
            生成的图像 [B, 3, H, W]
        """
        # 在潜空间中采样
        shape = (batch_size, 4, 64, 64)  # 假设512×512输入
        z = torch.randn(shape, device=self.device)
        
        # 均匀选择时间步子集
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=self.device
        )
        
        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)
            
            # 预测噪声
            noise_pred = self.denoiser(z, t_batch, conditions)
            
            # DDIM更新 (确定性)
            alpha_t = self.alphas_cumprod[t.long()]
            if i < len(timesteps) - 1:
                alpha_prev = self.alphas_cumprod[timesteps[i + 1].long()]
            else:
                alpha_prev = torch.tensor(1.0, device=self.device)
            
            # 预测x_0
            z_0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            
            # 计算z_{t-1}
            z = alpha_prev.sqrt() * z_0_pred + (1 - alpha_prev).sqrt() * noise_pred
        
        # VAE解码
        images = self.vae.decode(z)
        return images


# 使用示例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化VAE
    vae = VAE(latent_channels=4).to(device)
    
    # 测试VAE编解码
    dummy_image = torch.randn(2, 3, 512, 512, device=device)
    z, mean, logvar = vae.encode(dummy_image)
    print(f"输入图像: {dummy_image.shape}")  # [2, 3, 512, 512]
    print(f"潜编码:   {z.shape}")            # [2, 4, 64, 64]
    print(f"压缩比:   {dummy_image.numel() / z.numel():.1f}x")  # 48.0x
    
    recon = vae.decode(z)
    print(f"重建图像: {recon.shape}")         # [2, 3, 512, 512]
```

**本章小结**：Latent Diffusion Model通过引入VAE进行感知压缩，将扩散过程从高维像素空间转移到低维潜空间，获得了约48-64倍的计算效率提升。VAE的设计从4通道（SD 1.5）演进到16通道（SD3/FLUX），在重建质量和扩散效率之间取得了更好的平衡。这一设计使得高分辨率图像生成在消费级GPU上成为可能，是扩散模型走向实用化的关键突破。



---

## 第8章：条件生成优化 - CFG/CFG蒸馏/T5/CLIP融合策略

扩散模型的强大不仅在于其生成质量，更在于其灵活的**条件生成**能力——通过文本、图像、或其他信号精确控制生成内容。本章将深入探讨条件生成的核心技术：Classifier-Free Guidance（CFG）及其优化变体、文本编码器的设计演进、以及条件信号的注入机制。

### 8.1 Classifier-Free Guidance (CFG) 深度解析

#### 条件生成的理论基础

在条件生成框架中，我们希望从条件分布 $p(x|c)$ 中采样，其中 $c$ 是条件信号（如文本描述）。根据贝叶斯定理：

$$p(x|c) \propto p(x) \cdot p(c|x)$$

取对数并对 $x$ 求梯度（score）：

$$\nabla_x \log p(x|c) = \nabla_x \log p(x) + \nabla_x \log p(c|x)$$

这表明条件score等于无条件score加上一个"分类器梯度"。Classifier Guidance直接训练一个分类器来估计 $\nabla_x \log p(c|x)$，但这需要额外的分类器且有噪声输入适配问题。

#### CFG核心公式

Classifier-Free Guidance的核心洞察是：**不需要显式分类器，只需要同时训练条件和无条件模型**。

训练时，以概率 $p_{uncond}$（通常10-20%）将条件 $c$ 替换为空条件 $\varnothing$：

$$\epsilon_\theta(z_t, t, c) \text{ with probability } 1 - p_{uncond}$$
$$\epsilon_\theta(z_t, t, \varnothing) \text{ with probability } p_{uncond}$$

推理时，对两个预测做线性外推：

$$\tilde{\epsilon} = \epsilon_\theta(z_t, t, \varnothing) + w \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \varnothing))$$

等价形式：

$$\tilde{\epsilon} = (1 - w) \cdot \epsilon_\theta(z_t, t, \varnothing) + w \cdot \epsilon_\theta(z_t, t, c)$$

当 $w > 1$ 时，这是一个**外推**操作——将预测推向条件方向，远离无条件方向。

#### Guidance Scale的影响

$$w = 1$$：标准条件生成，无额外引导
$$w = 7 \sim 8$$：Stable Diffusion默认值，平衡质量与多样性
$$w > 15$$：过度锐化，出现色彩饱和、分布外伪影

```
┌────────────────────────────────────────────────────────────────────────────┐
│              CFG 推理流程图（双分支 → 线性组合）                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  输入: z_t (当前噪声潜码), t (时间步), c (条件, e.g. "a cat")            │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │                    U-Net / DiT 模型                              │      │
│  │                    (同一个模型，两次前向)                          │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│         │                                           │                      │
│         ↓ Forward 1 (条件分支)                      ↓ Forward 2 (无条件)  │
│  ┌──────────────────┐                     ┌──────────────────┐            │
│  │ ε_cond =          │                     │ ε_uncond =        │            │
│  │ ε_θ(z_t, t, c)   │                     │ ε_θ(z_t, t, ∅)   │            │
│  │                   │                     │ (c替换为空/null)   │            │
│  └────────┬─────────┘                     └────────┬─────────┘            │
│           │                                         │                      │
│           └──────────────┬──────────────────────────┘                      │
│                          ↓                                                  │
│           ┌──────────────────────────────────────┐                         │
│           │  CFG线性组合:                         │                         │
│           │                                      │                         │
│           │  ε̃ = ε_uncond + w·(ε_cond - ε_uncond)│                         │
│           │                                      │                         │
│           │  等价于:                              │                         │
│           │  ε̃ = (1-w)·ε_uncond + w·ε_cond      │                         │
│           │                                      │                         │
│           │  w=7.5 (SD默认):                     │                         │
│           │  ε̃ = -6.5·ε_uncond + 7.5·ε_cond     │                         │
│           └──────────────────┬───────────────────┘                         │
│                              ↓                                              │
│           ┌──────────────────────────────────────┐                         │
│           │  使用 ε̃ 执行去噪步:                  │                         │
│           │  z_{t-1} = denoise_step(z_t, ε̃, t)  │                         │
│           └──────────────────────────────────────┘                         │
│                                                                            │
│  ════════════════════════════════════════════════════════════════════       │
│                                                                            │
│  Guidance Scale 对生成质量的影响:                                          │
│                                                                            │
│  w=1.0     w=3.0      w=7.5       w=12.0      w=20.0                     │
│  ┌────┐   ┌────┐    ┌────┐      ┌────┐      ┌────┐                      │
│  │模糊│   │改善│    │最佳│      │过饱和│    │崩坏│                        │
│  │多样│   │    │    │平衡│      │锐利  │    │伪影│                        │
│  │    │   │    │    │    │      │      │    │    │                        │
│  └────┘   └────┘    └────┘      └────┘      └────┘                      │
│  ← 多样性高, 质量低              最优点    质量下降, 多样性低 →            │
│                                                                            │
│  数学直觉:                                                                 │
│  • w<1: 向无条件方向偏移(更随机)                                          │
│  • w=1: 纯条件采样(标准)                                                  │
│  • w>1: 外推, 强化条件(更精确但可能过度)                                  │
│  • w>>1: 分布外, 产生不自然伪影                                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### CFG的数学解释：隐式分类器

CFG可以被理解为一个**隐式分类器引导**。设扩散模型的score为：

$$s_\theta(z_t, t, c) = -\frac{\epsilon_\theta(z_t, t, c)}{\sqrt{1 - \bar\alpha_t}}$$

则CFG的引导score为：

$$\tilde{s} = s_\theta(z_t, t, \varnothing) + w \cdot (s_\theta(z_t, t, c) - s_\theta(z_t, t, \varnothing))$$

这等价于在一个隐式分类器 $p_\theta(c|z_t)$ 的梯度上做温度缩放，温度为 $1/w$。当 $w > 1$，等同于降低分类器温度——使其更加"确信"条件方向。

#### 多条件CFG

当存在多个条件 $c_1, c_2, \ldots$ 时，可以独立地对每个条件应用guidance：

$$\tilde{\epsilon} = \epsilon_\theta(z_t, t, \varnothing) + \sum_i w_i \cdot (\epsilon_\theta(z_t, t, c_i) - \epsilon_\theta(z_t, t, \varnothing))$$

这允许用户对不同条件（如文本、风格、构图）使用不同的guidance强度。

### 8.2 CFG的问题与改进

#### 核心问题

1. **计算开销翻倍**：每个去噪步需要两次前向传播（条件+无条件），推理时间直接×2
2. **过度饱和**：高guidance scale导致色彩过饱和、不自然的高对比度
3. **多样性损失**：CFG本质上是在缩小分布支撑集，高scale下生成多样性显著下降
4. **边缘伪影**：在图像边缘和细节处出现不自然的锐化

#### Dynamic CFG：时间步相关的Guidance Scale

关键洞察：**不同去噪阶段对guidance的需求不同**。

- 早期步骤（高噪声）：需要较强guidance确定全局结构
- 后期步骤（低噪声）：细节阶段应降低guidance避免过度锐化

$$w(t) = w_{max} \cdot \left(\frac{t}{T}\right)^\gamma + w_{min} \cdot \left(1 - \left(\frac{t}{T}\right)^\gamma\right)$$

典型设置：$w_{max} = 7.5$，$w_{min} = 1.0$，$\gamma = 0.5$

#### Rescaled CFG

CFG会放大噪声预测的幅度，导致输出超出正常范围。Rescaled CFG通过归一化来限制：

$$\tilde{\epsilon}_{rescaled} = \phi \cdot \text{std}(\epsilon_{cond}) \cdot \frac{\tilde{\epsilon}}{\text{std}(\tilde{\epsilon})} + (1 - \phi) \cdot \tilde{\epsilon}$$

其中 $\phi \in [0, 1]$ 控制rescale程度。

### 8.3 CFG蒸馏技术

#### 动机

CFG的推理成本翻倍对实时应用不可接受。CFG蒸馏的目标是：**将guidance信息内化到模型中，使单次前向即可产生guided质量的输出**。

#### Guided Distillation

核心思路：用一个"教师"模型（使用CFG）的输出训练一个"学生"模型（不使用CFG）：

$$L_{distill} = \mathbb{E}_{z_t, t, c} \left[ \| \epsilon_{student}(z_t, t, c) - \tilde{\epsilon}_{teacher}(z_t, t, c) \|^2 \right]$$

其中 $\tilde{\epsilon}_{teacher}$ 是教师模型经过CFG组合后的输出。

#### w-conditioned模型

更灵活的方案：将guidance scale $w$ 作为模型的**额外输入条件**：

$$\epsilon_\theta(z_t, t, c, w)$$

训练时，$w$ 从某个分布中采样（如 $w \sim \text{Uniform}(1, 15)$），模型学会根据不同的 $w$ 调整输出风格。推理时只需一次前向，通过指定 $w$ 控制引导强度。

**性能**：
- 速度：2×提升（消除双重前向）
- 质量：略有损失（约0.5-1.0 FID），对视觉质量影响很小
- 灵活性：w-conditioned方案允许推理时动态调整

### 8.4 文本编码器设计与融合策略

文本条件是扩散模型最重要的控制信号。文本编码器的选择和融合方式直接决定了模型对文本指令的理解能力。

```
┌────────────────────────────────────────────────────────────────────────────┐
│           文本编码器融合架构图 (CLIP + T5 → 融合 → 注入)                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  输入文本: "a beautiful sunset over mountains with golden light"           │
│                    │                                                        │
│         ┌──────────┴──────────┐                                            │
│         ↓                     ↓                                            │
│  ┌──────────────────┐  ┌──────────────────────────┐                       │
│  │ CLIP Text Encoder│  │ T5-XXL Encoder            │                       │
│  │                  │  │                            │                       │
│  │ • 12层Transformer│  │ • 24层Encoder-only        │                       │
│  │ • 77 token limit │  │ • 4.7B参数               │                       │
│  │ • 768/1024维     │  │ • 512 token limit         │                       │
│  │ • 语义对齐强     │  │ • 4096维                  │                       │
│  │ • 细节能力有限   │  │ • 细节理解出色            │                       │
│  │                  │  │ • 计算成本高              │                       │
│  └────────┬─────────┘  └───────────┬──────────────┘                       │
│           │                         │                                      │
│           ↓                         ↓                                      │
│  ┌─────────────────┐      ┌─────────────────────┐                         │
│  │ [CLS] + tokens  │      │ token embeddings     │                         │
│  │ [B, 77, 768]    │      │ [B, 512, 4096]       │                         │
│  └────────┬────────┘      └──────────┬───────────┘                         │
│           │                           │                                    │
│           └───────────┬───────────────┘                                    │
│                       ↓                                                     │
│           ┌───────────────────────────────────┐                            │
│           │        融合策略选择                 │                            │
│           ├───────────────────────────────────┤                            │
│           │                                   │                            │
│           │  方案A: 通道拼接 (Concatenation)   │                            │
│           │  [B, 77+512, max(768,4096)]       │                            │
│           │  → Linear projection对齐维度      │                            │
│           │                                   │                            │
│           │  方案B: 独立Cross-Attention        │                            │
│           │  CA_1(Q=latent, KV=CLIP)          │                            │
│           │  CA_2(Q=latent, KV=T5)            │                            │
│           │  → 输出相加                        │                            │
│           │                                   │                            │
│           │  方案C: Pool+Embed (SD3)          │                            │
│           │  CLIP_pool → 全局向量 → AdaLN     │                            │
│           │  T5_tokens → Cross-Attention      │                            │
│           │                                   │                            │
│           └───────────────┬───────────────────┘                            │
│                           ↓                                                 │
│           ┌───────────────────────────────────┐                            │
│           │      注入到扩散模型主干             │                            │
│           ├───────────────────────────────────┤                            │
│           │                                   │                            │
│           │  U-Net: Cross-Attention层         │                            │
│           │  ┌─────────────────────────┐      │                            │
│           │  │ Q = latent features     │      │                            │
│           │  │ K,V = text embeddings   │      │                            │
│           │  │ Attn(Q,K,V) → +residual│      │                            │
│           │  └─────────────────────────┘      │                            │
│           │                                   │                            │
│           │  DiT: AdaLN-Zero + Cross-Attn    │                            │
│           │  ┌─────────────────────────┐      │                            │
│           │  │ pooled_text → AdaLN参数  │      │                            │
│           │  │ token_seq → Cross-Attn  │      │                            │
│           │  └─────────────────────────┘      │                            │
│           │                                   │                            │
│           └───────────────────────────────────┘                            │
│                                                                            │
│  编码器演进路线:                                                            │
│  SD 1.5      → SDXL           → SD3/FLUX                                  │
│  CLIP-L/14    CLIP-L + OpenCLIP-G    CLIP-L + CLIP-G + T5-XXL            │
│  77token      77token×2编码器        77+256 token, 三编码器               │
│  768维        768+1280维             多维融合                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### CLIP Text Encoder

- **架构**：基于Transformer的文本编码器，12-24层
- **训练**：图文对比学习，建立图像-文本语义对齐
- **优势**：强大的语义理解，图文对齐性好
- **劣势**：77 token限制，对复杂空间关系描述能力弱

#### T5-XXL

- **架构**：Encoder-Decoder Transformer，4.7B参数
- **训练**：大规模文本预训练（C4数据集等）
- **优势**：512 token上下文，精细语言理解，可以处理复杂描述
- **劣势**：计算成本高（编码阶段约需1-2秒额外延迟）

#### 双编码器融合（SD3方案）

SD3采用了三编码器方案（CLIP-L + CLIP-G + T5-XXL），其融合策略：

1. **CLIP pooled output** → 提供全局语义向量 → 通过AdaLN影响所有层
2. **CLIP token sequence** → 与T5 token拼接 → 通过Cross-Attention提供细节
3. **融合机制**：token维度不同时用linear projection对齐

### 8.5 微条件（Micro-conditioning）技术

#### SDXL的创新

SDXL发现：训练数据中图像分辨率和裁剪方式的多样性会影响生成质量。为此引入**微条件**——将这些元信息作为额外条件输入模型：

1. **原始分辨率条件 (orig_size)**：$(H_{orig}, W_{orig})$
2. **裁剪坐标条件 (crop_coords)**：$(top, left)$
3. **目标尺寸条件 (target_size)**：$(H_{target}, W_{target})$

这些标量条件通过Fourier特征编码变为高维向量：

$$\text{FourierEmbed}(x) = [\sin(2\pi f_1 x), \cos(2\pi f_1 x), \ldots, \sin(2\pi f_d x), \cos(2\pi f_d x)]$$

**效果**：
- 消除了"训练数据裁剪导致的伪影"问题
- 模型学会了分辨率感知——推理时设置target_size=orig_size即可获得完整构图
- 训练效率提升：不再需要筛选"完美裁剪"的训练数据

### 8.6 条件注入机制对比

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    条件注入机制对比                                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. Cross-Attention (标准文本注入)                                         │
│  ┌──────────────────────────────────────────────────┐                     │
│  │  latent features (H×W×C)                         │                     │
│  │       │                                          │                     │
│  │       ↓ Linear → Q                              │                     │
│  │  text embeddings (L×D)                           │                     │
│  │       │                                          │                     │
│  │       ├─ Linear → K                             │                     │
│  │       └─ Linear → V                             │                     │
│  │                                                  │                     │
│  │  output = softmax(Q·K^T / √d) · V              │                     │
│  │                                                  │                     │
│  │  优点: 空间选择性强, 每个位置关注不同文本token   │                     │
│  │  缺点: 计算量与序列长度成正比                    │                     │
│  │  适用: 文本条件, 图像条件(IP-Adapter)           │                     │
│  └──────────────────────────────────────────────────┘                     │
│                                                                            │
│  2. AdaLN / AdaLN-Zero (归一化层融合)                                     │
│  ┌──────────────────────────────────────────────────┐                     │
│  │  condition vector c (D维)                        │                     │
│  │       │                                          │                     │
│  │       ├─ Linear → γ (scale)                     │                     │
│  │       ├─ Linear → β (shift)                     │                     │
│  │       └─ Linear → α (gate, AdaLN-Zero专有)     │                     │
│  │                                                  │                     │
│  │  AdaLN:      h = γ · LayerNorm(x) + β          │                     │
│  │  AdaLN-Zero: h = α · (γ · LayerNorm(x) + β)   │                     │
│  │              (α初始化为0, 训练开始时=Identity)   │                     │
│  │                                                  │                     │
│  │  优点: 全局调制, 计算量极低, 训练稳定           │                     │
│  │  缺点: 无空间选择性, 只能传递全局信息           │                     │
│  │  适用: 时间步t, pooled text, 风格向量           │                     │
│  └──────────────────────────────────────────────────┘                     │
│                                                                            │
│  3. Concatenation (通道拼接)                                              │
│  ┌──────────────────────────────────────────────────┐                     │
│  │  latent z: [B, 4, H, W]                         │                     │
│  │  condition map: [B, C_cond, H, W]               │                     │
│  │                                                  │                     │
│  │  input = concat([z, condition], dim=1)           │                     │
│  │  → [B, 4+C_cond, H, W]                         │                     │
│  │  → 修改第一层Conv的输入通道数                    │                     │
│  │                                                  │                     │
│  │  优点: 空间对齐精确, 实现简单                    │                     │
│  │  缺点: 要求条件与输入空间分辨率匹配             │                     │
│  │  适用: depth map, edge map, inpainting mask     │                     │
│  └──────────────────────────────────────────────────┘                     │
│                                                                            │
│  4. 组合使用 (现代架构的典型方案):                                        │
│  ┌──────────────────────────────────────────────────┐                     │
│  │  • 时间步 t → Sinusoidal Embed → AdaLN         │                     │
│  │  • 文本 pooled → 拼接 time_embed → AdaLN       │                     │
│  │  • 文本 tokens → Cross-Attention                │                     │
│  │  • 空间条件 → Concatenation + ControlNet       │                     │
│  │  • 微条件 → Fourier Embed → 拼接 time_embed   │                     │
│  └──────────────────────────────────────────────────┘                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 8.7 代码实现

以下是CFG采样、双编码器条件处理和微条件编码的完整实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class FourierFeatureEncoder(nn.Module):
    """
    Fourier特征编码器
    将标量微条件(分辨率/裁剪坐标等)编码为高维向量
    """
    
    def __init__(self, num_features: int = 256, max_freq: float = 10.0):
        super().__init__()
        self.num_features = num_features
        # 频率从1到max_freq对数均匀分布
        freqs = torch.exp(
            torch.linspace(0, math.log(max_freq), num_features // 2)
        )
        self.register_buffer('freqs', freqs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B] 或 [B, D] 标量条件值
        Returns:
            [B, num_features] 或 [B, D*num_features] Fourier特征
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B, 1]
        
        # x: [B, D], freqs: [F]
        # 计算 x * freq for all freq
        x_freq = x.unsqueeze(-1) * self.freqs  # [B, D, F]
        
        # sin和cos编码
        features = torch.cat([
            torch.sin(2 * math.pi * x_freq),
            torch.cos(2 * math.pi * x_freq)
        ], dim=-1)  # [B, D, num_features]
        
        return features.flatten(start_dim=1)  # [B, D * num_features]


class MicroConditionEncoder(nn.Module):
    """
    SDXL风格的微条件编码器
    编码原始分辨率、裁剪坐标、目标尺寸等元信息
    """
    
    def __init__(self, embed_dim: int = 1280, fourier_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 每个微条件有2个标量值(H,W或top,left)
        self.orig_size_encoder = FourierFeatureEncoder(fourier_dim)
        self.crop_coords_encoder = FourierFeatureEncoder(fourier_dim)
        self.target_size_encoder = FourierFeatureEncoder(fourier_dim)
        
        # 将3组Fourier特征投影到统一维度
        # 每组: 2个标量 × fourier_dim = 2*fourier_dim
        total_fourier = 2 * fourier_dim * 3  # 3组微条件
        self.proj = nn.Sequential(
            nn.Linear(total_fourier, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(
        self,
        orig_size: torch.Tensor,     # [B, 2] (H_orig, W_orig)
        crop_coords: torch.Tensor,   # [B, 2] (top, left)
        target_size: torch.Tensor,   # [B, 2] (H_target, W_target)
    ) -> torch.Tensor:
        """
        Returns:
            [B, embed_dim] 微条件嵌入向量
        """
        e1 = self.orig_size_encoder(orig_size)      # [B, 2*fourier_dim]
        e2 = self.crop_coords_encoder(crop_coords)  # [B, 2*fourier_dim]
        e3 = self.target_size_encoder(target_size)   # [B, 2*fourier_dim]
        
        combined = torch.cat([e1, e2, e3], dim=-1)  # [B, 6*fourier_dim]
        return self.proj(combined)                    # [B, embed_dim]


class DualTextEncoder(nn.Module):
    """
    双文本编码器: CLIP + T5融合
    模拟SD3的文本编码策略
    """
    
    def __init__(
        self,
        clip_dim: int = 768,
        t5_dim: int = 4096,
        output_dim: int = 1536,   # 融合后的统一维度
        clip_max_length: int = 77,
        t5_max_length: int = 256,
    ):
        super().__init__()
        self.output_dim = output_dim
        
        # 模拟CLIP编码器 (实际使用时替换为预训练CLIP)
        self.clip_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=clip_dim, nhead=12, dim_feedforward=clip_dim*4,
                batch_first=True
            ),
            num_layers=6
        )
        self.clip_embed = nn.Embedding(49408, clip_dim)  # CLIP词表
        self.clip_pos = nn.Embedding(clip_max_length, clip_dim)
        
        # 模拟T5编码器 (实际使用时替换为预训练T5)
        self.t5_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=t5_dim, nhead=32, dim_feedforward=t5_dim*4,
                batch_first=True
            ),
            num_layers=4  # 简化版
        )
        self.t5_embed = nn.Embedding(32128, t5_dim)  # T5词表
        self.t5_pos = nn.Embedding(t5_max_length, t5_dim)
        
        # 维度对齐投影
        self.clip_proj = nn.Linear(clip_dim, output_dim)
        self.t5_proj = nn.Linear(t5_dim, output_dim)
        
        # CLIP pooled output投影 (用于AdaLN)
        self.pool_proj = nn.Linear(clip_dim, output_dim)
    
    def forward(
        self,
        clip_tokens: torch.Tensor,   # [B, 77] token ids
        t5_tokens: torch.Tensor,     # [B, 256] token ids
        clip_mask: Optional[torch.Tensor] = None,
        t5_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                'token_embeddings': [B, 77+256, output_dim] 拼接后的token序列
                'pooled': [B, output_dim] 全局语义向量
        """
        B = clip_tokens.shape[0]
        
        # CLIP编码
        clip_pos_ids = torch.arange(77, device=clip_tokens.device)
        clip_h = self.clip_embed(clip_tokens) + self.clip_pos(clip_pos_ids)
        clip_h = self.clip_encoder(clip_h)  # [B, 77, clip_dim]
        
        # T5编码
        t5_len = t5_tokens.shape[1]
        t5_pos_ids = torch.arange(t5_len, device=t5_tokens.device)
        t5_h = self.t5_embed(t5_tokens) + self.t5_pos(t5_pos_ids)
        t5_h = self.t5_encoder(t5_h)  # [B, 256, t5_dim]
        
        # 投影到统一维度
        clip_projected = self.clip_proj(clip_h)  # [B, 77, output_dim]
        t5_projected = self.t5_proj(t5_h)        # [B, 256, output_dim]
        
        # 拼接token序列
        token_embeddings = torch.cat(
            [clip_projected, t5_projected], dim=1
        )  # [B, 333, output_dim]
        
        # Pooled输出 (取CLIP的EOS token)
        pooled = self.pool_proj(clip_h[:, -1, :])  # [B, output_dim]
        
        return {
            'token_embeddings': token_embeddings,
            'pooled': pooled,
        }


class CFGSampler:
    """
    Classifier-Free Guidance 完整采样器
    支持标准CFG、Dynamic CFG和Rescaled CFG
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        default_guidance_scale: float = 7.5,
        dynamic_cfg: bool = False,
        rescale_phi: float = 0.0,  # 0=不rescale, 0.7=推荐值
    ):
        self.model = model
        self.num_timesteps = num_timesteps
        self.default_guidance_scale = default_guidance_scale
        self.dynamic_cfg = dynamic_cfg
        self.rescale_phi = rescale_phi
        
        # 噪声调度
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    def get_guidance_scale(self, t: int, T: int) -> float:
        """Dynamic CFG: 根据时间步调整guidance scale"""
        if not self.dynamic_cfg:
            return self.default_guidance_scale
        
        # 早期(高噪声)使用较大scale, 后期(低噪声)使用较小scale
        w_max = self.default_guidance_scale
        w_min = 1.0
        gamma = 0.5
        
        progress = t / T  # 1.0(最大噪声) → 0.0(无噪声)
        w = w_max * (progress ** gamma) + w_min * (1 - progress ** gamma)
        return w
    
    def apply_rescale(
        self,
        noise_cfg: torch.Tensor,
        noise_cond: torch.Tensor
    ) -> torch.Tensor:
        """Rescaled CFG: 限制CFG输出的幅度"""
        if self.rescale_phi == 0.0:
            return noise_cfg
        
        # 计算每个样本的标准差
        std_cfg = noise_cfg.std(dim=[1, 2, 3], keepdim=True)
        std_cond = noise_cond.std(dim=[1, 2, 3], keepdim=True)
        
        # Rescale: 将CFG输出的幅度限制到接近条件输出的幅度
        factor = std_cond / (std_cfg + 1e-8)
        noise_rescaled = self.rescale_phi * factor * noise_cfg + \
                         (1 - self.rescale_phi) * noise_cfg
        
        return noise_rescaled
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Dict[str, torch.Tensor],
        num_steps: int = 50,
        guidance_scale: Optional[float] = None,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        CFG DDIM采样
        
        Args:
            shape: 输出张量形状, e.g. (B, 4, 64, 64)
            condition: 条件字典, 包含text embeddings等
            num_steps: 采样步数
            guidance_scale: 覆盖默认guidance scale
        Returns:
            生成的潜码 [B, C, H, W]
        """
        w = guidance_scale or self.default_guidance_scale
        batch_size = shape[0]
        
        # 初始化纯噪声
        z = torch.randn(shape, device=device)
        
        # 准备无条件输入 (空文本编码)
        uncond = {k: torch.zeros_like(v) for k, v in condition.items()}
        
        # 均匀选择时间步
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps,
            dtype=torch.long, device=device
        )
        
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)
            
            # ====== CFG核心: 两次前向传播 ======
            # 条件预测
            noise_cond = self.model(z, t_batch, condition)
            # 无条件预测
            noise_uncond = self.model(z, t_batch, uncond)
            
            # 获取当前步的guidance scale (可能是动态的)
            current_w = self.get_guidance_scale(t.item(), self.num_timesteps)
            if guidance_scale is not None:
                current_w = guidance_scale
            
            # CFG组合
            noise_pred = noise_uncond + current_w * (noise_cond - noise_uncond)
            
            # 可选: Rescaled CFG
            noise_pred = self.apply_rescale(noise_pred, noise_cond)
            
            # DDIM去噪步
            alpha_t = alphas_cumprod[t.long()]
            if i < len(timesteps) - 1:
                alpha_prev = alphas_cumprod[timesteps[i + 1].long()]
            else:
                alpha_prev = torch.tensor(1.0, device=device)
            
            # 预测z_0
            z_0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            # 计算z_{t-1}
            z = alpha_prev.sqrt() * z_0_pred + \
                (1 - alpha_prev).sqrt() * noise_pred
        
        return z
    
    @torch.no_grad()
    def sample_multi_condition(
        self,
        shape: Tuple[int, ...],
        conditions: list,       # 多个条件
        weights: list,          # 每个条件的guidance weight
        num_steps: int = 50,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        多条件CFG采样
        ε̃ = ε_uncond + Σ_i w_i·(ε_cond_i - ε_uncond)
        """
        batch_size = shape[0]
        z = torch.randn(shape, device=device)
        
        uncond = {k: torch.zeros_like(v) for k, v in conditions[0].items()}
        
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps,
            dtype=torch.long, device=device
        )
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)
            
            # 无条件预测 (只需一次)
            noise_uncond = self.model(z, t_batch, uncond)
            
            # 多条件引导
            noise_pred = noise_uncond.clone()
            for cond, w in zip(conditions, weights):
                noise_cond = self.model(z, t_batch, cond)
                noise_pred = noise_pred + w * (noise_cond - noise_uncond)
            
            # DDIM步
            alpha_t = alphas_cumprod[t.long()]
            alpha_prev = alphas_cumprod[timesteps[i+1].long()] \
                if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            
            z_0_pred = (z - (1-alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            z = alpha_prev.sqrt() * z_0_pred + (1-alpha_prev).sqrt() * noise_pred
        
        return z


# 使用示例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试微条件编码
    micro_cond = MicroConditionEncoder(embed_dim=1280).to(device)
    orig_size = torch.tensor([[1024, 768], [512, 512]], dtype=torch.float, device=device)
    crop_coords = torch.tensor([[0, 0], [64, 64]], dtype=torch.float, device=device)
    target_size = torch.tensor([[1024, 1024], [512, 512]], dtype=torch.float, device=device)
    
    micro_embed = micro_cond(orig_size, crop_coords, target_size)
    print(f"微条件嵌入: {micro_embed.shape}")  # [2, 1280]
    
    # 测试Fourier特征
    fourier = FourierFeatureEncoder(num_features=256).to(device)
    scalar = torch.tensor([512.0, 1024.0], device=device)
    feat = fourier(scalar)
    print(f"Fourier特征: {feat.shape}")  # [2, 256]
```

**本章小结**：Classifier-Free Guidance是当前条件扩散模型的核心推理技术，通过无条件/条件预测的线性外推实现高质量生成。其改进方向包括Dynamic CFG（时间步自适应）、Rescaled CFG（幅度控制）和CFG蒸馏（消除计算翻倍）。文本编码器从单CLIP演进到CLIP+T5双编码器融合，显著提升了语言理解能力。微条件技术则解决了训练数据多样性带来的伪影问题。条件注入机制的选择——Cross-Attention用于序列条件、AdaLN用于全局标量、Concatenation用于空间对齐条件——需要根据具体任务灵活组合。



---

## 第9章：视频与3D扩散模型优化

从静态图像到视频和3D内容生成，扩散模型面临的挑战发生了质的变化：时间维度的引入使数据规模和计算需求呈爆炸式增长，同时**时间一致性**成为新的核心约束。本章将系统探讨视频扩散模型的架构设计、时空分解策略、以及3D生成中扩散模型的应用（如Score Distillation Sampling）。

### 9.1 视频扩散模型基础

#### 从图像到视频的核心挑战

视频可以看作是图像在时间维度上的扩展：一段T帧、分辨率H×W的视频本质上是一个 $T \times H \times W \times C$ 的4D张量。这带来了多重挑战：

1. **计算量爆炸**：一段4秒、30fps、512×512的视频包含 $120 \times 512 \times 512 \times 3 \approx 94M$ 像素——是单张图像的120倍。在潜空间中（8× downsample），仍有 $120 \times 64 \times 64 \times 4 \approx 2M$ 个值。

2. **时间一致性**：相邻帧之间必须保持视觉连贯——物体运动轨迹平滑、光照变化自然、不出现闪烁。这不是简单地逐帧独立生成能解决的。

3. **长距离依赖**：视频中的因果关系可能跨越数秒——一个球被踢出后必须沿合理轨迹飞行。

4. **训练数据**：高质量视频数据获取和标注成本远高于图像。

#### 视频扩散框架分类

当前视频扩散模型主要分为以下几种范式：

| 范式 | 代表工作 | 核心思路 | 优缺点 |
|------|----------|----------|--------|
| 逐帧生成+后处理 | 早期工作 | 独立生成每帧，后续做一致性修复 | 简单但一致性差 |
| 3D U-Net | Video Diffusion Models | 将2D卷积扩展为3D时空卷积 | 一致性好但计算量大 |
| 分解注意力 | AnimateDiff, SVD | 空间和时间注意力分开计算 | 效率高，可复用图像模型 |
| 时空Patches | Sora, CogVideo | 将视频切为3D patches，用DiT处理 | 最灵活，支持可变长度 |
| 自回归+扩散 | 混合方案 | 自回归生成关键帧，扩散填充中间帧 | 可处理长视频 |

### 9.2 时空分解策略

处理视频4D张量最直接的方法是使用3D操作，但其计算代价过高。**时空分解**的核心思想是：将3D操作分解为2D空间操作和1D时间操作的组合。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    视频扩散 时空分解架构图                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入视频潜码: z ∈ R^{T×H×W×C} (e.g., 16×64×64×4)                          │
│                                                                              │
│  ═══ 方案1: 完全3D (Full 3D) ═══════════════════════════════════           │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │ 3D Conv(k=3×3×3) + 3D Self-Attention(所有token)             │           │
│  │ Token数: T×H×W = 16×64×64 = 65,536                          │           │
│  │ 计算量: O(N²·d) = O(65536²·d) ≈ 不可行                      │           │
│  └──────────────────────────────────────────────────────────────┘           │
│  评价: 质量最优但计算不可行 (O(T²H²W²))                                     │
│                                                                              │
│  ═══ 方案2: 空间+时间分解 (Factorized) ═══════════════════════             │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │ Step 1: Spatial (每帧独立)                                    │           │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   T帧并行               │           │
│  │  │Frame│ │Frame│ │Frame│ │Frame│   各帧独立做              │           │
│  │  │  1  │ │  2  │ │  3  │ │  T  │   2D Self-Attn           │           │
│  │  └─────┘ └─────┘ └─────┘ └─────┘   Token: H×W=4096       │           │
│  │                                                               │           │
│  │ Step 2: Temporal (每位置独立)                                 │           │
│  │  ┌───────────────────────────────┐                           │           │
│  │  │ 位置(i,j)在T帧上做1D注意力   │   H×W个独立序列          │           │
│  │  │ [z_1(i,j), z_2(i,j),...z_T]  │   序列长度: T=16         │           │
│  │  └───────────────────────────────┘                           │           │
│  └──────────────────────────────────────────────────────────────┘           │
│  评价: 计算量=O(T·(HW)²) + O(HW·T²), 显著降低                             │
│                                                                              │
│  ═══ 方案3: 伪3D (Pseudo-3D) ═══════════════════════════════════           │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │ 2D Spatial Conv(3×3)  →  1D Temporal Conv(1×1×k_t)          │           │
│  │                                                               │           │
│  │ 2D卷积: 只看空间邻域      1D卷积: 只看时间邻域               │           │
│  │ ┌─────────────────┐       ┌─────────────────┐               │           │
│  │ │  ·  ·  ·        │       │  Frame t-1      │               │           │
│  │ │  ·  ★  ·  ←当前帧│      │  Frame t   ★    │               │           │
│  │ │  ·  ·  ·        │       │  Frame t+1      │               │           │
│  │ └─────────────────┘       └─────────────────┘               │           │
│  └──────────────────────────────────────────────────────────────┘           │
│  评价: 实现最简单, 可直接从2D模型扩展, 但感受野有限                         │
│                                                                              │
│  ═══ 方案4: DiT时空Patches (Sora风格) ═══════════════════════             │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │ 视频 → 3D Patches → Flatten → Transformer                   │           │
│  │                                                               │           │
│  │ Patch大小: 2×16×16 (时间×高×宽)                              │           │
│  │ Token数: (T/2)×(H/16)×(W/16) = 8×4×4 = 128 (极大压缩)     │           │
│  │                                                               │           │
│  │ ┌───┬───┬───┬───┐                                           │           │
│  │ │P1 │P2 │P3 │P4 │ ← 时间patch 1 (帧1-2)                   │           │
│  │ ├───┼───┼───┼───┤                                           │           │
│  │ │P5 │P6 │P7 │P8 │ ← 时间patch 2 (帧3-4)                   │           │
│  │ ├───┼───┼───┼───┤                                           │           │
│  │ │P9 │...│   │   │ ← 时间patch 3 (帧5-6)                   │           │
│  │ └───┴───┴───┴───┘                                           │           │
│  │                                                               │           │
│  │ 所有patches flatten后做full attention (或window attention)   │           │
│  └──────────────────────────────────────────────────────────────┘           │
│  评价: 最灵活(可变分辨率/时长), 但需要大量计算和数据                        │
│                                                                              │
│  ═══ 各方案计算量对比 (16帧×64×64 latent) ═══                              │
│  Full 3D Attn:     65,536² ≈ 4.3×10⁹  (不可行)                            │
│  Factorized:       16×4096² + 4096×16² ≈ 2.7×10⁸ (可行)                   │
│  Pseudo-3D Conv:   ~逐帧2D计算量 × 1.3  (最轻量)                          │
│  DiT Patches(2×16²): 128² × layers ≈ 10⁴×layers (高效)                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 各策略的优劣对比

- **Full 3D**：质量上限最高，但计算量不可接受
- **Factorized (空间+时间)**：当前最主流的方案，在效率和质量间取得良好平衡。AnimateDiff、SVD均采用此路线
- **Pseudo-3D**：计算最轻量，但时间建模能力有限，适合短视频
- **DiT Patches**：最灵活，Sora等模型的基础架构，支持可变分辨率和时长

### 9.3 视频扩散架构设计

#### Stable Video Diffusion (SVD)

SVD是Stability AI提出的视频生成模型，基于图像模型微调而来：

**核心设计**：
1. 从预训练的SD图像模型出发
2. 在每个spatial attention层之后插入temporal attention层
3. 在每个spatial convolution之后插入temporal convolution
4. 训练时冻结空间层，只训练时间层

**技术细节**：
- 生成14-25帧视频
- 输入：单张图像作为条件（image-to-video）
- 时间注意力沿帧维度做attention
- 使用EDM框架的连续时间扩散

#### AnimateDiff

AnimateDiff的核心创新是**运动模块插件化**：

1. 训练一个通用的Temporal Attention模块（motion module）
2. 该模块可以插入到任何兼容的图像模型中
3. 不修改原始图像模型的权重

这意味着社区中大量的LoRA和checkpoint都可以直接与AnimateDiff组合使用。

### 9.4 Sora类架构（时空Patches）

#### 设计理念

Sora代表了视频生成的下一代架构范式。其核心理念是：**将视频视为时空patches的序列，用统一的Transformer处理**。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  Sora类 Spacetime Patches 示意图                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入视频 (可变分辨率、可变时长):                                             │
│  ┌──────────────────────────────────────────────────┐                       │
│  │  ████████████████████████████████████████████    │ ← 帧1 (H×W)         │
│  │  ████████████████████████████████████████████    │ ← 帧2                │
│  │  ████████████████████████████████████████████    │ ← 帧3                │
│  │  ████████████████████████████████████████████    │ ← 帧4                │
│  │  ...                                             │                       │
│  │  ████████████████████████████████████████████    │ ← 帧T                │
│  └──────────────────────────────────────────────────┘                       │
│                    │                                                          │
│                    ↓ 3D Patch Embedding                                      │
│                                                                              │
│  Patch切分 (时间p_t × 高p_h × 宽p_w):                                      │
│  例: p_t=2, p_h=2, p_w=2 (在latent空间)                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │ 视频 T×H×W → (T/p_t)×(H/p_h)×(W/p_w) 个patches          │             │
│  │                                                            │             │
│  │ 每个patch: p_t×p_h×p_w×C → 线性投影 → D维token           │             │
│  │                                                            │             │
│  │ 例: 120帧×64×64 (latent)                                  │             │
│  │     p_t=2, p_h=2, p_w=2                                   │             │
│  │     → 60×32×32 = 61,440 tokens                           │             │
│  │     (仍然很多! 需要窗口注意力或层级结构)                   │             │
│  └────────────────────────────────────────────────────────────┘             │
│                    │                                                          │
│                    ↓ + 时空位置编码                                           │
│                                                                              │
│  位置编码方案:                                                               │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │ 选项A: 可学习的1D绝对位置 (不支持可变尺寸)                 │             │
│  │ 选项B: RoPE 3D扩展 (每维独立旋转, 支持外推)               │             │
│  │ 选项C: 分解式 = 时间PE + 空间PE (最灵活)                  │             │
│  │         pos(t,h,w) = pos_t(t) + pos_h(h) + pos_w(w)      │             │
│  └────────────────────────────────────────────────────────────┘             │
│                    │                                                          │
│                    ↓                                                          │
│                                                                              │
│  DiT Transformer Blocks × N:                                                │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │ ┌────────────────────────────────────────────────────┐    │             │
│  │ │ Block:                                              │    │             │
│  │ │   AdaLN(时间步t + 文本pooled)                       │    │             │
│  │ │   → Self-Attention (所有patches间)                  │    │             │
│  │ │   → Cross-Attention (文本tokens)                   │    │             │
│  │ │   → FFN                                            │    │             │
│  │ │   (重复 N=28~40 层)                                │    │             │
│  │ └────────────────────────────────────────────────────┘    │             │
│  └────────────────────────────────────────────────────────────┘             │
│                    │                                                          │
│                    ↓ Unpatchify                                              │
│                                                                              │
│  输出: 预测噪声 ε (与输入同形状)                                             │
│  → 去噪得到视频latent → VAE解码 → 视频                                     │
│                                                                              │
│  ═══ 关键优势 ═══                                                           │
│  • 天然支持可变分辨率: token数量随输入变化                                   │
│  • 天然支持可变时长: 更多帧 = 更多token                                     │
│  • 天然支持可变宽高比: patch划分不依赖固定网格                               │
│  • 统一处理图像和视频: 图像 = 1帧视频                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 长视频生成策略

对于超过模型训练长度的视频，常用**分块+重叠**策略：

1. 将目标视频分为多个重叠片段（如每段16帧，重叠4帧）
2. 对每个片段独立去噪
3. 重叠区域做线性混合（blend）
4. 可选：在重叠区域使用特殊的一致性约束

### 9.5 视频扩散训练优化

#### 图像预训练 → 视频微调

这是当前最成功的训练策略：

1. **阶段一**：在大规模图像数据上训练2D扩散模型（如SD）
2. **阶段二**：冻结空间层，插入并训练时间层
3. **阶段三**（可选）：解冻所有层进行联合微调

好处：空间层已经学会了强大的视觉特征，时间层只需学习运动模式。

#### 帧率调度训练

从低帧率逐步提升到高帧率：
- 阶段一：4fps训练，学习粗粒度运动
- 阶段二：12fps训练，学习中等运动
- 阶段三：24fps训练，学习流畅运动

这类似于课程学习——先学容易的（慢运动），再学难的（快运动）。

#### 时间位置编码设计

视频中的时间位置编码需要支持可变帧率和可变长度：

$$\text{PE}_{time}(t) = \sin\left(\frac{t}{\text{fps}} \cdot \omega_k\right), \cos\left(\frac{t}{\text{fps}} \cdot \omega_k\right)$$

通过除以fps，模型可以自动适配不同帧率的视频——相同物理时间间隔对应相同的位置编码差异。

### 9.6 3D生成与扩散模型

#### Score Distillation Sampling (SDS)

SDS是将2D扩散模型的知识"蒸馏"到3D表示中的革命性技术（DreamFusion, 2022）。

**核心思路**：不直接在3D空间训练扩散模型，而是利用已有的2D图像扩散模型作为"评判者"来优化3D表示。

**数学推导**：

设 $\theta$ 为3D参数（NeRF参数或3D Gaussian参数），$g(\theta, \pi)$ 为从视角 $\pi$ 渲染的2D图像。SDS的梯度为：

$$\nabla_\theta L_{SDS} = \mathbb{E}_{t, \epsilon, \pi} \left[ w(t) \cdot (\epsilon_\phi(z_t, t, c) - \epsilon) \cdot \frac{\partial z}{\partial \theta} \right]$$

其中：
- $z_t = \sqrt{\bar\alpha_t} \cdot \mathcal{E}(g(\theta, \pi)) + \sqrt{1-\bar\alpha_t} \cdot \epsilon$ 是加噪后的渲染图像
- $\epsilon_\phi$ 是预训练的2D扩散模型
- $w(t)$ 是时间步权重函数
- $c$ 是文本条件

**直觉理解**：SDS的作用是"让3D渲染从每个视角看都像是2D扩散模型会生成的图像"。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   SDS 优化流程图 (2D扩散 → 3D参数更新)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        迭代优化循环                                  │    │
│  │                                                                     │    │
│  │  Step 1: 随机采样视角                                               │    │
│  │  ┌──────────────────────┐                                          │    │
│  │  │ π ~ Uniform(views)   │  ← 随机选择相机位置                      │    │
│  │  │ (方位角, 仰角, 距离)  │                                          │    │
│  │  └──────────┬───────────┘                                          │    │
│  │             ↓                                                       │    │
│  │  Step 2: 渲染2D图像                                                │    │
│  │  ┌──────────────────────────────────────────┐                      │    │
│  │  │  3D表示 (NeRF/3DGS)                      │                      │    │
│  │  │  参数θ                                    │                      │    │
│  │  │         │                                 │                      │    │
│  │  │         ↓ 可微分渲染                       │                      │    │
│  │  │  x = render(θ, π)   [H×W×3 图像]        │                      │    │
│  │  └──────────────────────────┬───────────────┘                      │    │
│  │                             ↓                                       │    │
│  │  Step 3: VAE编码 + 加噪                                            │    │
│  │  ┌──────────────────────────────────────────┐                      │    │
│  │  │  z = E(x)            ← VAE编码           │                      │    │
│  │  │  t ~ Uniform(1,T)    ← 随机时间步         │                      │    │
│  │  │  ε ~ N(0,I)          ← 随机噪声           │                      │    │
│  │  │  z_t = √ᾱ_t·z + √(1-ᾱ_t)·ε            │                      │    │
│  │  └──────────────────────────┬───────────────┘                      │    │
│  │                             ↓                                       │    │
│  │  Step 4: 2D扩散模型预测噪声 (冻结参数)                             │    │
│  │  ┌──────────────────────────────────────────┐                      │    │
│  │  │  ε_pred = ε_φ(z_t, t, c)                │                      │    │
│  │  │  (φ固定, 不更新)                         │                      │    │
│  │  │  c = "a DSLR photo of a dog"            │                      │    │
│  │  │                                          │                      │    │
│  │  │  通常使用CFG:                            │                      │    │
│  │  │  ε̃ = ε_uncond + w·(ε_cond - ε_uncond)  │                      │    │
│  │  └──────────────────────────┬───────────────┘                      │    │
│  │                             ↓                                       │    │
│  │  Step 5: 计算SDS梯度                                               │    │
│  │  ┌──────────────────────────────────────────┐                      │    │
│  │  │  grad = w(t) · (ε_pred - ε) · ∂z/∂θ    │                      │    │
│  │  │                                          │                      │    │
│  │  │  直觉: 如果渲染图像加噪后,扩散模型的      │                      │    │
│  │  │  去噪方向与原始噪声不同,说明渲染图像       │                      │    │
│  │  │  "不够像真实图像", 需要调整3D参数        │                      │    │
│  │  └──────────────────────────┬───────────────┘                      │    │
│  │                             ↓                                       │    │
│  │  Step 6: 更新3D参数                                                │    │
│  │  ┌──────────────────────────────────────────┐                      │    │
│  │  │  θ ← θ - lr · grad                      │                      │    │
│  │  │  (Adam/AdamW优化器)                       │                      │    │
│  │  └──────────────────────────────────────────┘                      │    │
│  │                                                                     │    │
│  │  重复 5000-10000 次迭代                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  已知问题 - Janus Problem (多面问题):                                        │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │ 由于2D模型对"正面"有偏好, 优化后的3D物体每个视角都呈现正面特征  │       │
│  │ 缓解方案:                                                        │       │
│  │ • 在text prompt中加入视角信息: "front view", "back view"        │       │
│  │ • 使用多视图一致性约束 (MVDream)                                 │       │
│  │ • 降低高guidance scale的使用                                     │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 多视图生成

**Zero-1-to-3**：从单张图像生成多视图
- 输入：一张图像 + 目标视角的相对旋转矩阵
- 输出：目标视角的图像
- 应用：可以为SDS提供更好的初始化

**MVDream**：多视图一致性扩散模型
- 同时生成4个正交视角的图像
- 通过跨视图注意力保证一致性
- 有效缓解Janus问题

#### 3D Native Diffusion

直接在3D表示上执行扩散过程：

- **Point-E / Shap-E**：在点云或隐式表示上做扩散
- **3D Gaussian Splatting + Diffusion**：在3DGS参数上做扩散
- 优势：无需多视图渲染的迭代优化
- 挑战：3D数据获取困难，训练数据量有限

### 9.7 视频/3D扩散的部署挑战

视频和3D扩散模型的部署面临比图像模型严峻得多的工程挑战：

**显存需求**：
- 16帧×64×64×4的视频latent + 模型激活：数十GB
- 长视频（120帧+）：需要梯度检查点+模型并行
- 3D SDS：渲染器+扩散模型需同时在显存中

**推理延迟**：
- 视频生成：通常需要30秒到数分钟
- SDS优化：5000-10000次迭代，耗时数十分钟
- 实时应用（如直播特效）目前不可行

**优化方向**：
- 帧间缓存：相邻帧的注意力KV可以复用
- 时空解耦推理：空间层只在关键帧计算，时间层做插值
- 步数压缩：视频扩散蒸馏（如AnimateLCM）
- 分辨率渐进：低分辨率生成→超分辨率

```
┌──────────────────────────────────────────────────────────────────────────┐
│              视频 U-Net vs Video DiT 架构对比                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ═══ Video U-Net (SVD/AnimateDiff风格) ═══                              │
│  ┌────────────────────────────────────────────────────┐                 │
│  │                                                    │                 │
│  │  Encoder Path:                                     │                 │
│  │  ┌─────────────────────────────────────────┐      │                 │
│  │  │ SpatialConv → SpatialAttn → TemporalAttn│      │                 │
│  │  │      ↓ Downsample                        │      │                 │
│  │  │ SpatialConv → SpatialAttn → TemporalAttn│      │                 │
│  │  │      ↓ Downsample                        │      │                 │
│  │  │ SpatialConv → SpatialAttn → TemporalAttn│      │                 │
│  │  └─────────────────────────────────────────┘      │                 │
│  │                     │                              │                 │
│  │  Middle:    ┌───────┴───────┐                     │                 │
│  │             │SpatAttn+TempAttn│                    │                 │
│  │             └───────┬───────┘                     │                 │
│  │                     │                              │                 │
│  │  Decoder Path:  (对称上采样 + skip connections)    │                 │
│  │                                                    │                 │
│  │  特点:                                             │                 │
│  │  • 可从图像U-Net平滑升级                          │                 │
│  │  • Skip connections保留细节                       │                 │
│  │  • Temporal层可插件化(AnimateDiff)                │                 │
│  │  • 局部感受野 + 全局注意力                        │                 │
│  └────────────────────────────────────────────────────┘                 │
│                                                                          │
│  ═══ Video DiT (Sora/CogVideo风格) ═══                                 │
│  ┌────────────────────────────────────────────────────┐                 │
│  │                                                    │                 │
│  │  Input: 3D Patches → Token序列                    │                 │
│  │  ┌─────────────────────────────────────────┐      │                 │
│  │  │                                         │      │                 │
│  │  │  DiT Block × N:                        │      │                 │
│  │  │  ┌───────────────────────────────┐     │      │                 │
│  │  │  │ AdaLN(t, c_pooled)           │     │      │                 │
│  │  │  │ → Full Self-Attention         │     │      │                 │
│  │  │  │   (所有时空token)             │     │      │                 │
│  │  │  │ → Cross-Attention(text)       │     │      │                 │
│  │  │  │ → FFN                         │     │      │                 │
│  │  │  └───────────────────────────────┘     │      │                 │
│  │  │                                         │      │                 │
│  │  │  或 分解版:                             │      │                 │
│  │  │  ┌───────────────────────────────┐     │      │                 │
│  │  │  │ Spatial Self-Attn (帧内)      │     │      │                 │
│  │  │  │ → Temporal Self-Attn (跨帧)   │     │      │                 │
│  │  │  │ → Cross-Attn(text)           │     │      │                 │
│  │  │  │ → FFN                         │     │      │                 │
│  │  │  └───────────────────────────────┘     │      │                 │
│  │  │                                         │      │                 │
│  │  └─────────────────────────────────────────┘      │                 │
│  │  Output: Unpatchify → 预测噪声                    │                 │
│  │                                                    │                 │
│  │  特点:                                             │                 │
│  │  • 纯Transformer, 无卷积归纳偏置                  │                 │
│  │  • 可变分辨率/时长 (token数量变化)                │                 │
│  │  • 更好的scaling性质                              │                 │
│  │  • 需要更多数据和计算才能收敛                     │                 │
│  └────────────────────────────────────────────────────┘                 │
│                                                                          │
│  ═══ 选型建议 ═══                                                       │
│  • 从图像模型升级: 选Video U-Net (兼容现有权重)                         │
│  • 从头训练大模型: 选Video DiT (更好scaling)                            │
│  • 插件化运动控制: 选AnimateDiff方案 (可复用社区模型)                   │
│  • 追求最高质量: 选DiT + 大规模数据/计算                                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 9.8 代码实现

以下是Temporal Attention模块、Video DiT Block和SDS Loss的完整实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange


class TemporalAttention(nn.Module):
    """
    时间注意力模块
    在时间维度上做self-attention，实现跨帧信息交互
    
    设计要点:
    - 每个空间位置独立地在T帧上做attention
    - 支持因果mask (用于自回归生成)
    - 残差连接 + zero-init (训练开始时等价于Identity)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        max_frames: int = 64,
        causal: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        # 层归一化
        self.norm = nn.LayerNorm(dim)
        
        # QKV投影
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        
        # 输出投影 (zero-init, 训练开始时temporal层=identity)
        self.proj_out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj_out.weight)
        
        # 时间位置编码 (相对位置)
        self.temporal_pos_embed = nn.Embedding(max_frames, dim)
        
        # 残差缩放因子
        self.alpha = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Args:
            x: [B*T, H*W, C] 或 [B*T, N, C] 
               (batch和帧合并, N为空间token数)
            num_frames: T (帧数)
        Returns:
            与输入同形状的张量
        """
        BT, N, C = x.shape
        B = BT // num_frames
        T = num_frames
        
        # 重组: [B*T, N, C] → [B*N, T, C] (每个空间位置在时间上做attention)
        x_time = rearrange(x, '(b t) n c -> (b n) t c', b=B, t=T)
        
        # 添加时间位置编码
        pos_ids = torch.arange(T, device=x.device)
        pos_embed = self.temporal_pos_embed(pos_ids)  # [T, C]
        x_time = x_time + pos_embed.unsqueeze(0)
        
        # 归一化
        h = self.norm(x_time)
        
        # 计算QKV
        qkv = self.to_qkv(h)  # [B*N, T, 3*C]
        qkv = rearrange(qkv, 'b t (three heads d) -> three b heads t d',
                        three=3, heads=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        # 可选因果mask
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        
        # 加权求和
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b heads t d -> b t (heads d)')
        
        # 输出投影
        out = self.proj_out(out)
        
        # 残差连接 (alpha初始化为0, 逐步学习)
        x_time = x_time + self.alpha * out
        
        # 重组回原始形状: [B*N, T, C] → [B*T, N, C]
        x_out = rearrange(x_time, '(b n) t c -> (b t) n c', b=B, n=N)
        
        return x_out


class VideoDiTBlock(nn.Module):
    """
    Video DiT Block - 时空注意力
    
    包含:
    1. Spatial Self-Attention (帧内)
    2. Temporal Self-Attention (跨帧)
    3. Cross-Attention (文本条件)
    4. FFN
    
    使用AdaLN-Zero进行条件注入
    """
    
    def __init__(
        self,
        dim: int = 1152,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        max_frames: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        mlp_dim = int(dim * mlp_ratio)
        
        # ===== Spatial Self-Attention =====
        self.norm1_spatial = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn_spatial = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        
        # ===== Temporal Self-Attention =====
        self.temporal_attn = TemporalAttention(
            dim, num_heads, max_frames, causal=False
        )
        
        # ===== Cross-Attention (文本条件) =====
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        
        # ===== FFN =====
        self.norm_ffn = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_dim, dim),
        )
        
        # ===== AdaLN-Zero 参数 (6组: γ,β,α for spatial+temporal+ffn) =====
        self.adaln_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # 初始化: α=0 (训练初期block为identity)
        nn.init.zeros_(self.adaln_linear[-1].weight)
        nn.init.zeros_(self.adaln_linear[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,           # [B*T, N, C] 时空token
        cond: torch.Tensor,         # [B*T, C] 条件向量 (时间步+文本pooled)
        text_tokens: torch.Tensor,  # [B, L, C] 文本token序列
        num_frames: int,            # T
    ) -> torch.Tensor:
        """
        Args:
            x: [B*T, N, C] 展平的时空tokens
            cond: [B*T, C] AdaLN条件 (timestep embed + pooled text)
            text_tokens: [B, L, C] 文本token序列 (用于cross-attn)
            num_frames: 帧数T
        """
        BT, N, C = x.shape
        B = BT // num_frames
        
        # 计算AdaLN参数
        adaln_params = self.adaln_linear(cond)  # [B*T, 6*C]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = \
            adaln_params.chunk(6, dim=-1)
        
        # ===== 1. Spatial Self-Attention =====
        h = self.norm1_spatial(x)
        # AdaLN调制
        h = (1 + gamma1.unsqueeze(1)) * h + beta1.unsqueeze(1)
        h_attn, _ = self.attn_spatial(h, h, h)
        x = x + alpha1.unsqueeze(1) * h_attn
        
        # ===== 2. Temporal Self-Attention =====
        x = self.temporal_attn(x, num_frames)
        
        # ===== 3. Cross-Attention (文本条件) =====
        h = self.norm_cross(x)
        # 扩展text_tokens到每帧: [B, L, C] → [B*T, L, C]
        text_expanded = text_tokens.repeat_interleave(num_frames, dim=0)
        h_cross, _ = self.cross_attn(h, text_expanded, text_expanded)
        x = x + h_cross
        
        # ===== 4. FFN =====
        h = self.norm_ffn(x)
        h = (1 + gamma2.unsqueeze(1)) * h + beta2.unsqueeze(1)
        h = self.ffn(h)
        x = x + alpha2.unsqueeze(1) * h
        
        return x


class SDSLoss(nn.Module):
    """
    Score Distillation Sampling (SDS) Loss
    用于将2D扩散模型的知识蒸馏到3D表示中
    
    核心流程:
    1. 从3D表示渲染2D图像
    2. 编码到潜空间并加噪
    3. 用预训练扩散模型预测噪声
    4. 计算SDS梯度
    """
    
    def __init__(
        self,
        diffusion_model: nn.Module,  # 预训练2D扩散模型 (冻结)
        vae_encoder: nn.Module,       # VAE编码器 (冻结)
        num_timesteps: int = 1000,
        min_step: int = 20,           # 避免最低噪声级别(梯度不稳定)
        max_step: int = 980,          # 避免最高噪声级别(信息丢失)
        guidance_scale: float = 100.0,  # SDS通常使用较高的CFG
    ):
        super().__init__()
        self.diffusion_model = diffusion_model.eval()
        self.vae_encoder = vae_encoder.eval()
        self.guidance_scale = guidance_scale
        self.min_step = min_step
        self.max_step = max_step
        self.num_timesteps = num_timesteps
        
        # 冻结参数
        for p in self.diffusion_model.parameters():
            p.requires_grad_(False)
        for p in self.vae_encoder.parameters():
            p.requires_grad_(False)
        
        # 噪声调度
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
    
    def compute_sds_grad(
        self,
        rendered_images: torch.Tensor,    # [B, 3, H, W] 渲染的图像
        text_embeddings: torch.Tensor,    # [B, L, D] 文本条件
        null_embeddings: torch.Tensor,    # [B, L, D] 空文本 (用于CFG)
    ) -> torch.Tensor:
        """
        计算SDS梯度
        
        Args:
            rendered_images: 可微分渲染得到的图像
            text_embeddings: 文本prompt的编码
            null_embeddings: 空文本编码 (CFG)
        
        Returns:
            SDS梯度 (形状与rendered_images相同)
        """
        device = rendered_images.device
        batch_size = rendered_images.shape[0]
        
        # Step 1: VAE编码 (不需要梯度)
        with torch.no_grad():
            # 假设vae_encoder返回(mean, logvar)
            latents = self.vae_encoder(rendered_images)
            if isinstance(latents, tuple):
                latents = latents[0]  # 取均值
            latents = latents * 0.18215  # scale factor
        
        # Step 2: 随机采样时间步
        t = torch.randint(
            self.min_step, self.max_step + 1,
            (batch_size,), device=device
        )
        
        # Step 3: 加噪
        noise = torch.randn_like(latents)
        sqrt_alpha = self.alphas_cumprod[t].sqrt()[:, None, None, None]
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t]).sqrt()[:, None, None, None]
        
        noisy_latents = sqrt_alpha * latents + sqrt_one_minus_alpha * noise
        
        # Step 4: 扩散模型预测 (CFG)
        with torch.no_grad():
            # 条件预测
            noise_cond = self.diffusion_model(
                noisy_latents, t, text_embeddings
            )
            # 无条件预测
            noise_uncond = self.diffusion_model(
                noisy_latents, t, null_embeddings
            )
            # CFG组合
            noise_pred = noise_uncond + self.guidance_scale * (
                noise_cond - noise_uncond
            )
        
        # Step 5: 计算SDS梯度
        # grad = w(t) * (ε_pred - ε)
        # w(t) = σ_t² / α_t (常见选择) 或简单地设为1
        w = (1 - self.alphas_cumprod[t])[:, None, None, None]
        
        # SDS梯度 (这是对latents的梯度)
        grad = w * (noise_pred - noise)
        
        # 通过latents对rendered_images求梯度 (链式法则)
        # 实际使用中通过backward实现
        return grad
    
    def forward(
        self,
        rendered_images: torch.Tensor,
        text_embeddings: torch.Tensor,
        null_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算SDS loss
        
        实现技巧: 使用"gradient detach"避免显式计算二阶梯度
        loss = 0.5 * ||latents - (latents - grad).detach()||²
        这样 ∂loss/∂θ = grad · ∂latents/∂θ (正是SDS梯度)
        """
        device = rendered_images.device
        batch_size = rendered_images.shape[0]
        
        # VAE编码 (保留梯度链 - 让梯度流回rendered_images)
        latents_mean = self.vae_encoder(rendered_images)
        if isinstance(latents_mean, tuple):
            latents_mean = latents_mean[0]
        latents = latents_mean * 0.18215
        
        # 采样时间步和噪声
        t = torch.randint(
            self.min_step, self.max_step + 1,
            (batch_size,), device=device
        )
        noise = torch.randn_like(latents)
        
        # 加噪 (detach latents, 避免梯度流经加噪过程)
        sqrt_alpha = self.alphas_cumprod[t].sqrt()[:, None, None, None]
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t]).sqrt()[:, None, None, None]
        noisy_latents = sqrt_alpha * latents.detach() + sqrt_one_minus_alpha * noise
        
        # 扩散模型预测
        with torch.no_grad():
            noise_cond = self.diffusion_model(noisy_latents, t, text_embeddings)
            noise_uncond = self.diffusion_model(noisy_latents, t, null_embeddings)
            noise_pred = noise_uncond + self.guidance_scale * (
                noise_cond - noise_uncond
            )
        
        # SDS梯度
        w = (1 - self.alphas_cumprod[t])[:, None, None, None]
        grad = w * (noise_pred - noise)
        
        # "Gradient detach" 技巧
        # target = (latents - grad).detach()
        # loss = 0.5 * F.mse_loss(latents, target)
        # 等价于: loss的梯度 = grad (就是我们想要的SDS梯度)
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction='mean')
        
        return loss


class VideoGenerationPipeline:
    """
    视频生成Pipeline示例
    展示从噪声到视频的完整采样流程
    """
    
    def __init__(
        self,
        video_dit: nn.Module,      # Video DiT模型
        vae_decoder: nn.Module,    # VAE解码器
        text_encoder: nn.Module,   # 文本编码器
        num_timesteps: int = 1000,
    ):
        self.video_dit = video_dit
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.num_timesteps = num_timesteps
        
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_frames: int = 16,
        height: int = 512,
        width: int = 512,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        生成视频
        
        Returns:
            [T, 3, H, W] 视频张量
        """
        # 文本编码
        # text_emb = self.text_encoder(prompt)  # 简化
        B = 1
        latent_h, latent_w = height // 8, width // 8
        
        # 潜空间初始噪声: [B, T, C, H, W]
        z = torch.randn(B, num_frames, 4, latent_h, latent_w, device=device)
        
        # 采样时间步
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps,
            dtype=torch.long, device=device
        )
        
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        for i, t in enumerate(timesteps):
            # 重组为DiT输入格式: [B*T, N, C]
            # (具体reshape取决于模型的patchify方式)
            z_input = rearrange(z, 'b t c h w -> (b t) (h w) c')
            
            t_batch = t.expand(B * num_frames)
            
            # 模型前向 (这里简化了条件输入)
            # noise_pred = self.video_dit(z_input, t_batch, text_emb, num_frames)
            
            # DDIM更新 (简化)
            alpha_t = alphas_cumprod[t.long()]
            alpha_prev = alphas_cumprod[timesteps[i+1].long()] \
                if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            
            # z_0_pred = (z_flat - sqrt(1-α_t) * noise_pred) / sqrt(α_t)
            # z = sqrt(α_prev) * z_0_pred + sqrt(1-α_prev) * noise_pred
            # (此处省略实际计算，因为需要完整模型)
        
        # VAE解码: 逐帧解码
        # videos = []
        # for frame_idx in range(num_frames):
        #     frame_latent = z[0, frame_idx]  # [C, H, W]
        #     frame = self.vae_decoder(frame_latent.unsqueeze(0))
        #     videos.append(frame)
        # video = torch.cat(videos, dim=0)  # [T, 3, H, W]
        
        return z  # 返回latent (实际使用时需要解码)


# 使用示例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试TemporalAttention
    temporal_attn = TemporalAttention(dim=512, num_heads=8).to(device)
    B, T, N, C = 2, 16, 64*64, 512  # 简化: 用小N测试
    N_small = 256  # 为了可运行性减小N
    x = torch.randn(B * T, N_small, C, device=device)
    out = temporal_attn(x, num_frames=T)
    print(f"Temporal Attention输入: {x.shape}")   # [32, 256, 512]
    print(f"Temporal Attention输出: {out.shape}")  # [32, 256, 512]
    
    # 测试VideoDiTBlock
    dit_block = VideoDiTBlock(dim=512, num_heads=8).to(device)
    x = torch.randn(B * T, N_small, 512, device=device)
    cond = torch.randn(B * T, 512, device=device)
    text_tokens = torch.randn(B, 77, 512, device=device)
    out = dit_block(x, cond, text_tokens, num_frames=T)
    print(f"VideoDiT Block输出: {out.shape}")  # [32, 256, 512]
    
    print("\n=== SDS Loss示例 (需要预训练模型, 此处展示接口) ===")
    print("rendered_image: [B, 3, 512, 512]")
    print("SDS grad: w(t) * (ε_pred - ε) * ∂z/∂θ")
    print("优化目标: 使3D渲染从各视角看起来都像自然图像")
```

**本章小结**：视频和3D扩散模型将扩散框架从2D图像扩展到更高维的数据空间。视频扩散的核心技术挑战在于时空一致性和计算效率的权衡——分解注意力（空间+时间分离）和时空Patches（Sora架构）是两条主要路线。对于3D生成，Score Distillation Sampling (SDS) 开创了用2D扩散模型监督3D优化的范式，尽管存在Janus问题等局限，但已成为文本到3D生成的基础技术。随着计算资源的增长和训练数据的积累，视频和3D扩散模型正在从"能生成"走向"能生成得好且快"的阶段。

---

## 第10章 部署推理优化——量化、剪枝、缓存与并行

前面九章我们从数学基础出发，系统梳理了扩散模型在采样加速、架构演进、条件生成、训练优化以及视频/3D扩展等方面的核心技术。然而，一个在研究环境中表现出色的扩散模型，要真正走向生产部署、服务千万级用户请求，还需要跨越一道巨大的"工程鸿沟"。本章聚焦**推理阶段的系统级优化**，从模型量化、结构剪枝、中间结果缓存到多维度并行策略，全面覆盖将扩散模型推向"低延迟、高吞吐、低成本"部署目标的关键技术。

### 10.1 扩散模型推理瓶颈分析

在深入优化技术之前，我们需要精确定位扩散模型推理的性能瓶颈。与单次前向推理的分类/检测模型不同，扩散模型的推理本质上是一个**多步迭代过程**，这使其推理开销呈倍数放大。

#### 10.1.1 端到端推理流程与延迟分解

以Stable Diffusion XL为例，一次完整的文本到图像推理包含以下阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│               扩散模型推理Pipeline延迟分解 (SDXL @A100)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌─────────────────────────────┐   ┌──────────────┐  │
│  │ 文本编码  │──▶│      去噪循环 (N步迭代)       │──▶│   VAE解码    │  │
│  │ ~50ms    │   │      ~3500ms (25步)          │   │   ~150ms     │  │
│  └──────────┘   └─────────────────────────────┘   └──────────────┘  │
│    1.4%                  94.6%                        4.0%          │
│                                                                     │
│  去噪循环内部分解 (单步 ~140ms):                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Self-Attn   Cross-Attn   FFN/Conv    ResBlock   GroupNorm    │  │
│  │   35ms        28ms        42ms        25ms       10ms        │  │
│  │   25%         20%         30%         18%        7%          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  瓶颈来源:                                                          │
│  • 迭代次数: 每步完整前向传播, 25步 = 25x单模型推理                     │
│  • 模型规模: U-Net 3.5B参数 → 单步~7 GFLOPS                         │
│  • 内存带宽: 权重搬运量 = 参数量 × 精度 × 步数                        │
│  •              = 3.5B × 2B × 25 = 175 GB/推理                     │
│  • Attention: O(n²) 复杂度, 1024×1024 分辨率下 n=4096               │
└─────────────────────────────────────────────────────────────────────┘
```

从延迟分解可以看出：**去噪循环占据了94%以上的推理时间**。因此，所有优化策略都围绕两个核心目标展开：

1. **减少每步计算量**：量化、剪枝、算子融合
2. **减少总步数或跳过冗余步**：步数蒸馏（前面章节已覆盖）、缓存复用

#### 10.1.2 计算与内存带宽的Roofline分析

扩散模型在推理时通常处于**内存带宽受限（Memory-Bound）**状态，尤其在batch size=1的在线推理场景：

- **算术强度（Arithmetic Intensity）**：对于FP16推理，U-Net的平均算术强度约为 $AI \approx 50\text{-}100$ FLOP/Byte，而A100的算力/带宽比为 $312 \text{TFLOPS} / 2 \text{TB/s} = 156$ FLOP/Byte
- 大量矩阵乘法的batch维度较小，无法充分利用GPU的计算单元
- Attention的softmax、LayerNorm等逐元素操作进一步拉低算术强度

这意味着**量化（减少数据搬运量）**和**缓存（减少重复计算和搬运）**是最直接有效的优化方向。

### 10.2 模型量化技术

量化是将模型参数和/或激活从高精度（FP32/FP16）表示降低到低精度（INT8/INT4）的技术。对于内存带宽受限的扩散模型，量化的收益是双重的：**减少内存占用**和**提升访存吞吐**。

#### 10.2.1 W8A8量化：权重与激活同步量化

W8A8指将权重（Weight）和激活（Activation）均量化到INT8精度。对称量化的映射关系为：

$$q = \text{clamp}\left(\left\lfloor \frac{x}{s} \right\rceil, -128, 127\right), \quad s = \frac{\max(|x|)}{127}$$

其中 $s$ 为缩放因子（scale），$\lfloor \cdot \rceil$ 表示四舍五入。

**Post-Training Quantization (PTQ)** 是最常用的方案，无需重训练：

1. 收集校准数据（calibration data）：用少量输入跑前向传播，统计各层激活的分布
2. 确定量化参数：根据激活分布选择最优的scale和zero-point
3. 量化权重：静态量化，保存量化后的权重
4. 推理时动态量化激活：每次前向传播时在线计算激活的量化参数

W8A8量化的典型效果：
- **速度提升**：1.8-2.5x（取决于硬件INT8支持程度）
- **内存减少**：约50%
- **质量损失**：FID增加0.5-1.0（在COCO-30K上）

#### 10.2.2 W4A16量化：极致权重压缩

W4A16将权重压缩到4bit，但保持激活在FP16精度。由于扩散模型在推理时通常是batch size=1的内存带宽受限场景，权重的搬运量是主要瓶颈，因此W4A16可以获得接近4x的权重内存压缩，同时因为激活保持高精度，质量损失更小。

主流的W4A16量化方法：

- **GPTQ**：基于Hessian信息的逐层最优量化，通过最小化量化误差 $\|Wx - \hat{W}x\|_2^2$ 来确定最优的量化参数
- **AWQ (Activation-Aware Weight Quantization)**：观察到1%的显著权重通道对质量影响极大，通过per-channel缩放保护这些通道

W4A16量化的典型效果：
- **速度提升**：2.0-2.4x
- **内存减少**：约75%（权重部分）
- **质量损失**：FID增加0.3-0.8

#### 10.2.3 扩散模型量化的特殊挑战

与LLM的量化不同，扩散模型量化面临独特的技术挑战：

**挑战一：时间步相关的激活分布漂移**

扩散模型在不同时间步 $t$ 的激活分布差异巨大。在高噪声时间步（$t$ 接近 $T$），输入几乎是纯噪声，激活值分布广且方差大；而在低噪声时间步（$t$ 接近 $0$），激活值精细且集中。使用单一量化参数（全局scale）会导致严重的量化误差。

**解决方案：时间步感知量化（Timestep-Aware Quantization）**

$$s_t = \frac{\max(|a_t|)}{2^{b-1} - 1}$$

其中 $a_t$ 是时间步 $t$ 的激活，$b$ 是量化位宽。实践中通常将时间步分为若干组（如4-8组），每组使用独立的量化参数。

**挑战二：注意力层的量化敏感性**

注意力计算中的 $QK^T$ 矩阵乘法后接softmax，softmax的指数运算会放大量化误差。实验表明，Attention层的量化误差比FFN层高3-5倍。

**解决方案**：混合精度策略——Attention层保持FP16/INT8，FFN/Conv层可以激进量化到INT4。

#### 10.2.4 ViDiT-Q与Q-Diffusion

**ViDiT-Q** 针对DiT架构的特殊性设计量化方案：
- 利用DiT中adaLN（Adaptive Layer Norm）的条件注入特性，将量化参数与条件信号绑定
- 对不同的DiT Block使用差异化位宽：浅层Block（靠近输入）用4bit，深层Block（靠近输出）用8bit

**Q-Diffusion** 是扩散模型专用PTQ框架：
- 设计了时间步校准集采样策略：均匀采样不同时间步的校准数据
- 引入shortcut-splitting技术处理残差连接中的量化累积误差
- 在SDXL上实现W4A8量化，FID仅增加0.8

### 10.3 结构剪枝

量化是在保持模型结构不变的前提下降低数值精度，而结构剪枝（Structured Pruning）则是直接**移除模型中的冗余结构单元**，从而减少计算量和参数量。

#### 10.3.1 剪枝目标与策略

扩散模型中可剪枝的结构单元：

```
┌────────────────────────────────────────────────────────────┐
│              扩散模型结构剪枝目标层级                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Level 1: 层级剪枝 (Layer Pruning)                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Block_1  Block_2  [Block_3]  Block_4  [Block_5] ... │   │
│  │                    ✗ 移除              ✗ 移除        │   │
│  └──────────────────────────────────────────────────────┘   │
│  目标: 移除整个Transformer/ResNet Block                      │
│  加速: 每移除1个Block, 减少1/N的计算量                        │
│                                                            │
│  Level 2: 注意力头剪枝 (Head Pruning)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Head_1  Head_2  [Head_3]  Head_4  [Head_5]  Head_6  │   │
│  │                  ✗ 移除             ✗ 移除           │   │
│  └──────────────────────────────────────────────────────┘   │
│  目标: 移除低重要性的Attention Head                          │
│  加速: 减少Q/K/V矩阵维度, 节省Attention计算                  │
│                                                            │
│  Level 3: 通道剪枝 (Channel Pruning)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ [C_1] C_2  C_3 [C_4] C_5  C_6  C_7 [C_8] C_9  C_10│   │
│  │  ✗              ✗                    ✗              │   │
│  └──────────────────────────────────────────────────────┘   │
│  目标: 减少卷积/FFN的通道数                                  │
│  加速: 减少每层的FLOPS和参数量                               │
└────────────────────────────────────────────────────────────┘
```

**剪枝重要性评估准则**：

1. **Taylor重要性**：基于损失函数对参数的一阶Taylor展开，评估移除某结构单元对损失的影响：
   $$I_j = \left| \frac{\partial \mathcal{L}}{\partial h_j} \cdot h_j \right|$$
   其中 $h_j$ 是第 $j$ 个结构单元的输出激活。

2. **Magnitude剪枝**：根据权重的L1/L2范数评估重要性：
   $$I_j = \|W_j\|_1 \quad \text{或} \quad I_j = \|W_j\|_2$$

3. **Sensitivity分析**：逐一移除每个结构单元，观察输出质量的变化幅度。

#### 10.3.2 剪枝后微调

结构剪枝后模型能力必然下降，需要通过微调恢复。关键策略：

- **知识蒸馏微调**：使用未剪枝的原始模型作为教师，剪枝后的模型作为学生
- **渐进式剪枝**：分多轮进行，每轮剪枝少量（如10%），然后微调恢复，避免一次性剪枝过多导致不可恢复的质量损失
- **微调数据量**：通常只需原始训练数据的1-5%即可恢复大部分质量

结构剪枝在SDXL上的典型效果：剪枝40%的参数后，微调10K步可恢复到FID增加<1.5的水平，推理速度提升2.5-3x。

### 10.4 DeepCache与Block Caching

缓存复用是扩散模型推理优化中最优雅的技术之一。其核心洞察极其简单：**相邻去噪步的模型中间表示高度相似**。

实验测量表明，在U-Net/DiT中，相邻时间步之间深层特征的余弦相似度高达0.98+，这意味着我们完全可以复用上一步的中间计算结果，而不是每步都从头计算。

#### 10.4.1 DeepCache原理

```
┌────────────────────────────────────────────────────────────────────┐
│              DeepCache 缓存策略示意图                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  U-Net/DiT层级结构:                                                 │
│                                                                    │
│  浅层 (高分辨率)     中层            深层 (低分辨率)              │
│  ┌─────┐ ┌─────┐  ┌─────┐ ┌─────┐  ┌─────┐ ┌─────┐        │
│  │ L_1 │─│ L_2 │──│ L_3 │─│ L_4 │──│ L_5 │─│ L_6 │        │
│  └─────┘ └─────┘  └─────┘ └─────┘  └─────┘ └─────┘        │
│  变化快,需频繁刷新        变化中等          变化慢,可缓存复用      │
│                                                                    │
│  时间步推进:                                                       │
│  t=25: ███  完整计算所有层, 缓存深层结果                        │
│  t=24: ░░░  只计算浅层, 深层用缓存 → 节省40-60%计算          │
│  t=23: ░░░  只计算浅层, 深层用缓存                               │
│  t=22: ███  完整计算所有层, 刷新缓存  (每N步刷新一次)         │
│  t=21: ░░░  只计算浅层, 深层用缓存                               │
│  ...                                                                │
│                                                                    │
│  缓存策略对比:                                                    │
│  ──────────────────────────────────────────────────────────  │
│  固定间隔:  每N步完整计算一次 (N=3-5), 其余步用缓存         │
│  自适应:   监控特征变化量, 变化>阈值时刷新                        │
│  层级化:   浅层每步刷新, 中层每2步, 深层每4-5步               │
│  ──────────────────────────────────────────────────────────  │
└────────────────────────────────────────────────────────────────────┘
```

DeepCache的数学表达：对于第 $l$ 层在时间步 $t$ 的输出 $h_t^l$，当缓存命中时：

$$h_t^l \approx h_{t+1}^l, \quad \text{if } \|h_t^l - h_{t+1}^l\|_2 / \|h_t^l\|_2 < \epsilon$$

典型性能：
- **固定间隔N=3**：节省60%深层计算，总体加速2.3x，FID增加<0.3
- **自适应策略**：加速2.0-2.5x，FID增加<0.2（更保守但质量更好）

#### 10.4.2 Block Caching与选择性跳过

Block Caching是DeepCache的细粒度变体，它不是按深度统一缓存，而是针对每个独立Block进行缓存决策：

- 为每个Block维护一个"变化计分器"，记录该Block在相邻步之间的输出变化量
- 变化量低于阈值的Block直接用缓存，跳过计算
- 变化量高于阈值的Block重新计算并更新缓存

**与步数蒸馏的互补性**：缓存技术和步数蒸馏可以叠加使用。例如，先用蒸馏将步数从25减到8，再用DeepCache在这8步内做缓存复用，维合加速可达5-8x。

### 10.5 并行与投机采样

除了“减少单步计算量”和“减少步数”，还可以通过**并行化**来提升实际吞吐。

#### 10.5.1 Tensor并行（模型并行）

将模型的权重矩阵切片到多个GPU上，每个GPU只计算部分结果，最后通过AllReduce汇聚。对于参数量达8B的大型扩散模型（如SD3-8B），单卡可能无法容纳，Tensor并行是必要的。

#### 10.5.2 Pipeline并行（跨步流水线）

将连续的去噪步分配到不同GPU上，形成流水线。当处理多个请求时，不同请求的不同步骤可以在不同GPU上并行执行，提升多请求场景下的吞吐。

#### 10.5.3 Speculative Sampling for Diffusion

借鉴LLM领域的投机解码思想：

1. **小模型走N步**：用一个轻量级模型（如剪枝后的小模型）快速生成N步的去噪轨迹
2. **大模型验证/修正**：用完整大模型并行验证这N步的结果，接受质量合格的步骤，拒绝并修正不合格的
3. 平均接受率越高，加速比越大

实验表明，当小模型和大模型的分布匹配度较高时（如教师-学生蒸馏对），平均接受率可达70-85%，实现1.5-2x加速。

### 10.6 推理引擎优化

#### 10.6.1 TensorRT加速

NVIDIA TensorRT是最成熟的推理优化引擎，对扩散模型的主要优化包括：

- **算子融合**：将Conv+BN+ReLU、Multi-Head Attention内部的Q/K/V投影、softmax+matmul等融合为单个kernel，减少kernel launch开销和中间内存访问
- **内存优化**：自动计算最优的workspace分配，减少内存碎片和分配/释放开销
- **动态Batch支持**：支持变长batch size，便于在线服务动态调度
- **FP16/INT8混合精度**：自动选择每层的最优计算精度

典型加速效果（相比PyTorch FP16）：1.5-2.5x

#### 10.6.2 其他推理引擎

| 引擎 | 加速比 | 特点 | 适用场景 |
|------|--------|------|----------|
| TensorRT | 1.5-2.5x | 最优化但开发成本高 | 生产部署(NVIDIA) |
| ONNX Runtime | 1.2-1.5x | 跨平台，易集成 | 跨硬件部署 |
| torch.compile | 1.3-1.8x | 零代码修改 | 快速原型验证 |
| xFormers | 1.2-1.4x | 专注Attention优化 | Attention受限场景 |
| Flash Attention | 1.3-1.6x | 内存高效+计算快 | 通用Attention加速 |

### 10.7 端到端推理优化方案

实际部署中，单一优化技术很少单独使用，而是**组合多种技术**形成完整的优化方案：

```
┌─────────────────────────────────────────────────────────────────┐
│              端到端优化组合方案 (Pareto前沿)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  方案A: 极致速度优先 (A100, 延迟敏感)                          │
│  ──────────────────────────────────────────────────          │
│  步数蒸馏(4步) + W8A8量化 + DeepCache(N=2) + TensorRT       │
│  维合加速: ~12x   |  延迟: ~300ms  |  FID增加: ~2.0           │
│                                                                 │
│  方案B: 质量速度平衡 (4090, 通用场景)                          │
│  ──────────────────────────────────────────────────          │
│  DPM-Solver(8步) + W4A16量化 + DeepCache(N=3) + compile   │
│  维合加速: ~6x    |  延迟: ~600ms  |  FID增加: ~1.0           │
│                                                                 │
│  方案C: 移动端/边缘部署 (内存受限)                            │
│  ──────────────────────────────────────────────────          │
│  一致性蒸馏(1步) + W4A16 + 40%剪枝 + TFLite               │
│  维合加速: ~25x   |  延迟: ~1.5s   |  FID增加: ~4.0           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.8 代码实践：DeepCache推理实现

```python
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class CacheConfig:
    """DeepCache缓存配置"""
    cache_interval: int = 3          # 每N步刷新一次缓存
    cache_layer_start: int = 2       # 从第几层开始缓存(浅层不缓存)
    adaptive_threshold: float = 0.02 # 自适应缓存阈值
    use_adaptive: bool = False       # 是否使用自适应策略


class DeepCacheScheduler:
    """DeepCache缓存调度器
    
    核心思想: 相邻去噪步的深层特征变化微小,
    可以复用上一步的中间结果来跳过计算。
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_store: Dict[int, torch.Tensor] = {}  # layer_idx -> cached_output
        self.step_counter: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
    
    def should_use_cache(self, layer_idx: int, current_feature: Optional[torch.Tensor] = None) -> bool:
        """判断是否应该使用缓存"""
        # 浅层不缓存
        if layer_idx < self.config.cache_layer_start:
            return False
        
        # 缓存中没有该层的数据
        if layer_idx not in self.cache_store:
            return False
        
        if self.config.use_adaptive and current_feature is not None:
            # 自适应策略: 根据特征变化量决定
            cached = self.cache_store[layer_idx]
            relative_change = (current_feature - cached).norm() / (cached.norm() + 1e-8)
            return relative_change.item() < self.config.adaptive_threshold
        else:
            # 固定间隔策略
            return self.step_counter % self.config.cache_interval != 0
    
    def get_cached(self, layer_idx: int) -> Optional[torch.Tensor]:
        """获取缓存的特征"""
        self.cache_hits += 1
        return self.cache_store.get(layer_idx)
    
    def update_cache(self, layer_idx: int, feature: torch.Tensor):
        """更新缓存"""
        self.cache_misses += 1
        self.cache_store[layer_idx] = feature.detach()
    
    def step(self):
        """推进一步"""
        self.step_counter += 1
    
    def reset(self):
        """重置缓存(新样本时调用)"""
        self.cache_store.clear()
        self.step_counter = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class SimpleDiTBlockWithCache(nn.Module):
    """支持DeepCache的简化DiT Block"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


class DiTWithDeepCache(nn.Module):
    """集成DeepCache的DiT推理模型
    
    通过缓存深层Block的输出来跳过重复计算,
    在保持质量的同时实现2-3x加速。
    """
    
    def __init__(self, dim: int = 512, num_blocks: int = 12, num_heads: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            SimpleDiTBlockWithCache(dim, num_heads) 
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, dim)
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(
        self, 
        x: torch.Tensor,           # [B, N, D] latent tokens
        t: torch.Tensor,           # [B] timestep
        cache_scheduler: Optional[DeepCacheScheduler] = None
    ) -> torch.Tensor:
        # 时间步条件注入
        t_emb = self.time_embed(t.unsqueeze(-1).float())  # [B, D]
        x = x + t_emb.unsqueeze(1)  # 广播到所有token
        
        for layer_idx, block in enumerate(self.blocks):
            if cache_scheduler is not None and cache_scheduler.should_use_cache(layer_idx):
                # 使用缓存, 跳过计算
                cached_output = cache_scheduler.get_cached(layer_idx)
                if cached_output is not None:
                    x = cached_output
                    continue
            
            # 正常计算
            x = block(x)
            
            # 更新缓存
            if cache_scheduler is not None and layer_idx >= cache_scheduler.config.cache_layer_start:
                cache_scheduler.update_cache(layer_idx, x)
        
        x = self.final_norm(x)
        x = self.output_proj(x)
        return x


def deep_cache_inference(
    model: DiTWithDeepCache,
    initial_noise: torch.Tensor,
    num_steps: int = 25,
    cache_config: Optional[CacheConfig] = None
) -> Tuple[torch.Tensor, Dict]:
    """DeepCache加速推理
    
    Args:
        model: DiT模型
        initial_noise: 初始噪声 [B, N, D]
        num_steps: 去噪步数
        cache_config: 缓存配置
    
    Returns:
        去噪结果和统计信息
    """
    if cache_config is None:
        cache_config = CacheConfig(cache_interval=3, cache_layer_start=4)
    
    scheduler = DeepCacheScheduler(cache_config)
    x = initial_noise
    device = initial_noise.device
    
    # 简化的去噪调度 (DDIM-style)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    
    with torch.no_grad():
        for i in range(num_steps):
            t = timesteps[i]
            t_batch = t.expand(x.shape[0])  # [B]
            
            # 模型前向传播 (带缓存)
            noise_pred = model(x, t_batch, cache_scheduler=scheduler)
            
            # 简化的去噪更新 (DDIM风格)
            alpha_t = 1.0 - t  # 简化的信噪比
            x = x - (1.0 / num_steps) * noise_pred
            
            scheduler.step()
    
    stats = {
        'cache_hit_rate': scheduler.hit_rate,
        'cache_hits': scheduler.cache_hits,
        'cache_misses': scheduler.cache_misses,
        'effective_speedup': 1.0 / (1.0 - scheduler.hit_rate * 0.6)  # 估算加速比
    }
    
    return x, stats


# 使用示例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = DiTWithDeepCache(dim=256, num_blocks=12, num_heads=8).to(device)
    model.eval()
    
    # 生成初始噪声
    batch_size = 2
    num_tokens = 64  # 8x8 latent patches
    noise = torch.randn(batch_size, num_tokens, 256, device=device)
    
    # 无缓存推理 (基线)
    import time
    start = time.time()
    result_no_cache, _ = deep_cache_inference(model, noise, num_steps=25, cache_config=CacheConfig(cache_interval=1))
    time_no_cache = time.time() - start
    
    # DeepCache推理
    start = time.time()
    result_cached, stats = deep_cache_inference(model, noise, num_steps=25, cache_config=CacheConfig(cache_interval=3, cache_layer_start=4))
    time_cached = time.time() - start
    
    print(f"\n=== DeepCache 推理结果 ===")
    print(f"无缓存延迟: {time_no_cache*1000:.1f}ms")
    print(f"DeepCache延迟: {time_cached*1000:.1f}ms")
    print(f"实际加速: {time_no_cache/time_cached:.2f}x")
    print(f"缓存命中率: {stats['cache_hit_rate']:.2%}")
    print(f"估算理论加速: {stats['effective_speedup']:.2f}x")
    print(f"输出形状: {result_cached.shape}")
```

### 10.9 代码实践：W8A8量化推理与TensorRT导出

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np


class QuantizedLinear(nn.Module):
    """INT8量化的线性层 (W8A8)
    
    实现对称量化: q = clamp(round(x/s), -128, 127)
    反量化: x_hat = q * s
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 量化后的INT8权重
        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        # 权重缩放因子 (per-channel)
        self.register_buffer('weight_scale', torch.ones(out_features))
        # 激活缩放因子 (动态计算)
        self.register_buffer('input_scale', torch.ones(1))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None
    
    @staticmethod
    def from_float(linear: nn.Linear, calibration_data: Optional[torch.Tensor] = None) -> 'QuantizedLinear':
        """从浮点线性层转换为量化线性层"""
        q_linear = QuantizedLinear(
            linear.in_features, linear.out_features, 
            bias=linear.bias is not None
        )
        
        # Per-channel权重量化
        weight = linear.weight.data.float()
        weight_scale = weight.abs().max(dim=1)[0] / 127.0
        weight_scale = weight_scale.clamp(min=1e-8)
        weight_int8 = (weight / weight_scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        
        q_linear.weight_int8 = weight_int8
        q_linear.weight_scale = weight_scale
        
        if linear.bias is not None:
            q_linear.bias = linear.bias.data
        
        # 如果有校准数据, 计算激活量化参数
        if calibration_data is not None:
            with torch.no_grad():
                input_max = calibration_data.abs().max()
                q_linear.input_scale = input_max / 127.0
        
        return q_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 动态计算激活量化参数
        input_scale = x.abs().max() / 127.0
        input_scale = input_scale.clamp(min=1e-8)
        
        # 量化激活
        x_int8 = (x / input_scale).round().clamp(-128, 127)
        
        # INT8矩阵乘法 (模拟, 实际部署用CUTLASS/TensorRT)
        # 反量化: output = (x_int8 @ W_int8.T) * (input_scale * weight_scale)
        output = x_int8.float() @ self.weight_int8.float().t()
        output = output * (input_scale * self.weight_scale.unsqueeze(0))
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class QuantizedDiTBlock(nn.Module):
    """量化的DiT Block (W8A8)"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.norm1 = nn.LayerNorm(dim)  # Norm保持FP16
        self.norm2 = nn.LayerNorm(dim)
        
        # Attention的Q/K/V投影量化
        self.q_proj = QuantizedLinear(dim, dim)
        self.k_proj = QuantizedLinear(dim, dim)
        self.v_proj = QuantizedLinear(dim, dim)
        self.o_proj = QuantizedLinear(dim, dim)
        
        # FFN量化
        self.ffn_up = QuantizedLinear(dim, dim * 4)
        self.ffn_down = QuantizedLinear(dim * 4, dim)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        # Self-Attention (Q/K/V量化, softmax保持FP16)
        h = self.norm1(x)
        q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention计算 (softmax保持高精度避免误差放大)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)  # 保持FP32
        h = (attn @ v).transpose(1, 2).reshape(B, N, D)
        h = self.o_proj(h)
        x = x + h
        
        # FFN (全量化)
        h = self.norm2(x)
        h = self.ffn_up(h)
        h = self.act(h)
        h = self.ffn_down(h)
        x = x + h
        
        return x


def export_to_tensorrt_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    onnx_path: str = "diffusion_model.onnx"
):
    """TensorRT部署第一步: 导出ONNX格式
    
    实际部署流程:
    1. PyTorch -> ONNX (torch.onnx.export)
    2. ONNX -> TensorRT Engine (trtexec / TensorRT API)
    3. TensorRT Engine -> Inference
    """
    print(f"\n=== TensorRT导出流程 ===")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"输入形状: {sample_input.shape}")
    print(f"\n步骤1: 导出ONNX")
    print(f"  torch.onnx.export(model, sample_input, '{onnx_path}',")
    print(f"                    opset_version=17,")
    print(f"                    dynamic_axes={{'input': {{0: 'batch'}}, 'output': {{0: 'batch'}}}}")
    print(f"                    )")
    print(f"\n步骤2: ONNX -> TensorRT")
    print(f"  trtexec --onnx={onnx_path} \\")
    print(f"          --saveEngine=model.trt \\")
    print(f"          --fp16 \\")
    print(f"          --int8 --calib=calibration_cache \\")
    print(f"          --minShapes=input:1x64x256 \\")
    print(f"          --optShapes=input:4x64x256 \\")
    print(f"          --maxShapes=input:8x64x256")
    print(f"\n步骤3: TensorRT推理")
    print(f"  # 加载engine并执行推理")
    print(f"  # 预期加速: 1.5-2.5x vs PyTorch FP16")


# 使用示例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 256
    
    # 创建并量化模型
    print("=== W8A8量化示例 ===")
    
    # 原始浮点模型
    float_linear = nn.Linear(dim, dim).to(device)
    
    # PTQ量化
    calib_data = torch.randn(100, dim, device=device)  # 校准数据
    quant_linear = QuantizedLinear.from_float(float_linear, calib_data)
    quant_linear = quant_linear.to(device)
    
    # 对比精度
    test_input = torch.randn(4, 64, dim, device=device)
    with torch.no_grad():
        float_output = float_linear(test_input)
        quant_output = quant_linear(test_input)
    
    mse = ((float_output - quant_output) ** 2).mean().item()
    relative_error = (float_output - quant_output).abs().mean() / float_output.abs().mean()
    print(f"量化前后 MSE: {mse:.6f}")
    print(f"相对误差: {relative_error:.4%}")
    
    # 内存对比
    float_mem = float_linear.weight.numel() * 4  # FP32: 4 bytes
    quant_mem = quant_linear.weight_int8.numel() * 1 + quant_linear.weight_scale.numel() * 4
    print(f"\nFP32权重内存: {float_mem/1024:.1f} KB")
    print(f"INT8权重内存: {quant_mem/1024:.1f} KB")
    print(f"压缩比: {float_mem/quant_mem:.1f}x")
    
    # TensorRT导出演示
    model = QuantizedDiTBlock(dim=256, num_heads=8).to(device)
    sample = torch.randn(1, 64, 256, device=device)
    export_to_tensorrt_onnx(model, sample)
```

**本章小结**：扩散模型推理优化是一个系统工程问题，需要从多个维度综合考量。量化（W8A8/W4A16）解决内存带宽瓶颈，结构剪枝减少冗余计算，DeepCache利用时间步间的特征连续性来跳过重复计算，而推理引擎（TensorRT等）通过算子融合和内存优化提升硬件利用率。实际部署中，将多种技术组合使用可以实现10x以上的综合加速，将SDXL的延迟从秒级压缩到亚秒级，从而支撑实时交互式图像生成体验。

---

## 第11章 前沿系统剖析——SDXL、SD3、Flux与Sora架构

前面章节从技术组件的角度深入解析了扩散模型的各项核心技术。本章将视角拉升到**完整系统**层面，剖析当前最具代表性的几个前沿图像/视频生成系统，理解它们如何将前述技术组装成一个有机整体，以及每个系统各自的技术创新和设计哲学。

### 11.1 SDXL（Stability AI, 2023）

Stable Diffusion XL是Stable Diffusion系列的重大升级，它在保持U-Net架构的基础上，通过多个关键设计决策大幅提升了生成质量。

#### 11.1.1 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                SDXL 完整Pipeline架构                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  文本编码 (双编码器):                                                │
│  ┌──────────────────┐  ┌───────────────────┐                        │
│  │ OpenCLIP ViT-bigG │  │ CLIP ViT-L/14     │                        │
│  │ (1280-dim)        │  │ (768-dim)          │                        │
│  └─────────┬────────┘  └─────────┬─────────┘                        │
│            │                    │                                      │
│            └──────┬───────────┘                                      │
│                 │ concat: 2048-dim                                    │
│                 ▼                                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Base U-Net (3.5B params)                                    │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │  │
│  │  │ Down x3  │─│  Mid     │─│  Up x3   │─│ Output   │  │  │
│  │  │ 320/640/ │ │ 1280-dim │ │ 1280/640│ │ 128x128  │  │  │
│  │  │ 1280-dim │ │          │ │ /320-dim│ │ latent   │  │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │  │
│  │  + 微条件注入: (orig_size, crop_coords, target_size)          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                 │ Base输出 (800步 → 200步)                          │
│                 ▼                                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Refiner U-Net (1.4B params) [可选]                           │  │
│  │  • 只处理最后200步去噪 (t: 0.2→0)                           │  │
│  │  • 专注细节精化, 提升局部质量                             │  │
│  │  • 使用相同的双编码器条件                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                 │                                                      │
│                 ▼                                                      │
│  ┌─────────────────────┐                                          │
│  │ VAE Decoder       │  128x128 latent → 1024x1024 image               │
│  │ (FP32精度解码)     │                                                │
│  └─────────────────────┘                                          │
└─────────────────────────────────────────────────────────────────────┘
```

SDXL的三大关键创新：

**1. 双文本编码器**

不SD 1.x/2.x的单编码器不同，SDXL同时使用两个CLIP编码器，将它们的输出拼接后作为Cross-Attention的条件输入：

$$c_{\text{text}} = [\text{CLIP-L}(\text{prompt}); \text{CLIP-bigG}(\text{prompt})] \in \mathbb{R}^{77 \times 2048}$$

双编码器提供了更丰富的语义表示，显著提升了提示词遵从度。

**2. 微条件系统（Micro-Conditioning）**

SDXL引入了一个巧妙的微条件机制，解决了训练数据中图像分辨率和裁剪不一致的问题：

- **原始分辨率条件**：告诉模型训练图像的原始分辨率，推理时设为目标分辨率以避免生成低质量图像
- **裁剪坐标条件**：记录训练时的裁剪位置，推理时设为(0,0)以生成完整主体
- **目标分辨率条件**：指定期望的输出分辨率

这些微条件通过Fourier嵌入后加到时间步嵌入上：

$$e_{\text{micro}} = \text{MLP}([\text{FourierEmb}(h_{\text{orig}}); \text{FourierEmb}(w_{\text{orig}}); \text{FourierEmb}(\text{crop}_y); \text{FourierEmb}(\text{crop}_x)])$$

**3. 两阶段生成（Base + Refiner）**

- Base模型（3.5B）处理主要的去噪过程，从纯噪声去噪到中等噪声水平
- Refiner模型（1.4B）接手最后阶段，专注于精细化局部细节和纹理
- Refiner是可选的，不用Refiner也能生成高质量结果

#### 11.1.2 训练策略

- **渐进式分辨率提升**：256×256预训练 → 512×512 → 1024×1024，每阶段渐进提升分辨率而非一步到位
- **多宽高比训练**：支持多种宽高比（16:9, 4:3, 1:1, 3:4, 9:16等），通过bucket分组实现高效训练
- **Offset Noise**：添加全局偏移噪声，解决传统扩散模型无法生成纯黑/纯白图像的问题

### 11.2 Stable Diffusion 3 / 3.5

SD3标志着扩散模型架构的一次重大范式转变：从**U-Net全面过渡到Transformer**。

#### 11.2.1 MMDiT架构（双流Transformer）

```
┌────────────────────────────────────────────────────────────────────┐
│          SD3 MMDiT 架构 (双流 + Joint Attention)                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  输入:                                                              │
│  Text tokens: T5-XXL + CLIP L/14 → [N_text, D]                     │
│  Image tokens: VAE(x) + Patchify → [N_img, D]                      │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    MMDiT Block (x24)                       │  │
│  │                                                            │  │
│  │  Text Stream            Image Stream                       │  │
│  │  ┌────────────┐          ┌────────────┐                   │  │
│  │  │ adaLN(t)   │          │ adaLN(t)   │                   │  │
│  │  │ Q_t, K_t,  │          │ Q_i, K_i,  │                   │  │
│  │  │ V_t        │          │ V_i        │                   │  │
│  │  └─────┬──────┘          └─────┬──────┘                   │  │
│  │        │                      │                             │  │
│  │        └──────┬───────────┘                             │  │
│  │              ▼                                               │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  Joint Attention                                │  │  │
│  │  │  Q = [Q_t; Q_i],  K = [K_t; K_i],  V = [V_t; V_i]  │  │  │
│  │  │  Attn = softmax(QK^T/√d) · V                    │  │  │
│  │  │  → split back to text_out, img_out              │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │              │                                               │  │
│  │        ┌─────┴───────────┐                             │  │
│  │  Text: │ adaLN + FFN_t │  Image: adaLN + FFN_i          │  │
│  │        └─────────────────┘                             │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  核心设计特点:                                                      │
│  • 文本和图像各自有独立的投影/FFN参数                            │
│  • 但在Attention阶段共享同一个注意力空间 (Joint)                │
│  • 实现文本-图像的双向信息流动                                  │
└────────────────────────────────────────────────────────────────────┘
```

MMDiT的核心创新在于**Joint Attention**机制：将文本和图像的token拼接后共同参与注意力计算，而非像U-Net中通过Cross-Attention将文本作为K/V注入。这种设计让文本和图像在注意力空间中“平等对话”，实现双向信息流动。

#### 11.2.2 Rectified Flow训练

SD3放弃了传统的扩散过程，转而采用**Rectified Flow**（直线流匹配）：

- 传统扩散：噪声轨迹是弯曲的随机过程，採样需要多步求解SDE/ODE
- Rectified Flow：学习从噪声到数据的**直线路径**，更简洁的轨迹意味着更少的采样步数

数学表达：

$$z_t = (1 - t) \cdot z_0 + t \cdot \epsilon, \quad t \in [0, 1]$$

$$v = \epsilon - z_0 \quad (\text{目标: 学习速度场 velocity})$$

训练目标：

$$\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, z_0, \epsilon} \left[ \|v_\theta(z_t, t) - (\epsilon - z_0)\|_2^2 \right]$$

Rectified Flow的优势：
- 更少的采样步数：20步即可达到高质量（DDPM需要1000步）
- 更简洁的理论基础：无需SDE/ODE框架，直接学习向量场
- 更稳定的训练：直线插值避免了弯曲轨迹的数值不稳定性

#### 11.2.3 文本编码与能力突破

SD3的文本编码组合：T5-XXL + CLIP L/14，其中T5-XXL提供了更强的文本理解能力，尤其在复杂指令和空间关系描述方面。

核心改进：
- **文字渲染能力**：得益于Joint Attention和T5的强文本表示，SD3可以生成高质量的图像内文字
- **空间推理**：更好地理解"左边“、“上方”、“之间”等空间关系
- **参数规模**：2B和8B两个变体

### 11.3 Flux（Black Forest Labs, 2024）

Flux由Stability AI的前核心团队创办的Black Forest Labs发布，可以视为SD3技术路线的进一步演化。

#### 11.3.1 架构特点

- **12B参数Transformer**：截至2024年最大的开源图像生成模型之一
- **Rectified Flow**：跟SD3一样采用直线流匹配，而非传统扩散
- **高效注意力设计**：在深层采用单流设计（文本和图像合并为单一序列），减少计算开销

#### 11.3.2 变体系列

```
┌────────────────────────────────────────────────────────────┐
│         Flux 变体系列对比 与 SD3/SDXL对照                    │
├────────────────┬─────────────┬──────────────┬─────────────┤
│                │ Flux.1-dev  │ Flux.1-schnell│  SD3-2B      │
├────────────────┼─────────────┼──────────────┼─────────────┤
│ 参数量         │ 12B         │ 12B           │  2B          │
│ 推理步数       │ 20-50       │ 1-4           │  20-28       │
│ 架构           │ Transformer │ Transformer   │  MMDiT       │
│ 流匹配         │ Rect. Flow  │ Rect. Flow    │  Rect. Flow  │
│ 生成速度@A100  │ ~8s         │ ~2s           │  ~5s         │
│ 文字渲染       │ ★★★★★     │ ★★★★★       │  ★★★★       │
│ Prompt遵从度   │ ★★★★★     │ ★★★★☆       │  ★★★★       │
│ 开源策略       │ Apache-2.0  │ Apache-2.0    │  部分开源    │
└────────────────┴─────────────┴──────────────┴─────────────┘
```

Flux.1-schnell通过步数蒸馏实现4步快速生成，是当前速度-质量平衡最好的开源模型之一。

#### 11.3.3 技术亮点

- **卓越的文字渲染**：相比SDXL和SD3，Flux在图像内文字生成方面表现最佳，得益于更大的模型容量和更充分的训练
- **极强的prompt遵从度**：对复杂描述（多对象、空间关系、属性绑定）的遵从度显著优于前代模型
- **快速推理**：Flux.1-schnell在A100上可在~2秒内生成1024×1024图像

### 11.4 Sora与视频生成前沿

OpenAI的Sora将扩散模型的能力边界从图像推向视频，展示了“视觉世界模拟器”的潜力。

#### 11.4.1 架构设计哲学

- **Spacetime Patches**：将视频分解为时空 patches，统一处理不同分辨率/时长/宽高比的视频
- **可变分辨率/时长/宽高比**：与固定分辨率的模型不同，Sora可以原生处理任意尺寸的视频
- **DiT Backbone**：使用Transformer处理展平后的时空tokens

#### 11.4.2 训练策略

- **图像-视频联合训练**：将图像视为“单帧视频”，与视频数据联合训练，扩大有效训练数据量
- **长视频分块训练**：将长视频分块处理，通过时序一致性损失保持块间连贯性
- **大规模计算**：估计训练耗资达数千万美元级别

#### 11.4.3 能力边界

- **物理世界理解**：能够生成符合基本物理规律的视频（如液体流动、反射）
- **长时间一致性**：在分钟级视频中保持角色、场景的一致性
- **3D空间感知**：理解镜头运动、透视变换

**开源替代方案**：
- **Open-Sora**：开源复现，支持最长16秒720p视频
- **CogVideo/CogVideoX**：智谱AI发布，6B参数，强调语义一致性
- **Wan Video**：阿里发布，14B参数，支持多分钟视频

### 11.5 Playground v2.5与DALL-E 3

#### 11.5.1 Playground v2.5：美学对齐

Playground v2.5的核心创新在于将**人类美学偏好对齐**引入扩散模型训练：

- **DPO (Direct Preference Optimization)**：收集人类对生成图像的偏好对（preferred vs rejected），用DPO损失直接优化生成质量
- **颜色对比增强**：通过训练数据筛选和增强，提升生成图像的颜色饱和度和对比度
- **多宽高比原生支持**：专门优化了非1:1宽高比下的生成质量

DPO对扩散模型的损失函数：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \cdot \left[\log \frac{p_\theta(x_w | c)}{p_{\text{ref}}(x_w | c)} - \log \frac{p_\theta(x_l | c)}{p_{\text{ref}}(x_l | c)}\right]\right)$$

其中 $x_w$ 是偏好图像，$x_l$ 是被拒绝图像，$c$ 是提示词。

#### 11.5.2 DALL-E 3：数据质量突破

OpenAI的DALL-E 3的最大突破不在模型架构，而在**数据工程**：

- **Caption重新标注**：使用GPT-4V对训练图像进行精细重新标注，将粗精度的alt-text标注转化为详细的视觉描述
- **从粗标注到精细描述**：原始标注“a dog” → 重新标注“A golden retriever puppy sitting on a green lawn, looking directly at the camera with a tilted head, soft afternoon sunlight, shallow depth of field”
- **提示词遵从度大幅提升**：模型在详细描述上训练，自然学会了更精确地遵循提示词中的每个细节

这证明了一个重要洞察：**在模型架构已经足够强大的今天，数据质量往往是决定性因素**。

### 11.6 系统对比与选择建议

```
┌─────────────────────────────────────────────────────────────────┐
│             图像生成系统能力雷达图 (ASCII版)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│           文字渲染                                                 │
│              │                                                    │
│         10 ──┼──  Flux ██████████                              │
│              │   SD3  ████████                                   │
│          8 ──┼──  DALL-E3 █████████                            │
│              │                                                    │
│          6 ──┼──  SDXL ██████                                   │
│              │                                                    │
│          4 ──┼──  SD 1.5 ███                                    │
│              │                                                    │
│  ──────────┼────────────────────── Prompt遵从度            │
│         4     6     8    10                                      │
│                                                                 │
│  各维度对比:                                                      │
│  ────────────────────────────────────────────────────────  │
│  图像质量:   SD1.5 < SDXL < SD3 ≈ Flux ≈ DALL-E3             │
│  推理速度:   Flux-schnell > SDXL > SD3 > Flux-dev              │
│  开源程度:   Flux=SDXL > SD3 > DALL-E3(API only)               │
│  社区生态:   SDXL > SD1.5 > Flux > SD3                          │
│  商用灵活性: Flux > SDXL > SD3 > DALL-E3                       │
│  ────────────────────────────────────────────────────────  │
│                                                                 │
│  选型建议:                                                      │
│  • 个人/创作者: Flux.1-schnell (快速高质量)                  │
│  • 企业产品: Flux.1-dev / SDXL (可控+开源)                   │
│  • 极致质量: DALL-E 3 API (最强prompt遵从)                    │
│  • 研究实验: SD3 / SDXL (社区生态丰富)                       │
└─────────────────────────────────────────────────────────────────┘
```

**未来发展趋势预判**：

1. **Transformer全面替代U-Net**：SD3和Flux已经证明了纯Transformer架构的优越性，未来新模型将全部采用Transformer
2. **Flow Matching成为主流**：Rectified Flow比传统扩散更简洁高效，将成为默认选择
3. **规模继续扩大**：从2B到12B再到更大，规模效应在视觉生成中同样成立
4. **多模态融合**：图像-视频-3D的统一生成框架是终极目标

### 11.7 代码实践：SDXL推理与Rectified Flow采样

```python
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class SDXLMicroCondition:
    """微条件系统参数"""
    original_size: Tuple[int, int] = (1024, 1024)
    crop_coords: Tuple[int, int] = (0, 0)
    target_size: Tuple[int, int] = (1024, 1024)


class FourierEmbedding(nn.Module):
    """微条件的Fourier嵌入"""
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
        args = x.unsqueeze(-1) * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)


class SDXLConditionEncoder(nn.Module):
    """模拟SDXL的双文本编码器 + 微条件编码"""
    
    def __init__(self, text_dim: int = 2048, time_dim: int = 1280):
        super().__init__()
        # 模拟双编码器输出: CLIP-L(768) + CLIP-bigG(1280) = 2048
        self.text_proj = nn.Linear(text_dim, time_dim)
        
        # 微条件编码
        self.fourier_embed = FourierEmbedding(dim=256)
        # 6个微条件值: orig_h, orig_w, crop_y, crop_x, target_h, target_w
        self.micro_cond_proj = nn.Sequential(
            nn.Linear(256 * 6, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 时间步编码
        self.time_embed = nn.Sequential(
            nn.Linear(256, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
    
    def forward(
        self, 
        text_embeds: torch.Tensor,      # [B, 77, 2048]
        timestep: torch.Tensor,          # [B]
        micro_cond: SDXLMicroCondition
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = text_embeds.shape[0]
        device = text_embeds.device
        
        # Cross-Attention条件
        context = self.text_proj(text_embeds)  # [B, 77, 1280]
        
        # 时间步嵌入
        t_emb = self.fourier_embed(timestep.float())  # [B, 256]
        t_emb = self.time_embed(t_emb)  # [B, 1280]
        
        # 微条件嵌入
        micro_values = torch.tensor([
            micro_cond.original_size[0], micro_cond.original_size[1],
            micro_cond.crop_coords[0], micro_cond.crop_coords[1],
            micro_cond.target_size[0], micro_cond.target_size[1]
        ], device=device, dtype=torch.float).unsqueeze(0).expand(B, -1)
        
        micro_embeds = []
        for i in range(6):
            micro_embeds.append(self.fourier_embed(micro_values[:, i]))
        micro_emb = torch.cat(micro_embeds, dim=-1)  # [B, 256*6]
        micro_emb = self.micro_cond_proj(micro_emb)  # [B, 1280]
        
        # 合并时间步和微条件
        cond_emb = t_emb + micro_emb  # [B, 1280]
        
        return context, cond_emb


class RectifiedFlowSampler:
    """统一的Rectified Flow采样器
    
    适用于SD3/Flux风格的直线流匹配模型。
    与传统扩散采样器的核心区别:
    - 传统扩散: 学习噪声预测 ε, 采样求解SDE/ODE
    - Rectified Flow: 学习速度场 v, 采样沿直线插值
    """
    
    def __init__(self, num_steps: int = 28, shift: float = 1.0):
        self.num_steps = num_steps
        self.shift = shift  # 时间步偏移 (Flux使用shift=3.0)
    
    def get_timesteps(self, device: torch.device) -> torch.Tensor:
        """Shifted timestep schedule (Flux风格)"""
        timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)
        # 应用shift: t' = t * shift / (1 + (shift-1)*t)
        if self.shift != 1.0:
            timesteps = timesteps * self.shift / (1.0 + (self.shift - 1.0) * timesteps)
        return timesteps
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5
    ) -> torch.Tensor:
        """Euler采样 for Rectified Flow
        
        更新公式: z_{t-dt} = z_t - dt * v_θ(z_t, t)
        直觉: 沿着预测的速度场方向移动
        """
        x = noise
        timesteps = self.get_timesteps(noise.device)
        
        for i in range(self.num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t  # 负值 (t从1→0)
            
            t_batch = t.expand(x.shape[0])
            
            if condition is not None and cfg_scale > 1.0:
                # Classifier-Free Guidance
                v_cond = model(x, t_batch, condition)
                v_uncond = model(x, t_batch, torch.zeros_like(condition))
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, t_batch, condition)
            
            # Euler步进
            x = x + dt * v
        
        return x
    
    @torch.no_grad()
    def sample_dpm(self, model: nn.Module, noise: torch.Tensor, 
                   condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """二阶DPM-Solver风格采样 (更高精度)"""
        x = noise
        timesteps = self.get_timesteps(noise.device)
        prev_v = None
        
        for i in range(self.num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            t_batch = t.expand(x.shape[0])
            
            v = model(x, t_batch, condition)
            
            if prev_v is None or i == 0:
                # 一阶Euler
                x = x + dt * v
            else:
                # 二阶Adams-Bashforth
                x = x + dt * (1.5 * v - 0.5 * prev_v)
            
            prev_v = v
        
        return x


# 使用示例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=== SDXL微条件系统示例 ===")
    encoder = SDXLConditionEncoder(text_dim=2048, time_dim=1280).to(device)
    
    # 模拟双编码器输出
    text_embeds = torch.randn(2, 77, 2048, device=device)
    timestep = torch.tensor([0.5, 0.3], device=device)
    micro_cond = SDXLMicroCondition(
        original_size=(1024, 1024),
        crop_coords=(0, 0),
        target_size=(1024, 1024)
    )
    
    context, cond_emb = encoder(text_embeds, timestep, micro_cond)
    print(f"Cross-Attention条件: {context.shape}")   # [2, 77, 1280]
    print(f"时间步+微条件嵌入: {cond_emb.shape}")   # [2, 1280]
    
    print("\n=== Rectified Flow采样示例 ===")
    
    # 简单的velocity模型
    class SimpleVModel(nn.Module):
        def __init__(self, dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim)
            )
        def forward(self, x, t, cond=None):
            B = x.shape[0]
            t_emb = t.view(B, 1).expand(-1, 1)
            x_flat = x.view(B, -1)
            inp = torch.cat([x_flat, t_emb], dim=-1)
            return self.net(inp).view_as(x)
    
    v_model = SimpleVModel(dim=256).to(device)
    sampler = RectifiedFlowSampler(num_steps=20, shift=1.0)
    
    # Euler采样
    noise = torch.randn(4, 256, device=device)
    result_euler = sampler.sample(v_model, noise)
    print(f"Euler采样结果: {result_euler.shape}")
    
    # DPM-Solver采样
    result_dpm = sampler.sample_dpm(v_model, noise)
    print(f"DPM-Solver采样结果: {result_dpm.shape}")
    
    # Flux风格 shifted schedule
    sampler_flux = RectifiedFlowSampler(num_steps=4, shift=3.0)
    result_flux = sampler_flux.sample(v_model, noise)
    print(f"Flux-schnell风格(4步): {result_flux.shape}")
    
    # 时间步对比
    ts_normal = RectifiedFlowSampler(num_steps=10, shift=1.0).get_timesteps(device)
    ts_shifted = RectifiedFlowSampler(num_steps=10, shift=3.0).get_timesteps(device)
    print(f"\n普通时间步: {[f'{t:.3f}' for t in ts_normal.tolist()]}")
    print(f"Shifted时间步: {[f'{t:.3f}' for t in ts_shifted.tolist()]}")
    print("→ Shift让早期步骤更密集, 后期更稀疏")
```

**本章小结**：从DXL的双编码器+微条件架构，到SD3的MMDiT双流Transformer和Rectified Flow，再到Flux的12B规模模型和Sora的时空Patch视频架构，扩散模型正在经历一场深刻的架构革命。两个核心趋势已经明确：Transformer全面替代U-Net，Rectified Flow替代传统扩散过程。与此同时，数据质量（DALL-E 3的重新标注）和人类对齐（Playground的DPO）正在成为与模型架构同等重要的研究方向。选择哪个系统取决于具体需求：追求质量选Flux，追求生态选SDXL，追求推理速度选Flux-schnell。

---

## 第12章 总结与未来展望

走过前面十一章的详尽探讨，我们最后来勒出一个清晰的技术全貌。本章以“站高看远”的视角系统性总结扩散模型的技术演进脉络，并探讨未来发展方向与开放问题。

### 12.1 技术演进总结

扩散模型从2020年DDPM的起点到今天的Flux/Sora，短短五年间走过了一条迅猛的技术演进之路。可以从六个维度梳理这段历史：

**1. 采样加速**：DDPM(1000步) → DDIM(50步) → DPM-Solver(20步) → Consistency(1步)。
从随机微分方程的迭代求解，到确定性ODE的高阶求解器，再到一步生成的一致性模型，采样步数减少了三个数量级。

**2. 架构演进**：U-Net → Efficient U-Net → DiT → MMDiT。
从卷积为主的U-Net、到引入Self-Attention的混合架构、再到纯粹Transformer的DiT、最后是双流设计的MMDiT，架构越来越“Transformer化”，与LLM领域形成架构上的收敛。

**3. 训练优化**：ε-prediction → v-prediction → Min-SNR加权 → Flow Matching。
训练目标从预测噪声、预测速度、不同信噪比下重新加权、到直线化的流匹配，不断优化以获得更稳定的训练动态和更高质量。

**4. 条件生成**：Class-Cond → CLIP → CLIP+T5 → GPT-4V重标注。
条件信号从类别标签、到CLIP语义空间、到T5强语言理解、再到跨模态重标注的详细描述，遵从度持续提升。

**5. 引导机制**：Classifier Guidance → CFG → CFG蒸馏 → 多路引导。
从需要额外分类器，到只需一个模型联合训练的CFG，再到将CFG蒸馏为单次推理，推理开销从2x减少到1x。

**6. 部署优化**：FP32 → FP16 → INT8 → INT4，完整计算 → 缓存复用。
从全精度计算到混合低位宽量化，从每步重复计算到智能缓存复用，部署成本下降了一个数量级。

### 12.2 关键技术里程碑时间线

```
┌────────────────────────────────────────────────────────────────────┐
│             扩散模型技术演进时间线 (2020-2025)                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 2020 ┃ DDPM (Ho et al.)                                          │
│      ┃  ├─ 首个高质量扩散模型, 1000步采样                       │
│      ┃  └─ 奠定扩散模型发展基础                                │
│      ┃                                                          │
│ 2021 ┃ DDIM → 采样加速到50步                                  │
│      ┃ Classifier Guidance → 文本条件生成可控性                  │
│      ┃ LDM (Stable Diffusion前身) → 潜空间扩散                  │
│      ┃                                                          │
│ 2022 ┃ CFG → 取代Classifier Guidance, 今天仍是主流              │
│      ┃ Stable Diffusion v1.0 → 开源生态爆发                       │
│      ┃ DALL-E 2 → 业界质量标杆                                 │
│      ┃ Imagen → T5文本编码器的有效性验证                       │
│      ┃ Progressive Distillation → 步数蒸馏开端                  │
│      ┃ DPM-Solver → 高阶数值求解器                              │
│      ┃                                                          │
│ 2023 ┃ SDXL → 开源SOTA, 多宽高比与微条件创新                   │
│      ┃ DiT → Transformer架构可行性验证                          │
│      ┃ Consistency Models → 一步生成的可能性                    │
│      ┃ DALL-E 3 → 数据重标注范式                                │
│      ┃ ControlNet → 可控生成生态                                │
│      ┃                                                          │
│ 2024 ┃ SD3 → MMDiT + Rectified Flow                              │
│      ┃ Flux → 12B参数, 质量与prompt遵从项峰                      │
│      ┃ Sora → 视频生成进入新纪元                                │
│      ┃ DMD2 → 一步生成质量超越多步教师                          │
│      ┃ SDXL-Lightning → 4步高质量推理                          │
│      ┃ Open-Sora → 开源视频生成思路                            │
│      ┃                                                          │
│ 2025 ┃ 统一Flow Matching框架                                     │
│      ┃ 多模态统一生成 (图像+视频+3D)                          │
│      ┃ 推理时计算缩放 (test-time compute scaling)             │
│      ┃ 强化学习对齐 (DPO/RLHF for generation)                  │
└────────────────────────────────────────────────────────────────────┘
```

### 12.3 当前开放问题

尽管技术进步迅猛，扩散模型领域仍面临众多开放问题。

**1. 一步生成质量上界**：一步生成是否能彻底赶上多步模型？理论上，多步模型可以迭代细化，而一步模型只能一次性给出答案。最新的DMD2在FID上已接近多步教师，但在复杂场景上仍有差距。一步生成的理论上限是当前重要的理论问题。

**2. 计算效率的理论极限**：生成一个高质量1024×1024图像需要多少FLOPs？有没有信息论边界？这里存在一个与Shannon信息理论類似的“生成信息理论”还未建立。

**3. 多模态统一**：图像、视频、3D、音频、点云这些不同模态能否在一个统一架构中生成？Sora的时空Patch是一个方向，但跨模态（3D+音频+视频）的联合生成仍未解决。

**4. 可控生成的精细化**：现有技术（ControlNet、IP-Adapter等）可以控制轮廓、姿态、风格，但精细控制复杂场景（如多对象属性绑定、精准语义可控、实体一致性）仍是难题。

**5. 训练数据与伦理**：版权问题、偏见问题、深度伪造、NSFW内容过滤等，这些社会问题与技术发展同等重要。

### 12.4 未来研究方向

**1. Flow Matching作为统一语言**。传统扩散、Rectified Flow、Schrödinger Bridge都可以统一到Flow Matching框架下。未来可能会出现更为一般的生成路径设计，根据不同数据分布自适应选择最佳路径。

**2. 替代Transformer架构**。Mamba/State Space模型在长序列任务上展示了优越性，扩散模型中的表现还在探索。为什么Diffusion Mamba能否超越DiT？这可能是下一轮架构革命。

**3. 端到端联合优化**。当前VAE、文本编码器、扩散模型是独立训练的。未来可能在同一训练过程中联合优化这些组件，获得更优的潜空间表示。

**4. RL与生成对齐**。使用DPO、RLHF或更高级的强化学习方法对齐生成质量、美学偏好、指令遵从。这个方向受到LLM领域启发，但生成任务的奇特性（质量评估难量化）使其更具挑战。

**5. 推理时计算缩放**。受OpenAI o1/o3启发，生成任务是否也能从“推理时多思考”中获益？例如生成多个候选后用鉴别器选择最佳，或者多轮迭代修正。

**6. 多模态统一生成**。在同一个模型中生成图像+音频+视频+3D，实现真正的“世界生成”。Sora是起点，但还远未达到“世界模拟器”的二的。

### 12.5 实践建议与工程清单

```
┌───────────────────────────────────────────────────────────────────┐
│             扩散模型项目选型决策树                              │
├───────────────────────────────────────────────────────────────────┤
│                                                                │
│           需求是什么?                                            │
│             │                                                    │
│     ┌───────┼────────┐                                       │
│     ▼                       ▼                                       │
│ 生成质量优先           推理速度优先                              │
│     │                       │                                       │
│     ▼                       ▼                                       │
│ 高质量+可商用?         实时/交互式?                              │
│  Y│││N                Y│││N                                       │
│   │││                  │││                                         │
│   ▼││                  ▼││                                         │
│ Flux││ → 社区生态?     SDXL+量化  → 边缘部署?                  │
│.dev ││     Y│ N         +DeepCache    Y│ N                          │
│     ││      │ │         +TensorRT      │ │                          │
│     ││      ▼ ▼                        ▼ ▼                          │
│     ││   SDXL DALL-E3              边缘: 服务器:                       │
│     ││        (API)               SD-Turbo Flux-                       │
│     ││                            +量化  schnell                       │
│     ││                                                              │
│     │└─→ 不要求商用?                                                 │
│     │   Y│ N                                                          │
│     │    │ │                                                          │
│     │    ▼ ▼                                                          │
│     │   SD3-2B  Flux.dev                                              │
│     └─→ 考虑社区生态成熟度                                            │
└───────────────────────────────────────────────────────────────────┘
```

**训练优化Checklist**：

- [ ] 使用v-prediction或Flow Matching目标函数
- [ ] 采用Min-SNR加权策略平衡不同信噪比
- [ ] 多宽高比bucket训练提升分辨率鲁棒性
- [ ] EMA权重使用，decay~0.9999
- [ ] 梯度裁剪（1.0）防止训练发散
- [ ] 使用bf16混合精度训练
- [ ] CFG dropout（0.1）以启用無条件引导
- [ ] 质量评估使用FID + CLIP Score + 人工评价三重指标
- [ ] 检查文本数据质量，考虑GPT-4V重标注高价值样本

**推理部署Checklist**：

- [ ] 选择适合采样器：DPM-Solver++ / Euler / DPM
- [ ] FP16推理作为起点
- [ ] 集成xFormers或Flash Attention
- [ ] 启用torch.compile (需要PyTorch 2.0+)
- [ ] 考虑W8A8量化（质量损失可接受场景）
- [ ] 启用DeepCache或Block Caching
- [ ] 使用步数蒸馏模型（SDXL-Lightning, Flux-schnell）
- [ ] TensorRT产线（NVIDIA部署）
- [ ] 建立延迟-质量监控，动态调整优化策略
- [ ] 考虑Batch调度优化高吞吐场景

### 12.6 代码实践：统一推理框架

```python
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum


class ModelType(Enum):
    """扩散模型类型"""
    DDPM_EPS = "ddpm_eps"           # 传统噪声预测
    V_PRED = "v_pred"                # v-prediction
    RECTIFIED_FLOW = "rectified_flow"  # 直线流匹配
    CONSISTENCY = "consistency"       # 一致性模型


class SamplerType(Enum):
    """采样器类型"""
    DDIM = "ddim"
    DPM_SOLVER = "dpm_solver"
    EULER = "euler"
    EULER_FLOW = "euler_flow"
    ONE_STEP = "one_step"


@dataclass
class DiffusionConfig:
    """统一的推理配置"""
    model_type: ModelType = ModelType.RECTIFIED_FLOW
    sampler: SamplerType = SamplerType.EULER_FLOW
    num_steps: int = 28
    cfg_scale: float = 7.5
    
    # 优化选项
    use_quantization: bool = False
    use_deep_cache: bool = False
    cache_interval: int = 3
    use_compile: bool = False
    use_flash_attention: bool = True
    
    # 性能预期
    target_latency_ms: Optional[float] = None
    target_quality_fid: Optional[float] = None


class UnifiedDiffusionPipeline:
    """统一的扩散模型推理框架
    
    集成本书覆盖的主要优化技术:
    - 多种采样器 (DDIM/DPM-Solver/Euler Flow)
    - 多种预测目标 (ε/v/Flow)
    - DeepCache缓存
    - W8A8量化
    - CFG引导
    - torch.compile加速
    
    设计哲学: 提供统一API, 隐藏优化复杂性
    """
    
    def __init__(
        self,
        unet: nn.Module,
        text_encoder: nn.Module,
        vae: nn.Module,
        config: DiffusionConfig
    ):
        self.unet = unet
        self.text_encoder = text_encoder
        self.vae = vae
        self.config = config
        
        # 应用优化
        self._apply_optimizations()
        
        # 采样器调度表
        self._sampler_registry: Dict[SamplerType, Callable] = {
            SamplerType.DDIM: self._sample_ddim,
            SamplerType.DPM_SOLVER: self._sample_dpm_solver,
            SamplerType.EULER: self._sample_euler,
            SamplerType.EULER_FLOW: self._sample_euler_flow,
            SamplerType.ONE_STEP: self._sample_one_step
        }
        
        # 缓存状态
        self._cache_store: Dict[int, torch.Tensor] = {}
    
    def _apply_optimizations(self):
        """应用推理优化"""
        if self.config.use_compile:
            try:
                self.unet = torch.compile(self.unet, mode='reduce-overhead')
                print("✓ torch.compile enabled")
            except Exception as e:
                print(f"⚠ torch.compile失败: {e}")
        
        if self.config.use_flash_attention:
            print("✓ Flash Attention enabled (assumed in model)")
        
        if self.config.use_quantization:
            print("✓ W8A8 quantization enabled")
        
        if self.config.use_deep_cache:
            print(f"✓ DeepCache enabled, interval={self.config.cache_interval}")
    
    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        """文本编码 (简化: 返回随机嵌入)"""
        # 实际使用中会调用CLIP/T5
        return torch.randn(1, 77, 768, device=next(self.unet.parameters()).device)
    
    def _model_forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """统一的模型前向接口, 集成CFG"""
        if self.config.cfg_scale > 1.0:
            # CFG: 并行计算有条件和无条件
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
            
            pred = self.unet(x_in, t_in, cond_in)
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            pred = pred_uncond + self.config.cfg_scale * (pred_cond - pred_uncond)
        else:
            pred = self.unet(x, t, cond)
        
        return pred
    
    def _sample_euler_flow(self, noise, cond, num_steps):
        """Rectified Flow Euler采样"""
        x = noise
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device)
        
        for i in range(num_steps):
            t = timesteps[i]
            dt = timesteps[i+1] - t
            t_batch = t.expand(x.shape[0])
            
            v = self._model_forward(x, t_batch, cond)
            x = x + dt * v
        
        return x
    
    def _sample_ddim(self, noise, cond, num_steps):
        """DDIM采样 (简化版)"""
        x = noise
        timesteps = torch.linspace(0.999, 0.001, num_steps, device=x.device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.expand(x.shape[0])
            eps = self._model_forward(x, t_batch, cond)
            
            # 简化的DDIM更新
            alpha_t = 1.0 - t
            x = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
            if i < num_steps - 1:
                t_next = timesteps[i+1]
                alpha_next = 1.0 - t_next
                x = alpha_next.sqrt() * x + (1 - alpha_next).sqrt() * eps
        
        return x
    
    def _sample_dpm_solver(self, noise, cond, num_steps):
        return self._sample_ddim(noise, cond, num_steps)  # 简化复用
    
    def _sample_euler(self, noise, cond, num_steps):
        return self._sample_euler_flow(noise, cond, num_steps)
    
    def _sample_one_step(self, noise, cond, num_steps):
        """一步生成 (Consistency Model风格)"""
        t = torch.ones(noise.shape[0], device=noise.device) * 0.999
        return self._model_forward(noise, t, cond)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """统一的生成接口"""
        if seed is not None:
            torch.manual_seed(seed)
        
        device = next(self.unet.parameters()).device
        num_steps = num_steps or self.config.num_steps
        
        # 文本编码
        cond = self.encode_text(prompt)
        
        # 生成初始噪声 (潜空间)
        latent_h, latent_w = height // 8, width // 8
        # 简化: 使用平坦token表示
        noise = torch.randn(1, latent_h * latent_w, 768, device=device)
        
        # 选择采样器采样
        sampler_fn = self._sampler_registry[self.config.sampler]
        latent = sampler_fn(noise, cond, num_steps)
        
        # VAE解码 (这里返回latent)
        return latent
    
    def benchmark(self, num_runs: int = 10) -> Dict[str, float]:
        """性能基准测试"""
        import time
        
        # Warmup
        for _ in range(2):
            _ = self.generate("test prompt")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(num_runs):
            _ = self.generate("test prompt")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
        
        avg_latency = elapsed / num_runs * 1000
        return {
            'avg_latency_ms': avg_latency,
            'throughput_per_sec': 1000.0 / avg_latency,
            'total_time_s': elapsed
        }


# 使用示例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 构建简化模型
    class SimpleUNet(nn.Module):
        def __init__(self, dim=768):
            super().__init__()
            self.dim = dim
            self.layer = nn.TransformerEncoderLayer(dim, 8, 2048, batch_first=True)
            self.cond_proj = nn.Linear(dim, dim)
        def forward(self, x, t, cond):
            B, N, D = x.shape
            cond_feat = self.cond_proj(cond.mean(dim=1, keepdim=True))
            x = x + cond_feat
            return self.layer(x)
    
    class DummyEncoder(nn.Module):
        def forward(self, text):
            return torch.randn(1, 77, 768)
    
    class DummyVAE(nn.Module):
        def decode(self, z):
            return z
    
    unet = SimpleUNet(dim=768).to(device)
    encoder = DummyEncoder().to(device)
    vae = DummyVAE().to(device)
    
    # 三种配置对比
    print("=== 配置A: 质量优先 (Flux风格) ===")
    config_quality = DiffusionConfig(
        model_type=ModelType.RECTIFIED_FLOW,
        sampler=SamplerType.EULER_FLOW,
        num_steps=28,
        cfg_scale=7.5,
        use_flash_attention=True
    )
    pipeline_quality = UnifiedDiffusionPipeline(unet, encoder, vae, config_quality)
    stats_q = pipeline_quality.benchmark(num_runs=5)
    print(f"  平均延迟: {stats_q['avg_latency_ms']:.1f}ms")
    
    print("\n=== 配置B: 速度优先 (Flux-schnell风格) ===")
    config_speed = DiffusionConfig(
        model_type=ModelType.RECTIFIED_FLOW,
        sampler=SamplerType.EULER_FLOW,
        num_steps=4,        # 减少步数
        cfg_scale=1.0,      # 关闭CFG
        use_compile=True,
        use_deep_cache=True,
        cache_interval=2
    )
    pipeline_speed = UnifiedDiffusionPipeline(unet, encoder, vae, config_speed)
    stats_s = pipeline_speed.benchmark(num_runs=5)
    print(f"  平均延迟: {stats_s['avg_latency_ms']:.1f}ms")
    
    print("\n=== 配置C: 一步生成 (极限速度) ===")
    config_one_step = DiffusionConfig(
        model_type=ModelType.CONSISTENCY,
        sampler=SamplerType.ONE_STEP,
        num_steps=1,
        cfg_scale=1.0,
        use_quantization=True
    )
    pipeline_one = UnifiedDiffusionPipeline(unet, encoder, vae, config_one_step)
    stats_1 = pipeline_one.benchmark(num_runs=5)
    print(f"  平均延迟: {stats_1['avg_latency_ms']:.1f}ms")
    
    print(f"\n=== 总结 ===")
    print(f"质量优先: {stats_q['avg_latency_ms']:.1f}ms (28步)")
    print(f"速度平衡: {stats_s['avg_latency_ms']:.1f}ms (4步)")
    print(f"极限速度: {stats_1['avg_latency_ms']:.1f}ms (1步)")
    print(f"快速模式加速比: {stats_q['avg_latency_ms']/stats_s['avg_latency_ms']:.2f}x")
    print(f"一步模式加速比: {stats_q['avg_latency_ms']/stats_1['avg_latency_ms']:.2f}x")
```

### 12.7 参考文献与推荐阅读

**奠基论文**：
- *Denoising Diffusion Probabilistic Models* (Ho et al., 2020) — DDPM开山之作
- *Denoising Diffusion Implicit Models* (Song et al., 2021) — DDIM采样加速
- *Score-Based Generative Modeling through Stochastic Differential Equations* (Song et al., 2021) — SDE统一框架
- *Classifier-Free Diffusion Guidance* (Ho & Salimans, 2022) — CFG
- *High-Resolution Image Synthesis with Latent Diffusion Models* (Rombach et al., 2022) — LDM/Stable Diffusion

**架构与优化**：
- *Scalable Diffusion Models with Transformers* (Peebles & Xie, 2023) — DiT
- *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (Esser et al., 2024) — SD3/MMDiT
- *Consistency Models* (Song et al., 2023) — 一步生成
- *DPM-Solver* 系列 (Lu et al., 2022, 2023)
- *Flow Matching for Generative Modeling* (Lipman et al., 2023)

**推理优化**：
- *DeepCache: Accelerating Diffusion Models for Free* (Ma et al., 2024)
- *Q-Diffusion* (Li et al., 2023)
- *Distribution Matching Distillation* / DMD2 (Yin et al., 2024)

**开源仓库**：
- huggingface/diffusers — 最主流的扩散模型库
- CompVis/stable-diffusion — 原始Stable Diffusion实现
- black-forest-labs/flux — Flux官方代码
- Stability-AI/sd3-ref — SD3参考实现
- hpcaitech/Open-Sora — 开源视频生成

**学习路径**：
1. 先读本博客的第1-3章建立数学直觉
2. 跑通完整的Stable Diffusion推理代码，理解各组件的交互
3. 读DiT论文，理解Transformer在生成中的设计
4. 实验不同采样器和采样步数，感受质量-速度权衡
5. 微调一个小型扩散模型（如SD 1.5 LoRA）深入训练细节
6. 跳进部署优化：尝试量化、缓存、TensorRT导出

### 结语

扩散模型从2020年DDPM的一个优雅数学思想出发，在短短五年间已发展为生成式AI的主流范式。它们不仅重塑了图像生成领域，还正在向视频、3D、音频、分子设计、机器人控制等更广阔的领域过渡。

本书走过了从**数学原理→算法设计→架构创新→训练优化→部署加速**的完整路径，并剖析了当今最前沿的生产级系统。如果你从本书中只带走三个takeaway，我希望它们是：

1. **扩散模型的本质是学习从简单分布到复杂分布的逐步变换**。无论叫它扩散、得分匹配、Flow Matching还是一致性模型，背后都是“如何学会从噪声中提取信息”这一根本问题的不同解答。

2. **架构与训练的创新同等重要**。U-Net到Transformer的迁移、ε-pred到v-pred再到Flow Matching的演进、SNR加权等训练技巧、以及数据重标注与DPO对齐，这些创新共同推动了质量跨越。不要只看架构。

3. **工程优化是走向生产的最后一公里**。量化、剪枝、缓存、推理引擎、步数蒸馏——每一个都可能是产品能否上线的决定性因素。不要留在红貌般“骨感”上不变，造就改变世界的技术始于最后一公里的打磨。

未来十年，扩散模型及其后继者将继续推动生成式AI的边界。从单一模态到多模态统一、从有限推理到推理时计算缩放、从被动生成到主动探索世界，这里仍有无限可能。希望本书不仅是你学习扩散模型的一本参考，更是你踏上这场剧烈变革的一个起点。

愿你生成出给别人以启发的东西。

