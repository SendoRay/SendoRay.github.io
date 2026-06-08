VeRL-Omni 全面解析
  
  一、项目是什么

  VeRL-Omni 是字节跳动 Seed MLSys 团队开源的多模态生成模型的 RL 后训练框架。它不像 Stable Diffusion
  那样做图像生成，而是用强化学习（RL）来微调已经训好的生成模型，让它们生成更高质量的图像/视频/音频。

  简单说：你有一个 Qwen-Image 模型能文生图，但生成质量不够好。VeRL-Omni 用 RL 的方式（比如
  FlowGRPO）来继续训练它，根据 reward（OCR 准确率、人类偏好评分等）来优化模型，让它生成更好的图片。

  二、为什么要做 VeRL-Omni（不用 verl 或 vllm）

  README 里说的很清楚：

  ▎ Multimodal generative RL training differs from text-only LLM RL not only in model structure, but also in I/O patterns, compute characteristics, and runtime bottlenecks.

  具体来说有这些关键差异：

  1. 模型结构不同：LLM 是自回归的（逐 token 生成），而扩散模型是迭代去噪的（逐 step 去噪）。verl 的训练循环是为
   LLM 设计的（token-level log-probs、PPO 在 token 维度做），不能直接用于扩散模型。
  2. 数据 I/O 模式不同：LLM 产生 token 序列，扩散模型产生高维 latent tensor（图片 latent 是 (B, C, H, W) 甚至
  (B, C, T, H, W) 的视频）。带宽和存储需求完全不同。
  3. 计算瓶颈不同：LLM rollout 的瓶颈在 KV cache 和 attention；扩散模型 rollout 的瓶颈在 VAE 编解码、多步
  UNet/DiT 推理。
  4. log-prob 计算方式完全不同：LLM 的 log-prob 直接从 softmax 输出拿到；扩散模型需要 SDE 逆过程 计算每步
  transition 的 log-prob。这是扩散 RL 核心的数学创新。
  5. vLLM 是为 LLM 推理优化的，它不懂扩散模型。因此需要 vLLM-Omni 来做扩散 rollout，提供高吞吐的扩散推理。
  6. 吞吐优势：README 说比 flow_grpo 实现快 ~25%，来自 vLLM-Omni rollout + FSDP 训练 + 异步 reward 计算的重叠。

  三、模型结构和支持情况

  支持的模型

  模型: Qwen-Image
  架构: QwenImageTransformer2DModel（DiT）
  模态: Text→Image
  算法: FlowGRPO, MixGRPO, GRPO-Guard, DiffusionNFT, DPO
  ────────────────────────────────────────
  模型: Wan2.2
  架构: WanTransformer3DModel（3D DiT）
  模态: Text→Video
  算法: DanceGRPO
  ────────────────────────────────────────
  模型: SD3.5
  架构: MMDiT（双流 DiT）
  模态: Text→Image
  算法: DPO
  ────────────────────────────────────────
  模型: BAGEL（WIP）
  架构: 统一理解+生成
  模态: Text+Image
  算法: FlowGRPO
  ────────────────────────────────────────
  模型: Qwen3-Omni（WIP）
  架构: 全模态统一
  模态: Text/Image/Video/Audio
  算法: GSPO
  ────────────────────────────────────────
  模型: HunyuanImage-3.0（计划）
  架构: 统一理解+生成
  模态: Text+Image
  算法: MixGRPO, SRPO
  ────────────────────────────────────────
  模型: LTX2.3（WIP）
  架构: 扩散生成器
  模态: Text→Video+Audio
  算法: FlowGRPO

  Qwen-Image 模型结构（核心支持模型）

  Qwen-Image 是一个 Flow Matching DiT（Diffusion Transformer）：

  1. VAE Encoder：将图像 (B, 3, H, W) 压缩到 latent (B, C, H/8, W/8)
  2. Text Encoder（T5/ByT5）：将文本 prompt 编码为 prompt_embeds (B, L, D)
  3. QwenImageTransformer2DModel：核心 DiT，输入 noisy latent + timestep + text embeddings，输出 noise
  prediction（也叫 velocity/flow prediction）
  4. VAE Decoder：将去噪后的 latent 解码回像素空间

  SDE 逆过程是 VeRL-Omni 的核心数学创新。标准扩散推理用 ODE（确定性），但 RL 训练需要随机性才能计算
  log-prob。FlowMatchSDEDiscreteScheduler（verl_omni/pipelines/schedulers/flow_match_sde.py）实现了三种 SDE：

  - "sde"（FlowGRPO）：std_dev_t = sqrt(sigma / (1-sigma)) * noise_level
  - "cps"（Conservative Proposal Step）：std_dev_t = sigma_prev * sin(noise_level * pi/2)
  - "dance_sde"（DanceGRPO）：基于 score-based SDE correction，数值更稳定

  每一步的 log-prob 计算为高斯分布的对数密度：
  log_prob = -(prev_sample - mean)^2 / (2 * std^2) - log(std) - log(sqrt(2*pi))

  四、怎么做——完整架构

  整个系统架构分 7 个阶段：

  1. 数据加载

  - StatefulDataLoader 从 parquet 文件读取 prompts（文本描述+ground truth）
  - 支持断点续训（checkpoint 恢复 dataloader 状态）

  2. Rollout（生成）— vLLM-Omni

  - vLLM-Omni Async Server 作为 Ray actor 部署
  - DiffusionAgentLoopWorker 把 prompt 发给 vLLM-Omni server
  - vLLM-Omni 的 Custom Pipeline（如 QwenImagePipelineWithLogProb）跑完整 SDE 去噪循环
  - SDE Window：只在指定的 timestep 窗口内注入噪声+收集 log-prob（比如 start=0, end=5 步），其余步用
  ODE（确定性）。这节省了计算和内存。
  - 返回：生成的图片 + all_latents（每个步骤的 latent）+ all_log_probs + all_timesteps + prompt embeddings

  3. Reward 计算

  - VisualRewardManager / MultiVisualRewardManager 对生成的图片打分
  - 支持多种 reward：
    - 规则型：OCR 准确率、JPEG 压缩率
    - 模型型：HPSv3（人类偏好评分）、UnifiedReward 2.0（Alignment + Coherence + Style）
    - HTTP 远程：外部 reward server
  - 异步计算：reward loop worker 和 rollout 可以重叠执行

  4. Old Log-Prob 重算

  - 用当前 actor 模型的权重，在 rollout 产生的 latent 轨迹上重新做 forward + SDE 逆过程
  - 得到 old_log_probs（rollout 时的策略下的 log-prob）
  - 这是 PPO 风格的 proximal optimization 必需的一步

  5. Advantage 估计（GRPO 风格）

  - Group-relative：对同一个 prompt 生成的多个样本（如 n=8），以组为单位归一化 reward
  - advantage = (reward - group_mean) / group_std（可配置是否除以 std）
  - 这样做的好处是不需要单独的 value function（critic）

  6. Actor 更新（FSDP 训练引擎）

  - DiffusersFSDPEngine：使用 PyTorch FSDP/FSDP2 做分布式训练
  - 支持 Ulysses Sequence Parallelism（把序列维度切分到多 GPU）
  - 支持 LoRA（lora_rank=64 等），大幅减少训练参数量
  - 三种子引擎：
    - PPODiffusersFSDPEngine：policy gradient 方法（FlowGRPO 等）
    - DPODiffusersFSDPEngine：DPO 方法
    - NFTDiffusersFSDPEngine：DiffusionNFT 方法（需要三套预测：old + current + ref）
  - 也支持 VeOmni 引擎（另一个训练后端，支持 FSDP2 风格 sharding）

  7. Weight Sync + Checkpointing

  - 训练完的权重同步回 rollout replica（通过 CheckpointEngine 异步同步或内存直接同步）
  - 定期保存 checkpoint（模型、优化器、dataloader 状态）

  五、各算法原理

  FlowGRPO（diffusion_algos.py:270）

  最核心的算法。PPO-style clipped objective：
  ratio = exp(log_prob_current - log_prob_old)
  loss = max(-adv * ratio, -adv * clip(ratio, 1-eps, 1+eps))
  就是标准的 PPO clip，但 log_prob 来自 SDE 逆过程而非 token 采样。

  GRPO-Guard（diffusion_algos.py:362）

  在 FlowGRPO 基础上加一个 "ratio-mean bias" 项，惩罚当前策略的 reverse-SDE mean 偏离 rollout 策略的 mean：
  ratio_mean_bias = ||prev_sample_mean - old_prev_sample_mean||^2 / (2 * scale^2)
  scale = sqrt_dt * std_dev_t，这使得 loss 在不同 timestep 上的梯度量级一致。

  DPO（diffusion_algos.py:486）

  直接偏好优化，不需要显式的 reward model。比较 chosen vs rejected 样本的 noise prediction 误差：
  loss = -log_sigmoid(-0.5 * beta * (model_err_chosen - ref_err_chosen - model_err_rejected +
  ref_err_rejected))
  
  DiffusionNFT（diffusion_algos.py:677）

  前向过程的 DPO（Forward-process direct-preference）。在 forward noising 方向上做优化：x0 -> xt 加噪后预测
  x0，用 positive/negative 预测的加权组合。

  六、双注册机制——可扩展性

  VeRL-Omni 的可扩展性来自于两个注册表（verl_omni/pipelines/model_base.py）：

  1. DiffusionModelBase：训练侧适配器，(architecture, algorithm) → 具体的训练逻辑
  2. VllmOmniPipelineBase：推理侧管道，(architecture, algorithm) → vLLM-Omni 自定义 pipeline

  要支持一个新模型，你只需要：
  1. 实现 DiffusionModelBase 子类（build_scheduler, set_timesteps, prepare_model_inputs,
  forward_and_sample_previous_step）
  2. 实现 vLLM-Omni 自定义 pipeline（diffuse 方法）
  3. 注册到对应 (architecture, algorithm) key 下

  配置系统中只需指定 external_lib=your.module.path 即可触发自动注册。

✻ Baked for 2m 43s

※ recap: You asked me to explain the verl-omni project in detail — its purpose, architecture, model support, 
  and design rationale. I delivered a comprehensive breakdown in Chinese covering all of it. No active task is 
  in progress; let me know if you want to dive deeper into any specific part. (disable recaps in /config)





好的，我来从零开始系统地给你讲。假设你精通 LLM 但完全不懂扩散模型。

  ---
  第一部分：扩散模型基础 —— 从 LLM 人的视角
  
  1.1 核心直觉：扩散模型是什么

  LLM 是自回归生成：给定 [我, 爱]，预测下一个 token [吃]，再预测 [苹果]...逐 token 串行。

  扩散模型是迭代去噪生成：先画一张纯噪声图，然后一步一步"擦掉"噪声，露出清晰的图像。

  LLM 推理:
    [BOS] → [我] → [爱] → [吃] → [苹] → [果] → [EOS]
    每一步: 选下一个 token (discrete sampling from softmax)

  扩散模型推理 (去噪):
    纯噪声 z_T → z_{T-1} → z_{T-2} → ... → z_0 (清晰 latent) → VAE decode → 图像
    每一步: 对 latent tensor 做连续数值的更新 (continuous denoising)

  关键对应：

  ┌──────────────────────────────────────┬───────────────────────────────────────────────┐
  │               LLM 概念               │                 扩散模型概念                  │
  ├──────────────────────────────────────┼───────────────────────────────────────────────┤
  │ token                                │ latent pixel/tensor element                   │
  ├──────────────────────────────────────┼───────────────────────────────────────────────┤
  │ autoregressive sampling (逐个 token) │ iterative denoising (逐步去噪，通常 20-50 步) │
  ├──────────────────────────────────────┼───────────────────────────────────────────────┤
  │ vocabulary size V                    │ latent 是连续值 (R^d)                         │
  ├──────────────────────────────────────┼───────────────────────────────────────────────┤
  │ softmax → categorical sampling       │ 高斯分布 → continuous sampling                │
  ├──────────────────────────────────────┼───────────────────────────────────────────────┤
  │ KV cache                             │ 无（每步独立，但 latent 尺寸大）              │
  ├──────────────────────────────────────┼───────────────────────────────────────────────┤
  │ next-token log-prob                  │ SDE transition log-prob                       │
  └──────────────────────────────────────┴───────────────────────────────────────────────┘

  ---
  1.2 扩散模型的两大流派
  
  DDPM (Denoising Diffusion Probabilistic Models)

  - 2020 年 Ho et al. 提出
  - 前向过程：逐步加高斯噪声 x_t = sqrt(1-β_t) * x_{t-1} + sqrt(β_t) * ε
  - 反向过程：学习去预测噪声 ε_θ(x_t, t)
  - 数学基础：离散时间步的马尔可夫链

  Flow Matching / Continuous-time Diffusion

  - 2023 年 Lipman et al. (Flow Matching) 和 Song et al. (Score-based SDE) 统一
  - 关键思想：扩散过程是连续时间的，由 ODE/SDE 描述
  - 训练目标：预测 velocity field v_θ(x_t, t) 或 score function ∇log p_t(x)
  - VeRL-Omni 主要使用 Flow Matching 框架

  为什么要 Flow Matching？
  - DDPM 把时间离散化了，步数少则质量差
  - Flow Matching 是连续时间框架，可以用任意步数采样（ODE solver），且数学上更"干净"——利于 RL 训练中计算
  log-prob（这个后面重点讲）

  ---
  1.3 Flow Matching 完整计算过程（带伪代码）
  
  阶段 A：训练（Pre-training，不是 RL）

  Flow Matching 的训练目标和扩散模型本质一样，只是参数化方式不同。

  直觉：你有一个清晰的图像 x_0（来自数据集），你随机挑一个时间 t ∈ [0,1] 和一个噪声水平 σ_t，然后：
  x_t = t * x_0 + (1-t) * ε    # 线性插值：t=0 时是纯噪声，t=1 时是清晰图
  模型要学习的是velocity（速度场） v = x_0 - ε，即"从噪声到清晰图像的方向"。

  # ============================================
  # Flow Matching 预训练伪代码
  # ============================================

  # x_0: 清晰图像的 latent，shape (B, C, H, W)，来自 VAE encode
  # text_embedding: 文本 prompt 的 embedding，shape (B, L, D)
  # model: DiT (Diffusion Transformer)，就是我们训的模型

  for x_0, text_embedding in dataloader:
      B = x_0.shape[0]

      # 1. 随机采样时间 t ∈ [0, 1]
      t = torch.rand(B, device=device)  # shape (B,)

      # 2. 采样噪声
      epsilon = torch.randn_like(x_0)   # 纯高斯噪声

      # 3. 线性插值：从噪声到清晰图
      #    t=0 → 纯噪声 epsilon
      #    t=1 → 清晰图 x_0
      #    中间 → 噪声和信号的混合
      x_t = (1 - t) * epsilon + t * x_0   # shape (B, C, H, W)
      # 注意：这里用 (1-t) 而不是 t，是 convention 差异，本质一样

      # 4. 目标 velocity：x_0 和 ε 的差
      target_velocity = x_0 - epsilon     # shape (B, C, H, W)

      # 5. 模型预测
      #    输入: x_t (噪声图) + t (时间) + text_embedding (文本条件)
      #    输出: predicted velocity v_θ(x_t, t, text)
      v_pred = model(
          hidden_states=x_t,
          timestep=t,                     # 时间条件
          encoder_hidden_states=text_embedding,  # 文本条件
      )  # shape (B, C, H, W)

      # 6. 简单的 MSE Loss！
      loss = F.mse_loss(v_pred, target_velocity)
      loss.backward()
      optimizer.step()

  重点：Flow Matching 的预训练 loss 就是 MSE！跟 LLM 的 cross-entropy loss 一样简单。因为 target x_0 - ε
  是已知的，模型只需要做回归。

  阶段 B：推理/采样（Inference/Sampling）

  训好模型后，怎么用？用 ODE solver 从噪声一步步去噪。

  # ============================================
  # Flow Matching 推理 (ODE sampling)
  # ============================================

  # 输入: text_embedding (文本 prompt)
  # 输出: 生成的 latent x_0

  # 1. 从纯噪声开始
  x_1 = torch.randn(B, C, H, W)  # t=1 → 纯噪声（注意 convention: t=1 是噪声）

  # 2. 时间步长列表 (由 scheduler 决定)
  #    例如 num_inference_steps=50, sigmas = [1.0, 0.98, 0.96, ..., 0.02, 0.0]
  sigmas = scheduler.sigmas  # len=51 (包含 0)，共 50 步

  # 3. 迭代去噪
  for step in range(num_inference_steps):
      sigma = sigmas[step]       # 当前噪声水平
      sigma_next = sigmas[step + 1]  # 下一噪声水平

      # 模型预测 velocity
      v_pred = model(
          hidden_states=x_t,
          timestep=sigma,              # 用 sigma 而不是 t
          encoder_hidden_states=text_embedding,
      )

      # Euler ODE step: x_{t-1} = x_t + (sigma_next - sigma) * v_pred
      # 因为 dσ < 0 (从大到小), dt = sigma_next - sigma < 0
      # 所以 x 更新方向是 v_pred 的方向
      x_t = x_t + (sigma_next - sigma) * v_pred

  # 4. VAE decode latent → 像素空间
  image = vae.decode(x_t)
  return image

  关键理解：Flow Matching 的采样（用 ODE）是确定性的！给定相同的初始噪声和 prompt，每次生成完全一样的图。这跟
  LLM 采样（有 randomness from softmax）完全不同。

  ---
  第二部分：为什么 RL 训练需要 SDE —— VeRL-Omni 的核心创新
  
  2.1 LLM RL 中的 log-prob

  在 LLM RL 中，log-prob 非常直接：
  # LLM: log_prob(y_t | x, y_{<t})
  logits = model(input_ids)              # (B, seq_len, vocab_size)
  log_probs = F.log_softmax(logits, dim=-1)
  token_log_prob = log_probs[:, t, token_id]  # 很简单！

  2.2 扩散模型 RL 中的 log-prob

  扩散模型没有 softmax 输出！每一步输出的是一个连续向量（velocity/noise prediction）。那 log-prob 怎么算？

  答案：把扩散模型的去噪过程当作一个随机微分方程（SDE）的离散化，然后计算高斯转移概率。

  这就是 VeRL-Omni 的 FlowMatchSDEDiscreteScheduler 做的事情。

  2.3 ODE vs SDE

  ODE (确定性):        SDE (随机性):
    dx = v(x,t) dt       dx = [v(x,t) + 噪声项] dt + 扩散项 dw

    固定输入 → 固定输出   固定输入 → 随机输出（可以算概率）

  FlowGRPO 的核心思想：
  1. 预训练用 ODE（MSE loss，简单高效）
  2. RL 训练用 SDE（需要随机性来定义 log-prob）
  3. 只在部分步（SDE Window）注入噪声，其余步用 ODE

  2.4 SDE 步的计算过程（完整数学+伪代码）

  # ============================================
  # SDE Reverse Step (FlowGRPO / "sde" 模式)
  # 这是 VeRL-Omni 最核心的计算
  # ============================================

  def sde_reverse_step(x_t, sigma, sigma_next, model_output, noise_level):
      """
      单步 SDE 逆过程
      
      输入:
          x_t: 当前 noisy latent, shape (B, C, H, W)
          sigma: 当前噪声水平 (float)
          sigma_next: 下一步噪声水平 (float), sigma_next < sigma
          model_output: 模型预测的 velocity v_θ(x_t, sigma), shape (B, C, H, W)
          noise_level: SDE 噪声强度 η ∈ [0, 1]
      
      输出:
          x_next: 下一步 latent x_{t-1}
          log_prob: 转移概率 log p(x_{t-1} | x_t), shape (B,)
          mean: 转移分布的均值
          std: 转移分布的标准差
      """

      dt = sigma_next - sigma  # 负数（sigma 减小）

      # ==========================================
      # Step 1: SDE 噪声方差 (这是关键!)
      # ==========================================
      # std_dev_t 决定了注入多少随机性
      sigma_max = scheduler.sigmas[1]  # 最大 sigma (~1.0)

      # 公式: std_dev_t = sqrt(sigma/(1-sigma)) * noise_level
      # 当 sigma→1: 分母→0，但 clamp 到 sigma_max
      std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma==1, sigma_max, sigma))) * noise_level
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

  2.5 SDE Window —— 为什么不全用 SDE？

  Timestep:   T ──────────────────────────────────────→ 0
              噪声                                      清晰

  SDE Window:  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
              只有这 4 步      其余都用 ODE (确定性的)
              注入噪声+收集     (快, 不需要 log-prob)
              log-prob

  为什么要 window？
  1. 计算开销：每步 SDE 需要存 latents + 额外计算，全部用 SDE 代价大
  2. 方差控制：只在噪声较大的步注入随机性（这些步方差大，梯度信号好），接近清晰的步用 ODE
  3. 经验发现：window_size=4-8 就够了，性能和全 SDE 几乎一样

  2.6 RL 训练中 log-prob 怎么用？

  回到 diffusion_algos.py:270 的 FlowGRPO Loss：

  # ============================================
  # FlowGRPO RL 训练的完整流程
  # ============================================

  # --- Phase 1: Rollout (vLLM-Omni) ---
  # 输入: prompt
  # 输出: image + rollout_log_probs (在 SDE window 内收集)

  # --- Phase 2: Reward ---
  reward = compute_reward(image, ground_truth)  # e.g. OCR accuracy, HPSv3

  # --- Phase 3: Old log-prob recompute (用当前模型权重) ---
  # 遍历 SDE window 内的每一步:
  for step in sde_window:
      v_pred = model(x_t, sigma, prompt)
      x_next, old_log_prob_step, _, _ = sde_reverse_step(...)
  old_log_probs = sum over steps

  # --- Phase 4: Advantage (GRPO) ---
  # 同一 prompt 的多个样本 (n=8)
  group_rewards = rewards[group_mask]
  advantage[i] = (reward[i] - group_rewards.mean()) / group_rewards.std()

  # --- Phase 5: Actor forward + Loss ---
  # 再次遍历 SDE window:
  for step in sde_window:
      v_pred = model(x_t, sigma, prompt)
      x_next, log_prob_step, _, _ = sde_reverse_step(...)
  log_probs = sum over steps

  # PPO-style clipped objective
  ratio = exp(log_probs - old_log_probs)
  loss = max(-adv * ratio, -adv * clip(ratio, 1-ε, 1+ε))
  loss.backward()

  ---
  第三部分：关键术语表（扩散模型 + VeRL-Omni）
  
  术语: Latent
  解释: 图像在 VAE 压缩空间的表示，shape (B,C,H,W)                
  LLM 类比: token embedding (B,L,D)
  ────────────────────────────────────────
  术语: Timestep / Sigma
  解释: 扩散过程的时间参数，σ 越大越"噪声"
  LLM 类比: 生成步骤 t
  ────────────────────────────────────────
  术语: Velocity (v)
  解释: Flow Matching 中的预测目标 = x_0 - ε
  LLM 类比: 类似预测 "下一个 token 的 logits"
  ────────────────────────────────────────
  术语: Score Function
  解释: ∇log p_t(x)，指向高概率密度区域的方向
  LLM 类比: 无直接类比
  ────────────────────────────────────────
  术语: ODE / SDE
  解释: 常微分方程 / 随机微分方程
  LLM 类比: ODE=贪心解码, SDE=带温度采样
  ────────────────────────────────────────
  术语: CFG (Classifier-Free Guidance)
  解释: 用 pred + scale * (pred_pos - pred_neg) 增强文本对齐
  LLM 类比: 类似 contrastive decoding
  ────────────────────────────────────────
  术语: True-CFG
  解释: 需要明确的无条件 prompt 预测
  LLM 类比: 同上但需要两个 forward
  ────────────────────────────────────────
  术语: DiT (Diffusion Transformer)
  解释: 用 Transformer 架构的扩散模型
  LLM 类比: Decoder-only Transformer
  ────────────────────────────────────────
  术语: VAE
  解释: 图像→latent 的编解码器
  LLM 类比: Tokenizer (BPE)
  ────────────────────────────────────────
  术语: SDE Window
  解释: 只在这几步注入随机性+收集 log-prob
  LLM 类比: N/A
  ────────────────────────────────────────
  术语: noise_level (η)
  解释: 控制 SDE 噪声强度
  LLM 类比: 类似 temperature
  ────────────────────────────────────────
  术语: num_inference_steps
  解释: 去噪步数（通常 20-50）
  LLM 类比: max_new_tokens
  ────────────────────────────────────────
  术语: FlowGRPO
  解释: Flow Matching + Group Relative Policy Optimization
  LLM 类比: PPO / GRPO
  ────────────────────────────────────────
  术语: DanceGRPO
  解释: Dance SDE 变体 + GRPO，数值更稳定
  LLM 类比: 同上
  ────────────────────────────────────────
  术语: GRPO-Guard
  解释: FlowGRPO + reverse-SDE mean 漂移惩罚
  LLM 类比: PPO + KL penalty
  ────────────────────────────────────────
  术语: DPO (diffusion)
  解释: 直接比较 chosen vs rejected 的 noise pred 误差
  LLM 类比: DPO for LLM
  ────────────────────────────────────────
  术语: DiffusionNFT
  解释: 前向过程 DPO（noise 清晰图然后预测 x_0）
  LLM 类比: N/A
  ────────────────────────────────────────
  术语: Ulysses SP
  解释: 序列并行（把 latent 的 spatial 维度切到不同 GPU）
  LLM 类比: Tensor Parallel 但切 seq 维度

  ---
  第四部分：VeRL-Omni 中两个不同 log-prob 的区分
  
  这是最容易被搞混的地方：

  ┌────────────┬─────────────────────────┬───────────────────────────┬────────────────────────────┐
  │            │    rollout_log_prob     │       old_log_prob        │          log_prob          │
  ├────────────┼─────────────────────────┼───────────────────────────┼────────────────────────────┤
  │ 谁算的     │ vLLM-Omni (推理 server) │ Actor (训练 engine)       │ Actor (训练 engine)        │
  ├────────────┼─────────────────────────┼───────────────────────────┼────────────────────────────┤
  │ 用的模型   │ 当前 rollout 权重       │ 当前 actor 权重（更新后） │ 当前 actor 权重            │
  ├────────────┼─────────────────────────┼───────────────────────────┼────────────────────────────┤
  │ 用的轨迹   │ 随机采样 (SDE 有噪声)   │ 同一轨迹（latent 复用）   │ 同一轨迹（latent 复用）    │
  ├────────────┼─────────────────────────┼───────────────────────────┼────────────────────────────┤
  │ 什么时候算 │ Rollout 阶段            │ 更新前                    │ 更新时（forward+backward） │
  ├────────────┼─────────────────────────┼───────────────────────────┼────────────────────────────┤
  │ 作用       │ metadata，不用于 loss   │ 用于 ratio 分母           │ 用于 ratio 分子            │
  └────────────┴─────────────────────────┴───────────────────────────┴────────────────────────────┘

  # 这三个 log_probs 的复用关系
  ratio = exp(log_prob - old_log_prob)
  # old_log_prob: 模型参数变了但 latent 轨迹不变，重新算一遍
  # log_prob: 当前 forward 的结果，有 grad
  # rollout_log_prob: rollout 原始轨迹的 log-prob（只用于 ref 计算或 debug）

  ---
  第五部分：模型架构 —— 从 LLM 人的视角看 DiT
  
  Qwen-Image 的 Transformer（DiT）

  输入:
    Noisy Latent: (B, C_latent, H, W)
    Timestep: (B,) → embedding → (B, D)
    Text Embedding: (B, L, D)  ← 来自 T5/Byt5 text encoder

                 ↓
    ┌─────────────────────────────────────────┐
    │  Patch Embed: 把 latent 切成 patch      │
    │  (B, C, H, W) → (B, H*W/p², D)         │
    │  类似 ViT 的 patch embedding            │
    └─────────────────────────────────────────┘
                 ↓
    ┌─────────────────────────────────────────┐
    │  N × DiT Block (类似 Decoder Layer)      │
    │  ┌─────────────────────────────────┐    │
    │  │  AdaLN (Adaptive Layer Norm)     │    │  ← 跟 LLM 不同！
    │  │  用 timestep embedding 调控      │    │    不是 Pre-LN/Post-LN
    │  │  LN 的 scale 和 shift            │    │    而是 adaptive
    │  └─────────────────────────────────┘    │
    │  ┌─────────────────────────────────┐    │
    │  │  Self-Attention (标准 MHA)       │    │
    │  └─────────────────────────────────┘    │
    │  ┌─────────────────────────────────┐    │
    │  │  Cross-Attention (可选)          │    │  ← 有些模型用
    │  │  Query: image tokens             │    │    Qwen-Image 是
    │  │  Key/Value: text embeddings      │    │    MM-DiT，有 text path
    │  └─────────────────────────────────┘    │
    │  ┌─────────────────────────────────┐    │
    │  │  FFN (MLP)                      │    │
    │  └─────────────────────────────────┘    │
    └─────────────────────────────────────────┘
                 ↓
    ┌─────────────────────────────────────────┐
    │  Unpatch: (B, H*W/p², D) → (B, C, H, W) │
    └─────────────────────────────────────────┘
                 ↓
    输出: velocity prediction v_θ (B, C_latent, H, W)

  与 LLM 的关键区别：
  1. AdaLN 代替 Pre-LN：LN 的参数由 timestep embedding 动态生成（scale(t), shift(t)），不是固定参数
  2. Cross-Attention：image queries attend to text keys/values（Qwen-Image 中 text 也走相同的 transformer
  blocks）
  3. 没有 causal mask：latent patch 之间是全 attention（双向），不需要 mask
  4. 输入是连续 latent 不是离散 token：不需要 embedding lookup
  5. 输出维度 = 输入维度：预测 velocity（同尺寸 tensor），不是下一个 token 的概率分布

  ---
  第六部分：推荐阅读资源
  
  必读论文

  1. Flow Matching for Generative Modeling (Lipman et al., 2023)
  https://arxiv.org/abs/2210.02747
  这是 Flow Matching 框架的奠基论文，写得很清晰
  2. FlowGRPO: Training Diffusion Models with Reinforcement Learning (2025)
  https://arxiv.org/abs/2505.05470
  这是 VeRL-Omni 核心算法的论文
  3. Denoising Diffusion Probabilistic Models (Ho et al., 2020)
  https://arxiv.org/abs/2006.11239
  DDPM 原始论文，虽然是离散框架但是非常好的入门
  4. Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., 2021)
  https://arxiv.org/abs/2011.13456
  统一了 Score-based 和 Diffusion 模型，SDE 框架的数学基础
  5. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stability AI, 2024)
  https://arxiv.org/abs/2403.03206
  SD3 的论文，Rectified Flow = Flow Matching，实战视角

  博客 / 教程

  6. What are Diffusion Models? (Lilian Weng, 2021)
  https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
  可能是最好的扩散模型入门博客，数学完整但易懂
  7. Understanding Diffusion Models: A Unified Perspective (Calvin Luo, 2022)
  https://calvinyluo.com/2022/08/26/diffusion-tutorial.html
  从 VAE 出发推导扩散模型，非常 pedagogic
  8. Flow Matching: Matching Flows instead of Scores (Yang Song, 2023)
  https://yang-song.net/blog/2023/flow-matching/
  Flow Matching 作者的博客，比论文更直观的讲解
  9. Diffusion Models from Scratch (Hugging Face Diffusers tutorial)
  https://huggingface.co/docs/diffusers/tutorials/basic_training
  用 diffusers 库训练扩散模型的实战教程
  10. Anime Keeps Making Diffusion Easier: GRPO for Image Generation (Lambda blog, 2025)
  https://lambdalabs.com/blog
  实战视角讲 Diffusion RL，通俗易懂

  关键 GitHub 仓库

  11. flow_grpo (原始 FlowGRPO 实现)
  https://github.com/yifan123/flow_grpo
  VeRL-Omni 的灵感来源，单文件 SD3 训练代码，比 VeRL-Omni 简单得多
  12. HuggingFace Diffusers
  https://github.com/huggingface/diffusers
  扩散模型的标准库，pipeline 设计是 VeRL-Omni 适配层的基础

  ---
  总结：你作为 LLM Infra 人的学习路线

  第1步: 理解扩散模型基础
    → Lilian Weng 博客 + DDPM 论文

  第2步: 理解 Flow Matching
    → Yang Song 博客 + Flow Matching 论文 (Lipman 2023)
    核心: Flow Matching = 用 MSE loss 直接回归 velocity = x_0 - ε

  第3步: 理解 SDE 和 log-prob
    → Score-based SDE 论文 (Song 2021)
    → VeRL-Omni 的 flow_match_sde.py (308 行，精读)
    核心: SDE 给出转移分布 → 噪声注入 → 高斯密度 = log-prob

  第4步: 理解 RL 训练流程
    → FlowGRPO 论文
    → flow_grpo GitHub (先看简单的，再回来看 VeRL-Omni)
    → VeRL-Omni 的 diffusion_algos.py (核心 loss 函数)

  第5步: 理解工程架构
    → VeRL-Omni 的 ray_diffusion_trainer.py (训练循环)
    → engine_workers.py (Actor + Rollout 的 colocated worker)
    → vllm_omni_async_server.py (Rollout server)

  最重要的一个 insight：扩散模型 RL 和 LLM RL 的本质是一样的——都是 PPO 的 ratio * advantage。唯一不同的是
  log-prob 的来源：LLM 来自 softmax 输出，扩散模型来自 SDE 高斯转移概率。理解了这个，其他都是工程细节。

✻ Worked for 1m 59s

───────────────────────────────────────────────────────────────────────────────────────────────────────────────
❯  
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ? for shortcuts



好，我现在有足够的信息来完整回答你的两个问题。

  ---
  一、DiT（Diffusion Transformer）模型结构 —— 拆开看
  
  先要有全局观，然后逐层拆解。Qwen-Image 是一个 20B 参数的 MM-DiT（多模态扩散 Transformer），三层结构：

  1.1 全局架构 ASCII 图

                              ┌─────────────────────────────┐
                              │      用户输入 Prompt          │
                              │   "一只猫坐在沙发上，旁边写着   │
                              │       'Hello World'"         │
                              └─────────────┬───────────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      │                                           │
                      ▼                                           ▼
      ┌──────────────────────────────┐         ┌──────────────────────────────┐
      │   Qwen2.5-VL (冻结，不做训练) │         │   纯噪声 ε ~ N(0, I)          │
      │   语义编码器                   │         │   shape: (B, 64, H/8, W/8)    │
      │                              │         │   比如 (1, 64, 128, 128)       │
      │   文本 → tokenize → embed     │         └──────────────┬───────────────┘
      │   output: text_emb            │                        │
      │   shape: (B, L_text, 3584)    │                        │
      │   例如 L_text=512 tokens      │                        │
      └──────────────┬───────────────┘                        │
                     │                                        │
                     │  text_emb            初始噪声 latent     │
                     │  (B, L_text, D)      (B, C, H_lat, W_lat)
                     │                     也叫 x_T               │
                     │                        │                   │
                     └───────────┬────────────┘                   │
                                 │                                │
                                 ▼                                │
                ┌────────────────────────────────────┐             │
                │      Patch Embedding (图像侧)       │             │
                │                                    │             │
                │  latent (B,64,H,W) → 切 patch      │             │
                │  patch_size = 2 → (B, 64, H/2, W/2)│             │
                │  每个 patch 是 2×2×64 = 256 维      │             │
                │  Linear(256 → hidden_dim=3584)       │             │
                │  → image_tokens (B, N_img, 3584)     │             │
                │  例如 H=128,W=128 → N_img = 64×64    │             │
                │                    = 4096 tokens     │             │
                └────────────────┬───────────────────┘             │
                                 │                                  │
                                 │  image_tokens                    │
                                 │  (B, 4096, 3584)                 │
                                 │                                  │
                                 ▼                                  │
                ┌────────────────────────────────────┐             │
                │   MSRoPE 位置编码 (Qwen-Image 独创)  │             │
                │                                    │             │
                │   图像 token: 2D RoPE (二维坐标)     │             │
                │   文本 token: 沿对角线方向 1D RoPE   │             │
                │                                    │             │
                │   image_tokens += pos_emb_2d(i,j)   │             │
                │   text_tokens  += pos_emb_1d(k)     │             │
                └────────────────┬───────────────────┘             │
                                 │                                  │
                                 ▼                                  │
      ┌──────────────────────────────────────────────────────────┐  │
      │                                                          │  │
      │              MMDiT Backbone (60层, 20B)                   │  │
      │                                                          │  │
      │   输入序列 = [image_tokens || text_tokens]                │  │
      │             (B, 4096+512, 3584)                          │  │
      │                                                          │  │
      │   ┌─────────────────────────────────────────────────┐    │  │
      │   │  MMDiT Block × 60 (每个 Block 内部见 1.2 节)    │    │  │
      │   │                                                 │    │  │
      │   │  关键：文本和图像 token 在同一个序列里做         │    │  │
      │   │  full self-attention（双向，无 causal mask）     │    │  │
      │   │                                                 │    │  │
      │   │  Attention 矩阵: (4608 × 4608)                  │    │  │
      │   │  ┌──────────────┬──────────────┐                │    │  │
      │   │  │ img→img self │ img→text crs │ ← 图像 token   │    │  │
      │   │  │   (4096²)    │  (4096×512)  │    作为 query   │    │  │
      │   │  ├──────────────┼──────────────┤                │    │  │
      │   │  │ text→img crs │ txt→txt self │ ← 文本 token   │    │  │
      │   │  │  (512×4096)  │   (512²)     │    作为 query   │    │  │
      │   │  └──────────────┴──────────────┘                │    │  │
      │   └─────────────────────────────────────────────────┘    │  │
      │                                                          │  │
      │   + Timestep Embedding 注入到每个 Block                   │  │
      │     t → sinusoidal → MLP → c (条件向量, dim=3584)         │  │
      │     c 通过 AdaLN 控制每个 Block 的 scale/shift/gate       │  │
      └──────────────────────────┬───────────────────────────────┘  │
                                 │                                  │
                                 ▼                                  │
      ┌──────────────────────────────────────┐                      │
      │  Final Layer (Unpatch)                │                      │
      │                                      │                      │
      │  只取 image_tokens 部分               │                      │
      │  (B, 4096, 3584) → Linear →          │                      │
      │  (B, 4096, patch_size² × C)          │                      │
      │  = (B, 4096, 256) → reshape          │          ┌───────────┘
      │  → (B, 64, 64, 64) velocity pred     │          │  x_T (初始噪声)
      │                                      │          │  (B,64,64,64)
      └──────────────────┬───────────────────┘          │
                         │                              │
                         │  velocity = x_0_pred - ε     │
                         │  (模型预测的速度场)            │
                         │                              │
                         ▼                              │
      ┌──────────────────────────────────────────────────────┐
      │        SDE Scheduler (FlowMatchSDEDiscreteScheduler) │
      │                                                      │
      │  输入: x_t (当前 latent) + velocity_pred              │
      │  输出: x_{t-1} (下一步 latent) + log_prob            │
      │                                                      │
      │  mean = x_t + ...  (公式)                            │
      │  x_{t-1} = mean + noise_level * sqrt(dt) * ε'       │
      │  log_prob = 高斯密度 (用于 RL 训练)                  │
      └──────────────────┬───────────────────────────────────┘
                         │
                         ▼  (迭代 50 步后)
                ┌────────────────────┐
                │  Clean Latent x_0  │
                │  (B, 64, 64, 64)   │
                └────────┬───────────┘
                         │
                         ▼
                ┌────────────────────┐
                │  VAE Decoder (冻结) │
                │  latent → 像素空间  │
                │  (B,3,512,512)     │
                └────────┬───────────┘
                         │
                         ▼
                ┌────────────────────┐
                │  生成的最终图像     │
                │  一只猫+文字       │
                └────────────────────┘

  ---
  1.2 单个 MMDiT Block 的完整内部结构
  
  这是最核心的模块，也是和标准 LLM Transformer Block 差异最大的地方。

  一层 MMDiT Block 的完整前向过程
  ================================

  输入:
    image_tokens: (B, N_img, D)      例如 (1, 4096, 3584)
    text_tokens:  (B, N_text, D)     例如 (1, 512,  3584)
    c:            条件向量 (B, D)     来自 timestep embedding

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────────────┐
  │  Step 1: adaLN_modulation — 从条件向量回归出 6 组参数   │
  │                                                         │
  │  adaLN_modulation = SiLU → Linear(D → 6*D)              │
  │                                                         │
  │  [shift_msa, scale_msa, gate_msa,                       │
  │   shift_mlp, scale_mlp, gate_mlp] =                     │
  │       adaLN_modulation(c).chunk(6, dim=-1)               │
  │                                                         │
  │  每组 shape: (B, D)  → 每个通道独立的 scale/shift/gate  │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Step 2: 文本-图像联合 Self-Attention (双流)             │
  │                                                         │
  │  拼接: joint = [image_tokens || text_tokens]            │
  │         shape: (B, N_img+N_text, D)                     │
  │                                                         │
  │  先做 AdaLN:                                            │
  │    joint_normed = LayerNorm(joint)  ← 无 affine 参数    │
  │    joint_modulated = joint_normed * (1+scale_msa)       │
  │                      + shift_msa                        │
  │                                                         │
  │  再做 Joint Self-Attention:                             │
  │    Q = joint_modulated @ W_Q   (B, N_total, D)          │
  │    K = joint_modulated @ W_K   (B, N_total, D)          │
  │    V = joint_modulated @ W_V   (B, N_total, D)          │
  │                                                         │
  │    # 先加 MSRoPE 到 Q, K:                               │
  │    Q_img += RoPE_2D(Q_img, pos_x, pos_y)  ← 图像用2D坐标│
  │    K_img += RoPE_2D(K_img, pos_x, pos_y)                │
  │    Q_txt += RoPE_1D(Q_txt, pos_k)         ← 文本用1D坐标 │
  │    K_txt += RoPE_1D(K_txt, pos_k)                       │
  │                                                         │
  │    attn_out = softmax(Q @ K^T / sqrt(d)) @ V            │
  │                                                         │
  │    完整 attention 矩阵包含 4 个区域:                     │
  │    ┌──────────────┬──────────────┐                      │
  │    │ img→img self │ img→txt crs  │                      │
  │    │  (4096,4096) │  (4096,512)  │                      │
  │    ├──────────────┼──────────────┤                      │
  │    │ txt→img crs  │ txt→txt self │                      │
  │    │  (512,4096)  │   (512,512)  │                      │
  │    └──────────────┴──────────────┘                      │
  │                                                         │
  │  Gate 控制:                                             │
  │    joint = joint + gate_msa * attn_out                  │
  │                                                         │
  │  拆分回双流:                                            │
  │    image_tokens = joint[:, :N_img, :]                   │
  │    text_tokens  = joint[:, N_img:, :]                   │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Step 3: Feed-Forward Network (双流独立 MLP)            │
  │                                                         │
  │  Image 侧:                                              │
  │    img_normed = LayerNorm(image_tokens)   ← 无 affine   │
  │    img_modulated = img_normed * (1+scale_mlp)           │
  │                    + shift_mlp                          │
  │    img_ffn = SiLU(img_modulated @ W_up) @ W_down        │
  │    image_tokens = image_tokens + gate_mlp * img_ffn     │
  │                                                         │
  │  Text 侧:                                               │
  │    txt_normed = LayerNorm(text_tokens)                  │
  │    txt_modulated = txt_normed * (1+scale_mlp)           │
  │                    + shift_mlp                          │
  │    txt_ffn = SiLU(txt_modulated @ W_up) @ W_down        │
  │    text_tokens = text_tokens + gate_mlp * txt_ffn       │
  │                                                         │
  │  (image 和 text 共享同一套 shift/scale/gate 参数)        │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  输出: image_tokens (B, N_img, D) + text_tokens (B, N_text, D)
        ↓ 进入下一层 Block (共 60 层)

  与 LLM Transformer Block 的核心差异总结：

  ┌────────────────┬────────────────────────────────────┬──────────────────────────────────────────────────┐
  │      组件      │         LLM Decoder Block          │                   MMDiT Block                    │
  ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ 归一化         │ Pre-LN 或 RMSNorm（固定参数）      │ AdaLN（scale, shift 由 timestep 动态回归）       │
  ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ 子层输出控制   │ 直接残差 x = x + attn(x)           │ Gated 残差 x = x + gate * attn(x)                │
  ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ Attention 类型 │ Causal Self-Attention（单向 mask） │ Full Bi-directional Attention（无 mask）         │
  ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ 位置编码       │ 1D RoPE                            │ MSRoPE（图像 2D RoPE + 文本 1D RoPE 对角线对齐） │
  ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ 输入模态       │ 纯文本 token                       │ 图像 latent token + 文本 token 拼接              │
  ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ 条件注入       │ 无（自回归）                       │ Timestep embedding 注入每一层                    │
  ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ 初始化         │ 标准初始化                         │ Zero 初始化（所有 gate=0，训练初期为恒等映射）   │
  └────────────────┴────────────────────────────────────┴──────────────────────────────────────────────────┘

  1.3 伪代码：完整模型定义

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

  ---
  二、迭代去噪 —— 复用了什么？
  
  这是你作为 LLM Infra 人最容易困惑的地方，因为和 LLM 推理的复用模式完全不同。

  2.1 核心结论：扩散模型去噪不复用 KV cache

  LLM 自回归推理:
    step 1: [BOS]        → "我"   ── KV cache[t=1]
    step 2: [BOS, 我]    → "爱"   ── 复用 KV cache[t=1], 只算新增 token
    step 3: [BOS,我,爱]  → "吃"   ── 复用 KV cache[t=1,2]
    每一步输入变长，但前面的 token 不变！

  扩散模型迭代去噪:
    step 1: latent x_1 (纯噪声)        → x_0.98  ── 整个 latent 变了
    step 2: latent x_0.98 (少一点噪声)  → x_0.96  ── 整个 latent 又变了
    step 3: latent x_0.96               → x_0.94  ── ...
    每一步输入是上一步的完整输出，尺寸相同，数值全变！

  2.2 每一步之间：复用什么，不复用什么

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

  2.3 详细列出每一步的计算过程

  def denoising_loop(noise_latent, text_embedding, text_mask, model, scheduler):
      """
      完整去噪循环，标明每一步什么变了什么没变
      """
      # ═══ 这些只算一次，所有步复用 ═══
      prompt_embeds = text_embedding        # (B, L, D) — 来自 frozen text encoder
      prompt_mask = text_mask               # (B, L)
      img_shapes = compute_img_shapes(H, W)  # [(H_lat, W_lat), ...]
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

  2.4 对比图：LLM vs Diffusion 迭代复用

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

  2.5 扩散模型实际可做的优化（不同于 LLM 的 KV cache）

  优化手段: Text embedding 一次编码
  说明: text encoder 只跑一次，输出复用全部步
  与 LLM 的对比: 类似 prompt encoding 只做一次
  ────────────────────────────────────────
  优化手段: CUDA Graph capture
  说明: 因为每步计算图完全相同（只是输入数值不同），可以 capture 整个 denoising step 的 CUDA graph
  与 LLM 的对比: LLM 不行因为每步 seq_len 变长
  ────────────────────────────────────────
  优化手段: VAE tiling
  说明: VAE decode 用 tiling 减少峰值显存
  与 LLM 的对比: 无类比
  ────────────────────────────────────────
  优化手段: Batch 并行去噪
  说明: 多个 prompt 的 latent 一起处理，矩阵乘法的 batch 维度
  与 LLM 的对比: LLM continuous batching
  ────────────────────────────────────────
  优化手段: LoRA
  说明: 只训练低秩适配器，全量模型权重冻结
  与 LLM 的对比: 跟 LLM LoRA 完全一样
  ────────────────────────────────────────
  优化手段: CFG 时的 batch 合并
  说明: pos 和 neg 的 latent 拼成 2B 一起算
  与 LLM 的对比: 无类比

  关键 insight：扩散模型的性能瓶颈不是 KV cache 和 decode 延迟，而是每一步都需要完整的 model 
  forward。优化方向是减少步数（fewer denoising steps）、提高单步吞吐（larger batch、CUDA
  Graph）、异步流水线（overlap rollout 和 reward compute）。

  ---
  推荐阅读
  
  - Flow Matching 论文: Flow Matching for Generative Modeling (Lipman et al., 2023)
  - FlowGRPO 论文: Training Diffusion Models with RL (VeRL-Omni 核心算法)
  - Qwen-Image 技术报告: arXiv:2508.02324
  - DiT 论文: Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
  - Score-based SDE: Score-Based Generative Modeling through SDEs (Song et al., 2021)
  - Lilian Weng 博客: What are Diffusion Models?
  - Yang Song 博客: Flow Matching Explained

  Sources:
  - Qwen‑Image 如何实现 95%+ 多语种文本渲染准确率
  - 详解Qwen-Image的MMDiT架构优势与工程优化
  - DiT详解
  - MMDiT Block for Multimodal Diffusion
