---
title: 训推工程师必备：超参设置与实验结果判断完全指南
date: '2026-06-26'
tags:
- LLM
- Engineering
draft: false
math: true
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

跑实验不是"随便调调参数看看效果"，而是一门有章法的工程实践。很多人把超参调优当玄学，本质上是因为没搞清楚每个超参的**物理含义**和它们之间的**耦合关系**。这篇文章从训推工程师的视角，系统梳理 LR、BS、WD 等核心超参的设置方法论，以及如何判断实验结果是否合理。

> TL;DR：**Learning Rate 决定步长、Batch Size 决定梯度噪声、两者必须联动调整**。合理的实验结果应该是：loss 平滑下降、eval 指标与 loss 方向一致、不出现 NaN/spike/plateau 三大异常。

---

## 一、超参的本质：控制优化轨迹

训练的核心循环：

```
for step in range(total_steps):
    batch = sample(dataset, batch_size)        # ← BS 控制
    loss = model(batch)
    loss.backward()                            # 计算梯度
    optimizer.step(lr=learning_rate)           # ← LR 控制
    optimizer.zero_grad()
```

每一步更新可以写成：

\[\theta_{t+1} = \theta_t - \eta \cdot \hat{g}_t\]

其中：
- \(\eta\) = Learning Rate（LR），控制每步走多远
- \(\hat{g}_t\) = 当前 mini-batch 计算出的梯度估计，由 Batch Size（BS）决定其方差

**核心直觉**：LR 决定步长大小，BS 决定方向的"噪声程度"。两者共同决定了优化轨迹在 loss landscape 上的行走方式。

---

## 二、Learning Rate：最敏感的超参

### 2.1 LR 过大 vs 过小的表现

| 现象 | 原因 | 诊断方法 |
|------|------|----------|
| Loss 震荡/发散/出现 NaN | LR 过大，更新步长超过了 loss 曲面的曲率半径 | 观察 loss 曲线前 100 步 |
| Loss 下降极慢，几乎不动 | LR 过小，步长微弱到每步几乎不改变参数 | 对比不同 LR 的 loss 下降斜率 |
| Loss 先降后突然 spike | LR 在当前参数区域还行，但随训练推进 loss landscape 变化后 LR 偏大 | 是否缺少 LR decay |
| Loss 平滑下降后 plateau | 可能 LR 已经小到跳不出当前局部最优 | 尝试 warmup restart |

### 2.2 如何选择初始 LR

**经验值速查表**（AdamW optimizer）：

| 模型规模 | 推荐初始 LR | 说明 |
|----------|-------------|------|
| ResNet-50 (ImageNet) | 1e-3 ~ 3e-3 | Adam 系优化器的经典起点 |
| BERT-base fine-tune | 2e-5 ~ 5e-5 | 预训练模型微调 LR 要小 1~2 个数量级 |
| GPT-2 (124M) pretrain | 6e-4 | OpenAI 原始设置 |
| LLaMA-7B pretrain | 3e-4 | Meta 论文设置 |
| LLaMA-65B pretrain | 1.5e-4 | 模型越大，LR 越小 |
| SFT fine-tune (7B) | 1e-5 ~ 2e-5 | 微调时 LR 比预训练小一个数量级 |
| LoRA fine-tune | 1e-4 ~ 3e-4 | 只更新低秩矩阵，可以用稍大的 LR |

**关键规律**：
1. **模型越大，LR 越小**——大模型参数空间更复杂，loss landscape 曲率更大
2. **预训练用大 LR，微调用小 LR**——微调时模型已在好的区域，大步长会把它踢出去
3. **Adam 系优化器 LR 比 SGD 小 1~2 个数量级**——Adam 自带自适应缩放

### 2.3 LR Warmup：为什么要热身

训练刚开始时，模型参数是随机初始化的（或加载预训练），此时梯度方向不稳定、方差大。如果一上来就用大 LR：

```
Step 0: 参数随机 → 梯度方向乱 → 大步长 → 参数飞到奇怪的地方 → 后续难以恢复
```

**Warmup 的做法**：前 N 步（或前 x% 的 steps）线性地把 LR 从 0 涨到目标值：

\[\eta_t = \eta_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}}, \quad t \leq T_{\text{warmup}}\]

**推荐设置**：
- 预训练：warmup steps = total steps 的 1%~5%（常见 2000 步）
- 微调：warmup steps = 总步数的 5%~10%（或固定 100~500 步）

### 2.4 LR Schedule：训练后期怎么衰减

Warmup 之后，LR 需要逐渐下降，常见方案：

**Cosine Decay（余弦衰减）**——目前 LLM 训练的标配：

\[\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - T_w}{T - T_w}\pi\right)\right)\]

```
LR
│
│  /‾‾‾‾‾‾‾‾‾‾‾\
│ /               \
│/                 \_____________
│← warmup →│← cosine decay →│
└──────────────────────────────── Step
```

**Linear Decay（线性衰减）**——简单直观：

\[\eta_t = \eta_{\max} \cdot \left(1 - \frac{t - T_w}{T - T_w}\right)\]

**常见设置**：
- \(\eta_{\min}\) = \(\eta_{\max}\) 的 10%（即最终衰减到峰值的 1/10）
- LLaMA 系列用 cosine decay 到 peak LR 的 10%

### 2.5 LR Finder：数据驱动选 LR

如果完全没有经验值参考，可以用 **LR Range Test**（Smith, 2017）：

1. 从一个极小的 LR（如 1e-7）开始，每步指数增长到一个很大的值（如 1）
2. 记录每步的 loss
3. 找到 loss 下降最快的区间，取该区间中间偏左的值作为最大 LR

```python
# PyTorch 伪代码
lr_min, lr_max = 1e-7, 1.0
num_steps = 200
gamma = (lr_max / lr_min) ** (1 / num_steps)

for step in range(num_steps):
    current_lr = lr_min * (gamma ** step)
    set_lr(optimizer, current_lr)
    loss = train_one_step()
    log(current_lr, loss)
```

```
Loss
│\
│ \
│  \         ← loss 下降最快的区域
│   \___
│       \___/‾‾‾  ← 开始发散
│
└────────────────── log(LR)
        ↑
     选这里
```

---

## 三、Batch Size：梯度估计的精度

### 3.1 BS 的物理含义

每步更新时，我们用一个 mini-batch 的梯度来估计全量梯度：

\[\hat{g}_t = \frac{1}{B}\sum_{i=1}^{B}\nabla L(x_i, \theta_t)\]

- **BS 大** → 梯度估计方差小、方向稳 → 类似"精准射击"
- **BS 小** → 梯度估计方差大、带噪声 → 类似"散弹枪"

### 3.2 BS 与泛化的关系

这是一个被广泛研究的话题：

| BS 大小 | 优点 | 缺点 |
|---------|------|------|
| **大 BS**（如 4096+） | GPU 利用率高、训练吞吐大、loss 曲线平滑 | 倾向收敛到 sharp minima，泛化可能差 |
| **小 BS**（如 8~64） | 梯度噪声帮助跳出局部最优，泛化通常更好 | GPU 利用率低、训练慢、loss 曲线抖动 |

**关键论文结论**：
- Goyal et al. (2017): 大 BS 可以工作，但需要配合 **Linear Scaling Rule**
- Hoffer et al. (2017): 小 BS 的"隐式正则化"效果来自梯度噪声
- McCandlish et al. (2018): 存在一个 **Critical Batch Size**，超过它再增大 BS 收益递减

### 3.3 Linear Scaling Rule：BS 翻倍 LR 也翻倍

这是最重要的 BS-LR 耦合规则：

> 当 Batch Size 增大 k 倍时，Learning Rate 也应增大 k 倍（或 \(\sqrt{k}\) 倍）。

**直觉解释**：
- BS 从 256 → 1024（4 倍），每步"看到"的数据多了 4 倍
- 相当于原来走 4 步的路现在 1 步走完
- 要达到同等效果，每步的步长也要放大

**实操建议**：

```
基准配置：BS=256, LR=1e-3

→ BS 翻倍到 512:  LR = 1e-3 × 2 = 2e-3     (linear scaling)
→ BS 翻4倍到1024: LR = 1e-3 × 4 = 4e-3     (linear scaling)
→ BS 翻4倍到1024: LR = 1e-3 × 2 = 2e-3     (sqrt scaling, 更保守)
```

**注意**：Linear Scaling 在 BS 非常大时会失效（超过 Critical Batch Size），此时用 sqrt scaling 或 LARS/LAMB 优化器。

### 3.4 Gradient Accumulation：小显存模拟大 BS

显存不够装大 BS？用梯度累积：

```python
accumulation_steps = 4  # 等效 BS = real_bs × 4

optimizer.zero_grad()
for micro_step in range(accumulation_steps):
    batch = next(dataloader)           # real_bs 个样本
    loss = model(batch) / accumulation_steps  # ← 注意要除以累积步数！
    loss.backward()                    # 梯度累加在 .grad 里
optimizer.step()                       # 累积完才更新一次
```

**易错点**：
- `loss` 要除以 `accumulation_steps`，否则等效 LR 被放大了
- 如果用了 DDP，`no_sync()` 上下文可以避免每个 micro step 都 allreduce

### 3.5 BS 选择实操指南

| 场景 | 推荐 BS | 理由 |
|------|---------|------|
| LLM 预训练（7B~70B） | 2M~4M tokens（global） | 大模型需要大 BS 保证梯度稳定 |
| SFT 微调 | 32~128 samples | 数据集小，不需要太大 BS |
| LoRA 微调 | 4~32 samples | 参数少，小 BS 足够 |
| CV 分类 (ImageNet) | 256~4096 | 经典设置 |
| Debug/验证 | 2~4 | 先确认流程跑通 |

---

## 四、Weight Decay：防止参数爆炸

### 4.1 什么是 Weight Decay

Weight Decay（权重衰减）本质是 L2 正则化的等价形式：

\[\theta_{t+1} = \theta_t - \eta \cdot \hat{g}_t - \eta \cdot \lambda \cdot \theta_t\]

等价于在 loss 上加了 \(\frac{\lambda}{2}\|\theta\|^2\) 项。

**直觉**：每步更新时，把所有参数向 0 拉一点。防止某些参数增长到非常大的值。

### 4.2 推荐设置

| 场景 | Weight Decay | 说明 |
|------|-------------|------|
| AdamW 预训练 | 0.01 ~ 0.1 | LLaMA 用 0.1 |
| AdamW 微调 | 0.01 ~ 0.05 | |
| SGD + Momentum | 1e-4 ~ 5e-4 | CV 经典设置 |

**重要细节**：
- **LayerNorm/Bias 不加 Weight Decay**——这些参数本身就不大，加了 WD 会导致训练不稳定
- AdamW ≠ Adam + L2 regularization（Loshchilov & Hutter, 2019 澄清了这一点）

```python
# PyTorch 中正确的做法：对不同参数组设置不同的 WD
no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
param_groups = [
    {'params': [p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.1},
    {'params': [p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
]
optimizer = torch.optim.AdamW(param_groups, lr=3e-4)
```

---

## 五、其他关键超参

### 5.1 训练步数 / Epochs

**预训练**：通常按 token 数计，而非 epoch 数。
- Chinchilla 定律：最优 token 数 ≈ 20 × 模型参数量
- 7B 模型 → ~140B tokens → 数据只过 1 遍（1 epoch 甚至不到）

**微调**：通常 2~5 epochs
- SFT: 2~3 epochs（过多 epoch → 过拟合到训练集的表达风格）
- RLHF/DPO: 1~3 epochs

**判断过拟合**：
- Train loss 持续下降但 eval loss 开始上升 → 过拟合的经典信号
- Eval metric 不再提升甚至下降 → 该停了

### 5.2 Adam 的 β₁ 和 β₂

Adam 优化器的两个动量参数：

\[m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{（一阶动量，方向的 EMA）}\]
\[v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{（二阶动量，幅度的 EMA）}\]

| 参数 | 默认值 | 调优空间 | 说明 |
|------|--------|----------|------|
| β₁ | 0.9 | 0.9~0.95 | 越大 → 动量越强，更平滑但反应慢 |
| β₂ | 0.999 | 0.95~0.999 | 越大 → 学习率自适应越保守 |
| ε | 1e-8 | 1e-8 | 基本不改 |

**LLM 常用设置**：
- GPT-3: β₁=0.9, β₂=0.95
- LLaMA: β₁=0.9, β₂=0.95
- 为什么 β₂ 用 0.95 而非默认 0.999？因为 LLM 训练步数极多，β₂=0.999 的 EMA 窗口太长（~1000 步），对分布变化反应太慢。

### 5.3 Gradient Clipping

防止梯度爆炸的保险措施：

\[\hat{g} = \begin{cases} g & \text{if } \|g\| \leq c \\ c \cdot \frac{g}{\|g\|} & \text{if } \|g\| > c \end{cases}\]

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**推荐设置**：
- LLM 预训练/微调：`max_norm = 1.0`（几乎所有主流工作都用这个值）
- 如果观察到 grad norm 经常超过 1.0 很多倍 → LR 可能偏大

### 5.4 Dropout

| 场景 | 推荐 Dropout | 说明 |
|------|-------------|------|
| LLM 预训练 | 0（不用） | 数据量大到不需要 dropout 正则化 |
| 小模型 fine-tune | 0.1~0.3 | 数据少时防过拟合 |
| CV 模型 | 0.2~0.5 | 经典设置 |

**趋势**：现代大模型预训练几乎不用 Dropout，靠大数据量和 Weight Decay 做正则化。

---

## 六、超参组合速查表

### 6.1 LLM 预训练标准配置

```yaml
# 7B 模型预训练参考（Chinchilla-optimal）
model_size: 7B
total_tokens: 2T
global_batch_size: 4M tokens   # = micro_bs × seq_len × dp_size × grad_accum
seq_len: 4096
micro_batch_size: 2~4          # 单卡能放下的最大 bs
grad_accumulation: 视 DP 规模调整

optimizer: AdamW
lr: 3e-4
lr_schedule: cosine decay → 3e-5 (10% of peak)
warmup_steps: 2000
weight_decay: 0.1
beta1: 0.9
beta2: 0.95
grad_clip: 1.0
dropout: 0.0
```

### 6.2 SFT 微调标准配置

```yaml
# 7B 模型 SFT 微调参考
base_model: pretrained_7B
dataset_size: 50K~500K samples
epochs: 2~3

optimizer: AdamW
lr: 2e-5
lr_schedule: cosine decay → 0
warmup_ratio: 0.03~0.1
batch_size: 128 (global)
weight_decay: 0.01
grad_clip: 1.0
```

### 6.3 LoRA 微调标准配置

```yaml
# 7B 模型 LoRA 微调参考
lora_rank: 8~64
lora_alpha: 16~128 (通常 = 2 × rank)
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj]

optimizer: AdamW
lr: 1e-4 ~ 3e-4     # LoRA 可以用比全参微调大的 LR
epochs: 3~5
batch_size: 16~64
weight_decay: 0.01
```

---

## 七、怎样的实验结果算"合理"

这是新手最容易困惑的部分。跑完实验，怎么判断结果是正常的还是有 bug？

### 7.1 Loss 曲线的健康形态

**正常的 loss 曲线**：

```
Loss
│
│\
│ \
│  \
│   \.
│     '·..
│         ''·····___________
│
└──────────────────────────── Step
  ↑         ↑           ↑
快速下降  减速下降    收敛/plateau
```

**各阶段的预期**：
1. **前 1% 步数**：loss 应该快速下降（从随机初始化的高位）
2. **中间 80%**：稳定下降，斜率逐渐变小
3. **最后 ~20%**：接近收敛，下降极缓（但不应该回升）

### 7.2 Loss 的量级参考

**语言模型（CrossEntropy Loss on next-token prediction）**：

| 阶段 | 预期 Loss 范围 | 换算 PPL |
|------|---------------|----------|
| 随机初始化 | ~ln(V) ≈ 10~12（V=32K~100K） | 几万到几十万 |
| 训练 1% 步数后 | 5~7 | ~150~1000 |
| 训练 10% 步数后 | 3~4 | ~20~55 |
| 收敛 | 1.5~2.5 | ~4.5~12 |

\[\text{PPL} = e^{\text{Loss}}\]

**对比基线**：
- GPT-2 (124M) 在 WebText 上最终 PPL ≈ 29（loss ≈ 3.37）
- LLaMA-7B 最终 loss ≈ 1.8~2.0
- LLaMA-65B 最终 loss ≈ 1.5~1.7

**SFT 微调的 loss**：
- 起始：通常 1.5~3.0（取决于预训练质量和 SFT 数据分布）
- 收敛：0.5~1.5（比预训练的 loss 低，因为 SFT 数据更"简单"/更规律）

### 7.3 五个异常信号及诊断

#### 异常 1：Loss = NaN / Inf

```
Step 100: loss = 2.34
Step 101: loss = 5.67
Step 102: loss = nan   ← 死了
```

**诊断清单**：
1. LR 过大 → 降低 LR 或加 warmup
2. 数据中有异常值（无穷大、空 batch）→ 检查数据管线
3. fp16 溢出 → 检查 GradScaler、尝试 bf16
4. 梯度爆炸 → 加 grad_clip 或减小 grad_clip 值

#### 异常 2：Loss Spike（突然尖刺后恢复）

```
Loss
│      *
│     / \
│    /   \
│···/     \···继续下降
│
└──────────── Step
```

**可能原因**：
- 数据中某个 batch 异常（超长序列、异常 token）
- 分布式训练中某个 worker 出问题
- LR 在某个阶段偏大

**是否需要干预**：
- 偶尔 1~2 个 spike 且 loss 能恢复 → 正常，无需处理
- 频繁 spike → 需要排查数据或降低 LR
- Spike 后 loss 不恢复 → 严重问题，需要 checkpoint 回滚重跑

#### 异常 3：Loss Plateau（过早停滞）

```
Loss
│\
│ \____________________  ← 太早就不降了
│
└──────────────────────── Step
```

**可能原因**：
- LR 过小 → 尝试增大 LR
- 模型容量不够 → 换更大的模型
- 数据质量差/重复度高 → 检查数据
- Bug：梯度没有正确回传（某些层 frozen 了）

#### 异常 4：Train Loss 降 但 Eval Loss 升

```
      Train Loss           Eval Loss
│\                    │      ____/
│ \                   │ \___/
│  \___               │
│      \___           │ ← 过拟合！
└──────── Step        └──────── Step
```

**诊断**：这是经典过拟合。

**解决方案**（按优先级）：
1. 减少 epochs
2. 加大 Weight Decay
3. 减小模型（或用 LoRA 减少可训练参数）
4. 增大数据量
5. 加 Dropout（微调场景）

#### 异常 5：Loss 正常但 Eval Metric 不动

Loss 在下降，但下游任务的准确率/BLEU/Rouge 不提升。

**可能原因**：
- Loss 在优化的目标和 eval metric 不一致（如：loss 在优化 CLM，但 eval 在测 QA）
- Eval 数据分布和 train 分布差异太大
- 评估方式有 bug（如 tokenizer 不一致、generation config 不对）

### 7.4 Eval 指标的合理区间

| 任务 | 指标 | 合理范围 | "不正常"的信号 |
|------|------|----------|----------------|
| 分类 | Accuracy | 随机基线 → ~SOTA | 比随机基线还低 = 有 bug |
| LLM QA | EM/F1 | 看 benchmark | F1=0 → 生成格式问题 |
| 翻译 | BLEU | 20~40 (一般) | BLEU < 5 → 基本没训好 |
| 摘要 | Rouge-L | 30~50 | Rouge < 10 → 对齐问题 |
| SFT (chat) | MT-Bench | 5~8 (7B) | < 4 → SFT 数据/超参问题 |

### 7.5 Sanity Check 清单

每次开跑实验前和实验中，按这个清单检查：

**开跑前**：
- [ ] LR 和 BS 是否匹配（有没有按 Linear Scaling Rule 调整）
- [ ] Warmup 步数是否合理（不是 0）
- [ ] Weight Decay 是否排除了 bias/LN
- [ ] 数据 pipeline：打印前 10 个 batch 的 shape 和内容，人眼确认
- [ ] 用 BS=2, 1 step 跑通（不 OOM、不 NaN）

**训练中**：
- [ ] 第 1 步的 loss ≈ ln(vocab_size)？（语言模型）
- [ ] 前 100 步 loss 有明显下降？
- [ ] Grad norm 在合理范围（通常 < 10）？
- [ ] GPU 利用率 > 50%?（否则 BS 太小或有通信瓶颈）

**训练后**：
- [ ] 最终 loss 是否达到同规模模型的参考值
- [ ] Eval metric 方向是否与 loss 一致
- [ ] 不同 seed 跑 2~3 次，结果方差是否合理

---

## 八、超参搜索策略

### 8.1 不要 Grid Search

Grid Search（网格搜索）在高维超参空间中效率极低。假设你要调 LR、BS、WD 三个参数，每个试 5 个值，就是 125 次实验。

**推荐策略**：

1. **Sequential Halving**：先用短实验（如 10% 的训练步数）淘汰明显差的配置
2. **Random Search**（Bergstra & Bengio, 2012）：在区间内随机采样，效率远高于 Grid
3. **手动 Binary Search**：固定其他参数，对目标超参做二分查找

### 8.2 超参调优的优先级

并非所有超参同等重要。按敏感度排序：

```
最敏感 ─────────────────────────────── 最不敏感
  LR  >  BS  >  Warmup  >  WD  >  β₂  >  Dropout
```

**实操建议**：
1. 先固定 BS（受限于硬件），调 LR（最重要）
2. LR 确定后微调 Warmup 步数
3. 最后微调 WD

### 8.3 短实验 ≈ 长实验

一个经验法则（适用于 LLM 预训练）：

> 在 10%~20% 的训练步数内，不同超参配置的相对排序和最终排序高度一致。

**这意味着**：你不需要跑满整个训练才能判断某个配置好不好。跑 10%~20% 看 loss 趋势，就能淘汰大部分差的配置。

```
Loss
│ Config A (最终最差)
│  \
│   Config B
│    \
│     Config C (最终最好)
│      \
└──────────── 10% steps 时的排序 = 最终排序
```

---

## 九、常见 Pitfall 合集

### Pitfall 1：BS 变了但 LR 没联动

```
"我把 BS 从 32 改成 256，怎么效果变差了？"
→ 因为你 LR 没跟着放大 8 倍（或至少 √8 倍）
```

### Pitfall 2：用了 Gradient Accumulation 但忘了除以 accumulation_steps

```python
# 错误写法
loss = model(batch)
loss.backward()  # 梯度被累加了 N 次，等效 LR 放大了 N 倍

# 正确写法
loss = model(batch) / accumulation_steps
loss.backward()
```

### Pitfall 3：微调时 LR 太大

```
"我用预训练的 LR（3e-4）去做 SFT，loss 震荡"
→ 微调 LR 应该比预训练小 10~100 倍，用 1e-5 ~ 2e-5
```

### Pitfall 4：没有 Warmup

```
"训练第一步就 NaN 了"
→ 加 warmup。尤其是从随机初始化训练时，前几步梯度巨大。
```

### Pitfall 5：评估方式和训练不一致

```
"Training loss 已经很低了，但 eval 上效果很差"
→ 检查：
  1. Eval 用的 tokenizer 和 train 一样吗？
  2. Eval 时的 generation config（max_len, temperature）合理吗？
  3. Eval 数据有没有被模型在 train 时见过？（数据泄露反向情况）
```

### Pitfall 6：多卡训练 LR 没有按 world_size 缩放

使用 DDP 时，global_batch_size = per_gpu_bs × world_size。如果从单卡实验配置迁移到多卡：

```
单卡：BS=32, LR=1e-4
4 卡 DDP：effective BS=128 → LR 应调整为 4e-4（或 2e-4 sqrt 缩放）
```

### Pitfall 7：混合精度训练的 Loss Scale 问题

fp16 训练时 GradScaler 可能导致前几步 loss 看起来不正常：

```
Step 0: loss = 10.5 (scale=65536)
Step 1: loss = inf  (scale overflow, scaler 自动降 scale)
Step 2: loss = 10.3 (正常恢复)
```

**解决**：前几步的 inf/nan 如果 GradScaler 能自动恢复，是正常的。如果持续不恢复，考虑用 bf16。

---

## 十、一张图总结决策流程

```
开始新实验
    │
    ▼
[确定硬件约束] → 能放多大 BS？
    │
    ▼
[选 BS] → 尽量大，但受限于显存
    │
    ├── 显存不够 → 用 Gradient Accumulation
    │
    ▼
[选 LR] → 查参考值表 → 按 BS 用 Linear Scaling 调整
    │
    ▼
[设 Warmup] → 总步数的 1%~5%
    │
    ▼
[设 LR Decay] → Cosine → final LR = peak 的 10%
    │
    ▼
[设 WD] → 0.01~0.1，排除 bias/LN
    │
    ▼
[Sanity Check] → BS=2 跑 1 步，确认不 NaN
    │
    ▼
[短实验] → 10%~20% 步数，对比 2~3 组 LR
    │
    ▼
[选最优配置] → 跑完整训练
    │
    ▼
[监控] → Loss 曲线 + Eval metric + Grad norm
    │
    ├── 异常 → 回到 [选 LR] 或检查数据
    │
    ▼
[收敛] → 对比 baseline，报告结果
```

---

## 附录：快速参考卡片

| 超参 | 预训练 (7B) | SFT (7B) | LoRA (7B) | CV (ResNet-50) |
|------|------------|-----------|-----------|----------------|
| LR | 3e-4 | 2e-5 | 1e-4~3e-4 | 1e-3 |
| BS (global) | 4M tokens | 128 | 32 | 256 |
| Warmup | 2000 steps | 5%~10% | 5%~10% | 5 epochs |
| LR Schedule | Cosine | Cosine | Cosine | StepLR/Cosine |
| WD | 0.1 | 0.01 | 0.01 | 1e-4 |
| Grad Clip | 1.0 | 1.0 | 1.0 | 不常用 |
| β₁, β₂ | 0.9, 0.95 | 0.9, 0.999 | 0.9, 0.999 | 0.9, 0.999 |
| Epochs | ~1 (token-based) | 2~3 | 3~5 | 90~300 |

---

**总结**：超参调优不是玄学，而是有明确物理直觉和经验法则的工程实践。掌握 LR-BS 联动规则、学会从 loss 曲线诊断问题、建立合理的 eval 预期——这三点做到了，你的实验效率会提升一个量级。
