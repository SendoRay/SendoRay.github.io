---
title: LLM训练全解
date: '2026-06-09'
tags:
- LLM
- GPU
- Training

draft: false
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

## 一、训练到底是什么？

深度学习的"训练"本质是一个**反复试错并改进**的过程。通俗讲：

1. 模型拿到一批数据，做出预测
2. 对比预测和真实答案，计算"错得有多离谱"（损失函数）
3. 根据"哪里错了、错多少"来调整模型参数，让下次预测更准

这三步循环往复数十万次，模型就从"瞎猜"变成"精通"。

用数学语言来说：
- 模型是一个参数化函数 \(f_\theta(x)\)，其中 \(\theta\) 是所有可调参数（权重）
- 给定训练样本 \((x, y)\)，计算损失 \(L = \text{loss}(f_\theta(x), y)\)
- 目标：找到一组 \(\theta^*\) 使得在所有训练数据上的平均损失最小

---

## 二、训练的核心流程：前向传播 → 损失计算 → 反向传播 → 参数更新

### 2.1 前向传播（Forward Pass）

数据从输入层逐层流过网络，每层做矩阵乘法 + 激活函数，最终输出预测结果。

```
输入 x → [Linear层1] → [ReLU] → [Linear层2] → [Softmax] → 预测 ŷ
```

每一层的计算：\(h = \sigma(Wx + b)\)
- \(W\)：权重矩阵（可训练参数）
- \(b\)：偏置（可训练参数）
- \(\sigma\)：激活函数（ReLU、GELU 等）

### 2.2 损失函数（Loss Function）

衡量预测和真实标签之间的差距：

| 任务类型 | 常用损失函数 |
|---|---|
| 分类 | CrossEntropyLoss |
| 回归 | MSELoss |
| 语言模型 | CrossEntropyLoss（预测下一个 token） |

**CrossEntropyLoss（交叉熵）**——分类/语言模型的标配：

\[L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)\]

- \(C\)：类别数（语言模型中 = 词表大小）
- \(y_i\)：真实标签的 one-hot 编码（正确类别为 1，其余为 0）
- \(\hat{y}_i\)：模型经过 softmax 后对第 \(i\) 类的预测概率

直觉：正确类别的预测概率越接近 1，\(\log(\hat{y}_i)\) 越接近 0，loss 越小；预测越离谱，loss 越大。

**MSELoss（均方误差）**——回归任务的标配：

\[L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2\]

- \(y_i\)：真实值
- \(\hat{y}_i\)：预测值
- \(n\)：样本数

直觉：逐个算"预测值和真实值差了多少"，平方后取平均。差得越远，惩罚呈平方级增长。

损失值是一个标量（单个数字），它汇总了"模型在这批数据上表现有多差"。

### 2.3 反向传播（Backward Pass / Backpropagation）

这是训练中最核心的数学机制。目标：**计算损失对每个参数的梯度**（偏导数）。

**梯度是什么？** 梯度告诉你"如果把某个参数稍微增大一点，损失会如何变化"：
- 梯度为正 → 增大该参数会让损失变大（方向错了）
- 梯度为负 → 增大该参数会让损失变小（方向对了）
- 梯度绝对值越大 → 该参数对损失影响越大

通过**链式法则（Chain Rule）**，从输出层反向逐层传递梯度。用一个最简单的 2 层网络来完整展示：

**模型定义（前向传播）：**

\[h = w_1 \cdot x + b_1 \quad \text{（第1层输出，即激活值）}\]
\[y = w_2 \cdot h + b_2 \quad \text{（第2层输出，即模型预测值）}\]

展开写：\(y = w_2(w_1 x + b_1) + b_2\)

**损失函数（以 MSE 为例，假设真实标签为 \(t\)）：**

\[L = (y - t)^2 = (w_2(w_1 x + b_1) + b_2 - t)^2\]

**反向传播——逐步求梯度：**

从最外层开始，一步步向里剥：

\[\frac{\partial L}{\partial y} = 2(y - t) \quad \text{（loss 对预测值的梯度，反向传播的起点）}\]

\[\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w_2} = 2(y - t) \cdot h \quad \text{（第2层权重的梯度）}\]

\[\frac{\partial L}{\partial h} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h} = 2(y - t) \cdot w_2 \quad \text{（激活值 h 的梯度，传给第1层）}\]

\[\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial w_1} = 2(y - t) \cdot w_2 \cdot x \quad \text{（第1层权重的梯度）}\]

**总结——每个量的角色：**

| 符号 | 是什么 | 数值来自 |
|---|---|---|
| \(h = w_1 x + b_1\) | 第1层激活值 | 前向传播时算出 |
| \(y = w_2 h + b_2\) | 模型预测值 | 前向传播时算出 |
| \(\frac{\partial L}{\partial y} = 2(y-t)\) | 输出层梯度（起点） | 反向传播第一步 |
| \(\frac{\partial L}{\partial w_2} = 2(y-t) \cdot h\) | 第2层权重梯度 | 上游梯度 × 本层激活值 |
| \(\frac{\partial L}{\partial h} = 2(y-t) \cdot w_2\) | 传给第1层的梯度 | 上游梯度 × 本层权重 |
| \(\frac{\partial L}{\partial w_1} = 2(y-t) \cdot w_2 \cdot x\) | 第1层权重梯度 | 上游梯度 × 输入 |

注意规律：**求某层权重的梯度 = 上游传来的梯度 × 本层的输入（激活值）**。这就是为什么反向传播需要前向传播时保存的激活值。

这就是为什么叫"反向"传播——梯度从损失出发，沿计算图逆向流动到每个参数。

### 2.4 参数更新（Optimizer Step）

拿到梯度后，优化器负责决定"怎么调参数"。最简单的是**梯度下降（SGD）**：

\[\theta_{new} = \theta_{old} - \eta \cdot \nabla_\theta L\]

- \(\eta\)：学习率（步长），控制每次更新的幅度
- \(\nabla_\theta L\)：损失函数对参数 \(\theta\) 的梯度，指示当前参数应该往哪个方向调整才能让 loss 下降最快

**为什么不直接用 SGD？** 因为它有很多问题：
- 所有参数用同一个学习率，不够灵活
- 容易在 loss landscape 的"窄谷"中震荡
- 没有"动量"概念，容易被噪声干扰

---

## 三、优化器（Optimizer）详解

### 3.1 SGD with Momentum

给梯度加一个"惯性"，避免频繁改变方向：

\[v_t = \beta \cdot v_{t-1} + \nabla_\theta L\]
\[\theta = \theta - \eta \cdot v_t\]

\(v_t\) 相当于历史梯度的指数加权平均，让更新方向更稳定。

### 3.2 Adam（Adaptive Moment Estimation）

**当前大模型训练的标配优化器**。它同时维护两个统计量：

- **一阶矩 \(m_t\)（exp_avg）**：梯度的指数移动平均（动量方向）
- **二阶矩 \(v_t\)（exp_avg_sq）**：梯度平方的指数移动平均（用于自适应学习率）

更新规则：
\[m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t\]
\[v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2\]
\[\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}\]
\[\theta = \theta - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\]

**关键点：**
- 每个参数都有自己的自适应学习率（\(\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\)）
- 对稀疏梯度友好
- 默认参数 \(\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}\)

**Adam 的显存代价：** 对每个可训练参数，需要额外存 2 个 fp32 状态（\(m_t\) 和 \(v_t\)），这就是优化器状态占显存的根本原因。

---

## 四、混合精度训练（Mixed Precision Training）

### 4.1 为什么需要混合精度？

| 数据类型 | 位数 | 每参数字节 | 特点 |
|---|---|---|---|
| fp32 | 32 | 4 bytes | 精度高，计算慢，占内存多 |
| fp16 | 16 | 2 bytes | 精度有限，容易溢出 |
| bf16 | 16 | 2 bytes | 范围同 fp32，精度较低，训练友好 |

用 fp32 训一个 7B 参数的模型，仅参数就需要 28GB 显存。改用 bf16 可以减半到 14GB，同时 GPU 的 Tensor Core 对 16 位计算有 2x 吞吐加速。

### 4.2 bf16 混合精度训练的工作方式

```
┌─────────────────────────────────────────────────────────────────────────┐
│  训练循环（每个 step 重复）                                              │
│                                                                         │
│  1. 前向传播                                                            │
│     输入：bf16 模型参数 W + 本 batch 训练数据 X                         │
│     计算：h = σ(W·X + b)，逐层向前直到输出 → 算出 loss                  │
│     输出：① 每层的 bf16 激活值（反向传播要用）                       │
│           ② 标量 loss（一个数，batch 内所有样本的误差已求平均）       │
│                                                                         │
│  2. 反向传播                                                            │
│     输入：步骤 1 的 loss + 每层的 bf16 激活值                            │
│     计算：从输出层开始，每层做两件事：                                │
│           a) 用上游传来的梯度 × 本层激活值 → 得本层参数的梯度 ∂L/∂W │
│           b) 用上游梯度 × 本层权重 → 得传给下一层的梯度 ∂L/∂h      │
│           （这就是链式法则的实操：梯度逐层向回“接力”）                │
│     输出：每个参数的梯度 g（bf16 或 fp32）                               │
│                                                                         │
│  3. 优化器更新                                                          │
│     输入：步骤 2 的梯度 g + 优化器状态 (m_t, v_t) + fp32 主权重          │
│     计算：Adam 公式 → θ_new = θ - η·m̂/(√v̂ + ε)                         │
│     输出：更新后的 fp32 主权重                                          │
│                                                                         │
│  4. 精度回写                                                            │
│     输入：步骤 3 输出的 fp32 主权重                                      │
│     计算：cast fp32 → bf16                                              │
│     输出：新的 bf16 模型参数 → 供下一个 step 的步骤 1 使用               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**为什么需要 fp32 主权重（Master Weights）？**

因为优化器的更新步长往往很小（如 \(\eta = 10^{-4}\)），直接在 bf16 上做加减法，小的更新量会被舍入为 0（bf16 只有 ~3 位有效数字）。所以必须维护一份 fp32 精度的参数副本来累积精确更新。

---

## 五、数据并行（Data Parallelism）与 DDP

当一张卡放不下一个 batch，或者想加速训练时，就需要**数据并行**：

- 每张 GPU 持有模型的完整副本
- 不同 GPU 处理不同的 mini-batch 数据
- 反向传播后，**AllReduce** 所有 GPU 的梯度取平均
- 每张 GPU 用相同的平均梯度更新参数（保持同步）

### DDP（DistributedDataParallel）关键机制

PyTorch DDP 的核心优化：

1. **Flat Buffer（扁平化缓冲区）**：把所有参数拼成一个大的一维 tensor，方便做通信（NCCL AllReduce 对大 tensor 更高效）
2. **Gradient Bucketing**：梯度分桶，反向传播和通信重叠（overlap computation & communication）
3. **Broadcast 初始参数**：训练开始时从 rank 0 广播参数保证一致

---

## 六、DistributedOptimizer（分布式优化器）

普通 DDP 的问题：每张卡都存完整的优化器状态（fp32 主权重 + Adam 的 m 和 v）。对 7B 模型，这意味着每卡额外 \(7B \times (4 + 4 + 4) = 84\) GB！

**DistributedOptimizer（类似 ZeRO Stage-2）的思路：**

把优化器状态按参数切分（shard）到各个 rank：
- 每个 rank 只负责一部分参数的优化器状态
- AllReduce 梯度后，每个 rank 只更新自己负责的那部分
- 更新后 AllGather 收集完整的更新后参数

这样每卡只需存 \(\frac{1}{N}\) 的优化器状态（N = GPU 数量）。

---

## 七、一个 Rank 的 GPU 显存里到底有什么？

以 **DistributedOptimizer + bf16 混合精度训练** 为例，一个 rank 上的 GPU 数据分为 5 类：

```
┌────────────────────────────────────────────────────────────────────┐
│                          GPU 显存                                  │
│                                                                    │
│  ① DDP flat buffer (param_data)     bf16   模型可训练参数           │
│  ② no_grad params                   bf16   冻结参数(embedding等)    │
│  ③ grad buffer (grad_data)          bf16/fp32  梯度                │
│  ④ fp32 master weights (shard)      fp32   优化器主权重副本         │
│  ⑤ Adam states (exp_avg/exp_avg_sq) fp32   优化器动量              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### ① DDP Flat Buffer（模型可训练参数）

**是什么：** 所有需要训练的参数（各层的 W 和 b）被拼接成一个连续的 bf16 flat tensor。

**为什么这样做：**
- NCCL AllReduce 在连续内存上效率最高
- 减少内存碎片
- DDP 内部直接对这个 buffer 做梯度同步

**大小估算（7B 模型）：** \(7 \times 10^9 \times 2\) bytes = **14 GB**

### ② No-Grad Parameters（冻结参数）

**是什么：** 标记为 `requires_grad=False` 的参数，如：
- 冻结的 embedding 层
- 冻结的底层 encoder（微调场景）
- 位置编码（如果是固定的）

**特点：**
- 不参与反向传播，不产生梯度
- 不在 DDP flat buffer 中
- 前向传播仍然需要它们

**大小：** 取决于冻结了多少层，可能从 0 到几个 GB 不等。

### ③ Gradient Buffer（梯度缓冲区）

**是什么：** 反向传播计算出的梯度值。通常也是一个与 flat buffer 等大的连续 tensor。

**生命周期：**
```
反向传播开始 → 梯度逐层累积到 buffer → AllReduce 同步 → 优化器消费 → 清零 → 下一轮
```

**精度选择：**
- bf16 梯度：节省显存，通信量小，但可能有精度损失
- fp32 梯度：精度更好，但占用翻倍

**大小（bf16）：** 和 flat buffer 一样，约 **14 GB**（7B 模型）

**注意：** 这里存的是"当前 step 的梯度"，不是历史梯度——那是优化器状态的事。

### ④ FP32 Master Weights（优化器主权重副本，Sharded）

**是什么：** 模型参数的 fp32 精度副本，用于优化器精确更新。

**为什么必须存在：**
- bf16 只有 ~3 位有效数字，学习率 1e-4 级别的更新会被舍入为 0
- 精确的参数累积必须在 fp32 下进行
- 每次 optimizer step 后，fp32 值 cast 回 bf16 写入 flat buffer

**Sharded 的含义：**
- DistributedOptimizer 下，每个 rank 只存自己负责的那一段 fp32 参数
- 比如 8 卡训 7B，每卡只存 \(\frac{7B}{8} \approx 875M\) 个参数的 fp32 副本

**大小（每卡，8 卡 7B 模型）：** \(875M \times 4\) bytes ≈ **3.5 GB**

### ⑤ Adam Optimizer States（优化器动量状态，Sharded）

**是什么：** Adam 优化器为每个参数维护的两个 fp32 统计量：
- `exp_avg`（一阶矩 \(m_t\)）：梯度的指数移动平均
- `exp_avg_sq`（二阶矩 \(v_t\)）：梯度平方的指数移动平均

**作用：**
- \(m_t\)：提供动量，让更新方向更稳定
- \(v_t\)：提供自适应学习率，让不同参数有不同的步长

**同样是 Sharded：** 和 master weights 一样按 rank 切分。

**大小（每卡，8 卡 7B 模型）：** \(875M \times 4 \times 2\) bytes ≈ **7 GB**（两个状态）

---

## 八、完整显存占用一览表

以 **7B 模型、8 卡 DistributedOptimizer + bf16** 为例，每卡显存占用：

| 类别 | 精度 | 大小估算 | 说明 |
|---|---|---|---|
| ① 可训练参数 | bf16 | 14 GB | 完整模型参数 |
| ② 冻结参数 | bf16 | ~0 GB | 全参训练时为 0 |
| ③ 梯度 | bf16 | 14 GB | 与参数等大 |
| ④ FP32 主权重 | fp32 | 3.5 GB | Sharded，每卡 1/8 |
| ⑤ Adam 状态 | fp32 | 7 GB | Sharded，每卡 1/8 |
| **激活值**（见下文） | bf16/fp32 | 10~30+ GB | 取决于 batch size 和序列长度 |
| **总计** | - | **~50+ GB** | A100 80GB 可以放下 |

---

## 九、别忘了：激活值（Activations）

上面 5 类是"静态"驻留显存的数据，但训练时还有一大块**动态**数据——**激活值**：

**什么是激活值？** 前向传播中每一层的中间输出结果。反向传播需要它们来计算梯度（链式法则需要用到前向时的中间值）。

**激活值有多大？**
- 与 batch size × sequence length × hidden dim × 层数 成正比
- 对 LLM（如 7B，seq_len=4096，micro_batch=1），激活值可达 10~30 GB

**节省激活值显存的技术：**
- **Activation Checkpointing（梯度检查点）**：只保存部分层的激活值，其余反向时重算
- **Selective Checkpointing**：只 checkpoint 计算量小但占显存大的操作

---

## 十、训练中的完整数据流

把上述所有概念串起来，一个完整的训练 step：

```
Step 1: 前向传播
  输入 token → Embedding(②冻结参数) → bf16 计算
  → 逐层 Transformer Block(①可训练参数)
  → 产生激活值（存显存/或 checkpoint）
  → 输出 logits → 计算 loss（标量）

Step 2: 反向传播
  loss → 逆向逐层计算梯度
  → 梯度写入 ③grad buffer
  → 与 DDP bucket 重叠做 AllReduce（各卡梯度平均）

Step 3: 优化器更新
  每个 rank 取自己负责的 shard：
  → 用 ③平均梯度 更新 ⑤Adam states (m_t, v_t)
  → 用 Adam 公式计算参数更新量
  → 更新 ④fp32 master weights
  → 将 fp32 cast 回 bf16

Step 4: 参数同步
  AllGather：各 rank 将更新后的 bf16 参数收集到 ①flat buffer
  → 所有卡恢复完整的 bf16 模型参数
  → 清零 ③梯度 buffer

Step 5: 进入下一个 step
```

---

## 十一、关键概念速查表

| 概念 | 解释 |
|---|---|
| 参数（Parameters/Weights） | 模型中可学习的数值，即各层的 W 和 b |
| 梯度（Gradient） | 损失对参数的偏导数，指示参数应该如何调整 |
| 学习率（Learning Rate） | 控制每次参数更新的步长大小 |
| 优化器（Optimizer） | 决定如何利用梯度来更新参数的算法（SGD/Adam等） |
| 损失函数（Loss Function） | 量化模型预测与真实标签之间差距的函数 |
| 前向传播（Forward Pass） | 数据流过模型得到预测结果 |
| 反向传播（Backward Pass） | 通过链式法则计算每个参数的梯度 |
| 混合精度（Mixed Precision） | 计算用低精度(bf16)，更新用高精度(fp32) |
| 激活值（Activations） | 前向传播中每层的中间结果，反向传播需要用到 |
| Master Weights | 优化器维护的 fp32 参数副本，防止精度丢失 |
| Adam States | Adam 优化器的两个动量统计量(m和v) |
| AllReduce | 分布式操作：所有 GPU 的梯度求和/平均 |
| AllGather | 分布式操作：收集所有 GPU 的分片组成完整数据 |
| Shard | 将数据切分到多个 GPU 上，每个只存一部分 |
| DDP Flat Buffer | 将所有参数拼成一维连续 tensor，加速通信 |
| Activation Checkpointing | 用计算换显存：部分激活值不存，反向时重算 |

---

## 十二、总结

理解 GPU 上到底有哪些数据，本质上就是理解训练的核心机制：

1. **模型参数** → 用于前向传播做预测
2. **梯度** → 反向传播告诉我们参数该怎么调
3. **优化器状态** → 利用历史信息让更新更智能
4. **Master Weights** → 保证精度不丢失
5. **激活值** → 反向传播的计算原料

每一类数据都是训练算法不可或缺的一环。DistributedOptimizer 通过 shard 优化器状态来节省显存，混合精度通过低精度计算来提升速度——但这些优化都不会减少数据的"种类"，只是改变了它们的精度和存储位置。

掌握了这张“GPU 显存地图”，就能理解后续所有训练优化技术（ZeRO、Pipeline Parallelism、Tensor Parallelism、Offload）的设计动机和原理。

---

## 附录：Offload 时哪些数据可以丢、哪些必须保留？

在 RL Infra（如 RLHF 训练框架）中，训练和 rollout 阶段交替进行，GPU 显存需要在两个角色间切换。Offload 的核心思路是：训练结束后把暂时不用的数据搬到 CPU，rollout 结束后再搬回来。

### 每种数据的 Offload 策略

| 数据 | 大小（每参数） | 能不能丢？ | 原因 |
|---|---|---|---|
| FP32 主权重 | 4 bytes | 不能丢，必须备份 | 参数的“真值”，optimizer.step() 在上面累加更新，丢了 = 丢训练进度 |
| 优化器状态 (m, v) | 8 bytes | 不能丢，必须备份 | 跨几千步累积的滑动平均，丢了等于 Adam 失忆，训练不稳定 |
| BF16 模型参数 | 2 bytes | 理论上可以丢 | 可从 FP32 主权重 cast 回来，但省的内存有限且会让 reload 路径变复杂 |
| 梯度 | 2~4 bytes | **可以直接丢掉** | 下次 backward() 会完全覆盖，备份旧梯度毫无意义 |

### 为什么梯度可以直接丢而不是 offload？

关键区别：

- **备份再恢复（普通 offload）：** offload 时 GPU→CPU 拷贝，reload 时 CPU→GPU 拷贝——两次内存拷贝 + 占用 CPU 内存
- **直接丢掉，reload 时分配零张量：** offload 时直接释放，reload 时 `torch.zeros(...)` 分配空 buffer

两种方式的最终效果一样——下一次 `backward()` 都会用新算出的梯度覆盖 buffer 里的值。前者多花了 CPU 内存去存一堆注定要被覆盖的零/旧值，完全是浪费。

> 所以不是“省掉了重算”，而是省掉了**无意义的 CPU 备份和恢复拷贝**。重算（backward）是怎么都要做的，跟备不备份无关。

### BF16 参数为什么不值得丢？

理论上可以从 FP32 主权重重建，但性价比不高：

- rollout 阶段只需要 BF16 参数做推理，如果没备份，就得先加载 FP32 再 cast，多一步依赖
- 省的只是 ~2 bytes/参数的存储，相比优化器状态的 8 bytes 不算大头
- 会让 reload 路径变复杂（RDMA 读 FP32 + cast 的延迟可能比直接读 BF16 更高）

### 小结

| 操作 | 策略 | 理由 |
|---|---|---|
| FP32 主权重 | offload 到 CPU | 不可替代，是训练进度本身 |
| 优化器状态 | offload 到 CPU | 不可替代，跨步累积信息 |
| BF16 参数 | offload 到 CPU | 可重建但省的少、复杂度高，不值得丢 |
| 梯度 | 直接释放，不备份 | 下次 backward 必然覆盖，备份是浪费 |
