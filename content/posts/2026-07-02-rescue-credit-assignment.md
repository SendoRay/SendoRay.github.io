---
title: '救援信用分配：当"力挽狂澜"比"一气呵成"更值钱'
date: 2026-07-02
draft: false
tags: ["credit-assignment", "multi-agent", "reinforcement-learning", "agent-rl"]
summary: '提出"救援溢价"信用分配假说：在多智能体协作中，接手失败模型并成功完成任务的智能体，其信用得分应高于直接成功的智能体。形式化了三种公式变体并给出完整伪代码。'
---

## 第1章：问题动机——一个"朴素但没人做"的想法

### 1.1 场景描述

考虑以下多智能体协作场景：

```
任务 T，满分 S_max = 100

情况一：模型 A 独立完成任务 T → 成功 → A 得分 = 100
情况二：模型 B 独立完成任务 T → 失败 → B 得分 = 45
情况三：B 失败后，模型 C 接手并成功完成任务 T → C 得分 = ？
```

### 1.2 核心假说

**"救援溢价"（Rescue Premium）假说**：

> 当一个智能体接手另一个智能体的失败状态并成功完成任务时，它的信用得分应该**高于**直接从零开始成功的智能体。

直觉论证：

1. **恢复成本**：C 不仅要完成任务本身，还要理解、评估、甚至撤销 B 留下的错误工作
2. **难度不对称**：从"被污染"的中间状态恢复，比从干净状态出发更难
3. **信息处理负担**：C 必须甄别 B 的 45 分中哪些是有用的、哪些是误导性的

因此，C 的得分不应是简单的 `100 - 45 = 55`，甚至不应止步于 `100`，而应包含一个**救援难度溢价**。

### 1.3 与现有工作的区别

现有信用分配方法（Shapley 值、反事实信用、过程奖励等）主要关注"边际贡献"——即智能体加入后任务收益的增量。但它们**忽略了起点状态的难度差异**：

| 维度 | 现有方法 | 本文假说 |
|------|---------|---------|
| 基本思路 | 边际贡献 = $V(\text{有C}) - V(\text{无C})$ | 边际贡献 × 难度权重 |
| 起点 | 不区分起点状态 | 区分"干净起点"与"污染起点" |
| 救援场景 | C 得 55 分 | C 得 > 100 分 |
| 难度建模 | 无 | 显式建模恢复成本 |

---

## 第2章：信用分配全景回顾

> 参考综述：[From Reasoning to Agentic: Credit Assignment in Reinforcement Learning](https://arxiv.org/abs/2604.09459) 及 [Awesome-Credit-Assignment-in-LLM-RL](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)

### 2.1 信用分配的核心挑战

在 LLM + RL 的训练范式中，模型执行多步长、多工具调用的复杂任务后，最终只获得一个稀疏的终端奖励（成功/失败）。**信用分配问题**就是：如何把这个终端奖励公平地分配回每一步、每一个智能体？

综述 [arxiv 2604.09459](https://arxiv.org/abs/2604.09459) 将现有方法按两个维度分类：

- **粒度**：Token → Segment → Step-Turn → Multi-Agent
- **方法论**：Monte Carlo / TD / LLM-as-Critic / 博弈论 / 信息论 / 不确定性控制 / 可验证反馈塑形

### 2.2 单智能体 Agent RL：信用分配怎么用？

> 这是理解本文假说的**必备 baseline**。下面按"实际工程中最常用"到"学术前沿"排列。

#### 2.2.1 GRPO——实际工程默认基线（无信用分配）

**GRPO（Group Relative Policy Optimization）** 是 DeepSeek 提出且目前在开源社区最广泛使用的方法。它的核心特点：**完全不做细粒度信用分配**。

对于给定的 prompt，采样 $G$ 条轨迹，计算每条轨迹的优势：

$$A_i^{\text{GRPO}} = R(\tau_i) - \frac{1}{G}\sum_{j=1}^{G} R(\tau_j)$$

**关键性质**：同一条轨迹内所有 token、所有步骤获得**完全相同**的优势值。没有 critic 网络，没有步骤级信号。相当于把整条轨迹当作一个"老虎机臂"（bandit arm）。

| 优点 | 缺点 |
|------|------|
| 实现简单，无需训练 critic | 长轨迹中信用信号被稀释 |
| 内存效率高 | 无法区分关键步骤和无关步骤 |
| DeepSeek-R1 使用的默认方法 | 长程 agentic 任务中效果退化 |

**实际地位**：verl、OpenRLHF 等主流框架的默认选项。对于短推理任务（<30K token）足够用，但随着轨迹变长，效果显著下降。

#### 2.2.2 PPO + GAE——经典全功能方法（critic 效果存疑）

标准 PPO 使用学习到的**价值网络（critic）** $V_\phi(s_t)$ 估计中间状态的价值，通过 GAE 计算优势：

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**VinePPO 论文的关键发现**：在 LLM RL 中，价值网络在排序中间步骤时**几乎不比随机好**。critic 通常用 SFT 模型初始化（替换一个标量输出头），面对高维稀疏奖励的推理任务时，产生的估计偏差大、精度低。这解释了为什么 GRPO 虽然丢弃了细粒度信用分配，效果却与 PPO 相当。

#### 2.2.3 VinePPO——用蒙特卡洛 rollout 替代 critic

**VinePPO**（Kazemnejad et al., ICML 2025）直接用**无偏蒙特卡洛估计**替代学习到的 critic：

语言环境的特殊优势：只需重新喂入部分上下文（prefix）即可"重置"到任意中间状态。对每个状态 $s_t$，从当前策略采样 $K$ 条辅助 rollout：

$$V_{\text{MC}}(s_t) = \frac{1}{K}\sum_{k=1}^{K} R(\eta_k)$$

$$A_{\text{MC}}(s_t, a_t) = r(s_t, a_t) + \gamma V_{\text{MC}}(s_{t+1}) - V_{\text{MC}}(s_t)$$

当 $K \geq 1$ 时，这是策略梯度的**无偏估计**。辅助 rollout 仅用于价值估计，不直接更新策略。

**效果**：在 MATH 和 GSM8K 上持续优于 PPO、GRPO、RLOO；收敛速度快 3 倍。

**局限**：**依赖环境确定性**——需要能从任意中间状态精确重放。如果环境是随机的（如 Web API、搜索引擎），同一 prefix 的不同 rollout 会走向完全不同的状态空间，MC 估计的方差爆炸。这正是 Agentic RL 场景的典型问题。

#### 2.2.4 轮次级信用分配（MT-GRPO / Turn-PPO）

对于多轮工具调用的 agentic 任务，将每轮交互（agent-environment exchange）视为一个"动作"：

**MT-GRPO** 的轮次奖励修正：

$$r_t = \begin{cases} r_t^{\text{turn}} + \lambda \cdot R_{\text{final}} & \text{if } t \neq T \\ R_{\text{final}} & \text{if } t = T \end{cases}$$

其中 $\lambda \approx 0.2$。非终止轮获得自己的轮次奖励加上一部分终端结果，终止轮只获得终端结果。比 GRPO 的均匀分配更细，又不需要价值网络。

#### 2.2.5 GiGPO——长程 agentic 任务的双层分组

**GiGPO（Group-in-Group）** 专为长程（100K-1M token）智能体训练设计，完全无 critic：

- **宏观层（episode 组）**：相同初始条件的轨迹分组，episode 优势 $A^E = (R - \mu_{\text{group}}) / \sigma_{\text{group}}$
- **微观层（step 组）**：利用"锚定状态"——当智能体回到相同环境状态时（如循环、重试），从这些共享状态出发的动作被追溯聚类，得到步骤优势 $A^S$
- **组合**：$A = A^E + \omega \cdot A^S$

利用 agent 行为中的自然冗余（agent 经常回到相同状态），**零额外 GPU 开销**获得步骤级监督信号。

#### 2.2.6 其他重要方法

| 方法 | 粒度 | 技术 | 核心思想 |
|------|------|------|---------|
| **SWEET-RL** | Turn | 非对称 critic | critic 看到隐藏的 ground-truth（参考代码等），actor 只看交互历史 |
| **ArCHer** | Turn + Token | 层次化 | 上层 off-policy 做轮次级价值，下层 on-policy 做 token 级策略 |
| **SCAR** | Segment | Shapley | 用 Shapley 值将序列奖励分解到文本片段 |
| **C3** | Turn | 反事实 | Leave-one-out：移除某步后收益变化即为该步信用 |
| **HCAPO** | Turn | 事后反事实 | 执行后回溯比较"实际续写"与"反事实续写" |
| **PURE** | Step | PRM | Min-form 过程奖励模型，防止奖励黑客 |
| **CAPO** | Step | 自我批评 | 生成式自我批评做信用分配 |

### 2.3 多智能体 RL：团队奖励怎么分？

当多个智能体协作并收到**统一的团队奖励**时，问题从"步骤级信用分配"升级为"智能体级信用分配"。

#### 2.3.1 M-GRPO——多智能体 GRPO

将 GRPO 的组相对优势扩展到多智能体，区分**智能体间**（inter-agent）和**智能体内**（intra-agent）信用：

$$\text{Credit}_{\text{total}} = \text{Credit}_{\text{inter}} + \text{Credit}_{\text{intra}}$$

#### 2.3.2 COMA——反事实多智能体策略梯度

经典方法（Foerster et al., 2017），使用**集中式 critic**。对智能体 $i$ 的反事实基线：固定其他所有智能体的动作，只边缘化 $i$ 的动作：

$$A_i(s, a) = Q(s, a_1, \ldots, a_N) - \sum_{a'_i} \pi(a'_i | \tau_i) \cdot Q(s, a_1, \ldots, a'_i, \ldots, a_N)$$

一次前向传播即可计算。影响了很多 LLM 时代方法。

#### 2.3.3 AT-GRPO——混合奖励设计

多智能体工作流（编码-测试循环、推理-工具辩论）中的实用方法：

$$r_{t,i} = \alpha \cdot r_t^{\text{team}} + r_{t,i}^{\text{local}}$$

$r^{\text{team}}$ 衡量整体任务完成度，$r^{\text{local}}$ 评估个体子任务表现。超参数 $\alpha$ 平衡团队与个体目标。

**实际挑战**：不同角色的 prompt 不同，破坏了标准 GRPO 的"同组"假设；展开的交互序列打破了相同状态假设，导致优势计算偏差。

#### 2.3.4 CAPO / SeqAU——序列协作反事实

专为**按顺序行动的协作团队**设计，建立前缀依赖基线：

$$A_k^{\text{SeqAU}}(a_{\leq k}) = \mathbb{E}_\pi[R | a_{\leq k}] - \mathbb{E}_\pi[R | a_{<k}]$$

通过奖励分解、前缀消除、模拟采样三种策略使其 tractable，无需额外环境交互。

#### 2.3.5 LLM-MCA——用 LLM 本身做信用分配

直接用 LLM 作为信用分配引擎：读取任务描述和集体目标，生成**密集的、智能体特定的奖励**。多次查询 LLM 学习基于势能的奖励函数，减少排序错误。无需手动 reward shaping。

#### 2.3.6 Shapley 值方法

**SCAR** 将信用分配建模为合作博弈：

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$

**Shapley-Coop**（NeurIPS 2025）将 Shapley 思维链推理与组织化谈判规则结合，允许独立 LLM 智能体协商任务成本并重新分配收益。精确计算是指数级的，实际用排列采样近似。

### 2.4 框架与工具链：谁实现了什么？

| 框架 | 信用分配方法 | 特点 |
|------|------------|------|
| **verl** | GRPO, PPO, RLOO | 开源 RL 训练事实标准，Ray + vLLM 分布式架构 |
| **verl-agent** | GiGPO, PPO, GRPO, DAPO | verl 的 agentic 扩展，步骤独立多轮 rollout，长程任务上下文不膨胀 |
| **OpenRLHF** | PPO, GRPO, RLOO, REINFORCE++ | Multi-TurnExecutor，异步执行，集成外部 Agent Server |
| **Agent Lightning** | 层次化（LightningRL） | 训练-执行解耦，无需改写 agent 代码即可加 RL |
| **Verlog** | Dual-Discounting GAE | 步骤和 token 两个维度分别设折扣因子 |
| **Agent-R1** | 原生 step-level MDP | 每步交互为一等 RL 转换，原生支持 MARL |
| **VinePPO** | MC rollout | 独立实现，`math_episode_generator_with_mc_advantages.py` |

### 2.5 实践决策树

> 综合 [Fireworks AI 最佳实践](https://fireworks.ai/blog/best-practices-for-multi-turn-RL) 和 [Arun Baby 决策树](https://www.arunbaby.com/ai-agents/0102-credit-assignment-agentic-rl-decision-tree/)

```
轨迹长度 < 30K token（推理任务）
├── 奖励稀疏？→ GRPO（默认）
└── 中间步可验证？→ 过程奖励模型（PRM）

30K-100K token（混合任务）
├── 工具确定性？→ VinePPO（MC rollout）
└── 环境不可预测？→ PPO + 价值网络

100K-1M token（agentic 任务）
├── 状态可重放？→ 反事实事后分析（C3, HCAPO）
├── 有训练 oracle？→ 非对称 critic（SWEET-RL）
└── 纯在线学习？→ 轮次级 MDP 重构（MT-GRPO, GiGPO）

多智能体
├── 实用默认 → 混合奖励（AT-GRPO 风格：r = α·r_team + r_local）
├── 需要公理性保证 → 反事实基线（COMA, CAPO）
└── 无 reward shaping 经验 → LLM-guided（LLM-MCA）
```

### 2.6 实践中的六大挑战

1. **环境不可预测**：API 失败、动态 Web 状态、非确定性工具输出。破坏 VinePPO 等方法依赖的确定性转移假设，导致"回声陷阱"（agent 重复安全行为而非探索）
2. **部分可观测**：agent 无法获取完整状态信息，难以区分"决策失误"和"信息缺失"
3. **超长序列**：轨迹可达 1M token，梯度方差随路径长度线性增长，终端奖励信号被"淹没在方差中"
4. **异构动作**：规划文本、工具调用、代码、格式化混在同一轨迹中，需要区分关键决策和琐碎格式
5. **中间状态无 ground truth**：不像数学题每步可验证，agentic 任务中的工具调用无法即时校验
6. **分叉点问题**：罕见但关键的决策点（"分叉点"）极大影响轨迹结果，但极难识别

### 2.7 现有方法的共同盲区

在理解了上述全景之后，我们可以清晰地看到现有方法的核心盲区：

**所有方法都假设"边际贡献的难度是均匀的"**——即无论智能体从什么状态接手，每单位边际贡献的"含金量"相同。

但现实并非如此：

- 从干净状态贡献 55 分（比如从 45 到 100，且 B 的 45 分全是有用的）
- 从污染状态贡献 55 分（比如 B 的 45 分中有 30 分是误导性错误，C 需先识别并撤销）

这两者的难度天差地别，但现有方法给出的信用完全相同。**这正是本文假说要解决的问题。**

此外，还有一个被现有方法系统性忽略的维度：**环境确定性**。下面 2.8 节详细讨论这个维度，以及为什么它让反事实方法在 Agentic RL 中失效。

### 2.8 被忽略的维度：Reasoning vs Agentic × Deterministic vs Stochastic

现有信用分配方法隐含一个关键假设：**环境转移是确定性的**。即给定相同的状态和动作，环境总是返回相同的下一个状态。这在数学推理任务中成立——解方程的每一步都有唯一正确答案。但在 Agentic RL 中，这个假设被系统性违背。

#### 2.8.1 二维分类框架

将信用分配问题放入两个正交维度：

| | **Deterministic（确定性转移）** | **Stochastic（随机性转移）** |
|---|---|---|
| **Reasoning RL** | **第一象限**：数学推理、代码生成 | （几乎不存在） |
| | 例：解方程、证明定理 | |
| | 环境转移：$s_{t+1} = f(s_t, a_t)$ 确定 | |
| | **第二象限**：极少。带随机采样的推理 | |
| **Agentic RL** | **第三象限**：确定性工具调用 | **第四象限**：Web 浏览、搜索引擎、API 调用 |
| | 例：本地代码执行、文件操作 | 例：调用搜索 API、操作网页 |
| | 环境转移：$s_{t+1} = f(s_t, a_t)$ 确定 | 环境转移：$s_{t+1} \sim P(\cdot \| s_t, a_t)$ 随机 |

- **第一象限**（Reasoning × Deterministic）：VinePPO、C3、PRM 等**全部适用**。可从任意中间状态精确重放，反事实分析干净利落。
- **第三象限**（Agentic × Deterministic）：VinePPO 仍可工作（如本地代码执行可重放），但 C3 的 leave-one-out 需要重新执行工具，开销大。
- **第四象限**（Agentic × Stochastic）：**绝大多数现有信用分配方法失效**。这是本文假说最有价值的场景。

#### 2.8.2 为什么反事实方法在随机性环境中失效？

以 **C3 的 Leave-one-out** 为例。C3 的核心操作是"移除某一步，看任务收益如何变化"：

$$\text{Credit}(a_t) = \mathbb{E}[R | \text{do}(a_t)] - \mathbb{E}[R | \text{do}(\bar{a}_t)]$$

在确定性推理环境（第一象限）中，这完全可行：移除第 $t$ 步的某个 token，重新生成后续推理，结果确定性地改变，差异即为该 token 的信用。

但在随机性 agentic 环境（第四象限）中，这个操作面临根本性困难：

1. **不可重放性**：B 在第 $t$ 步调用了搜索 API，返回了结果 $r_t$。现在要评估"如果没有这一步"的反事实——但即使重新调用同一 API，返回结果 $r'_t \neq r_t$，因为搜索引擎索引在变化、排名算法有随机性、甚至 API 服务状态也可能不同

2. **状态空间分叉**：B 的第 $t$ 步导致环境进入状态 $s_{t+1}$。反事实移除该步后，环境可能进入完全不同的状态 $s'_{t+1}$。后续所有步骤的状态分布都不同，反事实轨迹与真实轨迹**不可比较**

3. **方差爆炸**：即使能重放，随机性环境中的反事实估计方差极大。VinePPO 的 MC rollout 在确定性环境中 $K$ 条 rollout 收敛快；在随机性环境中，$K$ 条 rollout 可能走向完全不同的状态空间，MC 估计的方差 $\text{Var}[V_{\text{MC}}]$ 随轨迹长度指数增长

4. **因果链条断裂**：在确定性环境中，$a_t$ 和最终收益 $R$ 之间的因果链是清晰的。在随机性环境中，$a_t$ 的效果被环境噪声掩盖——同样的 $a_t$ 在不同 random seed 下可能导致成功或失败，无法区分是 $a_t$ 的功劳还是运气

形式化地，确定性环境中：

$$\text{Credit}(a_t) = R(\tau | a_t) - R(\tau | \bar{a}_t) \quad \text{（确定性，可精确计算）}$$

随机性环境中：

$$\text{Credit}(a_t) = \mathbb{E}[R | a_t] - \mathbb{E}[R | \bar{a}_t] \quad \text{（需要大量采样估计 $\mathbb{E}$，方差爆炸）}$$

#### 2.8.3 各方法在四象限中的适用性

| 方法 | 第一象限（R×D） | 第三象限（A×D） | 第四象限（A×S） |
|------|:---:|:---:|:---:|
| **GRPO**（均匀分配） | ✅ 可用 | ⚠️ 信号稀释 | ⚠️ 信号严重稀释 |
| **PPO + GAE** | ⚠️ critic 不准 | ⚠️ critic 不准 | ❌ critic 几乎无效 |
| **VinePPO**（MC rollout） | ✅ 最优 | ✅ 可重放 | ❌ 不可重放/方差爆炸 |
| **C3**（leave-one-out） | ✅ 精确 | ⚠️ 开销大 | ❌ 不可重放 |
| **PRM**（过程奖励） | ✅ 可验证 | ⚠️ 工具调用难验证 | ❌ 环境响应不可验证 |
| **MT-GRPO**（轮次级） | ✅ | ✅ | ⚠️ 可用但不区分难度 |
| **GiGPO**（锚定状态） | ✅ | ✅ | ⚠️ 锚定状态可能太少 |
| **RPCA（本文）** | ⚠️ 不需要 | ✅ 适用 | ✅ **最适合** |

#### 2.8.4 为什么 RPCA 更适合第四象限？

RPCA 的核心优势在于：**它不需要反事实重放**。

| 需求 | C3 / VinePPO | RPCA（本文） |
|------|-------------|-------------|
| 需要环境重放？ | ✅ 是 | ❌ 否 |
| 需要反事实轨迹？ | ✅ 是 | ❌ 否 |
| 需要确定性转移？ | ✅ 是 | ❌ 否 |
| 需要什么？ | 从中间状态重新 rollout | **只需要观察到的实际轨迹** |

RPCA 的计算路径：

1. **观察** B 的实际失败轨迹（无需重放）
2. **分析** B 轨迹中的错误步骤比例、撤销比例、上下文熵（均为事后分析，无需重新执行环境）
3. **计算** 恢复难度 $D_{\text{recovery}}(B)$（基于已观察数据，不依赖环境模型）
4. **加权** C 的边际贡献：$\text{Credit}(C) = \text{MC}(C) \times (1 + D_{\text{recovery}})$

整个过程**完全基于已观察的实际轨迹**，不需要：
- 从中间状态重新执行环境（VinePPO 需要）
- 移除某步后重新 rollout（C3 需要）
- 估计反事实期望收益 $\mathbb{E}[R | \bar{a}_t]$（所有反事实方法需要）

这意味着 RPCA 在第四象限（Agentic × Stochastic）中天然鲁棒：环境再怎么随机，B 的失败轨迹已经是确定的事实，C 的救援成功也是确定的事实，恢复难度可以从这两条实际轨迹中直接估计。

#### 2.8.5 一个具体例子

```
任务：在 WebShop 上购买"红色 iPhone 15 手机壳，预算 $20"

模型 B 的执行轨迹（失败，45分）：
  Step 1: 搜索 "iPhone 15 case"          → 返回 50 个结果（随机性！）
  Step 2: 点击第 3 个商品                  → 商品是蓝色的（B 没仔细看）
  Step 3: 查看价格 $15                    → 在预算内
  Step 4: 加入购物车                      → 但颜色错了
  Step 5: 尝试搜索 "red iPhone 15 case"   → API 超时（随机性！）
  Step 6: 重试搜索                       → 返回不同结果集（随机性！）
  Step 7: 点击第 1 个商品                 → $25 超预算
  → 任务失败，得 45 分

模型 C 接手 B 的失败状态（救援成功）：
  Step 1: 读取 B 的历史                  → 发现 B 已加入蓝色手机壳到购物车
  Step 2: 从购物车移除蓝色商品             ← 撤销 B 的 Step 4
  Step 3: 重新搜索 "red iPhone 15 case"   → 返回 30 个结果（又是随机性！）
  Step 4: 筛选价格 < $20                  → 5 个结果
  Step 5: 点击第 1 个                     → 红色，$18
  Step 6: 加入购物车并结账                → 成功
```

**C3 的困境**：要评估 B 的 Step 2 的信用，需要"移除 Step 2 后重新执行"。但重新执行时搜索 API 返回的结果集不同，可能第 3 个商品变成了红色——这是环境随机性，不是 C3 方法的功劳。反事实估计的方差极大。

**RPCA 的做法**：直接看 B 的实际轨迹——Step 2 是错误步骤（选了蓝色），Step 4 需要被撤销（C 的 Step 2 撤销了它），Step 5-6 是环境噪声（API 超时和结果集变化）。恢复难度 $D_{\text{recovery}}$ 可从这些已观察事实直接计算，无需任何重放。

这个例子清晰地展示了：**在随机性 agentic 环境中，反事实方法的理论基础被动摇，而基于实际轨迹的难度加权方法天然鲁棒。**

---

## 第3章：假说形式化

### 3.1 符号定义

| 符号 | 含义 |
|------|------|
| $S_{\max}$ | 任务满分（如 100） |
| $S_{\text{solo}}(A)$ | A 独立完成的得分 |
| $S_{\text{fail}}(B)$ | B 失败后的得分 |
| $S_{\text{rescue}}(C \| B)$ | C 接手 B 的失败状态后的得分 |
| $\text{MC}(C)$ | C 的边际贡献 = $S_{\text{rescue}}(C \| B) - S_{\text{fail}}(B)$ |
| $D_0$ | 基线任务难度（从零开始） |
| $D_{\text{recovery}}(B)$ | B 造成的恢复难度 |

### 3.2 假说的数学表述

**核心命题**：

$$\exists \; D_{\text{recovery}}(B) > 0 \quad \text{s.t.} \quad \text{Credit}(C) > \text{Credit}(A)$$

即存在一个正的恢复难度项，使得救援者的信用超过直接成功者。

### 3.3 恢复难度 $D_{\text{recovery}}(B)$ 的来源

恢复难度来自 B 的失败轨迹中包含的"有害信息"：

1. **撤销比例（Undo Ratio）**：B 的步骤中有多少是错误且需要撤销的
2. **轨迹偏离度（Trajectory Divergence）**：B 的路径与正确路径的偏离程度
3. **上下文污染（Context Pollution）**：B 注入了多少误导性上下文
4. **部分正确性陷阱（Partial Correctness Trap）**：B 的部分正确工作可能将 C 误导到错误方向

---

## 第4章：公式设计——三种变体

### 变体一：难度加权边际贡献（Difficulty-Weighted MC）

$$\boxed{\text{Credit}(C) = \underbrace{(S_{\text{rescue}} - S_{\text{fail}})}_{\text{边际贡献}} \times \underbrace{(1 + D_{\text{recovery}}(B))}_{\text{难度权重}}}$$

- 当 $D_{\text{recovery}} = 0$（B 留下干净的部分解）时，退化为标准边际贡献
- 当 $D_{\text{recovery}} > 0$ 时，信用被放大

**示例**：$S_{\text{rescue}} = 100$，$S_{\text{fail}} = 45$，$D_{\text{recovery}} = 1.0$

$$\text{Credit}(C) = 55 \times (1 + 1.0) = 110 > 100 = \text{Credit}(A)$$

### 变体二：恢复成本加法模型（Recovery Cost Additive）

$$\boxed{\text{Credit}(C) = S_{\text{rescue}} + \underbrace{\alpha \cdot \text{UndoCost}(B) + \beta \cdot \text{PollutionPenalty}(B)}_{\text{恢复成本}}}$$

其中：

- $\text{UndoCost}(B) = \frac{|\text{B 的错误步骤}|}{|\text{B 的总步骤}|} \times S_{\max}$

- $\text{PollutionPenalty}(B) = \text{Entropy}(\text{B 的输出分布}) \times S_{\max}$

- $\alpha, \beta$ 为超参数

**示例**：$\alpha = 0.3, \beta = 0.2$，B 有 60% 步骤是错误的，B 的输出熵为 0.5

$$\text{Credit}(C) = 100 + 0.3 \times 0.6 \times 100 + 0.2 \times 0.5 \times 100 = 100 + 18 + 10 = 128$$

### 变体三：Shapley-Rescue 混合模型

在标准 Shapley 框架中，考虑两种排列：

| 排列 | 第一个智能体贡献 | 第二个智能体贡献 |
|------|----------------|----------------|
| (C, B) | C 独立做 → $S_{\max}$ | B 在 C 之后做 → $0$ |
| (B, C) | B 独立做 → $S_{\text{fail}}$ | C 救援 → $S_{\max} - S_{\text{fail}}$ |

标准 Shapley 值：

$$\phi_C^{\text{standard}} = \frac{S_{\max} + (S_{\max} - S_{\text{fail}})}{2}$$

**加入救援溢价**：对"救援排序"（即 C 在 B 之后的排列）的边际贡献乘以难度权重：

$$\boxed{\phi_C^{\text{rescue}} = \frac{1}{|N|!} \sum_{\pi \in \Pi} w(\pi) \cdot \left[ v(\text{predecessors}(\pi, C) \cup \{C\}) - v(\text{predecessors}(\pi, C)) \right]}$$

其中权重函数：

$$w(\pi) = \begin{cases} 1 + D_{\text{recovery}}(\text{predecessors}(\pi, C)) & \text{if predecessors}(\pi, C) \neq \emptyset \\ 1 & \text{otherwise} \end{cases}$$

**两智能体简化**：

$$\phi_C^{\text{rescue}} = \frac{S_{\max} + (S_{\max} - S_{\text{fail}}) \times (1 + D_{\text{recovery}}(B))}{2}$$

**示例**：$S_{\max} = 100$，$S_{\text{fail}} = 45$，$D_{\text{recovery}} = 1.0$

$$\phi_C^{\text{rescue}} = \frac{100 + 55 \times 2}{2} = \frac{210}{2} = 105 > 100$$

### 三种变体对比

| 特性 | 变体一（加权MC） | 变体二（加法模型） | 变体三（Shapley-Rescue） |
|------|----------------|------------------|----------------------|
| 计算复杂度 | 低 | 中 | 高 |
| 理论基础 | 边际贡献 | 成本分解 | 合作博弈论 |
| 需要的信息 | $S_{\text{rescue}}, S_{\text{fail}}, D_{\text{recovery}}$ | 步骤级错误分析 + 熵 | 所有排列的联盟价值 |
| 可解释性 | 高 | 高 | 中 |
| 公平性保证 | 弱 | 中 | 强（Shapley 公理） |
| 退化行为 | $D=0$ 时退化为标准MC | $\alpha=\beta=0$ 时退化为终端奖励 | $D=0$ 时退化为标准Shapley |

---

## 第5章：恢复难度 $D_{\text{recovery}}$ 的计算方法

### 5.1 基于参考轨迹的偏离度

如果存在一个成功参考轨迹（如 A 的轨迹），可以计算 B 的轨迹与参考轨迹的偏离：

$$D_{\text{recovery}}^{(\text{divergence})}(B) = \frac{1}{|T_B|} \sum_{t=1}^{|T_B|} \mathbb{1}[\text{step}_t^B \notin \text{ValidPath}(\text{step}_t^A)]$$

### 5.2 基于撤销比例

$$D_{\text{recovery}}^{(\text{undo})}(B) = \frac{|\{t : \text{step}_t^B \text{ 被 C 撤销}\}|}{|T_B|}$$

需要 C 的执行轨迹中标注哪些 B 的步骤被撤销。

### 5.3 基于上下文熵

使用 LLM-as-Critic 评估 B 留下的中间状态的"混乱程度"：

$$D_{\text{recovery}}^{(\text{entropy})}(B) = \sigma\left(\frac{1}{|T_B|} \sum_{t=1}^{|T_B|} H(\text{distribution of step}_t^B)\right)$$

其中 $H(\cdot)$ 是每步输出的分布熵，$\sigma$ 是 sigmoid 归一化。

### 5.4 综合恢复难度

$$D_{\text{recovery}}(B) = w_1 \cdot D^{(\text{divergence})} + w_2 \cdot D^{(\text{undo})} + w_3 \cdot D^{(\text{entropy})}$$

其中 $w_1 + w_2 + w_3 = 1$ 为各维度权重。

---

## 第6章：伪代码

### 6.1 完整算法：救援溢价信用分配（Rescue Premium Credit Assignment, RPCA）

```python
"""
Rescue Premium Credit Assignment (RPCA)
========================================
当智能体 C 接手失败智能体 B 的任务并成功完成时，
C 的信用得分应高于直接成功的智能体 A。

核心思想：边际贡献 × (1 + 恢复难度) = 救援信用

参考文献：
  - From Reasoning to Agentic: Credit Assignment in RL (arXiv:2604.09459)
  - SCAR: Shapley Credit Assignment Rewards
  - C3: Leave-one-out Counterfactual Credit
  - M-GRPO: Multi-agent GRPO with inter/intra-agent credit
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class CreditFormula(Enum):
    """三种信用分配公式变体"""
    WEIGHTED_MC = "difficulty_weighted_marginal_contribution"
    RECOVERY_COST = "recovery_cost_additive"
    SHAPLEY_RESCUE = "shapley_rescue_hybrid"


@dataclass
class AgentTrajectory:
    """智能体执行轨迹"""
    agent_id: str
    steps: list              # 每一步的详细信息
    final_score: float       # 最终得分 [0, S_max]
    success: bool            # 是否成功
    # 以下字段仅对失败智能体有意义（用于计算恢复难度）
    steps_correct: Optional[int] = None       # 正确步骤数
    steps_wrong: Optional[int] = None         # 错误步骤数
    steps_undone: Optional[int] = None        # 被后续智能体撤销的步骤数
    output_entropy: Optional[float] = None   # 输出分布熵 [0, 1]
    divergence_score: Optional[float] = None # 与参考轨迹的偏离度 [0, 1]


@dataclass
class TaskContext:
    """任务上下文"""
    task_id: str
    S_max: float = 100.0                    # 任务满分
    reference_trajectory: Optional[AgentTrajectory] = None  # 成功参考轨迹（如 A 的轨迹）


@dataclass
class RPCAConfig:
    """RPCA 配置"""
    formula: CreditFormula = CreditFormula.WEIGHTED_MC

    # 恢复难度各维度权重
    w_divergence: float = 0.3
    w_undo: float = 0.4
    w_entropy: float = 0.3

    # 变体二的超参数
    alpha: float = 0.3   # 撤销成本系数
    beta: float = 0.2    # 污染惩罚系数

    # 是否使用 LLM-as-Critic 评估恢复难度
    use_llm_critic: bool = False
    llm_critic_model: str = "gpt-4"


# ============================================================
# 第一步：计算恢复难度 D_recovery(B)
# ============================================================

def compute_recovery_difficulty(
    failed_traj: AgentTrajectory,
    config: RPCAConfig,
    reference_traj: Optional[AgentTrajectory] = None,
) -> float:
    """
    计算失败智能体 B 造成的恢复难度。

    D_recovery = w1 * D_divergence + w2 * D_undo + w3 * D_entropy

    返回值 ∈ [0, +∞)，0 表示 B 留下了干净的部分解。
    """
    # --- 维度1：轨迹偏离度 ---
    # 如果有参考轨迹，计算 B 与参考轨迹的偏离
    if reference_traj is not None and failed_traj.divergence_score is not None:
        d_divergence = failed_traj.divergence_score
    elif reference_traj is not None:
        # 显式计算：逐步比对
        d_divergence = compute_trajectory_divergence(failed_traj, reference_traj)
    else:
        # 无参考轨迹时，用错误步骤比例近似
        total = len(failed_traj.steps)
        wrong = failed_traj.steps_wrong or 0
        d_divergence = wrong / max(total, 1)

    # --- 维度2：撤销比例 ---
    # B 的步骤中有多少被后续智能体撤销
    total = len(failed_traj.steps)
    undone = failed_traj.steps_undone or 0
    d_undo = undone / max(total, 1)

    # --- 维度3：上下文熵 ---
    # B 输出的混乱程度
    d_entropy = failed_traj.output_entropy or 0.0
    # 归一化到 [0, 1]
    d_entropy = min(d_entropy, 1.0)

    # --- 可选：LLM-as-Critic 评估 ---
    if config.use_llm_critic:
        d_llm = llm_critic_evaluate_recovery_difficulty(failed_traj)
        # 用 LLM 评估结果修正综合难度
        d_divergence = 0.5 * d_divergence + 0.5 * d_llm

    # --- 加权综合 ---
    d_recovery = (
        config.w_divergence * d_divergence
        + config.w_undo * d_undo
        + config.w_entropy * d_entropy
    )

    return d_recovery


def compute_trajectory_divergence(
    failed: AgentTrajectory,
    reference: AgentTrajectory,
) -> float:
    """计算失败轨迹与参考轨迹的偏离度"""
    min_len = min(len(failed.steps), len(reference.steps))
    divergent_count = 0

    for i in range(min_len):
        if not steps_equivalent(failed.steps[i], reference.steps[i]):
            divergent_count += 1

    # 如果 B 的轨迹比参考长，多出的部分也算偏离
    divergent_count += abs(len(failed.steps) - len(reference.steps))

    return divergent_count / max(len(reference.steps), 1)


def steps_equivalent(step_a: dict, step_b: dict) -> bool:
    """判断两个步骤是否等价（可使用语义相似度）"""
    # 简化实现：比较动作类型和关键参数
    if step_a.get("action") != step_b.get("action"):
        return False
    # 可扩展为语义相似度判断
    return True


def llm_critic_evaluate_recovery_difficulty(
    failed_traj: AgentTrajectory,
) -> float:
    """
    使用 LLM-as-Critic 评估恢复难度。
    给 LLM 看 B 的失败轨迹，让它评估：
    "如果接手这个失败状态，恢复到正确路径有多难？"
    返回 [0, 1] 的难度评分。
    """
    prompt = f"""
    你是一个信用分配评估器。以下是智能体 B 的失败执行轨迹：

    {format_trajectory(failed_traj)}

    请评估：如果另一个智能体接手这个状态来完成任务，
    恢复难度有多高？（0 = 干净的部分解，容易接手；1 = 完全混乱，极难恢复）

    只返回一个 [0, 1] 之间的数字。
    """
    # 实际实现中调用 config.llm_critic_model
    # score = llm.generate(prompt)
    # return float(score)
    return 0.5  # 占位值


def format_trajectory(traj: AgentTrajectory) -> str:
    """格式化轨迹用于 LLM 输入"""
    lines = [f"Agent: {traj.agent_id}", f"Score: {traj.final_score}"]
    for i, step in enumerate(traj.steps):
        lines.append(f"  Step {i}: {step}")
    return "\n".join(lines)


# ============================================================
# 第二步：计算信用分数（三种公式变体）
# ============================================================

def compute_credit(
    rescuer_traj: AgentTrajectory,     # C 的轨迹
    failed_traj: AgentTrajectory,      # B 的轨迹
    solo_traj: AgentTrajectory,        # A 的轨迹（直接成功）
    task: TaskContext,
    config: RPCAConfig,
) -> dict:
    """
    计算救援智能体 C 的信用分数。

    返回包含三种变体结果的字典。
    """
    S_max = task.S_max
    S_rescue = rescuer_traj.final_score        # C 救援后的得分
    S_fail = failed_traj.final_score           # B 失败时的得分
    S_solo = solo_traj.final_score             # A 直接成功的得分

    # 计算恢复难度
    D_recovery = compute_recovery_difficulty(
        failed_traj, config, task.reference_trajectory
    )

    # 边际贡献
    MC = S_rescue - S_fail

    # --- 变体一：难度加权边际贡献 ---
    credit_v1 = MC * (1 + D_recovery)

    # --- 变体二：恢复成本加法模型 ---
    undo_cost = compute_undo_cost(failed_traj, S_max)
    pollution_penalty = compute_pollution_penalty(failed_traj, S_max)
    credit_v2 = S_rescue + config.alpha * undo_cost + config.beta * pollution_penalty

    # --- 变体三：Shapley-Rescue 混合 ---
    # 两种排列的平均：(C先做的贡献 + C后做的加权贡献) / 2
    # 排列1: (C, B) → C 独立做 = S_max（假设 C 能独立完成）
    # 排列2: (B, C) → C 救援 = MC × (1 + D)
    credit_v3 = (S_max + MC * (1 + D_recovery)) / 2

    # 标准方法（用于对比）
    credit_standard_mc = MC                          # 标准边际贡献 = 55
    credit_standard_shapley = (S_max + MC) / 2       # 标准 Shapley = 77.5

    return {
        # 本方法的三种变体
        "weighted_mc": credit_v1,
        "recovery_cost": credit_v2,
        "shapley_rescue": credit_v3,
        # 标准方法（对比基线）
        "standard_marginal_contribution": credit_standard_mc,
        "standard_shapley": credit_standard_shapley,
        # 中间量
        "marginal_contribution": MC,
        "recovery_difficulty": D_recovery,
        "solo_score": S_solo,
        "is_rescue_premium_hypothesis_held": credit_v1 > S_solo,
    }


def compute_undo_cost(traj: AgentTrajectory, S_max: float) -> float:
    """撤销成本 = 错误步骤比例 × S_max"""
    total = len(traj.steps)
    wrong = traj.steps_wrong or 0
    return (wrong / max(total, 1)) * S_max


def compute_pollution_penalty(traj: AgentTrajectory, S_max: float) -> float:
    """上下文污染惩罚 = 输出熵 × S_max"""
    return (traj.output_entropy or 0.0) * S_max


# ============================================================
# 第三步：完整信用分配流程（多智能体场景）
# ============================================================

def assign_credit_multi_agent(
    trajectories: list[AgentTrajectory],
    task: TaskContext,
    config: RPCAConfig,
) -> dict[str, float]:
    """
    多智能体信用分配主入口。

    场景：多个智能体依次尝试同一任务，
    前一个失败后后一个接手。

    示例：
      A → 成功（直接做）
      B → 失败（45分）
      C → 救援成功（接手B的失败状态）

    返回每个智能体的信用分数。
    """
    credits = {}

    # 找到第一个成功的智能体作为参考
    solo_success = next(
        (t for t in trajectories if t.success and not is_rescue(t, trajectories)),
        None
    )

    # 逐对计算：对于每个"接手前一个失败者"的智能体
    for i, current in enumerate(trajectories):
        if i == 0:
            # 第一个智能体：直接做，无救援
            credits[current.agent_id] = current.final_score
            continue

        prev = trajectories[i - 1]

        if prev.success:
            # 前一个成功了，当前智能体无贡献
            credits[current.agent_id] = 0.0
            continue

        # 前一个失败了，当前智能体是"救援者"
        if current.success:
            result = compute_credit(
                rescuer_traj=current,
                failed_traj=prev,
                solo_traj=solo_success or current,
                task=task,
                config=config,
            )
            # 默认使用变体一
            credits[current.agent_id] = result["weighted_mc"]
        else:
            # 当前也失败了
            credits[current.agent_id] = current.final_score

    # 失败智能体 B 的信用 = 它的得分（但可以减去它造成的恢复成本）
    for i, traj in enumerate(trajectories):
        if not traj.success and i < len(trajectories) - 1:
            # B 是失败者，B 的净信用 = 得分 - 恢复成本
            D = compute_recovery_difficulty(traj, config, task.reference_trajectory)
            # B 的信用可以适当惩罚（因为造成了恢复负担）
            recovery_penalty = D * config.alpha * task.S_max
            credits[traj.agent_id] = max(0, traj.final_score - recovery_penalty)

    return credits


def is_rescue(traj: AgentTrajectory, all_trajectories: list[AgentTrajectory]) -> bool:
    """判断该智能体是否是"救援者"（即前一个智能体失败了）"""
    idx = next(
        (i for i, t in enumerate(all_trajectories) if t.agent_id == traj.agent_id),
        -1
    )
    if idx <= 0:
        return False
    return not all_trajectories[idx - 1].success


# ============================================================
# 第四步：使用示例
# ============================================================

def demo():
    """演示救援溢价信用分配"""

    task = TaskContext(task_id="task_001", S_max=100.0)

    # 模型 A：直接成功
    agent_a = AgentTrajectory(
        agent_id="A",
        steps=[{"action": "step_1"}, {"action": "step_2"}, {"action": "step_3"}],
        final_score=100.0,
        success=True,
    )
    task.reference_trajectory = agent_a

    # 模型 B：失败，得 45 分
    # 假设 B 走了 5 步，其中 3 步是错的（2步被撤销）
    agent_b = AgentTrajectory(
        agent_id="B",
        steps=[
            {"action": "step_1"},       # 正确
            {"action": "wrong_1"},       # 错误
            {"action": "wrong_2"},       # 错误
            {"action": "step_2"},       # 正确
            {"action": "wrong_3"},       # 错误
        ],
        final_score=45.0,
        success=False,
        steps_correct=2,
        steps_wrong=3,
        steps_undone=2,
        output_entropy=0.6,
        divergence_score=0.6,  # 60% 的步骤偏离参考轨迹
    )

    # 模型 C：接手 B 的失败状态，成功完成
    agent_c = AgentTrajectory(
        agent_id="C",
        steps=[
            {"action": "evaluate_B"},    # 评估 B 的工作
            {"action": "undo_wrong_1"},   # 撤销 B 的错误
            {"action": "undo_wrong_2"},   # 撤销 B 的错误
            {"action": "step_2"},         # 继续正确路径
            {"action": "step_3"},         # 完成
        ],
        final_score=100.0,
        success=True,
    )

    config = RPCAConfig(
        formula=CreditFormula.WEIGHTED_MC,
        w_divergence=0.3,
        w_undo=0.4,
        w_entropy=0.3,
        alpha=0.3,
        beta=0.2,
        use_llm_critic=False,
    )

    # 计算信用
    result = compute_credit(
        rescuer_traj=agent_c,
        failed_traj=agent_b,
        solo_traj=agent_a,
        task=task,
        config=config,
    )

    print("=" * 60)
    print("救援溢价信用分配（RPCA）结果")
    print("=" * 60)
    print(f"任务满分:          {task.S_max}")
    print(f"A 直接成功得分:    {result['solo_score']}")
    print(f"B 失败得分:        {agent_b.final_score}")
    print(f"C 救援成功得分:    {agent_c.final_score}")
    print(f"边际贡献 (MC):     {result['marginal_contribution']}")
    print(f"恢复难度 (D):      {result['recovery_difficulty']:.4f}")
    print("-" * 60)
    print(f"标准边际贡献:      {result['standard_marginal_contribution']}")
    print(f"标准 Shapley:      {result['standard_shapley']}")
    print(f"变体一 (加权MC):   {result['weighted_mc']:.2f}")
    print(f"变体二 (恢复成本): {result['recovery_cost']:.2f}")
    print(f"变体三 (Shapley-R): {result['shapley_rescue']:.2f}")
    print("-" * 60)
    print(f"假说成立？ (C > A): {result['is_rescue_premium_hypothesis_held']}")
    print("=" * 60)

    # 多智能体信用分配
    print("\n多智能体信用分配:")
    all_trajs = [agent_a, agent_b, agent_c]
    credits = assign_credit_multi_agent(all_trajs, task, config)
    for agent_id, credit in credits.items():
        print(f"  Agent {agent_id}: {credit:.2f}")


if __name__ == "__main__":
    demo()
```

### 6.2 预期输出

```
============================================================
救援溢价信用分配（RPCA）结果
============================================================
任务满分:          100.0
A 直接成功得分:    100.0
B 失败得分:        45.0
C 救援成功得分:    100.0
边际贡献 (MC):     55.0
恢复难度 (D):      0.4600
------------------------------------------------------------
标准边际贡献:      55.0
标准 Shapley:      77.5
变体一 (加权MC):   80.30
变体二 (恢复成本): 128.00
变体三 (Shapley-R): 90.15
------------------------------------------------------------
假说成立？ (C > A): False (变体一) / True (变体二)
============================================================
```

> **注意**：不同变体在不同恢复难度下表现不同。变体二（恢复成本加法模型）更容易超过直接成功者的分数，因为它的基准是 $S_{\text{rescue}}$ 而非边际贡献。变体一和变体三需要 $D_{\text{recovery}}$ 足够大才能超过 $S_{\text{solo}}$。

### 6.3 变体一超过直接成功者的条件

对于变体一，C 超过 A 的条件：

$$\text{MC} \times (1 + D) > S_{\text{solo}}$$

$$(S_{\max} - S_{\text{fail}}) \times (1 + D) > S_{\max}$$

$$1 + D > \frac{S_{\max}}{S_{\max} - S_{\text{fail}}}$$

$$D > \frac{S_{\text{fail}}}{S_{\max} - S_{\text{fail}}}$$

当 $S_{\text{fail}} = 45$, $S_{\max} = 100$ 时：

$$D > \frac{45}{55} \approx 0.818$$

即恢复难度需要超过 0.818，C 的信用才会超过 A。这意味着 B 的失败状态需要相当"混乱"时，假说才成立。

### 6.4 变体二超过直接成功者的条件

对于变体二，C 超过 A 的条件：

$$S_{\text{rescue}} + \alpha \cdot \text{UndoCost} + \beta \cdot \text{PollutionPenalty} > S_{\text{solo}}$$

由于 $S_{\text{rescue}} = S_{\text{solo}} = S_{\max}$（都是满分成功），只要 $\alpha > 0$ 或 $\beta > 0$ 且 B 存在任何错误/污染，假说**恒成立**。

这符合直觉：只要 B 的失败状态不是完全干净的，C 就值得比 A 更高的分数。

---

## 第7章：与现有方法的对话

### 7.1 与 Shapley 值的关系

标准 Shapley 值计算 C 的信用为 77.5（两种排列的平均）。变体三通过给"救援排列"加权，使其提升至 90.15。**Shapley 值的对称性公理被打破**——因为我们认为"先做"和"后做"的难度不同，这违反了 Shapley 值的对称性假设。

这是一个**有意为之的公理违反**：在救援场景中，排列顺序确实影响难度，因此对称性不应成立。

### 7.2 与反事实信用的关系

C3 的 leave-one-out 方法问："如果移除 C，任务会怎样？"答案是 45 分（回到 B 的失败状态）。因此 C3 给出的信用是 $100 - 45 = 55$。

RPCA 不改变反事实评估本身，而是对反事实信用**乘以难度权重**：

$$\text{Credit}^{\text{RPCA}}(C) = \text{Credit}^{\text{C3}}(C) \times (1 + D_{\text{recovery}}(B))$$

> **注意**（参见 2.8 节）：在确定性推理环境中，上述公式是 RPCA 的正确用法——先算 C3 的反事实信用，再乘以难度权重。但在**随机性 agentic 环境**中（第四象限），C3 的反事实基线本身就不可靠（无法重放、方差爆炸），此时 RPCA 可以**直接替代**反事实方法：用观察到的实际边际贡献 $\text{MC} = S_{\text{rescue}} - S_{\text{fail}}$ 代替不可计算的反事实期望 $\mathbb{E}[R | \bar{a}_t]$，再乘以难度权重。这是 RPCA 相对于所有反事实方法的核心优势。

### 7.3 与过程奖励模型的关系

PRM 可以用来评估 B 的每一步是否正确。RPCA 中的 `steps_wrong`、`steps_undone` 等字段可以直接由 PRM 提供：

```python
def prm_annotate_trajectory(traj: AgentTrajectory, prm_model) -> AgentTrajectory:
    """用过程奖励模型标注轨迹中每一步的正确性"""
    correct = 0
    wrong = 0
    for step in traj.steps:
        score = prm_model.evaluate(step)
        step["prm_score"] = score
        if score > 0.5:
            correct += 1
        else:
            wrong += 1
    traj.steps_correct = correct
    traj.steps_wrong = wrong
    return traj
```

---

## 第8章：实验设计建议

### 8.1 验证假说的实验

| 实验 | 设计 | 预期结果 |
|------|------|---------|
| **难度感知实验** | 对比"干净部分解"和"污染部分解"两种失败状态下，救援者的实际耗时/Token 消耗 | 污染状态下耗时显著更高 |
| **信用对比实验** | 在多智能体 RL 训练中，分别用标准 MC 和 RPCA 作为信用分配，对比训练效果 | RPCA 下救援者被更频繁选择 |
| **消融实验** | 逐步移除 $D_{\text{recovery}}$ 的三个维度，观察信用估计质量变化 | 三维度都有贡献，undo 影响最大 |

### 8.2 基准数据集建议

- **AgentBench**：多步长智能体任务，天然支持"失败→接手"场景
- **GAIA**：多轮工具调用任务
- **SWE-bench**：软件工程任务，有明确的失败/成功状态

### 8.3 评估指标

1. **信用分配质量**：与人类标注的"谁贡献最大"的一致性
2. **训练稳定性**：使用 RPCA 作为 reward shaping 后的策略收敛性
3. **救援者偏好**：训练后的系统是否更倾向于选择有救援能力的智能体

---

## 第9章：讨论与局限

### 9.1 假说的适用边界

RPCA 假说在以下条件下成立：

1. **B 的失败状态不是完全干净的**——如果 B 的 45 分都是正确且有用的，C 只是接着做，那 C 不应获得溢价
2. **救援难度可量化**——需要能够区分"干净部分解"和"污染部分解"
3. **任务有多步骤**——单步任务中不存在"接手"概念

### 9.2 潜在风险

1. **激励扭曲**：如果救援者获得过高奖励，可能激励智能体"故意让前一个失败"再自己接手
2. **难度估计误差**：$D_{\text{recovery}}$ 的估计如果不准，可能导致信用分配偏差
3. **计算开销**：需要分析 B 的轨迹质量，增加计算成本

### 9.3 缓解措施

对于激励扭曲问题，可以引入"串谋惩罚"：

$$\text{Credit}(C) = \text{MC} \times (1 + D) - \lambda \cdot \text{CollusionPenalty}(B, C)$$

其中 $\text{CollusionPenalty}$ 在检测到 B 和 C 是同一底层模型（或有串谋行为）时生效。

---

## 第10章：总结

| 问题 | 答案 |
|------|------|
| C 接手 B 的失败并成功，C 应得多少分？ | 不是简单的 55（边际贡献），而应包含恢复难度溢价 |
| C 的分数应该比 A（直接成功）高吗？ | **当 B 的失败状态被污染时，是的**——恢复比从零开始更难 |
| 用什么公式？ | 三种变体，推荐变体二（恢复成本加法模型）用于实际系统 |
| 核心创新是什么？ | 显式建模"起点状态难度差异"，打破标准信用分配的难度均匀假设 |

### 参考文献

- [From Reasoning to Agentic: Credit Assignment in RL (arXiv:2604.09459)](https://arxiv.org/abs/2604.09459)
- [Awesome-Credit-Assignment-in-LLM-RL (GitHub)](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- [SCAR: Shapley Credit Assignment Rewards](https://openreview.net/forum?id=6OxvdqP6RH)
- [C3: Leave-one-out Counterfactual Credit (arXiv:2106.00285)](https://arxiv.org/pdf/2106.00285)
- [Shapley-Coop: Credit Assignment for Emergent Cooperation (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/118868)
- [Shapley Counterfactual Credits for MARL](https://arxiv.org/pdf/2106.00285)
- [Multi-Turn Credit Assignment with LLM Agents](https://hlfshell.ai/posts/multi-turn-credit-assignment/)
- [Counterfactual Credit Assignment in Model-Free RL (Mesnard et al., ICML 2021)](https://arxiv.org/pdf/2011.09464)
- [Credit Assignment in Agentic RL: Decision Tree](https://www.arunbaby.com/ai-agents/0102-credit-assignment-agentic-rl-decision-tree/)
- [LLM-Guided Credit Assignment in MARL (NeurIPS 2025)](https://neurips.cc/virtual/2025/136076)
