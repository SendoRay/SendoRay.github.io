---
title: "TokenCake：面向多智能体 LLM 应用的 KV-Cache 中心服务框架"
date: '2026-07-13'
tags:
- LLM
- KV-Cache
- multi-agent
- inference
- scheduling
- memory-management
- function-call
draft: false
math: true
ShowToc: true
summary: "多智能体 LLM 应用的频繁函数调用带来 KV Cache 的时间低利用与空间争用两大问题。TokenCake 用 agent-aware 的时间调度器（函数调用期主动 offload 空闲 KV Cache）+ 空间调度器（为关键路径智能体预留显存），端到端延迟比 vLLM 降低 47%+，GPU 显存利用率提升最多 16.9 个百分点。"
---

## 论文 PDF

{{< pdf src="pdf/2510-tokencake.pdf" >}}

> arXiv 2510.18586v3（北京大学 & 阿里巴巴，2026-05）。

## 0. 论文图表

> 图表提取命令（见 paperreading.skill §1.3）：
> ```
> python3 scripts/extract_pdf_figures.py assets/pdf/2510-tokencake.pdf \
>     --page 3 --bbox 310 64 574 300 --name fig4-overview --slug 2510-tokencake --dpi 240
> ```

{{< figure src="/images/paperreading/2510-tokencake/fig4-overview.png" title="图 1: TokenCake 总体架构 — Frontend API 将多智能体应用描述为 DAG，并向下游暴露工具阶段、运行时元数据与图结构；Spatial Scheduler（空间调度器）按混合优先级为关键路径智能体预留 GPU 显存；Temporal Scheduler（时间调度器）在函数调用期事件驱动地 offload 空闲 KV Cache 并预测性上传；两者通过 Pressure Snapshot（压力快照）共享一致的显存视图进行协调" width="80%" >}}

{{< figure src="/images/paperreading/2510-tokencake/fig9-e2e-latency.png" title="图 2: 端到端延迟对比（3 模型 × 2 应用 × 2 数据集）— 横轴为应用到达率 QPS，纵轴为平均延迟（秒）。TokenCake（橙）在所有配置下延迟最低；vLLM / vLLM-Prefix（蓝 / 棕）在高负载下因停滞智能体的 KV Cache 占满显存、被迫缩小批量而急剧上升；Mooncake（绿）因 agent-agnostic 仅部分缓解" width="98%" >}}

## 1. 问题定义

### 1.1 解决了什么问题

**一句话**：在 GPU 显存受限的前提下，为「多智能体 + 频繁外部函数调用」这类新型 workload 高效管理 KV Cache，降低端到端延迟、提升显存利用率。

- **问题类别**：LLM 推理服务系统的显存与调度协同优化。
- **输入**：一个由多个专用智能体组成的应用，建模为有向无环图（DAG，节点是智能体 / 计算单元，边是数据依赖），其中智能体会频繁发起外部函数调用（工具调用）。
- **输出**：更低的应用级端到端延迟 + 更高的 GPU KV Cache 利用率。
- **新旧**：这是**旧组件（KV Cache 管理）在新 workload（agentic）下暴露的新问题**。传统 LLM 推理是「一次请求→一次响应」，而多智能体应用具有**双重交互模型**：对外通过函数调用与工具 / 数据源 / API 交互（agent-tool），对内通过 DAG 结构化协作（agent-agent）。

### 1.2 为什么要解决这个问题

动机来自函数调用的**长且高方差**的延迟特征。论文据 MCP（Model Context Protocol）文档与实测整理了常见工具的延迟（Table 1）：

| 工具 | 设备 | 延迟 | 方差 |
|------|------|------|------|
| File System | CPU | 100 ms | 50 ms |
| Git | CPU | 100 ms | 100 ms – 1 s |
| Database (SQLite) | CPU | 100–1000 ms | 500 ms |
| Web Search | CPU | 1–5 s | 1–10 s |
| AI Generation | GPU | 5–30 s | 10–60 s |

这种 workload 让 KV Cache 沿两个维度恶化：

- **时间维度：低利用（Temporal Underutilization）**。智能体执行遵循 `LLM Inference1 ⇒ Function Call ⇒ LLM Inference2` 模式。在函数调用等待外部返回的整段时间里，该智能体的 KV Cache（正是后续推理所需的前缀）**空占 GPU 显存却不产生任何计算**。论文实测峰值时高达 **18.5%** 的 GPU KV Cache 块被停滞智能体占据，直接削减了系统可用于活跃计算的容量。
- **空间维度：争用（Spatial Contention）**。不同**关键性**的智能体竞争有限显存。若一个非关键智能体先到并占据显存，会迫使关键路径上的智能体 KV Cache 被驱逐——论文称之为 **critical inversion（关键倒置）**：非关键工作挤掉关键工作，被驱逐的关键智能体只能付出昂贵的**上下文重计算**，拖慢整个应用。

**不解决的后果**：显存被空闲缓存占据 → 有效批量变小 → 吞吐下降、关键路径延迟升高、重计算浪费算力。**为什么是现在**：agentic 应用（自动编码、深度研究、金融分析）快速兴起，MCP 等标准化了工具调用，使这类 workload 成为主流。

### 1.3 这个问题难在哪

1. **何时 offload**：函数调用时长不可预测（见 Table 1 的高方差），offload/upload 走 PCIe 有固定往返成本，只有当「空闲窗口足够长且释放的显存能被有用工作利用」时 offload 才划算。
2. **区分关键性**：需要知道哪些智能体在关键路径上——这依赖应用级的 DAG 结构，而通用服务系统对此**完全无感知**。
3. **协同而非各自为战**：论文的核心经验发现是——**调度优化与内存管理必须协同**。只做 agent-aware 调度无法阻止内存层面的 critical inversion；只做 agent-agnostic 的 offload 会产生大量无用迁移（migration churn）。二者互不包含。

## 2. 方法

### 2.1 核心思想

**KV-Cache-centric（内存中心）+ Agent-aware（智能体感知）** 的协同设计。

> **Key Insight**：多智能体应用天然暴露了两类被现有系统忽视的信息——① 函数调用给出了**可预测的空闲窗口**；② DAG 给出了**关键性上下文**。TokenCake 正是利用这两点，把「被动、无差别」的 KV Cache 管理，升级为「主动、有区分」的管理。

如图 1，TokenCake 由三部分组成：**Frontend API**（把应用翻译成带注解的图）、**Temporal Scheduler**（管理 KV Cache 的时间维度生命周期）、**Spatial Scheduler**（管理空间维度的显存分区）。

### 2.2 技术细节

#### (1) Frontend API — 暴露被忽视的三类信息

用户把多智能体应用注册为 DAG，API 暴露现有系统缺失的：**图结构、细粒度函数调用阶段、性能元数据**。关键抽象是 `FuncNode`：它把一次函数调用**分解成多个顺序阶段**，从而给时间调度器一个「实时进度视图」而非单一的起止区间，让上传时机更精准；`predict_time` 参数则允许用户直接提供预估执行时间。

#### (2) Temporal Scheduler — 事件驱动的机会主义 offload

把函数调用的空闲期转化为有生产力的调度窗口。由两个运行时事件驱动：`call_start`（发起函数调用 → 触发 offload 决策）与 `call_finish`（工具返回 → 触发恢复）。

**动态预测**：维护 per-function-type 的时间估计，用指数加权移动平均融合用户估计 $t_{user}$ 与历史 $t_{history}$：

$$t_{estimate} = \alpha \cdot t_{user} + (1-\alpha) \cdot t_{history}$$

**机会主义门控（Opportunistic Gate）**：只有三个条件**同时**满足才 offload——① 预测停滞时长足够覆盖一次往返传输；② 存在能利用释放显存块的等待请求；③ 后续上传能在不挤占更重要工作的前提下准备好。传输开销与调度窗口：

$$T_{transfer} = T_{offload}(N_{blocks}) + T_{upload}(N_{blocks}), \quad T_{window} = T_{FC} - T_{transfer}$$

核心决策逻辑（Algorithm 1）：先估传输时间与函数调用时长，若 $T_{fc} \le T_{transfer}$ 则窗口太短、直接拒绝；否则把窗口按观测到的 decode 吞吐换算成 token 容量，用 **first-fit** 找一个增量 KV Cache 需求能装进释放块、且其工作量能在窗口内完成的等待请求；找不到则 offload 无收益，拒绝。此外还有「硬拒绝 + 软打分」：先用 4 条硬规则（CPU 容量不足、停滞短于传输、无请求能装下、显存压力低于阈值）快速否决，再用综合分数权衡 GPU 压力 / 块匹配 / 上传安全裕度，并惩罚 offload 关键路径智能体。

**预测性上传（Predictive Upload）**：在函数调用完成前渐进地把 KV Cache 传回 GPU，隐藏传输延迟。为避免上传抢占活跃请求的显存，采用「渐进预留」——上传预算保护关键等待请求：

$$B_{upload} = \max(0,\ B_{gpu}^{free} - \max(0,\ D_{critical} - B_{shared}^{free})), \quad B_{reserve} = \min(B_{remain},\ \lceil B_{deficit}/2 \rceil,\ B_{upload})$$

即每个调度步最多预留候选缺口的一半，跨多个周期摊销分配，既保证上传就绪又不驱逐关键等待工作。

#### (3) Spatial Scheduler — 动态内存分区 + 混合优先级

在**内存层面**（而非仅调度层面）解决 critical inversion。把 GPU KV Cache 显存动态划分为两个池：**shared pool**（全体可用）与 **reserved pool**（仅关键智能体可用），从而保证非关键工作**无法**驱逐关键智能体的缓存。

分区通过三步反馈循环周期性更新（Algorithm 2）：

- **Step 1 调整总预留池**：监控 GPU 块使用率，高于高水位（0.75）则预留比 $\rho$ 增 0.05，低于低水位（0.40）则减 0.05，clamp 到 $[0.05, 0.30]$。
- **Step 2 选择关键智能体类型**：按 per-agent-type 分数 $S_a$ 选出 top fraction（关键比 0.75）为关键类型。
- **Step 3 分配预留容量**：在关键类型间按「当前显存占用 + 优先级分数」加权分配，兼顾结构重要与内存密集。

**混合优先级指标（两级）**：

per-request 优先级（决定进入下一批的顺序）：

$$P_{req} = \alpha_{struct} \cdot f_{struct} + \alpha_{sync} \cdot f_{sync} + \alpha_{aging} \cdot f_{aging}$$

其中 $f_{struct}$（结构重要性，由节点深度与出入度合成，衡量解锁多少下游工作）、$f_{sync}$（同步压力，在 DAG 汇合点提升滞后分支优先级，防止拖尾）、$f_{aging}$（时间老化，融合图完成比例、排队等待、临近完成的推动力，防饿死、降尾延迟）。

per-agent-type 分数（决定哪类智能体获得预留显存）：

$$S_a = w_1 P_a + w_2 U_a + w_3 H_a + w_4 G_a$$

即结构优先级 $P_a$、运行时紧迫性 $U_a$（被抢占 / 等待次数，权重最大，因其直接反映 critical inversion）、重计算成本 $H_a$（保护重建代价高的缓存）、图上下文 $G_a$。

#### (4) 两个调度器的协调

时间与空间调度器优化 KV Cache 的不同维度却竞争同一显存。TokenCake 用**共享的压力感知协调协议**：每步先拍一份 **Pressure Snapshot**（捕获 GPU/CPU 块可用量、各类型预留容量、等待需求、可 offload 停滞块、待上传债务），两个调度器读同一快照，保证**每次显存移动都对应一个具体收益**——offload 仅当释放块能被有用工作利用时发生，upload 仅当恢复请求不会挤掉更重要的活跃请求时发生。每个调度步固定四阶段执行，避免相互矛盾的内存决策。

### 2.3 相关工作对比

论文把现有服务系统清晰地划成**两条互补但各有盲区**的技术路线，TokenCake 恰好桥接二者：

| 维度 | **TokenCake** | Mooncake / CachedAttention | Parrot / Autellix / Teola |
|------|---------------|----------------------------|---------------------------|
| 定位 | KV-Cache 中心 **且** Agent 感知 | KV-Cache 中心，**agent-agnostic** | Agent 感知，**compute-centric** |
| 管 KV Cache 内存分配 | ✅ 动态分区 + 预留 | ✅ 但不区分关键性 | ❌ 只调度请求顺序 |
| 用函数调用事件 | ✅ 主动（FC Start 触发） | ❌ 被动（内存压力 / LRU / 会话不活跃） | ❌ |
| 区分关键路径 | ✅ 图结构 + 运行时 | ❌ 所有 KV 同等对待 | ✅ 但仅在调度层，不管内存 |
| 能否防 critical inversion | ✅ 内存层解决 | ❌ | ❌ 最优调度也防不住 |

KV Cache offload/prefetch 策略对比（Table 2 精简）：TokenCake 的 offload 是 **proactive（FC Start 触发、Cost-Benefit 决策）**，prefetch 是 **predictive（预测 FC 完成、静态+动态）**；而 Mooncake 是 reactive（缓存池压力 / LRU）、CachedAttention 是 reactive（会话不活跃 / 层粒度）、InferCept 是 reactive（拦截信号 / 最小浪费启发式）。

### 2.4 Tradeoff 分析

- **优势（量化）**：端到端延迟比 vLLM 降低 **>47.06%**；GPU KV Cache 利用率从 69.9–74.1%（vLLM）提升到 **85.8–87.0%**（最多 +16.9 个百分点，32B 模型从 53.5% → 79.6%）。
- **代价**：
  - **侵入性**：需要用户显式把应用建成 DAG 并提供 `predict_time` / 工具阶段元数据——对已有应用有改造成本，且**动态图**（运行时才决定调用哪个下游）只能在请求级支持。
  - **迁移开销**：offload/upload 走 PCIe，单次往返 15.8–79.8 ms；高并发下无用迁移会吃掉 PCIe 带宽并占住显存——这正是必须用机会主义门控（而非 always-offload）的原因。
  - **实现复杂度**：约 9k 行 Python，两个调度器 + 协调协议 + CPU 迁移基础设施。
  - **对预测误差敏感**：见 §3.3 敏感性分析。
- **适用边界**：函数调用**密集**、单次调用**足够长**（能覆盖往返传输）、且存在**明确关键路径**的多智能体应用收益最大；反之，函数调用极短、或所有智能体同等重要的场景，收益会明显缩水。

## 3. 实验

### 3.1 实验设置

- **硬件 / 模型**：Qwen2.5-14B on 1×A100(80GB HBM)；Qwen2.5-32B on 1×H20(96GB)；Qwen2.5-72B on 2×H20（张量并行 TP=2，用于验证多 GPU 预留）。均保留 100 GB CPU 内存作为 offload 目标。
- **应用**：Code-Writer（11 种智能体类型，频繁调用文件 I/O、搜索、外部测试，制造高显存压力）与 Deep-Research（智能体更少但依赖链更深，压测关键路径优化）。
- **负载**：请求源自 ShareGPT + AgentCode 数据集，Poisson 到达，工具端点按 Table 1 的 MCP 延迟部署。
- **Baselines**：vLLM(0.8.6)、vLLM-Prefix（加前缀缓存）、Mooncake(0.3.0-beta，远程 KV Cache)、Parrot（agent-aware 但 compute-centric）。
- **指标**：应用级端到端延迟（均值 + 尾分位）与 GPU KV Cache 利用率。

### 3.2 关键结果

- **端到端延迟（图 2）**：TokenCake 在全部「模型 × 应用 × 数据集」配置下延迟最低。低负载（0.05 QPS）时与 vLLM 相近（争用小）；随到达率上升，vLLM / vLLM-Prefix 因保留停滞智能体的 KV Cache 而饱和显存、被迫缩小批量，延迟陡增。**Qwen2.5-14B Code-Writer D1 @1.0 QPS 时比 vLLM 降低 47.06%**，且优势跨模型规模保持。
- **GPU 显存利用率**：TokenCake 维持 85.8–87.0%，vLLM 仅 69.9–74.1%（最多 +16.9 pp）；32B 模型上从 53.5% → 79.6%。关键差异不在总内存量，而在**利用质量**——TokenCake 把占用块留给活跃、可计算的请求，而 vLLM 的块被停滞缓存占住、阻塞新请求入调度。
- **对比 Mooncake**：0.2 QPS（低压）时二者显存都充足，Mooncake 降延迟 24.8%（697→524 s），TokenCake 进一步到 499 s（仅领先 4.8%）；**0.5 QPS（高争用）时差距拉大**：Mooncake 533 s、TokenCake 384 s（比 Mooncake 降 28.0%、比 baseline 降 37.0%）。差异根源：Mooncake 不用函数调用事件预测空闲、也不按图关键性区分缓存。
- **对比 Parrot**：Code-Writer 上 Parrot 14.3k–18.6k s 而 TokenCake 2.0–2.1k s（差 **6.8–8.9×**）；Deep-Research 上 Parrot 3.5–4.2k s 而 TokenCake 496–646 s（差 6.5–7.1×）。根因是结构性的：Parrot 不管 KV Cache 内存，非关键智能体耗尽 GPU 块、无视调度顺序地驱逐关键缓存。（注：此对比用 Parrot 自家引擎，是**系统级对照**而非受控实验，差距含引擎差异，作者亦坦承。）

### 3.3 消融与敏感性

**组件必须协同（§7.3，Qwen2.5-14B Code-Writer，20 应用，1.0 QPS，0.5 显存利用）**：

| 配置 | 总延迟 | 说明 |
|------|--------|------|
| baseline（vLLM） | 502.2 s | — |
| agent-only（仅空间调度器） | 424.8 s（−15.4%） | 单独调度已显著改善 |
| offload-only（仅时间调度器、无 agent 感知） | 403.1 s | 但触发 **11,339 次** offload（≈全量 TokenCake 的 2 倍 swap），无差别迁移带来 churn |
| **full TokenCake** | **344.6 s** 总 / 313.7 s 均 / 328.3 s P90 | swap 量比 offload-only **减少 51%** |

结论：agent-only 在各负载下**一致优于** offload-only——即无差别 offload 不能替代 agent-aware 调度；只有两者结合、把释放块导向关键智能体，才能以更少迁移拿到最优延迟。

**敏感性（§7.5）**：

- **工具时间预测误差（非单调！）**：零噪声时降延迟 14.8%；噪声 0.25 时**回退到 +8.3%**（误估的上传通过了可行性检查却仍造成有害迁移）；噪声 0.5 时又恢复到 −3.4%（大误差直接让调度窗口不可行，门控正确地阻止了大多数迁移）。**中等预测误差是最难的区间**——硬拒绝能挡住大误差，但边际误差最易「蒙混过关」造成代价。
- **请求选择策略**：`first_fit` 最佳（均值 152.6 s、P95 164.7 s、吞吐最高），因其保持了空间调度器已优化好的队列顺序；`best_fit` 最差（187.0 s，扰乱队列）；`priority_first` 均值低（150.6 s）但尾部差（P95 173.2 s）。默认用 `first_fit`。
- **空间压力水位**：激进阈值（0.05/0.06）频繁 offload（~157 s），高水位（0.08）在该负载下拒绝所有 offload 候选，反而达 107.5 s（降 32%）——**印证选择性原则**：offload 应以「释放块能否被有用工作利用」为条件，而非越多越好。

**Offload 开销与实用性（§7.6，A100 PCIe，Qwen2.5-14B，16 tokens/块、3 MiB/块 bf16）**：

| 缓存上下文 | D2H offload | H2D upload | 重计算 | 重计算 / 往返 |
|-----------|-------------|------------|--------|--------------|
| 4096 tokens（256 块） | 32.0 ms | 31.7 ms | 1815 ms | **28.5×** |
| 全部长度（1024–5120） | — | — | — | **26.8–37.5× 慢** |

即「保留到 CPU 再传回」比「丢弃后重计算」快一个数量级，这是 TokenCake 用 offload 替代重计算的根本依据；但往返 15.8–79.8 ms 的成本在高并发下不可忽视，故用门控规避无用迁移。

## 4. 评估

### 4.1 局限性

- **方法局限**：以**单 GPU** 为主要目标；虽支持多 GPU 预留（per-GPU shared/reserved 池 + 跨设备准入），但**完整的多 GPU 性能评估留待未来工作**。
- **实验局限**：主要用**合成负载**（ShareGPT + AgentCode 合成，Poisson 到达），非真实生产 trace；Parrot 对比跨引擎，非受控实验，其 6–8× 差距含引擎实现差异。
- **假设局限**：① 假设应用图**静态**（为可复现）；现实中存在**动态边**（运行时才决定下游智能体），当前仅在请求级支持，作者建议集成 Hermes 的概率图模型改进预留准确性。② 依赖 per-function-type 历史 EMA 预测——对冷启动、或方差极大的工具（如 AI Generation 方差 10–60 s）预测困难，而敏感性分析恰恰显示中等误差最伤性能。③ 侵入式 API 需应用改造为 DAG 并标注元数据。

### 4.2 启示

- **对领域**：验证了 agentic serving 需要 **workload-aware**——把「函数调用事件」当作显式调度信号、把「DAG 关键性」当作显存分配依据，是通用 agent-agnostic 系统给不出的。
- **对工程实践**：① 调度与内存管理必须**协同设计**（本文最强的经验结论）；② offload 决策要以「释放的显存能否被有用工作利用」为门控，而非无差别 always-offload；③ 保留 KV 到 CPU 再取回，比重计算便宜一个数量级，是长上下文 / 长函数调用场景的通用优化点。
- **可扩展方向**：时间调度器的 offload 目标可从 CPU 扩展到 NVLink 相邻 GPU，形成**分层内存**；空间调度器的预留可跨设备协调，缓解跨 GPU critical inversion；引入动态图的概率模型可提升高变动图结构下的预留精度。

### 4.3 一句话总结

> TokenCake 抓住「函数调用暴露可预测空闲窗口 + DAG 暴露关键性」这一洞察，用**时间调度器**（函数调用期主动 offload 空闲 KV Cache 并预测性回传）与**空间调度器**（按混合优先级为关键路径智能体动态预留显存）协同解决了多智能体 LLM 应用的 KV Cache 时间低利用与空间争用问题，端到端延迟比 vLLM 降 47%+、显存利用率提升最多 16.9 pp；代价是需要应用提供 DAG 与函数调用元数据，以及 PCIe 迁移开销（靠机会主义门控规避）。

## 参考文献

- [TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications](https://arxiv.org/abs/2510.18586)（本文，arXiv 2510.18586v3）
- [vLLM: Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（PagedAttention，主要 baseline）
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)（远程 KV Cache 对比对象）
- [Parrot: Efficient Serving of LLM-based Applications with Semantic Variable](https://arxiv.org/abs/2405.19888)（agent-aware / compute-centric 对比对象）
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)（工具调用标准与延迟特征来源）
