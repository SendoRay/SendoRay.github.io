---
title: "An I/O Characterizing Study of Offloading LLM Models and KV Caches to NVMe SSD"
date: '2026-07-02'
tags:
- LLM
- serving
- SSD-offloading
- KV-cache
- characterization
draft: false
math: true
ShowToc: true
summary: "首次系统刻画 LLM 推理中模型权重与 KV 缓存卸载到 NVMe SSD 的块级 I/O 模式：128 KiB 读为主、SSD 远未饱和、KV cache 读写比达 186×。"
---

## 论文 PDF

{{< pdf src="pdf/2025-cheops-llm.pdf" >}}

## 1. 问题定义

### 1.1 解决了什么问题

本文解决的是 **"LLM 推理中 SSD 卸载的 I/O 特征缺失"** 问题。

具体来说：当 LLM 模型参数和 KV cache 超过 GPU 显存时，一种解决方案是将它们卸载（offload）到 NVMe SSD。但社区缺乏对这一卸载过程的 **I/O 访问模式和性能需求** 的系统性研究——没有人回答过以下问题：

- 卸载时的 I/O 请求大小分布是什么？
- 读写比例如何？
- SSD 带宽是否被充分利用？
- 不同框架（DeepSpeed vs FlexGen）的 I/O 模式有何差异？
- 模型卸载和 KV cache 卸载的 I/O 特征有何不同？

本文通过收集和分析 **块级（block layer）I/O trace**，首次系统刻画了这些问题。

**问题类别**：经验性研究 / 系统特征刻画（Empirical Characterization Study），非算法创新或系统设计。

**输入**：两个支持 SSD 卸载的 LLM 推理框架（DeepSpeed、FlexGen）在推理过程中产生的块级 I/O trace。

**输出**：四条核心发现 + 开源的 trace 数据集和分析脚本。

### 1.2 为什么要解决这个问题

**动机来源**：工程瓶颈 + 学术空白。

**现有方法的不足**：

| 维度 | 现状 | 不足 |
|------|------|------|
| GPU 显存增长 | A100 80GB → H200 141GB | 增长速度远慢于模型增长（Llama3-405B FP8 需 ~405GB） |
| 量化压缩 | FP16 → INT8 → INT4 | 减小了但不足以覆盖 100B+ 模型 |
| CPU 卸载 | vLLM、DeepSpeed 等广泛支持 | CPU 内存有限（通常 128-256GB），且与 GPU 间带宽受限 |
| SSD 卸载 | DeepSpeed、FlexGen 初步支持 | **无人系统研究过 I/O 特征**，优化方向不明 |

论文 Table 1 直观展示了问题严重性：即使 H200（141GB）也无法放下 FP8 的 GPT3-175B 和 Llama3-405B。

**不解决的后果**：社区在"盲调"——不知道 SSD 卸载的 I/O 瓶颈在哪、SSD 性能是否够用、如何针对 LLM 工作负载优化存储设备。SSD 厂商也不知道该为 LLM 推理工作负载做什么优化。

**为什么是现在**：
1. 长上下文 LLM（100K-1M token）使 KV cache 急剧增长，GPU 放不下
2. SSD 卸载框架刚出现（DeepSpeed、FlexGen），但缺乏特征研究
3. SSD 硬件能力（PCIe 4.0/5.0 NVMe 可达 7+ GB/s）已足够快，但不知道是否被充分利用

### 1.3 问题难点

1. **观测层选择**：需要在哪个层次观测 I/O？用户态 API 层？文件系统层？还是块设备层？不同层次看到的 I/O 模式不同（可能被合并、拆分）。本文选择 **块层（block layer）**，因为这是最接近 SSD 硬件的软件层，能反映真实的存储访问模式。

2. **实验环境控制**：真实 NVMe SSD 的性能受 FTL（Flash Translation Layer）、GC（Garbage Collection）、磨损均衡等内部机制影响，会干扰 I/O 特征刻画。需要一种"足够快且可控"的 SSD 模拟环境。

3. **框架异构性**：不同框架的卸载实现差异大（DeepSpeed 只卸载模型权重，FlexGen 同时支持模型和 KV cache 卸载），需要分别刻画。

4. **模型 vs KV cache 的 I/O 差异**：模型权重在推理期间不变（写一次、读多次），KV cache 在自回归生成中持续增长（读写交替）。两者的 I/O 模式可能完全不同，需要分开分析。

## 2. 方法

### 2.1 核心思想

> **用块级 I/O trace 刻画 LLM 推理中 SSD 卸载的真实访问模式，用 NVMeVirt 消除硬件干扰，用 bpftrace 捕获块层请求。**

关键洞察（Key Insight）：**在块设备层而非应用层观测 I/O**。应用层的 I/O 请求经过文件系统、页缓存、I/O 调度器后可能被合并、重排或拆分，只有块层的 trace 才反映 SSD 实际收到的请求。这一选择使得分析结果对 SSD 设计者和系统优化者都有直接价值。

### 2.2 技术细节

#### 实验架构

```
┌──────────────────────────────────────────────┐
│           LLM 推理框架                        │
│    (DeepSpeed / FlexGen)                     │
│    模型卸载 / KV cache 卸载                    │
├──────────────────────────────────────────────┤
│    DeepNVMe I/O 库                           │
│    (POSIX 同步 / libaio 异步)                  │
├──────────────────────────────────────────────┤
│    文件系统 (ext4 / xfs)                      │
├──────────────────────────────────────────────┤
│    Linux 块层 ← bpftrace 探针                  │
│    (block_rq_issue tracepoint)               │
├──────────────────────────────────────────────┤
│    NVMeVirt (DRAM 模拟 NVMe SSD)              │
│    96 GiB, 9.3μs 延迟, 16.9 GiB/s 峰值带宽     │
└──────────────────────────────────────────────┘
```

#### NVMeVirt：用 DRAM 模拟 NVMe SSD

关键设计决策：不使用真实 NVMe SSD，而是用 **NVMeVirt** 框架在 DRAM 中模拟一个 NVMe 设备。

| 参数 | NVMeVirt 模拟 SSD | 真实 Samsung 990 Pro |
|------|-------------------|---------------------|
| 容量 | 96 GiB (FlexGen) / 64 GiB (DeepSpeed) | 可选 |
| 4K 延迟 | 9.3 μs | ~10-20 μs |
| 4K IOPS | 2.6M (16线程) | ~1M |
| 单线程带宽 | 5.3 GiB/s (512K请求) | ~7 GiB/s |
| 多线程带宽 | 16.9 GiB/s (4线程) | ~7 GiB/s |

**为什么用模拟 SSD**：确保 SSD 不是瓶颈，从而观测到框架本身的 I/O 行为而非被硬件限制。真实 SSD 的 FTL、GC 等机制会引入噪声。

#### bpftrace：块级 I/O 采集

使用 `block_rq_issue` 内核 tracepoint，捕获每个 I/O 请求的：
- 类型（read / write）
- 请求大小
- 访问的扇区号

这是 eBPF 技术，开销极低，不影响推理性能。

#### DeepNVMe：张量传输库

DeepSpeed 的 I/O 库，提供两种接口。两者的根本差异在于 **I/O 深度（in-flight 请求数）** 和 **CPU 是否阻塞等待**。

##### POSIX 同步 I/O

```c
fd = open("model.bin", O_RDONLY);
read(fd, buf, size);   // ← 线程阻塞，直到数据到达 buf
// 此时才能使用 buf 中的数据
close(fd);
```

调用 `read()` 后当前线程被挂起，CPU 进入等待态，直到 SSD 把数据搬进内存。期间 CPU 空闲、SSD 内部并行度未被利用。**I/O 深度 = 1**：一个线程同一时刻只有 1 个在途请求。要提高吞吐只能开更多线程，但线程调度本身有开销。

##### libaio 异步 I/O

```c
struct iocb cb;
io_prep_pread(&cb, fd, buf, size, offset);
io_submit(ctx, 1, &cb);       // ← 立即返回，I/O 在后台进行
// ... CPU 可继续提交更多 I/O 或做其他计算 ...
io_getevents(ctx, 1, 1, events, NULL);  // 稍后收割完成结果
```

`io_submit()` 把请求丢进内核块设备队列后立即返回，SSD 在后台搬数据。线程可以继续提交更多请求或做其他工作，等数据真正需要时再调 `io_getevents()` 收割。**I/O 深度 > 1**：一个线程可同时维护多个在途请求，SSD 并行处理它们。

##### 时序对比

```
POSIX（单线程，I/O depth = 1）:
CPU: [提交] [等待......] [处理] [提交] [等待......] [处理]
SSD:         [搬数据]                [搬数据]
     ← 时间被串行化，SSD 大量空闲 →

libaio（单线程，I/O depth = N）:
CPU: [提交1][提交2][提交3][处理0] [提交4][处理1] ...
SSD:    [搬数据1] [搬数据2] [搬数据3] [搬数据4] ...
     ← SSD 持续工作，流水线重叠 →
```

POSIX 是"提交一个、等一个、再提交一个"的串行模式；libaio 是"提交一批、后台并行搬、按需收割"的流水线模式。NVMe SSD 内部有多个并行通道（多 queue + 多 die 交错），POSIX 的单请求根本喂不饱它。

##### 论文实测数据

| 场景 | POSIX 带宽 | libaio 带宽 | 提升 |
|------|-----------|------------|------|
| CPU 读 SSD | ~1.4 GiB/s | 4.1 GiB/s | 2.9× |
| CPU 写 SSD | ~1.2 GiB/s | 4.0 GiB/s | 3.3× |
| GPU 读 SSD | ~0.7 GiB/s | 2.0 GiB/s | 2.8× |
| **GPU 写 SSD** | **~0.5 GiB/s** | **2.8 GiB/s** | **5.5×** |

GPU 写场景差距最大（5.5×），因为 GPU→SSD 要经过 `GPU → CPU 内存 → SSD` 两跳。POSIX 下两跳串行等待；libaio 下可以在等 GPU 数据到达 CPU 内存的同时，把之前已到的数据异步推给 SSD，**两跳重叠**。

但 libaio 也没到极限——fio 基线是 5.3 GiB/s，libaio 最高才 4.1 GiB/s（低 22.6%），说明 DeepNVMe 库的实现本身还有优化空间，可能是拷贝开销或请求提交粒度不够。

#### 实验矩阵

| 维度 | DeepSpeed | FlexGen |
|------|-----------|---------|
| 模型卸载 | OPT-13B (FP16) | OPT-30B (FP16) |
| KV cache 卸载 | ✗ 不支持 | OPT-6.7B |
| 输入 token | 256 | 256 |
| 输出 token | 32 | 256 |
| SSD 容量 | 64 GiB | 96 GiB |
| 硬件 | GCP g2-standard-32, L4 GPU (24GB), 32 vCPU, 128GB RAM | 同左 |

### 2.3 相关工作对比

本文不属于算法创新，而属于 **系统特征刻画**。下表将本文与相关工作按技术路线分类：

| 技术路线 | 代表工作 | 核心思路 | 与本文关系 |
|---------|---------|---------|-----------|
| **SSD 卸载（训练）** | FlashNeuron [17], Behemoth [36], ZeRO-Offload [50] | 训练时卸载中间激活/梯度到 SSD | 本文聚焦推理而非训练 |
| **SSD 卸载（推理-模型）** | DeepSpeed [14], HuggingFace accelerate [4] | 推理时按需从 SSD 加载模型权重层 | 本文直接刻画其 I/O 模式 |
| **SSD 卸载（推理-KV cache）** | FlexGen [51], AttentionStore [29] | 将 KV cache 溢出到 SSD | FlexGen 是本文评估对象；AttentionStore 代码未公开 |
| **稀疏性优化卸载** | LLM-in-a-flash [13], PowerInfer [53], InfiniGen [39] | 利用模型权重/激活稀疏减少 I/O 流量 | 本文用最简策略（全量卸载），未利用稀疏性 |
| **GPU 显存管理** | vLLM PagedAttention [38] | 分页式 KV cache 管理避免显存浪费 | 正交方向，可与 SSD 卸载结合 |
| **Prefill-Decode 分离** | Mooncake [48], Splitwise [47] | 将 prefill 和 decode 分到不同服务器 | 需要迁移 KV cache，与单机 SSD 卸载正交 |
| **KV cache 压缩** | CacheGen [43] | 压缩 KV cache 减少存储/传输 | 可与 SSD 卸载叠加 |
| **在存储设备内计算** | InstInfer [46], OptimStore [35] | SSD 内部直接做注意力计算 | 依赖特殊硬件，本文关注标准 SSD |

**本文在技术路线版图中的位置**：本文不是提出新方法，而是为现有 SSD 卸载方法（DeepSpeed、FlexGen）提供 **第一手的 I/O 特征数据**，为后续优化（稀疏性、压缩、ZNS SSD 设计等）提供基线。

### 2.4 Tradeoff 分析

**本文方法的优势**：
- 块级 trace 反映 SSD 实际收到的请求，比应用层观测更准确
- NVMeVirt 消除硬件干扰，观测到框架纯软件行为
- 开源 trace 数据集，可复现和二次分析

**本文方法的代价**：

| 维度 | 代价 | 影响 |
|------|------|------|
| **模拟 vs 真实** | NVMeVirt 在 DRAM 中模拟，不反映真实 SSD 的 FTL、GC、磨损均衡 | 真实 SSD 上的性能可能更差（GC 引入尾延迟），结论的绝对值可能偏乐观 |
| **GPU 选择** | 仅用 L4 (24GB)，较慢 | 更快的 GPU（A100/H100）decode 更快，对 SSD I/O 压力更大，可能改变 "SSD 未饱和" 的结论 |
| **框架覆盖** | 仅 DeepSpeed + FlexGen | vLLM（主流框架）不支持 SSD 卸载；AttentionStore 代码未公开 |
| **卸载策略** | 最简策略（全量卸载） | 未利用稀疏性，I/O 流量是上界；真实部署中会用更聪明的策略 |
| **模型选择** | 仅 OPT 模型 | 不同模型架构（MoE 等）的 I/O 模式可能不同 |
| **量化精度** | 仅 FP16 | INT8/INT4 会减小 I/O 量，改变 I/O 模式 |
| **单机单卡** | 无多 GPU、多 SSD | 分布式场景的 NUMA、GPU-SSD 亲和性未考虑 |

**适用边界**：结论适用于 **单机单 GPU + 标准 NVMe SSD + 全量卸载** 的场景。超出此范围（多 GPU、稀疏卸载、真实 SSD）时结论可能不成立。

## 3. 实验

### 3.1 实验设置

| 项目 | 配置 |
|------|------|
| 实例 | GCP g2-standard-32 |
| CPU | Intel Xeon 2.2GHz, 32 核 |
| 内存 | 128 GiB |
| GPU | NVIDIA L4, 24 GiB |
| SSD | NVMeVirt 模拟 (96 GiB NVM, 4 核专用) |
| OS | Ubuntu 22.04.1, kernel 6.8.0 |
| 框架 | FlexLLMGen 0.1.7, DeepSpeed 0.16.1 |
| 模型 | OPT-13B (DeepSpeed), OPT-30B (FlexGen), OPT-6.7B (KV cache) |
| 量化 | FP16 |
| 追踪工具 | bpftrace (block_rq_issue tracepoint) |
| 文件系统 | ext4 + xfs（对比测试） |

### 3.2 关键结果

论文围绕四个问题给出四组结果：

#### Finding 1：libaio 显著优于 POSIX（微基准测试）

| 场景 | POSIX 带宽 | libaio 带宽 | libaio 提升 |
|------|-----------|------------|------------|
| CPU 读 SSD | ~1.4 GiB/s | 4.1 GiB/s | 2.9× |
| CPU 写 SSD | ~1.2 GiB/s | 4.0 GiB/s | 3.3× |
| GPU 读 SSD | ~0.7 GiB/s | 2.0 GiB/s | 2.8× |
| GPU 写 SSD | ~0.5 GiB/s | 2.8 GiB/s | 5.5× |

但即使 libaio 的最高带宽（4.1 GiB/s）也比 fio 基线（5.3 GiB/s）低 22.6%，说明 DeepNVMe 库本身还有优化空间。

#### Finding 2：模型卸载 I/O 以 128 KiB 读为主

DeepSpeed 和 FlexGen 在模型卸载时的块层 I/O 特征：

| 特征 | DeepSpeed | FlexGen |
|------|-----------|---------|
| 读写模式 | 启动时写模型 → 之后纯读 | 同左 |
| 主导请求大小 | 128 KiB | 128 KiB |
| 扇区访问模式 | 均匀 | 均匀 |
| 平均读带宽 | 4.9 GiB/s | 2.6 GiB/s |
| 文件系统影响 | ext4 ≈ xfs | ext4 ≈ xfs |

模型卸载的 I/O 模式非常规律：**一次写入（offload 模型到 SSD），之后持续读取**。这是因为模型权重在推理期间不变化。128 KiB 的请求大小由框架决定，用户无法配置。

#### Finding 3：模型卸载未饱和 SSD

| 框架 | 平均读带宽 | NVMeVirt 峰值 | 利用率 |
|------|-----------|-------------|-------|
| DeepSpeed | 4.9 GiB/s | 16.9 GiB/s (4线程) | ~29% |
| FlexGen | 2.6 GiB/s | 16.9 GiB/s | ~15% |
| fio 基线 | 5.3 GiB/s (单线程) | 16.9 GiB/s | ~31% |

**关键结论**：即使在模拟的超快 SSD 上，框架也未能充分利用带宽。瓶颈不在 SSD 硬件，而在 **框架的 I/O 调度策略**（单线程、无预取、无 I/O 合并优化）。

#### Finding 4：KV cache 卸载读写比极度不对称

| 指标 | 值 | 说明 |
|------|-----|------|
| 读带宽 | 2.0 GiB/s | 每个 token 生成时读取已有 KV cache |
| 写带宽 | 11.0 MiB/s | 新 token 的 KV cache 写入 |
| 读写比 | 186.2× | 读远大于写 |
| 主导请求大小 | 128 KiB | 与模型卸载一致 |
| 访问模式 | **非均匀** | 与模型卸载（均匀）不同 |
| 热点访问 | 最高 256 次/扇区 | 256 个输出 token 各访问一次 |

KV cache 的非均匀访问模式是关键差异：**最新生成的 KV cache 被访问最频繁**（每个后续 token 都要读它），而早期的 KV cache 访问频率逐渐降低。这暗示了 **缓存友好的布局** 或 **分热度存储** 的优化机会。

### 3.3 补充分析

本文是特征刻画研究，没有传统意义上的消融实验。但以下对照实验提供了类似信息：

| 对照实验 | 发现 |
|---------|------|
| POSIX vs libaio | libaio 全面优于 POSIX（Finding 1） |
| DeepSpeed vs FlexGen | DeepSpeed 读带宽更高（4.9 vs 2.6 GiB/s），但 I/O 模式相似 |
| ext4 vs xfs | 文件系统对 I/O 模式影响最小 |
| 模型卸载 vs KV cache 卸载 | 模型：均匀读；KV cache：非均匀读写混合 |
| 张量大小 vs 带宽 | 大张量（>1 GiB）能更好利用带宽（Fig. 2） |

## 4. 评估

### 4.1 局限性

**方法的局限**：

1. **NVMeVirt ≠ 真实 SSD**：DRAM 模拟没有 FTL、GC、NAND 特性。真实 SSD 上，128 KiB 顺序读可能触发 GC 而产生尾延迟；写工作负载可能触发 SLC cache 降级。论文承认这一点但未量化影响。

2. **L4 GPU 太慢**：L4 的 decode 速度远低于 A100/H100。论文自己在 Discussion 中承认需要探索更强 GPU——更强 GPU 意味着更快 decode，对 SSD 读带宽要求更高，"SSD 未饱和"的结论可能在 H100 上不成立。

3. **单线程 I/O**：DeepNVMe 使用单线程 I/O，而 NVMeVirt 的峰值带宽需要 4 线程。框架未做多线程 I/O 调度，这既是发现也是局限——真实部署中可能已经用多线程。

**实验的局限**：

4. **框架覆盖不全**：vLLM（最主流推理框架）不支持 SSD 卸载，未被测试。AttentionStore 代码未公开。结论的泛化性受限。

5. **模型覆盖不全**：仅 OPT 系列。MoE 模型（如 Mixtral）的 expert 权重卸载 I/O 模式可能完全不同——稀疏激活意味着按需加载 expert，I/O 更随机。

6. **量化精度单一**：仅 FP16。INT4/INT8 会使 I/O 量减半到 1/4，可能改变请求大小分布。

**假设的局限**：

7. **单机假设**：所有实验在单机单 GPU 上。多 GPU 场景中，SSD-SSD 间的 I/O 竞争、NUMA 效应未考虑。

8. **静态工作负载假设**：实验用固定输入/输出长度。真实 serving 场景中，variable-length 请求、batch 动态变化可能导致 I/O 模式更复杂。

### 4.2 启示

**对领域的贡献**：

1. **第一份块级 I/O trace**：为 SSD 卸载研究提供了可复现的基线数据。开源 trace 数据集是本文最大的价值——后续优化工作可以在此基础上评估改进效果。

2. **128 KiB 是关键 I/O 大小**：这对 SSD 设计有直接指导意义。SSD 的 FTL 可以针对 128 KiB 对齐优化，ZNS（Zoned Namespace）可以按 128 KiB zone 管理数据放置。

3. **SSD 未被饱和**：说明瓶颈在软件而非硬件，优化框架的 I/O 调度（多线程、预取、I/O 合并）比升级 SSD 更紧迫。

4. **KV cache 读写不对称**：186× 的读写比暗示了 SSD 的写寿命不是瓶颈——读远多于写，NAND 的写次数限制（TBW）不会快速耗尽。

**对工程实践的指导**：

- 选择 SSD 卸载框架时，优先选支持 libaio 的（DeepSpeed 的 DeepNVMe）
- 不需要追求最快 SSD——当前框架根本用不满
- KV cache 存储可以用更便宜的读优化型 SSD（而非读写均衡型）
- 文件系统选择（ext4 vs xfs）对 LLM 卸载工作负载影响可忽略

**可扩展的方向**：

- 在真实 NVMe SSD 上重复实验，量化 FTL/GC 影响
- 测试 vLLM + SSD 卸载（需自行实现）
- 结合稀疏性卸载（LLM-in-a-flash 方向）后重新刻画 I/O
- 探索 ZNS SSD 对 LLM 卸载的优化潜力
- 多 GPU + 多 SSD 场景的 I/O 调度

**未来工作方向**（论文自己提出的）：
- 更强 GPU（A100/H100）下的重新评估
- 多 GPU/多 SSD 的 NUMA、GPU-SSD 亲和性
- ZNS 和 FDP 接口对 LLM 工作负载的优化
- 分布式 KV cache 管理与存储的结合

### 4.3 一句话总结

> 这篇论文用块级 I/O trace 系统刻画了 LLM 推理中 SSD 卸载的 I/O 模式，核心发现是 128 KiB 读为主且 SSD 远未饱和，代价是仅覆盖单机单 GPU + 模拟 SSD + 两个框架，结论的泛化性有待更强硬件和更多框架验证。

## 参考文献

- [论文 PDF](pdf/2025-cheops-llm.pdf) — CHEOPS '25 Workshop
- [代码与 Trace 数据集](https://github.com/stonet-research/cheops25-IO-characterization-of-LLM-model-kv-cache-offloading-nvme)
- [DeepSpeed DeepNVMe](https://www.deepspeed.ai/tutorials/deepnvme/)
- [FlexGen (FMInference)](https://github.com/FMInference/FlexLLMGen)
- [NVMeVirt](https://github.com/snu-csl/nvmevirt)
- [vLLM PagedAttention (SOSP 2023)](https://doi.org/10.1145/3600006.3613165)
- [LLM-in-a-flash (ACL 2024)](https://doi.org/10.18653/V1/2024.ACL-LONG.678)
- [Mooncake KVCache-centric Architecture](https://arxiv.org/abs/2407.00079)
- [InfiniGen (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/lee)
- [CachedAttention (USENIX ATC 2024)](https://www.usenix.org/conference/usenix24)
