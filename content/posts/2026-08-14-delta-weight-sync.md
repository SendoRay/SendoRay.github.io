---
title: "Delta Weight Sync：当 RL 权重同步从硬件问题变成软件问题（编译）"
date: '2026-08-14'
tags:
- RL
- LLM
- Infra
- WeightSync

draft: false
math: true
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

> **出处声明**：本文为编译笔记，基于 Changyi Yang 的原文《[Tech] Delta Weight Sync》（[https://changyi.fun/zh/posts/delta-weight-sync/](https://changyi.fun/zh/posts/delta-weight-sync/)）整理改写而成。文中示意图按原图逻辑用 ASCII 重绘，所有技术观点与实现细节的版权归原作者所有。本文只是笔者消化后的重新组织，强烈建议读者移步阅读原文。

**TL;DR**：在 RL post-training 的解耦架构里，训练器（trainer）每一步都要把最新权重同步给推理引擎（rollout engine），这一直被当成一个"带宽问题"——要靠 NCCL、靠 RDMA、靠更粗的管子。但一个朴素的观察改变了整个故事：在典型的 RL 学习率下，把 Adam 更新量 cast 回 BF16 之后，**超过 99% 的权重元素逐字节完全没有变化**。既然如此，只传真正变了的那 1%~3%（即 delta），通信量直接降低约两个数量级；而且接收端的重建是**无损、bit-identical** 的——不是近似，不是压缩后的"差不多"。于是权重同步从"必须上 RDMA 的硬件问题"，变成了"如何高效编码稀疏 delta 的软件问题"。

全文结构鸟瞰：

```text
┌─────────────────────────────────────────────────────────────────┐
│  一、背景：weight sync 为什么是瓶颈                                │
│      解耦架构 / full broadcast / slime 60s→7s 仍不够 / 跨 DC 不可行 │
├─────────────────────────────────────────────────────────────────┤
│  二、关键观察：RL 权重更新天然稀疏                                  │
│      compute-visible sparsity / BF16 舍入阈值 / 1%~3% 实测        │
│      → 只传 (indices, values)，无损 bit-identical                 │
├─────────────────────────────────────────────────────────────────┤
│  三、三条线四个月内独立收敛                                        │
│      学术（PULSE/SparrowRL/Helix）× 工业（Fireworks）              │
│      × 开源（TRL/slime/vLLM），分歧在传输介质/位置编码/diff 基准     │
├─────────────────────────────────────────────────────────────────┤
│  四、slime 的实现详解（全文重心）                                   │
│      4.0 改动位置：发送端在 slime / 接收端在 SGLang                 │
│      4.1 发送端：diff → encode → bucket & flush → snapshot 更新    │
│          三条 CUDA stream + chunk 级 + 段级重叠      【ASCII 图 1】 │
│      4.2 接收端：NaN-masked overwrite                             │
│          checksum / densify / monkeypatch copy_     【ASCII 图 2】 │
│      4.3 侵入性评估   4.4 与 RDMA 正交（mode × transport）          │
├─────────────────────────────────────────────────────────────────┤
│  五、量化如何改变这套故事                                          │
│      全量任意精度都行 / delta 依赖三个条件 /                        │
│      BF16、FP8 block-wise 可行，NVFP4 难                          │
├─────────────────────────────────────────────────────────────────┤
│  六、小结与开放问题                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 一、背景：weight sync 为什么是瓶颈

### 1.1 解耦架构下的每步同步

现代 RL post-training 系统普遍采用解耦（disaggregated）架构：训练侧跑 Megatron 或 FSDP，rollout 侧跑 SGLang 或 vLLM，两边是不同的进程组，常常在不同节点上，甚至可以在不同的数据中心。这种拆分带来了资源利用率和弹性上的好处，但也引入了一个绕不开的动作——**每个训练步结束后，trainer 必须把新权重推给所有 rollout engine**，否则 rollout 采样用的就是旧策略。

最直接的默认做法是全量广播（full broadcast）：trainer 的 rank 0 把整份权重经 NCCL 广播到所有 inference rank。模型越大、engine 副本越多，这一步就越重。

### 1.2 优化到 7 秒之后，天花板还在

slime 团队曾经把 full-sync 的耗时从约 60 秒压到约 7 秒，用的都是经典工程手段：

- **异步 gather**：把参数聚合与其他工作重叠；
- **bucketing**：把约 2000 次 HTTP 请求合并成约 120 次；
- **tensor flatten**：把碎张量拍平成大块连续传输；
- **weight loading 缓存**：省掉接收端重复的格式解析。

但这些优化再漂亮，本质上仍然是**全量传输**——传的字节数一个没少，只是传得更整齐了。一旦离开数据中心内部的高速互联，这条路就走不通了：跨 DC 的商用网络带宽通常只有几百 MB/s，SparseRL-Sync 论文估算过，一个 8B 模型在这种链路上做全量同步要超过 100 秒。每步 100 秒的同步开销，对 RL 训练来说是不可接受的。

换句话说：只要还在传全量，优化的上限就被"模型总字节数 ÷ 链路带宽"死死钉住。想突破，要么换更粗的管子（硬件），要么少传字节（软件）。

---

## 二、关键观察：RL 权重更新天然稀疏

### 2.1 compute-visible sparsity：更新了，但字节没变

PULSE 论文提出了一个概念，叫 **compute-visible sparsity**（计算可见稀疏性）。它的观察是：Adam 优化器每一步确实会给几乎所有参数算出一个非零的更新量，但在典型的 RL 学习率下，这些更新量绝大多数都非常小——小到把 FP32 的 master weight cast 回 BF16 存储格式之后，**落在 BF16 的舍入阈值以下**。

BF16 的格式是 1 位符号 + 8 位指数 + **7 位显式尾数**（连同隐含位共 8 位有效精度），相邻可表示数之间的相对间距约为 $2^{-7}$。当 $|\Delta w| / |w|$ 明显小于这个量级（舍入阈值约 $2^{-8}$）时，cast 之后得到的 BF16 位模式（bit pattern）与更新前完全相同。逐字节去比对，这个参数**根本没变**。

PULSE 的结论是：约 99% 的 per-step 更新在 BF16 表示上是"不可见"的。注意这不是说训练没在进步——FP32 master weight 里的微小变化在持续累积，等累积量跨过舍入阈值，那个元素的 BF16 表示才会真正翻转。稀疏的是"每一步暴露给推理引擎的变化"，而不是学习本身。

### 2.2 多方独立实测：每步只变 1%~3%

这个比例被多个团队在真实训练中独立测得，数字高度一致：

| 来源 | 每步 BF16 逐字节变化比例 |
|---|---|
| SparrowRL | ~1% |
| Fireworks | ~1.98% |
| slime | ~3% |

也就是说，一个几百 GB 的模型，每步真正需要告诉推理引擎的信息只有几个 GB 甚至更少。

### 2.3 稀疏 delta：无损，而且是逐位无损

有了稀疏性，方案就自然浮现：不广播整份权重，只发一份 $(\text{indices}, \text{values})$ 对——哪些位置变了、变成了什么。接收端拿到之后**逐元素覆盖**对应位置即可。

这个方案有两个容易被低估的性质：

1. **无损（bit-identical）**：接收端重建出的权重与 trainer 端逐位相同。这不是有损压缩换带宽，而是把"本来就没变的字节"从传输中剔除。
2. **无浮点累积漂移**：整个过程只有按位覆盖，没有任何算术运算（对比"传更新量、接收端做加法"的 additive 方案——那条路上浮点加法的舍入误差会一步步累积）。所以不需要周期性的全量重同步来"校准"。

至此，问题的性质变了：不再是"怎样把几百 GB 塞进网络"，而是"怎样高效地找出、编码、传输、应用这 1%~3% 的稀疏 delta"。前者是硬件问题，后者是软件问题。

---

## 三、三条线四个月内独立收敛

有意思的是，这个想法并不是某一家的独门发明——在大约四个月的窗口里，学术界、工业界、开源社区三条线**各自独立**地走到了同一个结论上。

### 3.1 学术界

- **PULSE**：提出 compute-visible sparsity 概念，给出 ~99% 不可见更新的分析；
- **SparrowRL**：面向商用网络（无 RDMA）场景的稀疏同步，实测每步 ~1% 变化；
- **SparseRL-Sync / Helix**：报告 >99% 的稀疏度，并给出跨 DC 全量同步不可行的量化估算。

### 3.2 工业界

- **Fireworks**：已在生产环境落地，用于跨 region 的权重同步。一个值得注意的设计差异：Fireworks 的 diff 基准是**相邻 checkpoint**，而不是上一步的权重快照。

### 3.3 开源社区

- **HF TRL**：通过 Hub bucket 中转 delta，让不具备 RDMA 的用户也能跑解耦 RL；
- **slime**（THUDM）：本文第四章的主角，完整的在线 delta sync 实现；
- **vLLM**：issue #40096 正在讨论原生的 sparse in-place weight update（基于 NCCL，目前限定 TP=PP=1）。

### 3.4 殊途同归，分歧何在

三条线的共识是"只传 delta"，分歧集中在三个正交的维度上：

| 维度 | 可选项举例 |
|---|---|
| 传输介质 | NCCL 广播 / 磁盘文件 / Hub bucket / 对象存储 |
| 位置编码方式 | 绝对 int32 索引 / gap-delta / 再叠加通用压缩 |
| diff 基准 | 上一步快照（slime）/ 相邻 checkpoint（Fireworks） |

这三个维度的选择互不锁死，不同场景（同 DC 高带宽、跨 DC 慢链路、异步弹性集群）各有最优组合。

---

## 四、slime 的实现详解

这一章是全文的重心。slime 的实现之所以值得细读，是因为它把"只传 delta"这个一句话的想法，落成了一条考虑了 GPU 拷贝引擎、CUDA stream 重叠、跨框架边界、故障语义的完整流水线。

### 4.0 改动位置：一个"两头都要改"的跨框架特性

先回答一个架构问题：delta sync 的代码应该放在哪？答案是——**两头都要改**。

**发送端在 slime 内**。实现方式是继承 full-sync 的子类：NCCL group 管理、TP/EP 的参数 gather、Megatron→HF 格式转换这些基础设施全部复用，只重写 diff、编码、发送三个环节。这种子类化的代价是：发送端必须常驻一份 **pinned CPU 内存里的全量权重快照**作为 diff 基准。快照的首次建立（initial seed）不便宜——355B 模型约需 50 秒，但这是一次性成本。

**接收端在 SGLang 内**。SGLang 通过 PR #26519 提供了 sparse delta apply 的原生能力，slime 侧只保留一层很薄的 shim（胶水代码）。

这个分工揭示了 delta sync 的一个本质属性：它不是某个训练框架的内部优化，而是一个**跨越训练/推理框架边界的协议**——发送端定义 wire format，接收端必须能解析并正确应用。

### 4.1 发送端流水线

发送端逻辑只在 **PP-source rank** 上执行（流水线并行中持有该层参数的源 rank）。一次同步分四步：

#### 4.1.1 四步流程

**第 1 步：Diff（逐字节比较）**

```python
changed_mask = current.view(int_dtype) != snapshot.view(int_dtype)
```

关键技巧是先 `view` 成整数类型再比较。这样比较的是原始位模式而非浮点语义，因此天然 **dtype 无关**——BF16、FP8 还是 INT4 的字节，比法完全一样；也顺带规避了 NaN != NaN 之类的浮点比较陷阱。

**第 2 步：Encode（三种位置编码）**

变化位置的编码提供三档，按稀疏度和链路特性选择：

| 编码 | 格式 | 开销 |
|---|---|---|
| `indices` | int32 绝对位置 | 4 B/nnz |
| `deltas` | uint16 gap-delta（相邻索引差值） | 2% 稀疏度下约 2 B/nnz |
| `deltas_zstd` | gap-delta 再过一遍 zstd（level 1） | 更小，换少量 CPU |

gap-delta 能用 uint16 的原因很直观：2% 的稀疏度意味着相邻变化位置的平均间距约 50，uint16 的 65535 上限绰绰有余（超限位置有 escape 处理）。

**第 3 步：Bucket & flush（攒桶发送）**

编码结果按 buffer size 累积成桶。一个重要的布局决策：**positions 留在 CPU**（它是字节序列，编码本身在 CPU 做），**values 留在 GPU**（它是张量切片）。flush 时按 transport 分两路：

- **NCCL 路径**：positions 先 H2D 上 GPU，然后与 values 一起 broadcast；
- **disk 路径**：values D2H 下 CPU，交给后台线程写 safetensors 文件。

**第 4 步：Snapshot 更新**

diff 出的新值通过一条 side-stream 异步 D2H 回写到 pinned 快照——它就是**下一步的 diff 基准**。只回写变化的部分，不用重拷全量。

#### 4.1.2 三条 CUDA stream 的重叠

朴素实现里，diff、快照预取、快照回写会串行排队。slime 用三条 stream 把它们叠起来：

- **默认流**：对当前 chunk 做 diff + encode；
- **h2d_stream**：预取下一个 chunk 的旧快照（CPU→GPU）；
- **d2h_stream**：回写当前 chunk 的新值到快照（GPU→CPU）。

三者之间用 CUDA event 保证顺序正确性（比如回写必须等 diff 读完旧值）。现代 GPU 的拷贝引擎（copy engine）支持 H2D 与 D2H 全双工，这个设计恰好把两个方向同时吃满，diff 计算再叠在中间。

重叠还有两个更高的层级：

- **chunk 级**：1-step lookahead——处理第 $i$ 个 chunk 时，第 $i+1$ 个 chunk 的快照已经在路上；
- **段级**：参数被切成 5 段（非 expert 参数 1 段 + expert 参数 4 个 sub-pass）。每段 flush 之后接收端**立刻可以开始 apply**，与发送端对后续段的编码并行——发送和应用形成软流水线。

#### 4.1.3 disk transport 的异步化

磁盘路径由 `AsyncSafetensorsWriter` 后台线程承担：编码、zstd 压缩、写文件、fsync、原子改名（先写临时名再 rename，保证读者永远看不到半截文件），全部不占训练主线程。文件就绪后通过异步 RPC 发布通知；针对跨 DC 场景里"文件落盘 ≠ 远端可见"的持久化延迟，有一个 `_pre_push_hook` 扩展点做等待或校验。

#### 4.1.4 延迟怎么拆

同步耗时被拆成两段计量：

- **delta_encode**：diff 计算 + 编码 + 发送；
- **delta_finalize**：等待接收端 apply 完成的确认。

分开计量的意义在于：前者是发送端自己的活儿，后者反映接收端与链路的健康度，定位瓶颈时一目了然。

#### 4.1.5 发送端流水线全景图

```text
                      发送端流水线（仅 PP-source rank 执行）

 ┌───────────────────────┐
 │  GPU: 当前权重 W_t     │────────────┐
 └───────────────────────┘            │  bytewise diff
                                      │  current.view(int_dtype) !=
 ┌───────────────────────┐            │  snapshot.view(int_dtype)
 │  CPU pinned 全量快照   │────────────┤  （约 1~3% 元素命中）
 │  （h2d_stream 预取     │            ▼
 │   下一 chunk）         │   ┌─────────────────────────────────┐
 └───────────────────────┘   │ Encode: __positions__+__values__ │
        ▲                    │  · indices     int32   4B/nnz   │
        │ d2h_stream 回写     │  · deltas      uint16  ~2B/nnz  │
        │ 变化的新值           │  · deltas_zstd 再压 zstd L1     │
        │ =「下一步的          └───────────────┬─────────────────┘
        │   diff 基准」                        │ 按 buffer size 攒桶
        │                     ┌───────────────┴───────────────┐
        │                     ▼                               ▼
        │        ┌─────────────────────────┐   ┌─────────────────────────┐
        │        │ NCCL broadcast          │   │ disk: safetensors       │
        │        │ positions H2D 后        │   │ values D2H，后台线程     │
        │        │ 与 values 一起广播       │   │ 编码+zstd+fsync+原子改名 │
        │        └─────────────────────────┘   └─────────────────────────┘
        │
        └────────────（side-stream，与下一 chunk 的 diff 重叠）
```

### 4.2 接收端：NaN-masked overwrite

接收端的实现在 SGLang 里（PR #26519），核心设计可以概括为一句话：**wire 上稀疏，apply 时 densify，然后骗过原生 load 路径做掩码覆盖**。

#### 4.2.1 两个 transport 入口

- **NCCL 入口**：先经 RPC 拿到元数据（DeltaSpec）→ 按元数据预分配接收 buffer → 参与 broadcast → 进入 apply；
- **Disk 入口**：线程池并发读文件 → zstd 解压 → 解析 safetensors header → 进入 apply。

两条路最终汇入同一个 `_apply_delta_payload`。

#### 4.2.2 wire layout：索引与数据分离

一个 delta payload 由三部分组成：

- `__positions__`：所有参数的位置编码拼成的 uint8 blob；
- `__values__`：所有参数的变化值拼成的目标 dtype 张量；
- **DeltaSpec 元数据**：走元数据通道（NCCL 是 RPC JSON，disk 是 safetensors 的 `__metadata__` 字段）。

元数据里每个参数对应一条 **DeltaParam**，相当于该参数在两个 blob 里的"地址簿"：

```python
@dataclass
class DeltaParam:
    name: str          # 参数名
    dtype: str         # 值的 dtype
    shape: list[int]   # 参数完整 shape
    pos_start: int     # 在 __positions__ 中的起止
    pos_end: int
    pos_width: int     # 位置编码宽度（区分 indices/deltas）
    val_start: int     # 在 __values__ 中的起止
    val_end: int
```

#### 4.2.3 Checksum：先验伤，再动刀

apply 之前先做完整性校验。发送端对 payload 算 `torch.hash_tensor` 并做 XOR-reduce，结果随 DeltaSpec 走元数据通道下发；接收端用同样的方式重算比对，不一致直接 raise：

```python
local = xor_reduce(torch.hash_tensor(positions_blob),
                   torch.hash_tensor(values_tensor))
if local != spec.checksum:
    raise RuntimeError("delta payload checksum mismatch")
```

选 XOR-reduce 的好处是与分块顺序无关，可以在 GPU 上流式计算。

#### 4.2.4 单参数解码：4 步 densify

对每个 DeltaParam，解码在 GPU 上分 4 步：

1. 分配一个该参数**完整 shape** 的 buffer，用 NaN 填满（NaN 是"此处没变"的哨兵值）；
2. 按 `pos_start:pos_end` / `val_start:val_end` 从两个 blob 里切出自己的 slice；
3. 向量化位运算 unpack 位置字节 → 若是 gap-delta，`cumsum` 还原绝对索引；
4. `index_copy_` 把 values 散射写入 NaN buffer 的对应位置。

注意这里的取舍：**wire 上是稀疏的，但 apply 不是**——解码后每个参数都膨胀回全尺寸张量（只是绝大部分位置是 NaN）。牺牲一点临时显存，换来的是下一步可以直接走引擎原生的加载路径。

#### 4.2.5 apply：复用原生 load_weights

解码出的 (name, NaN 张量) 按 512 MB 的 chunk 攒成列表，然后调用**原生的** `model.load_weights(chunk)`，外面套一层 `_delta_apply_context`：

```python
with _delta_apply_context(model):
    for chunk in iter_chunks(decoded_params, max_bytes=512 << 20):
        model.load_weights(chunk)   # 引擎原生入口，一行没改
```

#### 4.2.6 `_delta_apply_context`：两处 monkeypatch 撬动整个引擎

为什么能直接用原生 `load_weights`？因为 `_delta_apply_context` 这个 contextmanager 在其作用域内临时替换了两个 Tensor 方法：

- **`Tensor.copy_`**：如果拷贝目标是模型权重，就把"整块覆盖"改成 **NaN 掩码覆盖**：

  ```python
  def patched_copy_(dst, src):
      if is_model_weight(dst):          # 见 4.2.7
          mask = ~torch.isnan(src)
          dst[mask] = src[mask]          # 只覆盖真正变化的位置
      else:
          orig_copy_(dst, src)
  ```

- **`Tensor.fill_`**：某些 load 路径会先把权重 fill 成初值再拷贝——对模型权重 fill NaN 哨兵值时改为 **no-op**，防止把没变的位置抹掉。

这两处补丁的杠杆效应极大：TP/EP/PP 的分片逻辑、量化格式转换、各模型的权重名映射……`load_weights` 里所有既有逻辑**原封不动全部复用**，引擎侧一行核心代码不用改。

#### 4.2.7 `_param_storage_index`：O(log n) 判断"这是不是模型权重"

`patched_copy_` 需要快速判断目标张量是否属于模型权重。做法是提前给所有 `named_parameters` 和 buffers 建一张地址区间表：

```python
intervals = sorted(
    (p.data_ptr(), p.data_ptr() + p.nbytes) for p in all_params_and_buffers
)

def is_model_weight(t):
    i = bisect.bisect_right(starts, t.data_ptr()) - 1
    return i >= 0 and t.data_ptr() + t.nbytes <= intervals[i][1]
```

用 `data_ptr() + nbytes` 做区间匹配的妙处：权重的 **view / slice 会自动命中**（它们落在原张量的地址区间内），而引擎内部的 scratch buffer **自动排除**（地址不在任何权重区间里）——不需要任何显式注册。

#### 4.2.8 无损性与已知缺陷

全程只有按位覆盖、没有算术运算，所以结果 bit-identical、无漂移、**不需要周期性 full re-sync**。

但有一个已知的故障语义缺陷：如果 apply 中途失败，发送端的快照已经推进到 $W_{t}$，而接收端还停在 $W_{t-1}$——下一步的 delta 是基于 $W_{t}$ 算的，接收端应用后会得到错误权重，且**没有任何报错**（silent drift）。这个问题记录在 slime issue #2104，修复见 PR #2119。作为对照，verl 的策略更保守：apply 失败就强制一次 full re-sync 兜底。

#### 4.2.9 接收端 apply 流程全景图

```text
                      接收端 apply 流程（SGLang 内）

  NCCL 入口                          Disk 入口
  RPC 拿 DeltaSpec → 预分配          线程池并发读文件 → zstd 解压
  buffer → broadcast                 → parse safetensors header
        └──────────────┬──────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ 稀疏 payload 上 GPU            │
        │ __positions__ + __values__    │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ checksum 校验                  │
        │ torch.hash_tensor XOR-reduce  │──✗ 不一致 → raise
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ 逐参数 decode / densify        │
        │ NaN buffer → unpack 位置       │
        │ → cumsum → index_copy_        │
        │ （wire 稀疏、apply 不稀疏）      │
        └───────────────┬───────────────┘
                        ▼
   ┌─ ─ ─ _delta_apply_context ─ ─ ─ ─ ─ ─ ─ ─ ┐
   │    ┌───────────────────────────────┐
   │    │ 原生 model.load_weights(chunk)│      │
   │    │ （512MB 一批，逻辑零修改）       │
   │    └───────────────┬───────────────┘      │
   │                    ▼
   │    ┌───────────────────────────────┐      │
   │    │ monkeypatch:                  │
   │    │  copy_ → NaN-masked 覆盖       │      │
   │    │  dst[~isnan(src)]=src[...]    │
   │    │  fill_(NaN) → no-op           │      │
   │    └───────────────┬───────────────┘      │
   └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                        ▼
        ┌───────────────────────────────┐
        │ GPU 权重就地更新（bit-identical）│
        └───────────────────────────────┘
```

### 4.3 侵入性评估

把账算清楚，delta sync 在 slime 里的成本结构是：

- **训练侧侵入小**：纯子类化，不碰 full-sync 的既有代码路径；
- **资源开销**：host RAM 里常驻一份完整模型的 pinned 快照，外加一次性的初始 seed 时间（355B 约 50 秒）；
- **部署耦合**：需要配套包含 PR #26519 的 SGLang build；
- **模式限制**：delta 与 colocate 模式互斥——colocate 下训练与推理同卡，权重经 CUDA IPC 只传一个 64 字节的 handle，本来就没有字节可省，delta 无用武之地。

### 4.4 与 RDMA 正交：mode × transport

一个常见误解是"delta sync 是 RDMA 的替代品"。实际上两者是**正交**的两个维度——mode（full / delta）× transport（nccl / disk），四种组合各有场景：

| | transport = NCCL | transport = disk |
|---|---|---|
| **mode = full** | 传统默认路径 | checkpoint 中转 |
| **mode = delta** | RDMA 高带宽 × 只传 1~3% 字节，同 DC 极致速度 | 无 RDMA 的跨 DC 场景 |

delta + nccl 是"两个数量级带宽 × 两个数量级减量"的乘法；delta + disk 则让根本没有 RDMA 的链路第一次变得可行。delta 不取代 RDMA，它取代的是"必须有 RDMA"这个前提。

---

## 五、量化如何改变这套故事

以上讨论默认权重是 BF16。一旦推理端上了量化（quantization），delta sync 的成立条件就需要重新审视。

### 5.1 量化发生在哪一端

先区分两种部署形态：

- **engine 端量化**：trainer 发送 canonical BF16 权重，engine 收到后本地做量化和 swizzle（内存重排）；
- **trainer 端量化**：trainer 直接发送量化后的表示。

发生的位置不同，delta 面对的"表示"就不同，难度也完全不同。

### 5.2 全量同步：任意精度都能做

先说结论的另一半：**full sync 在任何精度下都没有问题**。全量就是整份覆盖——不需要旧状态、不依赖稀疏性、不要求表示可逆。engine 收到 canonical 权重后自己量化自己 swizzle；集群弹性伸缩（新 engine 加入）也天然成立，因为不存在"必须先有一致基准"的前提。

### 5.3 delta 同步：三个隐含条件被量化逐一挑战

delta sync 能工作，其实暗含三个条件：**变化稀疏**、**表示可重建**、**位置可寻址**。量化会破坏它们：

**条件一：稀疏性 vs 共享 scale。** 量化格式普遍带 per-block 的共享 scale。一个块里只要有一个元素的幅值变化足以改变 scale，整个块的量化码字全部重算——一处小改动放大成一片字节变化。FP8 的 per-block scale 影响还有限；但 NVFP4 是 4-bit 的粗档位 + block scale 本身也被量化 + 还有一层 global scale，小扰动经这三层放大后会引发**大面积重量化**，稀疏性大幅退化。

**条件二 & 三：可重建性/可寻址性 vs "canonical ≠ 运行态"。** delta 的 (index, value) 要有意义，前提是发送端和接收端对"第 i 个元素在哪、长什么样"有一致认知。NVFP4 的运行态权重是 4-bit 交错打包 + swizzle 重排 + global scale 折叠之后的产物，而且这个布局**随并行度配置变化**——它是个移动靶，canonical 表示里的一个位置在运行态里没有稳定的字节地址。相比之下，FP8 block-wise 没有这些问题：8-bit 天然按字节可寻址、无 swizzle、无 global scale。

**结论**：delta sync 只在"**表示稳定 + canonical ≈ 运行态**"的精度上成立。BF16 和 FP8 block-wise 满足；NVFP4 目前很难。

### 5.4 各框架现状

- **slime**：在线支持 BF16 / FP8 / INT4 的 delta（disk 路径，byte-level 的 xor/overwrite）；
- **miles**（企业 fork）：额外支持 MXFP8；
- **NVFP4**：量化器本身已经存在，但在线 delta 链路尚未接入——与上面的分析一致，这不是没写代码，而是问题本身更硬。

---

## 六、小结与开放问题

回头看，整件事的起点只是一个足够朴素的观察：*在 RL 的学习率下，BF16 权重每步逐字节只变 1%~3%*。但它改变的是 RL 训练的**成本结构**——权重同步不再要求 RDMA 级别的互联，普通商用网络、甚至跨 region 的部署都变得可行。解耦式 RL 的地理边界被显著放宽了。

三条独立收敛的路线（学术、工业、开源）也从侧面说明：这不是某个 trick，而是 RL post-training 系统的一块基础设施拼图。

仍然开放的问题：

1. **有损 delta 压缩**：在无损 delta 之上再做有损压缩（例如对 values 降精度），带宽还能再降多少？RL 训练对这种误差的容忍边界在哪里？
2. **additive 方案的漂移边界**：传更新量、接收端做加法的方案会累积浮点舍入误差，多少步之内可以不做 full re-sync？这个边界能否被理论刻画？
3. **MoE 的稀疏度分布不均**：expert 参数的更新稀疏度并不均匀（热 expert 变得多、冷 expert 几乎不变），编码与调度策略是否应该 per-expert 自适应？

---

## 参考资料

- 原文：Changyi Yang, *[Tech] Delta Weight Sync*，[https://changyi.fun/zh/posts/delta-weight-sync/](https://changyi.fun/zh/posts/delta-weight-sync/)
- 论文：*PULSE*（compute-visible sparsity）；*SparrowRL*（商用网络稀疏同步）；*SparseRL-Sync / Helix*（>99% 稀疏度与跨 DC 估算）
- Fireworks 工程博客：跨 region delta 权重同步的生产实践
- HF TRL 文档：经 Hub bucket 的解耦 RL 权重流转
- THUDM/slime 仓库：[https://github.com/THUDM/slime](https://github.com/THUDM/slime)，及 issue #2104（apply 失败的 silent drift）、PR #2119（修复）
- SGLang PR #26519：sparse delta apply（NaN-masked overwrite）
- vLLM issue #40096：原生 sparse in-place weight update 讨论
