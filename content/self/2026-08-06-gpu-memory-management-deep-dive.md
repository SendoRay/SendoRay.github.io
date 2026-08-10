---
title: "GPU训练中的内存管理详解"
date: '2026-08-06'
tags:
- LLM
- GPU
- Training

draft: false
math: true
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

如果统计一下大模型训练工程师最常见的报错，冠军毫无悬念：

```text
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate
2.50 GiB (GPU 0; 79.35 GiB total capacity; 68.42 GiB already allocated;
1.83 GiB free; 74.51 GiB reserved in total by PyTorch)
```

这条报错里的每一个数字——`allocated`、`free`、`reserved`——背后都是一套精密的内存管理机制。为什么 `total` 减 `allocated` 明明还剩 10GB，却只有 1.83GB `free`？为什么 batch size 从 4 降到 2 还是 OOM？为什么训练跑了三天之后显存"莫名"涨了 5GB？

**本文的主线只有一句话：显存里到底放了什么、怎么省、怎么查泄漏。** 全文结构如下：

```text
┌─────────────────────────────────────────────────────────────────┐
│                  本文地图（GPU 训练内存管理）                     │
│                                                                 │
│  一、内存类型全景          寄存器 → HBM → RAM → NVMe             │
│      │                    （容量/带宽/延迟，pinned memory 详解） │
│      ▼                                                          │
│  二、显存构成剖析          每个 rank 的 80GB 里到底放了什么       │
│      │                    （五类静态数据 + 激活值 + 隐性占用）    │
│      ▼                                                          │
│  三、分配器机制            PyTorch caching allocator 与碎片      │
│      │                                          （下半部分）    │
│      ▼                                                          │
│  四、优化技术              checkpointing / offload / 混合精度    │
│      │                                          （下半部分）    │
│      ▼                                                          │
│  五、泄漏排查              工具链与实战 debug     （下半部分）    │
└─────────────────────────────────────────────────────────────────┘
```

**与已有文章的关系**：[LLM训练全解](/self/2026-06-09-training/) 讲的是训练机制本身——前向、反向、优化器为什么需要这些数据；本文则换一个视角，**从内存的角度深挖**：这些数据在物理上放在哪、怎么布局、生命周期如何、如何被分配器管理。两篇互为表里，建议先读那篇再读本文。

全文以一个贯穿始终的算例锚定所有数字：**7B 模型（$\Theta = 7 \times 10^9$）、8×A100 80GB、bf16 混合精度 + Adam、ZeRO-2 风格的 DistributedOptimizer**。

---

## 一、内存类型全景：从 GPU 寄存器到 NVMe

在讨论"怎么省显存"之前，必须先建立完整的存储层次感。训练系统里的数据不只住在 HBM 里——它们在寄存器、缓存、显存、内存、SSD 之间不断流动，而**每往下一层，容量涨一个数量级，带宽掉一个数量级**。

### 1.1 存储层次金字塔

先看直觉：越靠近计算单元（CUDA Core / Tensor Core）的存储越快、越小、越贵；越远的越慢、越大、越便宜。

```text
                          ┌───────────┐
                          │ Registers │  ~256KB/SM, ~1 cycle
                          ├───────────┤
                        ┌─┤ SMEM / L1 ├─┐  192KB/SM, ~30 cycles
                        │ └───────────┘ │
                      ┌─┴───────────────┴─┐
                      │      L2 Cache     │  40MB, ~200 cycles
                    ┌─┴───────────────────┴─┐
                    │      HBM2e (显存)     │  80GB, ~2TB/s
                  ┌─┴───────────────────────┴─┐
                  │     系统 RAM (DDR4/5)     │  0.5~2TB, ~200GB/s
                ┌─┴───────────────────────────┴─┐
                │         NVMe SSD              │  数TB, ~7GB/s
                └───────────────────────────────┘
                  ▲ 越往上：越快、越小、越贵
                  ▼ 越往下：越慢、越大、越便宜
```

用具体数字把这张金字塔量化（以 A100 80GB SXM 为参照）：

| 层级 | 典型容量 | 带宽数量级 | 延迟数量级 | 管理者 |
|---|---|---|---|---|
| Registers | 256KB / SM | ~数十 TB/s（聚合） | ~1 cycle | 编译器 |
| Shared Memory / L1 | 192KB / SM | ~20 TB/s（聚合） | ~30 cycles | kernel 代码显式管理 |
| L2 Cache | 40MB | ~7 TB/s | ~200 cycles | 硬件自动 |
| HBM2e 显存 | 80GB | ~2 TB/s | ~500 cycles | CUDA / PyTorch 分配器 |
| 系统 RAM（DDR4-3200 8通道） | 0.5~2TB | ~200 GB/s（CPU 侧） | ~100 ns | OS 虚拟内存 |
| GPU↔RAM（PCIe 4.0 x16） | — | 32 GB/s 理论（实测 ~25） | ~μs 级 | DMA 引擎 |
| NVMe SSD（PCIe 4.0 x4） | 数 TB | ~7 GB/s | ~100 μs | 文件系统 |

三个值得反复咀嚼的数字关系：

1. **HBM 带宽（2TB/s）是 PCIe（25GB/s）的 80 倍**。这就是为什么"把数据挪出显存"（offload）的代价如此昂贵——一旦数据要跨 PCIe 回来，速度直接掉两个数量级。这也是 [KV Cache 是算还是传？](/self/2026-06-10-kvcache-compute-vs-transfer/) 中"算 vs 传"权衡的物理根源。
2. **NVMe（7GB/s）又比 PCIe 传输慢 3~4 倍**。offload 到 SSD 只适合冷数据（checkpoint、极少访问的优化器状态）。
3. **寄存器到 HBM 之间差了约 500 倍延迟**。这是 kernel 优化（tiling、fusion）的战场，但不是本文重点——本文聚焦 HBM 及以下的"容量管理"问题。

**认知误区**："PCIe 4.0 x16 有 32GB/s，传 1MB 只要 30μs"。错。带宽是**大块传输的渐近值**，小传输由固定延迟主导：一次 H2D 拷贝的 launch + DMA 建立开销约 10~20μs，传 1MB 实测往往只有几 GB/s 的等效带宽。这就是为什么高效的数据搬运总是**攒大块再传**（flat buffer、bucket 机制都源于此），几百次 4KB 的小拷贝比一次 1GB 的大拷贝慢几个数量级。

### 1.2 GPU 显存（HBM）：带宽为什么高，容量为什么小

**物理直觉**：HBM（High Bandwidth Memory）是把多层 DRAM die 垂直堆叠起来，通过硅中介层（interposer）与 GPU die 并排封装。A100 的 HBM2e 用 5 个堆栈，总位宽高达 **5120 bit**——对比普通显卡 GDDR6 的 384 bit，宽了 13 倍。带宽 = 位宽 × 频率，所以 HBM 靠"宽"而不是"快"取胜：

$$
BW_{HBM} \approx 5120 \ \text{bit} \times 3.2 \ \text{Gbps} / 8 \approx 2 \ \text{TB/s}
$$

但堆叠封装同时决定了它的两个"贵"：

- **容量贵**：堆叠层数受散热和良率限制（当前单栈 16~24GB），无法像内存条一样随意插满，所以单卡 80GB 在 2026 年仍是主流上限量级；
- **成本贵**：硅中介层 + 3D 封装的制造成本远高于 GDDR，HBM 占据了整卡成本的相当比例。

三代训练卡的 HBM 演进，感受一下"带宽涨得快、容量涨得慢"：

| 显卡 | HBM 代际 | 容量 | 带宽 | 带宽/容量比 |
|---|---|---|---|---|
| V100 | HBM2 | 32 GB | 0.9 TB/s | 28 /s |
| A100 | HBM2e | 80 GB | 2.0 TB/s | 25 /s |
| H100 SXM | HBM3 | 80 GB | 3.35 TB/s | 42 /s |
| H200 | HBM3e | 141 GB | 4.8 TB/s | 34 /s |

"带宽/容量比"约 25~40/s 意味着：**把整卡显存完整读一遍只要 25~40ms**。这个数字是理解 decode 阶段 memory-bound 本质的钥匙（见 [KV Cache 是算还是传？](/self/2026-06-10-kvcache-compute-vs-transfer/)），也说明 HBM 的稀缺资源从来不是带宽，而是**容量**——这正是本文全部优化技术要解决的问题。

**训练时 HBM 里放什么**（此处只列清单，第二章逐一细讲）：

| 类别 | 性质 |
|---|---|
| 模型参数（bf16 flat buffer + 冻结参数） | 静态，常驻 |
| 梯度缓冲区 | 静态，常驻（内容每 step 覆盖） |
| 优化器状态（fp32 主权重 + Adam m/v） | 静态，常驻 |
| 激活值 | 动态，随 forward 增长、backward 释放 |
| 临时 workspace（cuBLAS/cuDNN/NCCL） | 动态，kernel 级生命周期 |
| CUDA context + 分配器缓存 | 隐性，常被忽略 |

### 1.3 系统内存（RAM）：训练系统的"二级仓库"

RAM 在训练中不是主角，但承担四个关键配角：

1. **DataLoader 预取与增广**：原始数据从磁盘读入 RAM，在 CPU 上完成 tokenize/增广，攒成 batch 后再传给 GPU。多进程 worker（`num_workers > 0`）的预取队列全部驻留 RAM。
2. **优化器状态 offload 的目的地**：ZeRO-Offload / CPU Adam 把 fp32 主权重和 Adam 状态挪到 RAM，用 PCIe 带宽换 HBM 容量——详见 [GPU 显存 Offload 技术](/self/2026-05-28-gpu-memory-offload-techniques/)。
3. **Checkpoint 落盘的中转站**：GPU → RAM → NVMe 两跳，异步 checkpoint 框架会在 RAM 里维护一份 staging 副本，让 GPU 尽快回去训练。
4. **Pinned staging buffer**：所有高效的 CPU↔GPU 传输都要经过锁页内存——这是下一小节的主角。

注意 RAM 有两种状态：**pageable（可分页，默认）** 和 **pinned（锁页）**。这个区别对训练性能的影响远超大多数人的预期。

### 1.4 锁页内存（Pinned Memory）：CPU↔GPU 传输的第一课

#### 分页与页交换：OS 的"腾挪自由"

现代操作系统用**虚拟内存**管理 RAM：物理内存被切成 4KB 的页（page），虚拟地址通过页表映射到物理页。关键在于，OS 保留两项"腾挪自由"：

- **换页（swap out）**：内存紧张时，把不活跃的页写到磁盘，物理页挪作他用；
- **页迁移（migration）**：出于 NUMA 优化或内存压缩，把页搬到另一个物理地址。

对 CPU 程序这完全透明——下次访问触发缺页中断，OS 把页搬回来即可。**但 GPU 的 DMA 引擎不吃这一套**：DMA 是按**物理地址**直接搬运数据的硬件，它绕过 CPU、不走页表。如果传输进行到一半，OS 把那块内存换页或迁移了，DMA 就会读到错误的物理页——数据损坏。

所以 CUDA 的规则是：**DMA 只能从锁页内存（pinned / page-locked memory）读写**。锁页 = 告诉 OS "这些页钉死在当前物理地址，不许换出、不许迁移"。

#### pageable 传输的隐藏拷贝：两跳 vs 一跳

那用普通（pageable）tensor 调 `.to('cuda')` 为什么也能工作？因为 CUDA runtime 在背后**偷偷做了一次额外拷贝**：先把数据从 pageable 内存复制到一块内部维护的临时 pinned staging buffer，再从 staging buffer DMA 到 GPU。

```text
路径 A：pageable 传输（默认，两跳）
┌──────────────┐  CPU memcpy   ┌───────────────┐   DMA over PCIe
│ pageable RAM │ ────────────▶ │ pinned staging│ ────────────────▶ GPU
│ (用户 tensor)│  ① 隐藏拷贝    │ buffer (CUDA  │  ② 真正的传输
└──────────────┘  占用 CPU     │ runtime 内部) │  同步、分块进行
                               └───────────────┘
                  实测带宽 ~6-12 GB/s（受 memcpy 拖累）

路径 B：pinned 传输（一跳直达）
┌──────────────┐                    DMA over PCIe
│  pinned RAM  │ ─────────────────────────────────▶ GPU
│ (锁页 tensor)│   DMA 直接读源地址，无需中转
└──────────────┘   可完全异步（配合 CUDA stream）
                  实测带宽 ~25 GB/s（打满 PCIe 4.0 x16）
```

两条路径的差距是**实测 2~4 倍带宽**，而且路径 A 的隐藏拷贝占用 CPU 核、且强制同步（CUDA 无法保证 pageable 源在异步期间不被动）。

| 维度 | pageable 传输 | pinned 传输 |
|---|---|---|
| 跳数 | 2（memcpy + DMA） | 1（DMA 直达） |
| 实测带宽（PCIe 4.0 x16） | ~6-12 GB/s | ~25 GB/s |
| 可否真正异步 | 否（隐式同步） | 是（配合 non_blocking） |
| CPU 占用 | 高（memcpy 烧核） | 几乎为零 |

不用相信书上的数字，30 秒实测自己机器的两条路径：

```python
import torch, time

def h2d_bw(t, n=20):
    torch.cuda.synchronize(); tic = time.perf_counter()
    for _ in range(n):
        t.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    return n * t.nbytes / (time.perf_counter() - tic) / 2**30

x = torch.empty(1024, 1024, 256)          # 1 GiB, pageable
print(f"pageable: {h2d_bw(x):.1f} GiB/s")  # 典型：~6-12
print(f"pinned:   {h2d_bw(x.pin_memory()):.1f} GiB/s")  # 典型：~25
```

#### PyTorch 三件套

```python
# ① DataLoader 层面：worker 攒好的 batch 直接放进 pinned buffer
loader = DataLoader(dataset, batch_size=..., num_workers=4,
                    pin_memory=True)

# ② 手动把某个 tensor 锁页（返回新 tensor）
t_pinned = t.pin_memory()

# ③ 异步拷贝：只有源是 pinned 时才真正异步
batch = batch.to('cuda', non_blocking=True)
```

三者配合的完整流水线：DataLoader worker 填 pinned buffer → 主进程发起 `non_blocking=True` 异步 H2D → GPU 在计算上一个 batch 的同时，下一个 batch 已经在 PCIe 上传输——**传输被计算完全掩盖**：

```text
时间 ─────────────────────────────────────────────────────▶
CPU worker:  ┌─预处理 batch2─┐┌─预处理 batch3─┐
             │ 写入 pinned   ││ 写入 pinned   │
             └───────┬───────┘└───────┬───────┘
PCIe (DMA):     ┌────▼─────┐     ┌────▼─────┐
             ...│ H2D b2   │     │ H2D b3   │...  ← 异步拷贝流
                └──────────┘     └──────────┘
GPU 计算流:  ┌──────────────────┐┌──────────────────┐
             │ fwd+bwd batch1   ││ fwd+bwd batch2   │
             └──────────────────┘└──────────────────┘
             ▲ 三级流水线并行：CPU 预处理 / PCIe 传输 / GPU 计算
               互相掩盖，GPU 永远不等数据
```

**认知误区**：`non_blocking=True` 不等于异步。如果源 tensor 是 pageable 的，CUDA runtime 仍然要走"隐藏拷贝"路径，`non_blocking=True` 会被**静默降级为同步拷贝**，不报错、不警告。很多训练代码写了 `non_blocking=True` 却忘了 `pin_memory=True`，异步流水线实际从未生效——profile 一下 `cudaMemcpyAsync` 是否真的和 kernel 重叠即可现出原形。

#### 代价与陷阱

锁页不是免费午餐：

1. **不可换页 = 挤占 OS 的机动空间**。锁页内存对 OS 而言是"死"的——不能 swap、不能整理。在 512GB RAM 的机器上锁 50GB 没问题；锁 300GB 就可能让系统卡顿、其他进程被 OOM killer 收割。经验上限：锁页总量不超过物理内存的 50%~60%。
2. **分配慢**。`cudaHostAlloc` 要逐页锁定并登记，分配 1GB pinned 内存耗时可达数百 ms（普通 malloc 是 μs 级）。**正确姿势是启动时分配一次、全程复用**，绝不能在训练循环里反复 `pin_memory()`：

```python
# 错误：每个 step 都现分现锁，每次百 ms 级开销
for batch in loader:
    staging = batch.pin_memory()          # ❌ 热路径上分配
    gpu_batch = staging.to('cuda', non_blocking=True)

# 正确：启动时预分配固定 staging buffer，循环内只做 copy_
staging = torch.empty(max_batch_shape, pin_memory=True)  # ✅ 一次性
for batch in loader:
    staging[:batch.shape[0]].copy_(batch)
    gpu_batch = staging[:batch.shape[0]].to('cuda', non_blocking=True)
```

3. **多进程共享受限**。pinned 内存的注册是进程级的，fork 出来的 DataLoader worker 需要各自处理，这也是 `pin_memory=True` 由专门的 pin_memory 线程统一负责的原因。

### 1.5 补充类型速览

除了上面三大主力，还有几种"特种内存"，各给一句适用场景：

| 类型 | API / 机制 | 一句话原理 | 适用场景 |
|---|---|---|---|
| Unified / Managed Memory | `cudaMallocManaged` | CPU/GPU 共享虚拟地址，按需缺页迁移，可超额订阅（分配量 > 显存） | 原型开发、访问稀疏的超大数据结构；性能不可控，训练慎用 |
| Zero-copy | host pinned + `cudaHostGetDevicePointer` | GPU kernel 直接经 PCIe 读 host 内存，不搬进 HBM | 只读一次的小数据（如 lookup 表），读多了 PCIe 立刻成瓶颈 |
| GPUDirect RDMA | 网卡直读 GPU 显存 | 跨机传输跳过 CPU 和 RAM 中转，GPU→NIC→GPU | 多机 NCCL 通信、分布式 KV Cache 传输的标配 |
| GPUDirect Storage | NVMe 直读/写 GPU 显存 | SSD↔GPU 直达，跳过 RAM bounce buffer | 大 checkpoint 快速加载、数据集直灌 |

### 1.6 本章小结：一张表看全

| 内存类型 | 容量 | 带宽 | 训练中的角色 |
|---|---|---|---|
| HBM（显存） | 80GB / 卡 | ~2 TB/s | 主战场：参数、梯度、优化器状态、激活值 |
| 系统 RAM（pageable） | 0.5~2 TB | ~200 GB/s（CPU 侧） | 数据预处理、checkpoint 中转 |
| 系统 RAM（pinned） | 手动分配，建议 <60% | PCIe ~25 GB/s（到 GPU） | H2D/D2H 高速通道、offload 目的地 |
| NVMe SSD | 数 TB | ~7 GB/s | checkpoint 落盘、冷数据 offload |
| Unified Memory | 虚拟无限 | 缺页迁移，不稳定 | 原型验证，生产训练罕用 |

记住本章最重要的一个对比：**HBM : PCIe : NVMe ≈ 2000 : 25 : 7（GB/s）**。后面所有优化技术，本质都是在这三级带宽悬崖之间做取舍。

---

## 二、训练时每个 Rank 的显存里到底有什么

> 本章沿用贯穿全文的算例：7B 模型、8×A100 80GB、bf16 混合精度 + Adam、ZeRO-2 风格 DistributedOptimizer。五类数据"为什么存在"的训练机制推导详见 [LLM训练全解](/self/2026-06-09-training/) 第七节；本章从**内存布局与生命周期**的视角展开——同样的数据，换个显微镜看。

### 2.1 静态大头：五类常驻数据

所谓"静态"，指的是**训练开始时分配、训练结束才释放**的显存——它们的地址和大小全程不变，变的只是内容。

#### ① DDP flat buffer：bf16 可训练参数（$2\Theta$ = 14GB）

**是什么**：模型全部可训练参数的 bf16 副本，前向/反向都用它做计算。

**为什么必须 flatten 成连续大块**——这是内存布局视角下最有意思的设计。一个 7B Transformer 有几百个 parameter tensor（每层的 $W_{q}$、$W_{k}$、$W_{v}$、$W_{o}$、MLP 权重、LayerNorm……），如果它们散落在显存各处：

1. **NCCL 通信效率崩塌**：AllReduce/Broadcast 对连续内存最友好——一次通信调用搬一大块。几百个离散 tensor 意味着几百次小通信，每次都有固定的 launch 延迟和 ring 建立开销，小包带宽利用率可能不足 10%；
2. **kernel launch 开销爆炸**：优化器更新、梯度清零这类逐元素操作，对连续 buffer 一个 kernel 搞定；离散 tensor 则要几百次 launch（每次 ~5μs，纯 CPU 侧开销）；
3. **碎片风险**：几百次独立分配在分配器里留下犬牙交错的块边界，为碎片化埋下伏笔（第三章细讲）。

所以 DDP/Megatron 的做法是：**分配一整块连续的 $2\Theta$ 大 buffer，每个 param tensor 只是这块 buffer 上的一个 view（视图）**——tensor 对象持有指向 buffer 内部某段偏移的指针，不拥有自己的存储：

```text
逻辑视角（模型代码看到的）             物理视角（显存里实际的）
                                    ┌──────────────────────────────┐
  layer0.wq  ─────┐                 │      flat buffer (14 GB)     │
  layer0.wk  ────┐│                 │  连续一整块 bf16 显存          │
  layer0.mlp ───┐││                 ╞══════════╤═══════╤═══════════╡
  ...           │││   view/offset   ║ offset 0 │ off. A│ offset B  ║
  layer31.wo ─┐ ││└───────────────▶ ║ wq 数据  │ wk数据│ mlp 数据  ║
              │ │└────────────────▶ ╚══════════╧═══════╧═══════════╝
              │ └─────────────────▶      ▲每个 param tensor 是
              └───────────────────▶       flat buffer 的一个切片 view
```

**生命周期**：训练开始分配 → 每个 step 末尾被优化器更新的新值覆盖（fp32 cast 回 bf16 后 AllGather 写入）→ 训练结束释放。

**大小**：$2\Theta = 2 \times 7 \times 10^9 = 14 \ \text{GB}$，每卡一份完整副本（数据并行不切参数）。

#### ② 冻结参数（no-grad params，bf16）

**是什么**：`requires_grad=False` 的参数——LoRA 微调时的 base 模型、部分冻结 SFT 中的 embedding/底层等。

**内存布局上的关键区别**：冻结参数**不进 flat buffer**（不参与梯度通信，没必要 flatten），也就**不产生梯度 buffer、不产生优化器状态**。它们只以 bf16 静静躺着，前向要用、反向只传递梯度不累积梯度。

**省显存的账**（LoRA 场景，7B base + 约 0.1B LoRA 参数）：

| 项目 | 全参训练 | LoRA（rank 较高的配置） |
|---|---|---|
| bf16 参数 | 14 GB（flat buffer） | 14 GB 冻结 + 0.2 GB 可训练 |
| 梯度 | 14 GB | 0.2 GB |
| fp32 主权重 + Adam（不切分） | 84 GB | 1.2 GB |
| 合计（激活值除外） | 112 GB | ~15.6 GB |

冻结 98.6% 的参数，砍掉的是**梯度和优化器状态这 12 bytes/param 的大头**——这就是 LoRA 单卡能训 7B 的根本原因。

**生命周期**：与 flat buffer 相同，全程常驻，但内容从不改变。

#### ③ 梯度缓冲区（grad buffer，bf16 或 fp32，14 或 28GB）

**是什么**：与 flat buffer 等大的另一块连续 buffer，存放当前 step 反向传播算出的梯度 $\nabla_{W} L$。

**bucket 机制**：如果等整个 backward 结束再一次性 AllReduce 14GB 梯度，通信完全无法与计算重叠。所以 grad buffer 被逻辑切分成若干个 **bucket**（典型 25~40MB 一个）：反向传播按从输出层到输入层的顺序进行，每当一个 bucket 内所有梯度就绪，立刻对这个 bucket 发起异步 AllReduce——**边算边传**：

```text
时间 ──────────────────────────────────────────────▶
计算流:  ┌─backward L31─┬─backward L30─┬─backward L29─┬─ ...
         └──────────────┴──────┬───────┴──────┬───────┴─
                bucket0 就绪 ──┘   bucket1 就绪┘
通信流:                 ┌─AllReduce b0─┐┌─AllReduce b1─┐ ...
                        └──────────────┘└──────────────┘
         ▲ 反向计算与梯度通信在两条 CUDA stream 上重叠
```

理想情况下通信被计算完全掩盖，只有最后一个 bucket 的通信暴露在关键路径上。bucket 大小是一个经典的工程权衡：

| bucket 大小 | 通信次数 | 单次带宽利用率 | 重叠粒度 | 结论 |
|---|---|---|---|---|
| 过小（~1MB） | 多 | 低（小包悬崖） | 细 | launch 开销吞噬收益 |
| 适中（25~40MB） | 适中 | 高 | 适中 | PyTorch DDP 默认 25MB 的由来 |
| 过大（~1GB） | 少 | 高 | 粗 | 第一个 bucket 迟迟不就绪，重叠窗口浪费 |

**精度选择**：bf16 梯度（$2\Theta$ = 14GB）省显存、通信量减半；fp32 梯度（$4\Theta$ = 28GB）在梯度累积多个 micro-batch 时精度更稳。本文算例取 bf16。

**生命周期**（一个 step 内走完全程）：

$$
\text{backward 逐 bucket 写入} \rightarrow \text{AllReduce（ZeRO-2 为 ReduceScatter）} \rightarrow \text{优化器消费} \rightarrow \text{清零}
$$

注意它存的是"当前 step 的梯度"——历史梯度信息属于 Adam 状态，不归它管。

#### ④ FP32 主权重（master weights，sharded，$4\Theta/N$ = 3.5GB）

**是什么**：参数的 fp32"真值"副本，`optimizer.step()` 在它上面做累加更新。

**为什么 bf16 尾数不够**：bf16 只有 8 位尾数，约 2~3 位十进制有效数字。设参数值 $w = 1.0$、学习率 $10^{-4}$、梯度 $g_{t} = 1.0$，则单步更新量为 $10^{-4}$：

$$
\underbrace{1.0 + 0.0001 = 1.0}_{\text{bf16：更新被舍入归零}} \qquad
\underbrace{1.0 + 0.0001 = 1.0001}_{\text{fp32：更新被保留}}
$$

bf16 在 1.0 附近的相邻可表示数间隔约 $2^{-8} \approx 0.0039$，远大于 $10^{-4}$ 的更新量——**成千上万个微小更新会被静默吞掉，模型训不动**。所以必须在 fp32 上累积，每步更新完再 cast 回 bf16 写入 flat buffer。

**认知误区**："bf16 不需要 loss scaling，所以也不需要 fp32 主权重"。前半句对，后半句错。bf16 和 fp32 指数位同为 8 位，动态范围一致，梯度不会像 fp16 那样下溢归零，确实免了 loss scaling；但**尾数精度问题与动态范围无关**——bf16 尾数（8 位）甚至比 fp16（10 位）更短，小更新被舍入吞掉的问题更严重，fp32 主权重一样不可省。

**Sharded 的含义**：DistributedOptimizer（ZeRO-1/2）下，flat buffer 被按 rank 均分成 $N$ 段，**每个 rank 只保管自己负责那一段的 fp32 副本**，只更新那一段，更新完 AllGather 拼回完整 bf16 参数。这就是分母 $N$ 的来源：

$$
M_{master} = \frac{4\Theta}{N} = \frac{4 \times 7 \times 10^9}{8} = 3.5 \ \text{GB}
$$

**生命周期**：全程常驻；每个 step 被 `optimizer.step()` 原地更新。

#### ⑤ Adam 状态（exp_avg + exp_avg_sq，sharded，$8\Theta/N$ = 7GB）

**是什么**：Adam 为每个参数维护的两个 fp32 统计量：

- `exp_avg`（一阶矩 $m_{t}$）：梯度的指数移动平均，提供动量；
- `exp_avg_sq`（二阶矩 $v_{t}$）：梯度平方的指数移动平均，实现逐参数自适应学习率。

$$
m_{t} = \beta_{1} m_{t-1} + (1-\beta_{1}) g_{t}, \qquad
v_{t} = \beta_{2} v_{t-1} + (1-\beta_{2}) g_{t}^{2}
$$

**内存视角的要点**：两个状态都是 fp32、都与参数等形状，即 $4\Theta + 4\Theta = 8\Theta$——**优化器状态是全部训练状态里最大的一块**（比参数本身大 4 倍）。好在它们只在 `optimizer.step()` 时被读写，与前向/反向完全无关，所以是 sharding 和 offload 的头号目标：

$$
M_{adam} = \frac{8\Theta}{N} = \frac{8 \times 7 \times 10^9}{8} = 7 \ \text{GB}
$$

**生命周期**：全程常驻，跨步累积——丢了等于 Adam 失忆，所以 checkpoint 必须包含它们。

#### 五类数据小结

| # | 数据 | 精度 | 是否 shard | 每卡大小（7B/8卡） | 生命周期 |
|---|---|---|---|---|---|
| ① | flat buffer 参数 | bf16 | 否 | 14 GB | 常驻，每 step 末更新 |
| ② | 冻结参数 | bf16 | 否 | 全参训练时 0 | 常驻，内容不变 |
| ③ | grad buffer | bf16 | 否（通信后逻辑上只用 1/N） | 14 GB | step 内：写→通信→消费→清零 |
| ④ | fp32 主权重 | fp32 | 是（/N） | 3.5 GB | 常驻，step 时原地更新 |
| ⑤ | Adam m/v | fp32 | 是（/N） | 7 GB | 常驻，跨步累积 |
| | **静态合计** | | | **38.5 GB** | |

### 2.2 动态大头：激活值

激活值是前向传播中每层的中间输出，反向传播用链式法则求梯度时必须用到（详见 [LLM训练全解](/self/2026-06-09-training/)）。它与五类静态数据的本质区别：**大小不由模型决定，而由 batch 和序列长度决定**——是显存里唯一的"弹性区"。

Megatron 论文给出的经典估算（不做任何重计算，标准 Transformer）：

$$
M_{act} \approx s \cdot b \cdot L \cdot (34d + 5sh)
$$

公式结果的单位直接就是**字节**：系数 34 与 5 已把 bf16/fp16（2 bytes/元素）的字节因子折算在内，不要再乘一次 bytes；若用 fp32 存激活则整体 ×2。其中 $s$ 为序列长度、$b$ 为 micro-batch、$L$ 为层数、$d$ 为 hidden size、$h$ 为注意力头数（公式推导与 70B 数值表见 [LLM 领域，你需要知道的数字](/self/2026-06-11-llm-panorama/)）。两项的物理含义：

- $34d$ 项：线性层/LayerNorm/GELU 的中间结果，**随 $s$ 线性增长**；
- $5sh$ 项：注意力分数矩阵 $s \times s$，**随 $s$ 平方增长**——长序列的头号杀手。

代入 7B（$L=32$，$d=4096$，$h=32$，$s=4096$，$b=1$）：

| 配置 | $34d$ 项 | $5sh$ 项 | 合计 |
|---|---|---|---|
| 朴素实现 | ~18 GB | ~86 GB | **~104 GB（单卡根本放不下）** |
| + FlashAttention（$5sh$ 项不落显存） | ~18 GB | ~0 | **~18 GB** |
| + Activation Checkpointing（每层只存输入） | | | 可压到 ~2 GB，反向重算换 |

这张表解释了两件事：**为什么 FlashAttention 是长序列训练的前提**（消灭平方项），以及**为什么还需要 Activation Checkpointing**（把线性项也压掉）——后者用计算换显存的完整账目，留到第四章展开，这里先埋下伏笔。本文算例后续按"开 FlashAttention、部分 checkpointing"取 **10~30GB 弹性区间**。

再看清弹性区的"弹性"到底有多大（开 FlashAttention，不重计算，只看 $34d$ 项）：

| 配置变化 | $M_{act}$ | 相对基准 |
|---|---|---|
| $s=4096, b=1$（基准） | ~18 GB | 1× |
| $s=4096, b=2$ | ~36 GB | 2×（随 b 线性） |
| $s=8192, b=1$ | ~36 GB | 2×（FlashAttention 下随 s 线性） |
| $s=8192, b=2$ | ~72 GB | 4×，静态 38.5GB + 72GB > 80GB，**OOM** |

这就是"降 batch 能救 OOM"的定量依据：**batch/seq 是训练工程师手里唯一不改代码就能拧的显存旋钮**。也顺带解释了梯度累积（gradient accumulation）的显存价值：用 4 个 $b=1$ 的 micro-batch 累积梯度，等效 $b=4$ 的优化效果，但激活值峰值始终只有 $b=1$ 的大小——因为每个 micro-batch 的 backward 做完，它的激活值就释放了，只有梯度（本来就常驻）在累加。

### 2.3 隐性占用：账本上找不到的那几个 GB

前两节的账加起来约 50GB，但 `nvidia-smi` 往往显示 55~60GB。差的那几个 GB 就是**隐性占用**——它们不出现在任何 tensor 的 `numel()` 里，却真实吃掉显存：

**（1）CUDA context：每进程 ~300MB-1GB**。进程第一次调用 CUDA API 时，driver 要在显存里建立上下文：kernel 代码段（所有编译好的 cubin）、driver 内部数据结构、printf/assert 缓冲区等。链接的库越多（PyTorch + cuBLAS + cuDNN + NCCL + FlashAttention + 自定义算子），context 越大。**它完全在 PyTorch 分配器视野之外**。

**（2）NCCL 通信 buffer：数百 MB 到数 GB**。每个通信组（communicator）要预分配环形缓冲区、FIFO 队列；开启 `NCCL_BUFFSIZE` 调大、或建了很多通信组（DP + TP + PP + EP 各一套）时，这块能悄悄涨到 2~3GB。

**（3）cuBLAS / cuDNN workspace：数十到数百 MB**。矩阵乘和卷积的高性能算法需要临时工作区（如 split-K 的部分和）。cuBLAS 默认按需分配并缓存；`CUBLAS_WORKSPACE_CONFIG` 可以控制。

**（4）分配器缓存（reserved − allocated）**。PyTorch caching allocator 从不把释放的显存立刻还给 driver，而是缓存起来备用。`torch.cuda.memory_reserved()` 与 `memory_allocated()` 的差值就是"已圈占但当前没装东西"的部分，波动的激活值会把它推高到数 GB。

**（5）碎片**。reserved 池内部被切割得七零八落，空闲总量够但没有一块连续空间能满足新请求——OOM 报错里 "1.83 GiB free" 的真相。（4）（5）都由分配器机制决定，是第三章的主角。

**认知误区**：`nvidia-smi` 显示的显存 ≠ 模型占的显存。三个数字的真实关系是：

$$
\underbrace{\text{nvidia-smi}}_{\text{driver 视角}} = \underbrace{\text{reserved}}_{\text{PyTorch 圈占}} + \underbrace{\text{context + NCCL 等}}_{\text{分配器视野之外}} \; \geq \; \text{reserved} \; \geq \; \underbrace{\text{allocated}}_{\text{tensor 实际用}}
$$

用 `nvidia-smi` 的数字去反推"模型多大"会高估 10%~20%；查泄漏时盯 `allocated` 曲线、查碎片时盯 `reserved − allocated`、查隐性占用时用 `nvidia-smi` 减 `reserved`——三个数字各司其职。

把三个数字一次性打印出来的最小代码（建议在每个 step 末采样）：

```python
import torch

def mem_report(tag=""):
    alloc = torch.cuda.memory_allocated() / 2**30      # tensor 实际占用
    reserv = torch.cuda.memory_reserved() / 2**30      # 分配器圈占
    peak = torch.cuda.max_memory_allocated() / 2**30   # 历史峰值
    print(f"[{tag}] allocated={alloc:.2f}GiB "
          f"reserved={reserv:.2f}GiB peak={peak:.2f}GiB "
          f"cache={reserv - alloc:.2f}GiB")

# 典型输出（7B/8卡, step 末）：
# [step-100] allocated=39.1GiB reserved=52.4GiB peak=61.8GiB cache=13.3GiB
```

读数方法：`allocated ≈ 39GB` 对应静态五件套（step 末激活值已释放）；`peak ≈ 62GB` 是 forward 最深处的"显存山峰"；`cache ≈ 13GB` 是激活值释放后留在分配器里的缓存。能把这三个数字对上 2.4 节的地图，就算真正读懂了自己的显存。

### 2.4 全景图：80GB HBM 的完整布局

把 2.1~2.3 拼起来，就是每个 rank 显存的完整地图（7B/8卡算例，自下而上）：

```text
┌────────────────────────────────────────────────────────────┐ 80 GB
│  free（安全余量，防碎片与峰值）                    ~5-10 GB │
├────────────────────────────────────────────────────────────┤
│  allocator cache（reserved − allocated）           ~2-5 GB │ ◀ 隐性
├────────────────────────────────────────────────────────────┤
│ ╔════════════════════════════════════════════════════════╗ │
│ ║  activations 激活值（弹性区）                  10-30 GB ║ │ ◀ 动态
│ ║  随 forward 逐层增长，backward 逐层释放                 ║ │   随 s·b
│ ║  每个 step 画出一座"显存山峰"                           ║ │   伸缩
│ ╚════════════════════════════════════════════════════════╝ │
├────────────────────────────────────────────────────────────┤
│  ⑤ Adam states m/v（fp32, sharded /8）               7 GB │ ◀ 静态
├────────────────────────────────────────────────────────────┤
│  ④ master weights（fp32, sharded /8）              3.5 GB │ ◀ 静态
├────────────────────────────────────────────────────────────┤
│  ③ grad buffer（bf16, 与参数等大）                   14 GB │ ◀ 静态
├────────────────────────────────────────────────────────────┤
│  ② no-grad 冻结参数（bf16）              全参训练时 ~0 GB │ ◀ 静态
├────────────────────────────────────────────────────────────┤
│  ① DDP flat buffer 参数（bf16, 完整副本）            14 GB │ ◀ 静态
├────────────────────────────────────────────────────────────┤
│  NCCL buffer + cuBLAS workspace                    ~1-2 GB │ ◀ 隐性
├────────────────────────────────────────────────────────────┤
│  CUDA context（kernel 代码 + driver 结构）       ~0.5-1 GB │ ◀ 隐性
└────────────────────────────────────────────────────────────┘ 0 GB
   静态（①-⑤ 38.5GB）训练全程不动；动态（激活值）每 step 呼吸一次；
   隐性（context/NCCL/cache）不在模型账本上，却真实存在。
```

读图要点：**静态区决定"能不能启动"，动态区决定"能开多大 batch"，隐性区决定"为什么账对不上"**。三类问题的排查工具完全不同（第五章展开）。

### 2.5 显存预算速算：16 bytes/param 规则

把上面的账收敛成一条可以心算的规则。bf16 + Adam 混合精度下，每个可训练参数的"全套行头"：

| 组成 | 精度 | bytes/param |
|---|---|---|
| 参数（flat buffer） | bf16 | 2 |
| 梯度 | bf16 | 2 |
| fp32 主权重 | fp32 | 4 |
| Adam $m_{t}$ + $v_{t}$ | fp32 | 4 + 4 |
| **合计** | | **16** |

即**训练状态总量 = $16\Theta$**——7B 模型就是 112GB，这是 8 张卡合计要背的账。ZeRO-2 视角下（参数、梯度不切，优化器状态切 $N$ 份），每卡：

$$
M \approx \underbrace{4\Theta}_{\text{bf16 参数+梯度}} + \underbrace{\frac{12\Theta}{N}}_{\text{fp32 主权重+Adam}} + M_{act} + M_{overhead}
$$

7B/8 卡完整对账（与 [LLM训练全解](/self/2026-06-09-training/) 第八节的 ~50GB 结论对齐）：

| 项目 | 公式 | 每卡大小 |
|---|---|---|
| bf16 参数 + 梯度 | $4\Theta$ | 28 GB |
| fp32 主权重 + Adam | $12\Theta / 8$ | 10.5 GB |
| 激活值 | $M_{act}$（FlashAttention + 部分重计算） | 10~30 GB |
| 隐性开销 | $M_{overhead}$ | ~3 GB |
| **总计** | | **~52-71 GB**，A100 80GB 放得下 |

同一公式外推到更大模型（$N=8$，激活值按同等配置估）：

| 模型 | $4\Theta$ | $12\Theta/8$ | 静态小计 | 结论 |
|---|---|---|---|---|
| 7B | 28 GB | 10.5 GB | 38.5 GB | ✅ 宽裕 |
| 13B | 52 GB | 19.5 GB | 71.5 GB | ⚠️ 静态就近 72GB，激活值没地方放——必须 ZeRO-3 或加卡 |
| 70B | 280 GB | 105 GB | 385 GB | ❌ 单卡装不下零头，必须 ZeRO-3 / TP / PP 组合 |

这张表也解释了工程上的经验法则：**ZeRO-2 + 8 卡的舒适区大约到 10B**；再往上，参数和梯度本身（$4\Theta$）成为不切不行的大头，游戏规则进入 ZeRO-3/FSDP 的世界。

最后把不同 ZeRO stage 的切分范围也收进同一个公式框架，方便心算（激活值与 overhead 另计）：

| 策略 | 切了什么 | 每卡静态显存 | 7B/8卡代入 |
|---|---|---|---|
| 纯 DDP（不切） | 无 | $16\Theta$ | 112 GB，单卡装不下 |
| ZeRO-1 | 优化器状态 | $4\Theta + 12\Theta/N$ | 38.5 GB |
| ZeRO-2（本文算例） | + 梯度 | $2\Theta + 14\Theta/N$（稳态） | 26.25 GB（峰值仍近 38.5） |
| ZeRO-3 / FSDP | + 参数 | $16\Theta/N$ | 14 GB |

注意 ZeRO-2 的梯度切分是"通信后只保留自己那 $1/N$"，但 backward 过程中 bucket 仍需临时存放完整梯度分片，所以**峰值显存接近 ZeRO-1**——这也是本文保守地按 38.5GB 做预算的原因：**预算要按峰值算，不能按稳态算**。

**认知误区**："ZeRO 省显存 = 训练状态变小了"。错。全局训练状态永远是 $16\Theta$（7B 就是 112GB），一个字节都不会少——ZeRO 只是把它**分摊到 $N$ 张卡上**，用通信（AllGather/ReduceScatter）换每卡容量。真正减少状态总量的只有两条路：冻结参数（② 的 LoRA 账）和换优化器（如 8-bit Adam，第四章展开）。分摊与减量，是两类本质不同的优化。

### 本章小结

每个 rank 的 80GB 显存里住着三类房客：**静态五件套**（bf16 参数 + 梯度 + sharded fp32 主权重 + Adam 状态，7B/8 卡合计 38.5GB）签的是整个训练周期的长租约；**激活值**（10~30GB）每个 step 潮起潮落，是唯一能用 batch/seq 调节的弹性区；**隐性占用**（context/NCCL/workspace/分配器缓存，约 3GB+）不在模型账本上，却是 `nvidia-smi` 与理论值对不上的全部原因。用 $M \approx 4\Theta + 12\Theta/N + M_{act} + M_{overhead}$ 一条公式即可完成任何配置的预算速算。

但静态账算得再准，OOM 依然可能发生——因为显存的**分配方式**本身会制造碎片和缓存膨胀。下半部分我们潜入 PyTorch caching allocator 的内部机制，再系统盘点省显存的优化技术与泄漏排查工具链。

---

## 三、PyTorch CUDA 缓存分配器：显存是怎么被"分配"的

前两章回答了"显存里放什么"，这一章回答"显存怎么被分出去"。这个问题之所以值得单开一章，是因为**绝大多数"账算得对却还是 OOM"的案例，根因都在分配器层**：碎片、缓存膨胀、跨 stream 归属——它们都不出现在任何模型账本上。

### 3.1 为什么需要 caching allocator

最朴素的做法是：每创建一个 tensor 就调一次 `cudaMalloc`，销毁就调 `cudaFree`。PyTorch 没有这么做，原因有三个：

1. **`cudaMalloc` 慢**。它要走驱动、改 GPU 页表，一次调用微秒到毫秒级；而训练循环里 tensor 的创建/销毁每个 step 成百上千次，逐次走驱动会把 CPU 端变成瓶颈。
2. **`cudaFree` 会隐式同步设备**。为了保证被释放的内存没有 kernel 还在使用，`cudaFree` 需要等待 GPU 上相关工作完成——这会把好不容易做到的 CPU/GPU 异步流水打断，制造大量气泡。
3. **训练的分配模式高度重复**。每个 step 申请的激活值 tensor 大小、顺序几乎一模一样，"上个 step 刚还回来的块，下个 step 原样再要一遍"——完美的缓存场景。

所以 PyTorch 的策略是：**向驱动要来的显存不再还回去，而是入池缓存、自己管理复用**。tensor "释放"（引用计数归零）时，它的显存块只是回到 PyTorch 自己的空闲链表，`cudaFree` 并没有发生。这就是第二章说"分配器缓存是隐性占用"的机制来源，也是 `nvidia-smi` 的数字**只增不减**的原因。

**认知误区**："我 `del tensor` 之后 `nvidia-smi` 没降，说明内存泄漏了"。错。`del` 之后显存回到了 PyTorch 的缓存池（`memory_allocated()` 会降），但池子不还给驱动（`nvidia-smi` 不降）。这是设计行为，不是泄漏。判断泄漏要看 `memory_allocated()`，第五章展开。

### 3.2 分配机制细节：pool、block、切分与合并

分配器的核心数据结构是两级：**Segment**（向 `cudaMalloc` 要来的整块）与 **Block**（Segment 内部切出来给 tensor 用的小块）。规则如下：

- 所有请求先向上取整到 **512B 的倍数**；
- 请求 **< 1MB**：走 **small pool**，Segment 按 **2MB** 一块向驱动申请，内部切成小 block（大量标量、norm 权重、索引 tensor 都住这里，避免小对象打碎大空间）；
- 请求 **≥ 1MB**：走 **large pool**，1~10MB 的请求按 **20MB** Segment 申请，更大的请求按取整后的实际大小单独开 Segment；
- **切分（split）**：从空闲链表找到一个够大的 block 后，切出请求的大小，剩余部分作为新的空闲 block 留在同一 Segment 内；
- **合并（coalesce）**：block 释放回池时，若同一 Segment 内的**相邻** block 也空闲，则合并成更大的空闲块——注意只能在 Segment 内部合并，跨 Segment 的空闲块永远无法拼成连续空间；
- **best-fit 查找**：按大小组织的空闲链表中找"最小的够用块"，找不到才向驱动要新 Segment。

```text
                     PyTorch Caching Allocator
┌───────────────────────────────────────────────────────────────┐
│  Large Pool（请求 ≥ 1MB）                                      │
│  ┌─ Segment 20MB（一次 cudaMalloc）──────────────┐             │
│  │ Block A 8MB │ Block B 4MB │  free 8MB         │             │
│  │  (in use)   │  (in use)   │  (空闲链表中)      │◀─ 切分残留  │
│  └─────────────┴─────────────┴───────────────────┘             │
│  ┌─ Segment 512MB（>10MB 请求按需单开）───────────┐            │
│  │          Block C 512MB (in use)                │            │
│  └────────────────────────────────────────────────┘            │
│                                                                │
│  Small Pool（请求 < 1MB，按 2MB Segment）                       │
│  ┌─ Segment 2MB ──────────────────────────────────┐            │
│  │ 512B │ 1KB │ 64KB │ 512KB │      free          │            │
│  └──────┴─────┴──────┴───────┴────────────────────┘            │
│                                                                │
│  释放 = 回空闲链表（相邻空闲块合并）；cudaFree 不发生            │
└───────────────────────────────────────────────────────────────┘
```

还有一个容易踩坑的维度：**block 归属于 stream**。分配发生在哪个 CUDA stream，这个 block 就记在哪个 stream 名下，空闲后默认只能被**同一个 stream** 的后续分配复用——因为分配器靠 stream 内的顺序性保证"复用时旧 kernel 一定已经用完了这块内存"。如果一个 tensor 被另一个 stream 的 kernel 使用（如通信/计算重叠的场景），必须调用 `tensor.record_stream(other_stream)` 告知分配器，否则会出现**数据竞争**（块被提前复用、内容被覆盖）或**显存虚高**（保守起见延迟复用）。多 stream + 自定义内存管理是 NCCL 重叠、pipeline 并行实现里最隐蔽的 bug 来源之一。

### 3.3 三个水位：allocated ≤ reserved ≤ nvidia-smi

有了池化机制，"显存用了多少"就有了三个不同答案，必须分清：

```text
 0GB                                                        80GB
 ├── allocated：活跃 tensor 真正占用 ──┤
 ├── reserved：allocated + 池中空闲缓存 block ──────────┤
 ├── nvidia-smi used：reserved + CUDA context + NCCL/驱动 ────┤
      （若有其他进程共卡，还要再加上它们的占用）
```

| 水位 | 查询方式 | 语义 | 谁能降低它 |
|---|---|---|---|
| allocated | `torch.cuda.memory_allocated()` | 所有存活 tensor 字节数之和 | 释放 tensor 引用 |
| reserved | `torch.cuda.memory_reserved()` | 分配器向驱动拿到的全部 Segment 总量 | `empty_cache()`（部分） |
| nvidia-smi | `nvidia-smi` / NVML | 进程在 GPU 上的全部占用 | 结束进程 |
| 峰值 | `torch.cuda.max_memory_allocated()` | 历史最高 allocated（判断"差多少 OOM"的依据） | `reset_peak_memory_stats()` 复位统计 |

三条恒等式式的经验：

1. **allocated ≤ reserved 恒成立**，差值就是缓存池里的空闲块。差值大不代表有问题（正常缓存复用），差值大**且分配失败**才是碎片（见 3.4）。
2. **reserved 与 nvidia-smi 的差**约等于 CUDA context（0.5~1GB）+ NCCL 缓冲 + cuBLAS/cuDNN workspace——即第二章的 $M_{overhead}$。
3. **容量规划要看 `max_memory_allocated()`**，而不是任意时刻的 allocated：OOM 发生在峰值处（通常是 backward 中段，激活值尚未释放完、梯度桶已开始堆积的时刻），稳态数字会给你虚假的安全感。

```python
import torch

GiB = 2**30
print(f"allocated: {torch.cuda.memory_allocated()/GiB:.2f} GiB")
print(f"reserved:  {torch.cuda.memory_reserved()/GiB:.2f} GiB")
print(f"peak:      {torch.cuda.max_memory_allocated()/GiB:.2f} GiB")
torch.cuda.reset_peak_memory_stats()  # 复位峰值，便于按阶段测量
```

### 3.4 碎片化：reserved 明明很大，为什么还 OOM

回到开篇那条报错，现在可以逐字段解读了：

```text
CUDA out of memory. Tried to allocate 2.50 GiB (GPU 0; 79.35 GiB
total capacity; 68.42 GiB already allocated; 1.83 GiB free;
74.51 GiB reserved in total by PyTorch)
```

| 字段 | 数值 | 含义 |
|---|---|---|
| Tried to allocate | 2.50 GiB | 本次请求大小 |
| total capacity | 79.35 GiB | 显卡总容量（80GB 卡去掉 ECC/预留后） |
| already allocated | 68.42 GiB | 活跃 tensor 占用（水位一） |
| reserved | 74.51 GiB | PyTorch 池子总量（水位二） |
| free | 1.83 GiB | 驱动侧还能给的新 Segment 空间 |

算一笔账：reserved − allocated = 74.51 − 68.42 = **6.09 GiB 的空闲缓存**，明明比 2.50 GiB 的请求大，为什么失败？因为这 6.09 GiB 是**散落在各个 Segment 里的不连续小块**，最大连续块不足 2.50 GiB；而驱动侧的 free 只剩 1.83 GiB，也开不出新 Segment。空闲总量够、连续块不够——这就是**外碎片（external fragmentation）**。

训练场景中碎片的两大来源：

1. **变长序列导致激活块大小抖动**。step 1 的 `seq_len=1897` 申请一批块，step 2 的 `seq_len=3041` 需要的块更大，旧块尺寸对不上——切一刀，残留一堆"差一点够用"的空闲块。数据不做 length bucketing / padding 对齐的训练任务，跑几千 step 后 reserved 与 allocated 的差值会持续爬升。
2. **长短生命周期的块交错**。长寿块（如手动缓存的 embedding、日志里存的 tensor）像钉子一样钉在 Segment 中间，两侧的短寿块释放后无法跨过它合并。

```text
碎片化前（step 1：块大小整齐，free 连续）：
┌──────┬──────┬──────┬──────┬──────┬────────────────┐
│ A 2G │ B 2G │ C 2G │ D 2G │ E 2G │   free 10G     │
└──────┴──────┴──────┴──────┴──────┴────────────────┘

碎片化后（数千 step：变长 seq + 长寿块钉住中间）：
┌──────┬────┬───────┬────┬──────┬────┬──────────────┐
│ A 2G │f 1G│ C' 3G │f.5G│ E 2G │f 2G│   free 3G    │
└──────┴────┴───────┴────┴──────┴────┴──────────────┘
  空闲总量 = 1 + 0.5 + 2 + 3 = 6.5G
  最大连续块 = 3G  →  申请 4G：OOM
```

### 3.5 调参与新机制：从 `max_split_size_mb` 到 expandable segments

分配器行为可以用环境变量 `PYTORCH_CUDA_ALLOC_CONF` 调整（必须在**第一次 CUDA 分配之前**设置才生效）：

```bash
# 老牌抗碎片参数：大于 128MB 的 block 禁止切分，
# 防止大块被切出难以合并的残片（代价：可能多占一点显存）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 请求大小按 2 的幂再分 4 档取整，减少"块大小的种类数"，
# 提高变长 seq 场景下旧块被原样复用的概率
export PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4

# 现代答案（PyTorch 2.x）：可扩展 Segment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**`expandable_segments:True` 值得单独讲**，因为它改变了游戏规则。传统模式下 Segment 一旦 `cudaMalloc` 出来大小就固定了，碎片本质上是"固定大小盒子装不下变化的货"。expandable segments 改用 CUDA 虚拟内存 API（`cuMemAddressReserve` 预留一大段**虚拟地址**，`cuMemMap` 按需映射**物理页**）：Segment 可以原地伸长，物理页可以在虚拟地址不变的前提下归还再映射。效果是"所有分配住在一个可伸缩的大 Segment 里"，相邻合并几乎总能成功，**变长 seq 场景的碎片问题大幅缓解**。vLLM 的 PagedAttention 用同样的"虚拟地址连续 + 物理页离散"思想管理 KV Cache——分页，是操作系统五十年前就给出的答案（参见 [KV Cache 是算还是传？](/self/2026-06-10-kvcache-compute-vs-transfer/) 中的显存管理讨论）。

再澄清两个常被误用的 API：

- **`torch.cuda.empty_cache()`**：把池中**完全空闲的 Segment** 还给驱动（正在使用或部分使用的 Segment 一个字节都还不了）。它**不解决碎片**（碎片恰恰是"Segment 部分空闲"），只降低 reserved 水位；且之后的分配要重新走 `cudaMalloc`，训练循环里频繁调用会明显拖慢速度。合理场景只有两个：训练/评估阶段切换时释放一大批缓存给同卡的其他进程；OOM 现场记录完信息后的清理。
- **`torch.cuda.set_per_process_memory_fraction(0.8)`**：给本进程设置显存占用上限（80GB × 0.8 = 64GB），超过就抛 OOM 而不是挤占别人。多进程/多租户共卡时用它划边界，单进程独占训练不需要。

### 3.6 本章小结：现象 → 根因 → 处理

| 现象 | 根因 | 处理手段 |
|---|---|---|
| `del` 后 `nvidia-smi` 不降 | 池化缓存的设计行为 | 无需处理；看 `memory_allocated()` |
| reserved − allocated 持续增大，最终 OOM | 外碎片（变长 seq / 生命周期交错） | `expandable_segments:True`；length bucketing；`max_split_size_mb` |
| "reserved 74.51 GiB" 却分不出 2.5 GiB | 空闲块不连续 | 同上；重启进程是最后的碎片整理 |
| 多 stream 下显存虚高或数值错误 | block 的 stream 归属未声明 | `record_stream()` 或显式同步 |
| `empty_cache()` 后显存没怎么降 | 空闲块所在 Segment 未完全空闲 | 属预期行为；别指望它救碎片 |
| 偶发 OOM 但平均水位不高 | 峰值毛刺（backward 中段） | 看 `max_memory_allocated()` 而非稳态值 |
| 与推理服务共卡互相挤爆 | 无进程级隔离 | `set_per_process_memory_fraction` |

分配器解决的是"怎么分"的效率问题，但它变不出容量。当 $4\Theta + 12\Theta/N + M_{act}$ 本身就超过 80GB 时，就需要下一章的武器库了。

---

## 四、内存优化技术全景

这一章把所有"省显存"的手段收进一张地图。核心心法只有一条：**显存优化的本质是用别的资源换容量**——用算力换（重计算）、用通信换（ZeRO/并行）、用带宽换（offload）、用精度换（量化）。没有免费的午餐，只有代价排序。

### 4.1 Activation Checkpointing：用 33% 算力买回激活值

**直觉**：第二章说过，激活值是为 backward 存的"草稿纸"——既然是草稿，丢了可以重新演算。Activation checkpointing（也叫 gradient checkpointing / 重计算）的策略是：前向时只保存少数**检查点边界**的激活，区间内部的激活直接丢弃；反向传播走到某个区间时，从最近的检查点**重新跑一遍该区间的前向**，把激活现算出来再做反向。

**数字**：设模型有 $L$ 层，每层激活占 $a$。全存策略的激活显存是 $O(L \cdot a)$；若把 $L$ 层均分成 $k$ 段、只存段边界，显存变为"边界 $k \cdot a$ + 重算时区间内峰值 $(L/k) \cdot a$"，对 $k$ 求极小值得 $k = \sqrt{L}$：

$$
M_{act} \approx \left( k + \frac{L}{k} \right) a \ \xrightarrow{\ k=\sqrt{L}\ } \ 2\sqrt{L} \cdot a = O(\sqrt{L})
$$

代价是每个区间的前向被算了两遍（一次正常前向 + 一次反向前的重算）。由于训练总 FLOPs 中前向约占 1/3（backward 约 2 倍前向），全量重计算的算力税约为 **+33%**。

```text
全存激活 O(L)：
  Forward:   L1 → L2 → L3 → L4 → L5 → L6 → L7 → L8
  显存驻留:  ▂▃▄▅▆▇█ （激活逐层堆到峰值，backward 才逐层释放）
  Backward:  L8 → L7 → ... → L1（直接消费已存激活）

Checkpointing O(√L)（每 4 层设一个检查点）：
  Forward:   [L1 L2 L3 L4] ─ckpt₁─ [L5 L6 L7 L8] ─ckpt₂
  显存驻留:  ▂▂（只存 2 个边界激活，区间内部即算即丢）
  Backward:  段二：从 ckpt₁ 重算 L5..L8 前向 ─▶ 反向 L8..L5
             段一：从输入  重算 L1..L4 前向 ─▶ 反向 L4..L1
  代价：多跑约一次前向 → 总 FLOPs ×1.33
```

**工程用法**（PyTorch 原生 API，Transformer 按层包裹是标准姿势）：

```python
import torch
from torch.utils.checkpoint import checkpoint

class Block(torch.nn.Module):
    def forward(self, x):
        return self.mlp(self.attn(x))

# 训练循环中：对每个 Transformer 层启用重计算
for blk in model.blocks:
    x = checkpoint(blk, x, use_reentrant=False)
```

`use_reentrant=False` 是必选项而非可选项：新实现（非重入版）支持 `torch.compile`、嵌套 checkpoint、关键字参数，且不再要求输入必须 `requires_grad`——老的重入实现已在弃用路径上。HuggingFace 模型一行 `model.gradient_checkpointing_enable()` 即可全局开启。

**更聪明的做法是选择性重计算（selective activation checkpointing）**：不同算子的"显存/重算代价比"差异巨大。attention 内部的中间结果（softmax 前后矩阵）显存大、重算便宜（memory-bound 算子），最值得丢；大矩阵乘输出重算贵（compute-bound），值得留。Megatron-LM 的 selective recompute 只重算 attention 部分，实测能拿到全量重计算大部分的显存收益，算力税却从 33% 降到 ~5%。用了 FlashAttention 之后（见 4.4），attention 中间矩阵本来就不落显存，两者天然互补。

**7B 算例**：第二章的激活值上界 30GB（micro-batch 偏大、不开重算的配置）。按层全量 checkpointing 后，只需保存 32 层的边界输入（形状 $[b, s, d]$）：$32 \times b \times s \times d \times 2\,\text{bytes}$，代入 $b=4, s=4096, d=4096$ 约 4GB，加上重算区间内的瞬时峰值，**约 5GB**——一举把激活从"最大的可变项"压成零头。省下的 25GB 可以换成 3~4 倍的 micro-batch，或者留给更长的序列。它与梯度累积是黄金搭档：checkpointing 压低单个 micro-batch 的激活峰值，梯度累积在时间上串行堆出大等效 batch，两者相乘可以在 80GB 卡上凑出原本不可能的 global batch size。

### 4.2 ZeRO 系列：把 112GB 的训练状态摊到 N 张卡上

第二章末尾已经给出结论表，这里把**每一级到底切了什么、为什么能切、通信代价是什么**讲透。记号沿用全文：bf16 参数 $2\Theta$、bf16 梯度 $2\Theta$、fp32 主权重 + Adam 状态 $12\Theta$，全局训练状态合计 $16\Theta$（7B = 112GB）。

**逐级拆解**（$N$ = 数据并行卡数）：

- **Stage 0（纯 DDP）**：每卡一份完整的 $16\Theta$。梯度用 AllReduce 同步。7B 单卡 112GB——直接出局。
- **Stage 1（切优化器状态）**：观察到 optimizer.step() 里**每个参数的更新是独立的**——第 $i$ 个参数的 Adam 更新只需要它自己的 $m_i, v_i$、主权重和梯度。所以 $12\Theta$ 的优化器状态可以按参数均分，每卡只负责更新自己那 $1/N$，更新完再把新参数广播回来。每卡显存 $4\Theta + 12\Theta/N$。通信上把 AllReduce 换成等价的 ReduceScatter + AllGather，**总通信量与 DDP 相同（约 $2\Theta$ 梯度字节的 2 倍传输）**，几乎白赚。
- **Stage 2（再切梯度）**：既然每卡只更新 $1/N$ 的参数，它也只需要那 $1/N$ 的梯度——backward 中梯度桶做完 ReduceScatter 后，非本卡负责的部分立即丢弃。稳态每卡 $2\Theta + 14\Theta/N$。通信量仍与 DDP 相同。注意第二章强调过的：backward 过程中 bucket 需要临时容纳未通信的完整梯度，**峰值接近 Stage 1**，预算按峰值算。
- **Stage 3（连参数也切）**：每卡常驻的参数只有 $1/N$ 分片；前向/反向算到某一层时，先把该层参数 AllGather 拼完整，用完立刻释放。每卡显存 $16\Theta/N$——训练状态被彻底摊平。代价：前向 AllGather 一遍参数（$2\Theta$）、反向再 AllGather 一遍（$2\Theta$）、梯度 ReduceScatter（$2\Theta$），相对 DDP 的通信量约 **1.5 倍**，且参数聚合在关键路径上，需要预取（prefetch）与计算重叠才能不掉吞吐。

```text
            rank0         rank1         rank2         rank3
         ┌─────────────┬─────────────┬─────────────┬─────────────┐
 Stage-1 │ P  G  o0    │ P  G  o1    │ P  G  o2    │ P  G  o3    │
 Stage-2 │ P  g0 o0    │ P  g1 o1    │ P  g2 o2    │ P  g3 o3    │
 Stage-3 │ p0 g0 o0    │ p1 g1 o1    │ p2 g2 o2    │ p3 g3 o3    │
         └─────────────┴─────────────┴─────────────┴─────────────┘
   大写 = 完整副本；小写+编号 = 只存 1/N 分片
   P = bf16 参数(2Θ)   G = bf16 梯度(2Θ)   O = fp32主权重+Adam(12Θ)
   Stage-3 计算某层时: AllGather 该层 p ─▶ 用完即弃 ─▶ 下一层
```

**7B / 8 卡各 stage 每卡静态显存对账**（激活值与 overhead 另计，与第二章表格自洽）：

| Stage | 切了什么 | 每卡公式 | 7B/8卡 | 相对 DDP 通信量 |
|---|---|---|---|---|
| 0（DDP） | 无 | $16\Theta$ | 112 GB ❌ | 1× |
| 1 | 优化器状态 | $4\Theta + 12\Theta/N$ | 38.5 GB | 1× |
| 2 | + 梯度 | $2\Theta + 14\Theta/N$（稳态） | 26.25 GB（峰值≈38.5） | 1× |
| 3 | + 参数 | $16\Theta/N$ | 14 GB | ~1.5× |

这张表回答了"该用哪级"：**通信免费的 Stage 1/2 是默认选项**（本文算例即 ZeRO-2 风格）；Stage 3 只在 $4\Theta$ 本身放不下时才启用（13B+ 单机、或 70B 这种量级），并且要接受通信换容量的税。

**与 FSDP 的对应关系**：PyTorch 原生 FSDP 是 ZeRO 思想的官方实现，`ShardingStrategy.FULL_SHARD` ≈ Stage 3，`SHARD_GRAD_OP` ≈ Stage 2，`NO_SHARD` ≈ DDP。FSDP2（per-parameter sharding，DTensor 化）进一步把分片粒度做到单参数，与 `torch.compile` 组合更干净。选型上：DeepSpeed ZeRO 配置驱动、功能全（offload/Infinity）；FSDP 与 PyTorch 生态咬合更紧。

**再往下切就出显存了**：ZeRO-Offload 把优化器状态和 fp32 更新挪到 CPU 内存（$12\Theta$ 从显存账上消失，代价是每 step 跨 PCIe 搬梯度和参数），ZeRO-Infinity 进一步用上 NVMe。记账摘要（7B/8卡，ZeRO-2 + 优化器状态 offload）：

- $M_{HBM} \approx 4\Theta + M_{act} + M_{overhead}$——bf16 参数 + 梯度留卡，静态部分 $4\Theta = 28$GB（对比不 offload 的 38.5GB，省的正是那 $12\Theta/N$）；
- $M_{RAM} \approx 12\Theta/N$ 每 rank——fp32 主权重 + Adam 状态搬去内存，$84/8 = 10.5$GB/rank，8 个 rank 合计 84GB 系统内存；
- 每 step 的代价：梯度下行、更新后参数上行各 $2\Theta/N$ 量级要过 PCIe，25GB/s 的带宽墙决定了它只适合"HBM 真不够"的场景。

| 方案 | 每卡 HBM 静态 | 每 rank RAM | 瓶颈链路 |
|---|---|---|---|
| ZeRO-2（不 offload） | 38.5 GB | ~0 | HBM 容量 |
| + 优化器状态 offload | **28 GB** | 10.5 GB | PCIe 25GB/s |
| + ZeRO-Infinity（NVMe） | 可再压 | 可再压 | NVMe 7GB/s，更慢 |

这条"容量无限但带宽骤降"的路线、以及 PCIe 25GB/s 如何成为新瓶颈的完整分析，见 [GPU 显存 Offload 技术](/self/2026-05-28-gpu-memory-offload-techniques/)，此处不重复。

### 4.3 并行策略的显存分布视角

[LLM 领域，你需要知道的数字](/self/2026-06-11-llm-panorama/) 的 2.4 节从**通信量**视角讲过 TP/PP/DP，这里换成**显存**视角：每种并行到底把哪部分内存切掉了、哪部分没切、付出了什么。

- **TP（张量并行）**：把每层内部的大矩阵按行/列切到 $tp$ 张卡（列切 $W_{up}$、行切 $W_{down}$ 的 Megatron 风格）。**权重、梯度、优化器状态都随之 $/tp$**，矩阵乘的输出激活也切了；但每层边界处 AllReduce 之后的激活（LayerNorm 输入、Dropout mask 等）**每卡仍是完整一份**。这就是 **SP（Sequence Parallel）** 存在的理由：在 TP 组内把这些"没切到的激活"沿序列维再切一刀（AllReduce 拆成 ReduceScatter + AllGather，通信量不变），让激活也完整地 $/tp$。Megatron 中 TP+SP 总是成对开启。
- **PP（流水线并行）**：按层切成 $pp$ 段，权重/梯度/优化器状态 $/pp$，简单直接。但激活账没那么便宜：为了填满流水线，1F1B 调度下 warmup 阶段第一个 stage 最多同时驻留 **$pp$ 份 in-flight micro-batch 的激活**——层数省了 $pp$ 倍、份数多了 $pp$ 倍，激活显存**基本没省**（interleaved 调度还会再多存一点）。PP 省的是权重不是激活，这一点常被误解。
- **CP（Context Parallel）**：超长序列场景，把激活沿 seq 维切到多卡、attention 用 Ring 方式跨卡交换 KV，一句带过。
- **EP（专家并行）**：MoE 模型把不同 expert 的权重放到不同卡（expert 权重 $/ep$），但 token 要按路由结果跨卡 all-to-all，需要额外的**路由缓冲区**——capacity factor 越大、缓冲越大，是 MoE 训练显存里独有的一项。

给 EP 算一笔小账（7B MoE，假设 expert 参数占总参数 50%，EP=4）：expert 权重 bf16 全量 $0.5 \times 2\Theta = 7$GB，按 EP 切分后每卡 $7/4 = 1.75$GB；expert 对应的梯度与 Adam 状态同比例 $/ep$，非 expert 部分（attention、embedding 等）仍由 DP/ZeRO 域照常处理。路由缓冲的量级估算 ≈ capacity factor × tokens/batch × $d$ × 2 bytes：代入 capacity factor 1.25、$b \times s = 16384$ tokens、$d = 4096$，单个缓冲约 170MB，几套收发 buffer 叠加后是数百 MB 量级——不致命，但必须记入 $M_{overhead}$。

| 静态项（每卡，bf16 权重口径） | 无 EP（expert 全量复制） | EP=4 |
|---|---|---|
| expert 权重 | 7 GB | **1.75 GB** |
| 非 expert 权重 | 7 GB | 7 GB（不切） |
| expert 的梯度 + Adam 状态 | 全量 | 同比例 $/4$ |
| 路由 all-to-all 缓冲 | 0 | +数百 MB |

这张表也解释了 MoE 显存调优的两个旋钮：capacity factor 调小省缓冲但丢 token（超出容量的 token 被 drop），调大则反之；而 expert 占比越高的模型，EP 的切分收益越大——非 expert 部分永远是 EP 切不动的地板。

| 并行 | 切了什么内存 | 没切什么 | 代价 |
|---|---|---|---|
| DP/ZeRO | 训练状态（按 stage） | 激活（每卡完整算自己的 batch） | 梯度/参数通信 |
| TP | 权重/梯度/优化器 $/tp$，矩阵激活 $/tp$ | 层边界激活（需 SP 补刀） | 每层 2 次 AllReduce，必须高带宽域 |
| SP | TP 没切完的激活 $/tp$ | — | 与 TP 绑定，通信量不变 |
| PP | 权重/梯度/优化器 $/pp$ | 激活（in-flight $pp$ 份抵消） | 流水线气泡，点对点通信 |
| CP | 激活沿 seq $/cp$ | 权重 | attention 跨卡交换 KV |
| EP | expert 权重 $/ep$ | 非 expert 部分、激活 | all-to-all + 路由缓冲 |

**为什么 TP 必须放机内、DP/PP 可以跨机**？答案在通信频率×数据量与物理带宽的匹配：

```text
            节点 0（8×A100，NVSwitch 全互联）          节点 1
 ┌─────────────────────────────────────┐  ┌─────────────────────┐
 │  GPU0   GPU1   GPU2   GPU3          │  │  GPU8  ...  GPU15   │
 │    ╚══════╩══════╩══════╝           │  │   ╚═════╦═════╝     │
 │    ║   NVSwitch ~600GB/s 双向  ║    │  │    NVSwitch         │
 │    ╔══════╦══════╦══════╗           │  │                     │
 │  GPU4   GPU5   GPU6   GPU7          │  │                     │
 │         │ NIC (IB/RoCE)             │  │        │ NIC        │
 └─────────┼───────────────────────────┘  └────────┼────────────┘
           │      ~25~50 GB/s per port             │
           │      (200~400 Gbps × 若干口)          │
           └────────────┬──────────────────────────┘
                   IB / RoCE 交换机
 布局原则（带宽差 ~10×，决定谁住哪）：
   TP+SP 组 ─▶ 机内 NVLink 域（每层都通信、量大、在关键路径）
   PP      ─▶ 可跨机（只传层边界激活，点对点、量小）
   DP/ZeRO ─▶ 跨机（每 step 一次梯度通信，可与 backward 重叠）
```

NVLink 域内 ~600GB/s 对跨机单口 ~25~50GB/s，一个数量级的落差——所以 TP 这种"每层两次、传完整激活"的高频大流量通信必须锁死在机内；DP 的梯度 AllReduce 每 step 才一次、还能藏进 backward 的计算时间里，跨机走 IB 完全扛得住。这条"通信密度决定物理位置"的原则，是所有 3D/4D 并行拓扑设计的第一定律（RDMA/RoCE 侧的细节见 [LLM 领域，你需要知道的数字](/self/2026-06-11-llm-panorama/)）。

**并行组合怎么记账**：真实训练往往是组合拳。心算口径一句话——**静态训练状态先 $/tp$ 再 $/pp$，剩下的 DP 域内再按 ZeRO 分片**（激活另算：TP+SP 切、PP 基本不省、DP/ZeRO 不切）。以 7B、8 卡为例，峰值口径，与 4.2 的 38.5GB 基准自洽：

| 组合（8 卡总并行度不变） | 每卡静态公式（峰值口径） | 7B 估算 |
|---|---|---|
| 仅 ZeRO-2（DP=8） | $4\Theta + 12\Theta/8$ | 28 + 10.5 = **38.5 GB** |
| ZeRO-2 + TP=2（DP=4） | $4\Theta/2 + 12\Theta/(2 \times 4)$ | 14 + 10.5 = **24.5 GB** |
| ZeRO-2 + TP=2 + PP=2（DP=2） | $4\Theta/(2 \times 2) + 12\Theta/(2 \times 2 \times 2)$ | 7 + 10.5 = **17.5 GB** |

优化器状态那一项三行都是 10.5GB 并非巧合：$12\Theta$ 除以的始终是全部 8 卡（$tp \times pp \times dp = 8$），切法怎么变、总分母不变；真正被 TP/PP 压下去的是每卡完整副本的 $4\Theta$。注意这只是量级估算——忽略了 embedding 层在 TP/PP 下的不均衡、PP 首 stage 的 in-flight 激活等细节。

顺带回答一个常见选型问题：同样 8 卡，纯 ZeRO-3 能切到 $16\Theta/8 = 14$GB，比 TP2+PP2 组合的 17.5GB 更狠，且不用改模型代码——那为什么还要 TP/PP？因为 ZeRO-3 只切静态状态、不切激活（每卡仍算完整的 batch 前向），而 TP+SP 连激活一起 $/tp$；长序列、大 micro-batch 场景下，激活才是峰值的大头，这时候 TP 的价值才显现。

### 4.4 其他优化技术速览

| 技术 | 一句话原理 | 省多少 | 适用场景 |
|---|---|---|---|
| FlashAttention | attention 分块在 SMEM 里算完，$O(s^2)$ 的 score 矩阵不落 HBM | 激活中最大的 $s^2$ 项归零 | 默认开启，无理由不用 |
| 8-bit 优化器 | bitsandbytes 把 Adam 的 m/v 量化到 int8（分块量化保精度） | 优化器状态 $8\Theta \to 2\Theta$ | 单卡/小规模微调最香 |
| QLoRA | 基座权重 NF4 量化冻结（$0.5\Theta$）+ LoRA 小分支训练 | 7B 微调进 10GB 级 | 消费级卡微调 |
| Fused kernel | 把 bias+gelu、norm+residual 等融合成单 kernel，中间结果不落显存 | 中间激活若干份 | apex/torch.compile 自动覆盖 |
| `zero_grad(set_to_none=True)` | 梯度置 None 释放显存，而非置零保留 | 峰值处省最多 $2\Theta$ | PyTorch 2.x 已是默认 |
| CPU/NVMe offload | 冷数据挪出 HBM，用 PCIe 带宽换容量 | 上不封顶 | 卡少模型大，详见 [GPU 显存 Offload 技术](/self/2026-05-28-gpu-memory-offload-techniques/) |
| 梯度累积 | 时间换空间：小 micro-batch 串行累积出大等效 batch | 激活按累积步数线性降 | 几乎所有大 batch 训练 |

### 4.5 OOM 决策树：按代价从小到大

把本章武器按"实施成本 + 副作用"排好序，OOM 时照着走：

```text
                        ┌───────────────┐
                        │   OOM 发生     │
                        └───────┬───────┘
                                ▼
              reserved − allocated 很大且分配失败？
                    │ 是                    │ 否
                    ▼                       ▼
             碎片问题（第三章）          真容量不足（本章）
             expandable_segments        按代价从小到大：
             max_split_size_mb                │
             length bucketing                 ▼
                              ① 减 batch / 截断 seq（先验证能跑通）
                                              ▼ 还不够
                              ② activation checkpointing（+33% 算力）
                                              ▼ 还不够
                              ③ ZeRO 升级 1→2→3 / FSDP（+通信）
                                              ▼ 还不够
                              ④ TP/PP/SP 并行改造（改拓扑与启动方式）
                                              ▼ 还不够
                              ⑤ CPU/NVMe offload（带宽税最重，殿后）
```

| 手段 | 省在哪一项 | 7B/8卡量级 | 代价 | 生效层 |
|---|---|---|---|---|
| 减 batch/seq | $M_{act}$ | 线性 | 吞吐/收敛超参变化 | 训练脚本 |
| checkpointing | $M_{act}$：$O(L) \to O(\sqrt{L})$ | 30GB → ~5GB | +33% FLOPs（selective 可降至 ~5%） | 模型代码一行 |
| ZeRO 1→2 | 梯度稳态 | 38.5 → 26.25GB | 峰值不降 | 框架配置 |
| ZeRO 2→3 | 参数常驻 | 26.25 → 14GB | 通信 ×1.5，需预取调优 | 框架配置 |
| TP+SP / PP | 权重与激活按组切 | $/tp$、$/pp$ | 改启动拓扑，TP 需机内 | 框架+集群 |
| 8-bit Adam | 优化器状态 | $8\Theta \to 2\Theta$ | 精度需验证 | 优化器替换 |
| offload | 任意冷数据 | 上不封顶 | PCIe 带宽墙，吞吐大降 | 框架配置 |

到这里，"放什么"（第二章）、"怎么分"（第三章）、"怎么省"（本章）都齐了。最后一块拼图：显存**不该涨的时候涨了**怎么办——泄漏的检测与防御。

---

## 五、内存泄漏检测与避免

"训练跑了三天显存涨了 5GB"是仅次于 OOM 的高频问题。这一章先教分类（大部分"泄漏"其实不是泄漏），再逐个剖析真泄漏的七大成因，最后给出工具链和防御清单。

### 5.1 先分类：泄漏、缓存增长还是碎片

Python 有 GC，PyTorch 的 tensor 靠引用计数释放——所以 GPU "泄漏"几乎从来不是 C++ 意义上的 malloc 忘 free，而是**有一条引用链把本该死掉的 tensor 拖着不放**。诊断第一步永远是分清三种"显存上涨"：

| 类型 | 特征信号 | 本质 | 严重性 |
|---|---|---|---|
| 真泄漏 | `memory_allocated()` 逐 step **单调上涨** | 存活 tensor 越来越多 | 🔴 必须修 |
| 缓存增长 | reserved 涨、allocated 平稳 | 分配器缓存膨胀/碎片累积 | 🟡 视水位 |
| 碎片 | reserved 大 + 分配失败 | 空闲块不连续（3.4 节） | 🟡 配置可解 |

判断三板斧——在训练循环里插一段逐 step 的水位差分，跑 50 个 step 看趋势：

```python
import torch

def memory_probe(step, tag=""):
    GiB = 2**30
    alloc = torch.cuda.memory_allocated() / GiB
    reserv = torch.cuda.memory_reserved() / GiB
    if not hasattr(memory_probe, "prev"):
        memory_probe.prev = (alloc, reserv)
    d_a = alloc - memory_probe.prev[0]
    d_r = reserv - memory_probe.prev[1]
    memory_probe.prev = (alloc, reserv)
    print(f"[{step}{tag}] alloc={alloc:.3f}GiB (Δ{d_a:+.3f}) "
          f"reserved={reserv:.3f}GiB (Δ{d_r:+.3f})")

for step, batch in enumerate(loader):
    train_step(batch)
    memory_probe(step)   # Δalloc 稳定为正 → 真泄漏，进入 5.2/5.4
```

前几个 step 的上涨是正常的（Adam 状态 lazy 初始化、分配器 warmup、cuBLAS workspace），**从第 3~5 个 step 起 Δalloc 应当归零**。持续为正哪怕每步只有几 MB，跑十万 step 就是几百 GB 的死刑判决。

### 5.2 七大泄漏成因：错误代码 → 为什么 → 修复

**① 累积带计算图的 tensor（占泄漏案例的半壁江山）**

```python
# ❌ 错误：loss 是图上的节点，+= 让 total_loss 持有整个计算图
total_loss = 0
for batch in loader:
    loss = model(batch).loss
    loss.backward()
    total_loss += loss        # 引用链：total_loss → loss → 整张图 → 所有激活

# ✅ 修复：取标量，切断图引用
    total_loss += loss.item()             # 同步取值，或
    total_loss += loss.detach()           # 仍在 GPU 但已脱离图
```

为什么致命：`loss` 这个 tensor 本身只有 4 字节，但它的 `grad_fn` 链向后指向本 step 的**全部中间激活**。`total_loss += loss` 之后，autograd 释放机制（backward 完成即释放图）被这条外部引用否决，每个 step 的激活全量滞留——这是"每步稳定涨几百 MB 到几 GB"最常见的原因。

**② 日志/列表里存 tensor**

```python
# ❌ 错误：history 持有每个 step 的 GPU tensor（连带其计算图）
history.append(output)

# ✅ 修复：立刻脱离 GPU 和图
history.append(output.detach().cpu())    # 需要值就搬 CPU
history.append(output.item())            # 标量直接取数
```

**③ 梯度未清理，或置零而非置 None**

```python
# ❌ 次优：置零——梯度 tensor 仍占 2Θ 显存，且多跑一次写零 kernel
optimizer.zero_grad(set_to_none=False)

# ✅ 正确：置 None——梯度显存立即归还缓存池，峰值错开
optimizer.zero_grad(set_to_none=True)    # PyTorch 2.x 起已是默认
```

严格说这不是泄漏而是恒定多占 $2\Theta$（7B 即 14GB 的峰值差），但"忘了调 zero_grad"就是真事故：梯度跨 step 累加，数值错误比显存问题更早爆发。

**④ 评估阶段忘关 autograd**

```python
# ❌ 错误：eval 也在建图，激活全保存，而且永远没有 backward 来释放它
for batch in val_loader:
    out = model(batch)

# ✅ 修复：
with torch.inference_mode():          # 或 torch.no_grad()
    for batch in val_loader:
        out = model(batch)
```

`inference_mode()` 比 `no_grad()` 更彻底（连 version counter 都省掉），评估/推理首选。症状很典型：训练稳定、一进 validation 显存暴涨甚至 OOM。

**⑤ RNN / 截断 BPTT 未 detach hidden state**

```python
# ❌ 错误：hidden 携带上一个 step 的图，图跨 step 无限串联
for batch in loader:
    out, hidden = rnn(batch, hidden)
    loss = criterion(out, target)
    loss.backward()                   # 图越拖越长，显存线性增长

# ✅ 修复：每个截断窗口切断历史
    hidden = hidden.detach()
```

Transformer 时代的等价物：把上一个 step 的 KV cache / memory bank 不 detach 就喂给下一个 step。

**⑥ 异常帧 / 闭包 / hook 持有引用**

```python
# ❌ 隐蔽：except 帧引用了局部变量，traceback 链住整张图
try:
    out = model(batch)          # 假设这里 OOM
except Exception as e:
    logger.error(e)
    self.last_error = e         # e.__traceback__ → 栈帧 → out → 图！

# ✅ 修复：不要长期持有异常对象；必要时显式清理
except Exception as e:
    logger.error(str(e))        # 只留字符串
    del e                       # 或确保 e 不逃逸出 except 块
```

同族问题：`register_forward_hook` 里把 `output` 存进外部容器忘了 detach；lambda 闭包捕获了大 tensor。这类泄漏 snapshot 工具（5.4）几乎是唯一解法——引用链藏在 Python 帧对象里，肉眼很难读出来。

**⑦ 自定义 autograd.Function 误存 tensor**

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # ❌ 错误：绕过 save_for_backward 直接挂在 ctx 上
        ctx.x = x                        # 逃过引用检查与释放管理
        # ✅ 正确：
        ctx.save_for_backward(x, weight) # 参与 autograd 生命周期管理
        return x @ weight
```

`save_for_backward` 会在 backward 结束后自动释放，且能检测 in-place 修改；直接 `ctx.x = x` 则把 tensor 的生死交给 ctx 对象本身，一旦图释放路径异常（如 retain_graph、异常退出），tensor 就滞留了。

### 5.3 缓冲区与进程级问题

有些"显存去哪了"不在训练循环里，而在进程和库这一层：

| 来源 | 现象 | 说明与对策 |
|---|---|---|
| DataLoader `pin_memory` + `persistent_workers` | **CPU 内存**（pinned）持续占用 | 这是 RAM 不是显存；worker 常驻是特性。CPU 内存告警时先查 `num_workers × prefetch_factor × batch 大小` |
| NCCL 通信 buffer | 每个通信组固定占几百 MB~GB，不归还 | 属于 $M_{overhead}$；通信组建得越多占越多，避免频繁新建 process group |
| 多进程退出未清理 | 训练崩了但 `nvidia-smi` 显示显存仍被占 | 僵尸进程持有 CUDA context；`fuser -v /dev/nvidia*` 找到残留 PID 处理 |
| CUDA Graph 私有池 | capture 过 graph 后 reserved 明显变高 | graph 的内存池独立于普通缓存池且不受 `empty_cache` 管理，属设计行为 |

### 5.4 检测工具链：从水位监控到分配栈定位

**工具一：`torch.cuda.memory_summary()`** ——一眼看全三水位 + 碎片指标：

```python
print(torch.cuda.memory_summary())
# 重点看三行：
#   Allocated memory  当前/峰值      → 真实占用与毛刺
#   Reserved memory   当前/峰值      → 缓存池规模
#   Non-releasable memory            → 部分占用 Segment 里锁死的空闲块，
#                                      这个数字大 = 碎片严重
```

**工具二：memory snapshot 三步法（定位泄漏的终极武器）**。原理：让分配器记录每一次 alloc/free 及其完整 Python 调用栈，导出后在官方可视化页面上看时间线——**"只涨不落"的那条颜色带对应的分配栈，就是泄漏点**。

```python
# 步骤 1：训练开始前，开启记录
torch.cuda.memory._record_memory_history(max_entries=100000)

# 步骤 2：跑若干 step 复现显存增长
for step, batch in enumerate(loader):
    train_step(batch)
    if step == 50:
        break

# 步骤 3：导出快照，拖入 https://pytorch.org/memory_viz 查看
torch.cuda.memory._dump_snapshot("snap.pickle")
torch.cuda.memory._record_memory_history(enabled=None)  # 关闭记录
```

memory_viz 页面提供两个视图：**Active Memory Timeline**（横轴时间、纵轴显存，每个色块是一个存活 tensor，跨越多个 step 仍不消失的色块即嫌疑人）和 **Allocator State History**（Segment 级布局，可直观看碎片）。点击色块即可看到它的分配调用栈，直接定位到代码行。

**工具三：`torch.profiler` 的内存视角**——顺带拿到算子级归因：

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA],
             profile_memory=True, record_shapes=True) as prof:
    train_step(batch)
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage", row_limit=10))
```

**工具四：Python 侧引用排查**——snapshot 告诉你"哪行代码分配的 tensor 没释放"，但"谁引用着它"要靠 GC 侧：

```python
import gc, torch

# 扫出所有存活的 GPU tensor，按大小排序看大头
tensors = [o for o in gc.get_objects()
           if torch.is_tensor(o) and o.is_cuda]
for t in sorted(tensors, key=lambda t: -t.numel())[:10]:
    print(t.shape, t.dtype, t.element_size() * t.numel() / 2**20, "MiB")

# 对嫌疑 tensor 用 objgraph 画引用链（pip install objgraph）
# import objgraph; objgraph.show_backrefs([suspect], max_depth=4)
```

**工具五：进程外监控**——不侵入训练代码的基线手段：

```bash
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1
# 或者粗暴一点：
watch -n 1 nvidia-smi
```

### 5.5 排查流程图

```text
                        显存持续增长
                             │
                             ▼
              逐 step 打印 allocated / reserved（5.1 探针）
                             │
                 allocated 单调上涨？
                │ 是                        │ 否
                ▼                           ▼
           【真泄漏】                  reserved 上涨？
   _record_memory_history()        │ 是              │ 否
   → 复现 → _dump_snapshot()       ▼                 ▼
   → memory_viz 时间线找      【缓存/碎片】      【进程外占用】
     "只涨不落"的分配栈       Non-releasable 大？  nvidia-smi 查
        │                     │是→碎片：          其他进程 /
        ▼                     │  expandable_      僵尸 context /
   按 5.2 七大成因对号：       │  segments        NCCL 组泄漏
   ①累积图 ②存tensor          │否→缓存膨胀：
   ③梯度 ④no_grad             │  阶段切换处
   ⑤detach ⑥异常帧            │  empty_cache()
   ⑦save_for_backward         ▼
```

### 5.6 防御性编程清单

与其事后排查，不如让泄漏无处发生。训练循环十条军规：

| # | 军规 | 防的是 |
|---|---|---|
| 1 | 记录指标一律 `.item()` / `.detach().cpu()` | 成因①② |
| 2 | `zero_grad(set_to_none=True)`（确认框架默认） | 成因③ |
| 3 | 评估/推理必包 `inference_mode()` | 成因④ |
| 4 | 跨 step 传递的状态（hidden/cache）显式 `detach()` | 成因⑤ |
| 5 | except 块不让异常对象逃逸，hook 内先 detach 再存 | 成因⑥ |
| 6 | 自定义 Function 只用 `save_for_backward` 存图内 tensor | 成因⑦ |
| 7 | 数据做 length bucketing / padding 对齐，减少块尺寸抖动 | 碎片 |
| 8 | 长跑任务默认 `expandable_segments:True` | 碎片 |
| 9 | 每 N step 记录三水位 + `max_memory_allocated()` 后复位，趋势入监控 | 一切（见第六章） |
| 10 | 上线前用 snapshot 跑 100 step 做一次"显存体检" | 一切 |

---

## 六、生产环境监控与最佳实践

前五章的知识最终要落成两件事：**上线前算得准，运行中看得见**。

### 6.1 训练启动前的显存预算 checklist

用第二章的公式做纸面演算，再留安全余量——这五分钟能省掉之后几小时的 OOM 调试：

| 步骤 | 动作 | 7B/8卡算例 |
|---|---|---|
| 1 | 静态项：$4\Theta + 12\Theta/N$（按 ZeRO **峰值**记账，不按稳态） | 38.5 GB |
| 2 | 激活项：按 micro-batch/seq/是否重算估 $M_{act}$，宁高勿低 | 10~30 GB |
| 3 | 隐性项：$M_{overhead}$（context+NCCL+workspace）拍 3 GB 起步 | ~3 GB |
| 4 | 合计后检查 **≤ 容量 × (85%~90%)**，留 10~15% 余量给毛刺与碎片 | ≤ 68~72 GB ✅ |
| 5 | 短跑冒烟测试：真实配置跑 20 step，`max_memory_allocated()` 对账 | 预算误差 <10% 才放行 |

余量为什么是 10~15%：碎片会吃掉几个 GB 的"名义空闲"（3.4），backward 中段的峰值毛刺高于稳态（3.3），变长数据偶尔来一条超长样本——三者叠加，贴着 100% 排的预算必炸。

### 6.2 运行中监控：把三水位送进 wandb/tensorboard

```python
import torch

def log_memory_stats(step, logger, interval=50):
    if step % interval:
        return
    GB = 2**30
    logger.log({
        "mem/allocated_gb": torch.cuda.memory_allocated() / GB,
        "mem/reserved_gb":  torch.cuda.memory_reserved() / GB,
        "mem/peak_gb":      torch.cuda.max_memory_allocated() / GB,
    }, step=step)
    torch.cuda.reset_peak_memory_stats()  # 复位，让 peak 反映"本窗口"峰值
```

复位峰值统计是关键细节：不复位的话 `max_memory_allocated()` 是开机以来的历史最高，曲线永远平的，看不出趋势；按窗口复位后，它变成"最近 50 step 的峰值"，异常毛刺立刻现形。告警阈值的经验设定：

| 指标 | 告警条件 | 含义 |
|---|---|---|
| allocated | 连续 3 个窗口上涨 | 疑似真泄漏（第五章流程启动） |
| peak | > 容量 × 90% | 距 OOM 一步之遥，主动降级 |
| reserved − allocated | 持续爬升 且 > 10GB | 碎片累积，考虑 expandable_segments |

### 6.3 OOM 事后处理：为什么 catch 住通常救不回

一个流传很广但基本无效的模式：

```python
# ❌ 常见但通常救不回的写法
try:
    train_step(batch)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()   # 只还完全空闲的 Segment，碎片纹丝不动
    train_step(batch)          # 大概率原地再炸
```

三个原因：**其一**，OOM 时刻的空闲显存以碎片形态锁在部分占用的 Segment 里，`empty_cache()` 还不出来（3.5）；**其二**，异常抛出点在前向/反向中途，模型的梯度桶、通信状态、autograd 图停在**不一致的中间态**，就地重试的数值正确性无人担保；**其三**，异常对象本身的 traceback 还引用着现场 tensor（5.2 成因⑥），显存反而更紧。分布式场景更糟：一个 rank 在重试、其他 rank 在集合通信里等它，整个 job 挂死。

正确姿势是**记录现场、干净退出、降级重启**：

```python
except torch.cuda.OutOfMemoryError:
    torch.cuda.memory._dump_snapshot(f"oom_rank{rank}.pickle")
    print(torch.cuda.memory_summary())
    raise   # 让进程退出，交给调度层按更小的 batch/更高的 ZeRO 级重启
```

配合 checkpoint 断点续训，损失的只是几分钟；而"带病重试"损失的可能是三天训练的数值正确性。

### 6.4 全文最佳实践 Top 10

| # | 实践 | 依据章节 |
|---|---|---|
| 1 | 启动前用 $M \approx 4\Theta + 12\Theta/N + M_{act} + M_{overhead}$ 做纸面预算，留 10~15% 余量 | 二、6.1 |
| 2 | 预算按**峰值**记账：ZeRO-2 看齐 Stage-1，看 `max_memory_allocated()` | 二、3.3 |
| 3 | 判断问题先分三水位：allocated（泄漏）/ reserved（缓存碎片）/ nvidia-smi（进程外） | 三、五 |
| 4 | 长跑任务默认 `expandable_segments:True`，变长数据做 bucketing | 3.5 |
| 5 | 显存不够时按代价排序动手：batch → checkpointing → ZeRO → 并行 → offload | 4.5 |
| 6 | checkpointing 用 `use_reentrant=False`，优先 selective 重算 | 4.1 |
| 7 | TP+SP 锁机内 NVLink 域，DP/PP 跨机走 IB——通信密度决定物理位置 | 4.3 |
| 8 | 训练循环遵守十条军规：`.item()` 记录、`inference_mode()` 评估、`set_to_none=True` | 5.6 |
| 9 | 三水位按窗口入监控，peak 复位统计，三窗口连涨即告警 | 6.2 |
| 10 | OOM 不就地重试：dump snapshot → 干净退出 → 降级重启 | 6.3 |

---

## 附录A：术语与单位速查

**单位换算**：

| 项 | 定义 | 易错点 |
|---|---|---|
| GiB vs GB | $2^{30}$ B vs $10^9$ B，差 7.4% | PyTorch 报错用 GiB，营销容量用 GB；80GB 卡 ≈ 74.5 GiB |
| GB/s vs Gbps | 字节 vs 比特，×8 | 400Gbps IB ≈ 50GB/s 理论 |
| bytes/param | fp32=4，bf16/fp16=2，fp8=1，int4=0.5 | 心算显存的乘数基础 |

**三水位定义**（第三章）：

| 水位 | 含义 |
|---|---|
| allocated | 存活 tensor 实际占用（判断泄漏看它） |
| reserved | PyTorch 缓存池总量 = allocated + 空闲缓存块 |
| nvidia-smi used | reserved + CUDA context + NCCL/驱动 + 其他进程 |

**常用环境变量**：

| 变量 | 用途 |
|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | 分配器调参：`expandable_segments:True` / `max_split_size_mb` / `roundup_power2_divisions` |
| `CUDA_VISIBLE_DEVICES` | 限定进程可见的 GPU |
| `NCCL_DEBUG=INFO` | 打印 NCCL 初始化与通信细节，排查通信 buffer 与拓扑 |

**本文缩写表**：

| 缩写 | 全称 | 一句话 |
|---|---|---|
| HBM | High Bandwidth Memory | 3D 堆叠显存，~2TB/s，容量是稀缺项 |
| DMA | Direct Memory Access | 不经 CPU 的内存搬运引擎，pinned memory 的服务对象 |
| DDP | Distributed Data Parallel | 每卡全量副本 + 梯度 AllReduce |
| ZeRO | Zero Redundancy Optimizer | 训练状态分片：Stage 1/2/3 递进切优化器/梯度/参数 |
| FSDP | Fully Sharded Data Parallel | PyTorch 原生 ZeRO-3 实现 |
| TP / PP | Tensor / Pipeline Parallel | 层内切矩阵 / 按层切段 |
| SP / CP | Sequence / Context Parallel | 激活沿序列维切分（TP 域内 / 跨卡 attention） |
| EP | Expert Parallel | MoE expert 权重分卡 + all-to-all 路由 |

---

## 总结

回收开篇的地图，现在每一站都有了答案：

```text
┌────────────────────────────────────────────────────────────┐
│ 放什么   静态五件套 38.5GB + 激活 10~30GB + 隐性 ~3GB       │
│    ▼     M ≈ 4Θ + 12Θ/N + M_act + M_overhead               │
│ 怎么分   caching allocator：池化复用，代价是碎片与缓存      │
│    ▼     allocated ≤ reserved ≤ nvidia-smi                 │
│ 怎么省   batch → checkpointing → ZeRO → 并行 → offload     │
│    ▼     一切优化 = 用算力/通信/带宽/精度换容量             │
│ 怎么查   三水位分类 → snapshot 定位栈 → 七大成因对号        │
└────────────────────────────────────────────────────────────┘
```

**显存里有什么**：每个 rank 的 80GB 里，静态五件套（bf16 参数 + 梯度 + sharded fp32 主权重 + Adam 状态，7B/8 卡 38.5GB）签长租约；激活值（10~30GB）随 step 潮汐涨落，是唯一的弹性区；隐性开销（context/NCCL/workspace/分配器缓存，~3GB+）解释了 `nvidia-smi` 与理论账的全部差值。一条公式 $M \approx 4\Theta + 12\Theta/N + M_{act} + M_{overhead}$ 走天下——但记住按峰值记账，不按稳态。

**怎么省**：所有优化都是交换——checkpointing 用 33%（selective 只要 5%）的算力把激活从 $O(L)$ 压到 $O(\sqrt{L})$；ZeRO 用通信把 $16\Theta$ 的训练状态摊到 $N$ 卡（Stage 1/2 通信免费应默认开启，Stage 3 付 1.5 倍通信税换 $16\Theta/N$）；TP/PP/SP 用拓扑约束换更细的切分；offload 用两个数量级的带宽损失换无限容量。动手次序永远从便宜的开始：减 batch → checkpointing → ZeRO 升级 → 并行改造 → offload。

**怎么查**：显存上涨先分三类——allocated 单调涨是真泄漏（九成是某条引用链拖住了计算图，`.item()`/`detach()`/`inference_mode()` 三件套先自查），reserved 涨而 allocated 平是缓存或碎片（`expandable_segments:True` 是现代答案），都不涨则去进程外找。真泄漏用 `_record_memory_history()` + snapshot + memory_viz 三步定位分配栈，比任何肉眼 review 都快。

显存管理没有魔法，只有会计学：**每一个字节都有名有姓、有生命周期、有存放理由**。把账算清楚，OOM 就从玄学变成算术。

**延伸阅读**：

- [LLM训练全解](/self/2026-06-09-training/)——本文的"表"：前向/反向/优化器为什么需要这些数据；
- [LLM 领域，你需要知道的数字](/self/2026-06-11-llm-panorama/)——并行策略的通信量视角与集群数字感；
- [GPU 显存 Offload 技术](/self/2026-05-28-gpu-memory-offload-techniques/)——"容量无限、带宽骤降"路线的完整展开；
- [KV Cache 是算还是传？](/self/2026-06-10-kvcache-compute-vs-transfer/)——推理侧的显存-带宽权衡与 Roofline 建模。
