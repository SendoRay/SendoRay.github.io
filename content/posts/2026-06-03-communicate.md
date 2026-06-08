---
title: AI 基础设施通信技术全景：从 DMA 原理到分布式推理
date: '2026-06-03'
tags:
- Engineering

draft: false
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

> 从一台机器的硬件结构出发，逐步讲清 DMA、Pinned Memory、Zero-Copy 等底层原理，再纵览 AI 基础设施通信的五层架构：硬件互连 → 节点间网络 → 通信库 → 底层框架 → 上层应用。

---

## 全局架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│  第五层：上层框架与生态                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  vLLM    │  │  SGLang  │  │ TRT-LLM  │  │  Dynamo  │  LMCache     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
├───────┼──────────────┼─────────────┼─────────────┼──────────────────────┤
│  第四层：底层传输框架                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                              │
│  │   UCX    │  │   UCC    │  │ NVSHMEM  │  (MPI)                       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                              │
├───────┼──────────────┼─────────────┼────────────────────────────────────┤
│  第三层：通信库 (xCCL)                                                    │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐               │
│  │  NCCL  │ │ MSCCL++│ │  NIXL  │ │ DeepEP │ │  RCCL  │  oneCCL Gloo │
│  │(训练)   │ │(可编程)│ │(推理)  │ │ (MoE)  │ │ (AMD)  │               │
│  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘               │
├───────┼──────────┼──────────┼──────────┼──────────┼─────────────────────┤
│  第二层：节点间网络                                                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │  InfiniBand    │  │    RoCEv2      │  │    iWARP       │            │
│  │  (原生RDMA)    │  │  (以太网RDMA)  │  │  (TCP/IP RDMA) │            │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘            │
├───────────┼───────────────────┼───────────────────┼─────────────────────┤
│  第一层：芯片与单机内部互连                                               │
│  ┌──────┐  ┌────────────┐  ┌─────┐  ┌───────────────────┐             │
│  │ PCIe │  │NVLink/NVSw │  │ CXL │  │ GPUDirect RDMA    │             │
│  └──────┘  └────────────┘  └─────┘  └───────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

  
**一句话理解每一层的作用：** 硬件决定带宽上限，网络决定延迟下限，通信库决定可达性能，上层框架决定业务价值的兑现效率。

---

## 基础篇：一台机器长什么样

```
┌─────────────────────────────────────────────────────────────────────┐
│                        一台物理机（一个节点）                         │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │                    主板 (Motherboard)                    │       │
│   │                                                          │       │
│   │   ┌──────────┐        ┌──────────────────────────────┐  │       │
│   │   │   CPU    │        │       内存条 (DRAM)           │  │       │
│   │   │  (大脑)  │◄──────►│    CPU 能直接读写的地方       │  │       │
│   │   │          │  内存总线│       DDR5, 64~512GB         │  │       │
│   │   └────┬─────┘        └──────────────────────────────┘  │       │
│   │        │                                                  │       │
│   │        │  PCIe 总线（高速公路，连接各种外设）             │       │
│   │   ─────┼──────────────────────────────────────           │       │
│   │        │         │              │            │            │       │
│   │   ┌────┴───┐ ┌───┴────┐ ┌──────┴───┐ ┌─────┴────┐      │       │
│   │   │  GPU   │ │  NIC   │ │  NVMe    │ │  其他    │      │       │
│   │   │(显卡)  │ │ (网卡) │ │  (硬盘)  │ │  外设    │      │       │
│   │   │ HBM内存│ │        │ │          │ │          │      │       │
│   │   └────────┘ └────────┘ └──────────┘ └──────────┘      │       │
│   └─────────────────────────────────────────────────────────┘       │
│                                                                      │
│   多台这样的机器用网线连起来 = 集群                                   │
└─────────────────────────────────────────────────────────────────────┘
```

**关键认知：**

- CPU 能直接读写的只有 DRAM（主内存）
- GPU/NIC/NVMe 都是"外设"，挂在 PCIe 总线上
- CPU 不能直接用指针访问 GPU HBM / NVMe 里的数据

---

## 基础篇：DMA（最重要的底层机制）

### PIO 时代：CPU 全程搬运

```
问题场景：CPU 想把 DRAM 里的数据搬到网卡发出去

  DRAM                CPU                 NIC
  ┌──────┐           ┌─────┐            ┌──────┐
  │data  │  读一个字 │     │  写一个字  │      │
  │[0]   │──────────►│ 寄 │───────────►│ buf  │
  │[1]   │  读一个字 │ 存 │  写一个字  │      │
  │[2]   │──────────►│ 器 │───────────►│      │
  │[3]   │    ...    │     │    ...     │      │
  └──────┘           └─────┘            └──────┘

  CPU 全程参与每一个字节的搬运！
  1GB 数据 → CPU 忙几秒钟 → 浪费！
```

### DMA 出现后：CPU 解放

```
DMA = Direct Memory Access（直接内存访问）

核心思想：加一个"搬运工"—— DMA Controller
CPU 只需要"下命令"，具体搬运由 DMA Controller 完成

  CPU 说：                DMA Controller 执行：
  "把 DRAM 地址 0x1000   ┌──────────────────────┐
   开始的 1GB 数据        │ 自己去 DRAM 读数据   │
   搬到 NIC 的缓冲区"     │ 自己写到 NIC         │
        │                 │ 搬完了通知 CPU(中断) │
        │ 下完命令就去     └──────────────────────┘
        │ 干别的了
        ▼
      CPU 解放！

  DRAM              DMA Controller           NIC
  ┌──────┐         ┌────────────────┐      ┌──────┐
  │data  │◄───────►│ 地址寄存器     │─────►│ buf  │
  │      │  总线访问│ 长度寄存器    │      │      │
  │      │         │ 方向寄存器     │      │      │
  └──────┘         └────────────────┘      └──────┘
                         ▲
                    CPU 写这几个寄存器 = "下命令"
```

### DMA 的本质

```
┌─────────────────────────────────────────────────────────────┐
│  让外设（NIC/GPU/NVMe）能够                                 │
│  直接访问主内存（DRAM），                                    │
│  而不需要 CPU 一个字节一个字节地中转                         │
│                                                             │
│  DMA Controller 本质上就是：                                │
│  一个能发起内存总线事务的小处理器                            │
│  跟 CPU 访问内存的方式完全一样，只是发起者不是 CPU           │
└─────────────────────────────────────────────────────────────┘
```

现代系统里，DMA Controller 通常内置在外设里：

- **NIC** 内置 DMA → NIC 自己去读写 DRAM
- **NVMe** 内置 DMA → NVMe 自己去读写 DRAM
- **GPU** 内置 DMA（叫 Copy Engine）→ GPU 自己去读写 DRAM

---

## 基础篇：Pinned Memory（Zero-Copy 的前提）

### 为什么 DMA 不能直接访问普通内存

```
问题：DMA 需要物理地址，但程序用的是虚拟地址

  程序的虚拟地址空间：        实际物理内存：
  0x0000 ┌──────────┐        ┌──────────┐ 物理页 A
         │  page 1  │───────►│  在这里  │
  0x1000 ├──────────┤        ├──────────┤ 物理页 B
         │  page 2  │  ┌───► │  在这里  │
  0x2000 ├──────────┤  │     ├──────────┤
         │  page 3  │  │     │  (被换出 │ ← 可能在磁盘上！
         └──────────┘  │     │   到磁盘)│
                       │     └──────────┘

  OS 可以随时把某页换出到磁盘 → 物理地址变了
  DMA 正在搬运时如果页被换出 → 读到错误数据 → 崩溃！
```

### Pinned Memory 解决方案

```
普通内存（Pageable）：
  malloc() 分配 → OS 可以随时换页 → DMA 不安全
  ┌──────────┐
  │  page    │ ← OS说：我可以把你换到磁盘
  └──────────┘

Pinned Memory（页锁定内存）：
  cudaMallocHost() 或 mlock() 分配
  告诉 OS：这块内存不许换出，永远在物理内存里
  ┌──────────┐
  │  page 📌 │ ← 钉在这里，OS不能动
  └──────────┘
  物理地址固定 → DMA 可以安全访问 ✅

代价：占用物理内存，OS 不能灵活调度
```

### 有无 Pinned Memory 的对比

```
没有 Pinned Memory（两次拷贝）：

  DRAM (pageable)    DRAM (pinned临时)      GPU HBM
  ┌─────────────┐    ┌─────────────┐       ┌─────────┐
  │你的数据     │───►│  临时缓冲区  │──────►│         │
  │(普通malloc) │CPU │(CUDA偷偷创建)│  DMA  │         │
  └─────────────┘拷贝└─────────────┘       └─────────┘
      第一次(慢)         第二次(快)

有 Pinned Memory（一次拷贝）：

  DRAM (pinned)                              GPU HBM
  ┌─────────────┐                           ┌─────────┐
  │你的数据 📌  │──────────────────────────►│         │
  │             │        DMA直接搬           │         │
  └─────────────┘                           └─────────┘
       只有一次拷贝，CPU不参与 ✅
```

### DMA 登记一览

| 设备 | DMA 访问的内存 | 如何登记 |
|------|----------------|----------|
| GPU (Copy Engine) | CPU DRAM（必须 pinned） | `cudaMallocHost()` |
| NIC (DMA Engine) | CPU DRAM（必须 pinned） | 普通 TCP: 内核处理；RDMA: `ibv_reg_mr()` |
| NVMe SSD | CPU DRAM（必须 pinned） | 内核 NVMe 驱动处理 |
| NIC → GPU HBM | GPU HBM（跨设备） | GPUDirect RDMA + nvidia-peermem |

---

## 基础篇：Zero-Copy 到底是什么

### 定义

数据从 A 到 B，中间 **没有任何额外的内存拷贝**（0 次 CPU 参与的拷贝，不是说数据完全没有移动）。

### 发送文件的对比

```
传统方式（4次拷贝）：
  NVMe       内核缓冲区      用户空间        内核Socket缓冲区     NIC
  ┌────┐    ┌──────────┐   ┌──────────┐    ┌──────────────┐   ┌────┐
  │文件│───►│page cache│──►│  你的    │───►│   socket     │──►│发送│
  └────┘    └──────────┘   │  buffer  │    │   buffer     │   └────┘
            ①DMA拷贝       ②CPU拷贝        ③CPU拷贝           ④DMA拷贝

sendfile() Zero-Copy（2次拷贝）：
  NVMe       内核缓冲区                                          NIC
  ┌────┐    ┌──────────┐                                       ┌────┐
  │文件│───►│page cache│──────────────────────────────────────►│发送│
  └────┘    └──────────┘                                       └────┘
            ①DMA拷贝          数据从没进入用户空间！             ②DMA拷贝
```

### GPU 场景的 Zero-Copy

```
有 CPU bounce（老方式）：3次拷贝
  GPU HBM ──► CPU staging ──► CPU socket buf ──► NIC
       ①PCIe DMA    ②CPU拷贝         ③DMA
                        ↑ CPU 被迫参与

GPUDirect RDMA（Zero-Copy）：1次拷贝
  GPU HBM ──────────────────────────────────► NIC
                    ①PCIe DMA
          NIC 直接读 GPU HBM，CPU 完全不在数据路径上！
```

---

## 第一层：芯片与单机内部互连

### PCIe（Peripheral Component Interconnect Express）

主板上的通用高速扩展总线。在 AI Infra 场景下扮演两个关键角色：

- **CPU ↔ GPU** 的数据通路：模型权重加载、调度指令下发都经过 PCIe
- **GPU ↔ 网卡** 的数据通路：GPUDirect RDMA 让网卡直接通过 PCIe 读写 GPU 显存

当前主流 PCIe 5.0（x16 双向约 128 GB/s），PCIe 6.0 进一步翻倍。

### NVLink / NVSwitch

NVIDIA 专有的节点内 GPU 高速互连技术：

- **NVLink**：连接两个 GPU 的高速链路
- **NVSwitch**：节点内 GPU 互联交换芯片，实现全互联拓扑

关键数据：H100 节点 NVLink 合计单向 **900 GB/s** vs PCIe 5.0 x16 的 **32 GB/s**，差距接近 30 倍。

> 做性能分析前，第一步用 `nvidia-smi topo -m` 确认实际互联拓扑。DGX H100 是 8 卡 NVSwitch 全互联；普通云 GPU 实例可能只有 PCIe。

### CXL（Compute Express Link）

PCIe 的缓存一致性升级版。核心升级点：CPU 和加速器可以共享同一块内存地址空间，不需要显式 DMA 拷贝。目前处于快速商业化阶段，是下一代 AI 集群的重要方向。

### GPUDirect RDMA

消除 GPU 显存 → CPU 内存 → 网卡的中间拷贝，让数据路径变为：

```
GPU 显存 → PCIe → 网卡 → 网络
```

CPU 内存完全不经过，是当前大规模 AI 训练/推理集群的标配技术。

### GDRCopy

基于 GPUDirect RDMA 的用户态库，CPU 通过 BAR 映射直接以极低延迟读写 GPU 显存。是 NVSHMEM 的重要依赖，间接影响 DeepEP 等库的性能。

---

## 第二层：节点间网络

### RDMA 本身

**RDMA（Remote Direct Memory Access）**：一台主机的网卡直接读写另一台主机的内存，完全绕过两端的 CPU 和 OS 内核。

- 延迟：微秒级
- 吞吐：逼近网卡物理带宽上限
- 适用范围：数据中心/集群内部，不适用于公网

RDMA 有三种网络实现载体：

### InfiniBand（IB）

为 RDMA 而生的专用高速无损网络（硬件主要来自 NVIDIA/Mellanox）：

- 原生支持 RDMA，从物理层到传输层全部优化
- 完全内核旁路：应用通过 verbs 接口直接操作网卡
- 原生无损：Subnet Manager 保证不丢包
- 当前旗舰 NDR：400 Gbps 单端口，亚微秒延迟

**地位：** HPC 超算和顶级 AI 训练集群首选，"用 IB" 常等价于 "用 RDMA"。

### RoCEv2（RDMA over Converged Ethernet）

在以太网上实现 RDMA，复用现有基础设施，成本低于 IB。

关键约束：以太网本身有损，RDMA 丢包性能断崖下跌，必须配置：

- **PFC**（Priority Flow Control）：链路级流控
- **ECN**（Explicit Congestion Notification）：端到端拥塞通知
- **DCQCN**：专为 RoCE 设计的拥塞控制算法

> 国内大多数 GPU 集群（含各大云厂商）都是 RoCE 方案。

### iWARP

TCP/IP 上实现 RDMA，无需无损网络，但延迟高（数十 µs）。AI 集群几乎不用。

### 网络方案对比

| 方案 | 典型延迟 | 带宽 | 需要无损配置 | 场景 |
|------|----------|------|:---:|------|
| InfiniBand NDR | 亚微秒 | 400 Gbps/端口 | 原生无损 | HPC/顶级 AI 集群 |
| RoCEv2 | 1–5 µs | 100–400 Gbps | 需要 | 云数据中心 AI 集群 |
| iWARP | 数十 µs | 受限于 TCP | 不需要 | 存储/兼容性场景 |
| 传统以太网 TCP/IP | 毫秒级 | 100G–800G | 不支持 RDMA | 通用/公网 |

---

## 第三层：通信库（xCCL 体系）

关键区分：

- **集合通信**（Collective）：多 GPU 同步完成一个操作（AllReduce 等）→ 训练
- **点对点传输**（P2P）：把数据从 A 搬到 B → 推理中的 KV Cache 搬运

NCCL 解决前者，NIXL 解决后者，是**分工关系**。

### 3.1 训练集合通信库

#### NCCL（NVIDIA Collective Communications Library）

AI 训练集合通信的行业事实标准。

- 实现 AllReduce、AllGather、ReduceScatter、AllToAll 等原语
- 自动检测硬件拓扑，选择最优路径（ring / tree）
- 支持通信-计算 overlap
- 支持 GPU-initiated networking（GPU 直接发起通信）

局限：算法固定，用户无法自定义。

#### 其他训练通信库

| 库 | 厂商 | 说明 |
|----|------|------|
| RCCL | AMD | NCCL API 兼容，已集成 MSCCL++ |
| oneCCL | Intel | oneAPI 生态 |
| Gloo | Meta | 通用，无硬件依赖，CPU 场景 |

### 3.2 可编程通信库

#### MSCCL++（微软研究院，2023 至今）

GPU 原语级通信栈，与原版 MSCCL 底层架构完全重建。

核心创新：提供 GPU 可直接调用的通信原语（put / get / signal / wait / flush），GPU 不需要通知 CPU，自己驱动跨节点传输。

性能对比 NCCL：集合通信平均提速 **1.7x**（最高 5.4x），SGLang 接入后 DeepSeek-V3 decode 吞吐提升 **1.31x**。

### 3.3 推理点对点传输库

#### NIXL（NVIDIA Inference Xfer Library）

专为 PD 分离推理（Prefill 和 Decode 跑在不同 GPU）设计。已集成至 vLLM、SGLang、TRT-LLM、LMCache、Dynamo。

核心问题：Prefill 算完的 KV Cache 需要传给 Decode 实例，这是推理吞吐的关键瓶颈。

核心能力：

- 统一 API 覆盖：GPU HBM → CPU DRAM → NVMe → 网络存储
- 弹性扩缩容：不重启推理服务即可增减 GPU
- 多云兼容：AWS EFA、Azure RDMA、Google Cloud

三个主要用途：

1. KV Cache 在 Prefill/Decode 实例间传输
2. 长上下文 KV Cache 分层存储（GPU → CPU → SSD 换入换出）
3. 模型权重快速分发与重分片

### 3.4 MoE 专用通信库

#### DeepEP（DeepSeek 开源，2025.02）

专为 MoE 模型的专家并行设计。DeepSeek-V3/R1 的训练和推理就是用它。

**为什么 NCCL AllToAll 不够用？**

```
标准 AllToAll：每个 GPU 给其他所有 GPU 各发一份数据
             → 流量均匀、规律

MoE dispatch：每个 token 只激活少数专家（256 选 8）
             → 流量极度稀疏且每次分布不同
             → 用均匀设计处理稀疏场景 = 大量无效传输
```

核心特性：

- **Normal Kernels**：高吞吐训练/Prefill，NVLink ~153 GB/s，RDMA ~43-47 GB/s
- **Low-Latency Kernels**：Decode 推理，dispatch 延迟低至 **163 µs**
- **Hook-based overlap**：通信-计算重叠，不占用任何 SM

### 3.5 通信库速查

| 库名 | 厂商 | 类型 | 场景 | 核心特点 |
|------|------|------|------|----------|
| NCCL | NVIDIA | 集合通信 | 训练 | 行业标准，自动拓扑优化 |
| RCCL | AMD | 集合通信 | 训练 | NCCL 兼容，集成 MSCCL++ |
| MSCCL++ | 微软 | GPU 原语通信 | 训练/推理 | GPU 自驱动，性能超 NCCL |
| NIXL | NVIDIA | 点对点传输 | 推理 | 统一异构存储 API |
| DeepEP | DeepSeek | MoE 专用 | 训练/推理 | 稀疏 All-to-All，超低延迟 |
| LLM-DataDist | 华为 | 点对点传输 | 推理 | 昇腾生态 |

---

## 第四层：底层传输框架

日常做推理不直接接触，了解概念即可。

### UCX / UCC

- **UCX**（Unified Communication X）：为上层库提供统一通信 API，自动选择 IB/RoCE/TCP/共享内存最优路径。NIXL 的 RDMA 能力通过 UCX 实现。
- **UCC**（Unified Collective Communication）：构建在 UCX 上的集合通信接口。

### NVSHMEM / IBGDA

解决的问题：**GPU 怎么自己主动发网络请求，不需要 CPU 帮忙？**

```
传统路径（有 CPU 参与）：
GPU 产生通信需求 → 通知 CPU → CPU proxy 轮询 → CPU 向 NIC 提交 → 传输
                                    ↑ CPU proxy 是瓶颈

IBGDA 路径（GPU 主动）：
GPU SM 直接向 NIC 写入请求 → 直接轮询 NIC completion queue → 传输
                    CPU 完全退出数据路径
                    小消息吞吐量提升最高 9.5x
```

- **NVSHMEM**：GPU 版 OpenSHMEM，提供 PGAS 模型——GPU 线程像访问本地内存一样读写远端 GPU 内存
- **IBGDA**：NVSHMEM 实现跨节点传输的核心机制

> DeepEP 的低延迟 Decode 内核依赖 NVSHMEM/IBGDA。

---

## 第五层：上层框架与生态

### NVIDIA Dynamo

GTC 2025 开源的分布式推理框架，在 DeepSeek-R1 上实现最高 **30x** 吞吐量提升。

核心组件：NIXL + KV 缓存管理器 + KV 感知路由器 + PD 分离调度。

### LMCache

KV Cache 复用库，跨请求、跨实例复用已计算的 KV Cache。长上下文和重复前缀场景效果显著。

### 推理引擎现状

| 框架 | 来源 | 通信集成 |
|------|------|----------|
| vLLM | UC Berkeley | NIXL |
| SGLang | CMU/LMSYS | MSCCL++ |
| TensorRT-LLM | NVIDIA | Dynamo + NIXL |

---

## 专题：通信-计算 Overlap 技术族

让 GPU 在做计算的同时并行完成通信，消除串行等待。

背景：张量并行要求每个 transformer layer 结束时做一次 AllReduce，即使 NVLink 互连，通信开销也占端到端延迟的 **10%–40%**。

### 技术对比

| 技术 | 粒度 | 主要收益 | 工程可用性 |
|------|------|----------|------------|
| **FLUX** (2024, 微软) | Kernel 级 | GEMM 与通信融合 | 开源 |
| **TokenWeave** (2025, 微软) | Token 批次级 | 延迟 -29%，吞吐 +26% | 开源，vLLM 集成 |
| **NanoFlow** (2024) | Nano-batch 级 | 三类资源并行 | 独立推理栈 |
| **TileLink** (2025) | Tile 级 | 理论最优 overlap | 学术论文 |

**TokenWeave** 思路：将 batch 的 token 切分为两个子集，子集 A 计算时子集 B 通信，同时融合 AllReduce-RMSNorm kernel（仅占 2–8 个 SM）。

---

## 速查表

| 概念 | 一句话说明 |
|------|-----------|
| PCIe | 通用高速总线，CPU↔GPU/网卡的物理通路 |
| NVLink/NVSwitch | NVIDIA 专有 GPU 间互连，带宽远超 PCIe |
| CXL | PCIe 的缓存一致性升级，CPU 与加速器共享内存 |
| GPUDirect RDMA | 网卡直接读写 GPU 显存，绕开 CPU 内存 |
| DMA | 让外设直接访问内存，不需要 CPU 中转 |
| Pinned Memory | 页锁定内存，物理地址固定，DMA 安全访问的前提 |
| Zero-Copy | 数据传输中 0 次 CPU 参与的额外拷贝 |
| RDMA | 跨机器直接内存访问，绕过 CPU 和内核 |
| InfiniBand | RDMA 原生高速无损网络，顶级 AI 集群首选 |
| RoCEv2 | 以太网上的 RDMA，需 PFC+ECN 无损配置 |
| NCCL | NVIDIA 训练集合通信标准库 |
| MSCCL++ | GPU 原语级通信栈，性能优于 NCCL |
| NIXL | 推理数据传输库，统一异构存储 API |
| DeepEP | MoE 专用通信库，稀疏 All-to-All |
| NVSHMEM | GPU 全局地址空间，直接读写远端 GPU 内存 |
| IBGDA | GPU 直接向 NIC 发请求，CPU 退出数据路径 |
| Dynamo | NVIDIA 分布式推理框架 |
| TokenWeave | Token 级计算-通信 overlap，延迟 -29% |

---



<think>
The user wants to understand same process vs different process in Python, with examples from LLM frameworks like Ray and PyTorch/Transformers, focusing on data transfer aspects.

</think>

# Python 进程详解：同进程 vs 不同进程

## 一、基础概念

### 什么是进程？

在操作系统层面，**进程是资源分配的最小单位**，每个进程拥有：
- 独立的内存地址空间
- 独立的文件描述符
- 独立的 CPU 寄存器状态

```
进程A [独立内存空间]     进程B [独立内存空间]
┌─────────────────┐     ┌─────────────────┐
│ 变量 x = 100   │     │ 变量 x = 200   │  ← 同名变量，完全独立
│ 模型权重 W     │     │ 模型权重 W     │  ← 各自加载，不共享
└─────────────────┘     └─────────────────┘
        ↑                       ↑
    PID: 1234               PID: 5678
```

---

## 二、同进程 vs 不同进程 对比

### 同进程（Same Process）

```python
import threading

model_weights = {"layer1": [1, 2, 3]}  # 共享内存中的对象

def worker():
    # 直接访问，零拷贝，极快
    print(model_weights["layer1"])

t = threading.Thread(target=worker)
t.start()
```

**关键特征：**
- 共享同一块内存地址空间
- 线程之间可以直接读写同一个变量
- 数据传输本质上是**指针传递**，几乎零开销
- Python 有 GIL（全局解释器锁），多线程无法真正并行执行 CPU 密集型任务

### 不同进程（Different Process）

```python
from multiprocessing import Process, Queue

def worker(q):
    data = q.get()  # 从队列取数据（需要序列化/反序列化）
    print(data)

q = Queue()
q.put({"tensor": [1.0, 2.0, 3.0]})  # 数据需要 pickle 序列化

p = Process(target=worker, args=(q,))
p.start()
```

**关键特征：**
- 完全隔离的内存空间
- 数据必须**序列化（pickle/protobuf）→ IPC → 反序列化**
- 有数据传输开销
- 可以绕过 GIL，真正并行

---

## 三、结合 LLM 框架深度解析

### 3.1 Ray 框架（分布式 LLM 推理核心）

Ray 是目前 LLM 推理（如 vLLM、LLaMA-Factory）的主流分布式框架。<sub index="2" url="https://qiankunli.github.io/2023/08/23/ray.html" title="通用分布式计算引擎Ray - 李乾坤的博客" snippet=""></sub>

```
Ray 架构图（LLM 推理场景）：

┌──────────────────────────────────────────────────┐
│                  Ray Cluster                      │
│                                                   │
│  ┌─────────────┐      ┌─────────────────────┐    │
│  │  Driver     │      │   Worker Node 1     │    │
│  │  Process    │      │  ┌───────────────┐  │    │
│  │  (主进程)   │─IPC─▶│  │  Actor进程   │  │    │
│  │             │      │  │  (GPU Worker) │  │    │
│  └─────────────┘      │  └───────────────┘  │    │
│                        └─────────────────────┘    │
└──────────────────────────────────────────────────┘
```

**Ray 中的同进程 vs 不同进程：**

```python
import ray

ray.init()

# ============ 不同进程：Ray Remote Function ============
@ray.remote
def load_model_shard(shard_id):
    # 这个函数运行在【独立的 Worker 进程】中
    import torch
    model = load_llama_shard(shard_id)
    return model.state_dict()  # 返回时需要序列化传回 Driver

# 调用时数据流：Worker进程 → pickle序列化 → Ray Object Store → Driver进程
result_ref = load_model_shard.remote(0)
result = ray.get(result_ref)  # ray.get 触发反序列化

# ============ 同进程：普通函数调用 ============
def local_preprocess(text):
    # 和 Driver 在同一进程，直接操作内存
    return tokenizer(text)
```

**Ray 数据传输的核心机制 —— Object Store（Plasma）：** <sub index="2" url="https://qiankunli.github.io/2023/08/23/ray.html" title="通用分布式计算引擎Ray - 李乾坤的博客" snippet=""></sub>

```
小对象 (< 100KB):
Driver → pickle → gRPC → Worker 内存   (进程间直接传输)

大对象 (>= 100KB，如模型权重/大 Tensor):
Driver → 写入 /dev/shm (共享内存) → Worker 通过 mmap 读取
                    ↑
            零拷贝！不需要真正复制数据
```

```python
import ray
import numpy as np

ray.init()

# 大 Tensor 放入 Object Store（共享内存）
big_tensor = np.random.randn(10000, 4096)  # ~300MB，模拟模型权重
ref = ray.put(big_tensor)  # 写入 Plasma Object Store

@ray.remote
def inference(tensor_ref):
    # Worker 进程通过共享内存 mmap 读取，接近零拷贝
    tensor = ray.get(tensor_ref)
    return tensor.mean()

result = ray.get(inference.remote(ref))
```

**Ray Actor（有状态进程）在 LLM 中的应用：**

```python
@ray.remote(num_gpus=1)
class LLMWorker:
    """每个 Actor 是一个独立进程，持有自己的 GPU 和模型"""
    
    def __init__(self, model_name, tp_rank):
        import torch
        from transformers import AutoModelForCausalLM
        self.rank = tp_rank
        # 模型加载在【当前 Actor 进程】的内存/显存中
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.cuda()
    
    def generate(self, input_ids):
        # 在当前进程的 GPU 上推理
        with torch.no_grad():
            return self.model.generate(input_ids)

# 创建 4 个 Worker 进程（Tensor Parallelism）
workers = [LLMWorker.remote("llama-7b", i) for i in range(4)]

# 向每个独立进程发送数据（IPC 传输）
input_ids = [[1, 2, 3, 4, 5]]
futures = [w.generate.remote(input_ids) for w in workers]
results = ray.get(futures)
```

---

### 3.2 PyTorch / Transformers（多进程训练）

**DataParallel vs DistributedDataParallel：**

```python
# ======= DataParallel：同进程，多线程 =======
import torch
import torch.nn as nn

model = nn.DataParallel(model)  # 所有 GPU 在同一个进程
# 数据分发：主线程 → 子线程（共享内存，快）
# 梯度汇聚：在主 GPU 上（同进程，直接访问）
output = model(input)  # 简单但有 GIL 瓶颈


# ======= DistributedDataParallel：不同进程，每卡一个进程 =======
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# 每个进程独立启动，有自己的内存空间
dist.init_process_group(backend="nccl")  # 用 NCCL 做进程间通信

model = DistributedDataParallel(model)

# 梯度同步：通过 NCCL（GPU 直接 P2P 通信，不经过 CPU）
# 数据传输路径：GPU0 显存 → NVLink/PCIe → GPU1 显存
loss.backward()  # 自动触发 All-Reduce 梯度同步
```

**Transformers 多进程数据加载：**

```python
from torch.utils.data import DataLoader
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# num_workers > 0 时，会 fork 出多个独立 Worker 进程来加载数据
loader = DataLoader(
    dataset["train"],
    batch_size=32,
    num_workers=4,   # 4 个独立进程并行读取数据
    pin_memory=True  # 数据放在锁页内存，加速 CPU→GPU 传输
)

# 数据流：
# Worker进程(磁盘读取) → 共享内存(shm) → 主进程(训练) → GPU显存
```

---

## 四、数据传输方式全景对比

```
┌─────────────────┬──────────────┬─────────────┬──────────────────────┐
│   场景          │  传输方式    │  速度       │  典型用法            │
├─────────────────┼──────────────┼─────────────┼──────────────────────┤
│ 同进程·同线程   │ 直接引用     │ 极快(0拷贝) │ 单卡推理             │
│ 同进程·多线程   │ 共享内存引用 │ 极快(0拷贝) │ DataParallel         │
│ 不同进程·同机   │ 共享内存/管道│ 快          │ DDP/Ray本地          │
│ 不同进程·不同机 │ 网络(NCCL   │ 中等        │ 多机DDP/Ray集群      │
│                 │ /gRPC/MPI)  │             │                      │
└─────────────────┴──────────────┴─────────────┴──────────────────────┘
```

### 数据传输的核心技术

```python
# 1. 共享内存（同机不同进程，最快）
import torch.multiprocessing as mp

tensor = torch.randn(1000, 1000)
tensor.share_memory_()  # 放入共享内存，其他进程可直接访问

# 2. pickle 序列化（通用但有开销）
import pickle
data = pickle.dumps(tensor)   # 序列化
tensor2 = pickle.loads(data)  # 反序列化（数据被复制了一份）

# 3. NCCL（GPU间直接通信，P2P，不经过CPU）
dist.all_reduce(gradient_tensor, op=dist.ReduceOp.SUM)

# 4. Ray Plasma（大对象零拷贝共享）
ref = ray.put(large_tensor)  # 写入共享内存
# 其他进程 ray.get(ref) 通过 mmap 读取，无需复制
```

---

## 五、总结：如何选择

```
需要共享大型模型权重？
    ├── 同机 → 用 共享内存 / Ray Object Store (mmap 零拷贝)
    └── 跨机 → 用 NCCL / Gloo 网络传输

需要并行计算？
    ├── IO 密集型 → 多线程（同进程）即可
    ├── CPU 密集型 → 必须多进程（绕过 GIL）
    └── GPU 计算 → DDP（每卡一个进程）+ NCCL 同步梯度

LLM 推理服务？
    └── vLLM/Ray Serve → Actor 模型（每个 Worker 独立进程）
        数据传输：请求通过 Ray IPC → Worker → 结果返回
```

---

## 六、推荐参考资料

以下是一些高质量的相关博客和文档：

**Ray 相关：**
- [Ray 分布式框架深度解析](https://qiankunli.github.io/2023/08/23/ray.html) — 包含 Object Store 和 Actor 模型的详细分析 <sub index="2" url="https://qiankunli.github.io/2023/08/23/ray.html" title="通用分布式计算引擎Ray - 李乾坤的博客" snippet=""></sub>
- [Ray 官方文档](https://docs.ray.io) — Ray Core / Ray Serve / vLLM 集成

**PyTorch 分布式：**
- PyTorch 官方：`pytorch.org/tutorials/distributed` — DDP vs DataParallel 详细对比
- [PyTorch 分布式训练原理](https://blog.csdn.net/m0_62464865/article/details/146095832) <sub index="1" url="https://blog.csdn.net/m0_62464865/article/details/146095832" title="如何使用Ray和PyTorch训练模型 - CSDN博客" snippet=""></sub>

**Python 进程/线程基础：**
- [廖雪峰 Python 教程 - 进程与线程](https://www.liaoxuefeng.com/wiki/1016959663602400) — 中文最佳入门 <sub index="8" url="https://www.liaoxuefeng.com/wiki/1016959663602400" title="简介 - Python教程 - 廖雪峰的官方网站" snippet=""></sub>
- [Python 官方中文文档](https://python.github.net.cn/) — multiprocessing 模块详解 <sub index="3" url="https://python.github.net.cn/" title="Python中文网" snippet=""></sub>

**LLM 推理优化：**
- vLLM 官方 Blog：`blog.vllm.ai` — PagedAttention + Ray 多进程架构
- HuggingFace 官方：`huggingface.co/docs/transformers` — 多进程训练 Trainer 章节



> 本文基础篇部分为原创图解，全景篇部分参考 [washing 的知乎文章](https://zhuanlan.zhihu.com/p/2019475140635801509)整理扩展。持续完善中。




