---
title: "程序员的硬核基础（四）：通信"
date: '2026-06-15'
tags:
- CS-Foundations

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

## 实战篇：RDMA 网口怎么用（mlx5 系列）

讲完协议层（IB/RoCE/iWARP），落到工程实战，开发者最先要解决三个问题：

1. **本机自测**：同一台机器、同一张网卡，能不能自己发自己收？（loopback）
2. **同机多卡**：同机插了多张 mlx5 网卡，能不能 A 卡发、B 卡收？
3. **跨机互通**：多台机器之间怎么连、怎么测、怎么压？

下面用最常见的 `mlx5_x`（Mellanox/NVIDIA ConnectX 系列）+ `perftest`（`ib_write_bw` / `ib_send_bw` / `ib_write_lat`）这套工具链给出标准答案。

### 准备：先看清楚机器上有几张 RDMA 网卡

```bash
# ① 列出所有 RDMA 设备及端口状态
ibstat
# 关键字段：
#   CA 'mlx5_0'   ← 设备名
#     Port 1:
#       State: Active            ← 必须 Active 才能用
#       Physical state: LinkUp   ← 物理链路 OK
#       Rate: 200                ← 当前协商带宽（Gbps）
#       Link layer: Ethernet     ← Ethernet=RoCE，InfiniBand=IB

# ② 一行一张卡，简洁版
ibv_devices

# ③ 查 GID（RoCEv2 通常用 index=3，IPv4-mapped IPv6）
show_gids
# 输出形如：
# DEV     PORT  INDEX  GID                                       IPv4
# mlx5_0  1     3      0000:0000:0000:0000:0000:ffff:c0a8:0101   192.168.1.1
# mlx5_1  1     3      0000:0000:0000:0000:0000:ffff:c0a8:0201   192.168.2.1

# ④ 看每张卡对应哪个内核网络接口
ls /sys/class/infiniband/mlx5_0/device/net   # 通常是 ib0 / eth0 / ens1f0...
```

> **关键认知**：`mlx5_0`、`mlx5_1` 是 RDMA verbs 层的设备名，和 Linux 网络栈的 `eth0`/`ens1f0` 一一对应但不同名。后续 perftest 命令（`-d mlx5_0`）用的是 **verbs 名**；建立连接的 IP 用的是**对应内核接口的 IP**。

### 为什么 mlx5/RoCE 必须和 IPv6「绑在一起」

第一次跑 RoCE 的人都会困惑这三个问题：

> 我集群明明是 IPv4，为什么 `show_gids` 里看到的 GID 全长得像 IPv6？
> 为什么 `-x 3` 一定要选那个「像 IPv6」的 entry？
> 为什么我把内核 IPv6 关了之后 RoCE 直接连不上？

答案藏在 RDMA 的历史里：**RoCE 的寻址模型完全继承自 InfiniBand，而 IB 的全局地址恰好是 128 位——和 IPv6 地址长度完全一样。**

#### GID = 128 bit，刚好等于 IPv6 长度

```
InfiniBand 时代（先于 RoCE）：
  每张 IB 卡都有一个 GID（Global Identifier）
  长度：128 bit          ← IB 规范定义
  作用：跨 Subnet 的全局唯一标识

  ╔══════════════ GID（128 bit）══════════════╗
  │ 64-bit Subnet Prefix │ 64-bit Interface ID │
  ╚══════════════════════╧═════════════════════╝
                         （由 MAC 派生）

RoCE 把 IB verbs 搬到以太网时，硬件/软件路径都不愿意改：
                ↓
 GID 字段保持 128 bit 不变 → 和 IPv6 地址长度天然一致
                ↓
 RoCEv2 干脆直接复用 IPv6 寻址语义 ✅
```

> **一句话**：GID 是 128 位是 IB 的历史遗产，IPv6 也是 128 位是巧合；两者长度撞车后，RoCEv2 顺水推舟把 GID 直接当 IPv6 地址用。

#### RoCEv1 vs RoCEv2：从 L2 到 L3 的关键升级

```
RoCEv1（早期，2010）：
  | Ethernet | InfiniBand GRH (含 GID) | RDMA Payload |
  ──────────────────────────────────────────────────
  ✗ 不带 IP 头，纯二层（L2）
  ✗ 不能跨子网 / 不能过路由器
  → 基本淘汰

RoCEv2（现在的事实标准，2014）：
  | Ethernet | IP | UDP(4791) | IB BTH | RDMA Payload |
  ──────────────────────────────────────────────────
  ✓ 走 UDP/IP（L3），可跨路由器、可走 ECMP
  ✓ IP 头：IPv4 / IPv6 都支持
  ✓ GID 表里同时存 IPv4-mapped GID 和原生 IPv6 GID
```

#### 为什么 IPv4 网络也「被迫」用 IPv6 形式？

`show_gids` 的输出最能说明问题：

```
DEV     PORT  INDEX  GID                                       IPv4         类型
mlx5_0  1     0      fe80:0000:0000:0000:1270:fdff:fe1c:0001                 ← RoCEv1 link-local
mlx5_0  1     1      0000:0000:0000:0000:0000:ffff:c0a8:0101   192.168.1.1   ← RoCEv1 over IPv4
mlx5_0  1     2      fe80:0000:0000:0000:1270:fdff:fe1c:0001                 ← RoCEv2 IPv6 link-local
mlx5_0  1     3      0000:0000:0000:0000:0000:ffff:c0a8:0101   192.168.1.1   ← RoCEv2 over IPv4（推荐）
```

看 Index 1 和 Index 3：GID 都是 `::ffff:192.168.1.1` —— 这是 **IPv4-mapped IPv6 地址**，IPv6 规范定义的特殊段（前缀 `::ffff:`），专门用来「用 IPv6 地址格式表达 IPv4 地址」。

```
你的视角：       192.168.1.1（普通 IPv4）
     ↓ 内核
IPv4-mapped:    ::ffff:192.168.1.1（IPv6 形式，128 bit）
     ↓ 填入 verbs
RoCE GID:       0000:0000:0000:0000:0000:ffff:c0a8:0101
     ↓
网卡硬件：只认 GID，不关心「原来」是 IPv4 还是 IPv6
```

> **关键认知**：即使你完全不用 IPv6 部署，**RoCE 协议层也强制使用 IPv6 地址格式存 GID**——只是用 `::ffff:` 前缀把 IPv4 地址「伪装」成 IPv6。这就是为什么所有 RoCE 教程都让你 `-x 3`：选 IPv4-mapped 的 RoCEv2 GID。

#### GID Index 速查（按常见顺序）

| Index | 类型 | 用法 |
|---|---|---|
| 0 | RoCEv1 / IPv6 link-local | 早期 RoCE，已基本淘汰 |
| 1 | RoCEv1 / IPv4-mapped | 早期 RoCE + IPv4 网络 |
| 2 | RoCEv2 / IPv6 link-local | 想用纯 IPv6（少见） |
| **3** | **RoCEv2 / IPv4-mapped IPv6** | **绝大多数生产环境（默认推荐）** |
| 4+ | RoCEv2 / IPv6 全局地址 | 真正的 IPv6 部署 |

> ⚠️ 不同发行版/驱动/RoCE 模式（v1-only/v2-only/dual）下 Index 顺序可能不同，**永远以 `show_gids` 实际输出为准**。NCCL 也提供 `NCCL_IB_GID_INDEX` 环境变量来覆盖默认值。

#### 致命陷阱：禁用 IPv6 = 禁用 RoCE

很多运维同学习惯把服务器 IPv6 关了「图清净」，然后 RoCE 直接寄：

```bash
# 必须为 0（启用），否则 RoCEv2 GID 注册不上
sysctl net.ipv6.conf.all.disable_ipv6
sysctl net.ipv6.conf.<iface>.disable_ipv6

# 如果是 1：show_gids 里 RoCEv2 entries 会全部消失
# QP 建连时找不到合适 GID → Failed to modify QP to RTR
```

**根因**：内核生成 RoCEv2 GID 的代码路径依赖 IPv6 子系统。即使你所有上层流量都是 IPv4，RoCE 的 GID 表也是由 IPv6 模块填充的——一旦 `disable_ipv6=1`，IPv4-mapped 的 RoCEv2 GID 都不会被注册。

```
禁用 IPv6 时的链路：
  你的应用 → ibv_post_send → 选 GID → ❌ 没有 RoCEv2 GID
                                       ↓
                              QP 状态机卡在 INIT/RTR
                                       ↓
                       Failed to modify QP to RTR
```

> **生产环境铁律**：**保持内核 IPv6 协议栈开启**（哪怕业务网段全是 IPv4）。这是 mlx5 + RoCE 在生产环境最容易踩、也最难定位的一个坑——因为运维同学通常不会想到 RDMA 跟 IPv6 有任何关系。

#### 一句话总结

```
IB 用 128 bit GID（历史） + IPv6 也是 128 bit（巧合）
                  ↓
        RoCEv2 直接复用 IPv6 寻址
                  ↓
     IPv4 流量 → IPv4-mapped IPv6 GID（::ffff:x.x.x.x）
                  ↓
   GID 表由内核 IPv6 子系统维护 → 关 IPv6 就废 RoCE
```

### 场景一：单机 Loopback（同机同卡，自己发自己收）

适用于 RDMA 软件栈/驱动连通性自测，**不需要任何远端机器**。

```
┌──────────── 一台机器 ────────────┐
│                                   │
│   ib_write_bw 进程 A (server)     │
│            │                      │
│            │ verbs                │
│            ▼                      │
│      ┌──────────┐                 │
│      │ mlx5_0   │  ← 走网卡内部   │
│      │  port 1  │   loopback 路径 │
│      └──────────┘                 │
│            ▲                      │
│            │ verbs                │
│            │                      │
│   ib_write_bw 进程 B (client)     │
│   连接 127.0.0.1                  │
└───────────────────────────────────┘
```

**两条命令搞定（开两个终端）**：

```bash
# 终端 1：起 server
ib_write_bw -d mlx5_0 -F --report_gbits

# 终端 2：起 client，连本地回环
ib_write_bw -d mlx5_0 -F --report_gbits 127.0.0.1
```

参数说明：`-F` 忽略 CPU 频率检查；`--report_gbits` 以 Gb/s 报告带宽。

> ⚠️ 同卡 loopback 的数据路径**不出网卡 SerDes**，性能数字不代表真实链路带宽，**只用来验证 verbs 栈是否畅通**。

### 场景二：同机不同网卡互通（mlx5_0 ↔ mlx5_1）

> **核心问题**：一台机器上插了两张 mlx5 卡，能不能 A 卡发、B 卡收？
>
> **答案**：**完全可以**。每张 RDMA 网卡是独立的 PCIe 设备，对操作系统而言就是两个独立的网络节点；只要两张卡在 IP 层（RoCE）或 IB Subnet 层（IB）能互通，就能正常跑 RDMA。

#### 前提：两张卡要"能看见对方"

- **RoCE 场景**：两张卡分别有 IP（例如 `192.168.10.1` 和 `192.168.10.2`），要求在同一子网 / 通过交换机互通；同机直连最常见的做法是用一根线把两张卡的口直接对插。
- **IB 场景**：两张卡需要被同一个 Subnet Manager（`opensm`）管理，端口 `State: Active` 即可。

```
┌──────────── 一台机器 ────────────┐
│                                   │
│   ib_write_bw  server  (-d mlx5_1)│
│              ▲                    │
│              │ verbs              │
│         ┌─────────┐               │
│         │ mlx5_1  │ 192.168.10.2  │
│         │ port 1  │               │
│         └────┬────┘               │
│              │  一根线直连/交换机 │
│         ┌────┴────┐               │
│         │ mlx5_0  │ 192.168.10.1  │
│         │ port 1  │               │
│         └─────────┘               │
│              │ verbs              │
│              ▼                    │
│   ib_write_bw  client  (-d mlx5_0)│
│   连接 192.168.10.2               │
└───────────────────────────────────┘
```

**测试命令**：

```bash
# 终端 1：server 用 mlx5_1
ib_write_bw -d mlx5_1 -F --report_gbits -x 3

# 终端 2：client 用 mlx5_0，连对端 mlx5_1 的 IP
ib_write_bw -d mlx5_0 -F --report_gbits -x 3 192.168.10.2
```

`-x 3` 指定 GID Index = 3（RoCEv2 IPv4-mapped IPv6），纯 IB 环境可省略。

#### 常见踩坑

| 现象 | 原因 | 解决 |
|---|---|---|
| `Couldn't connect to ...` | 两张卡 IP 不在同一子网，或没接线 | 先用 `ping <对端IP>` 验证 L3 通 |
| 带宽远低于卡能力（只有几 Gbps）| 走了内核 TCP 而非 RDMA | 加 `-R` 用 RDMA-CM 建连；确认两端 `link_layer` 一致 |
| `Failed to modify QP to RTR` | GID Index 不对（RoCE 没选 v2）| 显式 `-x 3` 指定 RoCEv2 GID |
| 两张卡 link_layer 不一致（一 IB 一 Eth）| 协议层不同，**不能直接 RDMA 互通** | 换同型号卡，或用 IPoIB/中间路由 |

> **直接结论**：同机两张 mlx5 **完全可以互通**，前提是两端 link_layer 相同（都是 IB 或都是 Ethernet/RoCE），且 IP/Subnet 层连通。**一张 IB + 一张 RoCE 的异构组合不能直接 RDMA 通信**——这是协议层硬约束。

### 场景三：多机互通（最常见的生产用法）

```
┌──── Node A ────┐                  ┌──── Node B ────┐
│                │                  │                │
│  ib_write_bw   │                  │  ib_write_bw   │
│   (client)     │                  │    (server)    │
│      │         │                  │      ▲         │
│      ▼         │     RDMA 网络    │      │         │
│  ┌────────┐    │   ┌──────────┐   │  ┌────────┐    │
│  │ mlx5_0 │────┼───┤ Switch / │───┼──│ mlx5_0 │    │
│  │  IP_A  │    │   │ 直连线   │   │  │  IP_B  │    │
│  └────────┘    │   └──────────┘   │  └────────┘    │
└────────────────┘                  └────────────────┘
```

**两步走**：

```bash
# Node B（server 端，先起）
ib_write_bw -d mlx5_0 -F --report_gbits -x 3

# Node A（client 端，连 Node B 的 IP）
ib_write_bw -d mlx5_0 -F --report_gbits -x 3 <Node_B_IP>
```

实测健康的 200G 网卡通常能跑出 **180–195 Gbps**，能稳定到 95% 以上线速就算正常。

#### 多机环境必查的 5 件事

```bash
# 1. 端口必须 Active
ibstat | grep -E "State|Rate|Link layer"

# 2. 两端 link_layer 必须一致（都 InfiniBand 或都 Ethernet）
# 3. RoCEv2 必须选对 GID Index（通常 3）
show_gids

# 4. 跨机 ping 通（RoCE 走以太网 IP）
ping <对端 IP>

# 5. MTU / PFC / ECN（RoCE 集群必查）
ip link show <iface>          # 两端 MTU 应一致，常见 4200 / 9000
# 交换机侧：PFC 优先级是否使能、ECN 是否开启、DCQCN 是否配置
```

#### 性能压测进阶

```bash
# 双向带宽（更接近真实业务）
ib_write_bw -d mlx5_0 -F -b --report_gbits <IP>

# 多 QP 并行（吃满网卡 / 多核）
ib_write_bw -d mlx5_0 -F -q 8 --report_gbits <IP>

# 测延迟（小消息，看单跳 latency）
ib_write_lat -d mlx5_0 -F <IP>

# 测 GPUDirect RDMA（数据直接出/入 GPU 显存）
ib_write_bw -d mlx5_0 --use_cuda=0 <IP>     # cuda=0 表示 GPU 0
```

### 一张表带走

| 场景 | 命令骨架 | 用途 |
|---|---|---|
| 单卡 loopback | `ib_write_bw -d mlx5_0` + `... 127.0.0.1` | 验证驱动/verbs |
| 同机异卡 | `-d mlx5_1` ↔ `-d mlx5_0 <对端IP>` | 验证两张卡硬件 + 配置 |
| 多机 | `-d mlx5_0 -x 3 <对端IP>` | 真实跨机带宽 |
| GDR | 加 `--use_cuda=N` | GPU 显存直传 |
| 双向 | 加 `-b` | 全双工压测 |
| 延迟 | `ib_write_lat` | 微秒级延迟测量 |

> **一句话原则**：先 `ibstat` 看端口 Active → 再 loopback 验证本地栈 → 再两端 perftest 验证链路 → 最后跑 NCCL / Mooncake / UCX 等上层库。出问题就按这个顺序逐层往下排。

---

## 第三层：通信库（xCCL 体系）

关键区分：

- **集合通信**（Collective）：多 GPU 同步完成一个操作（AllReduce 等）→ 训练
- **点对点传输**（P2P）：把数据从 A 搬到 B → 推理中的 KV Cache 搬运

NCCL 解决前者，NIXL 解决后者，是**分工关系**。

### 集合通信原语：六大操作图解

在深入各家通信库之前，先把"集合通信"本身的六大基础操作讲清楚——它们是 NCCL/MSCCL++/DeepEP/RCCL 这些库共同实现的语义底座。理解了这层语义，再回头看任何通信库，都只是"用什么算法、什么硬件，更高效地完成这六件事"。

约定：以下示例固定 4 张 GPU（GPU0~GPU3），每张 GPU 持有一段数据。M 表示单卡数据量，N 表示参与通信的 GPU 数量。

#### Broadcast（广播）

**语义**：一个 GPU（root）的数据 → 复制给所有 GPU。

```
初始：                       结果：
  GPU0: [A]  ← root           GPU0: [A]
  GPU1: [ ]                   GPU1: [A]
  GPU2: [ ]      ─────►       GPU2: [A]
  GPU3: [ ]                   GPU3: [A]

  数据流：GPU0 ──► GPU1
          GPU0 ──► GPU2
          GPU0 ──► GPU3
```

**典型用途**：模型权重从 rank 0 分发到所有 worker、超参广播。

#### Reduce

**语义**：所有 GPU 的数据 → 按 op（如 sum）聚合到一个 root GPU。

```
初始：                       结果（op = sum）：
  GPU0: [A]                   GPU0: [A+B+C+D]  ← root
  GPU1: [B]                   GPU1: [B]
  GPU2: [C]      ─────►       GPU2: [C]
  GPU3: [D]                   GPU3: [D]

  数据流：GPU1/2/3 的数据汇聚到 GPU0，逐元素相加
```

**典型用途**：把分散在各卡的损失/统计量收集到 rank 0 输出。

#### AllReduce

**语义**：所有 GPU 的数据 → 聚合 → 结果分发给所有 GPU。

```
初始：                       结果（op = sum）：
  GPU0: [A]                   GPU0: [A+B+C+D]
  GPU1: [B]                   GPU1: [A+B+C+D]
  GPU2: [C]      ─────►       GPU2: [A+B+C+D]
  GPU3: [D]                   GPU3: [A+B+C+D]

  逻辑等价：Reduce + Broadcast
  实际实现：Ring AllReduce 等高效算法（见下文），
            通信量比朴素 Reduce+Broadcast 少近一半。
```

**典型用途**：DDP 训练梯度同步——这是 AI 训练里出现频率最高的集合通信。

#### AllGather

**语义**：每个 GPU 各持一段 → 全部收集到所有 GPU（每个 GPU 拿到完整数据）。

```
初始：                       结果：
  GPU0: [A]                   GPU0: [A|B|C|D]
  GPU1: [B]                   GPU1: [A|B|C|D]
  GPU2: [C]      ─────►       GPU2: [A|B|C|D]
  GPU3: [D]                   GPU3: [A|B|C|D]

  与 Broadcast 区别：
    Broadcast 是 1 → N 复制同一份数据；
    AllGather  是 N → N 拼接 N 份不同数据。
```

**典型用途**：ZeRO/FSDP 中按需拉回完整权重、张量并行的输出收集。

#### ReduceScatter

**语义**：先聚合再分片——每个 GPU 拿到聚合结果的一部分。

```
初始（每卡持有 4 个 chunk）：       结果（op = sum，每卡持有 1 个 chunk）：
  GPU0: [a0|a1|a2|a3]                GPU0: [a0+b0+c0+d0]
  GPU1: [b0|b1|b2|b3]   ─────►       GPU1: [a1+b1+c1+d1]
  GPU2: [c0|c1|c2|c3]                GPU2: [a2+b2+c2+d2]
  GPU3: [d0|d1|d2|d3]                GPU3: [a3+b3+c3+d3]

  逻辑等价：Reduce 出全量结果 + Scatter 切片分发
  关键关系：AllReduce = ReduceScatter + AllGather
```

**典型用途**：ZeRO-2/3 梯度切片同步、Ring AllReduce 的前半段。

#### AllToAll

**语义**：每个 GPU 把自己的数据切成 N 段，第 i 段发给第 i 个 GPU。等价于一次"分布式矩阵转置"。

```
初始（每卡 4 段，下标 = 目的 GPU）：    结果（每卡收到 4 段，下标 = 来源 GPU）：
  GPU0: [A0|A1|A2|A3]                    GPU0: [A0|B0|C0|D0]
  GPU1: [B0|B1|B2|B3]      ─────►        GPU1: [A1|B1|C1|D1]
  GPU2: [C0|C1|C2|C3]                    GPU2: [A2|B2|C2|D2]
  GPU3: [D0|D1|D2|D3]                    GPU3: [A3|B3|C3|D3]

  GPU0 的 A1 → 发给 GPU1
  GPU0 的 A2 → 发给 GPU2
  GPU0 的 A3 → 发给 GPU3
  其他卡同理。
```

**典型用途**：MoE 专家并行的 dispatch/combine（这是 DeepEP 存在的根本原因）、序列并行重切分。

#### 通信量公式总览

以下公式假设单卡数据量为 M，GPU 数量为 N：

| 操作 | 每个 GPU 发送量 | 每个 GPU 接收量 | 总通信量 |
|---|---|---|---|
| Broadcast | M（root）/ 0 | M | M |
| Reduce | M | M（root）/ 0 | (N-1)×M |
| AllReduce | M | M | 2(N-1)/N × M |
| AllGather | M/N | M(N-1)/N | (N-1)×M |
| ReduceScatter | M(N-1)/N | M/N | (N-1)×M |
| AllToAll | M(N-1)/N | M(N-1)/N | N(N-1)×M / N |

> **关键观察**：AllReduce 总通信量 ≈ 2M（与 N 无关），这正是 Ring AllReduce 的核心红利——下面就来看它怎么做到的。

#### Ring AllReduce 算法图解

NCCL 的 AllReduce 默认实现就是 **Ring AllReduce**，把 N 张 GPU 串成一个环，分两个阶段完成：

**阶段 1：ReduceScatter（N-1 步，每步只发一个 chunk 并累加）**

```
Ring 拓扑：  GPU0 ──► GPU1 ──► GPU2 ──► GPU3 ──► GPU0

初始（每卡持有 4 个 chunk，下标对应 chunk 编号）：
  GPU0: [a0 | a1 | a2 | a3]
  GPU1: [b0 | b1 | b2 | b3]
  GPU2: [c0 | c1 | c2 | c3]
  GPU3: [d0 | d1 | d2 | d3]

Step 1：每张 GPU 把"自己负责的 chunk"沿环传给下家
  GPU0 ──[a0]──► GPU1   →  GPU1 的 b0 += a0
  GPU1 ──[b1]──► GPU2   →  GPU2 的 c1 += b1
  GPU2 ──[c2]──► GPU3   →  GPU3 的 d2 += c2
  GPU3 ──[d3]──► GPU0   →  GPU0 的 a3 += d3

Step 2：继续沿环传递"刚累加好的 chunk"
  GPU1 ──[b0+a0]──────► GPU2  →  c0 += (b0+a0)
  GPU2 ──[c1+b1]──────► GPU3  →  d1 += (c1+b1)
  GPU3 ──[d2+c2]──────► GPU0  →  a2 += (d2+c2)
  GPU0 ──[a3+d3]──────► GPU1  →  b3 += (a3+d3)

Step 3（最后一步）：
  GPU2 ──[c0+b0+a0]──► GPU3  →  d0 += ... = a0+b0+c0+d0   ✅ 完成
  GPU3 ──[d1+c1+b1]──► GPU0  →  a1 += ... = a1+b1+c1+d1   ✅ 完成
  GPU0 ──[a2+d2+c2]──► GPU1  →  b2 += ... = a2+b2+c2+d2   ✅ 完成
  GPU1 ──[b3+a3+d3]──► GPU2  →  c3 += ... = a3+b3+c3+d3   ✅ 完成

ReduceScatter 结束后，每卡都已经持有一个完整聚合的 chunk：
  GPU0: [ ?  | Σa1 | ?  | ?  ]
  GPU1: [ ?  | ?   | ?  | Σb2(=Σc3? 写法上每卡持一个 chunk) ]
  ...（每卡负责一个 chunk 的最终结果）
```

**阶段 2：AllGather（N-1 步，把各 GPU 的完成 chunk 复制传遍全环）**

```
这一阶段不再做 reduce，只是单纯把"已完成的 chunk"沿环复制：

Step 1：每张 GPU 把自己持有的最终 chunk 发给下家（覆盖对方对应位置）
Step 2：再传一步
Step 3：最后一步，全员持有完整的 [Σ0 | Σ1 | Σ2 | Σ3]

最终：所有 GPU 都拥有 [a0+b0+c0+d0 | a1+b1+c1+d1 | a2+b2+c2+d2 | a3+b3+c3+d3]
```

**为什么 Ring 高效**：每一步每张卡同时收发一个 chunk（M/N），N-1 步走完总传输量 = 2(N-1)/N × M ≈ 2M，与 N 无关；而朴素 Reduce+Broadcast 通信量是 2(N-1)×M，差距随 N 线性放大。

#### Ring vs Tree 算法对比

| 维度 | Ring | Tree |
|---|---|---|
| 步数（延迟） | 2(N-1) 步 | 2 × log₂ N 步 |
| 带宽利用 | 接近最优（每卡每步收发都打满） | 较差（根节点是瓶颈） |
| 适合场景 | 大消息（训练梯度，带宽受限） | 小消息（控制信号、低延迟优先） |
| NCCL 自动选择 | 消息 ≥ 256 KB | 消息 < 256 KB |

> **NCCL 的智能**：你只调用 `ncclAllReduce`，它会根据消息大小、拓扑（NVLink/PCIe/IB）自动选择 Ring 还是 Tree——这正是它能成为行业标准的原因之一。

#### 带宽计算实例：8 卡 H100 跑 80 GB AllReduce

场景：8 张 H100 通过 NVSwitch 全互联，单向 NVLink 带宽 450 GB/s，要做一次 80 GB（≈ 200 亿 fp32 参数的梯度）的 AllReduce。

```
参数：N = 8，M = 80 GB

① 总通信量（Ring AllReduce）
     = 2(N-1)/N × M
     = 2 × 7/8 × 80 GB
     = 140 GB

② 单卡发送量
     = 总通信量 / N
     = 140 GB / 8
     ≈ 17.5 GB

③ Ring 中每张卡同一时刻只用一个方向（顺时针发，逆时针收），
   所以单方向带宽 = 450 GB/s（不是双向 900 GB/s）

④ 理论时间下界
     = 17.5 GB / 450 GB/s
     ≈ 39 ms

⑤ 实际开销（含调度、同步、kernel launch、SM 抢占）
     ≈ 50–70 ms
```

这个数字很重要：它告诉你即使在最强 NVLink 拓扑下，一次百亿参数级别的梯度同步也要 **几十毫秒**——这就是为什么"通信-计算 overlap"（专题章节）值得专门优化。

> **钩子**：NCCL 根据消息大小自动选择 Ring（大消息，带宽优先）或 Tree（小消息，延迟优先）。而 DeepEP 面对 MoE 的 All-to-All 时，挑战完全不同——流量极度稀疏且不规则，标准 Ring/Tree 不适用，必须用定制化的稀疏通信方案。

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

### Transfer Engine（Mooncake）

专为大规模 GPU 集群的模型权重传输设计，控制层/传输层分离架构：

```
控制层（gRPC + REST）：
  mooncake_master (gRPC :50051)  ← 对象注册/发现、副本追踪
       ↕
  HTTP Metadata Server (:8083)   ← Segment 元数据、节点发现
  
  跑在普通 TCP/IP 上，传输元数据（带宽需求低）

传输层（Transfer Engine）：
  ├── RDMA (IB/RoCE)  ← 生产推荐，200Gbps+
  ├── TCP (fallback)  ← 开发测试用
  ├── GPUDirect RDMA  ← GPU 显存直接传输
  └── 多网卡带宽聚合 + 拓扑感知路径选择
  
  动态端口: 12300-14300 (数据), 15000-17000 (RPC)
```

**典型场景：** 模型权重在数十个节点间分发、KV Cache 跨节点迁移、训练检查点存储。

> 对比 NIXL：NIXL 专注推理场景的 KV Cache 传输（API 级），Transfer Engine 更底层，提供通用的高速数据传输能力（类似 UCX，但专为 AI 权重传输优化）。

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

## 专题：RPC 与 gRPC —— 跨进程通信的基石

### 什么是 RPC？

**RPC（Remote Procedure Call，远程过程调用）**：让调用远端函数像调用本地函数一样简单。

```python
# 本地函数调用（同进程）：
result = add(3, 5)  # 直接跳转执行，零开销

# RPC 远程调用（不同进程/不同机器）：
result = rpc_client.add(3, 5)  # 看起来一样，底层完全不同！
```

### RPC 的本质

```
本地调用：
  你的代码 ──────────────► 函数执行
              直接跳转，纳秒级

RPC 调用（跨进程/跨机器）：
  你的代码
    │
    ▼
  [RPC Client]  ① 参数序列化（protobuf/pickle/JSON）
    │
    ▼
  [网络传输]    ② TCP/HTTP/gRPC 发送到远端
    │
    ▼
  [RPC Server]  ③ 反序列化参数
    │
    ▼
  [函数执行]    ④ 实际执行逻辑
    │
    ▼
  [RPC Server]  ⑤ 结果序列化
    │
    ▼
  [网络传输]    ⑥ 返回结果
    │
    ▼
  [RPC Client]  ⑦ 反序列化结果
    │
    ▼
  你的代码收到 result
  
  总延迟 = 序列化 + 网络 + 执行 + 反序列化
  通常：毫秒级（同机）~ 数十毫秒（跨机）
```

### gRPC 是什么？

gRPC = Google 开源的高性能 RPC 框架。

```
gRPC 架构：

  定义接口（.proto 文件）：
  ┌──────────────────────────────────┐
  │  service LLMService {           │
  │    rpc Generate(Prompt)         │
  │      returns (Response);        │
  │  }                              │
  │                                 │
  │  message Prompt {               │
  │    string text = 1;             │
  │    int32 max_tokens = 2;        │
  │  }                              │
  └──────────────────────────────────┘
           │
     protoc 编译器自动生成
           ▼
  ┌──────────────┐      HTTP/2      ┌──────────────┐
  │  Client Stub │◄────────────────►│  Server Stub │
  │  (自动生成)   │   二进制流         │  (自动生成)   │
  └──────────────┘                  └──────────────┘
       │                                  │
       │ 你调用                           │ 你实现
       ▼                                  ▼
  client.Generate(prompt)          def Generate(request):
                                       # 你的业务逻辑
                                       return response

关键特性：
  ✅ HTTP/2 协议（多路复用，单连接并发）
  ✅ Protobuf 序列化（比 JSON 快 5-10x，体积小 3-5x）
  ✅ 自动生成客户端/服务端代码
  ✅ 支持流式传输（Server Streaming / Bidirectional）
```

### RPC/gRPC 在五层架构中的位置

```
┌─────────────────────────────────────────────────────────────────────────┐
│  第五层：上层框架与生态                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  vLLM    │  │  SGLang  │  │ TRT-LLM  │  │  Dynamo  │  LMCache     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
├───────┼──────────────┼─────────────┼─────────────┼──────────────────────┤
│  ✨ 第 4.5 层：RPC/gRPC 跨进程通信 ✨                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                              │
│  │   gRPC   │  │   Ray    │  │  FastAPI │  (HTTP/REST)                 │
│  │ (通用RPC) │  │ (IPC+RPC)│  │ (Web API)│                               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                              │
├───────┼──────────────┼─────────────┼────────────────────────────────────┤
│  第四层：底层传输框架                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐             │
│  │   UCX    │  │   UCC    │  │ NVSHMEM  │  │  Transfer    │  (MPI)       │
│  │          │  │          │  │          │  │   Engine     │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │ (Mooncake)  │              │
       │              │              │        └──────────────┘
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

**关键理解：RPC/gRPC 为什么在 4.5 层？**

```
第三层（NCCL/DeepEP）：
  - 面向 GPU 间高速通信
  - 追求极致性能（微秒级延迟）
  - 专用硬件（RDMA/NVLink）
  - 场景：训练梯度同步、MoE All-to-All

第四层（UCX/NVSHMEM）：
  - 底层传输框架
  - 给第三层提供统一接口

✨ 第 4.5 层（gRPC/Ray）：
  - 面向 CPU 进程间通信（控制平面）
  - 延迟容忍度高（毫秒级）
  - 跑在 TCP/IP 上（不走 RDMA）
  - 场景：请求调度、元数据管理、健康检查

第五层（vLLM/Dynamo）：
  - 调用 gRPC 做进程间协调
  - 调用 NCCL 做 GPU 间数据传输
```

### RPC vs NCCL 对比

```python
# ===== gRPC 场景：控制平面 =====

# vLLM 的 API Server 通过 gRPC 向 Worker 发送请求
async def handle_request(prompt):
    response = await grpc_client.generate(
        LLMGenerateRequest(text=prompt, max_tokens=100)
    )
    # gRPC 传输的是：控制指令 + 少量文本
    # 延迟：1-10ms
    return response

# ===== NCCL 场景：数据平面 =====

# DDP 训练时 GPU 间同步梯度
dist.all_reduce(gradient_tensor)  # 直接 NCCL 调用
# NCCL 传输的是：数十 GB 的 Tensor
# 延迟：微秒级，带宽：数百 GB/s
```

| 对比维度 | gRPC / RPC | NCCL / xCCL |
|----------|-----------|-------------|
| 通信对象 | CPU 进程间 | GPU 间（跨节点）|
| 传输内容 | 控制指令/元数据 | 大规模 Tensor 数据 |
| 协议 | HTTP/2 + Protobuf | 自定义二进制协议 |
| 底层网络 | TCP/IP | RDMA (IB/RoCE) |
| 延迟 | 毫秒级 (1-10ms) | 微秒级 (1-10μs) |
| 带宽 | ~1-10 GB/s | 100-900 GB/s |
| 典型场景 | 请求路由、元数据同步 | 梯度同步、KV Cache 传输 |

### Ray 中 RPC 的实际应用

```python
import ray

# Ray 内部用 gRPC 实现跨进程通信

@ray.remote
class LLMWorker:
    def __init__(self):
        self.model = load_model()
    
    def generate(self, prompt):
        return self.model.generate(prompt)

# 当你调用 .remote() 时，Ray 内部做了什么？
worker = LLMWorker.remote()  # ① 启动独立进程

future = worker.generate.remote("Hello")  # ② 通过 gRPC 发送调用请求
# ↓ 内部流程：
# Driver 进程 → gRPC 序列化 → Ray Object Store → Worker 进程

result = ray.get(future)  # ③ 通过 gRPC 接收结果
# ↓ 内部流程：
# Worker 进程 → 序列化结果 → gRPC → Driver 进程
```

### vLLM 中的 RPC 架构

```
vLLM Multi-Instance 架构：

┌─────────────────────────────────────────────┐
│              API Server 进程                 │
│  (接收 HTTP 请求，通过 gRPC 分发)            │
└──────────────────┬──────────────────────────┘
                   │ gRPC
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────┐     ┌──────────────┐
│ Worker 0     │     │ Worker 1     │
│ (GPU 0)      │     │ (GPU 1)      │
│              │     │              │
│  gRPC 控制面 │     │  gRPC 控制面 │  ← 请求调度
│  NCCL 数据面 │     │  NCCL 数据面 │  ← GPU 间 Tensor 通信
└──────────────┘     └──────────────┘
         │                   │
         └─────── NVLink ─────┘
```

**关键点：**
- 控制平面（请求路由、状态管理）→ gRPC
- 数据平面（GPU 间 Tensor 传输）→ NCCL

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

| 术语 | 一句话解释 | 关键数字 |
|---|---|---|
| PCIe | 通用高速总线，CPU↔GPU/网卡的物理通路 | Gen5 x16: 64 GB/s 单向 |
| NVLink/NVSwitch | NVIDIA 专有 GPU 间互连，带宽远超 PCIe | H100: 450 GB/s 单向 / 900 GB/s 双向 |
| CXL | PCIe 的缓存一致性升级，CPU 与加速器共享内存 | CXL 3.0 over PCIe 6.0 |
| GPUDirect RDMA | 网卡直接读写 GPU 显存，绕开 CPU 内存 | 数据路径减少 1 次拷贝 |
| DMA | 让外设直接访问内存，不需要 CPU 中转 | — |
| Pinned Memory | 页锁定内存，物理地址固定，DMA 安全访问的前提 | — |
| Zero-Copy | 数据传输中 0 次 CPU 参与的额外拷贝 | sendfile: 4 拷贝 → 2 拷贝 |
| RDMA | 跨机器直接内存访问，绕过 CPU 和内核 | 延迟 1–5 µs |
| InfiniBand | RDMA 原生高速无损网络，顶级 AI 集群首选 | NDR: 400 Gbps/端口 |
| RoCEv2 | 以太网上的 RDMA，需 PFC+ECN 无损配置 | 100–400 Gbps |
| NCCL | NVIDIA 训练集合通信标准库 | Ring/Tree 切换阈值 256 KB |
| MSCCL++ | GPU 原语级通信栈，性能优于 NCCL | 平均提速 1.7x（最高 5.4x） |
| NIXL | 推理数据传输库，统一异构存储 API | 已集成 vLLM/SGLang/TRT-LLM |
| DeepEP | MoE 专用通信库，稀疏 All-to-All | Decode dispatch 延迟 163 µs |
| NVSHMEM | GPU 全局地址空间，直接读写远端 GPU 内存 | 小消息吞吐最高 9.5x |
| IBGDA | GPU 直接向 NIC 发请求，CPU 退出数据路径 | — |
| RPC/gRPC | 跨进程远程调用框架，控制平面通信（第 4.5 层） | 延迟 1–10 ms |
| Ray | 分布式计算框架，内部用 gRPC + 共享内存 | Plasma 阈值 100 KB |
| Transfer Engine | Mooncake 底层传输引擎，模型权重高速分发（第四层） | RDMA 200 Gbps+ |
| Dynamo | NVIDIA 分布式推理框架 | DeepSeek-R1 最高 30x 吞吐 |
| TokenWeave | Token 级计算-通信 overlap，延迟 -29% | 吞吐 +26% |
| Ring AllReduce | 集合通信经典算法 | 总通信量 ≈ 2M（与 N 无关） |
| AllToAll | MoE 专家并行核心原语 | 总通信量 (N-1)×M |
| `perftest` (ib_write_bw) | RDMA 链路质量与带宽压测工具 | 200G 卡实测 180–195 Gbps |
| GID Index | RoCE 选择哪种地址，错了会连不上 | RoCEv2 通常选 3 |

---

> 本文基础篇部分为原创图解，全景篇部分参考 [washing 的知乎文章](https://zhuanlan.zhihu.com/p/2019475140635801509)整理扩展。

---

## 下一步

到这里你已经拥有了 AI 基础设施通信的完整地图——从芯片内部互连到跨机集合通信。这套知识和前三篇（体系结构、操作系统、网络）共同构成了理解现代分布式 AI 系统的硬件-软件-协议三维视角。

你可以回到系列起点，用全新的理解重读每一篇：

- [程序员的硬核基础（一）：体系结构](/posts/2026-06-13-cs-foundations-1-architecture/) — PCIe/NUMA/DMA 是本篇所有高速传输的物理基础
- [程序员的硬核基础（二）：操作系统](/posts/2026-06-14-cs-foundations-2-os/) — 内核旁路和 io_uring 模式在 RDMA/NCCL 中随处可见
- [程序员的硬核基础（三）：计算机网络](/posts/2026-06-15-cs-foundations-3-network/) — TCP 的局限性解释了为什么数据中心需要 RDMA

---

## 附录 B：Python 进程详解（同进程 vs 不同进程）

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




