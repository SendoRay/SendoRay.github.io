---
title: "程序员的硬核基础（一）：体系结构，从晶体管到 PCIe 拓扑"
date: '2026-06-13'
tags:
- CS-Foundations


draft: false
math: true
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

> 这篇文章不讲"CPU 怎么造出来的"，只回答一个问题：
>
> **当我们说"数据从内存搬到网卡"时，它到底走了哪条物理路径？**
>
> 这是《程序员的硬核基础三件套》的第一篇。读完它，你应该能在脑子里画出一台现代服务器的内部布线图，并且解释清楚：为什么 PCIe 拓扑会决定 RDMA 能跑多快。

---

## 一、冯诺依曼到现代 CPU：你写的 `a = b + c` 究竟发生了什么

### 1.1 冯诺依曼模型一页纸

冯诺依曼模型只有四个角色：**CPU、内存、I/O 设备、连接它们的总线**。所有现代计算机都没逃出这个抽象：

```
   ┌──────────┐         ┌──────────┐
   │   CPU    │◀───────▶│  内存     │
   │ (寄存器+ │   总线  │  (DRAM)  │
   │  ALU+    │◀──┐ ┌──▶│          │
   │  控制器) │   │ │   └──────────┘
   └────┬─────┘   │ │
        │ 总线    │ │
        ▼         │ │
   ┌──────────────┴─┴──┐
   │     I/O 设备       │  ← 网卡 / 磁盘 / GPU / ...
   └────────────────────┘
```

指令和数据共享同一段内存空间——这是冯诺依曼的核心约定。CPU 干的事则可以浓缩成一个循环：

```
   ┌──────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌──────────┐
   │ Fetch│──▶│ Decode │──▶│Execute │──▶│ Memory │──▶│Write-back│──┐
   │ 取指 │   │  译码  │   │  执行  │   │  访存  │   │   写回   │  │
   └──────┘   └────────┘   └────────┘   └────────┘   └──────────┘  │
        ▲                                                          │
        └──────────────────  下一条指令  ──────────────────────────┘
```

这五个阶段就是所谓的**经典五级流水线**。当我们说"主频 3 GHz"，意思是 CPU 每秒能完成 30 亿次时钟节拍——但**一次指令需要多个节拍**才能跑完整条流水线。

### 1.2 流水线、超标量、乱序、SIMD：四个让 CPU "看起来更快"的技巧

CPU 设计师为了把 IPC（Instructions Per Cycle）从 1 拉到 4 以上，发明了一系列把戏：

- **流水线（Pipelining）**：让取指、译码、执行像工厂流水线一样并行，每周期都能完成一条指令
- **超标量（Superscalar）**：CPU 内部有多个执行单元，一周期可以同时发射 4–8 条指令
- **乱序执行（Out-of-Order）**：CPU 自己重排指令顺序，让后面不依赖的指令提前算
- **SIMD**：一条指令同时对 4/8/16 个数据做相同运算（AVX-512、ARM NEON）

**程序员需要知道的副作用**：

1. **分支预测错误代价巨大**：一次预测失败要冲刷流水线，损失约 15–20 个周期。这就是为什么 hot path 里 `if/else` 比看起来贵。
2. **数据依赖会卡住流水线**：`a = b + c; d = a * 2;` 第二条必须等第一条算完。
3. **内存屏障 / volatile** 是为了对抗乱序：CPU 有权重排你的内存读写，多线程程序里你必须显式告诉它"这里不许动"。
4. **SIMD 是"免费"的 4–16 倍加速**：但前提是数据布局要对齐、要连续。这是 NumPy/PyTorch 比手写 Python 循环快几百倍的物理基础。

> **钩子**：当你以后看到"vectorized 实现比 naive 实现快 8 倍"，记住这不是软件玄学，是一条 AVX-256 指令一次干 8 个 float32 加法。

---

## 二、存储层级：所有性能问题的根源

### 2.1 把延迟和容量画成一座金字塔

```
              延迟 ↑                                       容量 ↓
              快                                            小
            ┌─────────┐                                  ┌──────┐
            │ Register│  0.3 ns                           │ ~1KB 
            └────┬────┘                                  └──────┘
                 │
           ┌─────┴─────┐
           │ L1 cache  │  1 ns                            ~32 KB
           └─────┬─────┘
                 │
         ┌───────┴─────-───┐
         │  L2 cache       │  4 ns                       ~256 KB
         └───────┬─────────┘
                 │
       ┌─────────┴───────-───┐
       │     L3 cache        │  12 ns                  几十 MB
       └─────────┬───────────┘
                 │
     ┌───────────┴────────-────┐
     │     DRAM (本地 NUMA)     │  80 ns                  GB ~ TB
     └───────────┬─────────────┘
                 │
   ┌─────────────┴───────────-───┐
   │       NVMe SSD (随机)        │  100 µs               TB
   └─────────────┬───────────────┘
                 │
 ┌───────────────┴────────────────┐
 │     网络 / 远端机器              │  100 µs ~ 150 ms       ∞
 ┌───────────────────────────────┐
              慢                                              大
```

### 2.2 一张数量级表把延迟刻进脑子

| 层级 | 容量 | 延迟（典型） | 带宽（典型） | 
|---|---|---|---|
| 寄存器 | ~1 KB | 0.3 ns | — |
| L1 cache | 32–64 KB | **1 ns** | 1 TB/s |
| L2 cache | 256 KB–1 MB | 4 ns | 500 GB/s | 
| L3 cache | 几十 MB | 12 ns | 200 GB/s | 
| DRAM（本地 NUMA）| GB–TB | **80 ns** | 50–100 GB/s | 
| DRAM（跨 NUMA） | — | 130 ns | — |
| NVMe SSD（4K 随机）| TB | 100 µs | 7 GB/s | 
| 同机房网络（RTT） | — | 100 µs | 100 Gbps | 
| 跨城（北京-上海）| — | 30 ms | — | 


### 2.3 性能优化的第一性原理：让数据离 CPU 更近

绝大多数性能优化的本质都是同一件事：**把数据搬到更高的存储层级**。

- 数据库做缓存：把磁盘搬到 DRAM
- Redis：把磁盘搬到 DRAM
- HugePages：让 TLB 命中率上升
- AI 推理用 KV Cache：把"重新算一遍"换成"从显存读"
- GPUDirect RDMA：跳过"显存→DRAM→网卡"那段绕路

**反过来**，所有让你性能崩盘的现象，本质都是数据被踢到下一级：cache miss、TLB miss、page fault、swap、网络不命中本地副本。

---

## 三、缓存：你以为按字节读，其实按 64B 读

### 3.1 cache line：CPU 的最小搬运单位

CPU 不会真的从内存里"读一个 byte"。它一次至少读 **64 字节**——也就是一条 cache line。即使你只用了 1 个字节，剩下 63 个字节也跟着进了 cache。

这带来三个工程上的后果：

1. **空间局部性**：访问 `a[0]` 之后再访问 `a[1]..a[15]` 几乎是免费的（同一条 cache line）
2. **结构体字段顺序很重要**：把常一起访问的字段放一起，能减少 cache miss
3. **伪共享（False Sharing）**：两个线程改的是不同变量，但变量恰好在同一条 cache line 上，CPU 会被迫不停同步，性能掉到 1/10

### 3.2 MESI：一句话讲清缓存一致性

多核 CPU 各有 L1/L2，怎么保证你看到的内存是一致的？答：**MESI 协议**。每条 cache line 有四种状态：

- **M**odified：我改过，内存还没同步
- **E**xclusive：只有我有，没改
- **S**hared：好几个核都有，没人改
- **I**nvalid：已失效

核之间通过总线广播状态变更。**核心心智模型**：写一个变量 = 让其他核的对应 cache line 变成 Invalid = 它们下次读必须从你这里 / 从内存重新拉。

### 3.3 工程后果：为什么并发计数器用 `padding[64]`

```c
struct Counter {
    std::atomic<int> value;
    char padding[60];   // 凑齐 64B，防止伪共享
};
```

每个核一个独立 cache line。少了 padding，多核高并发下吞吐能差 10 倍以上。这不是优化，是"避免反优化"。下面这张图直观看出区别：

```
  ❌ 伪共享：a 和 b 在同一条 cache line
              ┌──────── 64 B ────────┐
  内存:       │ a │ b │ ░░░░░░░░░░░░ │
              └────┬───┬─────────────┘
   CPU0 改 a ──▶  │   │  ◀── CPU1 改 b
   两个核被迫不停同步整条 line，吞吐掉到 1/10

  ✅ Padding 后：a 和 b 各占一条 cache line
              ┌──── 64 B ────┐  ┌──── 64 B ────┐
  内存:       │ a │ ░░░░░░░░ │  │ b │ ░░░░░░░░ │
              └───┘            └───┘
   CPU0 ──▶ 自己的 line       CPU1 ──▶ 自己的 line
   互不干扰
```

--

## 四、内存序与原子操作：多核世界的可见性契约

### 4.1 为什么乱序执行 + 多核 = 程序员必须关心内存序

第一章里我们讲过，单核 CPU 为了把 IPC 拉上去会**乱序执行**——后面不依赖的指令可以先算。在单核世界里这件事程序员不用关心，因为 CPU 会负责让"对外可观测的最终结果"和源码顺序保持一致。

但多核一上来，这个温柔的契约就破了。**核 A 写入内存的顺序，被核 B 看到的顺序，未必相同。**

经典反直觉例子：

```c
// 共享变量初始为 0
int x = 0;
int flag = 0;

// Core A 跑这段
x = 1;
flag = 1;

// Core B 跑这段
if (flag == 1) {
    assert(x == 1);   // ← 这条 assert 真的可能失败！
}
```

按源码"读"，flag=1 之前 x 必然=1。但实际硬件上：

- Core A 的两条 store 可能被 CPU 重排，flag 先到主存、x 还在 store buffer 里
- 即使 Core A 没重排，Core B 也可能因为自己的 load 重排，先看到 flag 的新值、再看到 x 的旧值
- 在 ARM/RISC-V 上这种"看到 flag=1 但 x 还是 0"是**家常便饭**

这不是 CPU 的 bug，是 CPU 设计师**故意保留的自由度**——为了性能。把这个自由度收回来的代价，就是程序员显式声明"这里要有顺序"。

### 4.2 Store Buffer：写入不是立刻可见的

要理解为什么乱序，先看一次 store 的物理路径。CPU 写一个变量并不是直接写到 cache，而是先扔进**Store Buffer**，再异步排队进 L1：

```
   Core 0                                Core 1
   ┌──────────┐                          ┌──────────┐
   │ register │                          │ register │
   └────┬─────┘                          └────┬─────┘
        │ store x=1                           │
        ▼                                     ▼
   ┌────────────┐  (异步)                ┌────────────┐
   │ Store Buf  │ ─────────┐             │ Store Buf  │
   └────┬───────┘          │             └────┬───────┘
        │                  │                  │
        ▼                  ▼                  ▼
   ┌──────────┐       ┌──────────┐       ┌──────────┐
   │ L1 cache │◀─MESI▶│ L1 cache │◀─MESI▶│ L1 cache │
   └────┬─────┘       └────┬─────┘       └────┬─────┘
        └─────────┬────────┴─────────┬────────┘
                  ▼                  ▼
              ┌──────────────────────────┐
              │   共享 L2 / L3 / DRAM    │
              └──────────────────────────┘
```

关键事实：

1. **store 写完不等于别人能看到**——它可能还在自己的 Store Buffer 里
2. Store Buffer 是**每个核私有**的，MESI 协议看不到它
3. 自己读自己刚写的值，会从 Store Buffer 里 forwarding 回来（所以单线程没问题）
4. 别的核要"看到"这个写，必须等 Store Buffer 把它推进 L1，并通过 MESI 让别人的 line 失效

这就是"写入有延迟可见"的硬件根源。**Store Buffer 的存在让 store-load 可以被重排**——因为 load 可以从 cache 直接拿，但前面的 store 还卡在 buffer 里没出去。

### 4.3 x86-TSO vs ARM/RISC-V 弱序模型

不同 ISA 给程序员的"乱序额度"完全不同。

**x86 用的是 TSO（Total Store Order）**——相对强：

- store-store **保序**（你 Core 内部多个写出去，别人看到的顺序就是你写的顺序）
- load-load **保序**
- load-store **保序**
- **唯独 store-load 可以重排**（因为 Store Buffer）

**ARM / RISC-V / POWER 用的是弱序模型**——几乎啥都能重排：

- store-store 可以重排
- load-load 可以重排
- load-store / store-load 都可以重排
- 只有数据依赖会强制保序（你不能在算出地址前就 load 那个地址）

一张对比表：

| 重排类型 | x86-TSO | ARM/RISC-V 弱序 |
|---|---|---|
| store → store | ❌ 不会 | ✅ 会 |
| load → load | ❌ 不会 | ✅ 会 |
| load → store | ❌ 不会 | ✅ 会 |
| store → load | ✅ 会 | ✅ 会 |

**程序员需要知道的副作用**：在 x86 上跑了三年没出 bug 的无锁代码，移植到 ARM 服务器（比如 Graviton、鲲鹏）上可能直接翻车。这不是软件 bug，是你**默认依赖了 x86 的强序保证**——而那个保证 ARM 不给你。

### 4.4 内存屏障（Memory Fence / Memory Barrier）

要把"乱序额度"收回来，硬件提供**屏障指令**：在屏障之前的内存操作必须全部对外可见，才允许执行屏障之后的操作。

x86 的三条屏障：

- `lfence`：load fence，前面的 load 全部完成才能往下
- `sfence`：store fence，前面的 store 全部刷出 Store Buffer 才能往下
- `mfence`：full fence，load 和 store 全保序

ARM 的 `dmb` / `dsb` / `isb` 也是类似的概念，只是分类更细。

但程序员不直接写屏障——直接写不可移植。语言层面给了三个**语义等级**：

- **Acquire**：用在"读取共享数据"那一侧。后面的读写**不能**移到 Acquire 之前。配合**加锁**这一侧的语义。
- **Release**：用在"发布共享数据"那一侧。前面的读写**不能**移到 Release 之后。配合**解锁**这一侧的语义。
- **Seq_cst**（Sequential Consistency）：所有线程看到的全局顺序完全一致。最强、最慢。

C++11 的 `memory_order` 枚举就是这套语义的直接映射：

```cpp
// 发布侧
data = 42;
ready.store(true, std::memory_order_release);   // 前面的 store 不能越过这一行

// 消费侧
if (ready.load(std::memory_order_acquire)) {     // 后面的 load 不能越到这一行前
    use(data);   // 此时一定能看到 data == 42
}
```

**Acquire/Release 是配对使用的**——一边 release、另一边 acquire，中间所有的修改都被这道围栏圈住。这就是无锁编程的最小可用工具。

### 4.5 原子操作：CAS 与 fetch-add 的硬件实现

光有屏障还不够。两个核同时对一个变量做 `++`，仍然会丢更新（read-modify-write 不是原子的）。**原子指令**才是终极武器。

**x86 的实现：`lock` 前缀**

```asm
lock incl (%rdi)        ; 原子自增 *rdi
lock cmpxchgq %rsi, (%rdi) ; 原子 CAS
```

`lock` 前缀的含义：本次访问期间**独占**这条 cache line（其他核的对应 line 强制 Invalid）。早期实现是锁总线，现在是锁 cache line——叫 **cache line locking**。

**ARM 的实现：LL/SC（Load-Link / Store-Conditional）**

```asm
retry:
    ldxr w0, [x1]       ; load-exclusive 标记 x1 为"我在监视"
    add  w0, w0, #1
    stxr w2, w0, [x1]   ; store-exclusive，如果中间被别人改过就失败
    cbnz w2, retry      ; 失败就重试
```

LL/SC 是**乐观重试**风格：先假设没人和我抢，写的时候硬件检查"中间有没有别的核动过这条 line"，没动就成功，动了就重来。RISC-V 的 `lr/sc` 是同一套思路。

**CAS（Compare-And-Swap）：无锁数据结构的基石**

```c
// 伪代码：原子地"如果当前值是 expected 就改成 desired"
bool CAS(int* p, int expected, int desired) {
    if (*p == expected) {
        *p = desired;
        return true;
    }
    return false;
}
```

用 CAS 写无锁栈的 push：

```c
void push(Node* new_node) {
    Node* old_top;
    do {
        old_top = top;            // 读当前栈顶
        new_node->next = old_top; // 把新节点指向它
    } while (!CAS(&top, old_top, new_node));  // 如果中间没人改 top 就成功
}
```

**fetch-add：原子计数器**

```c
int old = atomic_fetch_add(&counter, 1);  // 返回旧值，counter 已 +1
```

并发统计、生成全局递增 ID、引用计数都靠它。

### 4.6 工程后果与钩子

**程序员需要知道的副作用**：

1. **`std::atomic<int>` 默认是 `memory_order_seq_cst`，这是最贵的等级**。热路径上的原子计数器、引用计数，换成 `relaxed`（不要顺序保证，只要原子性）能差 2–5 倍。比如 `shared_ptr` 的引用计数自增就只需要 relaxed，递减到 0 才需要 acquire-release。
2. **无锁数据结构看起来不用加锁，但写对 memory order 比加锁难 10 倍**。教科书上无锁队列的论文每隔几年就发现一个新 bug，根源都是某处 memory order 用错了。能用锁就用锁，真到了瓶颈再考虑无锁。
3. **Java 的 `volatile`、Go 的 `sync/atomic`、Rust 的 `Ordering` 背后都是同样的硬件机制**——只是包装了不同的"语义等级"。换语言不换底层物理。
4. **跨架构移植要重新审视**：x86 上正确的代码，ARM 上要补屏障；ARM 上正确的代码，x86 上多半是过度同步、白白损失性能。

> **钩子**：在 RDMA 场景里，应用程序填好一个 WQE（Work Queue Entry）写到内存，然后通过 MMIO 写一下网卡的 doorbell 寄存器。这两步之间**必须有一道 `sfence`**——否则 doorbell 的写可能先到，网卡通过 PCIe 反过来 DMA WQE 的时候，看到的还是旧内容。这就是"内存序"在系统编程里最直接、最致命的体现：你以为你"写好了再敲门"，但没有屏障，门可能先被敲响。后面 RDMA 篇我们会回到这条 sfence。

---

## 五、虚拟内存与页表：操作系统给你的精美谎言

### 5.1 虚拟地址 vs 物理地址

每个进程都以为自己独占整个 64 位地址空间。这是**虚拟地址**。CPU 拿到虚拟地址后，要先翻译成**物理地址**才能去找 DRAM。这个翻译过程查的就是**页表（Page Table）**。

x86-64 用 4 级页表（Linux 5.x 之后是 5 级），每级有 512 个表项。一个虚拟地址被切成 5 段索引，逐级 walk：

```
虚拟地址 0x7fff_e1f0_2440
   │
   ├─ PGD index ─→ 找到 P4D
   ├─ P4D index ─→ 找到 PUD
   ├─ PUD index ─→ 找到 PMD
   ├─ PMD index ─→ 找到 PTE
   ├─ PTE      ─→ 物理页号 PFN
   └─ 页内偏移 ─→ 物理地址
```

光看流程不够直观，**来个带数据的真实例子**。栈上一个变量虚拟地址 `0x0000_7FFF_E1F0_2440`（4 KB 页、4 级页表，未启用 P4D），先按位切：

```
虚拟地址  0x0000_7FFF_E1F0_2440   ← 高 16 位是符号扩展（canonical）

  bit47..39   bit38..30   bit29..21   bit20..12   bit11..0
  ─────────   ─────────   ─────────   ─────────   ─────────
   PGD=255     PUD=511     PMD=271     PTE=258    off=0x440
```

假设 CR3 寄存器（PGD 表的物理基址）= `0x1A000`，每条表项 8 字节，整次 walk 走下来：

```
① PGD[255]  @ 0x1A000 + 255×8 = 0x1A7F8  → PUD 基址 0x2B000
② PUD[511]  @ 0x2B000 + 511×8 = 0x2BFF8  → PMD 基址 0x3C000
③ PMD[271]  @ 0x3C000 + 271×8 = 0x3C878  → PTE 基址 0x4D000
④ PTE[258]  @ 0x4D000 + 258×8 = 0x4D810  → PFN     0x98765
⑤ 物理地址 = (PFN << 12) | offset = 0x9876_5440
```

看清楚了：一次最普通的 `*ptr` 在 TLB miss 时，硬件要发 **4 次页表读 + 1 次实际数据读 = 5 次 DRAM 往返**。

每访问一次内存就要 walk 5 次页表？不可能。CPU 里有专门的 **TLB（Translation Lookaside Buffer）**，把"虚拟页号→物理页号"缓存起来。**TLB miss 是隐形性能税**——查一次完整页表大约要 100 ns，相当于一次 DRAM 访问。

### 5.2 Huge Page：用 2 MB / 1 GB 大页减少 TLB 压力

默认页大小是 4 KB。一台 1 TB 内存的机器要 2.5 亿条页表项，TLB 怎么够用？答：用 **2 MB 或 1 GB 的 Huge Page**，让一条 TLB 项覆盖更大范围。AI 训练、数据库、DPDK 几乎全是 Huge Page 用户。

### 5.3 钩子：RDMA 文里"在 CPU 页表上动动手脚"

"禁止用户态进程访问某段内存区间倒不是什么难事，内核只需要在 CPU 的页表上动动手脚就行了。"

翻译过来就是：**内核改一下 PTE 的权限位**（比如把 Writable 位清零），CPU 下次写这页就会触发 page fault，内核就拿到了"这块内存被锁定了"的硬件保证。RDMA 注册内存（`ibv_reg_mr`）干的就是这件事——把用户态缓冲区的页 pin 住、设上特殊属性，让网卡（通过 IOMMU）能直接访问，且应用程序不能随便改。

> 这就是为什么 RDMA 必须先调用 `register_mr` 这种慢路径——它是在改页表。但**改完之后**，数据面上每次 send/recv 都不用再过内核了。

---

## 六、总线：决定数据搬运上限的"高速公路"

### 6.1 从南北桥到 PCIe Gen5

二十年前 CPU 通过"前端总线"接北桥（管内存），北桥再接南桥（管 I/O）。现在内存控制器、PCIe Root Complex 都集成进 CPU 了，但**总线的概念没变**——它就是芯片之间的高速公路。

数据中心里你只需要记住一条总线：**PCIe**。它是 CPU、GPU、NVMe SSD、网卡之间的通用接口。

### 6.2 PCIe 的两个关键参数：lane 数 × generation

PCIe 用差分对（一对一收一发的物理线缆）传输，每对叫一条 **lane**。一条 PCIe 接口可以是 x1 / x4 / x8 / x16 等不同 lane 数。每条 lane 的带宽随 generation 翻倍：

| 代次 | 单 lane 单向 | x16 双向带宽 | 量产年份 |
|---|---|---|---|
| Gen3 | 1 GB/s | **32 GB/s** | 2010 |
| Gen4 | 2 GB/s | **64 GB/s** | 2017 |
| Gen5 | 4 GB/s | **128 GB/s** | 2022 |
| Gen6 | 8 GB/s | **256 GB/s** | 2025+ |

记忆口诀：**PCIe Gen5 x16 ≈ 64 GB/s 单向 / 128 GB/s 双向**。一张 H100 接 PCIe Gen5 x16，理论上对外有 64 GB/s 单向带宽——这就是它跟 CPU/网卡通信的天花板。

### 6.3 PCIe Switch：一个 Root，多个设备

CPU 上 PCIe lane 数是有限的（比如 EPYC 9004 单 socket 128 lane）。要挂更多设备，就在 CPU 下面再接一颗 **PCIe Switch**。Switch 就像网络交换机，下面又能挂多个 endpoint：GPU、网卡、NVMe SSD。

```
       CPU (Root Complex)
       │  PCIe Gen5 x16
       ▼
  ┌─ PCIe Switch ---─┐
  │   │   │   │      │
  GPU NIC NIC NVMe  ...
```

**关键观察**：Switch 下挂的设备之间可以**点对点（Peer-to-Peer, P2P）**直接通信，**不必绕回 CPU**。这就是 GPUDirect RDMA 的物理基础——同一个 Switch 下的 GPU 和网卡可以直接对话。



---

## 七、DMA 与 MMIO：CPU 不是数据搬运工

### 7.1 DMA：让外设自己读写内存

如果每次网卡收到数据都要 CPU 一字节一字节拷到内存，那 100 Gbps 网卡能把单核 CPU 拖死。**DMA（Direct Memory Access）** 解决这问题——外设里有专门的 DMA 引擎，CPU 只要告诉它"从这个地址读 N 字节"，剩下的搬运它自己来，搬完发个中断通知 CPU。

**DMA 是现代 I/O 的命脉**。NVMe、网卡、GPU 没一个不靠 DMA。

### 7.2 MMIO：把寄存器当内存读

CPU 怎么"指挥"这些外设？两种方式：

- **Port I/O**：x86 的 `in`/`out` 指令，用独立的端口空间
- **MMIO（Memory-Mapped I/O）**：把外设寄存器映射进物理内存地址，CPU 用普通的 load/store 指令读写

现代设备几乎全用 MMIO。RDMA 的 **Doorbell**——也就是"我的 work request 已经填好了，网卡你来取"——就是一次 MMIO 写：往网卡的某个固定地址写一个 64 位数。

### 7.3 IOMMU：DMA 的地址翻译

DMA 引擎用什么地址访问内存？早期是物理地址。但**应用程序拿到的是虚拟地址**，让用户态直接告诉网卡"这块内存在 0x7fff..."显然不行——用户根本不知道物理地址，而且不安全。

解决方案：**IOMMU（I/O Memory Management Unit）**。它是给设备用的"页表"。CPU 给设备一份"设备虚拟地址→物理地址"的映射，设备访问内存时，IOMMU 负责翻译并做权限检查。

**心智模型**：

- CPU 那边有 MMU + 页表，把进程虚拟地址翻译成物理地址
- 设备那边有 IOMMU + 设备页表，把设备虚拟地址翻译成物理地址
- 内核可以让 CPU 页表和 IOMMU 页表对**同一段虚拟地址**指向**同一段物理内存**

这就是为什么 RDMA 网卡能"直接读你应用程序的缓冲区"——内核已经在 IOMMU 里设好了通路。

下面这张图把 CPU/IOMMU/设备/页表的关系一次画清楚：

```
     用户态 buf 的虚拟地址  0x7fff_e1f0_2440
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
   ┌────────┐       ┌─────────┐
   │  MMU   │       │ IOMMU   │
   │ (CPU侧)│       │ (设备侧)│
   └────┬───┘       └────┬────┘
        │                │
   CPU 页表         设备页表
   （内核维护）       （内核维护）
        │                │
        ▼                ▼
        ┌────────────────┐
        │  物理页 (同一页)│   ← 应用 buf 实际所在
        └────────────────┘
               ▲
               │ DMA
               │
            ┌──────┐
            │ 网卡 │
            └──────┘
```

CPU 和网卡通过两套独立的页表，访问**同一段物理内存**——这就是 RDMA 让网卡"直接读用户态 buf"的硬件基础。

### 7.4 GPUDirect：DMA 链路再缩短一截

GPU 显存其实也能"被 DMA"。NVIDIA 把这套能力命名为 **GPUDirect** 家族：

- **GPUDirect P2P**：GPU 和 GPU 之间直接 DMA，不走 CPU 内存
- **GPUDirect Storage**：NVMe 直接 DMA 到显存
- **GPUDirect RDMA**：网卡直接 DMA 到/从显存

它们的物理基础都是同一件事——**只要两个设备挂在同一个 PCIe Switch 下，它们就可以通过 P2P 直接 DMA**。

---

## 八、NUMA：跨 socket 是一笔隐形性能税

### 8.1 多 socket 服务器的内存其实是分裂的

```
          ┌──── UPI / Infinity Fabric ────┐
          │                                │
   ┌──────▼──────┐                ┌────────▼─────┐
   │    CPU0     │                │     CPU1      │
   │  (NUMA 0)   │                │   (NUMA 1)    │
   └──┬─────┬────┘                └────┬─────┬────┘
      │     │                          │     │
      │     │ DDR5                     │ DDR5│
      ▼     ▼                          ▼     ▼
   ┌──────────┐                     ┌──────────┐
   │ DRAM 0   │  本地 80 ns         │ DRAM 1   │
   └──────────┘                     └──────────┘
         ▲                                ▲
         │  CPU0 远端访问 DRAM 1 = 130 ns │
         └────────────────────────────────┘
         (绕一大圈：CPU0 → UPI → CPU1 内存控制器 → DRAM 1)
```

本地访问 80 ns，远端访问 130 ns；带宽更是只剩一半。

一台双路服务器有两颗 CPU，每颗 CPU 有自己的内存控制器和**自己直连的 DRAM**。CPU 之间通过专用互联（Intel UPI / AMD Infinity Fabric）跨 socket 访问对方的内存。

- **本地访问**：CPU0 读自己直连的 DRAM，约 80 ns，带宽 100 GB/s
- **远端访问**：CPU0 读 CPU1 直连的 DRAM，约 130 ns，带宽减半

这就是 **NUMA（Non-Uniform Memory Access）**。Linux 在你看不见的地方做调度和内存分配——但分配错了，性能就掉 30%。

### 8.2 程序员要知道的三件事

1. **`numactl --hardware`** 看你的机器有几个 NUMA node
2. **关键进程绑核**：`numactl --cpunodebind=0 --membind=0 ./your_app`，避免内存跨 socket
3. **网卡和 GPU 也有 NUMA 亲和性**：一张网卡是挂在 CPU0 还是 CPU1 下，决定了它"该被哪个 NUMA node 的进程使用"

 AWS p5 的拓扑图就是个标准的双 NUMA：左半边 4 GPU + 16 网卡属于 CPU0，右半边属于 CPU1。**绝不要让进程在 CPU0 上、却用 CPU1 下的网卡**——那是双倍 NUMA 税 + 双倍 PCIe 跨域。

---

## 九、从单机到集群：GPU 拓扑全景

上面讲完了单颗 CPU 内的存储层级和多 socket 的 NUMA 税，现在把视角再抬高一层：**一台机器里装了 8 张 GPU，它们是怎么连在一起的？多台机器又是怎么组成一个训练集群的？** 搞清楚这张物理拓扑图，后面所有的并行训练策略才有根。

### 9.1 术语澄清：节点 = 机器，以及 CPU/GPU 的数量关系

先把三个被混用的词钉死：

- **节点（Node）= 一台物理机器（Machine）= 一台服务器**——三者完全等价
- 论文里说"128 节点训练"，就是 128 台物理机器

一台典型的 AI 训练服务器长这样：

| 组件 | 典型数量 | 举例 |
|---|---|---|
| CPU | 1–2 颗（单路/双路） | AMD EPYC 9004（96 核/颗） |
| GPU | 4 或 8 张 | DGX H100 = 8× H100 |
| RDMA 网卡 | 4–8 张（每张 200G 或 400G） | ConnectX-7 / CX-7 |
| 内存 | 数百 GB ~ 2 TB DDR5 | — |

一个集群就是**多个这样的节点通过高速网络连接**。把层级关系画出来：

```
集群（Cluster）
├── 节点 0（= 一台物理服务器）
│   ├── CPU 0（96 核）──── DDR5 DRAM（本地内存）
│   ├── CPU 1（96 核）──── DDR5 DRAM（本地内存）
│   ├── GPU 0 ~ GPU 3（挂在 CPU 0 的 PCIe 下）
│   ├── GPU 4 ~ GPU 7（挂在 CPU 1 的 PCIe 下）
│   ├── NIC 0 ~ NIC 3（RDMA 网卡，挂 CPU 0）
│   └── NIC 4 ~ NIC 7（RDMA 网卡，挂 CPU 1）
├── 节点 1（同构配置）
├── ...
└── 节点 N
```

**关键比例**：一台机器里 CPU 只有 1–2 颗，但 GPU 有 4–8 张——GPU 才是"需要被喂饱"的主角，CPU 更像是个管家。

### 9.2 单机多 GPU：PCIe + NVLink/NVSwitch 双层互连

单机内部的 GPU 不是随便插上去就能互相通信的——它们之间有两种截然不同的连接方式。

**拓扑 A：纯 PCIe（云实例常见，如 AWS p3/p4）**

GPU 通过 PCIe Switch 挂在 CPU 下面，GPU 之间通信只能走 PCIe：

```
         CPU 0                          CPU 1
          │                              │
    ┌─────┴─────┐                  ┌─────┴─────┐
    │PCIe Switch│                  │PCIe Switch│
    ├───┬───┬───┤                  ├───┬───┬───┤
   GPU0 GPU1 NIC0               GPU2 GPU3 NIC1
```

- 同 Switch 下的 GPU 可以 P2P 直通（如 GPU0↔GPU1），带宽 = PCIe Gen5 x16 = 64 GB/s 单向
- 跨 Switch 的 GPU 要绕 CPU（如 GPU0↔GPU2），带宽减半，延迟加倍
- 这种拓扑做多卡训练时，跨 Switch 的 all-reduce 是瓶颈

**拓扑 B：NVLink/NVSwitch 全互联（DGX H100 / DGX B200）**

8 张 GPU 通过 NVSwitch 形成**全互联（any-to-any）**——任意两张 GPU 之间都有直连高带宽通道：

```
        ┌─────────── NVSwitch ───────────┐
        │    (全互联，any-to-any)         │
   GPU0 ─── GPU1 ─── GPU2 ─── GPU3      │
   GPU4 ─── GPU5 ─── GPU6 ─── GPU7      │
        └────────────────────────────────┘
        │                                │
   ┌────┴────┐                     ┌────┴────┐
   │  CPU 0  │─── UPI/IF ──────── │  CPU 1  │
   │(PCIe域) │                     │(PCIe域) │
   └─────────┘                     └─────────┘
        │                                │
   NIC0~NIC3                        NIC4~NIC7
```

关键区分：

- **NVSwitch 是独立的交换芯片**，不是 CPU 的一部分，专门负责 GPU 间互联
- **NVLink 带宽**：H100 = 450 GB/s 单向，B200 = 900 GB/s 单向——远超 PCIe 的 64 GB/s
- **PCIe 仍然用于** CPU↔GPU 和 GPU↔NIC 通信（NVLink 只管 GPU 之间）
- 8 张 GPU 的 NVLink all-reduce 几乎不是瓶颈，瓶颈在跨机网络

**`nvidia-smi topo -m` 命令**可以查看实际拓扑。输出示例里常见的缩写：

| 缩写 | 含义 | 带宽量级 |
|---|---|---|
| NV18 | NVLink 连接（18 条 link） | 450 GB/s |
| PIX | 同一 PCIe Switch 下 | 64 GB/s |
| PHB | 同一 PCIe Host Bridge（同 CPU） | 32–64 GB/s |
| SYS | 跨 NUMA / 跨 CPU | 带宽减半 |
| NODE | 同 NUMA node 但非直连 | — |

### 9.3 多机多 GPU：节点间通过网络互连

一台机器内部 NVLink 带宽再高，数据要出机器就必须走网卡。跨机通信的物理路径：

```
GPU → PCIe → NIC → 网络交换机 → 对端 NIC → PCIe → 对端 GPU
```

如果 GPU 和 NIC 在同一个 PCIe Switch 下，还能用 GPUDirect RDMA 省掉 CPU 内存那一段拷贝：

```
GPU → PCIe Switch → NIC → 网络 → 对端 NIC → PCIe Switch → 对端 GPU
```

画出 2 节点的多机拓扑：

```
┌─────────── Node 0 ───────────┐     网络交换机      ┌─────────── Node 1 ───────────┐
│  GPU0..7 ←─NVLink─→ GPU0..7  │         │           │  GPU0..7 ←─NVLink─→ GPU0..7  │
│      │                        │         │           │      │                        │
│   NIC0~7 ─────────────────────┼────── Switch ──────┼── NIC0~7                      │
│   (各200G)                    │    (IB/RoCE)        │   (各200G)                    │
└───────────────────────────────┘                     └───────────────────────────────┘
```

**跨机带宽瓶颈在网卡**：即使每机 8 张 200G 网卡聚合 = 200 GB/s 单向出口，也只有 NVLink 的一半不到。

**关键数字对比表**：

| 通信路径 | 带宽（单向） | 延迟 | 典型场景 |
|---|---|---|---|
| 同机 GPU↔GPU（NVLink） | 450 GB/s（H100） | ~1 µs | 张量并行、NVLink AllReduce |
| 同机 GPU↔GPU（PCIe P2P） | 32–64 GB/s | ~2 µs | 无 NVLink 的云实例 |
| 跨机 GPU↔GPU（RDMA） | 25–50 GB/s | 5–10 µs | 数据并行梯度同步 |
| 跨机 GPU↔GPU（TCP） | 10–12 GB/s | 50–100 µs | 退化场景 |

从表中可以看出：NVLink 比 PCIe P2P 快 7–14 倍，比跨机 RDMA 快 10–18 倍，比 TCP 快 40 倍以上。**每跨一层物理边界，带宽都断崖式下降**。

### 9.4 工程后果：为什么拓扑决定并行策略

物理带宽的层级差异直接决定了分布式训练的并行策略切分：

| 并行策略 | 通信量 | 对带宽要求 | 放在哪一层 | 原因 |
|---|---|---|---|---|
| 张量并行（TP） | 极大（每层 2 次 AllReduce） | 极高 | 同机 NVLink 内（≤8 卡） | 只有 NVLink 带宽够用 |
| 专家并行（EP） | 大（All-to-All） | 很高 | 同机 NVLink 内 | DeepSeek 用 8 卡 NVLink 做机内 EP |
| 流水线并行（PP） | 小（只传激活值） | 较低 | 跨机 | 通信量小，容忍网络延迟 |
| 数据并行（DP/ZeRO） | 中（梯度 AllReduce） | 中等 | 跨机 | 可以 overlap 计算 |

**一句话总结**：通信量最大的并行维度放在带宽最高的物理层，通信量最小的并行维度跨到带宽最低的物理层。这就是 3D/4D 并行策略设计的第一性原理。

> **钩子**：这就是为什么第四篇"通信"里每个通信库都在解决同一个问题——**怎么在这些物理约束下把带宽吃满**。NCCL 的 Ring AllReduce、Tree AllReduce 都在做同一件事：把逻辑上的集合通信映射到物理拓扑上最高效的路径。

---

## 十、GPU 视角：把上面的概念全部叠在一起

### 10.1 GPU 是什么

把 CPU 的"几个核 + 大缓存 + 复杂控制逻辑"换成"几千个简单核 + 极大带宽显存"，就得到了 GPU。它不擅长分支密集的代码，极擅长"对一大块数据做相同运算"——这恰好是矩阵乘法、卷积、attention 的形状。

H100 的关键数字（记一下）：

| 指标 | 数值 |
|---|---|
| SM（Streaming Multiprocessor）数 | 132 |
| FP16 Tensor Core 算力 | ~1000 TFLOPs |
| HBM3 显存容量 | 80 GB |
| HBM3 显存带宽 | **3.35 TB/s** |
| NVLink 带宽（单向） | 450 GB/s |
| PCIe Gen5 x16（单向） | 64 GB/s |

### 10.2 HBM：贴在 GPU 旁边的高带宽内存

DRAM 是普通 DDR5，带宽 ~50 GB/s/通道。**HBM** 是把多块 DRAM die 用 TSV 堆叠起来直接焊在 GPU 旁边，带宽能飙到 **3 TB/s 以上**。这是 GPU 训练/推理性能的根本——没有 HBM，GPU 就是个跑不动的核数怪物。

### 10.3 NVLink：GPU 之间的“内部高速公路”

GPU 之间如果通过 PCIe 通信，单向只有 64 GB/s。NVIDIA 又造了一套**专属互联 NVLink**，单向 450 GB/s，比 PCIe 高 7 倍。8 卡 H100 服务器内部用 NVSwitch 把 8 张 GPU 全互联，做 all-reduce 时就用这条路。

**心智模型**：

- 同机内 GPU↔GPU：走 NVLink（450 GB/s）
- 跨机 GPU↔GPU：走 PCIe → 网卡 → 网络 → 对端网卡 → PCIe → GPU（受网卡带宽限制）

3200 Gbps = 32 张 100 Gbps 网卡聚合 = 单机 400 GB/s 单向出口带宽。这个数字刚好接近 NVLink 的量级——说明跨机和同机的带宽差距正在被网络追上。

### 10.4 GPUDirect RDMA：所有概念的合成

把前面的零件全装在一起，就得到了 GPUDirect RDMA 的完整图景：

1. 应用程序用 `cudaMalloc` 拿到一段**显存**的虚拟地址
2. 应用程序调用 `ibv_reg_mr` 把这段地址注册给网卡——内核在 **IOMMU** 里建好映射，让网卡知道这块"虚拟地址 → 显存物理页"
3. 应用程序提交 work request，按 **MMIO doorbell** 通知网卡
4. 网卡通过 **PCIe Switch** 直接 P2P **DMA** 显存
5. 数据从显存 → PCIe Switch → 网卡 → 网线，**完全不经过 CPU 内存**
6. 对端网卡反过来 DMA 到对端显存

```
   普通 TCP 路径（4 段总线 + CPU 内存被打爆）

     ┌──────┐   PCIe   ┌──────────┐  内存总线  ┌────────┐
     │ GPU  │─────────▶│ PCIe Sw  │──────────▶│  CPU   │
     │ HBM  │          └──────────┘            │  +DRAM │
     └──────┘                ▲                 │ (拷贝) │
                              │ 内存总线        └────┬───┘
                              │                     │ PCIe
                              │                     ▼
                          ┌─────────┐         ┌──────────┐
                          │ PCIe Sw │◀────────│   CPU    │
                          └────┬────┘         └──────────┘
                               │ PCIe
                               ▼
                          ┌──────┐
                          │ NIC  │ ──→ 网线
                          └──────┘

   GPUDirect RDMA 路径（一段总线 + CPU 完全不参与）

     ┌──────┐                                     ┌──────┐
     │ GPU  │────────┐                       ┌────│ NIC  │ ──→ 网线
     │ HBM  │        │   ┌───────────────┐   │    └──────┘
     └──────┘        └──▶│  PCIe Switch  │◀──┘
                          └───────────────┘
                          (P2P DMA, 同 Switch 子树内)
```

这条路径上每一个环节我们都在前面拆过了。从晶体管到 PCIe 拓扑——这就是为什么程序员需要懂体系结构。

---

## 十一、附录：术语速查表

| 术语 | 一句话解释 | 关键数字 |
|---|---|---|
| **IPC** | 每周期指令数 | 现代 CPU 4–6 |
| **Cache line** | CPU 最小搬运单位 | **64 B** |
| **L1 / L2 / L3** | CPU 内三级缓存 | 1 / 4 / 12 ns |
| **TLB** | 虚拟→物理地址翻译缓存 | miss 代价 ~100 ns |
| **Page** | 虚拟内存最小单位 | 默认 **4 KB**，大页 2 MB / 1 GB |
| **PTE** | Page Table Entry，一条页表项 | 8 字节 |
| **MMU** | CPU 端地址翻译硬件 | — |
| **IOMMU** | 设备端地址翻译硬件 | — |
| **DMA** | 设备直接读写内存 | — |
| **MMIO** | 把寄存器映射进内存空间 | — |
| **PCIe Gen5 x16** | 主流 GPU/网卡接口 | **64 GB/s 单向** |
| **PCIe Switch** | 多设备汇聚点，支持 P2P | — |
| **NVLink** | NVIDIA GPU 间专属互联 | **450 GB/s 单向**（H100） |
| **HBM3** | 堆叠在 GPU 旁的高带宽内存 | **3.35 TB/s**（H100） |
| **NUMA** | 多 socket 下的内存非对称 | 远端 +60% 延迟 |
| **GPUDirect RDMA** | 网卡直接 DMA 显存 | 路径不经 CPU 内存 |
| **Memory Order** | CPU 对内存读写的可见性顺序 | x86: TSO; ARM: 弱序 |
| **Store Buffer** | CPU 写入到 cache 之间的暂存队列 | 导致写不立刻可见 |
| **Memory Fence** | 强制内存操作的排序点 | sfence/lfence/mfence |
| **CAS** | Compare-And-Swap 原子操作 | 无锁数据结构基础 |
| **Acquire/Release** | 内存序语义：加锁/解锁方向 | 配对使用 |
| **节点（Node）** | 一台物理服务器，= Machine | 集群的基本单元 |
| **NVSwitch** | GPU 间全互联交换芯片 | 8 GPU any-to-any |
| **张量并行（TP）** | 将模型层内的张量切分到多 GPU | 需 NVLink 带宽 |
| **数据并行（DP）** | 每张 GPU 完整副本，梯度同步 | 可跨机，overlap 计算 |

---

## 十二、下一步

你现在有了硬件层的语言。但有一个角色还在阴影里——**操作系统**。当你调用 `send()` 或 `cudaMalloc()` 时，是它在背后协调所有这些硬件资源；而 RDMA 的核心命题"绕过内核"，就是要在保留它的安全性的前提下，把它从数据通路上请走。

下一篇：[程序员的硬核基础（二）：操作系统，从用户态到内核旁路](/posts/2026-06-14-cs-foundations-2-os/)。
