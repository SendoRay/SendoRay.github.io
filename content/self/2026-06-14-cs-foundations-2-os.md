---
title: "程序员的硬核基础（二）：操作系统，从用户态到内核旁路"
date: '2026-06-14'
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

> 这篇文章不写"Linux 入门"，只回答一个问题：
>
> **操作系统内核到底在你和硬件之间做了什么？什么时候它是帮手，什么时候是路障？**
>
> 这是《程序员的硬核基础三件套》的第二篇。前置阅读：[体系结构篇](/posts/2026-06-13-cs-foundations-1-architecture/)（如果你还不知道页表、MMIO、IOMMU，请先读它）。读完本篇你应该能回答：为什么 RDMA 要"绕过内核"，又如何只绕过一部分？

---

## 一、用户态 vs 内核态：CPU 自带的"二分法"

### 1.1 一颗 CPU，两种身份

x86-64 的 CPU 有四个特权环（Ring 0–3），但实际只用两个：

- **Ring 0 = 内核态（Kernel mode）**：可以执行任意指令，可以访问任意内存
- **Ring 3 = 用户态（User mode）**：只能访问自己进程的内存，只能执行非特权指令

"特权指令"是哪些？比如：

- 读写控制寄存器（CR3 = 当前页表指针）
- 操作 MMU / 关中断 / 设置 IDT
- 直接读写 I/O 端口
- 执行 `hlt` 让 CPU 进入低功耗

CPU 里有一个标志位记录"我现在是 Ring 0 还是 Ring 3"。这个标志只能由硬件机制（中断、异常、`syscall` 指令）来切换——用户态没办法自己把自己升权。这是整个操作系统安全模型的物理根基。

### 1.2 为什么要二分

**信任问题**。CPU 不能区分"这条 mov 指令是 Linux 内核写的、还是用户进程写的"——它只看权限位。如果让所有代码都跑在 Ring 0，那任何一段崩溃的应用都能把整台机器搞挂、读取别人的内存、操纵网卡发恶意包。

代价就是：**用户态做不了任何 I/O，必须求内核帮忙**。这个"求"的动作叫**系统调用（system call）**。

### 1.3 上下文切换不是免费的

每次 user → kernel → user 的来回（一次 syscall）至少包括：

```
   用户态 (Ring 3)                            内核态 (Ring 0)
   │                                            │
   │ ① 填寄存器：rax=syscall号, rdi/rsi/...      │
   │ ② SYSCALL 指令 ───────────────────────▶ ③ 保存用户寄存器、切栈
   │                                            │ ④ 检查参数 / dispatch
   │                                            │ ⑤ 执行真正的内核逻辑
   │                                            │ ⑥ copy_to_user
   │ ⑧ 拿到返回值，继续跑     ◀──────────── ⑦ SYSRET 返回用户态
   │
   ─── 总代价 ≈ 100 ns – 1 µs (Spectre 缓解还要加一截) ───
```

成本大约 **几百到几千个 CPU 周期**——量级是 **100 ns – 1 µs**。听起来不多，但如果你每秒做 100 万次 `recv()`，那就是 100 ms 的纯 syscall 开销。这就是为什么高性能场景要么用 batching（io_uring），要么干脆绕开内核（DPDK / RDMA）。


---

## 二、进程与线程：从"task_struct"看本质

### 2.1 进程 = 一份隔离的资源包

Linux 的"进程"是一组资源的打包：

- 独立的虚拟地址空间（独立页表）
- 打开的文件描述符表
- 信号处理表
- 当前工作目录、UID/GID
- 一段或多段执行流（线程）

`fork()` 复制一份这些资源给子进程；`exec()` 把当前进程的代码段替换成另一个程序但保留 PID。**Linux 内核里没有"进程"和"线程"两种结构**——只有 `task_struct` 这一种结构体，区别只是创建时哪些资源选择"共享"还是"独立"。

### 2.2 线程 = 共享地址空间的执行流

`pthread_create` 底下其实是 `clone()` 系统调用，参数告诉内核"创一个新 task_struct，但和当前的共享地址空间、共享文件描述符表"。所以多线程进程里：

- 所有线程**共享同一份页表**（同一个虚拟地址空间）→ 一个 malloc 出来的指针所有线程都能用
- 每个线程**独立的栈和寄存器**
- 调度单位是**线程**，不是进程

**性能后果**：线程切换比进程切换便宜（不用切页表、TLB 不会全部失效）；但跨线程数据竞争靠你自己加锁。

### 2.3 协程是用户态的"线程"

`goroutine`、Python `async`、Rust `tokio task` 都是**用户态调度**的执行流。内核完全不知道它们的存在。一个 OS 线程可以承载几千几万个协程，切换成本只有几十 ns（一次函数调用 + 寄存器保存）。

代价是：协程必须**协作式**让出 CPU（碰到 `await`/`yield`），否则一个协程死循环就会卡住整个线程上的所有协程。

### 2.4 同步原语：让线程不打架

线程共享地址空间是把双刃剑：方便通信，但也意味着两个线程可能同时改同一块内存。同步原语就是用来给"谁先谁后"画规矩的。

#### Mutex（互斥锁）

最朴素的：保护临界区，同一时刻只能有一个线程持有锁。

```c
pthread_mutex_lock(&m);
counter++;          // 临界区
pthread_mutex_unlock(&m);
```

但 Linux 的 `pthread_mutex` 底下不是简单的 syscall——它是 **futex** 实现的。

#### Futex：快路径不进内核

`futex` 的精妙在于：**无竞争时根本不进内核**，只在用户态做一次 atomic CAS。

```
   无竞争（fast path，全程用户态，~5 ns）：
   ┌──────────────────────────────────────┐
   │  CAS(state, 0, 1)  → 成功，拿到锁     │
   │  ...临界区...                          │
   │  CAS(state, 1, 0)  → 释放，没人等      │
   └──────────────────────────────────────┘

   有竞争（slow path，进内核排队，~µs）：
   ┌──────────────────────────────────────┐
   │  CAS(state, 0, 1)  → 失败，已被持有    │
   │  state := 2  (标记"有等待者")          │
   │  syscall futex_wait → 内核挂起线程     │
   │                                        │
   │  另一线程释放：CAS(state, 2, 0) 后      │
   │  syscall futex_wake → 内核唤醒等待者   │
   └──────────────────────────────────────┘
```

设计哲学：**让 99% 的无竞争情况完全免费，只为 1% 的竞争情况付内核代价**。这个思想几乎贯穿所有现代同步机制。

#### Spinlock：忙等就是不睡

```c
while (atomic_test_and_set(&lock)) { /* spin */ }
```

不挂起线程，疯狂 CAS 直到拿到锁。**适合临界区只有几十 ns 的场景**——比如内核的中断上下文（中断里不能睡眠！），或多核之间高频争抢的极小数据结构。代价是浪费 CPU：如果临界区超过几百 ns 还在 spin，那不如直接 sleep。

#### 读写锁（RWLock）

允许多个读者同时进入，但写者独占。适合"读多写少"的数据结构（路由表、配置、字典等）。

#### RCU（Read-Copy-Update）

读者**完全无锁**——直接读旧版本指针。写者复制一份、修改、原子替换指针，然后等所有读者都走出"旧版本"才释放旧内存（grace period）。

```
   读者：rcu_read_lock() / load(ptr) / use(*ptr) / rcu_read_unlock()  ← 零开销
   写者：new = copy(*ptr); modify(new); atomic_xchg(ptr, new);
         synchronize_rcu();   // 等所有读者退出
         free(old);
```

Linux 内核里大量数据结构（路由表、模块列表、命名空间）都用 RCU——读路径压到极致，写路径慢一点没关系。

#### 条件变量（Condition Variable）

用来等"某个条件成立"。比如生产者-消费者：消费者等队列非空，生产者放入元素后 `signal` 一下。

```c
pthread_mutex_lock(&m);
while (queue_empty()) pthread_cond_wait(&cv, &m);
take_one();
pthread_mutex_unlock(&m);
```

#### 信号量（Semaphore）

允许 **N 个**线程同时进入，常用作"令牌池"——比如限制最多 8 个并发请求。

#### 总结表格

| 原语 | 适用场景 | 无竞争开销 | 有竞争行为 |
|---|---|---|---|
| Futex/Mutex | 通用临界区 | 几 ns（CAS） | sleep/wake（µs） |
| Spinlock | 极短临界区/中断上下文 | 几 ns | 忙等（浪费 CPU） |
| RWLock | 读多写少 | 几 ns | 写者等读者全部退出 |
| RCU | 读极多写极少 | **0** | 写者等 grace period |
| 条件变量 | 等待特定条件 | — | sleep/signal |
| 信号量 | 限制并发数 | CAS | sleep |

> **钩子**：RDMA 的 completion queue 有两种消费模式——polling（用户态忙等，像 spinlock）和 event-driven（等中断通知，像 futex）。低延迟选前者，省 CPU 选后者。

---

## 三、虚拟内存：从 OS 视角再看一次

> 体系结构篇讲了页表怎么 walk、TLB 怎么用。本节讲 OS 怎么"用"页表来玩魔术。

### 3.1 mmap：把任何东西伪装成内存

`mmap()` 是 Linux 里最有诗意的 syscall。它能把这几个东西都塑造成"一段虚拟地址"，让你用普通的 load/store 指令访问：

```
              用户态虚拟地址空间
   0──────────────────────────────────────────────→ 2^48
     │代码│堆│ ......  mmap 区区块 ......                  │栈│
     └────┬──┬──────────────────────────────────────┘
          │  │  │         │         │         │
     页表映射│  │  │         │         │         │
          ▼  ▼  ▼         ▼         ▼         ▼
        ┌───┐ ┌────┐ ┌──────┐ ┌─────┐ ┌────────┐
        │匿名│ │文件│ │设备MMIO│ │共享 │ │另一进程│
        │ 0  │ │页缓存│ │网卡寄存器│ │ SHM │ │同一页│
        └───┘ └────┘ └──────┘ └─────┘ └────────┘
          堆/         读       进入 RDMA   跨进程      进程间
         malloc      mmap        的入口      交互        零拷贝
```

机制：内核在你的页表里建一些 PTE，但**不分配真实的物理页**。等你第一次访问触发 page fault 时，内核才把对应内容从后端读进来、分配物理页、修页表。这叫 **demand paging**。

### 3.2 page fault：所有"懒加载"的入口

Page fault 不全是错误。Linux 内核大量"按需分配"机制都是借 page fault 实现的：

- **首次访问匿名页**：分配一张零页
- **共享只读页**：所有进程映射到同一物理页
- **写时复制（COW）**：fork 后子进程读父进程的页，写时才真正复制
- **mmap 文件**：第一次读触发把磁盘内容载入页面缓存

代价是：每次 fault 大约 **几个 µs**。如果你有 1 GB 数据要 mmap 进来逐字节扫一遍，就有 25 万次 minor fault。这就是为什么 `MAP_POPULATE` 标志能加速（提前把全部页 fault 进来）。

### 3.3 共享内存：进程间最快的通信

两个进程 mmap 同一个 shm 段，相同的物理页就出现在它们各自的虚拟地址空间里。读写零拷贝，不过 syscall。Redis、PostgreSQL、CUDA IPC 都靠它。

> **钩子**：RDMA 注册的 MR（Memory Region），本质就是把"用户态进程 + 网卡"两个 IOMMU 域 mmap 到同一段物理页。

---

## 四、文件系统与页缓存：磁盘数据的高速公路

### 4.1 VFS：一套 API 统治所有文件系统

`open` / `read` / `write` / `close` 这套 API 你写了一辈子。神奇之处在于：**同一份代码既能读 ext4 的本地文件，也能读 NFS 的远程文件，还能读 procfs 这种“假文件”**。这一切靠的是 VFS（Virtual File System）这层抽象。

```
   ┌───────────────────────────────┐
   │  应用 (open/read/write/...)  │
   └──────────────┬───────────────┘
                  │ syscall
   ┌──────────────▼───────────────┐
   │      VFS（虚拟文件系统）       │   ← 统一抽象层
   └──┬─────┬─────┬──────┬─────┬──┘
      │     │     │      │     │
   ┌──▼┐ ┌─▼─┐ ┌─▼─┐ ┌──▼─┐ ┌▼────┐
   │ext4│ │XFS│ │NFS│ │proc│ │tmpfs│
   └──┬─┘ └─┬─┘ └─┬─┘ └────┘ └─────┘
      │     │     │
   ┌──▼─────▼─────▼──┐
   │    块设备层      │   (bio / 调度器 / IOMMU)
   └────────┬────────┘
            │
       磁盘 / SSD / NVMe
```

VFS 内部有三个核心抽象：

- **inode**：一个文件的元数据集合（大小、权限、修改时间、磁盘块指针）。**文件名不在 inode 里**——同一个 inode 可以有多个名字（硬链接）。
- **dentry**：目录项缓存。把“路径字符串 → inode”的查找结果缓存起来，避免每次都去磁盘 walk 目录树。
- **superblock**：一个挂载点的全局信息（块大小、总块数、空闲块位图）。

每种具体文件系统只要实现 VFS 定义的几十个回调（`->read_iter`、`->write_iter`、`->lookup`...），就自动获得了 `read()`、`write()`、`ls`、`cat` 这一整套用户态工具。

### 4.2 Page Cache：内核帮你做的磁盘缓存

**所有 `read`/`write` 默认都经过 Page Cache**——也就是 DRAM 里维护的一份磁盘副本。

```
   write 路径（默认走 Page Cache）：

   用户 buf  ──copy_from_user──▶  Page Cache（脏页）
                                       │
                                       │ （内核异步 flush，可能延迟几秒）
                                       ▼
                                    磁盘
```

读取流程：

1. `read(fd, buf, n)` 进内核
2. 查 Page Cache：命中 → 直接 `copy_to_user`，返回
3. miss → 触发块设备 I/O，把数据从磁盘读进 Page Cache，再 copy 给用户

写入流程：

1. `write(fd, buf, n)` 进内核
2. 把用户数据 copy 到 Page Cache，**标记该页为脏页**
3. **直接返回成功**（这时数据还在 DRAM！）
4. 内核后台线程（pdflush / writeback）在合适时机把脏页刷到磁盘

`free -h` 命令里的 `buff/cache` 列就是 Page Cache 占用的内存。Linux 的哲学是“DRAM 闲着也是闲着，全用来当磁盘缓存”——这是大多数 Linux 服务器看起来“内存几乎用满”的真正原因。

### 4.3 fsync：数据安全落盘的唯一保证

这是工程师最常踩的坑：

```
   write() 返回成功  ≠  数据已到磁盘
   write() 返回成功  =   数据已到 Page Cache
```

如果这时机器掉电、内核 panic，**Page Cache 里的脏页全部丢失**。要让数据真正落盘，必须显式调用：

- `fsync(fd)`：强制把当前文件的所有脏页 + inode 元数据刷到磁盘，等它确认才返回
- `fdatasync(fd)`：只刷数据脏页，不刷 inode 时间戳等元数据，更快（节省一次磁盘往返）

**工程后果**：

- 数据库每次 `COMMIT` 都得 `fsync`，否则掉电就丢已确认的事务（违反 D = Durability）
- MySQL 的 `innodb_flush_log_at_trx_commit=1` 就是每次 commit 强制 fsync；改成 0 性能高 10×，但崩溃可能丢最近 1s 的事务
- SSD 上 `fsync` 至少几百 µs，HDD 上 ms 级——这是数据库 TPS 的物理上限

### 4.4 Direct I/O：绕过 Page Cache

`open(path, O_DIRECT | ...)` 让该 fd 的读写**直接在用户 buf 和磁盘之间 DMA**，完全绕开 Page Cache。

为什么数据库爱用？因为数据库自己有 **Buffer Pool**——它比内核更懂“哪些页热、哪些页冷、什么时候刷盘”。Page Cache 在它眼里是多余的二级缓存：

- 浪费内存（同一份数据在 Buffer Pool 和 Page Cache 各一份）
- 干扰刷盘节奏（数据库想精确控制 WAL 何时落盘，但 Page Cache 自作主张）

代价：

- **必须对齐**：buf 地址、文件偏移、读写长度都得是块大小的倍数（通常 4 KB）
- **小 I/O 性能差**：没有了 Page Cache 的合并和预读
- **不能用 mmap**：mmap 本身就建立在 Page Cache 上

> **钩子**：`NVMe + io_uring + O_DIRECT` 是现代高性能存储的极致路径——异步、零拷贝（用户 buf 直接 DMA）、零内核缓存。SPDK 更进一步：连块设备驱动都搬到用户态，整个 I/O 路径不进内核。

### 4.5 工程后果

- `echo 3 > /proc/sys/vm/drop_caches` 会清掉 Page Cache。**AI 训练数据加载首次极慢**就是因为冷缓存——所有样本都得从磁盘重新读；第二个 epoch 全在 DRAM 里，I/O 几乎为零。
- **大模型 checkpoint 几十 GB**，一次 `write + fsync` 可能卡住训练数秒。这就是异步 checkpoint（先 dump 到 host DRAM，再后台慢慢刷盘）的存在意义。
- `dd` 测磁盘速度时若不加 `oflag=direct`，测的是 Page Cache 速度，不是磁盘速度。
- 容器里 Page Cache 是宿主机共享的——一个容器疯狂读大文件可能挤掉别人的缓存（噪声邻居问题）。

---

## 五、容器基石：cgroups 与 namespace

容器不是虚拟机，也不是某种“沙盒”。它就是一组**普通 Linux 进程**，只不过被两件武器包了起来：**namespace 让它看到的世界变小，cgroups 让它能用的资源变少**。

### 5.1 Namespace：给进程一个“独立宇宙”

每个 namespace 隔离内核里的一类全局资源，让属于这个 namespace 的进程仿佛“独享”这份资源。

| Namespace | 隔离什么 | 效果 |
|---|---|---|
| **PID** | 进程 ID | 容器内 PID 从 1 开始，看不到宿主机其他进程 |
| **Network** | 网络栈 | 容器有自己的 IP、端口、路由表、iptables |
| **Mount** | 文件系统挂载点 | 容器看到的 `/` 完全不同，可以挂载自己的镜像 |
| **UTS** | 主机名 / 域名 | 容器有自己的 hostname |
| **IPC** | 进程间通信对象 | 共享内存段、消息队列、信号量隔离 |
| **User** | UID / GID 映射 | 容器内的 root（uid=0）= 宿主机的某个普通用户 |

创建 namespace 只要一个 syscall：

```c
clone(child_func, stack, CLONE_NEWPID | CLONE_NEWNS | CLONE_NEWNET | ..., NULL);
```

或者命令行：

```bash
unshare --pid --net --mount --fork bash   # 进入新 namespace 的 shell
```

容器运行时（runc、containerd-shim）干的核心事就是 `clone` 时传齐六个 `CLONE_NEW*` 标志。

### 5.2 cgroups：给进程组设天花板

namespace 解决“看什么”，cgroups 解决“用多少”。可以限制一组进程的：

- **CPU**：`cpu.shares`（权重）、`cpu.cfs_quota_us`（绝对配额，比如“每 100 ms 最多用 40 ms CPU”）
- **内存**：`memory.limit_in_bytes`（超了直接 OOM kill）
- **磁盘 I/O**：`blkio.throttle.read_bps_device`（每秒最多读多少字节）
- **网络带宽**：配合 tc 做出口限速
- **设备访问**：`devices.allow` 控制能 open 哪些 `/dev/*`

**v1 vs v2**：v1 每种资源是一棵独立的 cgroup 树（互相打架，配置混乱）；v2 统一成一棵层级树，所有控制器挂在同一棵树上，干净很多。新的发行版默认 v2。

一个例子：限制某组进程最多用 4 核 + 16 GB 内存：

```bash
mkdir /sys/fs/cgroup/mygroup
echo "400000 100000" > /sys/fs/cgroup/mygroup/cpu.max     # 每 100 ms 最多 400 ms = 4 核
echo $((16*1024*1024*1024)) > /sys/fs/cgroup/mygroup/memory.max
echo $$ > /sys/fs/cgroup/mygroup/cgroup.procs              # 把当前 shell 扔进去
```

之后这个 shell 及其所有子进程都受这两条限制约束。

### 5.3 容器 vs 虚拟机：隔离边界不同

```
  虚拟机（硬件级隔离）                容器（OS 级隔离）
  ┌──────────────────────┐          ┌──────────────────────┐
  │ App A │ App B │ App C│          │ App A │ App B │ App C│
  ├───────┼───────┼──────┤          ├───────┼───────┼──────┤
  │Guest  │Guest  │Guest │          │ bins/libs │ bins/libs │
  │ OS    │ OS    │ OS   │          └────────┬─────────────┘
  ├───────┴───────┴──────┤                   │ namespace + cgroups
  │     Hypervisor       │          ┌────────┴─────────────┐
  ├──────────────────────┤          │   宿主机 Linux 内核   │
  │   宿主机 OS + 硬件    │          ├──────────────────────┤
  └──────────────────────┘          │      宿主机硬件       │
                                    └──────────────────────┘
  开销大、启动慢、隔离强            开销小、启动快、隔离弱
```

虚拟机每个 Guest 有自己的内核，hypervisor 在硬件级别（VT-x / AMD-V）做隔离，安全边界硬。容器全部共享宿主机内核——任何内核漏洞都会跨过容器边界。所以多租户公有云容器之外还要套一层 microVM（Firecracker、Kata Containers）。

### 5.4 Docker = namespace + cgroups + Union FS + Image

把 Docker 拆开：

- **namespace + cgroups**：进程隔离 + 资源限额（前两节）
- **Union FS**（OverlayFS）：镜像的层级文件系统，多个只读层 + 一个读写层叠加，节省空间和拉取时间
- **Image**：把镜像层打包成 tarball + manifest，可以推到 registry（Docker Hub），别处拉下来一模一样地跑

Docker 没有发明任何一项底层技术——它只是把“namespace + cgroups + OverlayFS + 镜像分发”这几样原本各自存在的东西工程化封装成一个体面的开发者工具。

### 5.5 工程后果

- Kubernetes Pod 里跑 `nvidia-smi` 只能看到分配的那几张 GPU——这是 NVIDIA 设备插件通过 cgroups 的 **devices 子系统**做的可见性控制（不允许 open 其他 `/dev/nvidia*`）。
- 容器内 `free -h` 看到的是**宿主机内存**，不是容器实际配额——`free` 读的是 `/proc/meminfo`，那是 namespace 透不过去的。要看真实限制：

  ```bash
  cat /sys/fs/cgroup/memory.max               # cgroup v2
  cat /sys/fs/cgroup/memory/memory.limit_in_bytes   # cgroup v1
  ```

- JVM 在老版本里也踩这个坑——按宿主机内存设堆大小，触发容器 OOM。后来加了 `-XX:+UseContainerSupport`（默认开启）才学会读 cgroup 配额。
- 容器里 `top` 看到的 load average 也是宿主机的，不是自己的。

> **钩子**：AI Infra 几乎全跑在 K8s 上——GPU 调度、网络隔离（CNI）、存储挂载（CSI）、训练任务的资源限额，全部建立在 namespace + cgroups 之上。理解这两样，等于拿到 K8s 内部运作的钥匙。

---

## 六、系统调用：用户态唯一的“出入境口岸”

### 6.1 一次 `read()` 在硬件上做了什么

```
用户态:
  调用 glibc 的 read()
  → 把参数填进寄存器（rax=系统调用号 0, rdi=fd, rsi=buf, rdx=count）
  → 执行 SYSCALL 指令
─────────────────────  CPU 切到 Ring 0
内核态:
  → 跳到 entry_SYSCALL_64
  → 保存用户寄存器到内核栈
  → 切换 GS/FS、检查参数合法性
  → dispatch 到 sys_read()
  → VFS 层 → 文件系统层 → 块设备层 → 真正读磁盘
  → 把数据拷到 buf（这里需要 copy_to_user）
─────────────────────
  → 执行 SYSRET 回到用户态
用户态:
  read() 返回，应用拿到字节数
```

最便宜的 syscall（如 `getpid`）只要 ~50 ns；像 `read()` 这种碰到 I/O 的，可能要等几 µs 甚至几 ms。

### 6.2 为什么 syscall 慢

不只是切环。还有这些隐藏成本：

1. **流水线冲刷**：CPU 切上下文等于把流水线清空重启
2. **TLB / cache 污染**：内核执行的代码踢掉你的 cache
3. **Spectre/Meltdown 缓解**：现代 CPU 切到内核态会做额外的间接分支屏障，慢上一截
4. **参数复制**：`copy_to_user` / `copy_from_user` 这种边界检查不能省

### 6.3 减少 syscall 的几种思路

| 思路 | 例子 |
|---|---|
| **Batching** | `writev`/`readv`、`sendmmsg`、io_uring 一次提交几百个 op |
| **轮询代替阻塞** | epoll edge-triggered + busy poll |
| **共享内存代替消息** | mmap 跨进程通信 |
| **完全绕开** | DPDK / RDMA / SPDK |

> **钩子**：RDMA 数据面（send/recv）一次都不进内核——work request 写进队列里，doorbell 是一次 MMIO 写而已，完成事件靠用户态轮询完成队列拿到。这就把 1 µs 的 syscall 成本压到了几十 ns 的"内存写"成本。

---

## 七、中断与软中断：硬件想说话

### 7.1 硬中断（Hardware IRQ）

外设要告诉 CPU "事情干完了"，靠的是**中断**。网卡收到一个包后，会通过 PCIe 发一个 MSI-X 消息，CPU 立刻打断当前任务，跳到中断处理函数（ISR）。

但 ISR 要尽量短——它是关中断在跑的，时间长了会丢中断。所以现代 Linux 把中断处理拆成两半：

- **上半部（top half）**：硬件 IRQ handler，只做最紧急的事（比如 ack、把数据拍进队列），快速返回
- **下半部（bottom half）**：软中断 / tasklet / 内核线程，做实际的协议栈处理

### 7.2 NAPI：高速网卡的专用模式

100 Gbps 网卡每秒能收 1500 万包（甚至更多）。如果每个包都中断 CPU，单这一项就能把 CPU 打满。**NAPI（New API）** 的做法是“忙时轮询、闲时中断”的混合模式：

```
   网卡进包速率低：                       网卡进包速率高：
   （中断驱动）                          （轮询驱动）

   网卡──IRQ──▶ CPU                       网卡 ─── IRQ──▶ CPU
         (每包一次)                          (首包触发后马上关)
                                                       │
                                                       ▼
                                                  调度 NET_RX
                                                  软中断
                                                       │
                                                       ▼
                                                  while (RX ring 非空)
                                                    poll 一批包
                                                  重开 IRQ
```

这是 Linux 内核网络栈在万兆/十万兆上还能撑住的关键技巧。

### 7.3 IRQ 亲和性：把中断绑到对的 CPU

多核机器上一个中断默认随便落到哪个核。如果网卡中断总是落到 CPU0，那 CPU0 永远在忙网络，其他核闲着。解决方法：**给每个 RX 队列绑一个 CPU**：

```bash
echo 2 > /proc/irq/<irq>/smp_affinity   # 2 = bitmask = CPU1
```

更进一步，配合 RSS（Receive Side Scaling）：网卡硬件按 hash 把不同的连接分到不同 RX 队列，每个队列绑一个 CPU——这样多连接的负载就被均匀打散到全部核上。

---

## 八、调度器：CFS 一句话原理

### 8.1 让所有线程"看起来公平"

Linux 默认调度器叫 **CFS（Completely Fair Scheduler）**。它的目标是让 N 个线程仿佛在 N 个独立 CPU 上各跑各的——等价于每个线程拿到 1/N 的 CPU 时间。

实现上每个线程有一个 `vruntime`（虚拟运行时间），调度器永远选 vruntime 最小的那个跑。跑过的线程 vruntime 增加；不同优先级（nice 值）让 vruntime 涨得快或慢——nice 值越低优先级越高，vruntime 涨得越慢。

### 8.2 抢占点

线程不会跑到永远，CFS 会在以下时机抢占：

- 时钟中断到来时检查（典型 1–10 ms 一次）
- 当前线程主动让出（`yield`、I/O 阻塞）
- 唤醒了一个更高优先级的线程

这意味着你的线程随时可能被踢下 CPU，回来时**所在的 CPU 都可能变了**——cache 全冷，TLB 部分失效。

### 8.3 CPU 亲和性 + isolcpus

高性能场景要避免这种"飘"。常见做法：

- `taskset -c 4 ./app`：把进程绑死在 CPU4
- `isolcpus=4-7` 内核启动参数：把这几个核从 CFS 调度池踢出去，专给指定进程用
- `nohz_full=4-7`：连时钟中断都关掉，CPU 真正"独占"

DPDK / 高频交易系统几乎全都这么干。**绑核 + 大页 + IRQ 亲和性**是高性能 Linux 的"圣三件套"。

---

## 九、I/O 模型四件套

> 这一节是为下一篇网络做铺垫。但它本身也是 OS 抽象的精华。

### 9.1 阻塞 I/O：最朴素的写法

```c
ssize_t n = read(fd, buf, sizeof(buf));   // 没数据就睡着，有数据返回
```

调用线程在内核里被挂起，加进 fd 的等待队列。一旦数据到来，内核唤醒它继续跑。**简单但伸缩性差**——一个连接一个线程，1 万连接就是 1 万线程，调度和栈内存都顶不住。

### 9.2 非阻塞 I/O：永远立即返回

```c
fcntl(fd, F_SETFL, O_NONBLOCK);
ssize_t n = read(fd, buf, sizeof(buf));   // 没数据返回 -1，errno=EAGAIN
```

线程不会被挂起，但你必须**自己轮询**——这又是另一种浪费。所以非阻塞几乎不单独用。

### 9.3 I/O 多路复用：用一个线程看一万个 fd

`select` / `poll` / `epoll` 让内核帮你看一组 fd 中"哪些就绪了"。

- **select / poll**：每次调用要把全部 fd 从用户态拷到内核态，O(N) 扫描，10K 连接就开始吃力
- **epoll**：fd 注册一次进内核（红黑树），就绪事件用回调机制塞进就绪链表，每次 `epoll_wait` 只返回**已就绪**的 fd，O(就绪数)

**Edge-triggered（ET）vs Level-triggered（LT）**：

- LT：只要 fd 还有数据可读，每次 `epoll_wait` 都告诉你
- ET：只在 fd 状态**变化**时告诉你一次，要求你**一次读到 EAGAIN 为止**，否则错过通知

ET 模式更省 syscall，但要求你彻底读干净。Nginx、Redis、Netty 都用 ET。

epoll 内部结构可以画成下面这样：

```
          用户态                              内核态
   ┌─────────────┐                  ┌─────────────────┐
   │ epoll_create│───────────────▶│  epoll 实例      │
   └─────────────┘                  │  ┌────────────┐│
   ┌─────────────┐                  │ │ 红黑树     ││  ← 所有注册的 fd
   │  epoll_ctl  │ ────加/减 fd ────▶│ │ fd1,fd2..││
   └─────────────┘                  │ └───────────┘│
                                    │  ┌───────────┐│  ← 只装就绪的 fd
   ┌─────────────┐ ◀── 拿就绪事件 ──││  就绪链表   ││
   │ epoll_wait  │                  ││  fd2,fd5    ││
   └─────────────┘                  │└───────────┘│
                                    └────▲─────────────┘
                                          │ fd 就绪时
                                          │ 内核回调插入
                                       ┌───┴───┐
                                       │ 网卡/磁盘/其他 fd 事件 │
                                       └────────────────────┘
```

所以 epoll 的 O(就绪数) 不是魔法——是因为 fd 注册进红黑树后只进一次内核，事件发生时被驱动代码推到就绪链表上。

### 9.4 异步 I/O：io_uring

epoll 解决了"等"的问题，但**操作本身**（read/write）还是 syscall。**io_uring** 把这层也抹平：

- 用户态和内核共享两个环形队列（提交队列 SQ + 完成队列 CQ），都在共享内存里
- 应用把 op 写进 SQ，调用 `io_uring_enter` 告诉内核（甚至可以省掉这一步）
- 内核异步完成 op，把结果写进 CQ
- 应用从 CQ 读结果

一次可以提交几百个 op，syscall 次数从 N 降到 1（甚至 0）。**架构上和 RDMA 的 work queue / completion queue 已经非常神似**——这不是巧合，**所有高性能 I/O 的终局都是"用户态-内核态共享环形队列"**。

---

## 十、零拷贝：内核和用户态的拉锯战

### 10.1 普通 read+write 走了几次拷贝

把磁盘文件发到网络上（典型 web server 场景）：

```
   普通 read + write（4 次搬运，其中 2 次是 CPU 硬搬）

   磁盘────DMA──▶内核页缓存 ─CPU拷─▶用户 buf ─CPU拷─▶内核 socket buf ─DMA─▶网卡
      ①                  ②↑         ③↑                       ④
                       (拷贝 1)    (拷贝 2)
```

四次“搬运”，其中 **②③ 两次是 CPU 在硬搬**——这就是普通的 `read() + write()`。100 Gbps 速率下，光 memcpy 就能把 CPU 打满。

### 10.2 zero-copy 的几种姿势

```
   sendfile：省掉 ②③

   磁盘────DMA──▶内核页缓存 ───内核内部───▶ socket buf ─DMA─▶网卡

   MSG_ZEROCOPY：省掉 ②，但要求用户不能马上修改 buf

   用户 buf ◀──────── (未修改) ─────────────── ─DMA─▶网卡
     ▲                                                  │
     └─── errqueue 通知: “发完了，你可以重用 buf” ◀───────┘
```

**`sendfile`**：内核内部直接从页缓存搬到 socket buf。

**`splice`**：通过管道把任意两个 fd 之间的数据在内核内移动，更通用。

**`MSG_ZEROCOPY`**：发送时网卡直接 DMA 用户态 buf。**前提**：用户态在内核异步完成发送前不能修改 buf——通过 errqueue 通知应用"好了你可以重用这个 buf 了"。

**`io_uring zero-copy`**：上面这些的统一接口。

### 10.3 钩子：内核为何愿意/不愿意信任用户态缓冲区

文里讨论"为什么不能让用户态直接提供缓冲区"——答案就在 `MSG_ZEROCOPY` 的限制里：**应用必须保证发送完成前不修改 buf**。普通 socket 编程模型里没有这个契约，所以默认行为是复制；提供了契约（用户愿意等内核通知）才能 zero-copy。

**RDMA 把这个契约做到了极致**：应用注册 MR 时承诺"我不会动这块内存，直到完成事件通知我为止"。换来的是数据面完全跳过内核 + 完全跳过 CPU 拷贝。

---

## 十一、内核旁路（Kernel Bypass）：一种通用模式

### 11.1 共同套路

DPDK（用户态网络栈）、SPDK（用户态存储栈）、ibverbs（RDMA 用户态库）有完全一致的架构：

1. **控制面经内核**：分配设备、注册内存、设置中断路由（慢路径，安全要紧）
2. **设备资源映射进用户态**：用 mmap 把设备的寄存器、队列直接暴露给用户进程
3. **数据面纯用户态**：work request 直接写共享队列，doorbell 是 MMIO 写，完成事件靠轮询 completion queue

```
              ┌────────────┐
   控制面 ─→ │   内核态    │  ← 设置好以后就不管了
              └─────┬──────┘
                    │  mmap 设备资源
                    ▼
   ┌──────────────────────────────┐
   │       用户态进程              │
   │  WQE → SQ doorbell → 设备    │   ← 数据面
   │       ↑                      │
   │   poll CQ ← 设备             │
   └──────────────────────────────┘
              ↑↓ DMA / IOMMU
              设备
```

### 11.2 控制面 vs 数据面的本质对比

| 维度 | 控制面（slow path） | 数据面（fast path） |
|---|---|---|
| 走哪条路 | syscall 进内核 | 用户态共享内存 + MMIO |
| 频率 | 启动时几次 | 每秒百万次 |
| 性能 | 不重要 | 决定整体吞吐 |
| 安全 | 内核检查 | 内核提前设好访问权限 |
| 例子 | 注册 MR、建立 QP | post send/recv、poll CQ |

> **这就是陈乐群 RDMA 文里"控制平面 vs 数据平面"的精确含义**。它不是 RDMA 的独家发明，而是"既要安全又要快"这个矛盾在系统设计上的通用解。

### 11.3 代价

绕过内核很爽，但有代价：

- **失去内核网络栈**：TCP、TLS、防火墙、Linux 路由——你要么自己实现，要么走特殊协议（RoCE/IB）
- **CPU 100% 占用**：busy poll 是必须的，但意味着这个核被"烧"掉了
- **应用变复杂**：内存生命周期、所有权、错误处理全要自己管

所以内核旁路只用在"必须榨干硬件"的场景。绝大多数业务代码用普通 socket 就够了。

---

## 十二、附录：常用 syscall 成本与术语速查

### 12.1 syscall 成本数量级（参考值，现代 x86-64）

| 操作 | 时间量级 |
|---|---|
| 空 syscall（`getpid`） | ~50 ns |
| `gettimeofday` (vDSO) | ~20 ns（不进内核） |
| `read` 缓冲区命中 | ~200 ns |
| `epoll_wait` 有事件 | ~500 ns |
| `read` 触发块设备 I/O | 几十 µs（NVMe） |
| 进程创建 `fork+exec` | ~1 ms |
| 一次 page fault（minor） | 1–5 µs |
| 一次 page fault（major，触发磁盘读） | 几百 µs – ms |

### 12.2 术语速查表

| 术语 | 一句话解释 |
|---|---|
| **Ring 0 / Ring 3** | CPU 的内核态 / 用户态 |
| **syscall** | 用户态请内核办事的唯一合法通道 |
| **task_struct** | Linux 里"线程/进程"的统一结构体 |
| **fork / exec / clone** | 创建进程 / 替换镜像 / 创建线程 |
| **mmap** | 把任意东西映射成内存 |
| **page fault** | 访问的虚拟页还没真实物理页时触发 |
| **COW** | 写时复制，fork 后默认行为 |
| **IRQ / softirq** | 硬中断 / 软中断 |
| **NAPI** | 网卡高负载下的轮询模式 |
| **CFS** | Linux 默认完全公平调度器 |
| **isolcpus / nohz_full** | 把核心从内核调度中"挖"出来 |
| **epoll ET/LT** | 边沿触发 / 水平触发 |
| **io_uring** | 用户态-内核态共享环形队列的异步 I/O 接口 |
| **sendfile / splice / MSG_ZEROCOPY** | 零拷贝家族 |
| **kernel bypass** | 数据面绕开内核的通用模式 |
| **MR / QP / CQ / WQE / Doorbell** | RDMA 词汇，全部对应“内核旁路”模式里的角色 |
| **Futex** | 用户态快路径 + 内核态慢路径的互斥锁 |
| **Spinlock** | 忙等锁，临界区极短时使用 |
| **RCU** | 读无锁、写延迟释放的同步机制 |
| **VFS** | 虚拟文件系统，统一所有文件系统的 API |
| **Page Cache** | 内核为磁盘文件维护的 DRAM 缓存 |
| **fsync** | 强制文件脏页写入磁盘 |
| **Direct I/O** | 绕过 Page Cache 直接读写块设备 |
| **Namespace** | Linux 进程隔离机制（六种） |
| **cgroups** | Linux 进程资源限额机制 |
| **容器** | namespace + cgroups + Union FS |

---

## 十三、下一步

到这里你应该能回答：syscall 为什么慢、内核旁路为什么可行、io_uring 和 RDMA 为什么长得像。但还有一块拼图——**网络协议本身**。下一篇我们从一根网线讲到 socket 编程，然后你就能从硬件、软件、协议三个方向同时夹击 RDMA 那篇文章里的每一个段落。

下一篇：[程序员的硬核基础（三）：计算机网络，从一根网线到 socket 编程](/posts/2026-06-15-cs-foundations-3-network/)。
