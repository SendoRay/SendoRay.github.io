---
title: "程序员的硬核基础（三）：计算机网络，从一根网线到 socket 编程"
date: '2026-06-15'
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

> 这篇文章不写"七层模型背诵题"，只回答一个问题：
>
> **为什么 TCP 慢、UDP 不可靠、socket API 简单？这些设计决定背后的取舍是什么？**
>
> 这是《程序员的硬核基础三件套》的第三篇。前置阅读：[体系结构篇](/posts/2026-06-13-cs-foundations-1-architecture/)、[操作系统篇](/posts/2026-06-14-cs-foundations-2-os/)。读完本篇你应该能回答：为什么 RDMA 要绕开 TCP，又为什么不能绕开"地址解析"和"拥塞控制"——它们换了形态继续存在。

---

## 一、五层模型一页纸

OSI 七层是教科书的，工业界实际只用 **TCP/IP 五层模型**：

| 层 | 职责一句话 | 典型协议 / 实体 | 地址 |
|---|---|---|---|
| **应用层** | 业务语义 | HTTP / gRPC / Redis / SQL | URL / API |
| **传输层** | 端到端可靠 / 端口区分 | TCP / UDP / QUIC | 端口号（16 bit） |
| **网络层** | 跨网段寻路 | IP / ICMP / IPv6 | IP 地址（32/128 bit） |
| **链路层** | 同一段物理链路上传一帧 | Ethernet / 802.11 / PCIe | MAC 地址（48 bit） |
| **物理层** | 比特怎么变成电/光信号 | 光纤、双绞线 | — |

**核心心智模型**：每一层只解决自己那一层的问题，并且**只信任直接相邻层的服务**。下层抽象出"我能传一个 X 单位"，上层就把自己的东西塞进 X 里——这叫**封装（encapsulation）**。一个 HTTP 请求发出去时，从上到下逐层加头、加完变成一串比特上线：

```
   应用层:                                  HTTP body
                                          ┌─────────────────┐
   传输层:                            TCPhdr│   HTTP body  │
                                    ┌──────────────────────┐
   网络层:                       IP hdr│  TCPhdr│   HTTP body  │
                            ┌────────────────────────────┐
   链路层:               Eth hdr│  IP hdr│  TCPhdr│   HTTP body  │  FCS
                       └────────────────────────────────────┘
                          14 B    20 B   20 B    ≤ 1460 B          4 B
   物理层:                  ▼
                       电子/光信号在光纤/双绞线上走
```

一个 1500 字节的以太网帧最多能装：1500 − 14（以太头）− 20（IP 头）− 20（TCP 头）= **1460 字节** payload。这个数叫 **MSS（Maximum Segment Size）**，记住它。

---

## 二、链路层：你以为是 IP 在走，其实是 MAC 在走

### 2.1 Ethernet 帧结构

```
| 目的 MAC (6) | 源 MAC (6) | EtherType (2) | Payload (46-1500) | FCS (4) |
```

一段以太网链路上，所有数据都包在这种帧里。**EtherType** 告诉接收方上层是什么：0x0800 是 IPv4、0x86DD 是 IPv6、0x8100 是 VLAN……

### 2.2 MAC 地址：链路层身份证

48 bit，固化在网卡里。它**只在同一段物理链路上有意义**——一旦帧被路由器转发出本网段，源 MAC 就被换成路由器自己的 MAC。所以 MAC 地址不能跨网段。

### 2.3 ARP：从 IP 找 MAC

应用程序只知道对方 IP，但发出去的帧必须填**目标 MAC**。怎么办？发一个**广播 ARP 请求**："谁是 192.168.1.10？请告诉 192.168.1.5"。目标主机回一个 ARP 响应，源主机把映射缓存进 ARP 表（`ip neighbor show` 能看见）：

```
   Host A (192.168.1.5)                            Host B (192.168.1.10)
   MAC: aa:aa:aa:aa:aa:aa                          MAC: bb:bb:bb:bb:bb:bb
          │                                              │
          │  广播: "Who has 192.168.1.10?               │
          │           Tell 192.168.1.5"                  │
          │  ──────────────────────────────────────────────▶ │
          │    目的 MAC = ff:ff:ff:ff:ff:ff             │
          │                                              │
          │  单播: "192.168.1.10 is at bb:bb.."          │
          │ ◀─────────────────────────────────────────────   │
          │                                              │
   ┌────┴─────┐                                            │
   │ ARP cache  │                                            │
   │ 1.10 → bb..│  以后发包直接查表、不再广播                  │
   └──────────┘                                            │
```

ARP 缓存通常几分钟过期。这就是为什么有时候第一个包会"卡一下"——在等 ARP 响应。

### 2.4 MTU 1500 的来历

以太网 v2 规范定的最大帧 payload。**整个互联网工程都在围绕这个数转**——TCP 选 MSS、IP 分片、MTU 探测，都是因为 1500 这个魔数。

数据中心可以开 **Jumbo Frame（9000 字节 MTU）** 减少包数量，但跨广域网通常还是 1500——任何一段中间链路 MTU 小于 1500 都会触发分片或者 PMTUD（Path MTU Discovery）。

> **钩子**：陈乐群 RDMA 文提到"地址解析"代价。它说的就是 ARP 这类 link-layer resolution——TCP 建连时一次解析就能缓存住，但**无连接的协议（如 UDP / RDMA UD 模式）每次发包都要查表**，这是个真实的延迟来源。

---

## 三、网络层：让数据跨过整个互联网

### 3.1 IP 头长什么样

IPv4 header 20 字节，最关键的字段：

- **Source / Destination IP**（4+4 字节）：寻路依据
- **Protocol**（1 字节）：6 = TCP，17 = UDP，1 = ICMP
- **TTL**（1 字节）：每经一个路由器减 1，到 0 就丢
- **Total Length**：包括 header 和 payload
- **Identification + Flags + Fragment Offset**：用来分片重组

### 3.2 路由表

每台主机和路由器都有一张路由表：

```
$ ip route
default via 10.0.0.1 dev eth0
10.0.0.0/24 dev eth0 proto kernel scope link
```

收到一个 IP 包，按 **最长前缀匹配** 找下一跳。下一跳如果在同一网段，就直接 ARP 拿目标 MAC；如果不在，就 ARP 拿默认网关的 MAC，把帧发给网关。

### 3.3 NAT：私网地址出墙

家用路由器、云上 VPC 都用 NAT。出方向把 source IP+Port 换成公网 IP+Port，记录映射；回包时反向替换。一个公网 IP 能藏住几万台内网机器。

### 3.4 MTU 与分片

如果上层给 IP 一个比 MTU 大的包，IP 层会把它**分片（fragmentation）**。每片是独立的 IP 包，到目的地由内核重组。分片很贵且脆弱：

- 任一片丢了整包重发
- 防火墙常常丢分片
- 现代 TCP 用 PMTUD 主动探测路径 MTU、避免分片
- UDP 大包是分片重灾区，QUIC 干脆要求一次写不超过 PMTU

---

## 四、TCP：可靠的代价

### 4.1 三次握手

```
Client                Server
  ──── SYN(seq=x) ────▶
  ◀── SYN(seq=y, ack=x+1) ──
  ──── ACK(ack=y+1) ──▶
                   状态 ESTABLISHED
```

这三次往返把双方的初始序号、窗口大小、拥塞控制参数同步好。**注意延迟成本**：建连本身要 1 个 RTT（同机房 0.1 ms，跨城 30 ms，跨洋 150 ms）。这就是为什么连接复用、HTTP/2 多路复用、连接池这么重要。

### 4.2 滑动窗口：流量控制

TCP 不是"一发一收"，而是**允许发出方塞 N 字节没被 ack 也继续发**。这个 N 就是窗口。接收方在 ack 里报告自己 buf 还剩多少（receive window），发送方按 min(rwnd, cwnd) 决定能发多少。

```
              已 ack          发出未 ack          可以发            不能发
        ────┼──────────────┼────────────────┼──────────────┼─────────────
   seq:    100         200            450             700
                       ▲                              ▲
                  snd_una                          snd_una + window
                       └──────── 窗口 = 500 B ────────┘

        ack=200 到达后，窗口整个右移 100 B，边界变成 [200, 700]
```

**带宽延迟积（Bandwidth-Delay Product）**：

$$
\text{BDP} = \text{带宽} \times \text{RTT}
$$

100 Gbps × 1 ms = 12.5 MB——这就是同机房想跑满 100 Gbps 时**TCP 发送窗口必须有 12.5 MB**。Linux 默认窗口才几 MB，跨城高带宽场景下不调参根本跑不满。

### 4.3 拥塞控制：BBR vs CUBIC 一句话

**CUBIC**（Linux 默认）：基于丢包。慢启动指数涨窗口，一丢包砍半，然后立方曲线再涨。**经典但在长肥管道+轻丢包时表现差**。

**BBR**（Google 推出）：基于带宽 + RTT 估计。主动测瓶颈带宽，把窗口控制在 BDP 附近。**跨大洋、有偶发丢包的链路上表现碾压 CUBIC**。

> 关键洞察：拥塞控制是 TCP 的**核心价值**之一——没有它互联网会立刻崩溃。但它也是性能开销之源。**RDMA 在 IB 协议上靠链路级流控（credit-based）取代它，在 RoCE 上靠 PFC + DCQCN，本质都是把"端到端反应"换成"链路级硬件反应"**——更快、但要求专门的网络硬件配合。

### 4.4 重传：可靠性来自这里

每个 TCP 段都有 sequence number。发送方设一个 RTO（Retransmission Timeout），超时没 ack 就重传。现代 TCP 还有 SACK（Selective ACK，告诉发送方"我收到了 1-100 和 200-300，少了 101-199"）和 Fast Retransmit（连续 3 个 dup ack 立刻重传）。

代价是：**应用永远不知道一个包到底是第几次才到的**——TCP 给的可靠性是有"延迟抖动"的。这对实时音视频、AI 训练通信都是致命的，所以这些场景宁可走 UDP / RDMA 自己处理。

### 4.5 TIME_WAIT：连接关闭的尾巴

`netstat` 看到一堆 TIME_WAIT？正常。TCP 主动关闭方会停留在这个状态 2×MSL（约 60 s），等收完所有迟到的包再彻底释放。短连接服务器上几十万 TIME_WAIT 是常态。

### 4.6 钩子：哪些"机制"恰恰是高性能场景想绕过的

总结一下 TCP 的开销来源：

| 机制 | 设计目的 | 高性能场景的态度 |
|---|---|---|
| 三次握手 | 状态同步 | 连接长期复用 / QUIC 0-RTT |
| 滑动窗口 | 流量控制 | RDMA 用硬件 credit |
| CUBIC/BBR 拥塞控制 | 互联网共享公平 | DC 内换 PFC/ECN 硬件级 |
| 重传 + SACK | 丢包恢复 | RDMA / IB 链路级低误码 |
| 内核协议栈 | 安全隔离 | DPDK / RDMA 用户态绕开 |

**不是 TCP 不好，是它的负担在数据中心场景里没必要**。

---

## 五、UDP：极简协议

### 5.1 头长什么样

UDP header 只有 8 字节：

```
| Source Port (2) | Dest Port (2) | Length (2) | Checksum (2) |
```

没了。没有连接、没有顺序保证、没有重传、没有拥塞控制。**应用程序自己负责所有这些**。

### 5.2 为什么数据中心反而爱 UDP

直觉上 UDP 不可靠，应该比 TCP 用得少。但实际上：

- **DNS**：一次请求一次响应，建连接亏
- **音视频**：丢一帧不要紧，迟到的帧反而碍事
- **QUIC（HTTP/3 的底层）**：在 UDP 上自己实现了可靠 + 多路复用 + 0-RTT，避开内核 TCP 的固化算法
- **AI 训练**：GPU 间 collectives 用的 NCCL 在 RoCE 模式下也是 UDP（其实是 RoCEv2 的 UDP 封装）
- **游戏**：低延迟最重要，丢包就丢，不能等

**核心原因**：内核 TCP 实现是"通用最优"，但每种应用都有自己的最优策略。UDP 给你一个**可编程的传输层**。

### 5.3 钩子：RoCEv2 = RDMA over UDP

RDMA 在以太网上跑（RoCEv2 模式）时，物理上就是 UDP 包。每个 RDMA 操作被封装成 UDP/IP 包，目的端口 4791。但**网卡硬件直接处理这些包**——内核根本看不到，应用也不会调用 socket。这是"用 UDP 端口号借道以太网，内容是完全自定义协议"的典型用法。

---

## 六、Socket API：一个 41 岁的接口为什么还活着

### 6.1 一张图看完所有 socket 调用

```
            服务端                          客户端
   socket()                           socket()
   bind(addr)                            │
   listen()                              │
   accept() ──┐                          │
              │  ◀── connect(addr) ──────│  3 次握手
              ▼                          ▼
            recv() ◀────── send() ──────send()
            send() ──────▶ recv() ──────▶recv()
            close()                    close()
```

七八个 syscall 把整个 TCP 状态机包装成几行代码——这是 Berkeley socket 留给世界的礼物。

### 6.2 为什么这个 API 必然带来一次复制

回到陈乐群 RDMA 文的核心论点：

```c
ssize_t send(int sockfd, const void* buf, size_t len, int flags);
```

这套接口的契约是：**`send` 返回后你可以立刻修改/释放 buf**。所以非阻塞模式下，内核必须在 syscall 里就把 buf 内容**复制到内核**——否则它没法答应这个契约。

要消除这次复制，要么改契约（`MSG_ZEROCOPY`，应用承诺等通知再改 buf），要么换一套 API（io_uring、ibverbs）。**每一个号称"零拷贝"的方案，本质都是在重新定义所有权契约**。

### 6.3 阻塞 vs 非阻塞 vs epoll：操作系统篇的回响

操作系统篇里讲过 I/O 模型。在 socket 上具体落地：

```c
// 阻塞：一个连接一个线程，10K 连接顶不住
int fd = accept(sfd, NULL, NULL);
recv(fd, buf, len, 0);   // 没数据就睡

// 非阻塞 + epoll：一个线程吃 10 万连接
int efd = epoll_create1(0);
epoll_ctl(efd, EPOLL_CTL_ADD, fd, &ev);
while (1) {
    int n = epoll_wait(efd, events, MAX, -1);
    for (int i = 0; i < n; i++) {
        recv(events[i].data.fd, buf, len, 0);   // 立刻能读
    }
}
```

Nginx、Redis、Netty 都是后面这种结构。

---

## 七、延迟 vs 带宽：高带宽时代的窗口

### 7.1 RTT 决定了一切等待型操作

| 场景 | 典型 RTT |
|---|---|
| 同机不同进程 (loopback) | < 50 µs |
| 同 ToR 交换机 | 0.1 ms |
| 同机房不同机架 | 0.2–0.5 ms |
| 同城不同机房 | 1–3 ms |
| 跨城（北京-上海） | 30 ms |
| 跨大洋（中-美） | 150 ms |
| 卫星 | 600 ms |

**任何"一来一回"的协议步骤都至少花 1 个 RTT**。三次握手、TLS 握手、HTTP/1 一次请求——全是 RTT 的倍数。

### 7.2 带宽延迟积再算一次

跨城 30 ms RTT × 10 Gbps 带宽 = **37.5 MB**。你的 TCP 发送窗口必须有 37.5 MB 才能跑满。Linux 内核默认 `tcp_rmem`/`tcp_wmem` 上限大约 6 MB——意味着默认配置下跨城你顶天跑 1.6 Gbps，浪费了 84% 的带宽。

调参方法：

```bash
sysctl -w net.core.rmem_max=67108864
sysctl -w net.core.wmem_max=67108864
sysctl -w net.ipv4.tcp_rmem='4096 87380 67108864'
sysctl -w net.ipv4.tcp_wmem='4096 87380 67108864'
```

### 7.3 钩子：3200 Gbps 网络的窗口

如果是同机房 0.1 ms RTT × 3200 Gbps：BDP = 40 MB。这数量级的窗口 + 32 个并发流，正是陈乐群文里 AWS p5 那 32 张 EFA 网卡能聚合的物理意义。

---

## 八、数据中心网络：跟你想的不太一样

### 8.1 Spine-Leaf 架构

数据中心不是树形，而是 **Clos 网络 / Spine-Leaf**：

```
   Spine 层 (高位交换机)
     ┌─────────────────────────────────┐
     │ [SP1]   [SP2]   [SP3]   [SP4]    │
     └───┬─────┬─────┬─────┬─────────────┘
         │ ╲╱ │ ╲╱ │ ╲╱ │           ← 每个 Leaf 全连所有 Spine
         │ ╱╲ │ ╱╲ │ ╱╲ │
     ┌───┴─────┴─────┴─────┴─────────────┐
     │ [LF1]   [LF2]   [LF3]   [LF4]    │   Leaf 层 (ToR)
     └───┬─────┬─────┬─────┬─────────────┘
         │     │     │     │
         ▼     ▼     ▼     ▼
        机架   机架  机架   机架
      (服务器)(服务器)(服务器)(服务器)
```

**任意两台服务器之间都有多条等价路径**（LF1 → SP1/SP2/SP3/SP4 任选一条 → LF3）。靠 **ECMP（Equal-Cost Multi-Path）** 按 5-tuple hash 把不同流分到不同路径上——一台机器内部一个流跑一条路径，但成千上万的流被均匀打散。

这就是为什么数据中心总带宽远大于单链路带宽——并行才是关键。

### 8.2 RoCE vs InfiniBand

RDMA 有两条物理路：

- **InfiniBand**：原生 RDMA 协议，专用交换机和网卡，链路级流控（credit-based），几乎无丢包。HPC 老牌选择。
- **RoCEv2**：在 UDP/IP 上跑 RDMA，复用以太网。需要 **PFC（Priority Flow Control）** 防丢包 + **DCQCN** 做拥塞控制。AWS EFA、Azure 大多走这条。

**为什么 RDMA 这么怕丢包**：RDMA 没有 TCP 那种"只丢一段、SACK 重传"的机制；丢一个包整个 connection 进入复杂恢复，性能掉一个数量级。所以 RDMA 网络通常是**无损以太网（lossless Ethernet）**。

### 8.3 3200 Gbps 是怎么来的

AWS p5 单机 32 张 100 Gbps EFA 网卡——单机出口聚合带宽 **3200 Gbps = 400 GB/s**。把 32 张网卡分成 8 组，每组 4 张配 1 张 H100，让 H100 通过 GPUDirect RDMA 同时用这 4 张网卡（多路并行）跟远端通信。这不是"一根 3200 Gbps 管子"，而是"32 根 100 Gbps 管子并联"。

---

## 九、常用调试工具

### 9.1 一句话用法表

| 工具 | 一句话 | 经典用法 |
|---|---|---|
| `ping` | 测连通性 + RTT | `ping -c 4 host` |
| `traceroute` | 看到底走了哪些路由器 | `traceroute host` / `mtr host` |
| `iperf3` | 测两点间吞吐 | server: `iperf3 -s` / client: `iperf3 -c host` |
| `ss` | 看本机所有 socket 状态 | `ss -tan` 看 TCP / `ss -lntp` 看监听 |
| `netstat -i` | 看网卡丢包错包计数 | 持续观察 RX-DRP/TX-ERR |
| `tcpdump` | 抓包 | `tcpdump -i eth0 -nn 'port 80'` |
| `wireshark / tshark` | 解析包 | 与 tcpdump 配合 |
| `ethtool` | 看/调网卡能力 | `ethtool -S eth0` 看硬件计数 |
| `ibv_devinfo` / `ibstat` | 看 RDMA 设备 | InfiniBand/RoCE 调试 |
| `bpftrace / bcc` | 内核网络栈追踪 | `tcpconnect`、`tcptracer` |

### 9.2 排错思路

性能慢/不通问题，按这个顺序排：

1. **物理层**：网卡 link 起来没？`ethtool eth0` 看 Speed/Duplex
2. **链路层**：ARP 表对吗？`ip neighbor`
3. **网络层**：路由对吗？`ip route get <ip>`
4. **传输层**：端口通吗？`telnet host port` / `nc -zv`
5. **应用层**：日志说啥？

很多看似"应用 bug"最后查到是 IRQ 不均、MTU 不一致、TCP 窗口太小、DNS 解析慢——网络问题永远要从下往上查。

---

## 十、附录：协议头速查 + TCP 状态机 + 术语表

### 10.1 头部字段长度速查

| 协议 | 头部长度 | 关键字段 |
|---|---|---|
| Ethernet | 14 B | 源/目的 MAC、EtherType |
| IPv4 | 20 B | 源/目的 IP、Protocol、TTL |
| IPv6 | 40 B | — |
| TCP | 20 B | 源/目的端口、seq、ack、window、flags |
| UDP | 8 B | 源/目的端口、长度、校验 |

以太网帧典型: **14 + 20 + 20 = 54 B 头**，剩下 1446 B 装 payload。

### 10.2 TCP 状态机（精简版）

```
                CLOSED
                  │ connect / accept
                  ▼
              SYN_SENT / SYN_RCVD
                  │
                  ▼
              ESTABLISHED   ◀──── 数据传输都在这里
              ╱        ╲
   主动关闭                被动关闭
   FIN_WAIT_1              CLOSE_WAIT
       │                       │
   FIN_WAIT_2                LAST_ACK
       │                       │
   TIME_WAIT (60s)           CLOSED
       │
     CLOSED
```

记住：**主动关闭方走 TIME_WAIT；被动关闭方走 CLOSE_WAIT**。CLOSE_WAIT 堆积通常意味着应用没正确关 fd——这是个常见 bug。

### 10.3 术语速查表

| 术语 | 一句话 |
|---|---|
| **MSS / MTU** | TCP 段最大 / 帧最大 |
| **MSL** | Maximum Segment Lifetime，约 30 s |
| **RTT** | Round-Trip Time |
| **BDP** | 带宽 × RTT，决定窗口 |
| **ARP** | IP 找 MAC 的协议 |
| **NAT** | 私网地址映射成公网 |
| **TIME_WAIT** | 主动关闭后等 2×MSL |
| **CLOSE_WAIT** | 被动方还没 close |
| **三次握手 / 四次挥手** | 建连 / 拆连流程 |
| **SACK** | 选择性 ack |
| **CUBIC / BBR** | Linux 默认 / Google 拥塞控制 |
| **PMTUD** | 路径 MTU 探测 |
| **ECMP** | 多路径等价分流 |
| **Spine-Leaf** | 数据中心标准拓扑 |
| **PFC** | 优先级流控，无损以太网基础 |
| **DCQCN** | RoCE 的拥塞控制 |
| **RoCEv2** | RDMA over UDP/IP |
| **InfiniBand** | RDMA 原生协议 / 网络 |
| **QUIC** | UDP 上的可靠协议，HTTP/3 底层 |

---

## 十一、下一步

到这里你已经有了硬件、操作系统、网络三套语言。下一篇我们回到陈乐群那篇 RDMA 长文，**逐段照亮**——他每说一句让你"无力"的细节，我们都用前三篇里的概念翻译一遍。读完那篇收尾文，你应该能合上文章自己重述整套 RDMA 设计哲学。

下一篇：[程序员的硬核基础（四）：用前三篇重读《驾驭 3200Gbps 网络》](/posts/2026-06-16-cs-foundations-4-rdma-recap/)。
