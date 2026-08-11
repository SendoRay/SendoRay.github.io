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
> 这是《程序员的硬核基础四部曲》的第三篇。前置阅读：[体系结构篇](/posts/2026-06-13-cs-foundations-1-architecture/)、[操作系统篇](/posts/2026-06-14-cs-foundations-2-os/)。读完本篇你应该能回答：为什么 RDMA 要绕开 TCP，又为什么不能绕开"地址解析"和"拥塞控制"——它们换了形态继续存在。

---

> **系列导航 · 程序员的硬核基础四部曲**
>
> | 篇 | 负责讲清 | 不负责 |
> |---|---|---|
> | [一：体系结构](/posts/2026-06-13-cs-foundations-1-architecture/) | CPU 微架构、缓存与内存序、页表/TLB、PCIe/NVLink、NUMA 硬件拓扑、GPU 硬件 | OS 如何调度使用这些硬件 |
> | [二：操作系统](/posts/2026-06-14-cs-foundations-2-os/) | 进程/线程、同步原语、虚拟内存与 Page Cache、容器、I/O 模型、零拷贝、内核旁路 | 具体网络协议细节 |
> | [三：计算机网络](/posts/2026-06-15-cs-foundations-3-network/) | 五层协议栈、TCP/UDP/QUIC、DNS/HTTP/TLS、数据中心网络、RSS/XDP/DPDK | 上层通信库与分布式框架 |
> | [四：通信](/posts/2026-06-16-cs-foundations-4-communicate/) | DMA/Pinned Memory、RDMA 实战、NCCL 等通信库、集合通信、RPC/gRPC | 硬件拓扑细节（见篇一） |

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

> **钩子**："地址解析"代价。它说的就是 ARP 这类 link-layer resolution——TCP 建连时一次解析就能缓存住，但**无连接的协议（如 UDP / RDMA UD 模式）每次发包都要查表**，这是个真实的延迟来源。

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

## 四、DNS：互联网的电话簿

到这里你已经知道一台机器怎么把帧送到下一跳、IP 包怎么跨过整个互联网。但有一个前提一直被悄悄默认了：**你怎么知道对方的 IP？** 应用程序看到的永远是 `mail.google.com` 这种字符串，不是 `93.184.216.34`。中间这层翻译，就是 DNS。

### 4.1 域名层级：一棵倒过来的树

域名是一棵层级树，从右往左读：

```
                          .  (根域，写出来是个空字符)
                          │
              ┌───────────┼─────────────┐
            .com         .org           .cn        ← 顶级域 TLD
              │
        ┌─────┴────────┐
     google.com   example.com                       ← 二级域
        │
   mail.google.com / drive.google.com               ← 三级域
```

每一级都有自己的**权威 DNS 服务器**——只有它能回答"我这一级下面的子域应该问谁"。根 DNS 全球只有 13 组（A-M），由几家机构维护。

### 4.2 一次完整查询走过的路

假设你在浏览器输入 `example.com`，本地 OS 没缓存，会发生这样一连串问询：

```
 用户程序                 本地递归DNS           根DNS        .com DNS      example.com权威DNS
   │                      (ISP/公司)
   │ "example.com的IP?"  │
   │─────────────────────▶│
   │                      │ "." 在哪？(已知)
   │                      │──────────────────▶│
   │                      │ ".com 找 xx"      │
   │                      │◀──────────────────│
   │                      │─────────────────────────────▶│
   │                      │ "example.com 找 yy"          │
   │                      │◀─────────────────────────────│
   │                      │──────────────────────────────────────────▶│
   │                      │ "A 记录 = 93.184.216.34"                  │
   │                      │◀──────────────────────────────────────────│
   │ "93.184.216.34"     │
   │◀─────────────────────│
```

注意区分两种查询模式：

- **递归查询**：客户端 → 本地递归 DNS。"我不管你怎么查，给我一个最终答案。"
- **迭代查询**：本地 DNS → 根 / TLD / 权威。"我自己一步步问下去，每次只往下走一级。"

这么设计是为了让本地 DNS 替成千上万的客户端做"压缩中转"——根服务器永远不需要直接面对终端用户，只需要面对全球几万台递归 DNS。

### 4.3 常用记录类型

| 类型 | 含义 | 示例 |
|---|---|---|
| **A** | 域名 → IPv4 | `example.com → 93.184.216.34` |
| **AAAA** | 域名 → IPv6 | — |
| **CNAME** | 别名 → 真名 | `www.example.com → example.com` |
| **MX** | 邮件服务器 | `example.com → mail.example.com` |
| **NS** | 管这个域的 DNS 服务器是谁 | — |
| **TXT** | 任意文本（SPF / 域名验证 / 配置） | — |

CNAME 经常用来把 `www.xxx.com` 指到 CDN 域名上，迁 CDN 时不用动 A 记录。

### 4.4 缓存与 TTL

DNS 不可能每次都跑一遍上面那张图——那互联网早就崩了。所以每一层都有缓存：

```
浏览器缓存  →  OS 缓存（含 /etc/hosts，最优先）  →  本地递归 DNS 缓存  →  ISP 缓存
```

每条 DNS 记录都带一个 **TTL（Time To Live）**——这条结果可以被缓存多久。到期后下游必须重新查询。

- **低 TTL（如 60 s）**：迁移快，代价是查询多、根服务器和权威 DNS 压力大
- **高 TTL（如 86400 s = 1 天）**：查询少、稳定，代价是改一次 IP 全网生效要等一天

### 4.5 工程后果

1. **服务迁移前先降 TTL**。如果你计划下周切 IP，今天就把 TTL 从 1 天改成 60 秒；等旧 TTL 自然过期后再改 A 记录。否则有些客户端会缓存到旧 IP 一整天。
2. **`nslookup` / `dig` 是 DNS 调试的标配**。`dig +trace example.com` 能模拟完整的迭代查询过程。
3. **生产环境"莫名其妙连不上"**，第一排查方向永远是 DNS——不是网线、不是防火墙。`ping IP` 能通但 `ping 域名` 不通，基本就是 DNS 出问题了。

> **钩子**：数据中心内部不用公网 DNS——而是 CoreDNS（Kubernetes 默认）或 Consul 做服务发现。但它们的工作原理完全一样：一个名字 → 一个地址的映射 + 缓存 + TTL。把"DNS"换成"服务发现"后，所有公网 DNS 的工程经验都还能用。

### 4.6 实操：dig +trace 亲眼看一次完整递归解析

4.2 那张问询图不用背，一条命令就能亲眼看一遍（无需 root，任何 Linux/macOS 都行）：

```bash
dig +trace example.com

# ;; 第一段：本地递归 DNS 返回根服务器列表
# .            518400  IN  NS  a.root-servers.net.
# .            518400  IN  NS  b.root-servers.net.
# ;; Received 525 bytes from 10.0.0.2#53(10.0.0.2) in 2 ms
#
# ;; 第二段：根服务器把 .com 委派给 TLD 服务器
# com.         172800  IN  NS  a.gtld-servers.net.
# ;; Received 1170 bytes from 198.41.0.4#53(a.root-servers.net) in 145 ms
#
# ;; 第三段：.com TLD 委派给 example.com 的权威 DNS
# example.com. 172800  IN  NS  a.iana-servers.net.
# ;; Received 76 bytes from 192.5.6.30#53(a.gtld-servers.net) in 32 ms
#
# ;; 第四段：权威 DNS 给出最终 A 记录
# example.com. 300     IN  A   93.184.216.34
# ;; Received 56 bytes from 199.43.135.53#53(a.iana-servers.net) in 160 ms
```

逐段解读：每一段是一次**迭代查询**，`NS` 记录就是"下一级去问谁"的委派链——根 → `.com` TLD → 权威，三跳定位。注意 TTL 的差别：委派记录 TTL 高达 172800 s（2 天，NS 极少变动），最终 A 记录只有 300 s（5 分钟，方便迁移）。`from` 后面的 IP 和耗时告诉你每一跳问的是谁、花了多久——145 ms 那跳就是跨洋到根服务器的真实 RTT。

---

## 五、HTTP 与 TLS：应用层的门面

DNS 帮你把名字翻译成 IP，IP + TCP 把字节流送到对端。剩下最后一公里：**应用层用什么格式说话？** 90% 的互联网答案是 HTTP，外加一层 TLS 加密。这一节讲 HTTP/1.1 → HTTP/2 → HTTP/3 的演进，每一步都在解决前一代的瓶颈。

### 5.1 HTTP/1.1：文本时代

最朴素的 HTTP 就是一段文本：

```
GET /index.html HTTP/1.1\r\n
Host: example.com\r\n
User-Agent: curl/7.79.1\r\n
\r\n
```

服务器回一段同样格式的文本。简单到可以 telnet 手敲。

但 HTTP/1.1 有两个原罪：

- **一个 TCP 连接同一时刻只能跑一个请求**。即使用 keep-alive 复用连接，多个请求也是**串行**的——前一个请求还没回完，后一个不能发。这叫**应用层队头阻塞（HOL blocking）**。
- 浏览器只好对每个域名开 6 个 TCP 连接来"并行"——结果是建连开销和拥塞控制各跑各的，整体吞吐反而被分割。

### 5.2 HTTP/2：二进制帧 + 多路复用

HTTP/2 把文本协议改成了**二进制 frame**：

- 一个 TCP 连接上同时跑多条 **stream**（逻辑上的请求/响应）
- 每个 stream 切成多个 frame，frame 上带 stream ID，乱序发、按 ID 重组
- **HPACK 头部压缩**：相同的 header（`User-Agent`、`Cookie`）只发一次，后续用编号引用
- **Server Push**：服务器主动把客户端"还没要但马上会要"的资源推过去（实际生态里用得不多）

但 HTTP/2 没解决一个更深的问题：**TCP 层的队头阻塞**。一旦底层 TCP 丢一个包，所有 stream 都要等这个包重传完才能继续——上层并行了，下层还是串行的。

### 5.3 HTTP/3 与 QUIC：在 UDP 上重建一切

HTTP/3 干脆放弃了 TCP，改用 **QUIC**（基于 UDP）：

```
QUIC = UDP + TLS 1.3 + 可靠传输 + 多路复用
```

它在用户态把 TCP 的可靠性、TLS 的加密、HTTP/2 的多路复用全部重新实现了一遍。带来的好处：

- **解决 TCP 队头阻塞**：每个 stream 独立维护重传，丢一个 stream 的包不影响别的
- **0-RTT 建连**：首次握手 1 个 RTT，重连可以做到 0 个 RTT（带着上次的密钥就能发数据）
- **连接迁移**：QUIC 用 Connection ID 而非 IP+端口四元组识别连接——你从 WiFi 切到 4G、IP 变了，连接也不会断

为什么不一开始就用 UDP 改 TCP？因为内核 TCP 的算法早已固化，全球设备升级要十年；QUIC 在用户态实现，浏览器/服务器自己迭代就行。

### 5.4 TLS 1.3 握手

不管 HTTP/1/2/3，跑在公网上都要套 TLS。TLS 1.3 把握手压缩到了 1 个 RTT：

```
Client                                   Server
  │── ClientHello (支持的密码套件) ──────▶│
  │                                       │
  │◀─ ServerHello + Certificate          │
  │   + Finished ─────────────────────────│
  │                                       │
  │── Finished ──────────────────────────▶│
  │                                       │
  │◀════════ 加密通道建立 ════════════════▶│

  总共 1-RTT（TLS 1.2 要 2-RTT）
```

关键设计点：

- **密钥交换走 ECDHE**：每次握手生成一对临时密钥，握手完就丢——即使私钥日后泄露，过去抓的流量也解不开（**前向安全 forward secrecy**）
- **证书验证靠 CA 签名链**：浏览器内置一组根 CA 公钥，服务器证书必须能从根 CA 一路签下来才被信任
- **工程后果**：HTTPS 比 HTTP 多 1 个 RTT 的握手开销。同机房可以忽略，跨城多 30 ms，跨洋多 150 ms——这就是为什么 CDN、连接复用、0-RTT 这么重要

### 5.5 三代 HTTP 对照

| 特性 | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---|---|---|---|
| 传输层 | TCP | TCP | **UDP (QUIC)** |
| 多路复用 | 无 | 有 | 有 |
| 头部压缩 | 无 | HPACK | QPACK |
| 队头阻塞 | 应用层 + 传输层 | 传输层 | **无** |
| 加密 | 可选 | 实际必须 | **强制内置** |
| 0-RTT 重连 | — | — | **支持** |

### 5.6 工程后果与钩子

- **gRPC 必选 HTTP/2**：因为它需要多路复用 + 双向流 + 头部压缩，HTTP/1.1 根本撑不起 RPC 高并发
- **AI 推理 API**（OpenAI、Anthropic 等）全走 HTTPS——既要安全，又要 HTTP/2 多路复用让一个连接上同时挂几十路流式响应
- **为什么不是 HTTP/3**？生态还在早期，企业内的中间件、负载均衡器、抓包工具对 QUIC 的支持参差不齐；公网消费场景（CDN、视频）已经大规模上了，但内部服务一般还是 HTTP/2

> **钩子**：HTTP/QUIC 追求的"减少 RTT、多路复用、避免队头阻塞"在数据中心有更激进的解法——RDMA 直接绕过整个协议栈，连 TCP/UDP/HTTP 这些字母都不需要。但它只在无损网络里能工作。一边是公网"什么都得自己处理"，一边是数据中心"硬件帮你处理"，殊途同归。

### 5.7 gRPC 最小示例：从 .proto 到一次远程调用

上面说"gRPC 必选 HTTP/2"，光说不练假把式。gRPC 的开发流程是三步：`.proto` 文件定义接口 → 代码生成器生成桩代码 → 填业务逻辑。一个最小 Echo 服务只要 5 行 proto：

```protobuf
syntax = "proto3";
service Echo {
  rpc Say (Msg) returns (Msg);
}
message Msg { string text = 1; }
```

生成 Python 桩代码（需 `pip install grpcio grpcio-tools`）：

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. echo.proto
# 生成两个文件：
# echo_pb2.py       ← 消息类（Msg 的序列化/反序列化）
# echo_pb2_grpc.py  ← 服务桩（EchoServicer 基类 + EchoStub 客户端）
```

服务端 12 行：

```python
import grpc, echo_pb2, echo_pb2_grpc
from concurrent import futures

class EchoService(echo_pb2_grpc.EchoServicer):
    def Say(self, request, context):
        return echo_pb2.Msg(text=f"echo: {request.text}")

server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
echo_pb2_grpc.add_EchoServicer_to_server(EchoService(), server)
server.add_insecure_port("[::]:50051")
server.start()
server.wait_for_termination()
```

客户端 6 行：

```python
import grpc, echo_pb2, echo_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = echo_pb2_grpc.EchoStub(channel)
reply = stub.Say(echo_pb2.Msg(text="hello"))
print(reply.text)   # 输出: echo: hello
```

抓包看的话，这次调用就是一个 HTTP/2 POST，路径 `/Echo/Say`，body 是 protobuf 二进制。关键在 `channel`：**一个 channel 底下是一条 TCP 连接，上面可以同时跑成百上千个并发 RPC**——每个 RPC 是一条 HTTP/2 stream，正是 5.2 节多路复用的直接受益者。AI 基础设施里到处是它：vLLM 分布式控制面、Triton/KServe 的推理接口，控制面协议基本都是 gRPC。

### 5.8 QUIC 上手：一条 curl 命令的差距

HTTP/3 也不用背，curl 直接对比（需 curl ≥ 8.x 且编译了 HTTP/3 支持，`curl -V` 输出里有 `HTTP3` 才行）：

```bash
curl --http2 -v https://cloudflare-quic.com -o /dev/null 2>&1 | grep -E 'Trying|SSL conn|using HTTP'
# *   Trying 104.16.102.207:443...          ← TCP 三次握手，1 RTT
# * SSL connection using TLSv1.3            ← TLS 握手，再 1 RTT
# * using HTTP/2                            ← 之后才能发第一个请求
# 总计：先花 2 个 RTT 才开始传数据（TLS 1.2 则是 3 个）

curl --http3-only -v https://cloudflare-quic.com -o /dev/null 2>&1 | grep -E 'Trying|QUIC|using HTTP'
# *   Trying 104.16.102.207:443...
# * Connected ... using QUIC                ← QUIC 握手把传输 + 加密合并成一步
# * using HTTP/3                            ← 1 个 RTT 后就能发请求
```

差距的来源：TCP 和 TLS 是两个独立协议，只能**串行握手**（TCP 1 RTT + TLS 1 RTT）；QUIC 把加密内置进传输层，一次握手同时干两件事，**首连 1 RTT**。跨洋 150 ms RTT 的场景下，这就是省 150~300 ms 的体感差距。

重连更狠：QUIC 支持 **0-RTT**——带着上次的会话密钥，第一个包就捎带应用数据。代价是 0-RTT 数据可能被攻击者截获重放，所以只能承载幂等请求（GET 可以，转账 POST 不行）。

| 对比项 | TCP + TLS 1.3 | QUIC |
|---|---|---|
| 首次建连 | 2 RTT（TLS 1.2 为 3） | **1 RTT** |
| 重连 | 1~2 RTT | **0 RTT**（有重放风险） |
| 队头阻塞 | TCP 层仍有 | 无 |

---

## 六、TCP：可靠的代价

### 6.1 三次握手

```
Client                Server
  ──── SYN(seq=x) ────▶
  ◀── SYN(seq=y, ack=x+1) ──
  ──── ACK(ack=y+1) ──▶
                   状态 ESTABLISHED
```

这三次往返把双方的初始序号、窗口大小、拥塞控制参数同步好。**注意延迟成本**：建连本身要 1 个 RTT（同机房 0.1 ms，跨城 30 ms，跨洋 150 ms）。这就是为什么连接复用、HTTP/2 多路复用、连接池这么重要。

### 6.2 滑动窗口：流量控制

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

### 6.3 拥塞控制：BBR vs CUBIC 一句话

**CUBIC**（Linux 默认）：基于丢包。慢启动指数涨窗口，一丢包砍半，然后立方曲线再涨。**经典但在长肥管道+轻丢包时表现差**。

**BBR**（Google 推出）：基于带宽 + RTT 估计。主动测瓶颈带宽，把窗口控制在 BDP 附近。**跨大洋、有偶发丢包的链路上表现碾压 CUBIC**。

> 关键洞察：拥塞控制是 TCP 的**核心价值**之一——没有它互联网会立刻崩溃。但它也是性能开销之源。**RDMA 在 IB 协议上靠链路级流控（credit-based）取代它，在 RoCE 上靠 PFC + DCQCN，本质都是把"端到端反应"换成"链路级硬件反应"**——更快、但要求专门的网络硬件配合。

### 6.4 重传：可靠性来自这里

每个 TCP 段都有 sequence number。发送方设一个 RTO（Retransmission Timeout），超时没 ack 就重传。现代 TCP 还有 SACK（Selective ACK，告诉发送方"我收到了 1-100 和 200-300，少了 101-199"）和 Fast Retransmit（连续 3 个 dup ack 立刻重传）。

代价是：**应用永远不知道一个包到底是第几次才到的**——TCP 给的可靠性是有"延迟抖动"的。这对实时音视频、AI 训练通信都是致命的，所以这些场景宁可走 UDP / RDMA 自己处理。

### 6.5 TIME_WAIT：连接关闭的尾巴

`netstat` 看到一堆 TIME_WAIT？正常。TCP 主动关闭方会停留在这个状态 2×MSL（约 60 s），等收完所有迟到的包再彻底释放。短连接服务器上几十万 TIME_WAIT 是常态。

### 6.6 钩子：哪些"机制"恰恰是高性能场景想绕过的

总结一下 TCP 的开销来源：

| 机制 | 设计目的 | 高性能场景的态度 |
|---|---|---|
| 三次握手 | 状态同步 | 连接长期复用 / QUIC 0-RTT |
| 滑动窗口 | 流量控制 | RDMA 用硬件 credit |
| CUBIC/BBR 拥塞控制 | 互联网共享公平 | DC 内换 PFC/ECN 硬件级 |
| 重传 + SACK | 丢包恢复 | RDMA / IB 链路级低误码 |
| 内核协议栈 | 安全隔离 | DPDK / RDMA 用户态绕开 |

**不是 TCP 不好，是它的负担在数据中心场景里没必要**。

### 6.7 实战：ss -ti 读懂一条连接的健康状况

本章讲的窗口、拥塞控制、重传都不是玄学，Linux 上一条命令全能看到（无需 root）：

```bash
ss -ti dst 10.0.0.8
# State  Recv-Q  Send-Q  Local Address:Port   Peer Address:Port
# ESTAB  0       36720   10.0.0.5:44612       10.0.0.8:5201
#      cubic wscale:7,7 rto:204 rtt:1.85/0.32 mss:1448 cwnd:512
#      ssthresh:380 bytes_acked:8812340224 retrans:0/17
#      pacing_rate 6.4Gbps delivery_rate 3.2Gbps busy:212450ms
```

逐字段解读（数字是同机房 iperf3 打流时的典型值）：

| 字段 | 含义 | 怎么读 |
|---|---|---|
| `cubic` | 该连接的拥塞控制算法 | 想换 BBR：`sysctl net.ipv4.tcp_congestion_control` |
| `wscale:7,7` | 窗口缩放因子（本端,对端） | TCP 头窗口字段只有 16 bit，左移 7 位才能表达 >64 KB 的窗口 |
| `rto:204` | 重传超时（ms） | 下限 200 ms 左右；RTO 频繁触发说明丢包严重 |
| `rtt:1.85/0.32` | 平滑 RTT / 抖动（ms） | 1.85 ms 是同机房跨机架的正常水平 |
| `cwnd:512` | 拥塞窗口（单位：MSS 个数） | 512 × 1448 B ≈ 741 KB 在途数据上限 |
| `ssthresh:380` | 慢启动阈值 | cwnd 低于它指数涨、高于它线性涨；有值说明曾经丢过包 |
| `retrans:0/17` | 当前在重传 / 累计重传段数 | 累计 17 相对 88 亿字节可忽略；持续增长就要查了 |
| `pacing_rate` | 内核平滑发包速率 | 一般是当前吞吐的 2 倍（给窗口增长留空间） |
| `delivery_rate` | 实测交付速率 | **最接近"真实吞吐"的数字** |

诊断口诀：**rtt 高→路径问题；cwnd 上不去→丢包或 ssthresh 压制；retrans 涨→网络质量；delivery_rate 低但以上都正常→接收窗口不够**。

接收窗口的上限由内核参数决定：

```bash
sysctl net.ipv4.tcp_rmem
# net.ipv4.tcp_rmem = 4096 131072 6291456
#                     ↑最小  ↑默认   ↑最大（6 MB）
```

三个值是每条连接接收缓冲的最小/默认/最大字节数，内核在此范围内自动调节。**最大值 6 MB 就是第九章 BDP 撞上的那个天花板**——跨城 30 ms × 10 Gbps 需要 37.5 MB 窗口，默认配置直接把吞吐封死在 1.6 Gbps。怎么调、调多大，见第九章 9.2 节，公式和参数这里不重复。

---

## 七、UDP：极简协议

### 7.1 头长什么样

UDP header 只有 8 字节：

```
| Source Port (2) | Dest Port (2) | Length (2) | Checksum (2) |
```

没了。没有连接、没有顺序保证、没有重传、没有拥塞控制。**应用程序自己负责所有这些**。

### 7.2 为什么数据中心反而爱 UDP

直觉上 UDP 不可靠，应该比 TCP 用得少。但实际上：

- **DNS**：一次请求一次响应，建连接亏
- **音视频**：丢一帧不要紧，迟到的帧反而碍事
- **QUIC（HTTP/3 的底层）**：在 UDP 上自己实现了可靠 + 多路复用 + 0-RTT，避开内核 TCP 的固化算法
- **AI 训练**：GPU 间 collectives 用的 NCCL 在 RoCE 模式下也是 UDP（其实是 RoCEv2 的 UDP 封装）
- **游戏**：低延迟最重要，丢包就丢，不能等

**核心原因**：内核 TCP 实现是"通用最优"，但每种应用都有自己的最优策略。UDP 给你一个**可编程的传输层**。

### 7.3 钩子：RoCEv2 = RDMA over UDP

RDMA 在以太网上跑（RoCEv2 模式）时，物理上就是 UDP 包。每个 RDMA 操作被封装成 UDP/IP 包，目的端口 4791。但**网卡硬件直接处理这些包**——内核根本看不到，应用也不会调用 socket。这是"用 UDP 端口号借道以太网，内容是完全自定义协议"的典型用法。

---

## 八、Socket API：一个 41 岁的接口为什么还活着

### 8.1 一张图看完所有 socket 调用

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

### 8.2 为什么这个 API 必然带来一次复制

回到一个核心论点：

```c
ssize_t send(int sockfd, const void* buf, size_t len, int flags);
```

这套接口的契约是：**`send` 返回后你可以立刻修改/释放 buf**。所以非阻塞模式下，内核必须在 syscall 里就把 buf 内容**复制到内核**——否则它没法答应这个契约。

要消除这次复制，要么改契约（`MSG_ZEROCOPY`，应用承诺等通知再改 buf），要么换一套 API（io_uring、ibverbs）。**每一个号称"零拷贝"的方案，本质都是在重新定义所有权契约**。

### 8.3 阻塞 vs 非阻塞 vs epoll：操作系统篇的回响

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

## 九、延迟 vs 带宽：高带宽时代的窗口

### 9.1 RTT 决定了一切等待型操作

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

### 9.2 带宽延迟积再算一次

跨城 30 ms RTT × 10 Gbps 带宽 = **37.5 MB**。你的 TCP 发送窗口必须有 37.5 MB 才能跑满。Linux 内核默认 `tcp_rmem`/`tcp_wmem` 上限大约 6 MB——意味着默认配置下跨城你顶天跑 1.6 Gbps，浪费了 84% 的带宽。

调参方法：

```bash
sysctl -w net.core.rmem_max=67108864
sysctl -w net.core.wmem_max=67108864
sysctl -w net.ipv4.tcp_rmem='4096 87380 67108864'
sysctl -w net.ipv4.tcp_wmem='4096 87380 67108864'
```

### 9.3 钩子：3200 Gbps 网络的窗口

如果是同机房 0.1 ms RTT × 3200 Gbps：BDP = 40 MB。这数量级的窗口 + 32 个并发流，正是陈乐群文里 AWS p5 那 32 张 EFA 网卡能聚合的物理意义。

---

## 十、数据中心网络：跟你想的不太一样

### 10.1 Spine-Leaf 架构

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

### 10.2 RoCE vs InfiniBand

一句话结论：**InfiniBand 原生无损（credit-based 链路级流控）；RoCEv2 复用以太网，需要 PFC + DCQCN 才能凑出无损网**。为什么非要无损？RDMA 没有 TCP 那种 SACK 局部重传机制，丢一个包性能直接断崖式下跌。两条路线的完整对比表见通信篇的[第二层：节点间网络](/posts/2026-06-16-cs-foundations-4-communicate/#第二层节点间网络)。

### 10.3 3200 Gbps 是怎么来的

AWS p5 单机 32 张 100 Gbps EFA 网卡——单机出口聚合带宽 **3200 Gbps = 400 GB/s**。把 32 张网卡分成 8 组，每组 4 张配 1 张 H100，让 H100 通过 GPUDirect RDMA 同时用这 4 张网卡（多路并行）跟远端通信。这不是"一根 3200 Gbps 管子"，而是"32 根 100 Gbps 管子并联"。

### 10.4 容器网络：Pod 的包怎么出机器

前面讲的都是"物理机到物理机"，但现在的服务几乎都跑在容器里。容器有自己的 network namespace（见操作系统篇），它的包要多走几段路才能出机器：

```text
┌───────────────────────────────────────────┐
│  宿主机                                       │
│  ┌─────────────┐                            │
│  │ Pod (netns)  │                            │
│  │   eth0 ──────┼── veth pair ──▶ cbr0/cni0 │
│  └─────────────┘   (虚拟网线)     (宿主 bridge)│
│                                    │          │
│                        路由 + SNAT │          │
│                                    ▼          │
│                                物理网卡 eth0  │
└─────────────────────────────────│─────────┘
                                     ▼
                             Leaf 交换机 (ToR) ─▶ Spine
```

veth pair 是一根"虚拟网线"：一头插在容器 netns 里叫 `eth0`，另一头插在宿主机上接进 bridge。宿主机上能直接看到：

```bash
ip link show type veth
# 7: veth3a2b1c@if4: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 ... master cni0
# 9: veth8d4e2f@if4: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 ... master cni0
# → 每个 Pod 对应一条 veth；@if4 表示对端是容器里编号 4 的接口；
#   master cni0 表示宿主端已接入名为 cni0 的 bridge

bridge fdb show | head -3
# 33:33:00:00:00:01 dev cni0 self permanent
# aa:14:52:e0:3b:07 dev veth3a2b1c master cni0
# be:66:01:9c:22:d8 dev veth8d4e2f master cni0
# → bridge 的 MAC 学习表：哪个容器 MAC 在哪条 veth 后面，
#   和物理交换机的 FDB 是同一套逻辑
```

出了宿主机之后，跨节点 Pod 互通有两条主流路线：

| 方案 | 原理 | 代价 |
|---|---|---|
| **VXLAN overlay**（Flannel 默认） | Pod 包外面再套一层 UDP 头（端口 4789），宿主机之间隧道传输 | 每包多 50 B 封装 + 封解包 CPU，MTU 要相应调小 |
| **BGP 路由**（Calico） | 把 Pod 网段通过 BGP 宣告给物理网络，Pod IP 直接可路由，不封装 | 要求物理网络可控（自建机房友好，公有云受限） |

这一切由谁搭建？**CNI（Container Network Interface）**：kubelet 在创建 Pod 时调一个插件二进制，后者负责创建 veth、分 IP、配路由。Flannel/Calico/Cilium 都只是 CNI 接口的不同实现。

**AI 钩子：GPU 集群的 RDMA 流量为什么不走 CNI overlay**——veth + bridge + VXLAN 每一段都是内核协议栈开销，而 RDMA 的命根子就是绕开内核。所以训练 Pod 的做法是：业务面（控制、日志）走 CNI，训练面通过 **host network / SR-IOV（把物理网卡切成多个 VF 直通给 Pod）/ Multus（给 Pod 插第二张网卡）**直接拿到物理 RDMA 网卡——训练面与业务面物理分离，互不争资源。

---

## 十一、100G+ 网卡时代：RSS、多队列与中断绑核

先算一笔账：100 Gbps 线速、每包 1500 B，每秒要处理 100 Gbps ÷ (1500 × 8 bit) ≈ **830 万个包**。单核 CPU 处理一个包（中断 + 协议栈 + 拷贝）大约要几千 ns，满打满算每秒百万包量级——**单核根本扛不住 100G**，必须把包分给多个核并行处理。

### 11.1 RSS：网卡硬件把包分流到多个队列

**RSS（Receive Side Scaling）**：网卡硬件对每个包的五元组（源/目的 IP + 源/目的端口 + 协议）做 hash，按 hash 值把包分到 N 条硬件接收队列，每条队列绑一个 CPU 的中断：

```text
            到达的包
               │
               ▼
     ┌───────────────┐
     │ 网卡: hash(五元组) │   同一条流永远落同一队列
     └─┬────┬────┬───┘   （保证 TCP 不乱序）
       ▼      ▼      ▼
   RX queue0 queue1 queue2 ...
       │      │      │
     IRQ→CPU0 IRQ→CPU1 IRQ→CPU2
```

看自己机器有几条队列（无需 root）：

```bash
ethtool -l eth0
# Channel parameters for eth0:
# Pre-set maximums:
# Combined:       63          ← 硬件最多支持 63 条收发合一队列
# Current hardware settings:
# Combined:       63          ← 当前开满（高端网卡常见 ≈ CPU 核数）

ethtool -x eth0 | head -6
# RX flow hash indirection table for eth0 with 63 RX ring(s):
#     0:      0     1     2     3     4     5     6     7
#     8:      8     9    10    11    12    13    14    15
# → 间接表：hash 值模表长后查表得到队列号，默认均匀轮询分布，
#   可用 ethtool -X 改权重（比如避开某些核）
```

### 11.2 中断在哪个核上：/proc/interrupts

```bash
cat /proc/interrupts | grep mlx5 | head -4
#  98:  8812340   0        0        0   ... mlx5_comp0@pci:0000:3b:00.0
#  99:  0         7723101  0        0   ... mlx5_comp1@pci:0000:3b:00.0
# 100:  0         0        6519773  0   ... mlx5_comp2@pci:0000:3b:00.0
# 101:  0         0        0        7031225 ... mlx5_comp3@pci:0000:3b:00.0
# → 每行一个队列的 IRQ；每列是一个 CPU 的中断计数。
#   健康状态：对角线分布（每队列落在不同核）；
#   如果所有计数堆在第一列 = 中断全打在 CPU0，单核瓶颈。
```

手动绑核（**需 root，且要先 `systemctl stop irqbalance`**，否则它会把你的设置改回去）：

```bash
echo 4 > /proc/irq/98/smp_affinity_list
# 把 IRQ 98（即 mlx5_comp0 队列）的中断固定到 CPU4
cat /proc/irq/98/smp_affinity_list
# 4                          ← 确认生效
```

硬件队列不够分或者虚拟网卡没 RSS 时，还有软件兼容方案：**RPS**（软件模拟 RSS，内核用软中断把包摄到其他核）和 **RFS**（进一步把包引向"应用正在运行的那个核"，提升缓存命中）。

### 11.3 AI 钩子：IRQ 亲和必须跟 GPU 同 NUMA

多路径服务器上，RDMA 网卡和 GPU 各自挂在某个 NUMA 节点的 PCIe 根下。如果网卡中断绑在了另一个 NUMA 节点的核上，每次中断处理都要跨 NUMA 访内存，延迟和带宽双重惩罚——**IRQ 亲和、网卡、GPU 三者必须同 NUMA**（`cat /sys/class/net/eth0/device/numa_node` 能查网卡归属）。NUMA 拓扑本身为什么长这样，见[体系结构篇](/posts/2026-06-13-cs-foundations-1-architecture/)。

---

## 十二、eBPF/XDP：在网卡驱动层就把包处理掉

上一章把中断分到了多核，但每个包仍要走完整个内核协议栈。有些场景（DDoS 清洗、负载均衡）根本不需要协议栈——包进来看一眼就丢掉或转发。**XDP（eXpress Data Path）** 就是干这个的：把一段 eBPF 程序挂在网卡驱动收包路径的最早位置。

### 12.1 XDP 在哪里：sk_buff 分配之前

```text
 网卡 DMA 收包
     │
     ▼
 驱动收包函数 ──▶ 【XDP 钩子】── XDP_DROP ──▶ 直接丢，成本几十 ns
     │              │     └── XDP_TX ───▶ 原网卡弹回（改包后反射）
     │              │     └── XDP_REDIRECT ▶ 转发到其他网卡/CPU/AF_XDP
     │           XDP_PASS
     │              │
     ▼              ▼
 分配 sk_buff（昂贵的内核包结构）
     │
     ▼
 内核协议栈（IP → TCP/UDP）
     │
     ▼
  socket → 应用
```

关键在位置：**XDP 跑在 sk_buff 分配之前**。sk_buff 是内核包描述符，分配 + 初始化就要几百 ns——在这之前丢包，等于把最贵的步骤直接短路了。

### 12.2 最小 XDP 程序：15 行丢掉所有包

```c
// drop.c —— 最简 XDP 程序：收到什么丢什么（演示用！）
#include <linux/bpf.h>

#define SEC(name) __attribute__((section(name), used))

SEC("xdp")
int xdp_drop_all(struct xdp_md *ctx)
{
    return XDP_DROP;   // 每个包都返回"丢弃"
}

char _license[] SEC("license") = "GPL";
```

编译与挂载（编译需 clang；挂载需 root，**别在自己 ssh 的网卡上试，会断连**）：

```bash
clang -O2 -target bpf -c drop.c -o drop.o
# 编译成 BPF 字节码，加载时内核 verifier 会验证它不会崩内核

ip link set dev eth0 xdp obj drop.o sec xdp     # 挂载（需 root）
ip link set dev eth0 xdp off                    # 卸载
```

性能量级（示意）：XDP 丢包可以做到**单核 10~20M pps**，而走完整协议栈再丢只有百万量级——这就是 DDoS 清洗为什么必须在 XDP 层做。

### 12.3 谁在用它

- **Cilium**：基于 eBPF 的 Kubernetes CNI，用 eBPF 替代 iptables 做转发和网络策略——GPU 集群里最常见的 CNI 之一（呼应 10.4 节）
- **Katran**：Meta 的 L4 负载均衡器，XDP 层改包转发，单机扛百万级 QPS

eBPF 本身远不止网络——它是一套通用的内核可编程机制，跟踪、安全、观测都靠它。它的工作原理、verifier、bpftrace 用法，见操作系统篇 [7.4 eBPF 与 bpftrace](/posts/2026-06-14-cs-foundations-2-os/#74-ebpf-与-bpftrace不改内核不重启的系统透视)。

---

## 十三、DPDK：完全用户态网络栈

XDP 还是"在内核里加钩子"，DPDK（Data Plane Development Kit）则彻底掎旗：**内核靠边站，网卡直接交给用户态进程**。架构三要素：

- **PMD（Poll Mode Driver）**：用户态驱动独占网卡，一个 CPU 核 100% 死循环轮询收包——零中断、零系统调用，代价是那个核永远显示 100% 占用
- **大页内存**：收发缓冲区用 2 MB/1 GB HugePage，减少 TLB miss（见操作系统篇大页小节）
- **用户态协议栈**：内核协议栈不参与，要 TCP 得自己带（F-Stack、mTCP）或者干脆只处理自定义协议

使用前要把网卡从内核驱动解绑，换绑到 vfio-pci（需 root；解绑后内核里这张网卡直接消失，ssh 会断，别动管理口）：

```bash
dpdk-devbind.py --status
# Network devices using kernel driver
# ===================================
# 0000:3b:00.0 'MT2892 Family [ConnectX-6]' if=eth0 drv=mlx5_core unused=vfio-pci
# 0000:3b:00.1 'MT2892 Family [ConnectX-6]' if=eth1 drv=mlx5_core unused=vfio-pci

dpdk-devbind.py --bind=vfio-pci 0000:3b:00.1    # 把 eth1 交给 DPDK
dpdk-devbind.py --status
# Network devices using DPDK-compatible driver
# ============================================
# 0000:3b:00.1 'MT2892 Family [ConnectX-6]' drv=vfio-pci unused=mlx5_core
# → eth1 从 ip link 里消失了，内核不再认识它，只有 DPDK 进程能用
```

最小收发验证用自带的 testpmd（需 root + 大页已配置）：

```bash
dpdk-testpmd -l 0-1 -n 4 -- -i --forward-mode=macswap
# -l 0-1: 用 CPU0/CPU1 两个核；--forward-mode=macswap: 收包、交换源目的 MAC 后发回
# testpmd> start
# testpmd> show port stats 0     ← 能看到 RX-pps/TX-pps 实时包速
```

### 13.1 三条 kernel-bypass 路线选型

| 路线 | 侵入性 | 性能 | 是否独占网卡 | 典型场景 |
|---|---|---|---|---|
| **XDP** | 低（内核内，协议栈共存） | 单核 10~20M pps（丢包/转发） | 否 | DDoS 防护、L4 LB、CNI 加速 |
| **DPDK** | 高（应用重写，独占核） | 线速，延迟最可控 | 是 | 网关、NFV、量化交易 |
| **RDMA** | 中（换 verbs API，需专用网卡） | 延迟 µs 级，CPU 几乎不参与 | 否（多进程共享） | AI 训练/推理、存储网络 |

三者都在回答同一个问题——"内核协议栈太慢怎么办"，区别只是绕开多少、谁来干活。RDMA 这条线的完整实战见[通信篇](/posts/2026-06-16-cs-foundations-4-communicate/)。

---

## 十四、常用调试工具

### 14.1 一句话用法表

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

### 14.2 三个最常用工具的输出解读

排错时 90% 的时间在看这三个命令的输出，值得逐个过一遍。

**① `ss -tin`：连接级体检**（完整字段解读见 6.7 节，这里只看三个数）：

```bash
ss -tin state established | head -3
# Recv-Q Send-Q  Local Address:Port  Peer Address:Port
# 0      1248640 10.0.0.5:44612      10.0.0.8:5201
#      cubic rtt:1.85/0.32 cwnd:512 retrans:0/17 delivery_rate 3.2Gbps
# → 先看 Send-Q：持续堆积 = 发不出去（网络慢或对端收窗小）；
#   再看 retrans 和 delivery_rate，定位思路同 6.7
```

**② `ethtool -S`：网卡硬件计数器**，专治"丢包了但协议栈没看见"：

```bash
ethtool -S eth0 | grep -E "discard|error|pause"
# rx_discards_phy: 48213        ← 非零！网卡物理层缓冲溢出丢的包，
#                                 典型原因：CPU 收不过来（回看十一章绑核）
# rx_errors: 0                  ← CRC 错包，非零通常是线缆/光模块问题
# rx_pause_ctrl_phy: 1893       ← 收到的 pause/PFC 帧：对端交换机在叫停，
#                                 RoCE 无损网里看到它 = PFC 正在生效
# tx_pause_ctrl_phy: 0          ← 本机发出的 pause 帧，非零 = 本机收不过来
```

**③ `tcpdump`：看包本身**（抓包需 root）：

```bash
tcpdump -i eth0 -nn -c 3 'tcp port 443'
# 14:23:01.824512 IP 10.0.0.5.44612 > 10.0.0.8.443: Flags [S], seq 1839203145, win 64240, length 0
# 14:23:01.825103 IP 10.0.0.8.443 > 10.0.0.5.44612: Flags [S.], seq 902817364, ack 1839203146, win 65160, length 0
# 14:23:01.825141 IP 10.0.0.5.44612 > 10.0.0.8.443: Flags [.], ack 1, win 502, length 0
```

逐字段：`14:23:01.824512` 是微秒精度时间戳（相邻两行相减≈单向延迟）；`10.0.0.5.44612 > 10.0.0.8.443` 是五元组里的四元（协议已由过滤器限定 TCP）；`Flags [S]/[S.]/[.]` 分别是 SYN / SYN+ACK / ACK——这三行正好是一次完整三次握手；`seq/ack` 是序号（第三行起 tcpdump 默认显示相对值，所以是 `ack 1`）；`win` 是接收窗口——注意第三行变成 502，因为握手完成后要乘上 wscale（502 × 2⁷ ≈ 64 KB）。

### 14.3 排错思路

性能慢/不通问题，按这个顺序排：

1. **物理层**：网卡 link 起来没？`ethtool eth0` 看 Speed/Duplex
2. **链路层**：ARP 表对吗？`ip neighbor`
3. **网络层**：路由对吗？`ip route get <ip>`
4. **传输层**：端口通吗？`telnet host port` / `nc -zv`
5. **应用层**：日志说啥？

很多看似"应用 bug"最后查到是 IRQ 不均、MTU 不一致、TCP 窗口太小、DNS 解析慢——网络问题永远要从下往上查。

---

## 十五、附录：协议头速查 + TCP 状态机 + 术语表

### 15.1 头部字段长度速查

| 协议 | 头部长度 | 关键字段 |
|---|---|---|
| Ethernet | 14 B | 源/目的 MAC、EtherType |
| IPv4 | 20 B | 源/目的 IP、Protocol、TTL |
| IPv6 | 40 B | — |
| TCP | 20 B | 源/目的端口、seq、ack、window、flags |
| UDP | 8 B | 源/目的端口、长度、校验 |

以太网帧典型: **14 + 20 + 20 = 54 B 头**，剩下 1446 B 装 payload。

### 15.2 TCP 状态机（精简版）

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

### 15.3 术语速查表

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
| **DNS** | 域名 → IP 的分布式查询系统 |
| **TTL** | DNS 记录缓存有效期 |
| **CNAME** | 域名别名记录 |
| **HTTP/2** | 二进制帧 + 多路复用的 HTTP |
| **TLS 1.3** | 1-RTT 加密握手，前向安全 |
| **HPACK / QPACK** | HTTP/2 / HTTP/3 头部压缩 |
| **0-RTT** | 重连时零往返建立加密通道 |
| **eBPF** | 内核内安全运行的可编程字节码 |
| **XDP** | 挂在网卡驱动最早收包点的 eBPF 钩子 |
| **DPDK** | 用户态轮询驱动，完全绕开内核协议栈 |
| **RSS** | 网卡硬件按五元组 hash 把包分到多队列多核 |
| **RPS/RFS** | RSS 的软件版 / 再把包导向应用所在核 |
| **veth** | 成对出现的虚拟网线，容器接入宿主机的桥梁 |
| **CNI** | 容器网络插件接口，负责给 Pod 建网分 IP |
| **VXLAN** | 用 UDP 封装二层帧的 overlay 隧道协议 |
| **SR-IOV** | 把一张物理网卡切成多个 VF 直通给容器/虚机 |

---

## 十六、下一步

- [一：体系结构](/posts/2026-06-13-cs-foundations-1-architecture/)——读它你会明白十一章"IRQ 要跟 GPU 同 NUMA"背后的硬件拓扑：PCIe 根、NUMA 节点、缓存层次到底长什么样。
- [二：操作系统](/posts/2026-06-14-cs-foundations-2-os/)——读它你会明白本篇反复出现的 namespace、大页、中断、eBPF 在内核里到底是怎么回事，socket 之下那层地基就稳了。
- [四：通信](/posts/2026-06-16-cs-foundations-4-communicate/)——读它你会拿到本篇留的所有钩子的答案：RDMA 怎么真正上手、NCCL 怎么把 3200 Gbps 用满、集合通信原语长什么样。
