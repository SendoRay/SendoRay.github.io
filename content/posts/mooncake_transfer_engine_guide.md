# Mooncake Store & Transfer Engine 网络架构指南

## 概述

Mooncake 是一个分布式键值存储系统，专为大规模 GPU 集群中的模型权重传输和参数离线存储设计。其核心组件 **Transfer Engine** 提供了高效的跨节点数据传输能力，支持 RDMA 和 TCP 两种传输协议。

本文档重点阐述：
1. 控制层 (Control Plane) 与传输层 (Data Plane) 的分离架构
2. RDMA 与 TCP 协议的适用场景与区别
3. MetaServer 与 Storage Client 的关键配置

---

## 架构总览：控制层 vs 传输层

Mooncake 采用经典的 **控制层/传输层分离** 架构：

```
┌─────────────────────────────────────────────────────────────────┐
│                        控制层 (Control Plane)                      │
│                                                                   │
│  ┌─────────────────────┐        ┌──────────────────────────┐    │
│  │   mooncake_master    │        │  HTTP Metadata Server     │    │
│  │   (gRPC :50051)      │◄──────►│  (REST :8083)             │    │
│  │                       │        │                            │    │
│  │  - 对象注册/发现       │        │  - Segment 元数据注册       │    │
│  │  - 副本追踪            │        │  - 节点发现                 │    │
│  │  - 租约管理            │        │  - 拓扑信息广播             │    │
│  │  - 驱逐协调            │        │                            │    │
│  └─────────────────────┘        └──────────────────────────┘    │
│                                                                   │
│          使用普通 TCP/IP 即可（带宽需求低，延迟不敏感）               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        传输层 (Data Plane)                         │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Transfer Engine                               │    │
│  │                                                           │    │
│  │  - RDMA (InfiniBand/RoCE) ← 推荐，200Gbps+              │    │
│  │  - TCP (fallback)                                         │    │
│  │  - GPUDirect RDMA (GPU 显存直接传输)                       │    │
│  │  - 多网卡带宽聚合                                          │    │
│  │  - 拓扑感知路径选择                                        │    │
│  │                                                           │    │
│  │  动态端口: 12300-14300 (数据), 15000-17000 (RPC)          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│          生产环境必须使用 RDMA 加速（海量数据，延迟敏感）             │
└─────────────────────────────────────────────────────────────────┘
```

### 关键设计原则

| 层次 | 协议要求 | 原因 |
|------|----------|------|
| **控制层** | TCP/IP 即可 | 仅传输元数据（对象位置、租约、拓扑），数据量小，对延迟不敏感 |
| **传输层** | **必须使用 RDMA**（生产环境） | 传输模型权重（数十 GB），需要极低延迟和极高带宽，CPU 零开销 |

> **简单类比**：控制层就像"电话簿"——告诉你数据在哪里；传输层就像"搬运卡车"——实际搬运数十 GB 的模型权重。电话簿用普通网络就行，但搬运卡车必须走高速公路（RDMA）。

---

## RDMA vs TCP：核心区别

### 对比表

| 维度 | RDMA | TCP |
|------|------|-----|
| **吞吐量** | 单网卡可达 200 Gbps，多卡聚合更高 | 受 CPU 处理能力限制，通常 10-25 Gbps |
| **延迟** | 亚微秒级（< 1μs） | 毫秒级 |
| **CPU 开销** | 几乎为零（硬件卸载） | 高（内核协议栈处理） |
| **GPU 直通** | 支持 GPUDirect RDMA（GPU 显存直接读写） | 不支持，必须先拷贝到 CPU 内存 |
| **多网卡聚合** | 支持拓扑感知带宽聚合 | 单连接 |
| **内存类型** | DRAM/VRAM → 远端 DRAM/VRAM | 仅 DRAM → 远端 DRAM |
| **内存注册** | 需要 `register_buffer()` 进行 RDMA 内存注册 | 不需要 |
| **硬件要求** | InfiniBand / RoCE / eRDMA 网卡 | 任意以太网 |
| **适用场景** | 生产训练集群 | 开发测试、无 RDMA 硬件环境 |
| **自动回退** | 检测不到 RDMA 网卡时自动降级到 TCP | 始终可用 |

### 何时使用 TCP

- 本地开发调试环境
- 没有 InfiniBand/RoCE 硬件的集群
- 控制层通信（MetaServer 的 HTTP/gRPC 接口）
- 小规模数据传输（KB 级别的元数据）

### 何时必须使用 RDMA

- 生产环境的模型权重传输（数十 GB）
- 需要 GPUDirect 零拷贝的场景
- 对延迟敏感的同步更新
- 多节点大规模训练集群

### 协议自动检测

Mooncake 会自动检测 RDMA 可用性：

```python
# 检测逻辑（roll/mooncake/config.py）
# 扫描 /sys/class/infiniband/*/ports/*/state
# 如果发现 "ACTIVE" 状态的 IB 端口 → protocol = "rdma"
# 否则 → protocol = "tcp"
```

也可通过环境变量强制指定：
```bash
export MOONCAKE_PROTOCOL=rdma   # 强制 RDMA
export MOONCAKE_PROTOCOL=tcp    # 强制 TCP
```

---

## MetaServer 配置详解

MetaServer 即 `mooncake_master` 进程，是整个 Mooncake 集群的控制中枢。

### 角色与职责

`mooncake_master` 同时承担两个服务角色：

| 服务 | 端口 | 协议 | 职责 |
|------|------|------|------|
| HTTP Metadata Server | 8083 | REST over TCP | Segment 元数据注册/发现、节点拓扑广播 |
| Master gRPC Service | 50051 | gRPC over TCP | 对象注册表、副本追踪、租约管理、驱逐协调 |

### 启动命令

```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=${NODE_IP} \
  --http_metadata_server_port=8083 \
  --default_kv_lease_ttl=300000       # 对象租约 TTL（毫秒），默认 5 分钟
```

### 启用 SSD Offload

```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=${NODE_IP} \
  --http_metadata_server_port=8083 \
  --default_kv_lease_ttl=300000 \
  --enable_offload=true \
  --root_fs_dir=/mnt/nvme/mooncake \
  --global_file_segment_size=107374182400   # 100GB SSD 容量
```

### HTTP Metadata Server API

Transfer Engine 通过以下 RESTful 接口注册和发现 Segment：

```
GET    /metadata?key=$KEY    # 查询 segment 元数据
PUT    /metadata?key=$KEY    # 注册 segment 元数据
DELETE /metadata?key=$KEY    # 注销 segment
```

Segment 元数据格式（JSON）：
```json
{
  "device_info": [...],
  "priority_matrix": [...],
  "buffer_registration": {...}
}
```

### 关键配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--http_metadata_server_host` | 监听地址 | 必填 |
| `--http_metadata_server_port` | HTTP 端口 | 8083 |
| `--default_kv_lease_ttl` | 对象租约 TTL（ms） | 300000 (5min) |
| `--enable_offload` | 是否启用 SSD 驱逐 | false |
| `--root_fs_dir` | SSD 存储路径 | 无 |
| `--global_file_segment_size` | SSD 可用空间 | 0 |

> **注意**：MetaServer 的所有通信都走普通 TCP/IP。它是控制层组件，不参与数据传输。

---

## Storage Client 配置详解

Storage Client 即 `mooncake_store_service` 进程，是数据层的核心——负责实际的张量存储和传输。

### 配置文件格式

```json
{
    "local_hostname": "192.168.1.100",
    "metadata_server": "http://192.168.1.1:8083/metadata",
    "global_segment_size": 21474836480,
    "local_buffer_size": 5368709120,
    "protocol": "rdma",
    "device_name": "mlx5_0",
    "master_server_address": "192.168.1.1:50051"
}
```

### 字段说明

| 字段 | 说明 | 建议值 |
|------|------|--------|
| `local_hostname` | 本节点的网络地址（Transfer Engine 用此地址建立连接） | 节点 IP（RDMA 时用 RDMA 网卡的 IP/IPv6） |
| `metadata_server` | HTTP Metadata Server 的 URL | `http://<master_ip>:8083/metadata` |
| `global_segment_size` | DRAM 池大小（字节），用于缓存模型权重 | 10-20 GB |
| `local_buffer_size` | RDMA 暂存缓冲区大小，必须 ≥ 单个最大张量 | 3-5 GB |
| `protocol` | 传输协议：`"rdma"` 或 `"tcp"` | `"rdma"`（生产） |
| `device_name` | RDMA 设备名（如 `mlx5_0`），空字符串表示自动检测 | `""` 或具体设备名 |
| `master_server_address` | Master gRPC 地址 | `<master_ip>:50051` |

### 启用 SSD Offload 的额外字段

```json
{
    "file_storage_path": "/mnt/nvme/mooncake/data",
    "file_storage_size": 107374182400,
    "global_file_segment_size": 107374182400
}
```

### 启动命令

```bash
python -m mooncake.mooncake_store_service \
  --config="/path/to/config.json" \
  --port=8085
```

### 环境变量配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MOONCAKE_STORAGE_HOST` | Storage Service 监听地址 | `hostname -I` |
| `MOONCAKE_STORAGE_PORT` | HTTP Metadata 端口 | 8083 |
| `MOONCAKE_MASTER_HOST` | Master gRPC 地址 | 同 storage_host |
| `MOONCAKE_MASTER_PORT` | Master gRPC 端口 | 50051 |
| `MOONCAKE_PROTOCOL` | `tcp` / `rdma` | 自动检测 |
| `MOONCAKE_INTERFACE` | RDMA 网卡设备名 | 第一个活跃 IB 设备 |
| `MOONCAKE_NODE_ADDR` | 节点地址（用于 Transfer Engine） | auto |

---

## Transfer Engine 网络配置

Transfer Engine 是 Mooncake 的数据传输引擎，直接负责字节搬运。

### 核心概念

- **Segment**：一段连续的地址空间（DRAM 或 VRAM），可被远程读写
- **BatchTransfer**：一组传输请求，在不连续的地址空间之间同步数据
- **Endpoint**：RDMA QP（队列对）连接，按需建立，使用 SIEVE 算法池化管理

### 拓扑感知路径选择

Transfer Engine 的一个关键特性是 **拓扑感知的多路径选择**：

1. 每个节点启动时生成本地拓扑矩阵（NIC-to-Memory affinity）
2. 拓扑信息通过 MetaServer 广播到集群
3. 传输请求 > 64KB 时，自动拆分成多个 slice，每个 slice 可能走不同 NIC
4. NIC 按 NUMA 亲和性分为 preferred 和 secondary 列表

### 关键环境变量

#### RDMA 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MC_GID_INDEX` | RDMA GID 索引（RoCE 网络需要正确设置） | 3 |
| `MC_USE_IPV6` | 使用 IPv6 地址做 RDMA 数据面 | 0 |
| `MC_IB_PCI_RELAXED_ORDERING` | 启用 PCIe Relaxed Ordering（提升 RDMA 性能） | 0 |
| `MC_NUM_QP_PER_EP` | 每个 Endpoint 的 QP 数量 | 2 |
| `MC_MAX_EP_PER_CTX` | 每个设备最大 Endpoint 数 | 65536 |
| `MC_FORCE_HCA` | 强制使用指定 HCA 设备 | 无 |
| `MC_MLX5_QP_UDP_SPORTS` | UDP 源端口（用于 ECMP/LAG 负载均衡） | 无 |

#### TCP 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MC_FORCE_TCP` | 强制使用 TCP 传输 | 0 |
| `MC_TCP_SLICE_SIZE` | TCP 传输分片大小 | 65536 (64KB) |

#### 通用配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MC_SLICE_SIZE` | RDMA 传输分片大小（用于多路径并行） | 自动 |
| `MC_RETRY_CNT` | 最大重试次数 | 3 |
| `MC_MIN_RPC_PORT` | Transfer Engine RPC 端口范围下界 | 15000 |
| `MC_MAX_RPC_PORT` | Transfer Engine RPC 端口范围上界 | 17000 |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | 启用 Rail-optimized 拓扑匹配 | 0 |
| `MC_PATH_ROUNDROBIN` | Round-robin 模式（适合大批量传输） | 0 |

---

## 端口规划总览

| 组件 | 端口 | 协议 | 层次 |
|------|------|------|------|
| Master HTTP Metadata | 8083 | HTTP (TCP) | 控制层 |
| Master gRPC | 50051 | gRPC (TCP) | 控制层 |
| Store Service API | 8085 | HTTP (TCP) | 控制层 |
| Transfer Engine RPC | 15000-17000 | TCP | 传输层辅助 |
| Transfer Engine Data | 12300-14300 | **RDMA** 或 TCP | 传输层 |

---

## 多节点部署示例

### 场景：3 节点 RDMA 集群

```
Node 1 (Head):  mooncake_master + mooncake_store_service
Node 2 (Worker): mooncake_store_service
Node 3 (Worker): mooncake_store_service
```

#### Node 1 (Head 节点)

```bash
# 环境变量
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_INTERFACE=mlx5_0
export MC_GID_INDEX=3
export MC_IB_PCI_RELAXED_ORDERING=1

# 启动 Master
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=10.0.0.1 \
  --http_metadata_server_port=8083 \
  --default_kv_lease_ttl=300000

# 启动 Store Service
cat > /tmp/store_config.json << EOF
{
    "local_hostname": "10.0.0.1",
    "metadata_server": "http://10.0.0.1:8083/metadata",
    "global_segment_size": 21474836480,
    "local_buffer_size": 5368709120,
    "protocol": "rdma",
    "device_name": "mlx5_0",
    "master_server_address": "10.0.0.1:50051"
}
EOF

python -m mooncake.mooncake_store_service --config=/tmp/store_config.json --port=8085
```

#### Node 2/3 (Worker 节点)

```bash
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_INTERFACE=mlx5_0
export MC_GID_INDEX=3
export MC_IB_PCI_RELAXED_ORDERING=1

cat > /tmp/store_config.json << EOF
{
    "local_hostname": "10.0.0.2",
    "metadata_server": "http://10.0.0.1:8083/metadata",
    "global_segment_size": 21474836480,
    "local_buffer_size": 5368709120,
    "protocol": "rdma",
    "device_name": "mlx5_0",
    "master_server_address": "10.0.0.1:50051"
}
EOF

python -m mooncake.mooncake_store_service --config=/tmp/store_config.json --port=8085
```

### 多节点 IPv6 RDMA 配置

当 RDMA 网卡使用 IPv6 时（常见于 RoCE 环境）：

```bash
export MC_USE_IPV6=1
export MC_GID_INDEX=3

# local_hostname 使用 RDMA 网卡的 IPv6 地址
RDMA_NETDEV="eth6"
IPV6_ADDR=$(ip -6 addr show dev $RDMA_NETDEV | grep "scope global" | awk '{print $2}' | cut -d'/' -f1)

# 在配置中使用 IPv6 地址
"local_hostname": "${IPV6_ADDR}"
```

---

## 在 ROLL 训练框架中的集成

### Pipeline 配置示例

```yaml
# enable mooncake offload for model weights
enable_mooncake_offload: true

# mooncake 服务配置
mooncake_config:
  service:
    service_dram_size: 21474836480     # 20GB DRAM 池
    local_buffer_size: 5368709120      # 5GB 暂存缓冲
    ssd_capacity: 0                    # 0 = 不启用 SSD
  client:
    segment: 0                         # 0 = thin client 模式
    local_buffer_size: 3221225472      # 3GB

# 稀疏权重更新配置
actor_train:
  enable_sparse_model_update: true
  sparse_model_update_threshold: 1e-9
  sparse_model_update_full_sync_frequency: 10

# 系统环境变量
system_envs:
  MOONCAKE_ENABLE_VERIFY: '1'         # 启用数据校验
  MOONCAKE_DISABLE_PARALLEL_IO: '1'   # 禁用并行 IO（调试用）
```

### Client 模式说明

| 模式 | `segment` 值 | 说明 |
|------|--------------|------|
| Thin Client | 0 | 不贡献本地 DRAM，所有数据存到集中式 store_service |
| Fat Client | > 0 | 贡献指定大小的本地 DRAM 参与分布式存储 |

---

## 常见问题 FAQ

### Q: 控制层为什么不需要 RDMA？

控制层传输的是元数据（键的位置、节点拓扑、租约信息），数据量仅 KB 级别，频率低（秒级），对延迟容忍度高。使用普通 TCP/IP 完全满足需求，且无需额外硬件。

### Q: 传输层为什么必须用 RDMA？

传输层搬运的是实际模型权重（30B 模型约 60GB），要求：
- **高带宽**：模型同步需要在秒级完成
- **低延迟**：训练/推理切换的间隔越短，GPU 利用率越高
- **零 CPU 开销**：CPU 应专注于训练计算，不应浪费在数据拷贝上
- **GPU 直通**：GPUDirect RDMA 允许跳过 CPU 直接访问 GPU 显存

TCP 虽然能工作，但在生产集群中会成为严重瓶颈。

### Q: TCP 模式何时合理？

- 开发和调试阶段（本地单机或无 IB 的测试集群）
- 小模型实验（< 1B 参数，权重 < 2GB）
- 作为 RDMA 不可用时的自动 fallback

### Q: `MC_GID_INDEX` 是什么？为什么默认是 3？

GID (Global Identifier) 是 RDMA 设备的网络地址标识。不同 GID Index 对应不同的地址类型：
- Index 0: IB 默认 GID
- Index 1: IPv4 based RoCE
- Index 2: IPv6 link-local
- Index 3: IPv6 global (最常用于 RoCEv2)

可通过 `show_gids` 命令查看本机 GID 表。

### Q: 如何确认 RDMA 正常工作？

```bash
# 检查 IB 设备状态
ibstat

# 检查端口是否 Active
cat /sys/class/infiniband/mlx5_0/ports/1/state
# 应输出: 4: ACTIVE

# 测试 RDMA 连通性
ib_write_bw -d mlx5_0   # server
ib_write_bw -d mlx5_0 <server_ip>  # client
```

### Q: `local_buffer_size` 设多大合适？

必须大于等于最大单个张量的大小。对于 7B 模型，最大张量通常是 embedding 层（约 1-2GB）。建议：
- 7B 模型：3GB
- 30B 模型：5GB
- 70B+ 模型：8-10GB

---

## 总结

```
┌──────────────────────────────────────────────┐
│               Mooncake 架构要点               │
├──────────────────────────────────────────────┤
│                                              │
│  控制层 = TCP/IP (低带宽，低频率)              │
│    ├── HTTP Metadata Server (:8083)          │
│    ├── Master gRPC (:50051)                  │
│    └── Store Service API (:8085)             │
│                                              │
│  传输层 = RDMA (高带宽，低延迟，零拷贝)        │
│    ├── Transfer Engine Data (12300-14300)     │
│    ├── 支持 GPUDirect RDMA                   │
│    ├── 多网卡拓扑感知聚合                     │
│    └── 自动 fallback 到 TCP                  │
│                                              │
│  核心原则:                                    │
│    控制信令走 TCP ✓                           │
│    模型权重走 RDMA ✓                          │
│    GPU 显存直通 RDMA ✓                        │
│                                              │
└──────────────────────────────────────────────┘
```
