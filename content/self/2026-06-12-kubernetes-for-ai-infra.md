---
title: "Kubernetes：AI Infra 视角的最小够用版"
date: '2026-06-12'
tags:
- Engineering

draft: false
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

> 这篇文章不写 K8s 教程，只回答一个问题：
>
> **作为做训练 / 推理基础设施的人，对 K8s 知道到什么程度就够了？**
>
> 目标：能看懂 yaml、能 debug 卡住的训练任务、能跟 SRE 对齐资源需求，仅此而已。

---

## 一、为什么 AI Infra 绕不开 K8s

GPU 集群的现实是：几十到几千张卡，几十个研究员同时抢资源，训练任务可能跑几小时也可能跑几周，推理服务要 7×24 在线。**这本质就是一个"多租户 + 异构硬件 + 长任务 + 在线服务"的调度问题**，而 K8s 是目前业界事实标准的答案。

具体到 AI 场景，K8s 帮你解决：

- **资源抽象**：把"哪台机器、哪张卡、哪条 IB 网卡"抽象成 `resources.limits`
- **任务编排**：分布式训练几十个进程要同时拉起来，要么全起要么全不起（gang scheduling）
- **故障恢复**：节点挂了 Pod 能自动重建，配合 checkpoint 实现训练续跑
- **服务上线**：推理服务要灰度、要 HPA、要负载均衡，K8s 原生支持
- **环境一致**：镜像 = 环境，再也不用调研究员的 conda

---

## 二、一张图看懂 K8s 架构

```
                      ┌──────────────────────────────────────────────┐
                      │              Control Plane (master)          │
                      │                                              │
   kubectl apply ───► │   ┌──────────┐    ┌──────────┐   ┌────────┐  │
   (yaml)             │   │ API      │◄──►│ etcd     │   │ sched  │  │
                      │   │ Server   │    │ (KV 存储) │   │ uler   │  │
                      │   └─────┬────┘    └──────────┘   └────┬───┘  │
                      │         │                             │       │
                      │   ┌─────▼────────────────────────────▼────┐  │
                      │   │         Controller Manager             │  │
                      │   │ (Deployment / Job / ReplicaSet ...)    │  │
                      │   └────────────────────────────────────────┘  │
                      └──────────────────┬───────────────────────────┘
                                         │
                ┌────────────────────────┼────────────────────────┐
                │                        │                        │
        ┌───────▼────────┐      ┌────────▼───────┐      ┌─────────▼──────┐
        │   Node 1       │      │   Node 2       │      │   Node N       │
        │  (8×H100)      │      │  (8×H100)      │      │  (8×H100)      │
        │ ┌────────────┐ │      │ ┌────────────┐ │      │ ┌────────────┐ │
        │ │  kubelet   │ │      │ │  kubelet   │ │      │ │  kubelet   │ │
        │ ├────────────┤ │      │ ├────────────┤ │      │ ├────────────┤ │
        │ │ container  │ │      │ │ container  │ │      │ │ container  │ │
        │ │ runtime    │ │      │ │ runtime    │ │      │ │ runtime    │ │
        │ ├────────────┤ │      │ ├────────────┤ │      │ ├────────────┤ │
        │ │ Pod  Pod   │ │      │ │ Pod  Pod   │ │      │ │ Pod  Pod   │ │
        │ └────────────┘ │      │ └────────────┘ │      │ └────────────┘ │
        └────────────────┘      └────────────────┘      └────────────────┘
```

记住三件事：

1. **API Server 是唯一入口**：你 `kubectl` 也好、Operator 也好，所有操作都过它
2. **etcd 是唯一真相源**：整个集群状态存在这里，挂了集群就废了
3. **kubelet 在每个节点上**：负责真正把容器拉起来、汇报状态

---

## 三、Docker 镜像与容器：K8s 管什么东西

上一章图里每个 Node 上都有个 **container runtime**，K8s 调度来调度去，最后落地跑的就是容器，而容器的"安装包"就是镜像。这一章把这块地基补上——只讲原理和 AI 场景要紧的部分。

> Docker 常用命令速查可移步[《程序员速查手册》的 Docker 章节](/posts/2026-05-25-programmer-cheatsheet/#6-docker-常用命令)，这里不重复列命令。

### 3.1 容器 ≠ 虚拟机：本质是 namespace + cgroup

先破除一个误解：**容器不是轻量级虚拟机，它就是宿主机上一个被隔离的普通进程**。隔离靠内核的两个机制：

- **namespace 管"看得见什么"**：pid（进程树）、net（网卡端口）、mnt（挂载点）、uts（主机名）、ipc（共享内存）、user（用户）——每个维度给进程套一层独立视图，让它"以为"自己独占一台机器
- **cgroup 管"能用多少"**：CPU、内存、设备访问的配额上限——你写的 `resources.limits` 最终就是落到 cgroup 上

```
        虚拟机                              容器
 ┌─────────┬─────────┐           ┌─────────┬─────────┐
 │  App A  │  App B  │           │  App A  │  App B  │
 ├─────────┼─────────┤           ├─────────┴─────────┤
 │ Guest OS│ Guest OS│           │ namespace + cgroup │
 │ (独立内核)│(独立内核)│           │  (只是隔离的进程)   │
 ├─────────┴─────────┤           ├────────────────────┤
 │     Hypervisor    │           │  宿主机内核（共享）  │
 ├───────────────────┤           ├────────────────────┤
 │      宿主机内核    │           │       硬件          │
 ├───────────────────┤           └────────────────────┘
 │       硬件         │
 └───────────────────┘
```

| | 虚拟机 | 容器 |
|---|------|------|
| **隔离单位** | 完整操作系统 | 进程 |
| **内核** | 每个 VM 独立内核 | 全部共享宿主机内核 |
| **启动速度** | 分钟级 | 秒级 |
| **性能损耗** | 有虚拟化开销 | 接近裸机 |

一个重要推论：**容器里没有独立内核**——在容器里跑 `uname -r`，看到的就是宿主机的内核版本。所以宿主机的内核版本、GPU 驱动直接影响容器里的程序能不能跑，**这正是 GPU 容器化的关键**（3.6 和 6.1 会回到这一点）。

### 3.2 镜像是什么：分层 + UnionFS

**镜像 = 一叠只读层；容器启动 = 在镜像顶上加一层可写层**。每条 Dockerfile 指令生成一层，层与层叠起来由 **UnionFS**（现在主流是 overlayfs）合成一个统一的文件系统视图，容器里改文件走 copy-on-write，只发生在最上面的可写层。

```
         容器 A                     容器 B
   ┌───────────────┐         ┌───────────────┐
   │  可写层 (RW)   │         │  可写层 (RW)   │   ← 各容器独立
   └───────┬───────┘         └───────┬───────┘
           └────────────┬────────────┘
           ┌────────────▼────────────┐
           │  Layer 3: COPY 代码      │   ← 只读
           ├─────────────────────────┤
           │  Layer 2: pip install   │   ← 只读
           ├─────────────────────────┤
           │  Layer 1: CUDA 基础镜像  │   ← 只读，节点上只存一份
           └─────────────────────────┘
```

分层的好处是**复用**：同一节点上 10 个容器都基于同一个 CUDA 基础镜像，磁盘上底层只存一份；推新版本镜像时，没变的层也不用重新拉。

构建时同样按层缓存：**每一层的缓存 key 依赖它前面所有层**。所以改了 Dockerfile 前面的一条指令，后面所有层的缓存全部失效、全部重建——这就是 Dockerfile 指令顺序如此重要的原因，下一节展开。

### 3.3 Dockerfile：最小构建示例

一个 AI 推理镜像的最小 Dockerfile：

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# 1. 系统依赖（很少变，放最前面）
RUN apt-get update && apt-get install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. Python 依赖（偶尔变）
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# 3. 业务代码（天天变，放最后面）
COPY . /app
WORKDIR /app

CMD ["python3", "server.py"]
```

核心原则只有一条：**变得少的放前面，变得勤的放后面**。依赖装在前、代码 COPY 在后，改一行代码只重建最后几层，几秒完事；顺序反过来，每改一行代码都要重装一遍几个 GB 的 pytorch。

再提一嘴**多阶段构建**（multi-stage build）：需要编译的场景（比如编 CUDA 算子）用一个 `FROM ... AS builder` 阶段装全套工具链做编译，最终镜像只 `COPY --from=builder` 拿产物——工具链不进最终镜像，体积能小一大截。

### 3.4 镜像仓库、tag 与 digest

镜像存在 **Registry**（镜像仓库）里：公网有 Docker Hub、nvcr.io，企业内部常用 **Harbor** 或云厂商的 ACR / ECR。镜像名的完整格式长这样：

```
registry.example.com/ai-infra/vllm-serve:v0.6.0@sha256:3f2a9c...
└────────┬─────────┘ └──┬───┘ └───┬────┘ └─┬──┘ └─────┬──────┘
    Registry 地址    namespace   repo      tag      digest
```

| | tag | digest |
|---|-----|--------|
| **本质** | 可变指针，随时能指向新镜像 | 内容的 sha256，不可变 |
| **例子** | `:latest`、`:v0.6.0` | `@sha256:3f2a9c...` |
| **适合** | 日常开发 | 生产环境、复现实验 |

生产环境的经典坑：**`latest` 是可变指针**——今天拉的 `latest` 和明天拉的可能根本不是同一个镜像，训练结果对不上都不知道去哪查。生产至少用固定版本 tag（并且别复用同一个 tag 推新内容），要求严格就直接钉 digest。

### 3.5 Docker → containerd → CRI：K8s 到底用哪个 runtime

很多人以为 K8s 底下跑的是 Docker，其实早就不是了。演进过程：

1. **Docker 一体机时代**：build、分发、运行全包在 dockerd 一个进程里
2. **OCI 标准**：镜像格式和运行时行为被标准化，**runc** 是运行时的参考实现
3. **containerd 独立**：Docker 把"管理容器生命周期"这块拆出来，成了独立项目
4. **K8s 通过 CRI 直连**：K8s 定义了 **CRI**（Container Runtime Interface），1.24 起移除 dockershim，kubelet 直接对接 containerd

```
 Docker 时代:
 ┌─────────┐   ┌────────────┐   ┌─────────┐   ┌────────────┐   ┌──────┐   ┌──────────┐
 │ kubelet │──►│ dockershim │──►│ dockerd │──►│ containerd │──►│ runc │──►│ 容器进程  │
 └─────────┘   └────────────┘   └─────────┘   └────────────┘   └──────┘   └──────────┘

 现在 (K8s ≥ 1.24):
 ┌─────────┐   CRI (gRPC)   ┌────────────┐   ┌──────┐   ┌──────────┐
 │ kubelet │───────────────►│ containerd │──►│ runc │──►│ 容器进程  │
 └─────────┘                └────────────┘   └──────┘   └──────────┘
```

两个实用结论：

- **docker build 出来的镜像 K8s 照样能跑**——镜像遵循的是 OCI 规范，跟 Docker 这个工具没有绑定
- 生产节点上多半**没有 `docker` 命令**，排查容器用 `crictl`（参数和 docker 高度相似）

### 3.6 AI 场景的镜像三件事

**(a) 大镜像分发**。CUDA 基础镜像动辄 10–20 GB，千卡任务一起拉镜像能直接打爆 Registry，Pod 全卡在 `ContainerCreating`。常见对策：节点**镜像预热**（提前拉好）、**P2P 分发**（Dragonfly 之类，节点之间互传）、**精简基础镜像**（跑推理用 `runtime` 版而不是 `devel` 版基础镜像）。

**(b) nvidia-container-toolkit 怎么把 GPU 塞进容器**。3.1 说过容器没有独立内核，自然也没有独立驱动。toolkit 做的事是在容器启动时（通过 runc 的 hook）**把宿主机的驱动库（libcuda.so 等）和设备文件（/dev/nvidia*）挂进容器**——所以镜像里只需要 CUDA Toolkit，不需要也不应该装驱动。它和 6.1 的 Device Plugin 分工明确：Plugin 决定分哪几张卡，toolkit 负责真正挂进去。

**(c) 权重不进镜像**。镜像只放"环境"（框架、依赖、代码），模型权重和数据集走存储卷挂载（见 6.4）。理由很直接：权重一变就得重打几百 GB 的镜像，分发和存储都是灾难。

---

## 四、八个必须知道的对象（按重要程度排序）

| 对象 | 一句话解释 | AI 场景里干啥 |
|------|----------|------------|
| **Pod** | 一个或多个容器的最小调度单元 | 一个训练 worker = 一个 Pod |
| **Node** | 集群里的一台物理机 / 虚机 | 一台 8 卡 GPU 机 = 一个 Node |
| **Deployment** | 管理一组无状态副本 | 部署推理服务 |
| **Service** | 给一组 Pod 一个稳定的访问入口（VIP + DNS）| 推理服务的 endpoint |
| **Job** | 跑一次就退出的任务 | 单机训练 / 数据预处理 |
| **PV / PVC** | 持久化存储声明 | 挂 NAS、CephFS、放 checkpoint 和数据集 |
| **ConfigMap / Secret** | 注入配置和密钥 | wandb token、HF token、训练超参 |
| **Namespace** | 逻辑隔离 | 按团队 / 项目划资源 |

> Pod ⊂ Deployment / Job / StatefulSet，Pod 是被"管"的，你日常打交道更多是上层对象。

---

## 五、最小可运行示例

### 5.1 一个推理服务（Deployment + Service）

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-70b
spec:
  replicas: 2
  selector:
    matchLabels: { app: vllm-llama3 }
  template:
    metadata:
      labels: { app: vllm-llama3 }
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.6.0
        args: ["--model", "/models/llama3-70b", "--tensor-parallel-size", "8"]
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 8       # 关键：申请 8 张 GPU
            memory: 800Gi
            cpu: 64
        volumeMounts:
        - { name: models, mountPath: /models }
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-weights-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama3-svc
spec:
  selector: { app: vllm-llama3 }
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

看几个 AI 场景特有的点：

- `nvidia.com/gpu: 8` —— GPU 通过 **Device Plugin** 暴露，不是普通资源
- `tensor-parallel-size: 8` —— 一个 Pod 占一台 8 卡机，TP 通信走 NVLink
- 模型权重通过 PVC 共享，避免镜像里塞几百 GB

### 5.2 一个分布式训练任务

单个 Job 不够用，分布式训练通常用 **Kubeflow PyTorchJob** 或 **MPIJob**：

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llama-pretrain
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: my-registry/llama-train:v1
            command: ["torchrun", "--nproc_per_node=8", "train.py"]
            resources:
              limits: { nvidia.com/gpu: 8, rdma/hca: 1 }
    Worker:
      replicas: 31              # 32 节点 × 8 卡 = 256 卡训练
      template:
        spec:
          containers:
          - name: pytorch
            image: my-registry/llama-train:v1
            command: ["torchrun", "--nproc_per_node=8", "train.py"]
            resources:
              limits: { nvidia.com/gpu: 8, rdma/hca: 1 }
```

`rdma/hca: 1` 意味着这个 Pod 需要一张 **RDMA / IB 网卡**，由 SR-IOV Device Plugin 提供。这是大规模训练的命门——没有 RDMA，AllReduce 直接拉胯。

---

## 六、AI Infra 必懂的几个深水点

### 6.1 GPU 是怎么"被看到"的：Device Plugin

K8s 自己只认 CPU 和内存，GPU 是通过 **Device Plugin 机制**注入的：

```
   ┌─────────────┐    1. 注册       ┌────────────┐
   │  NVIDIA     │ ───────────────► │  kubelet   │
   │ Device      │                  │            │
   │ Plugin      │ ◄─── 2. 上报 ───┤            │
   │ (DaemonSet) │     "我有8张卡"   └─────┬──────┘
   └─────────────┘                       │
          ▲                              │ 3. 汇报到 API Server
          │ 4. Pod 调度到本节点          │
          │    分配卡 → 注入 env         ▼
          │    NVIDIA_VISIBLE_DEVICES   ┌──────────┐
          │                             │ scheduler│
          └─────────────────────────────┴──────────┘
```

知道这个机制能解释很多奇怪现象：

- Pod 起来了但 `nvidia-smi` 看不到卡 → Device Plugin 没装好 / nvidia-container-toolkit 配置错
- 申请了 8 卡但只有 4 卡可见 → Device Plugin 上报数错了，或被其他 Pod 占了
- 想用 MIG 切片 → 需要换 `nvidia.com/mig-1g.10gb` 这种细粒度资源名

### 6.2 调度：Gang Scheduling 不是默认的

K8s 默认调度器**一个一个 Pod 调**。这对训练是灾难：

```
你: 申请 32 个 worker
K8s 默认: 调度了 30 个，剩 2 个等不到资源
结果: 30 个 Pod 占着 240 张卡空等，整个集群死锁
```

解决方案是**整组调度（Gang Scheduling）**：要么 32 个一起调度成功，要么一个都不起。生产里通常上 **Volcano** 或 **Koordinator** 替换默认调度器。

### 6.3 拓扑感知：8 卡机内部也分远近

一台 8×H100 机器内部不是完全对称的：

```
   GPU0 ── NVLink ── GPU1       GPU2 ── NVLink ── GPU3
     │                 │          │                 │
     └──── NVSwitch ───┴──────────┴──── NVSwitch ───┘
              │                            │
           CPU0  ◄── UPI/QPI/IB ──►      CPU1
              │                            │
           NIC0                           NIC1
```

跨 NUMA、跨 PCIe Switch 的卡之间通信慢得多。所以调度时要：

- **NUMA 对齐**：Pod 用的 GPU 和 CPU 在同一个 NUMA node
- **整机分配**：训练任务尽量独占整机，不要 4+4 分给两个任务

K8s 用 **Topology Manager** + **CPU Manager** 处理，但默认是关的，要在 kubelet 里显式打开 `--topology-manager-policy=single-numa-node`。

### 6.4 存储：模型权重和数据集放哪

| 数据类型 | 大小 | 推荐方案 |
|--------|-----|---------|
| 容器镜像 | 5–20 GB | 镜像仓库 + 节点本地缓存 |
| 训练数据集 | TB 级 | 高吞吐并行文件系统（Lustre / GPFS / JuiceFS / 3FS）|
| 模型权重（推理）| 100GB+ | 共享只读 PVC，或预拉到本地 NVMe |
| Checkpoint | TB 级、高写入 | 对象存储（S3）或并行文件系统 |
| 日志 / 临时 | 视情况 | emptyDir + node 本地盘 |

**别把几百 GB 的权重打进镜像**——拉镜像能拉到怀疑人生。

### 6.5 网络：Pod 之间到底怎么通信

K8s 网络由 **CNI 插件**（Calico、Cilium、Flannel...）实现。AI 场景关键的不是 CNI 本身，而是：

- **训练通信**通常**不走** CNI 的 Pod 网络，而是直接用 host 网络下的 RDMA（IB / RoCE）—— SR-IOV / Multus 让 Pod 直挂 IB 网卡
- **推理服务**走正常 Service VIP，性能要求没那么高

所以一个训练 Pod 经常会挂两套网络：

```
                        ┌────────────────────┐
                        │       Pod          │
                        │                    │
      管理面 / 调度 ───► │ eth0 (CNI)         │
                        │   ↓                │
                        │ kubelet / sidecar  │
                        │                    │
      AllReduce ──────► │ ib0 (SR-IOV/RDMA)  │
                        │   ↓                │
                        │ NCCL / pytorch     │
                        └────────────────────┘
```

---

## 七、kubectl 速查（够用版）

```bash
# === 看 ===
kubectl get pods -n <ns>                       # 列 Pod
kubectl get pods -o wide                       # 看在哪个 Node 上
kubectl describe pod <pod>                     # 看事件、调度失败原因 ★最常用
kubectl logs <pod> -c <container> -f           # 跟日志
kubectl logs <pod> --previous                  # 看上一次崩溃前的日志
kubectl get events --sort-by=.lastTimestamp    # 看集群事件
kubectl top pod / node                         # CPU / 内存使用

# === 进容器 ===
kubectl exec -it <pod> -- bash
kubectl exec -it <pod> -- nvidia-smi

# === 改 ===
kubectl apply -f x.yaml
kubectl delete pod <pod>                       # Deployment 会自动重建
kubectl scale deploy/<name> --replicas=4
kubectl rollout restart deploy/<name>          # 强制重启所有副本

# === GPU 相关 ===
kubectl describe node <node> | grep -A5 Allocatable     # 看 GPU 资源
kubectl get pods -A -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].resources.limits.nvidia\.com/gpu}{"\n"}{end}'  # 谁用了卡

# === Debug ===
kubectl debug node/<node> -it --image=ubuntu   # 起一个特权调试 Pod
```

---

## 八、定位训练任务卡住：一个排错心法

> 训练任务起不来 / 起一半挂了，按这个顺序看：

```
1. kubectl get pods                    # Pod 都在吗？
   ├─ Pending      → 资源不够 / 调度失败 → describe 看 Events
   ├─ ContainerCreating → 拉镜像？挂卷？→ describe 看
   ├─ Error/CrashLoopBackOff → 看 logs
   └─ Running 但训练没动 → 进容器看进程

2. kubectl describe pod <pod>          # 看 Events 一栏 ★
   常见错误：
   - FailedScheduling: insufficient nvidia.com/gpu  → 集群没空卡
   - FailedScheduling: node affinity                 → 标签不对
   - Failed to pull image                            → 仓库 / 网络问题
   - MountVolume.SetUp failed                        → PVC / Secret 问题

3. kubectl logs <pod> -f               # 看进程输出
   - NCCL 报错？      → 网络 / RDMA 问题
   - OOM Killed？     → 减 batch / 调 ZeRO stage
   - hang 住没输出？  → 进容器 py-spy dump

4. kubectl exec -it <pod> -- bash      # 进去看
   - nvidia-smi   → 卡的状态、是否 hang
   - ibstat       → IB 链路是否 up
   - ss -tnp      → 端口连接情况
```

---

## 九、AI Infra 工程师的 K8s "知识下沉"

最后，回到题目——AI infra 的人到底要懂多少？我的体感：

**必须会的（每天都用）**

- 看懂 yaml、写一个 Deployment / Job
- `kubectl get / describe / logs / exec` 排错四件套
- 知道 `requests` / `limits` 区别，知道 GPU 是怎么申请的
- 看得懂 Pod 状态机（Pending → Running → Succeeded / Failed）

**应该懂的（出问题需要）**

- Device Plugin / Topology Manager / NUMA
- PV / PVC / StorageClass 与底层文件系统的关系
- Volcano / Kubeflow Operator 怎么做 gang scheduling
- HPA、PDB、taint/toleration、nodeSelector、affinity

**了解就行（SRE 的活）**

- etcd 怎么备份、控制面怎么 HA
- CNI 内部实现、kube-proxy iptables / IPVS
- 自己写 Operator、CRD、Admission Webhook

---

## 十、一句话总结

K8s 之于 AI infra，就像 Linux 之于普通后端——**不需要会写内核，但要知道每个进程在哪里、为什么 hang 住、怎么救活**。掌握这篇文章涵盖的内容，足够你跟 SRE 平等对话，足够你独立把训练任务和推理服务运行在生产集群上。
