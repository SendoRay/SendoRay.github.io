---
title: "Understanding Data Movement in Tightly Coupled Heterogeneous Systems: A Case Study with the Grace Hopper Superchip"
date: '2026-08-13'
tags:
- GPU
- NVLink-C2C
- Grace-Hopper
- data-movement
draft: false
summary: "以 GH200 为案例分析紧耦合异构系统的数据移动：本地 HBM 带宽 4000 GB/s，NVLink-C2C 450 GB/s/方向，跨互联路径均受 C2C 限制。"
---

{{< figure src="/images/paperreading/2408-grace-hopper/fig6-datapaths.png" title="Fig. 6: GH200 读写（上）与拷贝（下）数据路径 — 本地 HBM 带宽 4000 GB/s，其余路径受限于 C2C 450 GB/s" width="80%" >}}


https://arxiv.org/pdf/2408.11556

{{< figure src="/images/paperreading/2408-grace-hopper/fig-topology.png" title="Quad GH200 拓扑 — NVLink-C2C 450 GB/s/方向、NVLink 150 GB/s、Grace 互联与 Slingshot" width="95%" >}}



第一个是NVLink-C2C
NVLink 是一种互连技术，最初由 NVIDIA 设计，旨在替代 PCIe，提供更多功能并面向多 GPU 系统。它经过几代迭代，链路速度不断提升，目前已发展到第四代。NVLink-C2C (C2C) 扩展了 NVLink 系列，提供了一种高速互连方案，用于构建由多个芯片组组合而成的集成设备。它支持不同类型 PU 之间的快速且缓存一致的通信。其架构允许每个数据信号的带宽达到 40 Gbps，每个链路支持 9 个数据信号。在 GH200 中，Grace 和 Hopper 各自集成了 10 个链路，每个方向的总带宽为 450 GB/s 

