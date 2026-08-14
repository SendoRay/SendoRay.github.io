---
title: "论文浅读笔记"
date: '2026-08-14'
tags:
- prefetch
- TMA
- offload
- KV-Cache
draft: false
summary: "不值得单独成文的论文浅读合集：只抓核心机制与关键数字，不做精读级展开。"
---

> 本页收录**浅读**（非精读）笔记：不值得单独写一篇文章的论文，每篇只抓一个核心机制。

## 2604.26074 — kernel 级 TMA 预取（DAK）

https://papers.cool/arxiv/2604.26074

它只是 kernel 级别的预取，主要通过 TMA 把对应的数据拿到 SMEM 上，直接绕过 HBM。

{{< figure src="/images/paperreading/2604-dak/fig5-dak.png" title="Fig. 5: DAK 将矩阵乘法分区执行 — 权重按 tile chunk 拆分于 host/GPU，TMA 从 host 直取 SMEM" width="80%" >}}

### 预取粒度谱系

```text
细 ◄──────────────────────────────────────────────► 粗
kernel 级          tensor 级           layer 级         expert/模块级
(DAK: kernel       (按单个权重张量      (逐层 prefetch    (MoE 按 expert
 直接访存 host,     调度, 感知生命周期/   权重, 计算通信     驻留/换入换出)
 不搬整块数据)       张量间异构性)        重叠)
```

```text
细 ◄────────────────────────────────────────────────────────────► 粗
kernel级    tensor级    layer级/expert级    组件/状态级           阶段/工作流级
(DAK)      (ATSInfer,  (ITME, chunked     (weights/optimizer/  (rollout ↔ train
           TERAIO)      prefetch)          grads/act/KV 按      整栈切换, 跨任务
                                           生命周期保留或丢弃)    phase 复用)
```
