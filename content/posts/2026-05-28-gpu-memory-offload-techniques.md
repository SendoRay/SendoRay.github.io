---
title: GPU 显存 Offload 技术：训练与推理中的工程实践
date: '2026-05-28'
tags:
- LLM
- GPU
- CUDA

draft: false
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

> 作者：菽陌松囿
> 链接：https://www.zhihu.com/question/1989041714380416374/answer/1990864712833967977
> 来源：知乎


---

有钱有卡的话（团队可操控千卡+左右，模型几百B），边际收益（做东西你要承担风险）有限，但是你如果只有 256/512 卡，单卡 80G-140G 显存，offload 还是有不少收益的，不是没事可做，只是大家还没玩溜，没有现成可抄的。

---

## 训练层面：在有限的卡跑更大模型以及 Context Length，优化 MFU

### 1. 激活 Offload——把训练激活显存收敛到可控 Size

做全量微调和预训练的时候，如果训练语料上下文很长比如 128k，显存开销大头在激活，单层 hidden states 为 2G 左右（`mb_size * seq * hidden_dim`，全量 recompute）。即使用 PP，且这里是连续显存，其实显存剩余 20G 其实是申请不了的（显存碎片问题，另外计算图的显存可以估算下），大部分玩家要 CP，但是 CP 要把模型参数/梯度复制 CP 份。

结合 Ring Attention，其实你可以结合 recompute，把 hidden states 在前向 **D2H** 到 CPU（和前向计算 overlap），反向把上一层 hidden states **H2D** 到显存同当前层 backward 计算 overlap。D2H/H2D 用不同的 stream，这种情况下基本可以做到**无损 overlap**，把显存开销收敛到单层计算图的激活。

进一步把单层从 attention 和 FFN 切两部分，可以把激活收敛到某个 block。通过以上的优化，训练的激活显存能优化到可控的 size。记得当时用 128 卡 80G 显存全量微调了 128k 的 DeepSeek-V2。

另外你其实增加 batch size，然后通信占比可以忽略不计，可大幅提升 MFU，因为 80/90% 都是 attention 和 GEMM。

### 2. 优化器状态 Offload——节省数 TB 显存

对于模型训练过程中，显存固定开销，参数、梯度、优化器状态比值为 **1:1:6**。如果显卡数量不足以 shard 优化器状态，其实选择把 opt states 放在 CPU 并用 SIMD 指令或者 CPU 线程去加速也是很好的（这个 Colossal-AI 最先做的），因为 Adam 更新也不是算力密集型，基本对训练 MFU 影响忽略不计。

比如 V3 模型你可以 `671 * 2 * 6`，差不多能省 **6-7T 显存**，那多余的显存让你支持更大模型/seq/batch size，调整 TP/PP/CP/EP 配置，达到更好的 MFU。

### 3. RL 训练——推理与训练复用资源

做 RL 时，Reference Model、Rollout 阶段 Actor Model、Train 阶段 Policy Model 都可以交织做 offload，可以让推理和训练**复用资源**，并且合理设置推理 TP 等配置，有利于把 batch size 做大，达到比较高的资源利用率。

---

## 推理层面：优化吞吐

### 1. Diffusion Model 的无损 Weight Offload

对于 Diffusion Model，因为是 full attention，你可以把 weight 放 CPU，然后**计算当前层预取下一层参数**可以做到无损 offload。可以在低端卡去做 Diffusion Model 推理，只需要 **2-4G 显存**就可以跑 14B 的 DiT 模型，性能无损，多余的显存可以用来混布或者组 batch。

### 2. LLM Prefill 阶段的 Weight Offload

对于 LLM 先按照请求分 bucket，把 prompt 很长的请求 route 到特定实例。因为 prefill 是算力密集型，也可以**无损的 offload weight**，支持更大吞吐，而且多余的显存可以部署 decode 实例。

### 3. KV Cache Offload

把冷数据 offload 到 CPU，很多推理引擎应该做过。

### 4. 综合实践——KTransformers 等项目

譬如 KTransformers 项目就做了以上大部分工作。别人跑不了的模型和配置，通过合理的 offload 策略能搞定，并且性能不错。
