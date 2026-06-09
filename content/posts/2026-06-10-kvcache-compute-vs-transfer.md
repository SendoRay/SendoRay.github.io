---
title: KV Cache 是算还是传？一份 Roofline 视角的全景账
date: '2026-06-10'
tags:
- LLM
- GPU
- Inference
- AI-Infra

draft: false
math: true
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

## 一、问题的起点

做 LLM 推理优化时，几乎所有"花活"都绕不开同一个问题：

> **这段 KV Cache，是该重新算一遍，还是从别处搬过来？**

它出现在很多场景里：

- **Prefix Caching / RadixAttention**：命中前缀时，是把 KV 从 host 内存（甚至 SSD）拉回 GPU，还是直接 prefill 一遍？
- **PD 分离（Prefill / Decode 解耦）**：Prefill 节点把 KV 通过 NIC 推给 Decode 节点，是否真的比让 Decode 自己 prefill 更划算？
- **KV Offload**：显存吃紧把 KV 换到 CPU，等下一次访问再 H2D 回来，划不划算？
- **InfiniGen 这类异构 KV 管理**：哪些 layer / head 该留在 HBM，哪些可以下放？

这些问题其实是同一个**"算 vs 传"**的取舍，本质上可以用一条非常朴素的不等式来判断：

$$
\boxed{\,T_{\text{recompute}} \;>\; T_{\text{transfer}} \;\Longrightarrow\; \text{传比算划算}\,}
$$

这篇博客就把这条不等式拆开，写一份带数值的全景账。

## 二、两边各是什么

### 2.1 重算时间 $T_{\text{recompute}}$

如果选择**不传、自己重新 prefill**，所花的时间由 GPU 的算力 roofline 决定：

$$
T_{\text{recompute}} \;=\; \frac{F_{\text{prefill}}}{P_{\text{gpu}} \cdot \text{MFU}}
$$

- $F_{\text{prefill}}$：prefill 阶段需要的总 FLOPs
- $P_{\text{gpu}}$：GPU 在该精度下的峰值算力（如 H100 BF16 约 1000 TFLOPs/s）
- $\text{MFU}$：实际能跑到的算力利用率，prefill 阶段一般 0.3 ~ 0.5

**$F_{\text{prefill}}$ 的估算**：对一个参数量为 $\Theta$ 的 dense Transformer，处理 $N$ 个 token 的 prefill，主干 GEMM 的 FLOPs 约为：

$$
F_{\text{gemm}} \approx 2\,\Theta\,N
$$

注意 attention 部分有 $O(N^2)$ 的项：

$$
F_{\text{attn}} \approx 2 \cdot L \cdot N^2 \cdot d_{\text{model}}
$$

其中 $L$ 是层数。当 $N$ 很大（长上下文）或者用了 GQA 让 GEMM 部分相对变小时，attention 这一项会逐渐压过 GEMM，roofline 会从 compute-bound 滑向 attention 的 memory-bound。

### 2.2 传输时间 $T_{\text{transfer}}$

如果选择**把现成的 KV Cache 搬过来**，所花的时间由"链路带宽"决定：

$$
T_{\text{transfer}} \;=\; \frac{S_{\text{kv}}}{B_{\text{eff}}}
$$

其中传输带宽是整条路径上**最窄的那一段**：

$$
B_{\text{eff}} \;=\; \min\bigl(B_{\text{h2d}},\; B_{\text{nic}},\; B_{\text{nvlink}},\; B_{\text{ssd}},\; \dots\bigr)
$$

常见瓶颈带宽（典型上限，实际打折）：

| 链路 | 单向带宽 | 实测可用 |
|---|---|---|
| HBM3（卡内显存） | 3 TB/s | ~80% |
| NVLink 4.0（卡间） | 900 GB/s | ~80% |
| PCIe Gen5 x16（H2D） | 64 GB/s | ~50 GB/s |
| PCIe Gen4 x16（H2D） | 32 GB/s | ~25 GB/s |
| 400 GbE / IB NDR（NIC） | 50 GB/s | ~40 GB/s |
| 100 GbE / IB EDR（NIC） | 12.5 GB/s | ~10 GB/s |
| NVMe SSD（顺序读） | 7 GB/s | ~5 GB/s |

**$S_{\text{kv}}$ 的估算**：每个 token 的 KV Cache 大小为：

$$
s_{\text{kv}} \;=\; 2 \cdot L \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot \text{bytes}
$$

- 系数 2 对应 K 和 V
- $H_{\text{kv}}$：KV head 数（GQA / MQA 在这里发挥作用）
- $\text{bytes}$：dtype 字节数（FP16/BF16=2，FP8=1）

总传输量：$S_{\text{kv}} = N \cdot s_{\text{kv}}$。

## 三、把不等式两边都化成"每 token"

把 $T_{\text{recompute}}$ 和 $T_{\text{transfer}}$ 都除以 $N$（先忽略 attention 二次项），得到非常干净的两个量：

$$
t_{\text{recompute}} \approx \frac{2\,\Theta}{P_{\text{gpu}} \cdot \text{MFU}},\qquad
t_{\text{transfer}} = \frac{s_{\text{kv}}}{B_{\text{eff}}}
$$

定义 **算传比** $\rho$：

$$
\rho \;=\; \frac{t_{\text{recompute}}}{t_{\text{transfer}}} \;=\; \frac{2\,\Theta \cdot B_{\text{eff}}}{P_{\text{gpu}} \cdot \text{MFU} \cdot s_{\text{kv}}}
$$

- $\rho > 1$：**传比算快**，应该传；
- $\rho < 1$：**算比传快**，重 prefill 反而更优；
- $\rho \approx 1$：边界，要结合 overlap、显存压力、batch 重组等二阶因素再判。

这个比值的好处是：**它把模型结构、硬件参数、链路类型全部塞进了一个数**，做架构决策时一眼就能扫出来。

## 四、几个真实场景的算账

下面以 **Llama-3-70B（GQA，8 KV heads，80 层，head_dim=128，BF16）** 为例。

- $\Theta \approx 70 \times 10^9$
- $s_{\text{kv}} = 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ B $\approx$ **320 KB / token**
- $P_{\text{gpu}} \cdot \text{MFU}$：H100 BF16 取 $1000 \times 0.4 = 400$ TFLOPs/s

代入：

$$
t_{\text{recompute}} \approx \frac{2 \times 70 \times 10^9}{4 \times 10^{14}} = 350\ \mu s\,/\,\text{token}
$$

注意这是**单卡**的数。实际 70B 通常 TP=4 / TP=8，每张卡只承担一部分 GEMM，所以单 token prefill 的真实壁钟时间会更小，但**单位算力消耗**没变。下面比较时直接用"每 token 的算力时间"。

### 场景 A：同机内 H2D（CPU → GPU，PCIe Gen5）

$$
t_{\text{transfer}} = \frac{320\ \text{KB}}{50\ \text{GB/s}} \approx 6.4\ \mu s\,/\,\text{token}
$$

$$
\rho \approx \frac{350}{6.4} \approx 55
$$

**结论**：H2D 链路下，把 KV Cache 从 CPU 拉回来比重新 prefill 快 50 倍以上。这就是 Prefix Caching / KV Offload 在单机场景下几乎"无脑划算"的根源。

### 场景 B：跨机 PD 分离（400 GbE / IB NDR）

$$
t_{\text{transfer}} = \frac{320\ \text{KB}}{40\ \text{GB/s}} \approx 8\ \mu s\,/\,\text{token}
$$

$$
\rho \approx \frac{350}{8} \approx 44
$$

**结论**：高速 NIC 下 PD 分离依然有近一个数量级优势。这解释了为什么 Mooncake / DistServe 这类方案敢把 prefill 拆出去单独组池子。

### 场景 C：低端集群（100 GbE）

$$
t_{\text{transfer}} = \frac{320\ \text{KB}}{10\ \text{GB/s}} \approx 32\ \mu s\,/\,\text{token}
$$

$$
\rho \approx \frac{350}{32} \approx 11
$$

**结论**：依然显著划算，但留给"传输和计算 overlap"的 margin 已经明显变小，链路抖动会直接打到尾延迟上。

### 场景 D：SSD 持久化 KV

$$
t_{\text{transfer}} = \frac{320\ \text{KB}}{5\ \text{GB/s}} \approx 64\ \mu s\,/\,\text{token}
$$

$$
\rho \approx \frac{350}{64} \approx 5.5
$$

**结论**：仍然 > 1，但加上 SSD 的随机访问延迟、文件系统 overhead，实际收益会打折。一般用于**离线 prefix 命中率高、且显存/内存都装不下**的场景。

### 场景 E：MHA 模型（无 GQA，比如 GPT-3 175B）

如果模型是 64 KV heads（典型 MHA），$s_{\text{kv}}$ 直接放大 8 倍到 ~2.6 MB/token。这时即便 H2D：

$$
\rho \approx \frac{2 \times 175 \times 10^9 / (4 \times 10^{14})}{2.6\ \text{MB} / 50\ \text{GB/s}} = \frac{875\ \mu s}{52\ \mu s} \approx 17
$$

**结论**：还是传划算，但比例从 ~50× 滑到 ~17×。**GQA 不仅省显存，也让"传 KV"这件事的性价比下降——重算变得相对没那么亏了**。这也是为什么有些 GQA-heavy 的新模型反而更倾向"重 prefill"路线。

## 五、不等式之外的二阶因素

上面只是 roofline 的"主项"。真实工程里至少还要叠加几条修正：

### 5.1 Overlap

传输和计算可以走不同 stream，理想情况下 H2D / NIC RDMA 完全和当前层的计算重叠。这时 $T_{\text{transfer}}$ 不是直接加在端到端延迟上，而是：

$$
T_{\text{e2e}} \approx \max\bigl(T_{\text{compute}},\; T_{\text{transfer}}\bigr)
$$

只要两边在一个数量级，就有空间通过预取做到**近无损**（参见 [GPU 显存 Offload 技术：训练与推理中的工程实践](/posts/2026-05-28-gpu-memory-offload-techniques/) 里 Diffusion weight offload 的例子）。

### 5.2 Attention 的二次项

当上下文 $N$ 很大（比如 128k），prefill 的 attention 部分不再可忽略。这时 $T_{\text{recompute}}$ 的增长比线性更快，**$\rho$ 会进一步放大，传 KV 的优势越长越明显**。这是 long context + prefix cache 命中场景下收益爆炸的根本原因。

### 5.3 显存和 batch 重组

传 KV 要先有地方放。显存压力大的时候，"传过来"也意味着挤掉别的 request 的 KV，间接降低 batch size 和 MFU。这时候 $\rho$ 高也未必是赢。

### 5.4 命中率

Prefix Caching 的真实收益要乘上一个**命中概率** $p$：

$$
\mathbb{E}[\text{saved}] \;=\; p \cdot (T_{\text{recompute}} - T_{\text{transfer}}) - (1-p) \cdot T_{\text{lookup overhead}}
$$

命中率低的时候，**索引和 lookup 本身的开销可能把收益吃掉**。

### 5.5 量化

KV Cache 量化（FP8 / INT8 / KIVI 之类）会**同时**改变两边：
- $s_{\text{kv}}$ 直接减半甚至 1/4，$t_{\text{transfer}}$ 同比下降；
- 但量化-反量化本身要消耗算力，$t_{\text{recompute}}$ 相对优势略降。

总体仍然是放大 $\rho$，所以 KV 量化在 PD 分离 / Offload 场景里几乎是默认开。

## 六、一张总表

把上面的场景汇总成一张可贴墙上的表（70B GQA，BF16，H100，MFU=0.4）：

| 场景 | 链路有效带宽 | $t_{\text{transfer}}$ | $\rho$ | 结论 |
|---|---|---|---|---|
| 同机 H2D（PCIe Gen5） | 50 GB/s | 6.4 μs | ~55 | 几乎无脑传 |
| PD 分离（400G IB） | 40 GB/s | 8 μs | ~44 | 强烈推荐传 |
| PD 分离（100G） | 10 GB/s | 32 μs | ~11 | 划算但要谨慎抖动 |
| SSD 持久化 KV | 5 GB/s | 64 μs | ~5.5 | 仅离线 / 命中率高时 |
| MHA 175B + H2D | 50 GB/s | 52 μs | ~17 | 仍传，但优势缩水 |

## 七、给工程决策的几句话

1. **先估 $\rho$，再谈架构**。不要凭感觉决定"是不是要做 prefix cache / PD 分离"，先用上面那个公式算一下，能不能拿到 5× 以上的潜在收益。
2. **盯住链路上最窄那一段**。$B_{\text{eff}} = \min(\cdot)$ 不是修辞，决定一切的就是那一段。把 NIC 升到 400G、PCIe 升到 Gen5，往往比改任何 kernel 都管用。
3. **结构优化会反向影响这笔账**。GQA / MQA / KV 量化让 $s_{\text{kv}}$ 变小，"传"的优势在缩；MoE 让 $\Theta$（激活参数）变小，"算"的劣势也在缩。这两条的组合方向决定了一个推理栈到底要不要押注 PD 分离。
4. **Overlap 是二阶但能决定生死**。$\rho$ 算出来等于 2 的场景，做不到 overlap 就是收益归零；做到 overlap 就是端到端时延减半。
5. **命中率比公式更难估**。线下 trace、线上灰度，是工程上唯一靠谱的办法。

## 八、小结

> 「算 vs 传」的本质是 GPU 算力 roofline 和链路带宽 roofline 的赛跑。

记住这一条公式就够用了：

$$
\rho \;=\; \frac{2\,\Theta \cdot B_{\text{eff}}}{P_{\text{gpu}} \cdot \text{MFU} \cdot s_{\text{kv}}}
$$

下次再听到 "Prefix Cache / PD 分离 / KV Offload / InfiniGen" 这些名词，先别急着读论文，把模型的 $\Theta$、$s_{\text{kv}}$ 和你机房里那段最窄的带宽塞进这条式子，先看看 $\rho$ 长什么样。

数能告诉你，这一仗到底值不值得打。
