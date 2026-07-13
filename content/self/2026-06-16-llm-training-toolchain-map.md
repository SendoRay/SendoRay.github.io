---
title: "大模型训练工具链地图：从 CUDA 到 verl，到底谁在哪一层？"
date: '2026-06-16'
tags:
- LLM
- Training
- GPU

draft: false
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

> 很多人刚接触大模型训练时，会同时看到一堆名字：CUDA / PCIe / NVLink / InfiniBand / NCCL，PyTorch / TensorFlow，Transformers / modelscope，Qwen / Llama / DeepSeek，FSDP / DeepSpeed / Megatron，Accelerate / Ray，Trainer / LLaMA-Factory / SWIFT / TRL / OpenRLHF / verl，vLLM / SGLang。
>
> 它们经常出现在同一个 README、同一份训练配置、同一套分布式启动命令里，所以初学者很容易产生一种错觉：**它们好像都是"训练框架"。**
>
> 但它们不是同一层级的东西。这篇文章不讲安装命令，不推 loss 公式，也不展开 optimizer 细节。我们只做一件事：**建立一张大模型训练工具链地图**。

读的时候可以抓住三件事：

1. 它在哪一层？
2. 它解决什么问题？
3. 它和上下游工具怎么配合？

看完之后，你再看到 LLaMA-Factory 的多种启动命令、verl 的多 worker 配置、DeepSpeed ZeRO、FSDP、vLLM rollout，就不会再把它们揉成一团。

---

## 一、先看总图

可以先把大模型训练系统分成这几层：

```
大模型训练 / 后训练工具链
│
├── 0. GPU 计算、互联与通信基础
│   ├── CUDA：让程序使用 NVIDIA GPU 做通用计算
│   ├── PCIe：服务器内部连接 CPU、GPU、网卡、硬盘等设备的通用高速总线
│   ├── NVLink：同一台机器内 GPU 和 GPU 之间的高速互联
│   ├── InfiniBand / IB：多机训练中常见的高速网络
│   └── NCCL：利用底层互联和网络做高效 GPU 通信
│
├── 1. 底层深度学习框架
│   ├── PyTorch
│   └── TensorFlow
│
├── 2. 模型结构接口 / 模型生态 SDK
│   ├── Transformers：模型结构、权重加载、训练和推理接口
│   └── modelscope：ModelScope 平台 SDK、模型下载和任务 pipeline
│
├── 3. 模型系列 / 具体 checkpoint
│   ├── Qwen / Llama / DeepSeek / GLM 等模型系列
│   └── Qwen2.5-7B-Instruct / DeepSeek-R1 等具体 checkpoint
│
├── 4. 分布式训练后端 / 并行与显存优化方案
│   ├── FSDP：PyTorch 原生全分片数据并行方案
│   ├── DeepSpeed：覆盖 DP / ZeRO / PP / TP / SP / EP / Offload 等能力
│   └── Megatron：面向大模型的 TP / PP / SP / CP / EP 并行训练框架
│
├── 5. 启动、适配与任务调度
│   ├── Accelerate：Hugging Face 生态里的训练启动与分布式适配工具
│   └── Ray：多机多进程任务调度与资源管理框架
│
├── 6. 训练工具 / 训练库 / 后训练框架
│   ├── Trainer：Transformers 里的通用训练器
│   ├── PEFT：Hugging Face 参数高效微调库，常用于 LoRA / QLoRA
│   ├── TRL：Hugging Face 生态里的后训练 trainer / 算法库
│   ├── LLaMA-Factory：高层微调入口 / 微调工作台
│   ├── SWIFT / ms-swift：训练、推理、评测、部署一体化工具
│   ├── OpenRLHF：RLHF 工程框架
│   └── verl：大模型 RL post-training 框架
│
└── 7. 推理 / rollout 引擎
    ├── vLLM
    └── SGLang
```

先用一句话抓住主线：

> 底层硬件和通信负责"算得动、传得快"；PyTorch 负责"怎么计算和求导"；Transformers / modelscope 负责"怎么加载和使用模型"；FSDP / DeepSpeed / Megatron 解决"多卡怎么训练"；PEFT 负责"怎么做参数高效微调"；Trainer / LLaMA-Factory / SWIFT / TRL / OpenRLHF / verl 负责"训练任务怎么组织"；vLLM / SGLang 负责"怎么高吞吐生成"。

更短一点：

- **硬件通信层**：让计算和通信发生
- **计算框架层**：定义张量、模型、自动求导
- **模型接口层**：加载模型、tokenizer、checkpoint
- **分布式训练层**：切参数、切梯度、切计算、做同步
- **高层工具层**：组织数据、配置、训练任务和算法流程
- **推理 / rollout 层**：高吞吐生成 token

---

## 二、第零层：CUDA、PCIe、NVLink、IB、NCCL

这一层最靠近硬件。

| 名词 | 先怎么理解 |
|---|---|
| CUDA | 让程序使用 NVIDIA GPU 做通用计算 |
| PCIe | 服务器内部连接 CPU、GPU、网卡、硬盘等设备的通用高速总线 |
| NVLink / NVSwitch | 单机内部 GPU 和 GPU 之间的高速互联 |
| InfiniBand / IB | 多机训练中常见的高速网络 |
| NCCL | 利用 PCIe / NVLink / IB 等通道做高效 GPU 通信 |

可以这样记：

- **CUDA**：让 GPU 做计算。
- **PCIe**：服务器内部通用设备互联。
- **NVLink**：同一台机器里 GPU 和 GPU 高速互联。
- **IB**：多台机器之间高速互联。
- **NCCL**：在这些通道之上做 all-reduce / all-gather / reduce-scatter 等通信。

训练代码里你可能不会直接写 NCCL，也不会手动操作 NVLink 或 IB，但它们在底层非常重要。比如：

- PyTorch 调 CUDA 做张量计算。
- DeepSpeed / FSDP / Megatron 调 NCCL 做多卡通信。
- NCCL 根据机器环境利用 PCIe / NVLink / IB 等路径传数据。

如果你看到 `CUDA out of memory`，通常是显存不够；如果看到 NCCL 错误，往往和多卡通信、驱动、网络、IB、环境变量或集群配置有关。

---

## 三、PyTorch / TensorFlow：底层深度学习框架

PyTorch 和 TensorFlow 是同一层级的东西：**底层深度学习框架**。

PyTorch 最早由 Facebook AI Research，也就是 Meta 旗下 FAIR 推动开发；现在由 Linux Foundation 旗下的 PyTorch Foundation 治理。

PyTorch 负责的是：

- 张量计算
- GPU 计算
- 自动求导
- 模型定义
- 训练循环
- 分布式通信接口

很多上层工具最终都依赖 PyTorch：

- Transformers 里的大模型通常是 PyTorch 模型。
- FSDP 是 PyTorch 原生分布式能力。
- DeepSpeed 通常基于 PyTorch 使用。
- Megatron 主要基于 PyTorch。
- verl 训练 actor / critic 时也依赖 PyTorch 生态。

TensorFlow 也是底层深度学习框架，由 Google 推出。当前开源 LLM 训练生态里，PyTorch 更常见。

---

## 四、Transformers 和 modelscope：别简单并列成"两个模型库"

这里要先分清两个概念。

### 4.1 Transformers 是什么？

这里说的 Transformers，指 Hugging Face 的 `transformers` Python 库，不是 Transformer 模型架构本身。

它更像：

> 预训练模型的结构实现、权重加载、tokenizer、训练、保存和推理接口库。

常见入口包括：

- `AutoTokenizer`
- `AutoModelForCausalLM`
- `from_pretrained()`
- `generate()`
- `Trainer`
- `TrainingArguments`
- `config.json`
- `modeling_xxx.py`

所以 Transformers 不只是推理库。它至少覆盖四类能力：

- **模型定义**：提供 BERT、GPT、Llama、Qwen 等模型结构实现。
- **模型加载**：用 `from_pretrained()` 加载配置、tokenizer 和权重。
- **模型训练**：提供 `Trainer` / `TrainingArguments`，也可以配合 PyTorch 自己写训练循环。
- **模型推理**：提供 `generate()` 等生成接口。
- **模型保存**：保存训练后的权重、配置和 tokenizer。

两个最常见函数可以这样理解：

- `from_pretrained()`：从模型名或本地路径加载 tokenizer、模型结构、权重和配置。
- `generate()`：让已经加载好的生成模型根据输入继续生成文本。

可以粗略理解为：

- `from_pretrained()`：把模型请进内存里。
- `generate()`：让模型开始续写、回答、生成。

如果要训练或微调，Transformers 也可以参与：

- 用 `AutoModelForCausalLM.from_pretrained(...)` 加载基座模型
- 用 `Trainer` / `TrainingArguments` 组织标准训练流程
- 或者直接拿 Transformers 模型接入 PyTorch 自定义训练循环
- 训练后再 `save_pretrained(...)` 保存模型和 tokenizer

### 4.2 modelscope 是什么？

modelscope 是 ModelScope / 魔搭生态的 Python SDK。它更偏：

- 从 ModelScope 下载模型
- 管理本地缓存
- 提供 pipeline 任务接口
- 连接 ModelScope 平台、数据集和工具生态

常见组合是：

```python
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = snapshot_download("qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
```

这时分工很清楚：

- **modelscope**：负责从 ModelScope 下载模型。
- **Transformers**：负责加载 tokenizer、模型结构和权重。
- **PyTorch**：负责底层计算。

所以，transformers 和 modelscope 可以放在同一层讨论，因为它们都靠近"模型怎么被找到、下载、加载、运行"；但它们不是完全同类工具。

---

## 五、Qwen / Llama / DeepSeek：模型系列，不是工具

Qwen、Llama、DeepSeek、GLM、Mistral、InternLM、Baichuan 等，更准确地说是**模型系列、模型家族或模型品牌**。

真正落到训练脚本里的，通常是某个具体 checkpoint，例如：

- Qwen2.5-7B-Instruct
- Qwen3-32B
- Llama-3.1-8B-Instruct
- DeepSeek-V3
- DeepSeek-R1
- DeepSeek-R1-Distill-Qwen

一句话：

> 模型是被训练和部署的对象，框架和工具是训练、微调、部署模型的手段。

你不会说"我训练了一个 PyTorch"，你会说"我用 PyTorch / Transformers / DeepSpeed 训练了某个 Qwen 系列 checkpoint"。

### 5.1 怎么在 Hugging Face 上看一个模型？

打开任意一个模型主页，例如 `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct`，几个最值得看的入口：

| 入口 | 看什么 |
|---|---|
| `Model card`（README.md） | 模型介绍、训练数据、能力范围、license |
| `Files and versions` | 实际权重文件、tokenizer 文件、配置文件 |
| `config.json` | 模型架构超参（层数、hidden size、head 数、词表大小等） |
| `tokenizer_config.json` / `tokenizer.json` | 分词器类型、特殊 token、chat template |
| `generation_config.json` | 默认采样参数（temperature、top_p、max_new_tokens 等） |
| `model.safetensors.index.json` | 权重分片索引，能看到所有 tensor 的 key 和形状 |
| `Use this model` 按钮 | Transformers / vLLM / Ollama 等的快速调用代码 |

`config.json` 是判断"这是个什么模型"最直接的依据。它会写明 `architectures`，对应 Transformers 里的某个 `modeling_xxx.py` 实现类，比如：

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_hidden_layers": 28,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "max_position_embeddings": 32768,
  "rope_theta": 1000000.0,
  "vocab_size": 152064,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16"
}
```

只看这一段，就能算出这是个 **dense Transformer + GQA + RoPE** 的 7B 量级模型：

- `num_hidden_layers = 28`：28 个 Transformer block
- `num_attention_heads = 28`、`num_key_value_heads = 4`：query 头是 KV 头的 7 倍 → **GQA（Grouped-Query Attention）**
- `hidden_size = 3584`、`head_dim = 128`：每层的隐藏维度
- `intermediate_size = 18944`：FFN 中间层维度，约等于 `hidden_size × 5.3`，是 SwiGLU 风格 FFN 的典型比例
- `vocab_size = 152064`：词表大小
- `max_position_embeddings = 32768`：最长上下文 32K
- `rope_theta = 1000000`：RoPE 长上下文外推用的 base

如果想再往细节看，可以在仓库里点开 `model.safetensors.index.json`，能看到所有权重张量的 key，比如：

```
model.embed_tokens.weight                 [152064, 3584]
model.layers.0.self_attn.q_proj.weight    [3584, 3584]
model.layers.0.self_attn.k_proj.weight    [512,  3584]
model.layers.0.self_attn.v_proj.weight    [512,  3584]
model.layers.0.self_attn.o_proj.weight    [3584, 3584]
model.layers.0.mlp.gate_proj.weight       [18944, 3584]
model.layers.0.mlp.up_proj.weight         [18944, 3584]
model.layers.0.mlp.down_proj.weight       [3584, 18944]
model.layers.0.input_layernorm.weight     [3584]
model.layers.0.post_attention_layernorm.weight [3584]
...
model.norm.weight                         [3584]
lm_head.weight                            [152064, 3584]
```

K/V 投影是 `[512, 3584]` = `4 heads × 128 dim`，对应 `num_key_value_heads=4`，再次印证是 GQA。MLP 有 `gate_proj` / `up_proj` / `down_proj` 三件套，是典型的 **SwiGLU** 实现。

如果你只想要在本地一行打印出整张网络结构，最快的方式是：

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto"
)
print(model)
```

输出就是每一层 module 的层级嵌套，能一眼看到 28 层 `Qwen2DecoderLayer` 是怎么堆起来的。

### 5.2 举个例子：把 Qwen2.5-7B-Instruct 画出来

把上面 `config.json` 的数字翻译成结构图，整张网络长这样：

```
输入 token ids  [B, T]
        │
        ▼
┌──────────────────────────────────────────────────────┐
│ Embedding (vocab=152064, dim=3584)                   │
└──────────────────────────────────────────────────────┘
        │  hidden  [B, T, 3584]
        ▼
╔══════════════════════════════════════════════════════╗
║   × 28  Qwen2DecoderLayer                            ║
║                                                      ║
║   ┌──────────────────────────────────────────────┐   ║
║   │ RMSNorm (input_layernorm)                    │   ║
║   ├──────────────────────────────────────────────┤   ║
║   │ Self-Attention (GQA + RoPE)                  │   ║
║   │   Q: [3584] → 28 heads × 128                 │   ║
║   │   K: [3584] → 4  heads × 128  ← KV 共享       │   ║
║   │   V: [3584] → 4  heads × 128  ← KV 共享       │   ║
║   │   每 7 个 Q 头共用 1 组 KV                    │   ║
║   │   RoPE(theta=1e6), max_pos=32768             │   ║
║   │   O: 28×128 → 3584                            │   ║
║   ├──────────────────────────────────────────────┤   ║
║   │ + Residual                                   │   ║
║   ├──────────────────────────────────────────────┤   ║
║   │ RMSNorm (post_attention_layernorm)           │   ║
║   ├──────────────────────────────────────────────┤   ║
║   │ MLP (SwiGLU)                                 │   ║
║   │   gate_proj: 3584 → 18944                    │   ║
║   │   up_proj:   3584 → 18944                    │   ║
║   │   SiLU(gate) ⊙ up                             │   ║
║   │   down_proj: 18944 → 3584                    │   ║
║   ├──────────────────────────────────────────────┤   ║
║   │ + Residual                                   │   ║
║   └──────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════╝
        │  hidden  [B, T, 3584]
        ▼
┌──────────────────────────────────────────────────────┐
│ Final RMSNorm (model.norm)                           │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│ lm_head: 3584 → 152064  (tie_word_embeddings=false)  │
└──────────────────────────────────────────────────────┘
        │
        ▼
   logits  [B, T, 152064]
```

参数量也可以粗算一下，和 7B 对得上：

- 单层 attention：`Q + K + V + O ≈ 3584² + 2 × (512 × 3584) + 3584² ≈ 29.4 M`
- 单层 MLP（SwiGLU 三件套）：`3 × 3584 × 18944 ≈ 203.7 M`
- 单层合计 ≈ **233 M**，× 28 层 ≈ **6.5 B**
- Embedding + lm_head（不共享）：`2 × 152064 × 3584 ≈ 1.09 B`
- 总计 ≈ **7.6 B**，与官方 "7B" 标称一致

换成别的模型，套路完全一样：在 Hugging Face 仓库里翻 `config.json`，看 `architectures` 决定属于哪种结构（`LlamaForCausalLM` / `Qwen2MoeForCausalLM` / `DeepseekV3ForCausalLM` …），看 `num_hidden_layers` / `hidden_size` / `num_attention_heads` / `num_key_value_heads` / `intermediate_size` 这几个数字，就能在脑子里把网络拼出来。MoE 模型还会多出 `num_experts`、`num_experts_per_tok`、`moe_intermediate_size` 等字段，对应的图里就是把 MLP 那一块换成 router + N 个 expert。

> 一个习惯：拿到任何一个 checkpoint，先打开它的 `config.json` 扫一眼，再决定用什么框架、要多少显存、能不能开 FlashAttention、能不能直接上 vLLM。

---

## 六、FSDP、DeepSpeed、Megatron：分布式训练三大高频名词

先用 DDP 做参照。DDP 是 PyTorch 里的基础数据并行机制：

- 每张 GPU 放一整份模型
- 不同 GPU 处理不同数据
- 最后同步梯度

问题是：每张卡都存完整模型。模型大了之后，这种方式很快撑不住。

### 6.1 FSDP：PyTorch 原生分片方案

FSDP 是 PyTorch 官方的 Fully Sharded Data Parallel。它的核心是：

- 参数分片
- 梯度分片
- 优化器状态分片

普通 DDP：

```
GPU0：完整模型状态
GPU1：完整模型状态
GPU2：完整模型状态
GPU3：完整模型状态
```

FSDP：

```
GPU0：一部分模型状态
GPU1：一部分模型状态
GPU2：一部分模型状态
GPU3：一部分模型状态
```

一句话：**FSDP 是 PyTorch 原生的省显存分布式训练方案。**

### 6.2 DeepSpeed：不只是 ZeRO

DeepSpeed 是 Microsoft 的综合训练优化框架。很多人一提 DeepSpeed 就想到 ZeRO，这没错，但不完整。

DeepSpeed 常见能力包括：

- **DP**：Data Parallel，数据并行
- **ZeRO**：切分 optimizer states、gradients、parameters 等训练状态
- **PP**：Pipeline Parallel，流水线并行
- **TP**：Tensor Parallel，张量并行
- **SP**：Sequence Parallel，序列并行
- **EP**：Expert Parallel，专家并行，常见于 MoE
- **3D Parallelism**：DP / TP / PP 的组合
- **Offload**：把部分状态卸载到 CPU / NVMe，缓解显存压力

ZeRO 分级可以这样记：

- **ZeRO-1**：切 optimizer states
- **ZeRO-2**：切 optimizer states + gradients
- **ZeRO-3**：切 optimizer states + gradients + parameters

DeepSpeed ZeRO-3 和 FSDP 在目标上很像：都希望减少每张 GPU 上重复存储的模型状态。但 DeepSpeed 不等于 ZeRO。它也有 PP、TP、SP、EP、offload 等能力。

需要注意的是，DeepSpeed 内部不同并行能力不是任意组合都能开，具体能否组合取决于 DeepSpeed 版本、训练模式和配置。比如：

- Pipeline Parallel 在官方文档中明确不兼容 ZeRO-2 / ZeRO-3。
- AutoTP、AutoSP 等能力也要按对应版本文档确认支持范围。

所以更准确的记法是：

> DeepSpeed 是综合训练优化框架；ZeRO 是其中最常用的显存优化 / 数据并行能力，但不是全部。

### 6.3 Megatron：大模型并行训练框架

Megatron / Megatron-Core 更强调大模型并行训练。常见能力包括：

- **TP**：Tensor Parallel，张量并行
- **PP**：Pipeline Parallel，流水线并行
- **SP**：Sequence Parallel，序列并行
- **CP**：Context Parallel，上下文并行
- **EP**：Expert Parallel，专家并行

可以先这样理解：

- **TP**：把大矩阵计算切到多张 GPU 上。
- **PP**：把模型不同层放到不同 GPU 上。
- **SP / CP**：围绕序列维度和长上下文做并行。
- **EP**：MoE 模型里把专家分布到不同 GPU 上。

Megatron 的核心标签不是 offload。某些实现或组合方案里可以出现 offload，但从入门角度看，不应把 offload 当成 Megatron 的核心卖点；如果看到 Megatron + DeepSpeed，offload 往往主要来自 DeepSpeed。

### 6.4 三者对比

| 对比项 | FSDP | DeepSpeed | Megatron |
|---|---|---|---|
| 来源 | PyTorch 官方 | Microsoft | NVIDIA |
| 先怎么理解 | PyTorch 原生状态分片 | 综合训练优化框架 | 大模型并行训练框架 |
| 常见能力 | 全分片数据并行 | DP / ZeRO / PP / TP / SP / EP / Offload | TP / PP / SP / CP / EP |
| 入门关键词 | 省显存 | ZeRO 最常见，但不只 ZeRO | 模型并行 |
| 典型场景 | 中大模型微调、研究实验 | 大模型训练、省显存、多种并行组合 | 70B、百B、MoE、长上下文、多机多卡 |

入门时可以说：

- FSDP 和 DeepSpeed ZeRO 更容易先理解为"切模型状态，降低显存冗余"。
- Megatron 更容易先理解为"切模型计算，做大规模模型并行"。

但这只是入门视角。真实大规模训练里，DeepSpeed 和 Megatron 都包含多种并行与系统优化能力，边界不是绝对的。

---

## 七、Accelerate 和 Ray：一个管启动，一个管调度

### 7.1 Accelerate

Accelerate 是 Hugging Face 给 PyTorch 训练脚本准备的"启动器 + 适配器"。它解决的问题是：

- 这份训练代码跑在哪个设备上？
- 模型和 batch 放到 CPU 还是 GPU？
- 8 张 GPU 要不要启动 8 个进程？
- 混合精度怎么开？
- DeepSpeed / FSDP 怎么接？

核心动作可以概括成：

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
```

重点不是语法，而是思想：

- `accelerator.prepare(...)`：帮模型、优化器、dataloader 适配设备和分布式环境。
- `accelerator.backward(loss)`：按当前环境正确做反向传播。

一句话：

> Trainer 是"帮你组织训练流程"；Accelerate 是"帮训练流程在不同硬件和分布式环境里跑起来"。

### 7.2 Ray

Ray 是通用分布式计算和任务调度框架。它不是深度学习框架，不负责自动求导；也不是 NCCL，不负责具体 all-reduce 通信；也不是 vLLM，不负责推理加速。

Ray 关心的是：

- 多个任务放到哪些机器上？
- 每个 worker 用多少 CPU / GPU？
- actor、reward、rollout、trainer 这些角色怎么调度？
- 任务失败、资源分配、集群提交怎么管理？

在普通 SFT 里，你可能不直接关心 Ray。但在 RLHF / RL post-training 中，系统常有多个角色：

- **actor**：正在被优化的策略模型
- **reference model**：通常冻结，用来提供 KL 约束或参考 logprob
- **reward model / reward function**：给回答打分
- **critic / value model**：估计 value，PPO 类算法里常见
- **rollout worker**：负责批量生成样本
- **trainer worker**：负责训练、更新和同步

这些角色可能分布在多机多卡上。Ray 的价值就是把它们组织成一个分布式系统。

一句话：

> Accelerate 更像训练脚本启动适配器；Ray 更像集群任务调度器。

### 7.3 那 Kubernetes 在哪一层？

K8s 在这张地图里**比 Ray 更底层**，是**集群基础设施层**——它不关心你跑的是 SFT 还是 RLHF，只关心"这堆机器、这些 GPU、这些容器怎么管"。可以这样排：

```
应用 / 算法层：    LLaMA-Factory / verl / OpenRLHF / Trainer
                          ↓
角色调度层：       Ray（actor / rollout / trainer worker 编排）
                          ↓
作业 / 容器编排层：Kubernetes  或  Slurm
                          ↓
节点 / 硬件层：    物理机、GPU、网卡、IB、存储
```

K8s 关心的事情和 Ray 不重叠：

| 维度 | Kubernetes | Ray |
|---|---|---|
| 抽象单位 | Pod / Container / Node | Actor / Task / Placement Group |
| 关心什么 | 容器生命周期、资源配额、网络、存储、调度策略 | Python 函数 / actor 怎么放到哪台机器上跑 |
| 谁在用 | 平台 / 运维 / SRE | 算法 / 训练框架开发者 |
| 看到的世界 | 一堆节点上的 Pod | 一个逻辑上的 Ray cluster |

在大模型训练里常见的几种组合：

- **物理机直接跑**：登录裸金属或虚机，`torchrun` / `deepspeed` 起进程，不用 k8s 也不用 Ray。小规模实验最常见。
- **Slurm 集群**：HPC 风格的作业调度器，`sbatch` 提任务、按节点分配 GPU。学术界、超算中心、很多大厂内部训练平台仍以 Slurm 为主。
- **Kubernetes 集群**：云原生平台、内部训练平台多用这种。再配合一些专门的算子让它"懂训练"：
    - **KubeRay / RayCluster Operator**：在 k8s 上一键拉起一个 Ray cluster，verl / OpenRLHF 在上面跑。
    - **Kubeflow Training Operator**（PyTorchJob、MPIJob 等）：把多机多卡训练封装成 k8s CRD，自动起 master/worker Pod 并配好 NCCL 环境。
    - **Volcano / YuniKorn**：批处理调度器，支持 gang scheduling（要么所有 worker 一起起，要么都不起），解决 k8s 默认调度器不适合 MPI 类作业的问题。
    - **GPU Operator / Network Operator**：管 NVIDIA driver、CUDA、GPU Feature Discovery、IB / RDMA 网络配置。

所以同一个 verl 训练任务，可以是：

```
verl
  └── 在 RayCluster 上调度 actor / rollout worker
        └── RayCluster 由 KubeRay Operator 拉起
              └── 跑在 Kubernetes 上
                    └── 集群节点是带 GPU + IB 的物理机
```

也可以是：

```
verl
  └── 用 Ray 做角色调度
        └── Ray cluster 直接起在裸金属节点上（不走 k8s）
              └── 节点是带 GPU + IB 的物理机
```

K8s 和 Ray 不是替代关系：

- **没 k8s 也能跑训练**：物理机 + Slurm + Ray 是经典组合。
- **没 Ray 也能用 k8s**：单机或纯 SFT 任务直接 PyTorchJob 起 `torchrun` 就够了。
- **两者一起用**：大规模 RLHF / 多租户训练平台常见，k8s 管资源和容器、Ray 管算法角色。

一句话：

> Kubernetes 是"机房和容器的操作系统"；Ray 是"分布式 Python 程序的运行时"；它们一个偏运维基础设施，一个偏应用层调度，叠在一起才构成大规模训练平台。

---

## 八、Trainer、LLaMA-Factory、SWIFT、TRL、OpenRLHF、verl

这几个名字经常一起出现，但层级不同。

| 工具 | 更准确的定位 | 典型任务 | 怎么理解 |
|---|---|---|---|
| Trainer | Transformers 里的通用训练器 | 分类、SFT、小中规模微调 | 训练循环封装 |
| PEFT | Hugging Face 参数高效微调库 | LoRA、QLoRA、Adapter 等 | 只训练少量新增参数 |
| TRL | Hugging Face 后训练库 | SFT、DPO、Reward Model、PPO、GRPO | 后训练 trainer / 算法库 |
| LLaMA-Factory | 高层微调入口 | SFT、LoRA、QLoRA、DPO、RM、部分 PPO | 微调工作台 |
| SWIFT / ms-swift | ModelScope 生态训练部署工具 | SFT、LoRA、多模态、评测、部署 | 一体化工程入口 |
| OpenRLHF | RLHF 工程框架 | PPO、GRPO、RLOO、Ray + vLLM + DeepSpeed | 大规模 RLHF |
| verl | RL post-training 框架 | PPO、GRPO、复杂 rollout-reward-update | 大规模 RL 后训练总控 |

简单区分：

- **Trainer / TRL**：更像训练 API / 算法库。
- **PEFT**：更像参数高效微调能力库。
- **LLaMA-Factory / SWIFT**：更像面向用户的微调平台。
- **OpenRLHF / verl**：更像大规模 RLHF / RL 后训练工程框架。

PEFT 全称是 Parameter-Efficient Fine-Tuning，参数高效微调。它解决的问题是：

> 大模型太大了，全量更新所有参数成本高；能不能只训练一小部分新增参数，同时尽量保留原模型能力？

LoRA / QLoRA 就是最常见的 PEFT 方法。关系可以这样看：

- **Transformers**：加载基座模型
- **PEFT**：给模型挂 LoRA / QLoRA 等可训练小模块
- **PyTorch**：负责底层计算和参数更新
- **Trainer / LLaMA-Factory / SWIFT**：把训练流程封装起来

所以 PEFT 不是训练启动器，也不是分布式后端。它更像"微调方法层"的库：告诉模型哪些参数要训练、哪些参数冻结，以及 LoRA adapter 怎么插入模型。

---

## 九、为什么 LLaMA-Factory 有这么多分布式启动命令？

看 LLaMA-Factory 分布式训练文档时，很多人会被一堆命令吓到：

```bash
llamafactory-cli train ...
FORCE_TORCHRUN=1 llamafactory-cli train ...
torchrun ...
accelerate launch ...
deepspeed ...
USE_RAY=1 llamafactory-cli train ...
ray job submit ...
```

先抓住一句话：

> 这些命令不是"很多个 LLaMA-Factory"，而是同一个 LLaMA-Factory 训练任务，可以用不同方式启动，并接到不同分布式训练后端。

可以类比成：

- **LLaMA-Factory**：你要运送的货物，也就是训练任务本身
- **torchrun / accelerate / deepspeed / ray**：不同司机或调度系统
- **NativeDDP / DeepSpeed / FSDP**：真正负责多卡训练的车辆或引擎
- **CUDA / NCCL**：底层道路和通信基础设施

所以看到很多命令时，不要先问"哪个才是 LLaMA-Factory"。它们大多是在回答另一个问题：**这个 LLaMA-Factory 训练任务，要用什么方式启动？底层用什么分布式后端跑？**

可以分成四层：

```
第 1 层：训练任务
  LLaMA-Factory 负责读配置、加载模型、处理数据、执行 SFT / LoRA / DPO 等训练逻辑

第 2 层：启动方式
  llamafactory-cli / torchrun / accelerate launch / deepspeed / ray job submit

第 3 层：分布式训练后端
  NativeDDP / DeepSpeed / FSDP / FSDP2

第 4 层：底层计算和通信
  PyTorch / CUDA / NCCL
```

再看这些命令分别是什么意思：

| 命令 / 名字 | 它是什么 | 主要作用 |
|---|---|---|
| `llamafactory-cli train` | LLaMA-Factory 自己的入口 | 读取 yaml 配置，进入训练流程 |
| `torchrun` | PyTorch 分布式启动器 | 启动多进程，常用于 NativeDDP |
| `accelerate launch` | Hugging Face 启动器 / 适配器 | 根据配置启动多卡、DeepSpeed、FSDP |
| `deepspeed` | DeepSpeed 启动器 | 直接用 DeepSpeed 引擎启动训练 |
| `ray job submit` | Ray 集群任务提交方式 | 把训练任务提交到 Ray 集群调度 |

注意：上面这几项主要是"怎么启动"。真正训练时用什么后端，还要看配置。

常见组合可以这样记：

- **普通单机多卡**：LLaMA-Factory + torchrun / FORCE_TORCHRUN + NativeDDP
- **想用 Hugging Face 风格管理分布式配置**：LLaMA-Factory + accelerate launch + DDP / DeepSpeed / FSDP
- **显存压力大，想用 ZeRO**：LLaMA-Factory + DeepSpeed 配置 + ZeRO-2 / ZeRO-3
- **想用 PyTorch 原生分片方案**：LLaMA-Factory + Accelerate + FSDP / FSDP2
- **多机集群上提交任务**：LLaMA-Factory + Ray + 具体分布式训练引擎

这里最容易混的是 Accelerate、DeepSpeed、Ray：

- **Accelerate**：更像"启动器 + 适配器"，可以帮你启动 DDP / DeepSpeed / FSDP。
- **DeepSpeed**：更像"训练优化后端"，真正负责 ZeRO、offload、并行训练等能力。
- **Ray**：更像"集群任务调度器"，负责多机资源管理，不等于 DeepSpeed / FSDP。

再压缩成一张关系图：

```
LLaMA-Factory 训练配置
        ↓
选择启动方式：
  llamafactory-cli / torchrun / accelerate / deepspeed / ray
        ↓
选择训练后端：
  NativeDDP / DeepSpeed / FSDP / FSDP2
        ↓
底层执行：
  PyTorch / CUDA / NCCL
```

一句话总结：

> LLaMA-Factory 是高层训练应用；这些命令只是不同启动方式和不同训练后端的组合。命令多，不代表概念乱；先分清"训练任务、启动方式、训练后端、底层通信"四层，就清楚了。

---

## 十、为什么 RL 后训练框架更像一个系统？

SFT 相对简单：给模型输入和标准答案，让模型学会模仿。

RLHF / GRPO / PPO 这类后训练更像一个系统，因为它有多个角色：

- actor 模型生成回答
- rollout 引擎批量生成
- reward 计算
- advantage / return 计算
- actor 更新
- reference model 做约束
- critic / value model 可选

以 verl 为例，它通常会组织这些组件：

- **FSDP / Megatron**：承载 actor / critic / reference 等模型的分布式计算；其中 actor / critic 通常会更新，reference 通常冻结
- **vLLM / SGLang**：高吞吐 rollout 生成
- **Ray / 多进程编排**：多角色任务调度
- **PyTorch**：底层计算
- **Transformers**：模型结构与加载接口

所以 verl 不只是一个 trainer，也不是推理引擎。它更像 **RL 后训练总控框架**。

OpenRLHF 也类似，但常见组合更偏：**Ray + vLLM + DeepSpeed**。

---

## 十一、以 verl 为例：训练框架怎么集成不同层级？

前面讲了很多层：CUDA / NCCL、PyTorch、Transformers、FSDP、Megatron、Ray、vLLM、SGLang。单独看每个词还好，真正进入 RL 后训练时，它们会同时出现在一个系统里。verl 就是一个很典型的例子。

先抓住一句话：

> verl 不是 PyTorch、不是 Transformers、不是 vLLM，也不是 Ray；它是把这些组件组织起来做 RL 后训练的框架。

也就是说，verl 更像"总控层"：它决定一次 RL 后训练里先 rollout，还是先算 reward；什么时候算 old_log_prob；什么时候算 advantage；什么时候算 loss；什么时候更新 actor；更新后又怎么把新权重同步给 rollout 引擎。

### 11.1 verl 里各层分别负责什么？

可以把 verl 拆成这样看：

```
任务层：
  PPO / GRPO / RLHF / RLAIF 等后训练任务

总控层：
  verl 负责算法流程编排：
  rollout、reward、advantage、logprob、loss、update 的先后关系和数据流

模型接口层：
  Transformers 提供模型结构、tokenizer、config、权重加载等接口

训练后端层：
  FSDP / Megatron 负责 actor、critic、reference 等模型的分布式执行
  actor / critic 可能参与 backward 和参数更新，reference 通常只做前向计算和 logprob 参考

推理 / rollout 层：
  vLLM / SGLang 负责高吞吐生成 response 样本

调度层：
  Ray 或多进程编排负责把 actor、rollout、reward、trainer 等 worker 调度到多机多卡资源上运行

底层计算与通信层：
  PyTorch 负责张量计算和自动求导
  CUDA / NCCL / PCIe / NVLink / IB 支撑 GPU 计算和多卡通信
```

这里最容易混的是"总控层"和"调度层"：

- **verl**：决定 RL 训练流程怎么走，是算法和数据流的总控。
- **Ray**：决定 worker 放到哪些机器、哪些 GPU 上跑，是资源和任务调度。

所以不要把 Ray 理解成 verl 的"算法大脑"。Ray 更像集群调度系统；verl 才是 RL 后训练流程本身。

### 11.2 一轮 RL 后训练到底怎么跑？

以 verl + vLLM + FSDP / Megatron 为例，一轮训练可以简化成：

1. verl 准备一批 prompts。
2. verl 调用 vLLM / SGLang，让 rollout worker 批量生成 responses。
3. prompt + response 组成一批固定训练样本。
4. reward model 或 reward function 给 response 打分。
5. 在 actor 更新前，用采样时对应的旧策略对固定样本 forward，计算并缓存 `old_log_prob`。
6. verl 根据 reward 计算 advantage / return 等训练信号。
7. 进入更新阶段后，用当前 actor 参数对同一批固定样本 forward，计算 `new_log_prob`。
8. verl 根据 PPO / GRPO / GSPO 等算法计算 loss。
9. 在 FSDP / Megatron 封装下调用 PyTorch autograd 做 backward 和参数更新。
10. 更新后的 actor 权重同步给 vLLM / SGLang，进入下一轮 rollout。

这条链路里，几个组件的分工非常清楚：

- **vLLM / SGLang**：负责把回答"采出来"。
- **reward model / reward function**：负责给回答"打分"。
- **FSDP / Megatron**：封装训练 actor，介入 logprob 计算、梯度同步和参数更新。
- **Ray / 多进程**：负责把不同 worker 调度到多机多卡上。
- **verl**：负责把这些步骤按 RL 算法串起来。

### 11.3 vLLM 生成了 token，但不等于 vLLM 在训练

很多人会卡在这里：

> vLLM 生成回答时不是也会算 logits / logprob 吗？那为什么还要训练 actor 重新算 old_log_prob / new_log_prob？

原因是：vLLM 的目标是高吞吐推理，不是训练。

vLLM 做 rollout 时，大概是：

```
prompt
↓
自回归生成 response token
↓
返回 response token ids
```

生成过程中，vLLM 确实会计算 logits，也可能返回生成时的 logprob。但它通常不负责保存训练计算图，不负责 `loss.backward()`，也不负责更新模型参数。

训练时真正参与 loss 和 backward 的 logprob，通常由训练 actor 在 FSDP / Megatron 等训练后端封装下计算，底层仍然调用 PyTorch 的张量计算和自动求导。

更准确地说：

- **old_log_prob**：rollout 样本生成后、actor 更新前，对固定 prompt + response 计算并缓存的旧策略概率。
- **new_log_prob**：进入更新阶段后，用当前 actor 参数对同一批固定 prompt + response 重新 forward 得到的当前策略概率。

所以 old_log_prob 是采样时旧策略的概率快照，new_log_prob 是当前正在更新的策略概率。二者比较的是"同一条 response 在旧策略和新策略下分别有多大概率"，**不是让新模型重新生成一条回答**。

### 11.4 π_rollout、π_old、π_θ 是三个模型吗？

更准确地说，它们不是"三个完全不同的模型"，而是**同一个 actor / policy 模型在 RL 流程里的不同策略身份**：

- **π_rollout**：rollout 策略，负责生成回答，通常由 vLLM / SGLang 执行
- **π_old**：旧策略，也就是采样后、更新前的策略快照，用来计算 old_log_prob
- **π_θ**：当前训练策略，也就是正在被更新的参数版本，用来计算 new_log_prob

可以把它理解成：

```
同一个 actor 模型
├── 同步到 vLLM / SGLang 里，用来生成样本：π_rollout
├── 采样时对应的旧版本，用来算 old_log_prob：π_old
└── 当前正在训练的版本，用来算 new_log_prob 并更新：π_θ
```

所以这三个符号更像"角色 / 状态快照 / 执行形态"，不一定代表三份彼此独立的模型。

### 11.5 old_log_prob 和 new_log_prob 到底怎么算？

注意，算 `new_log_prob` **不是让新模型重新生成一条回答**。

RL 后训练里通常先固定一条已经生成好的 response。`old_log_prob` 在 actor 更新前计算并缓存，表示采样时旧策略怎么看这条 response；`new_log_prob` 在更新阶段用当前 actor 参数重新 forward，表示当前策略怎么看同一条 response。

假设：

```
prompt tokens   = [x1, x2]
response tokens = [y1, y2, y3]
input_ids       = [x1, x2, y1, y2, y3]
```

语言模型是 next-token prediction：

- x2 位置的 logits 预测 y1
- y1 位置的 logits 预测 y2
- y2 位置的 logits 预测 y3

训练 actor 会对固定的 `[x1, x2, y1, y2, y3]` 做 forward，然后对真实 response token 取 logprob：

```
old_log_prob(y1), old_log_prob(y2), old_log_prob(y3)
new_log_prob(y1), new_log_prob(y2), new_log_prob(y3)
```

这本质上是 **teacher forcing 下的 logprob 计算**：固定答案，不重新采样，只计算模型认为这些 token 有多大概率。

### 11.6 GSPO / PPO / GRPO 里这些 logprob 怎么用？

不同算法细节不一样，但入门可以先抓住一个共同点：

- 先比较同一条 response 在 current policy 和 old policy 下的 logprob 差异；
- 再结合 reward / advantage；
- 最后形成 loss，反向传播更新 actor。

PPO / GRPO 里常会看到 token-level 的 ratio；GSPO 更强调 sequence-level 的 ratio。先不用急着背公式，只要理解：

- vLLM 负责采样；
- actor 负责评估样本概率；
- reward 负责提供优化方向；
- verl 负责把概率、reward、advantage 组合成 loss；
- FSDP / Megatron 封装训练过程，底层借助 PyTorch 把 loss 反传回模型参数。

最后把 verl 放回整篇文章的工具链里：

- verl 用 Transformers 管模型加载；
- 用 PyTorch 做张量计算和自动求导；
- 用 FSDP / Megatron 封装分布式训练执行；
- 用 vLLM / SGLang 做高吞吐 rollout；
- 用 Ray / 多进程做 worker 调度；
- 自己负责 RL 后训练的算法流程和数据流。

所以理解 verl 时，不要问"verl 替代了 PyTorch / vLLM / Ray 吗"。更准确的问题是：

> verl 把哪些底层能力组织成了一个完整的 RL 后训练系统？

这就是工具链地图的价值：一个复杂训练框架不是凭空替代所有东西，而是把不同层级的工具组合成系统。

---

## 十二、vLLM / SGLang：推理和 rollout 引擎

Transformers 可以用：

```python
model.generate()
```

但大规模推理和 RL 后训练 rollout 需要高并发、高吞吐、长时间批量生成。普通 `generate()` 往往不够快。

vLLM 适合这样理解：

- 高吞吐大模型推理引擎
- KV Cache 管理
- continuous batching
- PagedAttention

SGLang 也是高性能推理和 serving 框架，更强调：

- 低延迟
- 高吞吐
- 复杂生成流程
- 结构化输出
- 多轮推理
- 大规模 serving

一句话：

> vLLM / SGLang 不是训练框架，而是高性能生成引擎。它们既能服务线上推理，也能在 RL 后训练里加速 rollout。

---

## 十三、不要画成线性流程，要画成运行栈

很多文章喜欢把工具链画成：

```
数据 → tokenizer → 模型 → PyTorch → CUDA → 分布式训练
```

这个图容易误导。CUDA、NCCL、PCIe、NVLink、IB、PyTorch 不是"数据流到模型之后的下一步"，它们是底层运行支撑。

更准确的是把"被训练 / 推理的对象"和"运行它的工具链"分开看。

先看对象：

```
被训练 / 推理的对象：
  - Qwen / Llama / DeepSeek / GLM 等模型系列
  - 具体 checkpoint、tokenizer、config、weights
```

再看运行与训练工具链：

```
用户任务：SFT / DPO / PPO / GRPO / 推理 / 评测
        ↓
训练工具生态：
  - Trainer / PEFT / TRL
  - LLaMA-Factory / SWIFT
  - OpenRLHF / verl
        ↓
任务调度与启动：
  - Ray：复杂多机多角色调度
  - Accelerate：Hugging Face / PyTorch 训练脚本启动适配
        ↓
模型接口：Transformers / modelscope
        ↓
训练并行与显存优化：FSDP / DeepSpeed / Megatron
        ↓
底层框架：PyTorch / TensorFlow
        ↓
GPU 计算与通信库：CUDA / NCCL
        ↓
硬件互联与网络：PCIe / NVLink / InfiniBand
```

这张图的重点不是说每个项目都必须用全套工具，而是让你知道每个名字站在哪一层。

---

## 十四、常见 FAQ

### Q1：PyTorch 和 Transformers 是什么关系？

PyTorch 是底层计算框架。Transformers 是模型结构实现和统一接口库。Transformers 里的很多模型运行在 PyTorch 之上。

- **PyTorch**：负责算。
- **Transformers**：负责把模型结构、权重加载、tokenizer、训练和推理接口组织好。

### Q2：FSDP、DeepSpeed、Megatron 是什么关系？

三者都和大模型分布式训练有关，但侧重点不同。

- **FSDP**：PyTorch 原生分片方案，先理解成"省显存"。
- **DeepSpeed**：综合训练优化框架，ZeRO 最常见，但不只 ZeRO。
- **Megatron**：大模型并行训练框架，重点是 TP / PP / SP / CP / EP。

DeepSpeed ZeRO-3 和 FSDP 在目标上接近：都希望减少每张 GPU 上重复存储的模型状态。但 DeepSpeed 还覆盖 PP、TP、SP、EP、offload 等能力；Megatron 则更强调大模型并行训练。

### Q3：Accelerate、Ray、DeepSpeed 是什么关系？

- **Accelerate**：训练脚本启动和分布式适配。
- **Ray**：多机多角色任务调度。
- **DeepSpeed**：训练优化后端，负责 ZeRO、offload、并行训练等。

它们不互相替代。

### Q4：LLaMA-Factory 和 SWIFT 是什么关系？

二者都可以理解成高层微调入口，用来降低大模型训练和微调门槛。

- **LLaMA-Factory**：更常见于开源 LLM 微调场景。
- **SWIFT**：更偏 ModelScope / Qwen 生态，也覆盖多模态、评测、部署。

### Q5：TRL、OpenRLHF、verl 怎么区分？

- **TRL**：Hugging Face 生态里的后训练库。
- **OpenRLHF**：偏 Ray + vLLM + DeepSpeed 的 RLHF 工程框架。
- **verl**：偏 FSDP / Megatron + vLLM / SGLang 的大规模 RL post-training 总控框架。

### Q6：vLLM / SGLang 和 Transformers 是什么关系？

Transformers 能训练也能推理，适合通用模型接口和实验调试。vLLM / SGLang 是专门优化过的高性能推理 / serving / rollout 引擎，适合高吞吐生成。

---

## 十五、最后总结

学习大模型训练工具链，最重要的不是把所有名字背下来，而是建立一个判断框架：

> 先问它在哪一层，再问它解决什么问题，最后看它和谁配合。

如果只看名字，PyTorch、Transformers、DeepSpeed、Ray、verl、vLLM 好像都在"训练大模型"。但拆开以后会发现，它们负责的是完全不同的事情。可以把全文压缩成这样一张图：

- **CUDA / NCCL** 负责 GPU 计算和通信；**PCIe / NVLink / IB** 是硬件互联和网络。
- **PyTorch / TensorFlow** 是底层深度学习框架。
- **Transformers** 是模型结构和接口库；**modelscope** 是 ModelScope 生态 SDK。
- **Qwen / Llama / DeepSeek** 是模型系列，真正训练部署的是具体 checkpoint。
- **FSDP / DeepSpeed / Megatron** 解决分布式训练、显存优化和模型并行。
- **Accelerate** 是训练脚本启动适配器；**Ray** 是分布式任务调度器。
- **Trainer / TRL** 是训练器或后训练库。
- **PEFT** 是参数高效微调库，常用于 LoRA / QLoRA。
- **LLaMA-Factory / SWIFT** 是高层微调入口。
- **OpenRLHF / verl** 是 RLHF / RL 后训练工程框架。
- **vLLM / SGLang** 是高吞吐推理和 rollout 引擎。

这背后的核心逻辑是：

1. **底层硬件和通信**：让 GPU 算得动、卡和卡之间传得快。
2. **深度学习框架**：负责张量计算、自动求导、模型训练。
3. **模型接口层**：负责加载 tokenizer、config、模型结构和权重。
4. **分布式训练层**：负责多卡、多机、显存切分和大模型并行。
5. **高层训练工具**：负责把数据、模型、训练参数和训练流程组织起来。
6. **RL 后训练框架**：负责 rollout、reward、advantage、loss、update 等复杂流程。
7. **推理 / rollout 引擎**：负责高吞吐生成 token。

所以这些工具大多数不是互相替代关系，而是**上下游组合关系**。

比如做一个分布式 LoRA / SFT 微调任务时，常见组合可以这样理解：

```
LLaMA-Factory / SWIFT / Transformers Trainer
↓
Transformers 加载模型结构、tokenizer、config 和 checkpoint
↓
PEFT 挂载 LoRA / QLoRA 等参数高效微调模块
↓
DDP / FSDP / DeepSpeed 按各自方式封装模型、参数、梯度同步、optimizer 或训练引擎
↓
训练过程在封装后的模型上执行 forward、loss、backward、optimizer step
↓
底层仍然调用 PyTorch 的张量计算和 autograd
↓
CUDA 负责 GPU 计算；NCCL 负责多 GPU / 多机通信
↓
GPU / PCIe / NVLink / IB 提供底层硬件和互联能力
```

这里要注意三点：

- **第一**，PEFT 主要出现在 LoRA / QLoRA 这类参数高效微调里；如果是全参数微调，就不一定需要 PEFT。
- **第二**，DDP / FSDP / DeepSpeed 不是简单地排在 PyTorch 后面，而是会介入训练执行过程：它们会按各自方式封装模型、参数、梯度同步、optimizer 或训练引擎，再在训练时调用 PyTorch 的计算和自动求导能力。
- **第三**，DDP、FSDP、DeepSpeed 的侧重点不一样：DDP 主要做基础数据并行和梯度同步；FSDP 通过分片参数、梯度和优化器状态减少显存冗余；DeepSpeed 则通过 ZeRO、offload、并行策略等能力做更综合的训练优化。

如果是 RL 后训练，组合会更复杂：

```
verl / OpenRLHF
↓
Transformers 管模型接口
↓
FSDP / Megatron 封装 actor / critic 等模型的分布式训练；reference 通常冻结，只参与前向和 logprob 参考
↓
vLLM / SGLang 负责 rollout 生成
↓
Ray 或多进程负责 worker 调度
↓
底层仍然调用 PyTorch / CUDA / NCCL 完成计算和通信
```

再看 LLaMA-Factory 里那些分布式启动命令，也就没那么吓人了。它们不是"很多套完全不同的训练框架"，而是同一个训练任务可以通过不同启动方式接入不同后端：

- **torchrun**：偏 PyTorch 原生多进程启动。
- **accelerate launch**：偏 Hugging Face 生态的统一启动和适配。
- **deepspeed**：偏 DeepSpeed 后端启动。
- **ray job submit**：偏多机任务调度和 worker 管理。

同理，理解 verl 也不要问"verl 是不是替代 PyTorch / vLLM / Ray"。更准确的理解是：

- verl 负责 RL 后训练流程编排；
- PyTorch 负责计算和求导；
- FSDP / Megatron 封装分布式训练执行；
- vLLM / SGLang 负责高吞吐生成；
- Ray 负责调度 worker；
- CUDA / NCCL 负责底层 GPU 计算和通信。

真正重要的问题不是"哪个工具更高级"，而是：

1. 它在哪一层？
2. 它解决什么问题？
3. 它和上下游工具怎么配合？

只要这张地图建立起来，再看复杂训练脚本、LLaMA-Factory 分布式命令、verl 配置、DeepSpeed ZeRO 配置、vLLM rollout 配置，就不会被一堆名词淹没。

最后可以用一句话收束：

> 大模型训练不是某一个框架单独完成的，而是一整套分层工具链协作完成的。看懂层级关系，就看懂了这些工具为什么会同时出现。
