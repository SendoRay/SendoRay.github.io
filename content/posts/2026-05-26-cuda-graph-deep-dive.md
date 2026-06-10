---
title: CUDA Graph 深度解析：原理、机制与 vLLM/SGLang 中的工程实践
date: '2026-05-26'
tags:
- Inference
- GPU

draft: false
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---


## 一、背景：为什么需要 CUDA Graph

GPU 越来越快，单个 kernel 的执行时间已经降到微秒级别。但 **每次从 CPU 提交一个 GPU 操作（kernel launch、memcpy 等）本身也有微秒级的 overhead**：

- CPU 调用 CUDA 驱动接口
- 驱动验证参数、编码命令
- 将命令写入 command queue
- GPU 轮询 queue、解码命令、启动执行

当你的推理循环每次迭代都要 launch 几十上百个 kernel（QKV projection、attention、FFN、layernorm……），这些 launch overhead 累积起来，可能占总耗时的 30%~70%。

| 场景 | 耗时来源 |
|---|---|
| kernel 实际执行 | 2.9 μs（示例） |
| 单次 launch overhead | ~3-6 μs |
| 20 个 kernel 逐个同步 launch | 9.6 μs/kernel（overhead 占 70%） |
| 20 个 kernel 流水 launch | 3.8 μs/kernel（仍有 overhead） |
| CUDA Graph | 3.4 μs/kernel（接近理论极限） |

---

## 二、CUDA Graph 核心原理

CUDA Graph 的思路很简单：**把反复执行的 kernel 序列"录制"成一张图，后续每次只需一次 CPU 调用就可以回放整张图**。

```
传统方式（每次迭代）:
  CPU: launch(k1) → launch(k2) → ... → launch(kN)   ← N 次 CPU 调用
  GPU: [k1]  [k2]  ...  [kN]

CUDA Graph 方式:
  第一次: capture(k1, k2, ..., kN) → instantiate → graph
  后续每次: GraphLaunch(graph)                        ← 1 次 CPU 调用
  GPU:     [k1][k2]...[kN]
```

### 关键 API 三件套

```
cudaGraph_t     graph;      // 图结构（节点 + 依赖边）
cudaGraphExec_t instance;   // 可执行图（instantiate 后的产物）

// 录制
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  ... 正常写 kernel launch 代码 ...
cudaStreamEndCapture(stream, &graph);

// 实例化（将图编译为可执行形式，只做一次）
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// 每次迭代只需这一行
cudaGraphLaunch(instance, stream);
```

### 两种建图方式

**Stream Capture（最常用）**：在 BeginCapture/EndCapture 之间正常写代码，CUDA 自动录制，零代码侵入性。

**Explicit API**：手动调用 `cudaGraphAddKernelNode` 添加节点、`cudaGraphAddDependencies` 指定依赖关系，适合动态构建复杂拓扑。

---

## 三、完整代码示例

### 示例 1：Stream Capture（C++）

```cpp
// cuda_graph_demo.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define N       500000
#define NSTEP   1000
#define NKERNEL 20

__global__ void shortKernel(float* out, float* in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = 1.23f * in[idx];
}

// ===== 方案一：传统逐个 launch =====
void runWithoutGraph(float* out, float* in, cudaStream_t stream) {
    int threads = 512, blocks = (N + 511) / 512;
    for (int step = 0; step < NSTEP; step++) {
        for (int k = 0; k < NKERNEL; k++)
            shortKernel<<<blocks, threads, 0, stream>>>(out, in);
        cudaStreamSynchronize(stream);
    }
}

// ===== 方案二：CUDA Graph =====
void runWithGraph(float* out, float* in, cudaStream_t stream) {
    int threads = 512, blocks = (N + 511) / 512;

    cudaGraph_t     graph;
    cudaGraphExec_t graphExec;
    bool graphCreated = false;

    for (int step = 0; step < NSTEP; step++) {
        if (!graphCreated) {
            // 步骤1：录制（Stream Capture）
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for (int k = 0; k < NKERNEL; k++)
                shortKernel<<<blocks, threads, 0, stream>>>(out, in);
            cudaStreamEndCapture(stream, &graph);

            // 步骤2：实例化（只做一次）
            cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
            cudaGraphDestroy(graph);   // graph 定义已不需要
            graphCreated = true;
        }

        // 步骤3：每次迭代只需一次 GraphLaunch
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
    }
    cudaGraphExecDestroy(graphExec);
}

int main() {
    float *in_d, *out_d;
    cudaMalloc(&in_d,  N * sizeof(float));
    cudaMalloc(&out_d, N * sizeof(float));
    cudaMemset(in_d, 1, N * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 计时：无 Graph
    cudaEventRecord(start, stream);
    runWithoutGraph(out_d, in_d, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms1; cudaEventElapsedTime(&ms1, start, stop);
    printf("Without Graph: %.2f ms (%.2f us/kernel)\n",
           ms1, ms1 * 1000.0f / (NSTEP * NKERNEL));

    // 计时：有 Graph
    cudaEventRecord(start, stream);
    runWithGraph(out_d, in_d, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms2; cudaEventElapsedTime(&ms2, start, stop);
    printf("With    Graph: %.2f ms (%.2f us/kernel)\n",
           ms2, ms2 * 1000.0f / (NSTEP * NKERNEL));

    printf("Speedup: %.2fx\n", ms1 / ms2);

    cudaFree(in_d); cudaFree(out_d);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
```

编译运行：
```bash
nvcc -O2 -o demo cuda_graph_demo.cu && ./demo
# Without Graph: 192.00 ms (9.60 us/kernel)
# With    Graph:  68.00 ms (3.40 us/kernel)
# Speedup: 2.82x
```

### 示例 2：Explicit API（手动构建 A→B→C 依赖图）

```cpp
__global__ void kernelA(float* x) { *x = 1.0f; }
__global__ void kernelB(float* x) { *x *= 2.0f; }
__global__ void kernelC(float* x) { *x += 10.0f; }

int main() {
    float* d; cudaMalloc(&d, sizeof(float));
    cudaStream_t stream; cudaStreamCreate(&stream);

    cudaGraph_t graph; cudaGraphCreate(&graph, 0);

    void* argsA[] = {&d};
    void* argsB[] = {&d};
    void* argsC[] = {&d};

    cudaKernelNodeParams p = {};
    cudaGraphNode_t nodeA, nodeB, nodeC;

    p.func = (void*)kernelA; p.gridDim = {1,1,1}; p.blockDim = {1,1,1}; p.kernelParams = argsA;
    cudaGraphAddKernelNode(&nodeA, graph, NULL,   0, &p);   // 无前置依赖

    p.func = (void*)kernelB; p.kernelParams = argsB;
    cudaGraphAddKernelNode(&nodeB, graph, &nodeA, 1, &p);   // 依赖 A

    p.func = (void*)kernelC; p.kernelParams = argsC;
    cudaGraphAddKernelNode(&nodeC, graph, &nodeB, 1, &p);   // 依赖 B

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);

    float result; cudaMemcpy(&result, d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result = %.1f (expected 12.0)\n", result);  // (1.0*2.0)+10.0 = 12.0
    return 0;
}
```

### 示例 3：PyTorch API

```python
import torch

model = torch.nn.Linear(1024, 1024).cuda()
x = torch.randn(1024, 1024, device='cuda')

# 步骤1：Warmup（让 CUDA 完成内存分配、初始化）
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):
        y = model(x)
torch.cuda.current_stream().wait_stream(s)

# 步骤2：录制
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_y = model(x)  # x 的地址被固定到 graph 中

# 步骤3：反复 replay
for i in range(1000):
    x.copy_(torch.randn_like(x))   # 修改数据（地址不变，graph 感知不到变化）
    g.replay()                      # 一次 CPU 调用，GPU 执行整张图
    # static_y 自动包含最新结果
```

> **关键约束**：Graph 录制后，输入 tensor 的**内存地址**被固定。只能修改数据内容（`.copy_()`），不能换新 tensor。这是 LLM 框架设计 "static buffer" 的根本原因。

---

## 四、CUDA Graph 在 LLM 推理中为什么特别有用

LLM 推理有两个阶段：

- **Prefill（预填充）**：处理用户输入，batch size 和 token 数量每次不同，计算量大，kernel 数量少 → **不适合 Graph**（拓扑变化频繁）
- **Decode（自回归生成）**：每步只生成 1 个 token，每次 batch size 相对固定，kernel 序列完全相同，但每 step 都要 launch 几十个 kernel → **极其适合 Graph**（高频重复 + 短 kernel）

Decode 阶段的典型 kernel 序列（以 LLaMA 为例，每个 step 约 40+ 个 kernel）：

```
[RMSNorm] → [QKV Proj] → [RoPE] → [Attention] → [O Proj]
          → [RMSNorm] → [Gate Proj] → [Up Proj] → [SiLU] → [Down Proj]
          → ... × N_layers
          → [Logits] → [Sampling]
```

每个 step 都 launch 这一套，Graph 一次性录制，后续 replay 即可。

---

## 五、vLLM 中的 CUDA Graph 实现

### 5.1 整体架构

vLLM v1 的 CUDA Graph 实现分为两种模式：

- **FULL 模式**：对整个模型 forward 录制一张大 graph，replay 时一次提交全部 kernel
- **PIECEWISE 模式**：对 Attention 之外的部分（QKV、FFN 等）录制 sub-graph，Attention 保持 eager 执行（因为 FlashAttention 本身通过 CUDA event 就能高效调度）

核心文件：`vllm/v1/worker/gpu/cudagraph_utils.py`（`CudaGraphManager`、`ModelCudaGraphManager`）

### 5.2 Capture Sizes（录制哪些 batch size）

vLLM 不会对每个可能的 batch size 都录制，而是按照固定的 size 列表录制，推理时 **向上对齐（padding）** 到最近的录制尺寸：

```python
# vllm/config/compilation.py
# 默认 capture sizes 生成规则（伪代码）：
sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_size + 1, 16))
# 例：[1, 2, 4, 8, 16, 24, 32, ..., 248, 256, 272, 288, ..., 512]
```

`max_cudagraph_capture_size` 默认为 `min(max_num_seqs * 2, 512)`，防止 OOM 和过长启动时间。

你也可以自定义：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B \
    --cudagraph-capture-sizes 1 2 4 8 16 32 64 128
```

### 5.3 Capture 流程（源码解析）

```python
# vllm/v1/worker/gpu/cudagraph_utils.py: CudaGraphManager.capture()

@torch.inference_mode()
def capture(self, create_forward_fn, ...):
    with graph_capture(device=self.device):
        # 先录制 PIECEWISE，再录制 FULL（FULL 所需 buffer 更小，可复用 PW 的内存池）
        for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
            for desc in self._capture_descs[mode]:
                # 1. 准备 dummy 输入（固定地址的静态 buffer）
                forward_fn, attn_state = create_forward_fn(desc)

                # 2. Warmup 一次（不录制，让 CUDA 分配好所有内存）
                forward_fn(CUDAGraphMode.NONE)

                if desc.cg_mode == CUDAGraphMode.FULL:
                    # 3. 录制
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, self.pool):
                        forward_fn(CUDAGraphMode.NONE)
                    self.graphs[desc] = graph
```

Warmup 这一步至关重要：CUDA Graph 录制阶段禁止任何新的内存分配，所有 tensor 必须在 warmup 时就已经分配好。

### 5.4 Dispatch & Replay 流程

```python
# 推理时（decode step）
def dispatch(self, num_reqs, num_tokens, ...):
    """找到能覆盖当前 batch 的最小录制尺寸"""
    for desc in self._candidates[num_tokens]:
        if _is_compatible(desc, num_reqs, num_tokens, ...):
            return desc
    # 如果没有匹配的 graph，fallback 到 eager 模式

def run_fullgraph(self, desc):
    """直接 replay"""
    self.graphs[desc].replay()
    return self.hidden_states[:desc.num_tokens]
```

**Padding 机制**：若当前 decode batch 有 33 个请求，vLLM 会找到第一个 `>= 33` 的录制尺寸（比如 40），用 pad 过的 dummy 数据填充多出来的 7 个位置（通常是全零输入），只取前 33 个结果。

### 5.5 静态 Buffer 设计

Graph 录制时使用的所有 tensor 必须是**静态分配好的 buffer**，replay 时写入实际数据：

```python
# vllm/v1/worker/gpu/input_batch.py（概念性示意）
class InputBuffers:
    # 启动时一次性分配最大尺寸的 buffer
    input_ids = torch.zeros((max_tokens,), dtype=torch.int64, device='cuda')
    positions  = torch.zeros((max_tokens,), dtype=torch.int64, device='cuda')
    seq_lens   = torch.zeros((max_num_seqs,), dtype=torch.int32, device='cuda')
    # ...

# 每个 decode step 的流程：
# 1. 将当前 batch 数据 .copy_() 到对应 buffer（地址不变）
# 2. GraphLaunch → 读取 buffer → 计算 → 写入 output buffer
# 3. 从 output buffer 读取结果
```

---

## 六、SGLang 中的 CUDA Graph 实现

### 6.1 整体思路

SGLang 的 CUDA Graph 实现在 `python/sglang/srt/model_executor/cuda_graph_runner.py`，设计上与 vLLM 类似但有几处不同：

- 只对 **Decode 阶段**录制 Graph（Prefill 不录制）
- 使用 **batch size（请求数）** 作为 Graph key，而非 token 数
- 额外支持 **BreakableCUDAGraph**（可在图中暂停执行 eager 代码，用于调试）

### 6.2 自动计算 Capture Sizes

```python
# sglang/srt/server_args.py - set_ulysses_sequence_parallelism_and_verify() 自动计算
# cuda_graph_max_bs 根据 GPU 显存自动推断：

# 小显存 GPU（< 35GB HBM）:
#   chunked_prefill_size=2k → cuda_graph_max_bs=8
# 中显存 GPU（H100 80GB, TP=1）:
#   chunked_prefill_size=4k → cuda_graph_max_bs=32
# 大显存 GPU（H100, TP=8）:
#   chunked_prefill_size=8k → cuda_graph_max_bs=512
```

默认的 `cuda_graph_bs` 列表（`sglang/srt/server_args.py`）：
```
[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, ...]
```

capture 最大到 `cuda_graph_max_bs`，推理时 bisect 向上对齐。

### 6.3 Capture 流程（源码解析）

```python
# cuda_graph_runner.py: CudaGraphRunner.capture()

def capture(self) -> None:
    # 从大到小录制（大尺寸先录，小尺寸可复用大尺寸申请的内存池）
    for bs in reversed(self.capture_bs):
        graph, output_buffers = self.capture_one_batch_size(bs, forward)
        self.graphs[bs] = graph
        self.output_buffers[bs] = output_buffers

def capture_one_batch_size(self, bs: int, forward, ...):
    graph = torch.cuda.CUDAGraph()    # 或 BreakableCUDAGraph
    stream = self.stream

    # 准备 static buffer（从 DecodeInputBuffers 中截取 bs 大小的切片）
    input_ids     = buffers.input_ids[:bs]
    req_pool_indices = buffers.req_pool_indices[:bs]
    seq_lens      = buffers.seq_lens[:bs]
    out_cache_loc = buffers.out_cache_loc[:bs]
    # ...

    # Warmup（保证所有 CUDA 内存在录制前分配好）
    forward(input_ids, req_pool_indices, ...)
    torch.cuda.synchronize()

    # 正式录制
    with self._capture_graph(graph, pool, stream, lambda: forward(...)):
        output_buffers = forward(input_ids, req_pool_indices, ...)

    return graph, output_buffers
```

### 6.4 Replay 流程

```python
# cuda_graph_runner.py: CudaGraphRunner.replay()

def replay(self, forward_batch: ForwardBatch):
    raw_bs = forward_batch.batch_size

    # 找到最小的、>= raw_bs 的录制尺寸（bisect）
    index = bisect.bisect_left(self.capture_bs, raw_bs)
    bs = self.capture_bs[index]   # padding 后的 bs

    # 将真实 batch 数据 copy 到静态 buffer
    buffers.populate_from_forward_batch(
        forward_batch, raw_bs=raw_bs, bs=bs, ...
    )

    # 更新 attention backend 的 metadata（seq_lens 等）
    attn_backend.init_forward_metadata_replay_cuda_graph(
        bs, buffers.req_pool_indices[:bs], buffers.seq_lens[:bs], ...
    )

    # 一次 GraphLaunch 完成整个 decode step
    self.graphs[bs].replay()

    # 从 output_buffers 读取结果，只取前 raw_bs 个
    return self.output_buffers[bs][:raw_bs]
```

### 6.5 共享内存池（Graph Memory Pool）

多个不同尺寸的 Graph 共享同一个 `graph_memory_pool`，避免显存碎片化：

```python
# 全局内存池，所有 CUDAGraph 共享
global_graph_memory_pool = None

# capture 时
with torch.cuda.graph(graph, pool=global_graph_memory_pool):
    ...
```

这是 CUDA 10.0+ 的特性：多个 Graph 可以共享同一个 memory pool，大尺寸 Graph 分配的内存可以被小尺寸 Graph 复用，**极大降低 Graph 录制的总显存开销**。

---

## 七、两个框架实现对比

| 维度 | vLLM | SGLang |
|---|---|---|
| 核心文件 | `vllm/v1/worker/gpu/cudagraph_utils.py` | `sglang/srt/model_executor/cuda_graph_runner.py` |
| Graph 模式 | FULL + PIECEWISE | 主要 FULL（支持 BreakableCUDAGraph） |
| 录制 key | `(num_tokens, cg_mode, uniform_token_count)` | `batch_size (int)` |
| Capture sizes 来源 | `[1,2,4]+range(8,256,8)+range(256,max,16)` | 自动根据 GPU 显存推断，默认 `[1,2,...,max_bs]` |
| Padding 策略 | token 数向上对齐 | 请求数（batch size）向上对齐 |
| 内存池共享 | `current_platform.get_global_graph_pool()` | `global_graph_memory_pool` |
| Prefill 是否录制 | 否 | 否（默认） |
| 特殊能力 | PIECEWISE 支持中断 Attention 单独 eager | BreakableCUDAGraph 支持调试 |

---

## 八、约束与注意事项

### 8.1 Graph 录制阶段的限制（极重要）

在 `BeginCapture` 和 `EndCapture` 之间，以下操作**不能有**，否则会报错或结果错误：

| 禁止的操作 | 原因 |
|---|---|
| `cudaMalloc` / `torch.empty()` | 图录制期间不允许分配新内存 |
| CPU↔GPU 同步（`cudaDeviceSynchronize`） | 会中断录制流 |
| `print`、Python 控制流依赖 GPU 结果 | GPU 结果在 replay 时才可用 |
| 动态 shape 的 kernel | 图拓扑固定，shape 不能变 |
| `torch.unique(...).tolist()` | 隐含 CPU-GPU 同步 |

这也是为什么 vLLM/SGLang 要在录制前做完整的 **warmup**：所有 CUDA kernel 的内存分配必须在 warmup 时完成。

### 8.2 Padding 的开销

推理时若当前 batch 有 33 个请求，但录制尺寸为 40，则多出来 7 个 slot 需要用 dummy 数据填充（通常是全零 token_id），这 7 个位置的计算结果会被丢弃。计算量浪费为 \((40-33)/40 = 17.5\%\)。

这就是为什么 capture sizes 要设计得足够密集：尺寸间隔越小，浪费越少，但录制时间和显存开销越大。

### 8.3 显存开销

每录制一张 Graph，需要额外显存存储图的结构和静态 buffer。录制尺寸越大、数量越多，显存开销越大。这是 `cuda_graph_max_bs` 参数存在的原因。

SGLang 的注释中给出了一个估算公式：

$$
\text{reserved\_mem} \approx \text{chunked\_prefill\_size} \times 1.5 + \text{cuda\_graph\_max\_bs} \times 2 \quad (\text{单位: GB})
$$

### 8.4 何时 fallback 到 Eager 模式

- batch size 超过 `max_capture_bs`
- Prefill 阶段（每次 token 数量不同）
- 模型中有动态 shape 的算子
- 启动时传入 `--enforce-eager` 参数（强制禁用 Graph）

---

## 九、实战调参建议

### 针对 vLLM

```bash
# 标准配置（自动推断 capture sizes）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B \
    --max-num-seqs 256

# 自定义 capture sizes（减少启动时间）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B \
    --cudagraph-capture-sizes 1 2 4 8 16 32 64 128 256

# 禁用（调试时用）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B \
    --enforce-eager
```

### 针对 SGLang

```bash
# 标准配置（自动推断）
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --port 30000

# 手动指定最大 batch size
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --cuda-graph-max-bs 160

# 完全禁用 Graph
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --disable-cuda-graph
```

---

## 十、总结

CUDA Graph 是 LLM 推理系统在 decode 阶段消除 CPU-GPU launch overhead 的核心技术。

其本质是一次性录制 GPU 操作序列，后续每步推理通过单次 `GraphLaunch` 完成，将原本 N 次 CPU 调用（N 为 kernel 数量）压缩为 1 次，在 decode 阶段能带来 **2-5x 的 launch overhead 降低**，对整体吞吐量的提升通常为 **10%-30%**（具体取决于模型大小和 batch size）。

vLLM 和 SGLang 的工程实现都围绕以下三个核心设计：

1. **静态 Buffer**：启动时分配最大尺寸的 buffer，Graph 的内存地址永不改变
2. **Capture Size 分档**：只录制有限个 batch size，推理时向上 padding 对齐
3. **共享内存池**：多张 Graph 共享 pool，大尺寸 Graph 的内存可被小尺寸复用







ques;
可以在训练里面用吗