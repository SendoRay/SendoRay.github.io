---
title: "CUDA 入门：从第一个 Kernel 到 PyTorch 自定义算子"
date: '2026-08-14'
tags:
- CUDA
- GPU
- PyTorch

draft: false
math: true
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

> **出处声明**：本文编译自 tinkerd.net 的 [Writing CUDA Kernels for PyTorch](https://tinkerd.net/blog/machine-learning/cuda-basics/)（原文发表于 tinkerd.net，站点作者未具名，2024-04-20）。文中所有示意图均为按原图逻辑用 ASCII 字符**重绘**，代码示例源自原文（属标准 CUDA / PyTorch API 的教学用法），版权归原作者所有。本文以自己的中文表述改写组织，不构成对原文的逐句翻译。

如果你每天都在用 PyTorch 训练模型，却从来没有亲手写过一行 CUDA 代码，这篇文章就是为你准备的。它不要求任何 GPU 编程经验，只要求你会写基本的 C++ 和 Python。读完之后你将能够回答三个问题：

1. **一个 CUDA Kernel 到底是什么**——线程（Thread）、线程块（Block）、Warp、SM 这些词分别指什么，它们如何映射到硬件上？
2. **GPU 的内存层级长什么样**——为什么同样的算法，用不用 Shared Memory 能差出 7 倍性能？
3. **怎么把自己写的 Kernel 接进 PyTorch**——`torch.utils.cpp_extension` 和 pybind11 是如何把一个 `.cu` 文件变成可以 `import` 的 Python 模块的？

全文结构鸟瞰如下：

```text
┌────────────────────────────────────────────────────────────────────┐
│           本文地图（CUDA 入门 → PyTorch 自定义算子）                 │
│                                                                    │
│  第一部分 CUDA 基础                                                 │
│  ├── 一、为什么并行        CPU 顺序执行 vs GPU 数千线程              │
│  ├── 二、第一个 Kernel     __global__ / threadIdx / <<<>>>          │
│  ├── 三、读写数据          cudaMalloc / cudaMemcpy / cudaFree       │
│  ├── 四、Thread Block      1D/2D/3D 线程块、1024 线程上限           │
│  └── 五、SM 与 Warp        Block→SM 分配、32 线程编组、调度器        │
│                          ▼                                         │
│  第二部分 CUDA 内存层级                                             │
│  ├── 六、三类逻辑内存      Global / Local & Register / Shared       │
│  ├── 七、逐层拆解          sector、L1/L2、寄存器、__shared__        │
│  └── 八、Softmax 实战      __syncthreads() + Shared Memory ≈ 7.5×  │
│                          ▼                                         │
│  第三部分 接入 PyTorch                                              │
│  ├── 九、pybind11 桥接     kernel(.cu) + binding(.cpp) + setup.py   │
│  ├── 十、操作 Tensor       data_ptr / AT_DISPATCH 多 dtype          │
│  └── 十一、速查小结与结语                                            │
└────────────────────────────────────────────────────────────────────┘
```

---

## 一、为什么需要 CUDA：从顺序到并行

CUDA（Compute Unified Device Architecture）是 NVIDIA 提供的 GPU 编程平台：它扩展了 C/C++ 语言，让你可以写出在 GPU 上运行的函数。

要理解它存在的意义，先看一个最朴素的任务：给一个数组里的每个元素加 1。CPU 的做法是一个循环，**一个接一个**地处理：

```cpp
// CPU：顺序执行，处理 N 个元素需要 N 轮循环
for (int i = 0; i < N; i++) {
    data[i] = data[i] + 1;
}
```

CPU 的核心数通常是个位数到几十，单核很强但数量有限，这种"少量强核"的架构适合分支复杂的串行逻辑。而 GPU 走的是另一条路：**成千上万个弱核同时干活**。同样的任务，GPU 的思路是启动 N 个线程，第 $i$ 个线程只负责 `data[i] = data[i] + 1` 这一件事——所有元素**同时**被处理。

深度学习的核心计算（矩阵乘法、逐元素运算、归约）恰好都是这种"对海量数据做同构操作"的形态，这就是 GPU 成为深度学习标配硬件的根本原因。而 CUDA，就是你指挥这数千个线程的语言。

## 二、第一个 Kernel：点名

### 2.1 Kernel 定义的是"单个线程"的行为

在 CUDA 术语里，**Kernel（核函数）是一个由 CPU 发起、在 GPU 上执行的函数**，用 `__global__` 修饰符标记。理解 CUDA 最关键的一个思维转变是：

> **Kernel 代码描述的不是整个任务，而是单个线程要做的事。** 你写一份代码，GPU 把它复制给成千上万个线程同时跑，每个线程用自己的索引区分"我该处理哪块数据"。

看第一个完整例子——让每个线程"报数"：

```cuda
#include <stdio.h>

// __global__ 表示：这是一个 kernel，由 CPU 调用、在 GPU 上执行
__global__ void roll_call() {
    // threadIdx.x 是当前线程在 block 内的索引
    const int threadIndex = threadIdx.x;
    printf("Thread %d here!\n", threadIndex);
}

int main() {
    // <<<numBlocks, numThreads>>>：1 个 block，每个 block 10 个线程
    roll_call<<<1, 10>>>();

    // kernel 是异步执行的，CPU 必须显式等待 GPU 完成
    cudaDeviceSynchronize();

    return 0;
}
```

逐行拆解几个新面孔：

- **`__global__`**：告诉编译器这个函数是 kernel——从 CPU（Host，主机）侧发起，在 GPU（Device，设备）侧执行。
- **`threadIdx.x`**：CUDA 内置变量，取当前线程在所属 block 内的索引。10 个线程分别拿到 0~9。
- **`<<<numBlocks, numThreads>>>`**：kernel 的启动配置（launch configuration），三对尖括号是 CUDA 对 C++ 的语法扩展。这里表示启动 1 个 block、每个 block 含 10 个线程。
- **`cudaDeviceSynchronize()`**：kernel 启动是**异步**的——CPU 发出启动命令后立即继续往下跑，不等 GPU。如果 `main` 不等待就 `return`，程序可能在 GPU 打印之前就退出了。这个调用会阻塞 CPU，直到 GPU 上所有已提交的工作完成。

### 2.2 编译与运行

CUDA 源文件以 `.cu` 为后缀，用 NVIDIA 的编译器 `nvcc` 编译：

```bash
nvcc roll_call.cu -o roll_call.o
./roll_call.o
```

输出：

```text
Thread 0 here!
Thread 1 here!
...
Thread 9 here!
```

同一份 kernel 代码被 10 个线程各执行了一次，每个线程打印出自己的编号——这就是"一份代码、万人齐跑"的 SIMT（Single Instruction, Multiple Threads，单指令多线程）模型的最小演示。

## 三、让 Kernel 读写数据

### 3.1 Kernel 没有返回值

看 `roll_call` 的签名：返回类型是 `void`。这不是巧合——**所有 kernel 都不能有返回值**。想想也合理：几千个线程同时执行同一个函数，"返回值"该是谁的？

因此 kernel 与外界交换数据只有一种方式：**通过指针读写内存**——要么原地修改输入，要么把结果写进调用方预先分配好的输出缓冲区。

但这里有个坑：CPU 和 GPU 用的是两块物理上独立的内存（主机内存 RAM vs 设备显存 VRAM）。CPU 上 `malloc` 出来的指针，GPU 线程是无法解引用的。于是需要一套显存管理 API：

| API | 作用 | 类比 |
|---|---|---|
| `cudaMalloc(&ptr, size)` | 在 GPU 显存上分配 size 字节 | `malloc` |
| `cudaFree(ptr)` | 释放显存 | `free` |
| `cudaMemcpy(dst, src, size, direction)` | 在主机/设备间拷贝数据 | `memcpy` + 方向 |

`cudaMemcpy` 的第四个参数指明拷贝方向，最常用的两个枚举值：

- `cudaMemcpyHostToDevice`：主机 → 设备（把输入送上 GPU）
- `cudaMemcpyDeviceToHost`：设备 → 主机（把结果取回 CPU）

另外一个重要性质：**`cudaMemcpy` 对 host 是同步的**，它会等待**默认流**上先前提交的操作（包括 kernel）完成后再开始拷贝，所以"默认流上 kernel 之后紧跟 D2H 拷贝"的场景不需要额外的 `cudaDeviceSynchronize()`。但在多 stream 场景下，这条隐式顺序保证不覆盖其他流，需要用 `cudaStreamSynchronize` / `cudaDeviceSynchronize` 显式控制顺序。

### 3.2 端到端示例：数组加一

把上面所有 API 串起来，写一个完整的"数组每个元素加 1"程序：

```cuda
#include <stdio.h>

__global__ void array_increment(int* in) {
    const int threadIndex = threadIdx.x;
    // 每个线程只负责自己那一个元素，原地加 1
    in[threadIndex] = in[threadIndex] + 1;
}

int main() {
    const int arraySize = 10;
    const int arrayByteSize = arraySize * sizeof(int);

    // 1. 主机侧：分配并初始化数组 [0, 1, ..., 9]
    int* h_in = (int*)malloc(arrayByteSize);
    for (int i = 0; i < arraySize; i++) {
        h_in[i] = i;
    }
    printf("Before: ");
    for (int i = 0; i < arraySize; i++) printf("%d ", h_in[i]);
    printf("\n");

    // 2. 设备侧：分配显存
    int* d_in;
    cudaMalloc(&d_in, arrayByteSize);

    // 3. H2D：把输入拷贝上 GPU
    cudaMemcpy(d_in, h_in, arrayByteSize, cudaMemcpyHostToDevice);

    // 4. 启动 kernel：1 个 block、10 个线程，每个线程处理 1 个元素
    array_increment<<<1, arraySize>>>(d_in);

    // 5. D2H：取回结果（cudaMemcpy 会先隐式等 kernel 跑完）
    cudaMemcpy(h_in, d_in, arrayByteSize, cudaMemcpyDeviceToHost);

    printf("After:  ");
    for (int i = 0; i < arraySize; i++) printf("%d ", h_in[i]);
    printf("\n");

    // 6. 清理两侧内存
    free(h_in);
    cudaFree(d_in);
    return 0;
}
```

输出：

```text
Before: 0 1 2 3 4 5 6 7 8 9
After:  1 2 3 4 5 6 7 8 9 10
```

习惯上用 `h_` 前缀表示主机（host）指针、`d_` 前缀表示设备（device）指针，避免把只能在一侧解引用的指针传错地方。这个"**malloc → cudaMalloc → H2D → kernel → D2H → free/cudaFree**"六步曲，是几乎所有 CUDA 程序的骨架。

## 四、Thread Block：线程的组织方式

### 4.1 线程块可以是 1D、2D 或 3D 的

前面 `<<<1, 10>>>` 里的第二个参数其实不只是一个整数——它的完整类型是 `dim3`，一个含 x/y/z 三个维度的结构体。相应地，`threadIdx` 也有 `.x`、`.y`、`.z` 三个分量。也就是说，**一个 block 内的线程可以按一维、二维甚至三维网格来编号**：

```text
【图 1】Thread Block 的三种形态

  1D block（如 dim3(6)）           每个线程一个 threadIdx.x
  ┌───┬───┬───┬───┬───┬───┐
  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ ──▶ threadIdx.x = 0..5
  └───┴───┴───┴───┴───┴───┘

  2D block（如 dim3(4, 3)）        每个线程一对 (x, y)
        x=0   x=1   x=2   x=3
      ┌─────┬─────┬─────┬─────┐
  y=0 │(0,0)│(1,0)│(2,0)│(3,0)│
      ├─────┼─────┼─────┼─────┤
  y=1 │(0,1)│(1,1)│(2,1)│(3,1)│
      ├─────┼─────┼─────┼─────┤
  y=2 │(0,2)│(1,2)│(2,2)│(3,2)│
      └─────┴─────┴─────┴─────┘

  3D block（如 dim3(4, 3, 2)）     再叠一层 z 维
        z=1 ┌─────┬─────┬─────┬─────┐
       ┌────┴┬────┴┬────┴┬────┴┐    │
   z=0 │(0,0)│(1,0)│(2,0)│(3,0)│    │
       ├─────┼─────┼─────┼─────┤    │
       │(0,1)│(1,1)│(2,1)│(3,1)│    │
       ├─────┼─────┼─────┼─────┤────┘
       │(0,2)│(1,2)│(2,2)│(3,2)│
       └─────┴─────┴─────┴─────┘
```

需要强调：多维只是**编号上的便利**，不改变硬件执行方式。它存在的意义是让处理图像、矩阵这类天然多维的数据时，索引计算更直观。

与 `threadIdx` 配套的内置变量是 **`blockDim`**：它存放当前 block 的各维尺寸（如 `blockDim.x`、`blockDim.y`），供线程把多维坐标换算成一维内存偏移。

### 4.2 2D 示例：矩阵转置

用一个 2D block 做矩阵转置，能直观看到多维索引的好处。矩阵在内存中按行优先（row-major）一维排布，转置就是把 `(x, y)` 位置的元素搬到 `(y, x)`：

```cuda
__global__ void transpose_matrix(int* in, int* out) {
    // 输入中的一维偏移：行 y 中的第 x 个元素
    const int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;
    // 输出中的一维偏移：x/y 互换
    const int outIdx = threadIdx.y + threadIdx.x * blockDim.y;
    out[outIdx] = in[threadIndex];
}

// 启动方式：block 尺寸直接取矩阵形状，每个线程搬一个元素
// 注意 dim3 的 x 维对应列（num_cols）、y 维对应行（num_rows），
// 与上面索引公式里 blockDim.x = 每行列数 的假定一致
// dim3 numThreadsPerBlock(num_cols, num_rows);
// transpose_matrix<<<1, numThreadsPerBlock>>>(d_in, d_out);
```

每个线程按自己的 `(threadIdx.x, threadIdx.y)` 算出"从哪读、往哪写"，一次启动整块矩阵同时完成转置——没有任何循环。

### 4.3 多 Block 与 1024 线程上限

一个 block 能容纳的线程数是有硬上限的：**1024 个**（对 x/y/z 各维乘积生效）。试图启动 1025 个线程的 kernel，不会崩溃、也不会打印任何东西——它会**静默失败**。要抓住这类启动错误，需要 `cudaGetLastError()`：

```cuda
__global__ void roll_call() {
    printf("Thread %d here!\n", threadIdx.x);
}

int main() {
    roll_call<<<1, 1025>>>();   // 超过 1024，启动失败

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // 输出：CUDA Error: invalid configuration argument
    }

    cudaDeviceSynchronize();
    return 0;
}
```

那需要超过 1024 个线程怎么办？用启动配置的**第一个**参数：启动多个 block。`<<<4, 256>>>` 就是 4 个 block × 每块 256 线程 = 1024 个线程。全体 block 合称一个 **Grid（网格）**。block 数量没有 1024 这种小上限，可以轻松启动几万个 block——这正是 GPU 规模化并行的入口。

## 五、SM 与 Warp：线程在硬件上如何运行

到目前为止说的 thread/block/grid 都是**软件抽象**。这一节下沉到硬件，回答"这些线程到底在哪儿跑"。

### 5.1 SM：流式多处理器

一块 GPU 芯片由多个 **SM（Streaming Multiprocessor，流式多处理器）** 组成，例如 A100 有 108 个 SM。SM 才是真正执行线程的硬件单元，而 block 到 SM 的分配遵循一条铁律：

> **同一个 block 的所有线程必定运行在同一个 SM 上；不同的 block 可以被分派到不同的 SM。**

这条规则是后面 Shared Memory、`__syncthreads()` 一切"block 内协作"能力的硬件基础——只有物理上住在一起的线程，才能共享高速内存、低成本同步。

想验证这一点，可以用内联 PTX（CUDA 的汇编层）读出特殊寄存器 `%smid`——当前线程所在 SM 的编号：

```cuda
__global__ void sm_roll_call() {
    unsigned int smID;
    // 内联 PTX：把特殊寄存器 %smid 的值搬进变量 smID
    asm("mov.u32 %0, %smid;" : "=r"(smID));

    printf("Block %d, Thread %d, running on SM %d\n",
           blockIdx.x, threadIdx.x, smID);
}

int main() {
    sm_roll_call<<<4, 2>>>();   // 4 个 block、每块 2 线程
    cudaDeviceSynchronize();
    return 0;
}
```

典型输出（`blockIdx.x` 是 block 在 grid 中的索引）：

```text
Block 0, Thread 0, running on SM 0
Block 0, Thread 1, running on SM 0
Block 1, Thread 0, running on SM 2
Block 1, Thread 1, running on SM 2
Block 2, Thread 0, running on SM 4
Block 2, Thread 1, running on SM 4
Block 3, Thread 0, running on SM 6
Block 3, Thread 1, running on SM 6
```

同一 block 的两个线程总是报出同一个 SM 编号，而 4 个 block 被摊到了 4 个不同的 SM 上：

```text
【图 2】Block 到 SM 的分配

   软件侧（Grid）                     硬件侧（GPU）
  ┌──────────────────┐
  │ Block 0          │ ───────────▶ ┌──────────┐
  │  [T0] [T1]       │              │   SM 0   │
  ├──────────────────┤              ├──────────┤
  │ Block 1          │ ───────────▶ │   SM 2   │
  │  [T0] [T1]       │              ├──────────┤
  ├──────────────────┤              │   SM 4   │
  │ Block 2          │ ───────────▶ │          │
  │  [T0] [T1]       │              ├──────────┤
  ├──────────────────┤              │   SM 6   │
  │ Block 3          │ ───────────▶ │          │
  │  [T0] [T1]       │              └──────────┘
  └──────────────────┘
   规则：一个 Block 的全部线程 ── 必在同一个 SM
         不同 Block ──────────── 可在不同 SM
```

### 5.2 Warp：32 个线程一组

block 内的线程也不是被逐个调度的。SM 会把 block 里的线程**按 32 个一组**编成 **Warp（线程束）**，warp 才是 SM 的基本调度与执行单位——同一 warp 里的 32 个线程在同一拍里执行同一条指令（这正是 SIMT 名字的来源）。

每个线程在 warp 内的位置叫 **lane（车道）**，取值 0~31。如果 block 有 40 个线程，就会编成 2 个 warp：warp 0 满编 32 个，warp 1 只有 8 个（剩下 24 个 lane 空转浪费——所以 block 大小最好取 32 的倍数）。

warp 编号和 lane 编号同样能用 PTX 特殊寄存器 `%warpid`、`%laneid` 读出来：

```cuda
__global__ void warp_roll_call() {
    unsigned int smID, warpID, laneID;
    asm("mov.u32 %0, %smid;"   : "=r"(smID));
    asm("mov.u32 %0, %warpid;" : "=r"(warpID));
    asm("mov.u32 %0, %laneid;" : "=r"(laneID));

    printf("SM %d | Warp %d | Lane %2d | Thread %2d\n",
           smID, warpID, laneID, threadIdx.x);
}

int main() {
    warp_roll_call<<<1, 40>>>();   // 40 线程 = 1 个满编 warp + 1 个 8 人 warp
    cudaDeviceSynchronize();
    return 0;
}
```

输出会显示：线程 0~31 属于 warp 0（lane 0~31），线程 32~39 属于 warp 1（lane 0~7），且全部落在同一个 SM 上。

### 5.3 Warp Scheduler：SM 的发射引擎

warp 由谁来调度？每个 SM 内有若干个 **Warp Scheduler（线程束调度器）**——A100 上是 **4 个**。每个时钟周期，调度器从手头所有"就绪"的 warp 中挑一个，把它的下一条指令发射（issue）到执行单元的 issue slot 里。

关键在于"就绪"二字。一个 warp 随时可能**停摆（stall）**：比如它在等一次几百周期的全局内存读取，或者在 `__syncthreads()` 屏障前等待同伴。停摆的 warp 不可发射，调度器就换别的 warp 顶上——这正是 GPU 隐藏内存延迟的核心机制：**用海量 warp 之间的切换填满每一个周期**。

原文用了一个很形象的类比：把每个周期想象成流水线上经过的一只桶，调度器的工作是往每只桶里装一条指令。如果某个周期所有 warp 都在停摆、没有指令可装，**这只桶就空着流走了**——这就是算力浪费。性能优化的很大一部分工作，就是减少"空桶"。

把 SM 的内部结构画出来（按 A100 白皮书逻辑简化重绘）：

```text
【图 3】SM 内部结构简图（A100，逻辑示意）

  ┌─────────────────────────────────────────────────────────┐
  │                        SM (×108)                        │
  │  ┌────────────┬────────────┬────────────┬────────────┐  │
  │  │  Warp      │  Warp      │  Warp      │  Warp      │  │
  │  │ Scheduler 0│ Scheduler 1│ Scheduler 2│ Scheduler 3│  │
  │  │ Dispatch   │ Dispatch   │ Dispatch   │ Dispatch   │  │
  │  │  Unit      │  Unit      │  Unit      │  Unit      │  │
  │  └─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┘  │
  │        ▼            ▼            ▼            ▼         │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │            Register File（寄存器文件）           │    │
  │  └─────────────────────┬───────────────────────────┘    │
  │                        ▼                                │
  │  ┌──────────┬──────────┬──────────────┬────────────┐    │
  │  │  FP32    │  INT32   │ Tensor Core  │  LD/ST 等  │    │
  │  │ 计算单元 │ 计算单元  │              │            │    │
  │  └──────────┴──────────┴──────────────┴────────────┘    │
  │                        ▼                                │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │        L1 Data Cache / Shared Memory（同址）      │    │
  │  └─────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────┘
   每周期：4 个调度器各自从就绪 warp 中选指令 ──▶ 发射到计算单元
   warp 停摆（等内存 / 等同步）时 ──▶ 调度器切换到其他 warp
```

至此，软件抽象与硬件的映射关系齐了：**Thread 由 kernel 代码定义 → 32 个 Thread 编成 Warp 被调度 → 若干 Warp 组成 Block 落在一个 SM 上 → 全部 Block 组成 Grid 铺满整块 GPU**。

---

## 六、CUDA 内存层级：三类逻辑内存

第一部分解决了"计算在哪发生"，第二部分解决"数据在哪存放"。CUDA 向程序员暴露三类**逻辑**内存，按作用域从大到小排列：

1. **Global Memory（全局内存）**：所有线程可见，容量最大（就是显卡标称的那几十 GB），速度最慢。
2. **Shared Memory（共享内存）**：**同一 block 内**的线程共享，速度接近 L1 缓存。
3. **Local Memory 与 Registers（本地内存与寄存器）**：单线程私有；寄存器最快，local memory 名字里带"local"但物理上其实很慢（后面细说）。

```text
【图 4】CUDA 逻辑内存层级：作用域 vs 速度

   作用域大 ▲ 速度慢
            │
  ┌─────────┴──────────────────────────────────────┐
  │ Global Memory        所有 Block 的所有线程可见   │
  │ （cudaMalloc 分配，容量 = 显存，~数百周期延迟）    │
  └─────────┬──────────────────────────────────────┘
            │
  ┌─────────┴──────────────────────────────────────┐
  │ Shared Memory        仅本 Block 内线程可见       │
  │ （__shared__ 声明，物理上与 L1 同址，~几十周期）  │
  └─────────┬──────────────────────────────────────┘
            │
  ┌─────────┴──────────────────────────────────────┐
  │ Registers / Local    仅本线程可见                │
  │ （寄存器 ~1 周期最快；Local 实际落在显存、很慢）   │
  └────────────────────────────────────────────────┘
            │
   作用域小 ▼ 速度快（Local 除外，注意这个陷阱）
```

这个层级和第一部分的执行层级严格对应：**线程私有寄存器 ↔ Thread，Shared Memory ↔ Block（因为 block 独占一个 SM），Global Memory ↔ Grid**。

### 6.1 观测工具：Nsight Compute

后面所有关于内存行为的结论都不是拍脑袋，而是能用 NVIDIA 的 kernel 级 profiler **Nsight Compute** 实测出来的。命令行采集：

```bash
sudo ncu --set=full -f -o report ./your_binary.o
```

生成的 `report.ncu-rep` 用 Nsight Compute GUI 打开，重点看 **Memory Workload Analysis** 一节——它会把 kernel 对每一级内存发出的请求数、命中率、传输字节数全部列出来。

## 七、逐层拆解：Global、Register/Local、Shared

### 7.1 Global Memory：L1 → L2 → 显存三级支撑

`cudaMalloc` 分配的就是 global memory。逻辑上它是一整块"所有线程可见的大内存"，物理上则由三级硬件支撑：每个 SM 私有的 **L1 Cache** → 全 GPU 共享的 **L2 Cache** → 真正的 **Device Memory（显存颗粒）**。线程读 global memory 时逐级查找：L1 命中最快，L1 miss 则查 L2，L2 再 miss 才去显存取。

两个关键的硬件常数：

- 内存读取的基本粒度是 **32 字节的 sector（扇区）**——哪怕你只要 4 字节的一个 float，硬件也会搬完整的 32 字节。
- L2 从显存取数时**一次取 2 个相邻 sector（64 字节）**——一种朴素的预取：你要了这 32B，隔壁 32B 大概率马上也要。

用一个刻意写成**单线程**的点积 kernel 来验证（两个 `float[16]` 数组做点积）：

```cuda
__global__ void dot_product(float* a, float* b, float* c, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    *c = sum;
}
// 启动：dot_product<<<1, 1>>>(d_a, d_b, d_c, 16);
```

先算账：每个数组 16 个 float = 64 字节 = **2 个 sector**，两个数组共 128 字节 = **4 个 sector**。再看 Nsight Compute 报出的数字：

- **L1 收到 32 次读请求**：循环 16 轮 × 每轮读 a、b 各一次。
- **L1 向 L2 只发了 4 次 sector 读**：32 次请求里只有 4 次真正 miss——每个 sector 第一次被碰到时 miss 一次，之后同 sector 的读全部 L1 命中。
- **L2 从显存读了 4 个 sector、Hit Rate 33.33%**：L2 收到 L1 的 4 次 miss 请求；由于"一次取 2 个 sector"的预取，第 1 次请求实际搬回 2 个 sector，于是第 2 次请求直接命中，a、b 两个数组各贡献 1 miss + 1 hit。但 profiler 统计的分母还包括预取本身产生的访问，最终 6 次访问中 2 次命中 = 33.33%。

```text
【图 5】dot_product 的内存请求流（Nsight 数据流逻辑重绘）

  ┌────────────────┐
  │  dot_product   │  单线程循环 16 轮，每轮读 a[i]、b[i]
  │    Kernel      │
  └───────┬────────┘
          │ 32 次读请求（16×2）
          ▼
  ┌────────────────┐
  │    L1 Cache    │  28 次命中
  │   (SM 私有)    │
  └───────┬────────┘
          │ 4 次 sector 请求（4 次 miss）
          ▼
  ┌────────────────┐
  │    L2 Cache    │  Hit Rate 33.33%（2 hit / 6 访问）
  │   (全卡共享)   │
  └───────┬────────┘
          │ 读 4 sector = 128 字节（每次成对取 64B）
          ▼
  ┌────────────────┐
  │ Device Memory  │  a、b 两数组共 128B，恰好 4 sector
  └────────────────┘
```

数组与 sector 的对应关系：

```text
【图 6】float[16] 如何切成 sector

  一个 float = 4B；一个 sector = 32B = 8 个 float
  float[16] = 64B = 2 个 sector

  数组 a：
  ┌─── sector 0 (32B) ────────────┬─── sector 1 (32B) ────────────┐
  │ a[0] a[1] a[2] ... a[6] a[7] │ a[8] a[9] a[10] ... a[15]     │
  └───────────────────────────────┴───────────────────────────────┘
  读 a[0] 时：整个 sector 0 被搬进缓存 ──▶ a[1..7] 的后续读直接命中
```

这个小实验给出的直觉非常值钱：**访问模式的局部性直接决定缓存效率**。相邻线程读相邻地址（合并访存，coalesced access）时，一个 sector 服务一整组线程；跳着读则每次都触发新的 sector 搬运，带宽瞬间被浪费掉数倍。

### 7.2 Local Memory 与寄存器：快慢两重天

线程私有的数据放在哪，由编译器决定，且两种去向的性能天差地别：

- **标量局部变量 → 寄存器（Register）**：SM 上最快的存储，1 周期可达。A100 上每个线程最多可用 **255 个寄存器**。
- **需要动态索引的局部数组、或寄存器放不下的变量 → Local Memory**：名字带"local"仅指作用域（线程私有），**物理位置其实在 Device Memory**——和 global memory 一样慢。局部数组一旦用变量做下标，编译器通常只能把它放进 local memory。

怎么验证一个变量究竟去了哪？看 PTX。Nsight Compute 的 Source 页面可以并排显示 CUDA 源码与 PTX 汇编：

- 落在寄存器的操作直接以寄存器为操作数，例如 `float sum = 0;` 编译成 `mov.f32 %f4, 0f00000000;`——`%f4` 就是一个浮点寄存器，说明 `sum` 住在寄存器里。
- 落在 local memory 的变量会出现 `.local` 标记的指令（如 `ld.local` / `st.local`），一眼可辨。

实践建议由此而来：**警惕 kernel 里的局部数组**；若 profiling 发现大量 `.local` 访存，通常意味着寄存器溢出（register spilling）或动态索引数组，都是性能红灯。

### 7.3 Shared Memory：block 内的协作白板

第三类内存是 shared memory：**同一 block 内所有线程共享**，用 `__shared__` 修饰符声明：

```cuda
__shared__ float buffer[256];   // block 内所有线程看到同一份 buffer
```

物理上，shared memory 和 L1 cache 是**同一块片上 SRAM**——这解释了它为什么快（接近 L1 速度），也解释了为什么它是 block 作用域（block 独占 SM，而这块 SRAM 在 SM 上）。两者的配比还可以调，比如把更多空间划给 shared memory：

```cuda
// carveout：期望把这块 SRAM 的多少比例留给 shared memory（此处 20%）
cudaFuncSetAttribute(my_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout, 20);
```

shared memory 的典型用法是：**每个线程从 global memory 各搬一部分数据进来，大家在高速白板上协作计算**。但"协作"立刻引出一个新问题——怎么保证你写完了我再读？这就是下一章的主角。

## 八、线程同步与 Softmax 实战：7.5 倍加速

### 8.1 __syncthreads()：block 级屏障

`__syncthreads()` 是 block 级同步屏障（barrier）：**block 内所有线程都执行到这一行之前，谁也不许往下走**。它是"先写 shared memory、再读别人写的结果"这类协作模式的安全保障。注意它只同步 block 内部——跨 block 同步在 kernel 内没有廉价手段（这也呼应了硬件事实：不同 block 可能在不同 SM 上，根本没有共享的快速通道）。

### 8.2 案例：Softmax

用一个真实算子看 shared memory + 同步的威力。Softmax 是把一组分数归一化成概率分布的函数，数值稳定版会先减去最大值：

$$Softmax(x_i) = \frac{e^{x_i - x_{max}}}{\sum_j e^{x_j - x_{max}}}$$

计算分三步：① 全数组求最大值 $x_{max}$；② 全数组求 $\sum_j e^{x_j - x_{max}}$；③ 每个元素算 $e^{x_i - x_{max}}$ 除以总和。前两步都是**全局归约**——需要看到整个数组，这正是并行化的难点。

**Naive 版**：数组切给各线程，第 ③ 步各管一段；但 ①② 两步偷懒让**每个线程都自己全量扫一遍数组**：

```cuda
__global__ void softmax_kernel(float *input, float *output, int size) {
    int num_threads = blockDim.x;
    int num_elements_per_thread = size / num_threads;
    int thread_index = threadIdx.x;
    int start_idx = thread_index * num_elements_per_thread;
    int end_idx = min(size, start_idx + num_elements_per_thread);

    // 步骤①：每个线程都全量扫描找 max（大量重复劳动！）
    float max_val = -INFINITY;
    for (int i = 0; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // 步骤②：每个线程又全量扫描求 sum_exp（again！）
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_exp += expf(input[i] - max_val);
    }

    // 步骤③：只有这一步真正做到了分工
    for (int i = start_idx; i < end_idx; i++) {
        output[i] = expf(input[i] - max_val) / sum_exp;
    }
}
```

问题一目了然：1024 个线程，每个都把 6144 个元素的数组从头到尾读两遍 global memory——**同样的活儿被干了 1024 遍**。

**Shared Memory 版**：把归约拆成"先局部、后汇总"两级。每个线程只扫**自己那一段**，把局部结果写上 shared memory 白板，`__syncthreads()` 等所有人写完，再从白板上汇总出全局结果。注意其中 `NUM_THREADS` 是编译期常量（两个 shared 数组按它静态分配），启动时必须保证 `blockDim.x == NUM_THREADS`，否则会数组越界：

```cuda
__global__ void softmax_kernel_smem(float *input, float *output, int size) {
    int num_threads = blockDim.x;
    int num_elements_per_thread = size / num_threads;
    int thread_index = threadIdx.x;
    int start_idx = thread_index * num_elements_per_thread;
    int end_idx = min(size, start_idx + num_elements_per_thread);

    // 步骤①：局部求 max → 写白板 → 同步 → 汇总全局 max
    __shared__ float shared_max_val[NUM_THREADS];
    float max_val = -INFINITY;
    for (int i = start_idx; i < end_idx; i++) {      // 只扫自己的分片
        if (input[i] > max_val) max_val = input[i];
    }
    shared_max_val[thread_index] = max_val;
    __syncthreads();                                  // 等所有线程写完
    for (int i = 0; i < num_threads; i++) {           // 从白板汇总
        if (shared_max_val[i] > max_val) max_val = shared_max_val[i];
    }

    // 步骤②：局部求 sum_exp → 写白板 → 同步 → 汇总全局 sum
    __shared__ float shared_sum_exp[NUM_THREADS];
    float sum_exp_local = 0.0f;                       // 局部和用独立变量
    for (int i = start_idx; i < end_idx; i++) {
        sum_exp_local += expf(input[i] - max_val);
    }
    shared_sum_exp[thread_index] = sum_exp_local;
    __syncthreads();
    float sum_exp = 0.0f;                             // 从零汇总，避免重复计入自己
    for (int i = 0; i < num_threads; i++) {
        sum_exp += shared_sum_exp[i];
    }

    // 步骤③：与 naive 版相同
    for (int i = start_idx; i < end_idx; i++) {
        output[i] = expf(input[i] - max_val) / sum_exp;
    }
}
```

> 说明：以上两版示例在原文版本基础上做了数值稳健性修正——max 初值由 `0.0` 改为 `-INFINITY`（否则全负输入会出错），smem 版的 sum 归约改为"局部和独立变量 + 同步后从零汇总"（否则每个线程的分母会重复计入自己的局部和）。max 归约段在同步后重复比较自身值是无害的，保持原样。核心优化思路——用 Shared Memory 消除跨线程重复扫描——与原文一致。

每个线程的扫描量从 $2 \times size$ 降到约 $2 \times (size / num\_threads + num\_threads)$，且大部分读取落在高速的 shared memory 上。实测（input size = 6144，1024 线程，Nsight Compute 计时）：

| 版本 | 耗时 | 相对加速 |
|---|---|---|
| Naive（全量重复扫描） | 11.23 ms | 1× |
| Shared Memory 两级归约 | 1.5 ms | **≈ 7.5×** |

同时执行的总指令数大幅下降——省掉的正是那 1024 份重复劳动。**这就是 CUDA 性能优化的第一性原理：让数据尽量待在快的内存里，让线程之间少做重复功。**

## 九、CUDA 速查小结

进入 PyTorch 部分之前，把第一、二部分的知识收拢成一张总图和一张 API 表：

```text
【CUDA 速查总图】执行层级 × 内存金字塔

  执行层级                          内存金字塔（快→慢）
  ┌────────────────────────┐      ┌────────────────────────────┐
  │ Grid（一次 kernel 启动）│      │ Registers   255 个/线程     │
  │  ├─ Block（≤1024 线程） │      │  ▲ 线程私有，~1 周期        │
  │  │   必落在同一个 SM    │      ├────────────────────────────┤
  │  │   ├─ Warp（32 线程） │      │ Shared / L1  block 内共享   │
  │  │   │   调度基本单位   │      │  ▲ 与 L1 同址片上 SRAM      │
  │  │   │   ├─ Thread     │      ├────────────────────────────┤
  │  │   │   │  kernel 代码 │      │ L2 Cache     全卡共享       │
  │  │   │   │  描述的对象  │      │  ▲ sector=32B，成对预取     │
  └──┴───┴───┴─────────────┘      ├────────────────────────────┤
                                  │ Global / Local（显存颗粒）  │
   对应关系：                      │  容量最大，数百周期延迟      │
   Thread ↔ Register              └────────────────────────────┘
   Block  ↔ Shared Memory
   Grid   ↔ Global Memory
```

| 类别 | API / 语法 | 说明 |
|---|---|---|
| Kernel 定义 | `__global__ void f(...)` | CPU 发起、GPU 执行，无返回值 |
| 启动 | `f<<<numBlocks, numThreads>>>(...)` | 参数可为 `dim3`，异步执行 |
| 索引 | `threadIdx` / `blockIdx` / `blockDim` | 均为 dim3，线程定位三件套 |
| 同步 | `cudaDeviceSynchronize()` | 主机等 GPU 全部完成 |
| 同步 | `__syncthreads()` | block 内屏障 |
| 显存 | `cudaMalloc` / `cudaFree` | 设备内存分配/释放 |
| 拷贝 | `cudaMemcpy(dst, src, n, dir)` | 同步；dir 取 H2D/D2H 枚举 |
| 共享内存 | `__shared__ float buf[N];` | block 作用域，L1 同址 |
| 错误检查 | `cudaGetLastError()` / `cudaGetErrorString()` | 捕获静默的启动错误 |
| 编译 | `nvcc file.cu -o file.o` | CUDA 编译器 |
| Profiling | `ncu --set=full -f -o report ./bin` | Nsight Compute 采集 |

---

## 十、编写 PyTorch 自定义 Kernel

掌握了 CUDA 本身，最后一步是把自己写的 kernel 接进 PyTorch——毕竟没人想用裸 CUDA 重写整条训练流水线。PyTorch 官方提供的桥梁是 **C++ 扩展机制**（`torch.utils.cpp_extension`），底层靠 **pybind11** 完成 Python ↔ C++ 的绑定。

### 10.1 三文件结构：kernel + binding + setup

一个最小可用的 PyTorch CUDA 扩展由三个文件组成。还是用 `roll_call` 当例子。

**文件 1：`roll_call.cu`** —— kernel 本体，外加一个普通 C++ 包装函数（launcher）。因为 `<<<>>>` 启动语法只有 nvcc 认识，必须把启动逻辑留在 `.cu` 文件里，向外只暴露一个普通函数签名：

```cuda
// csrc/roll_call/roll_call.cu
#include <stdio.h>

__global__ void roll_call_kernel() {
    printf("Thread %d here!\n", threadIdx.x);
}

// launcher：普通 C++ 函数，封装 kernel 启动细节
void roll_call_launcher() {
    roll_call_kernel<<<1, 5>>>();
    cudaDeviceSynchronize();
}
```

**文件 2：`roll_call_binding.cpp`** —— 用 pybind11 把 launcher 注册成 Python 可调用的函数。注意 launcher 在这里只做**前向声明**，链接期才和 `.cu` 里的实现对上：

```cpp
// csrc/roll_call/roll_call_binding.cpp
#include <torch/extension.h>

// 前向声明：实现在 roll_call.cu 里
void roll_call_launcher();

void roll_call_binding() {
    roll_call_launcher();
}

// TORCH_EXTENSION_NAME 宏在编译时替换为 setup.py 里指定的扩展名，
// 避免在 C++ 里硬编码模块名
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "roll_call",              // Python 侧函数名
        &roll_call_binding,       // 绑定的 C++ 函数
        "Launches the roll_call kernel"   // docstring
    );
}
```

**文件 3：`setup.py`** —— 用 PyTorch 提供的 `CUDAExtension` + `BuildExtension`，它们替你搞定 nvcc 调用、头文件路径、和 libtorch 的链接等所有脏活：

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension('example_kernels', [
        'csrc/roll_call/roll_call_binding.cpp',
        'csrc/roll_call/roll_call.cu',
    ])
]

setup(
    name="cuda_basics",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
```

执行 `pip install -e .` 编译安装后，就能在 Python 里直接调用 GPU kernel 了：

```python
import example_kernels

example_kernels.roll_call()
# Thread 0 here!
# Thread 1 here!
# Thread 2 here!
# Thread 3 here!
# Thread 4 here!
```

三层调用关系一图看懂：

```text
【图 7】Python → binding.cpp → .cu 的三层桥接

  Python 层
  ┌─────────────────────────────────────────────┐
  │ import example_kernels                      │
  │ example_kernels.roll_call()                 │
  └──────────────────────┬──────────────────────┘
                         │ pybind11 生成的绑定
                         ▼
  C++ 绑定层（roll_call_binding.cpp，g++ 编译）
  ┌─────────────────────────────────────────────┐
  │ PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)    │
  │   m.def("roll_call", &roll_call_binding)    │
  │ roll_call_binding() ──▶ roll_call_launcher()│ ←── 仅前向声明
  └──────────────────────┬──────────────────────┘
                         │ 链接期解析符号
                         ▼
  CUDA 层（roll_call.cu，nvcc 编译）
  ┌─────────────────────────────────────────────┐
  │ roll_call_launcher()                        │
  │   └─▶ roll_call_kernel<<<1, 5>>>()   [GPU]  │
  │       cudaDeviceSynchronize()               │
  └─────────────────────────────────────────────┘
          ▲
          └── setup.py 用 CUDAExtension 把两个文件编译链接成
              同一个 Python 扩展模块 example_kernels
```

### 10.2 在 C++ 里操作 PyTorch Tensor

真正有用的 kernel 需要吃 tensor。`torch/extension.h` 暴露了 C++ 侧的 `torch::Tensor` 类型，launcher 用引用接收，再用 **`.data_ptr<T>()`** 取出底层显存的裸指针传给 kernel。把第三章的 `array_increment` 改造成 PyTorch 版：

```cpp
// launcher：接 PyTorch tensor，取裸指针喂给 kernel
void array_increment_launcher(torch::Tensor& in) {
    int size = in.numel();
    // tensor 已经在 GPU 上（调用方保证 .cuda()），
    // data_ptr 拿到的就是 global memory 地址，不需要任何 cudaMemcpy
    array_increment_kernel<<<1, size>>>(in.data_ptr<int>());
    cudaDeviceSynchronize();
}
```

教学提示：这里直接拿 `numel()` 当线程数只是为了演示方便，受单 Block 1024 线程上限约束，元素一多就会静默失败；真实工程应按 `(size + threads - 1) / threads` 计算 Block 数、划分多 Block 启动。

Python 侧：

```python
import torch
import example_kernels

x = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
example_kernels.array_increment(x)
print(x)   # tensor([1, 2, 3, 4], device='cuda:0', dtype=torch.int32)
```

注意两点：其一，tensor 本来就住在显存里，kernel 直接原地改，**修改对 Python 侧立即可见**（in-place 语义），全程零拷贝；其二，`data_ptr<int>()` 的模板参数必须与 tensor 的实际 dtype 一致——这引出最后一个问题。

### 10.3 处理多 dtype：AT_DISPATCH 宏

PyTorch 的 tensor 可能是 float32、float64、half……而 C++ 指针类型是编译期定死的。标准解法分两步。第一步，把 kernel 写成模板：

```cuda
template <typename T>
__global__ void array_increment_kernel(T* in) {
    const int threadIndex = threadIdx.x;
    in[threadIndex] = in[threadIndex] + 1;
}
```

第二步，在 launcher 里用 PyTorch 的 **`AT_DISPATCH_FLOATING_TYPES`** 宏按运行时 dtype 分发。它的本质是一个 switch：检查 tensor 的实际类型，把宏内部的 `scalar_t` 替换成对应的 C++ 类型，再执行你给的 lambda：

```cpp
void array_increment_launcher(torch::Tensor& in) {
    int size = in.numel();
    AT_DISPATCH_FLOATING_TYPES(in.type(), "array_increment", [&]() {
        // scalar_t 由宏根据 in 的运行时 dtype 决定（float / double）
        array_increment_kernel<<<1, size>>>(in.data_ptr<scalar_t>());
    });
    cudaDeviceSynchronize();
}
```

这样同一份代码就能同时服务 float32 和 float64 的 tensor。除了 `AT_DISPATCH_FLOATING_TYPES`，PyTorch 还提供了覆盖半精度、整型、复数等各种组合的一整族分发宏，完整清单见 PyTorch 源码 [aten/src/ATen/Dispatch.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch.h)。

## 十一、结语

回顾一下这趟旅程走过的路：

- **执行模型**：kernel 描述单线程行为，`<<<blocks, threads>>>` 决定规模；线程按 32 个编成 warp 被 SM 上的调度器发射，同一 block 必落同一 SM——这条硬件规则是一切 block 内协作的根基。
- **内存模型**：Global（大而慢，sector 粒度经 L1/L2 缓存）、Register（快但每线程 255 个）、Local（名字骗人、物理在显存）、Shared（block 白板、与 L1 同址）；Softmax 案例证明，仅仅是"让线程别做重复功 + 把中间结果放上白板"，就能换来 7.5 倍加速。
- **工程接入**：`.cu`（kernel + launcher）→ `binding.cpp`（pybind11 注册）→ `setup.py`（CUDAExtension 编译）三件套，加上 `data_ptr` 与 `AT_DISPATCH` 宏，就足以把手写 kernel 变成一个普通的 Python 函数。

这些内容只是 CUDA 世界的地基——访存合并、bank conflict、warp divergence、Tensor Core 编程、CUDA Graph 都还在前面。但地基最重要的作用是消除恐惧：**自定义算子不是黑魔法，只是三个文件加一次 `pip install`。** 最好的下一步是找一块能摸到的 GPU（Colab 也行），把文中的 kernel 逐个敲一遍、用 `ncu` 看一看自己代码的内存流。

延伸阅读：

- 原文：[Writing CUDA Kernels for PyTorch — tinkerd.net](https://tinkerd.net/blog/machine-learning/cuda-basics/)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)：CUDA 官方权威手册
- [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)：SM 结构与硬件参数的一手来源
- [PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)：官方扩展教程
