{{< figure src="/images/posts/image.png" width="95%" >}}


在这个架构中，**Batcher（位于 Rust Router 中）** 到 **Model Shard（Python Server）** 的通信是通过 **gRPC over Unix Domain Socket (UDS)** 完成的。

以下是基于 TGI 源码逻辑简化的伪代码，展示了从 Batcher 组批到通过 gRPC 发送给 Model Shard 的全过程：

### 1. Proto 定义 (接口契约)
首先，Rust 和 Python 之间通过 `.proto` 文件约定接口。TGI 定义了 `Prefill`（预填充）和 `Decode`（解码）等 RPC 方法。

```protobuf
// generate.proto (简化版)
service TextGenerationService {
  // 处理新请求的 Prompt (Prefill 阶段)
  rpc Prefill(Batch) returns (PrefillResponse);
  
  // 处理已缓存 Batch 的下一个 Token (Decode 阶段)
  rpc Decode(DecodeRequest) returns (DecodeResponse);
  
  // 清理显存中的 Batch
  rpc ClearCache(ClearCacheRequest) returns (ClearCacheResponse);
}

message Batch {
  uint64 id = 1;
  repeated Request requests = 2; // 包含 tokenized 的 input_ids
  uint32 size = 3;
}
```

### 2. Rust 端 (Router/Batcher) 伪代码
Router 是用 Rust 写的，负责接收 HTTP 请求、Tokenize、组批（Batching），然后作为 **gRPC Client** 调用 Python 端。

```rust
// router/src/server.rs (简化逻辑)

use tonic::transport::Channel; // gRPC 库
use crate::pb::text_generation_service_client::TextGenerationServiceClient;

struct Router {
    // 指向 Python Model Shard 的 gRPC 客户端
    // 注意：TGI 默认使用 Unix Domain Socket (uds://...) 而非 TCP，速度更快
    grpc_client: TextGenerationServiceClient<Channel>,
    batcher: Batcher,
}

impl Router {
    async fn generation_loop(&mut self) {
        loop {
            // 1. Batcher 从 Buffer 中取出请求，组成一个 Batch
            // 这里实现了 Continuous Batching 逻辑
            let next_batch = self.batcher.next_batch().await;
            
            if let Some(batch) = next_batch {
                if batch.is_prefill {
                    // --- Prefill 阶段 (处理新 Prompt) ---
                    
                    // 2. 构造 gRPC 请求
                    let grpc_request = tonic::Request::new(PbBatch {
                        id: batch.id,
                        requests: batch.requests.into_iter().map(|r| r.into()).collect(),
                        size: batch.len() as u32,
                    });

                    // 3. 【关键步骤】通过 gRPC 发送给 Python Model Shard
                    // 这一步就是图中 "Batcher -> gRPC -> Model Shard" 的过程
                    let response = self.grpc_client.prefill(grpc_request).await?;
                    
                    // 4. 处理返回结果 (生成的第一个 token + KV Cache ID)
                    self.handle_generations(response.into_inner());
                    
                } else {
                    // --- Decode 阶段 (生成后续 Token) ---
                    
                    // 2. 构造 Decode 请求 (只传 Batch ID，数据已在 Python 端显存中)
                    let grpc_request = tonic::Request::new(PbDecodeRequest {
                        batches: vec![PbCachedBatch { id: batch.id, ... }],
                    });

                    // 3. 【关键步骤】通过 gRPC 发送 Decode 指令
                    let response = self.grpc_client.decode(grpc_request).await?;
                    
                    // 4. 流式返回给客户端
                    self.handle_generations(response.into_inner());
                }
            }
        }
    }
}
```

### 3. Python 端 (Model Shard) 伪代码
Python 端运行着 PyTorch 模型，作为 **gRPC Server** 等待 Rust 端的调用。

```python
# server/text_generation_server/server.py (简化逻辑)

import grpc
from concurrent import futures
import generate_pb2
import generate_pb2_grpc
from text_generation_server.models import Model

class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(self, model: Model):
        self.model = model
        self.cache = {} # 存储 KV Cache

    def Prefill(self, request, context):
        """
        对应图中的 gRPC -> Model Shard
        接收 Rust 发来的 Batch，执行模型的前向传播 (Prefill)
        """
        # 1. 将 Protobuf 对象转换为 PyTorch Tensor
        batch = self.model.batch_type.from_pb(request, self.model.tokenizer)
        
        # 2. 执行模型推理 (Flash Attention / Paged Attention)
        # 这一步真正消耗 GPU 算力
        generations, next_batch, timings = self.model.forward(batch)
        
        # 3. 将生成的 Token 和 Cache 信息存起来
        if next_batch:
            self.cache[next_batch.id] = next_batch
            
        # 4. 返回结果给 Rust Router
        return generate_pb2.PrefillResponse(
            generations=[g.to_pb() for g in generations],
            batch=next_batch.to_pb() if next_batch else None,
        )

    def Decode(self, request, context):
        """
        处理 Decode 阶段，利用显存中已有的 KV Cache 生成下一个 Token
        """
        # 1. 从缓存中取出之前的 Batch 状态
        batches = [self.cache[b.id] for b in request.batches]
        batch = self.model.batch_type.concatenate(batches)
        
        # 2. 执行 Decode 步 (只计算最新的一个 Token)
        generations, next_batch, timings = self.model.forward(batch)
        
        # 3. 更新缓存
        self.cache[next_batch.id] = next_batch
        
        return generate_pb2.DecodeResponse(...)

def serve():
    # 启动 gRPC 服务器，监听 Unix Socket
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=1))
    generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(
        TextGenerationService(model), server
    )
    # 监听 uds:///tmp/text-generation-server-0
    server.add_insecure_port("unix:///tmp/text-generation-server-0")
    await server.start()
    await server.wait_for_termination()
```

### 总结：数据流向

1.  **HTTP Request** 到达 Rust Router。
2.  **Buffer** 暂存请求。
3.  **Batcher** 根据策略（如 max_wait_time, max_batch_size）将多个请求打包成一个 `Batch` 对象。
4.  **Rust gRPC Client** 将这个 `Batch` 序列化为 Protobuf 二进制流。
5.  通过 **Unix Domain Socket** (高性能本地 IPC) 发送到 Python 进程。
6.  **Python gRPC Server** 反序列化得到 `Batch`，转换为 Tensor，送入 **GPU** 进行计算。
7.  如果是多卡（Model Shard），Python 进程内部通过 **NCCL** 进行 All-Reduce 通信 [[1]]。
8.  计算结果沿原路返回给 Rust Router，最终流式输出给用户。

**为什么这么设计？**
*   **Rust (Router):** 处理高并发网络 IO、Tokenization、复杂的 Batching 调度逻辑，性能极高且内存安全。
*   **Python (Model Shard):** 专注于 GPU 计算和模型逻辑，利用 PyTorch 生态。
*   **gRPC:** 提供了强类型约束和高性能的序列化，比 REST/JSON 快得多，适合这种高频、低延迟的内部通信。