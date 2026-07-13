---
title: "Agentic Infra 前沿：多 Agent 通信、调度与工程实践"
date: '2026-06-26'
tags:
- LLM
- Agent

draft: false
math: true
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

> 这篇文章不讲 "什么是 Agent"、也不讲 DP/TP/PP，只聚焦一个问题：
>
> **当你有 N 个 Agent 需要协作完成复杂任务时，底层 Infra 到底在解决什么问题？**

---

## 一、为什么需要 Agentic Infra

单 Agent 的瓶颈在 2025 年已经很明显：

| 问题 | 根因 | 传统方案上限 |
|------|------|-------------|
| 上下文爆炸 | 单 Agent 要塞进所有 tool/memory/context | 128k~1M 窗口仍然不够 |
| 延迟不可控 | 串行 tool call → LLM → tool call | 用户等不起 10+ 轮 |
| 单点故障 | 一个幻觉就全盘皆输 | retry 治标不治本 |
| 能力天花板 | 一个 prompt 不可能精通所有子领域 | role prompt 难以 scale |

**Agentic Infra 的核心命题：把"一个超人 Agent" 拆成 "一群专家 Agent 的协作网络"**，并为这个网络提供通信、发现、调度、容错、可观测等基础设施。

---

## 二、Agent 间通信拓扑

### 2.1 三种基本拓扑

```
┌─────────────────────────────────────────────────────────────┐
│  (A) Centralized Orchestrator                               │
│                                                             │
│              ┌───────────┐                                  │
│              │ Orchestrator│                                 │
│              └──┬──┬──┬──┘                                  │
│                 │  │  │                                      │
    │          ┌──┘  │  └──┐                                      │
    │          ▼     ▼     ▼                                      │
    │        Agent  Agent  Agent                                  │
    │          A      B      C                                    │
│                                                             │
│  优点：简单、可控、易 debug                                    │
│  缺点：瓶颈、单点故障、延迟累积                                │
├─────────────────────────────────────────────────────────────┤
│  (B) Peer-to-Peer (P2P)                                    │
│                                                             │
│        Agent A ◄──────► Agent B                             │
│          ▲                 ▲                                │
│          │                 │                                │
│          └──► Agent C ◄───┘                                 │
│                                                             │
│  优点：无单点、低延迟、弹性扩展                                  │
│  缺点：消息风暴、一致性难、调试困难                               │
├─────────────────────────────────────────────────────────────┤
│  (C) Hierarchical (层级式)                                   │
│                                                             │
│              ┌──────────┐                                   │
│              │ Supervisor │                                 │
│              └──┬─────┬──┘                                  │
│                 │     │                                     │
│          ┌─────┘     └─────┐                                │
│          ▼                 ▼                                │
│     ┌────────┐       ┌────────┐                             │
│     │Team Lead│       │Team Lead│                           │
│     └──┬──┬──┘       └──┬──┬──┘                             │
│        │  │              │  │                               │
│        ▼  ▼              ▼  ▼                               │
│       A1  A2            B1  B2                              │
│                                                             │
│  优点：可控扩展、局部自治、适合复杂任务分解                        │
│  缺点：层级延迟、Supervisor 是瓶颈                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 P2P 通信的工程挑战

真正做 Agent P2P 通信，你要解的不是"发个消息"，而是：

**1) 消息格式标准化**

```
┌─────────────────────────────────────────┐
│          Agent Message Envelope          │
├─────────────────────────────────────────┤
│ sender_id:    "agent-code-review-01"    │
│ receiver_id:  "agent-test-gen-03"       │
│ msg_type:     REQUEST / RESPONSE / EVENT│
│ correlation_id: "uuid-xxx"              │
│ payload:      { structured data }       │
│ metadata:                               │
│   - priority: HIGH / NORMAL / LOW       │
│   - ttl: 30s                            │
│   - retry_policy: exponential_backoff   │
│   - trace_id: "span-yyy"               │
└─────────────────────────────────────────┘
```

**2) 路由与发现**

Agent 怎么知道该把消息发给谁？

- **Capability-based routing**: 每个 Agent 注册自己的能力向量（"我擅长 code review"、"我能调用 K8s API"），请求方按能力匹配
- **Intent-based routing**: 用一个轻量 LLM 做 router，根据消息 intent 分发到合适的 Agent
- **Topic-based pub/sub**: Agent 订阅自己关心的 topic，发布者不需要知道谁会消费

**3) 背压与流控**

当 Agent B 处理速度跟不上 Agent A 的请求速度：

```
Agent A ──[msg1, msg2, msg3, ...]──► Agent B (过载)

解法：
  ① Token bucket: 每个 Agent 有发送速率限制
  ② 优先级队列: HIGH 优先消费，LOW 可以被丢弃
  ③ Circuit breaker: 连续 N 次超时后断路，fallback 到其他 Agent
  ④ Backpressure signal: Agent B 主动告知 "slow down"
```

---

## 三、Google A2A 协议：Agent-to-Agent 通信标准

### 3.1 核心设计理念

Google 在 2025 年发布的 A2A（Agent-to-Agent）协议，试图解决一个根本问题：**不同框架、不同厂商的 Agent 如何互操作？**

```
┌─────────────────────────────────────────────────────┐
│                  A2A Protocol Stack                  │
├─────────────────────────────────────────────────────┤
│ Layer 4: Application    │ Task execution & results  │
├─────────────────────────┼───────────────────────────┤
│ Layer 3: Collaboration  │ Task lifecycle mgmt       │
├─────────────────────────┼───────────────────────────┤
│ Layer 2: Communication  │ JSON-RPC over HTTP/SSE    │
├─────────────────────────┼───────────────────────────┤
│ Layer 1: Discovery      │ Agent Card (/.well-known) │
└─────────────────────────┴───────────────────────────┘
```

### 3.2 Agent Card：自描述机制

每个 Agent 通过 `/.well-known/agent.json` 暴露自己的能力：

```json
{
  "name": "CodeReviewAgent",
  "description": "Reviews code for bugs, security issues, and style",
  "url": "https://agents.example.com/code-review",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": true
  },
  "skills": [
    {
      "id": "code-review",
      "name": "Code Review",
      "description": "Analyze code for potential issues",
      "inputModes": ["text/plain", "application/json"],
      "outputModes": ["text/plain", "application/json"]
    }
  ],
  "authentication": {
    "schemes": ["OAuth2", "API-Key"]
  }
}
```

### 3.3 Task 生命周期

```
┌──────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐
│ submitted │───►│  working   │───►│ completed  │    │  failed  │
└──────────┘    └───────────┘    └───────────┘    └──────────┘
                      │                                  ▲
                      │         ┌───────────┐            │
                      └────────►│  blocked   │───────────┘
                                └───────────┘
                                (等待其他 Agent)
```

关键设计点：
- **Task 是一等公民**：每个协作单元是一个 Task，有唯一 ID、状态机、超时策略
- **Artifact 传递**：Agent 产出的结果作为 Artifact 附在 Task 上，下游 Agent 可以消费
- **Long-running support**：通过 SSE 支持流式返回、通过 webhook 支持异步回调

### 3.4 A2A vs MCP：定位差异

```
┌────────────────────────────────────────────────────────────────┐
│                    协议定位对比                                  │
├──────────────┬─────────────────────┬───────────────────────────┤
│              │       MCP           │          A2A              │
├──────────────┼─────────────────────┼───────────────────────────┤
│ 解决什么     │ Agent ↔ Tool 连接   │ Agent ↔ Agent 协作        │
│ 比喻         │ USB 接口标准        │ HTTP 协议标准             │
│ 通信方向     │ Agent 调用 Tool     │ Agent 互相调用            │
│ 状态管理     │ 无状态（每次调用）   │ 有状态（Task 生命周期）   │
│ 典型场景     │ 读数据库、调 API    │ 多 Agent 分工协作         │
│ 谁发起的     │ Anthropic           │ Google                    │
└──────────────┴─────────────────────┴───────────────────────────┘
```

**两者是互补的**：MCP 让 Agent 能"用工具"，A2A 让 Agent 能"找同事"。一个完整的 Agentic 系统两者都需要。

---

## 四、Agent 调度与资源管理

### 4.1 问题建模

多 Agent 系统的调度问题比传统微服务更复杂：

| 维度 | 微服务调度 | Agent 调度 |
|------|-----------|-----------|
| 负载预测 | 相对稳定 | 高度不确定（LLM token 数不定） |
| 执行时间 | ms ~ s 级 | s ~ min 级（多轮推理） |
| 资源类型 | CPU/Mem | GPU/Context Window/Tool quota |
| 依赖关系 | 静态（API 调用图） | 动态（运行时决定找谁） |
| 失败模式 | crash/timeout | 幻觉/死循环/资源耗尽 |

### 4.2 调度策略

```
┌─────────────────────────────────────────────────────────────┐
│              Agent Scheduler Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Task Queue          Scheduler            Agent Pool        │
│  ┌──────┐         ┌───────────┐        ┌──────────────┐   │
│  │Task 1│────────►│           │───────►│ Agent A (idle) │   │
│  │Task 2│         │  Policy:  │        │ Agent B (busy) │   │
│  │Task 3│         │  - LLM    │        │ Agent C (idle) │   │
│  │ ...  │         │    cost   │        │ Agent D (warm) │   │
│  └──────┘         │  - Latency│        └──────────────┘   │
│                   │  - Quality│                             │
│                   │  - Budget │         Resource Monitor    │
│                   └───────────┘        ┌──────────────┐   │
│                                        │ GPU util: 78% │   │
│                                        │ Token/s: 1.2k │   │
│                                        │ Queue depth: 5│   │
│                                        └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**核心调度维度**：

- **Cost-aware**: 不同 Agent 用不同模型（GPT-4o vs Claude vs 本地 7B），同一个任务的 token 成本差 100x
- **Latency-sensitive**: 用户交互型任务需要 < 3s 响应，后台分析型任务可以慢慢来
- **Quality-driven**: 关键决策路径用强模型，日常格式化用弱模型
- **Budget-constrained**: 总 token 预算限制下的最优分配

### 4.3 Warm Pool 与 Context 复用

Agent 不是无状态的——它的 KV Cache、System Prompt、Tool Schema 都是"预热成本"：

```
Cold start:
  Load system prompt (2k tokens)
  + Tool schemas (5k tokens)
  + Memory retrieval (3k tokens)
  + Few-shot examples (2k tokens)
  = 12k tokens "浪费" 在 prefill
  ≈ 0.5-2s latency penalty

Warm pool strategy:
  ┌──────────────────────────────────────────────┐
  │ Agent Instance Pool                          │
  ├──────────────────────────────────────────────┤
  │ Instance 1: [System + Tools cached]  → WARM  │
  │ Instance 2: [System + Tools cached]  → WARM  │
  │ Instance 3: [spinning up...]         → COLD  │
  │ Instance 4: [idle > 5min]            → EVICT │
  └──────────────────────────────────────────────┘

  关键指标：
  - Warm hit rate > 80% → 延迟稳定
  - Cache eviction policy: LRU + priority weight
  - Prefill 复用: KV Cache sharing across requests
```

---

## 五、Agent 可靠性工程

### 5.1 故障模式分类

传统服务的故障是"挂了"，Agent 的故障更加微妙：

```
┌─────────────────────────────────────────────────────────────┐
│              Agent Failure Taxonomy                          │
├──────────────────┬──────────────────────────────────────────┤
│ 类型              │ 表现                                     │
├──────────────────┼──────────────────────────────────────────┤
│ Hallucination    │ 自信地返回错误结果                        │
│ Infinite Loop    │ 反复调用同一个 tool 不收敛                │
│ Context Overflow │ 塞太多信息导致遗忘关键约束                │
│ Cascading Fail   │ Agent A 的幻觉被 Agent B 当真            │
│ Resource Drain   │ 不停生成 token 耗尽预算                   │
│ Deadlock         │ A 等 B 的结果，B 等 A 的确认              │
│ Byzantine        │ 同一问题，多次回答不一致                   │
└──────────────────┴──────────────────────────────────────────┘
```

### 5.2 可靠性模式

**1) 投票与共识（Voting / Consensus）**

```
         Task: "这段代码有 bug 吗？"
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 Agent A   Agent B   Agent C
 (GPT-4o)  (Claude)  (Gemini)
    │         │         │
    ▼         ▼         ▼
 "有bug,    "有bug,   "没有
  L42空指针"  L42 NPE"  问题"
    │         │         │
    └─────────┼─────────┘
              ▼
      Consensus: 2/3 判定有 bug
      置信度: 0.67
      Action: 标记为 HIGH confidence issue
```

**2) 守卫 Agent（Guardian Pattern）**

```
┌─────────────────────────────────────────┐
│         Guardian Pattern                │
│                                         │
│  User ──► Worker Agent ──► Guardian ──► Output
│              │                  │       │
│              │    ┌─────────────┘       │
│              │    │ 检查项：             │
│              │    │ - 事实一致性         │
│              │    │ - 安全合规           │
│              │    │ - 格式正确性         │
│              │    │ - 逻辑自洽           │
│              │    │                     │
│              │    │ 不通过 → 打回重做    │
│              └────┘                     │
└─────────────────────────────────────────┘
```

**3) 断路器 + 降级（Circuit Breaker + Fallback）**

```
Agent A 调用 Agent B：

正常:  A ──► B ──► 结果 (< 10s)

降级链:
  B 超时 3 次 → 断路器 OPEN
    → 尝试 Agent B' (备用实例)
      → B' 也超时
        → 降级到本地规则引擎（不用 LLM）
          → 返回 "low-confidence" 结果 + flag

状态机:
  CLOSED ──[3次失败]──► OPEN ──[30s冷却]──► HALF-OPEN
     ▲                                          │
     └──────────[成功]─────────────────────────────┘
```

---

## 六、Agent Memory 基础设施

### 6.1 多 Agent 共享记忆架构

单 Agent 的 memory 是私有的，多 Agent 系统需要分层记忆：

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Agent Memory Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 3: Global Knowledge (所有 Agent 共享)                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ - 项目文档、代码库索引、API schema                     │ │
│  │ - 全局约束（"不要修改 prod 数据库"）                   │ │
│  │ - 长期事实（"用户偏好 TypeScript"）                    │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Layer 2: Team Memory (同组 Agent 共享)                      │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ - 当前任务上下文、中间产物                             │ │
│  │ - 已做决策 & 原因（避免重复讨论）                      │ │
│  │ - 冲突解决记录                                        │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Layer 1: Private Memory (Agent 私有)                        │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ - 个人 scratchpad、推理中间状态                        │ │
│  │ - Tool 调用历史、error log                            │ │
│  │ - 自我反思 & 策略调整                                  │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  底层存储:                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐             │
│  │ Vector DB │  │ KV Store  │  │ Event Stream │             │
│  │(语义检索) │  │(精确查找) │  │(时序追溯)    │             │
│  └──────────┘  └──────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Memory Consistency 问题

多 Agent 写同一份记忆时，面临经典的并发问题：

```
时间线:
  t=0: Agent A 读取 memory["status"] = "in_progress"
  t=1: Agent B 读取 memory["status"] = "in_progress"
  t=2: Agent A 写入 memory["status"] = "completed"
  t=3: Agent B 写入 memory["status"] = "failed"  ← 覆盖了 A 的结果!

解法:
  ① Optimistic Locking: 带版本号写入，冲突时 retry
  ② Event Sourcing: 只追加 event，不直接改状态
  ③ CRDT (Conflict-free Replicated Data Type): 自动合并无冲突
  ④ 分区写入: 每个 Agent 只写自己的 namespace，读取时 merge
```

---

## 七、Agent 可观测性

### 7.1 Trace 模型

多 Agent 的 trace 不是线性的，而是 DAG：

```
Trace: user_request_001
│
├── Span: Orchestrator.plan (120ms)
│   └── LLM call: GPT-4o (token_in=500, token_out=200)
│
├── Span: Agent_CodeGen.execute (8.5s)  ──────────────────┐
│   ├── LLM call: Claude-3.5 (token_in=3000, token_out=1500)
│   ├── Tool call: file_read("main.py") (15ms)           │
│   ├── Tool call: file_write("fix.py") (20ms)           │ 并行
│   └── LLM call: Claude-3.5 (token_in=2000, token_out=800)
│                                                         │
├── Span: Agent_TestGen.execute (6.2s)  ──────────────────┘
│   ├── LLM call: GPT-4o-mini (token_in=2000, token_out=1000)
│   └── Tool call: run_tests() (3.1s)
│
├── Span: Agent_Review.execute (4.8s)  [depends on CodeGen + TestGen]
│   ├── LLM call: GPT-4o (token_in=5000, token_out=500)
│   └── Decision: APPROVE (confidence=0.92)
│
└── Total: 13.3s, 15000 tokens consumed, $0.08 cost
```

### 7.2 关键可观测指标

```
┌─────────────────────────────────────────────────────────────┐
│              Agent Observability Metrics                     │
├──────────────────┬──────────────────────────────────────────┤
│ 类别              │ 指标                                     │
├──────────────────┼──────────────────────────────────────────┤
│ 性能             │ E2E latency, per-agent latency           │
│                  │ Token throughput (tok/s)                  │
│                  │ Queue wait time                          │
├──────────────────┼──────────────────────────────────────────┤
│ 成本             │ Token consumption per task               │
│                  │ Cost per successful completion           │
│                  │ Wasted tokens (failed attempts)          │
├──────────────────┼──────────────────────────────────────────┤
│ 质量             │ Task success rate                        │
│                  │ Hallucination rate (via guardian)         │
│                  │ Human override rate                      │
├──────────────────┼──────────────────────────────────────────┤
│ 可靠性           │ Agent crash rate                         │
│                  │ Circuit breaker trigger rate             │
│                  │ Retry rate & success-after-retry rate    │
├──────────────────┼──────────────────────────────────────────┤
│ 协作             │ Message volume between agents            │
│                  │ Avg handoff count per task               │
│                  │ Deadlock detection count                 │
└──────────────────┴──────────────────────────────────────────┘
```

---

## 八、前沿方向：Agent Mesh & Runtime

### 8.1 Agent Mesh（类比 Service Mesh）

就像 Istio 给微服务加了 sidecar，Agent Mesh 给每个 Agent 加"通信代理层"：

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Mesh Architecture                   │
│                                                             │
│  ┌─────────────────┐         ┌─────────────────┐          │
│  │    Agent A       │         │    Agent B       │          │
│  │  (business logic)│         │  (business logic)│          │
│  └────────┬─────────┘         └────────┬─────────┘         │
│           │                             │                   │
│  ┌────────▼─────────┐         ┌────────▼─────────┐        │
│  │  Agent Sidecar    │◄───────►│  Agent Sidecar    │        │
│  │  - Auth/AuthZ     │         │  - Auth/AuthZ     │        │
│  │  - Rate limiting  │         │  - Rate limiting  │        │
│  │  - Tracing inject │         │  - Tracing inject │        │
│  │  - Circuit break  │         │  - Circuit break  │        │
│  │  - Schema validate│         │  - Schema validate│        │
│  │  - Cost metering  │         │  - Cost metering  │        │
│  └──────────────────┘         └──────────────────┘         │
│           │                             │                   │
│           └──────────┬──────────────────┘                   │
│                      ▼                                      │
│           ┌──────────────────┐                              │
│           │   Control Plane   │                              │
│           │  - Agent Registry │                              │
│           │  - Policy Engine  │                              │
│           │  - Load Balancer  │                              │
│           │  - Observability  │                              │
│           └──────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

Agent Mesh 的价值：**业务 Agent 只管推理逻辑，通信/安全/限流/计费全部下沉到基础设施层**。

### 8.2 Agent Runtime：从框架到平台

当前 Agent 框架（LangGraph、CrewAI、AutoGen）的问题是**只管编排，不管运维**。下一代 Agent Runtime 需要：

```
┌─────────────────────────────────────────────────────────────┐
│              Agent Runtime 能力矩阵                          │
├──────────────────┬──────────────────────────────────────────┤
│ 能力              │ 说明                                     │
├──────────────────┼──────────────────────────────────────────┤
│ Hot Deploy       │ 不停机更新 Agent 的 prompt / tool / model │
│ Auto-scaling     │ 根据 queue depth 自动扩缩 Agent 实例      │
│ Canary Release   │ 新版 Agent 先接 5% 流量，观测质量          │
│ State Checkpoint │ Agent 执行到一半可以暂停、恢复、回滚       │
│ Cost Governor    │ 实时监控 token 消耗，超预算自动降级        │
│ Sandbox          │ Agent 的 tool call 在隔离环境执行          │
│ Replay & Debug   │ 从任意 checkpoint 重放，修改输入观测变化   │
│ Multi-tenancy    │ 同一 Agent 服务多租户，数据隔离            │
└──────────────────┴──────────────────────────────────────────┘
```

### 8.3 Speculative Execution for Agents

借鉴 CPU 分支预测的思路：Agent 可以"推测执行"多条路径：

```
User: "帮我重构这个函数，要求性能提升 2x"

                    ┌─────────────────────────┐
                    │ Orchestrator 预判：       │
                    │ 可能需要 3 种重构方向     │
                    └─────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │Path A:     │  │Path B:     │  │Path C:     │
     │算法优化    │  │并行化      │  │缓存策略    │
     │(speculate) │  │(speculate) │  │(speculate) │
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │                │                │
           ▼                ▼                ▼
     benchmark         benchmark         benchmark
     1.8x ✓            2.3x ✓✓           1.2x ✗

  → 选择 Path B，丢弃 A 和 C
  → 代价：多花了 2x token，但延迟从串行 30s → 并行 12s
```

**Trade-off**: token 成本 vs 延迟，适合**用户在线等待**且**方案不确定性高**的场景。

---

## 九、实战参考：典型 Agentic 系统架构

### 9.1 代码开发 Agentic System

```
┌─────────────────────────────────────────────────────────────────┐
│                   Code Development Agent System                  │
│                                                                 │
│  User Request: "给这个 API 加分页功能"                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────┐                                               │
│  │ Planner Agent │ ── 分析需求，拆解子任务                       │
│  └──────┬───────┘                                               │
│         │ Task DAG:                                              │
│         │ ① 理解现有代码结构                                     │
│         │ ② 设计分页方案                                         │
│         │ ③ 实现代码变更                                         │
│         │ ④ 编写测试                                             │
│         │ ⑤ Review & 合并                                        │
│         ▼                                                       │
│  ┌──────────────┐     ┌──────────────┐                          │
│  │ Search Agent  │────►│ Architect    │ ── 输出设计方案           │
│  │ (理解代码)    │     │ Agent        │                          │
│  └──────────────┘     └──────┬───────┘                          │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    ▼                   ▼                         │
│           ┌──────────────┐    ┌──────────────┐                  │
│           │ Coder Agent   │    │ Test Agent    │  ← 并行执行      │
│           │ (写实现代码)  │    │ (写测试用例)  │                  │
│           └──────┬───────┘    └──────┬───────┘                  │
│                  │                   │                           │
│                  └─────────┬─────────┘                          │
│                            ▼                                    │
│                   ┌──────────────┐                               │
│                   │ Review Agent  │ ── 检查一致性、安全性          │
│                   └──────┬───────┘                               │
│                          │                                       │
│                          ▼                                       │
│                   ┌──────────────┐                               │
│                   │ Guardian      │ ── 最终安全门禁               │
│                   └──────────────┘                               │
│                                                                 │
│  通信方式：A2A 协议                                              │
│  调度策略：关键路径用 GPT-4o，并行任务用 Claude-3.5-Sonnet        │
│  容错：Review Agent 不过 → Coder Agent 重试（max 3 次）          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 十、总结与展望

| 层次 | 当前状态 (2025) | 趋势 (2026+) |
|------|---------------|-------------|
| 通信 | 框架私有协议为主 | A2A 标准化，跨框架互操作 |
| 发现 | 硬编码 Agent 列表 | Agent Registry + 能力匹配 |
| 调度 | 简单 round-robin | Cost/Latency/Quality 多目标优化 |
| 可靠性 | 手动 retry | 自动断路器 + 投票共识 + Guardian |
| 记忆 | 单 Agent RAG | 分层共享记忆 + CRDT 同步 |
| 运维 | 无（开发者自己搞） | Agent Runtime 平台化 |
| 安全 | 信任所有 Agent | Zero-trust Agent Mesh |

**核心观点**：Agentic Infra 正在重走微服务基础设施 2015-2020 的路，从"手动拼接"走向"平台化运维"。区别在于 Agent 的不确定性远高于传统服务，这要求基础设施具备**概率性思维**——不追求 100% 正确，而是通过冗余、投票、降级把系统级可靠性提上去。

---

## 附录：术语速查

| 术语 | 含义 |
|------|------|
| A2A | Agent-to-Agent Protocol，Google 提出的 Agent 间通信标准 |
| MCP | Model Context Protocol，Anthropic 提出的 Agent-Tool 连接标准 |
| Agent Card | A2A 中 Agent 的自描述元数据，类似微服务的 service descriptor |
| Agent Mesh | 类比 Service Mesh，为 Agent 通信提供基础设施层（认证、限流、追踪） |
| Guardian Agent | 专门负责验证其他 Agent 输出正确性的"看门人"Agent |
| Circuit Breaker | 断路器模式，连续失败后停止调用，防止级联故障 |
| Speculative Execution | 并行尝试多条路径，选最优结果，用 token 换延迟 |
| CRDT | Conflict-free Replicated Data Type，无冲突复制数据类型 |
| Warm Pool | 预热的 Agent 实例池，避免冷启动的 prefill 开销 |
| Byzantine Failure | 拜占庭故障，Agent 对同一问题给出不一致答案 |
