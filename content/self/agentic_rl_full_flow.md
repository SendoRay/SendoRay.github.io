# Agentic RL 全流程详解

本文档详细阐述 ROLL 框架中 Agentic RL（智能体强化学习）的完整流程，重点说明**容器化环境隔离机制**（`env.reset` 通过 Docker 镜像拉取与容器启动完成环境初始化、`env.step` 执行智能体动作），以及**专用 Kubernetes 集群如何管理和隔离每个容器化环境**。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [容器化环境隔离机制](#2-容器化环境隔离机制)
3. [env.reset()：环境初始化全流程](#3-envreset环境初始化全流程)
4. [env.step()：智能体动作执行](#4-envstep智能体动作执行)
5. [Kubernetes 集群的容器管理与隔离](#5-kubernetes-集群的容器管理与隔离)
6. [Rollout 调度：多环境并发采样](#6-rollout-调度多环境并发采样)
   - 6.1 EnvironmentWorker — 多环境进程
   - 6.2 GroupQueueManager — 分组采样的中央调度器
   - 6.3 TrajEnvManager — 单环境轨迹管理器
7. [奖励计算与优势估计](#7-奖励计算与优势估计)
8. [PPO 训练更新](#8-ppo-训练更新)
9. [完整训练循环：14 个阶段](#9-完整训练循环14-个阶段)
10. [两种信用分配范式：StarPO vs GiGPO](#10-两种信用分配范式starpo-vs-gigpo)

---

## 1. 整体架构概览

Agentic RL 的核心是一个**多角色分布式系统**，基于 Ray 构建，包含以下关键组件：

```
┌─────────────────────────────────────────────────────────────────┐
│                     AgenticPipeline (主控)                       │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │actor_train│  │actor_infer│  │ reference│  │    critic    │    │
│  │ (训练集群) │  │ (推理集群) │  │(参考模型) │  │ (价值网络GAE) │    │
│  │Megatron/  │  │ vLLM/     │  │           │  │              │    │
│  │FSDP2      │  │ SGLang   │  │           │  │              │    │
│  └─────┬─────┘  └─────▲─────┘  └──────────┘  └──────────────┘    │
│        │  model_update  │                                        │
│        └────────────────┘                                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │            RolloutScheduler (Ray Actor)              │        │
│  │  ┌──────────────────────────────────────────────┐   │        │
│  │  │          GroupQueueManager (Ray Actor)        │   │        │
│  │  │  GroupQueue[0] ... GroupQueue[N]              │   │        │
│  │  └──────────────────────────────────────────────┘   │        │
│  └───────────────────────┬──────────────────────────┘        │
│                          │                                      │
│  ┌───────────────────────▼──────────────────────────┐        │
│  │          EnvironmentWorker (Ray Worker)            │        │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐    │        │
│  │  │EnvManager 0│ │EnvManager 1│ │EnvManager N│    │        │
│  │  │  (线程)    │ │  (线程)    │ │  (线程)    │    │        │
│  │  │    ↓       │ │    ↓       │ │    ↓       │    │        │
│  │  │  env.reset │ │  env.reset │ │  env.reset │    │        │
│  │  │  env.step  │ │  env.step  │ │  env.step  │    │        │
│  │  └────────────┘ └────────────┘ └────────────┘    │        │
│  └──────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │   XRL Sandbox 服务 (容器编排)  │
          │   ┌─────────┐ ┌─────────┐   │
          │   │Container│ │Container│   │
          │   │  env_0  │ │  env_1  │   │
          │   └─────────┘ └─────────┘   │
          └─────────────────────────────┘
```

**核心思想**：每个 RL 环境实例运行在一个独立的 Docker 容器中，由 XRL Sandbox 服务统一管理和隔离。训练集群负责模型参数更新，推理集群负责 rollout 时的 LLM 生成，环境集群负责运行智能体交互的沙箱。

---

## 2. 容器化环境隔离机制

### 2.1 为什么需要容器化隔离

Agentic RL 中的智能体需要与真实环境交互（如执行 Shell 命令、编辑代码、操作终端）。这些操作：
- **有副作用**：修改文件系统、安装包、运行脚本
- **需要隔离**：不同 episode 的环境不能互相干扰
- **需要可复现**：相同 seed 应产生相同初始状态
- **需要安全**：防止智能体执行危险操作影响宿主机

因此，ROLL 采用**一容器一环境**的隔离模型：每个环境实例拥有独立的 Docker 容器，提供完整的文件系统和进程隔离。

### 2.2 XRL Sandbox 服务

ROLL 不直接调用 Docker CLI 或 Kubernetes API，而是通过 **XRL Sandbox SDK**（`rock.sdk.sandbox.client.Sandbox`）与内部的容器编排服务交互。该服务底层基于专用 Kubernetes 集群管理容器生命周期。

**架构层级**：

```
ROLL Agentic 代码
     │
     ▼
XRL Sandbox SDK (Python)
     │  SandboxConfig → Sandbox.start() → sandbox_id, host_ip
     ▼
XRL Sandbox 服务 (REST API)
     │  接收容器创建请求，调度到 K8s 集群
     ▼
Kubernetes 集群 (专用)
     │  创建 Pod、拉取镜像、启动容器
     ▼
Docker 容器 (每个 env 一个)
     │  bash 会话: agent / test / model_service / proxy
```

### 2.3 环境注册表 (gem)

所有环境类型通过 `gem` 注册表统一管理（类似 `gym.make`）：

```python
# roll/pipeline/agentic/env/__init__.py
gem.register("cli",              entry_point="...cli_env.env:CLIEnv")
gem.register("swe_native_env",   entry_point="...swe_env.swe_native_env:SWENativeEnv")
gem.register("tb_native_env",    entry_point="...terminal_env.terminal_native_env:TerminalNativeEnv")
gem.register("rock_tb_native_env", entry_point="...terminal_env.rock_tb_native_env:RockTBNativeEnv")
gem.register("sokoban_sandbox",  entry_point="...sandbox:SokobanSandboxEnv")
# ... 以及 alfworld, sokoban, frozen_lake 等非容器环境
```

使用容器化隔离的环境类型包括：
- `cli` — 通用 CLI 环境
- `swe_native_env` — SWE-Bench 代码修复环境
- `tb_native_env` / `rock_tb_native_env` — Terminal-Bench 终端环境
- `sokoban_sandbox` — 推箱子游戏沙箱

---

## 3. env.reset()：环境初始化全流程

### 3.1 调用链路

```
TrajEnvManager.reset()
    └── env.reset(seed)                    # Gym 标准接口
         └── SandboxManager.start_sandbox()  # 启动容器
              └── Sandbox(SandboxConfig).start()  # XRL SDK 调用
                   └── XRL 服务 → K8s → 拉取镜像 + 启动容器
         └── SandboxManager.create_session()  # 创建 bash 会话
         └── (可选) setup_swe_env()           # 环境内安装依赖
         └── (可选) reset_agent_status()      # 启动容器内 Agent 服务
```

### 3.2 TrajEnvManager.reset() — 管理器层

`roll/pipeline/agentic/env_manager/traj_env_manager.py:145-188`

```python
def reset(self) -> RolloutCache:
    # 1. 创建本轮 rollout 缓存
    self.rollout_cache = RolloutCache(
        env_id=self.env_config['env_id'],
        group_id=self.env_config['group_id'],
        tag=self.env_config['tag']
    )

    # 2. 从 GroupQueueManager 获取下一个 episode_id
    self.episode_id = ray.get(self.output_queue.get_episode_id.remote(
        self.env_config['group_id'],
        self.env_config['env_id']
    ))
    if self.episode_id is None:  # 数据收集完成
        return None

    # 3. 计算 seed（保证同组环境使用相同 seed → 相同初始状态）
    seed = self.group_seed + self.episode_id

    # 4. 加锁调用 env.reset —— 这是容器启动的入口
    with self.thread_lock, self.env_step_limiter:
        observation, info = self.env.reset(seed=seed)
        if observation is None:
            return None

    # 5. 记录初始观测到历史
    self.rollout_cache.history.append({
        "observation": observation,
        "actions_left": self.env_config.max_steps - self.rollout_cache.step,
        "messages": None,
        **info,
    })
    return self.rollout_cache
```

**关键设计**：
- `group_seed` 在同组环境间共享（配置时由 `random.randint(0, 2^31-1)` 生成），保证同组环境初始状态一致——这是 GRPO/GiGPO 方差消除的基础
- `thread_lock` 和 `env_step_limiter` 控制并发，防止过多容器同时启动导致资源耗尽

### 3.3 SWENativeEnv.reset() — 环境层（以 SWE-Bench 为例）

`roll/pipeline/agentic/env/swe_env/swe_native_env.py:221-249`

```python
def reset(self, seed=None) -> Tuple[List[Dict], Dict]:
    super().reset(seed)
    self.clean_record()

    # 1. 从数据集中获取一条数据（包含 docker_image 名称）
    data_line = ray.get(self.dataset.get_data_item.remote(seed=seed))
    self.docker_image = data_line["docker_image"].replace("rex-registry-vpc", "rex-registry")

    # 2. 启动容器（拉取镜像 + 创建 Pod + 启动容器）
    self.start_sandbox()

    # 3. 在容器内安装环境依赖（上传脚本并执行）
    self.setup_swe_env()

    # 4. 获取问题描述
    self.problem_statement, self.issue = self.get_instruction()

    # 5. 启动容器内的 Agent 服务（iflow-cli / rock model-service）
    observation, tools, error_msg = self.reset_agent_status(prompt=self.problem_statement)

    return observation, {"tools": tools, "error_msg": error_msg}
```

### 3.4 SandboxManager.start_sandbox() — 容器启动核心

`roll/pipeline/agentic/env/rock/sanbox_manager.py:327-362`

```python
def start_sandbox(self, max_retry=3, backoff=20.0):
    """启动一个沙箱容器实例"""
    # 1. 构建容器配置
    config = SandboxConfig(
        image=self.image_id,               # Docker 镜像名（如 "rex-registry/swebench:xxx"）
        xrl_authorization=self.xrl_authorization,  # 认证 token
        auto_clear_seconds=self.auto_clear_seconds,  # 容器 TTL（默认 3600 秒后自动回收）
        startup_timeout=self.startup_timeout,       # 启动超时（默认 600 秒）
        user_id=self.user_id,              # 用户标识（配额管理）
        experiment_id=self.experiment_id,  # 实验标识
        cluster="zb-b",                    # 目标 K8s 集群（zb-b / sg / nt-a 等）
    )

    # 2. 创建 Sandbox 对象并启动 —— SDK 内部调用 XRL 服务
    sandbox = Sandbox(config)
    asyncio.run(sandbox.start())  # ← 这里触发：镜像拉取 → Pod 创建 → 容器启动

    # 3. 获取容器信息
    self.sandbox = sandbox
    self.sandbox_id = sandbox.sandbox_id   # 容器唯一 ID
    return True, sandbox.host_ip           # 容器可访问的 IP
```

**容器启动重试机制** (`_initialize_sandbox_with_times`, L199-231)：
- 最多重试 3 次
- 退避时间递增：`120 * attempt` 秒（第 1 次等 120s，第 2 次等 240s）
- 如果 AgentNativeStepEnvManager 层重试仍失败，升级到长达 1 小时的退避

### 3.5 create_session() — 容器内会话创建

容器启动后，需要在其内部创建持久的 bash 会话：

```python
def create_session(self, session="agent"):
    asyncio.run(
        self.sandbox.create_session(
            CreateBashSessionRequest(
                session=session,           # 会话名: "agent" / "test" / "model_service"
                startup_source=["/root/.bashrc"],
                env_enable=True,
                env={
                    "HOME": "/root",
                    "IFLOW_ENV": "train",
                    "HF_ENDPOINT": "https://hf-mirror.com"
                }
            )
        )
    )
```

一个容器内通常创建多个会话，各司其职：

| 会话名 | 用途 |
|--------|------|
| `agent` | 运行智能体（iflow-cli / swe-agent），接收 LLM 输出并执行 |
| `test` | 运行测试脚本（如 SWE-Bench 的 `run_tests.sh`），计算 reward |
| `model_service` | 运行 `rock model-service`，处理 LLM ↔ Agent 的消息转换 |
| `proxy` | 代理会话，用于 Harbor 模式的 LLM 回调 |

### 3.6 SokobanSandboxEnv.reset() — 简洁参考实现

`roll/pipeline/agentic/env/sandbox/sokoban_sandbox_env.py:74-96` 是最清晰的容器环境示例：

```python
def reset(self, seed=None, **kwargs):
    """重置环境到新游戏"""
    self._lazy_init()  # 懒初始化：首次调用时启动容器

    # 在容器内执行 reset 命令
    command = f"python client.py reset --seed {seed} --json"
    result = self._execute_command(command)  # 通过 bash 会话执行

    if isinstance(result, dict):
        obs = result.get('observation')
        info = result.get('info', {})
        return obs, info
```

其中 `_lazy_init()` (L201-285) 完成完整的容器启动流程：

```python
def _lazy_init(self):
    # 1. 创建 Sandbox 配置（指定镜像、TTL、超时）
    self.sandbox_config = SandboxConfig(
        base_url="http://localhost:8080",
        image='rock-n-roll-registry.cn-hangzhou.cr.aliyuncs.com/rock/sokoban-sandbox:latest',
        auto_clear_seconds=60 * 120,  # 2 小时 TTL
        startup_timeout=360,
    )

    # 2. 启动容器（3 次重试）
    for attempt in range(3):
        self.sandbox = Sandbox(self.sandbox_config)
        asyncio.run(self.sandbox.start())        # ← K8s 拉取镜像 + 启动 Pod
        asyncio.run(self.sandbox.is_alive())      # 等待容器就绪
        # ... 创建 default 和 server_session 会话

    # 3. 在容器内启动游戏服务器
    asyncio.run(self.sandbox.run_in_session(
        BashAction(session="server_session",
                   command="nohup python server.py --port 8001 &")
    ))

    # 4. 健康检查（轮询直到服务器就绪）
    self._wait_for_server(timeout=60)
```

---

## 4. env.step()：智能体动作执行

### 4.1 TrajEnvManager.step() — 管理器层

`roll/pipeline/agentic/env_manager/traj_env_manager.py:190-217`

```python
def step(self, llm_output: DataProto):
    # 1. 将 LLM 生成的 token_ids 解码为文本
    responses = self.tokenizer.batch_decode(
        llm_output.batch['responses'], skip_special_tokens=False
    )

    # 2. 加锁调用 env.step —— 这是动作执行的入口
    with self.thread_lock, self.env_step_limiter:
        observation, reward, terminated, truncated, info = \
            self.env.step(action=responses[0])

    # 3. 更新 rollout 缓存
    self.rollout_cache.step += 1
    self.rollout_cache.terminated = terminated
    self.rollout_cache.truncated = truncated

    # 4. 步数上限检查
    if self.rollout_cache.step >= self.env_config.max_steps:
        self.rollout_cache.terminated = True
        if not terminated:
            self.rollout_cache.truncated = True

    # 5. 记录奖励和 LLM 响应到历史
    self.rollout_cache.history[-1]['reward'] = reward
    self.rollout_cache.history[-1]['llm_response'] = responses[0]
    self.rollout_cache.history[-1].update(info)

    # 6. 追加新的观测（作为下一步的输入）
    self.rollout_cache.history.append({
        "observation": observation,
        "actions_left": self.env_config.max_steps - self.rollout_cache.step,
        "messages": None
    })

    return self.rollout_cache
```

### 4.2 SWENativeEnv.step() — 在容器内执行智能体动作

`roll/pipeline/agentic/env/swe_env/swe_native_env.py:251-321`

```python
def step(self, action: str):
    """执行智能体的一个动作

    Args:
        action: LLM 生成的文本响应（可能包含 tool_call 和 content）
    """
    self.current_step += 1
    self.current_session_step += 1

    # 1. 超时检查
    if self.rollout_time > self.max_env_time:
        action = EpisodeStopReason.ENV_TIMEOUT

    # 2. 控制类动作（达到最大长度或超时）→ 终止并计算 reward
    if isinstance(action, EpisodeStopReason) and action in [...]:
        observation, tools, error_msg, self.reward = \
            self.check_terminated(next_request_payload="", force_terminated=True)
        return observation, self.reward, True, True, {}

    # 3. 将 LLM 响应格式化为 Agent 可理解的消息
    response_payload, info = \
        self.sandbox_manager.format_response_payload(response=action)

    # 4. ★ 核心：将动作发送到容器内的 Agent 服务 ★
    request_response: RunSessionResponse = \
        self.sandbox_manager.fetch_agent_request(
            index=self.current_session_step,
            response_payload=response_payload
        )

    # 5. 解析 Agent 的下一步请求
    if request_response.exit_code != 0:
        # 交互失败，标记错误
        self.env_reset_failed = True
        next_request_payload = "SESSION_END"
    else:
        next_request_payload = request_response.output

    # 6. 检查是否终止（Agent 输出 SESSION_END 表示任务完成）
    observation, tools, error_msg, self.reward = \
        self.check_terminated(next_request_payload=next_request_payload)

    # 7. 返回 Gym 标准五元组
    return observation, self.reward, self.terminated, self.truncated, info
```

### 4.3 fetch_agent_request() — 容器内动作执行的核心

`roll/pipeline/agentic/env/rock/sanbox_manager.py:1628-1750`

这是将智能体动作真正送入容器的关键函数：

```python
def fetch_agent_request(self, index, response_payload):
    """在容器的 model_service 会话中执行 anti-call-llm 脚本

    "Anti-call-LLM" 模式：
    - 容器内的 Agent 服务 (iflow-cli) 原本会调用外部 LLM
    - ROLL 通过 "anti-call" 拦截这个调用，将 ROLL 的 LLM 输出注入回去
    - Agent 收到 LLM 响应后继续执行，产生下一个请求
    """
    cmd = rock_setup_config.model_service_anti_call_llm_script.format(
        index=index,
        response_payload=shlex.quote(response_payload)
    )
    # 在容器的 model_service 会话中执行
    response_obj = self.run_script_with_timeout(
        session_name=self.proxy_session_name,
        script_text=cmd,
        ...
    )
    # 解析输出，获取 Agent 的下一步请求（messages + tools）
    return response_obj
```

### 4.4 run_in_session() — 容器内命令执行

`roll/pipeline/agentic/env/rock/sanbox_manager.py:382-472`

```python
def run_in_session(self, command, session, max_retry=3, backoff=10.0, timeout=None):
    """在容器的指定 bash 会话中执行命令"""
    for attempt in range(1, max_retry + 1):
        try:
            # 通过 XRL SDK 向容器发送 BashAction
            result = asyncio.run(
                asyncio.wait_for(
                    self.sandbox.run_in_session(
                        BashAction(
                            session=session,      # 目标会话
                            command=command,      # 要执行的命令
                            check="silent"
                        )
                    ),
                    timeout=timeout
                )
            )
            return result  # BashObservation(output, exit_code, failure_reason)
        except asyncio.TimeoutError:
            # 超时重试
            ...
        except Exception as exc:
            # 异常重试
            ...
```

### 4.5 SokobanSandboxEnv.step() — 简洁参考

```python
def step(self, action: str):
    """在容器内执行一步动作"""
    sanitized_action = action.replace('\n', ' ').replace('\r', '')
    command = f"python client.py action {shlex.quote(sanitized_action)} --json"
    result = self._execute_command(command)  # 在容器内执行

    if isinstance(result, dict):
        return (
            result.get('observation'),
            result.get('reward', 0),
            result.get('terminated', True),
            result.get('truncated', False),
            result.get('info', {})
        )
```

### 4.6 动作执行的数据流

```
LLM 生成 tokens
    │
    ▼
tokenizer.batch_decode → action (文本)
    │
    ▼
env.step(action)
    │
    ├── format_response_payload(action) → JSON payload
    │       将 LLM 文本包装为 OpenAI 格式的 {"choices":[{"message":{...}}]}
    │
    ├── fetch_agent_request(payload)
    │       在容器 model_service 会话中执行 anti-call-llm 脚本
    │       将 LLM 响应注入容器内 Agent 的请求流
    │
    ├── Agent 在容器内执行动作
    │       例如：编辑文件、运行测试、执行 Shell 命令
    │
    ├── Agent 产生下一步请求
    │       输出 messages + tools（下一个 LLM 输入）
    │
    └── check_terminated()
            判断是否任务完成 (SESSION_END)
            如果完成 → 运行测试脚本计算 reward
```

---

## 5. Kubernetes 集群的容器管理与隔离

### 5.1 专用 K8s 集群架构

ROLL 的容器化环境由**专用 Kubernetes 集群**管理，通过 XRL Sandbox 服务层进行调度：

```
┌─────────────── ROLL 训练集群 ──────────────────┐
│                                                │
│  EnvironmentWorker (Ray Worker 进程)            │
│  ├── EnvManager 0 → env_0.reset()              │
│  ├── EnvManager 1 → env_1.reset()              │
│  └── EnvManager N → env_N.reset()              │
│         │                                      │
│         │  SandboxConfig(image, cluster, ...)  │
│         ▼                                      │
│  ┌─── XRL Sandbox SDK ───────────────────┐    │
│  │  Sandbox.start() → POST /api/sandbox   │    │
│  │  Sandbox.run_in_session() → POST /api/  │    │
│  │  Sandbox.stop() → DELETE /api/sandbox   │    │
│  └──────────────────┬─────────────────────┘    │
│                     │                          │
└─────────────────────┼──────────────────────────┘
                      │  HTTPS REST API
                      ▼
┌─── XRL Sandbox 服务 (控制面) ──────────────────┐
│                                                │
│  1. 接收容器创建请求                            │
│  2. 鉴权 (xrl_authorization)                    │
│  3. 选择目标 K8s 集群 (zb-b / sg / nt-a)       │
│  4. 生成 sandbox_id, 分配 host_ip              │
│  5. 调度到 K8s API Server                       │
│                                                │
└────────────────────┬───────────────────────────┘
                     │  Kubernetes API
                     ▼
┌─── 专用 Kubernetes 集群 (数据面) ───────────────┐
│                                                │
│  ┌────────── Pod: sandbox-abc123 ──────────┐  │
│  │  Container: swebench-python-abc         │  │
│  │  ├── bash session: agent                │  │
│  │  ├── bash session: test                 │  │
│  │  ├── bash session: model_service        │  │
│  │  └── 完整文件系统 (git repo, conda env)  │  │
│  │  auto_clear: 3600s 后自动回收            │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ┌────────── Pod: sandbox-def456 ──────────┐  │
│  │  Container: sokoban-sandbox-xyz         │  │
│  │  ├── python server.py (游戏服务器)       │  │
│  │  └── bash session: default              │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ┌────────── Pod: sandbox-ghi789 ──────────┐  │
│  │  Container: terminal-bench-uvw          │  │
│  │  └── ...                                │  │
│  └──────────────────────────────────────────┘  │
│                                                │
└────────────────────────────────────────────────┘
```

### 5.2 隔离保证

| 隔离维度 | 实现方式 |
|---------|---------|
| **进程隔离** | 每个环境运行在独立的 Docker 容器中，进程命名空间隔离 |
| **文件系统隔离** | 每个容器拥有独立的可写层，互不干扰 |
| **网络隔离** | 每个容器分配独立的 host_ip，通过 K8s 网络策略隔离 |
| **资源隔离** | K8s Pod 的 resource requests/limits 控制 CPU/内存 |
| **用户隔离** | 容器内以 root 运行，但与宿主机用户命名空间隔离 |
| **时间隔离** | `auto_clear_seconds` (默认 3600s) 后自动回收容器 |

### 5.3 生命周期管理

```
创建 ──────────────────────────────────────── 回收
  │                                            ▲
  ▼                                            │
env.reset()                                auto_clear_seconds 到期
  │                                            │
  ├── SandboxConfig(image, cluster, TTL)       │
  ├── Sandbox.start()                          │
  │     → XRL 服务调度到 K8s                    │
  │     → K8s 拉取 Docker 镜像                  │
  │     → 创建 Pod + 启动容器                   │
  │     → 返回 sandbox_id + host_ip           │
  │                                            │
  ├── create_session("agent")                  │
  ├── create_session("test")                  │
  ├── create_session("model_service")          │
  │                                            │
  │  ── 多轮交互 ──                            │
  │                                            │
  ├── env.step(action_1)                      │
  │     → run_in_session(command)              │
  ├── env.step(action_2)                      │
  │     → run_in_session(command)              │
  ├── ...                                      │
  ├── env.step(action_N)                      │
  │     → 任务完成 or 步数上限                   │
  │                                            │
  └── env.close()                              │
        → Sandbox.stop()                       │
          → 主动销毁容器                         │
          OR                                    │
        TTL 到期 → XRL 服务自动回收 ─────────────┘
```

### 5.4 集群选择与多区域部署

```python
# 根据运行区域选择 K8s 集群
if self.run_region == "sg":    # 新加坡
    cluster = "sg"
else:                          # 默认国内
    cluster = "zb-b"
```

支持的集群：
- `zb-b` — 国内主集群
- `sg` — 新加坡集群
- `nt-a` — 备用集群
- `vpc-sg-sl-a` — VPC 新加坡

### 5.5 容器镜像管理

容器镜像从阿里云容器镜像服务 (ACR) 拉取：

| 环境类型 | 镜像来源 |
|---------|---------|
| SWE-Bench | `rex-registry/.../swebench:<hash>` (从数据集获取) |
| Sokoban | `rock-n-roll-registry.cn-hangzhou.cr.aliyuncs.com/rock/sokoban-sandbox:latest` |
| CLI | `hub.docker.alibaba-inc.com/chatos/iflow-cli:4.0` |
| Terminal-Bench | 配置文件指定 |

镜像名中的 `-vpc` 后缀会被替换为公共端点：
```python
self.docker_image = data_line["docker_image"].replace("rex-registry-vpc", "rex-registry")
```

### 5.6 认证与配额

```python
SandboxConfig(
    xrl_authorization="t-r8c4rjh0por8gwc5",  # 认证 token
    user_id="337866",                        # 用户 ID（配额跟踪）
    experiment_id="test_swe",                # 实验标识
)
```

XRL 服务通过这些字段进行：
- **认证**：验证请求合法性
- **配额管理**：限制每个用户的并发容器数
- **计费归因**：追踪资源消耗到具体实验

---

## 6. Rollout 调度：多环境并发采样

### 6.1 EnvironmentWorker — 多环境进程

`roll/pipeline/agentic/environment_worker.py:24-167`

每个 EnvironmentWorker 是一个 Ray Worker 进程，持有多个 EnvManager（线程级并发）：

```python
class EnvironmentWorker(Worker):
    def initialize(self):
        # 为每个 env_id 创建一个 EnvManager
        for env_id, env_config in self.env_configs[rank].items():
            self.env_managers[env_id] = create_env_manager(env_id, env_config)

    def run_rollout_loop(self, seed):
        # 所有 EnvManager 并发运行（每个一个线程）
        loop = asyncio.new_event_loop()
        tasks = [
            loop.run_in_executor(
                self.thread_pool,
                env_manager.run_rollout_loop,
                DataProto(meta_info={"seed": seed})
            )
            for env_manager in self.env_managers.values()
        ]
        asyncio.gather(*tasks)  # 等待全部完成
```

### 6.2 GroupQueueManager — 分组采样的中央调度器

`roll/distributed/scheduler/rollout_scheduler.py:362-540`

#### 6.2.1 它是做什么的

`GroupQueueManager` 是一个 **Ray Remote Actor**，是整个 rollout 采样的**中央调度器**。它解决一个核心问题：**如何在数百个并发环境中，保证同一组（group）的环境都使用相同的 seed 运行，并且只有当同组所有环境都完成后才一起释放数据？**

这个分组保证是 GRPO/GiGPO 等组内相对优势估计的基础——只有同组轨迹面对相同初始状态，才能通过 `(score - group_mean) / group_std` 消除环境随机性、分离出策略本身的贡献。

#### 6.2.2 核心数据结构

```python
@ray.remote
class GroupQueueManager:
    """管理所有 group 的中央调度器"""

    def __init__(self, config, env_manager_config, mode):
        self.mode = mode                    # "train" 或 "val"
        self.group_size = env_manager_config.group_size  # 每组环境数（如 8）
        self.async_generation_ratio = config.async_generation_ratio  # 异步比例
        self.max_traj_per_env = ...         # 每个 env 最多跑多少 episode

        # ★ 核心：每个 group_id 对应一个独立的 GroupQueue
        self.group_queue: Dict[int, GroupQueue] = {}
        for env_config in all_envs:
            group_id = env_config["group_id"]
            if group_id not in self.group_queue:
                self.group_queue[group_id] = GroupQueue(
                    group_id=group_id,
                    group_size=self.group_size,
                    group_size_redundancy=env_manager_config.group_size_redundancy,
                    max_traj_per_env=self.max_traj_per_env,
                    async_generation_ratio=self.async_generation_ratio,
                    group_filter=group_filter,
                )
```

**GroupQueue** (`rollout_scheduler.py:217-360`) 是单个组内的状态机：

```python
@dataclass
class GroupData:
    """一个 episode 的组数据容器"""
    group_id: int
    episode_id: int           # 第几个 episode（从 0 递增）
    create_step: int          # 在哪个训练 step 创建的（用于异步过期判断）
    rollouts: List[DataProto] = []    # 已完成的 rollout 数据
    running_rollouts: int = 0        # 正在运行的 env 数量

class GroupQueue:
    """单个 group 的 episode 生命周期管理"""

    def __init__(self, group_id, group_size, ...):
        self.group_id = group_id
        self.group_size = group_size               # 需要多少条轨迹才完成
        self.group_size_redundancy = ...           # 允许的冗余（容错）
        self.max_traj_per_env = ...                # 每个 env 最多跑几个 episode
        self.async_generation_ratio = ...          # 异步训练的提前生成比例

        self.groups: Dict[int, GroupData] = {}    # episode_id → GroupData
        self.next_episode_id = 0                   # 下一个待创建的 episode_id
        self.progress = asyncio.Event()           # 有新 episode 可用时 set
        self.complete = asyncio.Event()           # 一个 group 收满时 set
        self.quit = False                          # 是否关闭
```

#### 6.2.3 Episode 生命周期

一个 episode 从创建到释放的完整生命周期：

```
advance_step(global_step=k)
    │
    ▼
GroupQueue.advance_group(create_step=k)
    │  创建 GroupData(episode_id=N, create_step=k)
    │  next_episode_id += 1
    ▼
EnvManager 调用 get_episode_id(group_id, env_id)
    │  遍历 self.groups，找到第一个 running_rollouts < group_size 的 episode
    │  running_rollouts += 1
    │  返回 episode_id
    ▼
EnvManager 执行 rollout（reset → step × N → formulate_rollouts）
    │
    ▼
EnvManager 调用 put(group_id, episode_id, rollout)
    │  group.rollouts.append(rollout)
    │
    ├── len(group.rollouts) < group_size → 等待其他 env 完成
    │
    └── len(group.rollouts) == group_size → ★ 组完成 ★
        │
        ├── group_filter.filter(...) → True? 丢弃，重新创建新 episode
        │
        └── group_filter.filter(...) → False? complete.set()
            │  通知 get_batch 可以取走了
            ▼
GroupQueueManager.get_batch 取走该组数据
    │  groups.pop(episode_id)  → 释放
    │  env_monitor.cleanup_episode(...)
```

#### 6.2.4 三大核心方法

**1. `get_episode_id` — 分配工作给环境**

```python
async def get_episode_id(self, group_id, env_id=None) -> Optional[int]:
    """环境调用此方法获取下一个要跑的 episode_id"""
    while not self.quit:
        for episode_id, group in self.groups.items():
            # 找到一个还没被 group_size + redundancy 个 env 占满的 episode
            if group.running_rollouts < self.group_size + self.group_size_redundancy:
                group.running_rollouts += 1
                return episode_id

        # 没有可用的 episode，等待 advance_step 创建新的
        self.progress.clear()
        await self.progress.wait()

    return None  # 关闭信号
```

**2. `put` — 环境提交 rollout 数据**

```python
def put(self, group_id, episode_id, start_step, rollout, env_id=None):
    """环境完成一个 episode 后调用"""
    group = self.groups[episode_id]
    group.rollouts.append(rollout)

    if len(group.rollouts) == self.group_size:
        # ★ 组完成：所有同组环境都跑完了 ★
        if all(r is None for r in group.rollouts):
            # 全部是 None → 该 env 组退出
            self.complete.set()
        elif self.group_filter.filter(...):
            # 组过滤器拒绝 → 丢弃这组，重新创建
            self.groups.pop(episode_id)
            self.advance_group(create_step=self.current_step)
        else:
            # 正常完成 → 通知 get_batch
            self.complete.set()
```

**3. `get_batch` — 收集足够多的已完成组**

```python
async def get_batch(self, batch_size, current_step) -> List[DataProto]:
    """Pipeline 调用此方法收集一个 training batch"""
    ret = []
    while len(ret) < batch_size:
        # 为每个未完成的 group 创建 asyncio task
        if not self.pending_gets:
            pending = {
                asyncio.create_task(group_queue.get())
                for group_id, group_queue in self.group_queue.items()
                if str(group_id) not in self.rollout_complete
            }

        # 等待任意一个 group 完成
        done, pending = await asyncio.wait(pending, return_when=FIRST_COMPLETED)

        for task in done:
            group = await task
            # 异步过期检查：丢弃太旧的 rollout
            if current_step - group.create_step > self.async_generation_ratio:
                continue  # 这组数据太旧了（异步训练中策略已更新太多）
            ret.extend(group.rollouts)

    return ret[:batch_size]
```

#### 6.2.5 异步训练支持

在异步训练模式下，`async_generation_ratio` 控制提前生成多少 step 的数据：

```python
def advance_step(self, step):
    if self.current_step is None:
        # 首次：提前生成 async_generation_ratio 个 step 的 episode
        for _ in range(self.async_generation_ratio):
            self._advance_step(step)
    else:
        # 清理过期 episode（超过 async_generation_ratio 步的）
        for episode_id, group in list(self.groups.items()):
            if step - group.create_step > self.async_generation_ratio:
                self.groups.pop(episode_id)  # 丢弃过期数据
        self._advance_step(step)
```

这允许 rollout 与训练并行：当 step k 正在训练时，step k+1、k+2 的 rollout 已经在后台进行了。

#### 6.2.6 组过滤 (Group Filter)

`group_filter` 可以基于自定义逻辑拒绝某些组（如所有轨迹得分相同、全部失败等），被拒绝的组会被丢弃并重新生成新的 episode：

```python
class GroupFilter:
    def filter(self, group_id, episode_id, group) -> bool:
        """返回 True 表示丢弃这组"""
        # 默认实现：永远不丢弃
        return False
```

#### 6.2.7 分组机制图解

```
GroupQueueManager
├── GroupQueue[group_id=0]  (seed_seed=42)
│   ├── Episode 0: [env_0✓, env_1✓, env_2✓, ... env_7✓] → 完成 → 释放给 get_batch
│   ├── Episode 1: [env_0✓, env_1⏳, env_2  , ... env_7  ] → 等待中
│   └── Episode 2: (尚未有 env 开始)
│
├── GroupQueue[group_id=1]  (seed_seed=99)
│   ├── Episode 0: [env_8✓, env_9✓, ... env_15✓] → 完成 → 释放给 get_batch
│   └── Episode 1: [env_8⏳, env_9  , ... env_15  ] → 等待中
│
└── GroupQueue[group_id=N] ...
```

**关键不变量**：一个 Group 只有在所有 `group_size` 个环境都完成 rollout 后才会被释放给 `get_batch`，保证组内方差估计的完整性。

---

### 6.3 TrajEnvManager — 单环境轨迹管理器

`roll/pipeline/agentic/env_manager/traj_env_manager.py`

#### 6.3.1 它是做什么的

`TrajEnvManager` 是**单个环境实例的完整生命周期管理者**。它封装了一个 `gem.Env`（如 Sokoban、SWE-Bench），负责：

1. **管理一个完整 episode 的交互循环**：reset → (make_decision → step) × N → formulate_rollouts
2. **桥接 LLM 推理与环境交互**：将对话历史格式化为 LLM 输入，将 LLM 输出转换为环境动作
3. **组装训练数据**：将多轮交互的 token 序列、奖励、mask 打包成 `DataProto`
4. **与 GroupQueueManager 通信**：获取 episode_id、提交 rollout 数据

它是 StarPO（轨迹级信用分配）的默认 EnvManager。GiGPO 使用其子类 `StepEnvManager`。

#### 6.3.2 构造函数

```python
class TrajEnvManager(BaseEnvManager):

    def __init__(self, worker_config, pipeline_config, env_config,
                 tokenizer, generate_scheduler, output_queue, thread_lock, mode):
        # 1. 存储配置
        self.env_config = env_config
        self.pipeline_config = pipeline_config
        self.tokenizer = tokenizer
        self.output_queue = output_queue              # → GroupQueueManager (Ray Actor)
        self.generate_scheduler = generate_scheduler   # → RouterManager (推理路由)
        self.mode = mode                               # "train" 或 "val"

        # 2. 线程安全控制
        self.use_thread_lock = env_config.get("use_thread_lock", False)
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()

        # 3. 并发限制器（防止大量容器同时启动导致资源耗尽）
        self.max_env_step_concurrent = env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=...)

        # 4. ★ 创建环境实例 ★
        self.env = gem.make(env_id=env_config["env_type"], **env_config['config'])
        # 例如: gem.make("sokoban_sandbox", ...) → SokobanSandboxEnv 实例
        # 该实例内部持有 XRL Sandbox SDK，通过容器化方式运行

        # 5. 可选：工具包装器（Tool Use 场景）
        if "tool_wrapper" in env_config:
            self.env = tool_wrapper(self.env, tool_configs=...)

        # 6. 创建 LLM 代理（通过推理集群生成动作）
        self.llm_proxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            tokenizer=self.tokenizer,
            env=self.env
        )

        # 7. 模板配置（prompt 构建）
        self.agent_system_template = custom_envs[tag]["agent_system_template"]
        self.agent_template = custom_envs[tag]["agent_template"]
```

#### 6.3.3 核心交互循环 run_rollout_loop()

这是每个环境线程的主循环，持续运行直到收到停止信号：

```python
def run_rollout_loop(self, data: DataProto):
    """持续执行 episode 直到数据收集完成"""

    self.running = True
    # ★ Seed 计算：保证同组环境使用相同 seed ★
    # group_seed = base_seed(来自 pipeline) + group_seed(来自 config)
    self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']

    # 首次 reset → 启动容器、获取初始观测
    rollout_cache = self.reset()
    start_step = self.current_step

    while self.running and rollout_cache is not None:

        # ━━━━━━━━━━ Phase 1: LLM 生成决策 ━━━━━━━━━━
        lm_output = self.make_decision(rollout_cache)
        # → 调用推理集群 (vLLM/SGLang) 生成 token_ids
        # → 返回 DataProto(responses, infer_logprobs, stop_reason)

        stop_reason = lm_output.meta_info.pop("stop_reason")

        # ━━━━━━━━━━ Phase 2: 环境执行动作 ━━━━━━━━━━
        if stop_reason == GenerateStopReason.FINISH:
            rollout_cache = self.step(lm_output)
            # → 将 LLM 输出解码为文本
            # → 调用 env.step(action) 在容器内执行
            # → 返回 (observation, reward, terminated, truncated, info)

        # ━━━━━━━━━━ Phase 3: Episode 结束处理 ━━━━━━━━━━
        if rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH:
            # 3a. 组装训练数据（一条轨迹 → 一个训练样本）
            rollout = self.formulate_rollouts(rollout_cache)

            # 3b. 附加分组标识（用于后续的优势归一化）
            traj_group_id = f"{tag}_{group_id}_{episode_id}_{group_seed}"
            traj_id = f"{traj_group_id}_{env_id}"
            rollout.non_tensor_batch["traj_group_id"] = [traj_group_id]
            rollout.non_tensor_batch["traj_id"] = [traj_id]

            # 3c. ★ 提交给 GroupQueueManager ★
            ray.get(self.output_queue.put.remote(
                group_id, episode_id, start_step, rollout, env_id
            ))

            # 3d. 开始下一个 episode（新容器）
            rollout_cache = self.reset()
            start_step = self.current_step

    # 发送结束哨兵（None 表示该 env 退出）
    ray.get(self.output_queue.put.remote(
        group_id, episode_id, start_step, None, env_id
    ))
```

#### 6.3.4 reset() — 启动新 Episode

```python
def reset(self) -> RolloutCache:
    # 1. 创建新的 rollout 缓存
    self.rollout_cache = RolloutCache(
        env_id=self.env_config['env_id'],
        group_id=self.env_config['group_id'],
        tag=self.env_config['tag']
    )

    # 2. ★ 向 GroupQueueManager 请求下一个 episode_id ★
    self.episode_id = ray.get(
        self.output_queue.get_episode_id.remote(group_id, env_id)
    )
    if self.episode_id is None:
        # None 表示 GroupQueueManager 已关闭，停止运行
        return None

    # 3. 计算 episode seed
    # ★ group_seed + episode_id → 保证同组不同 episode 有不同但确定的 seed ★
    seed = self.group_seed + self.episode_id

    # 4. 加锁 + 并发限制
    with self.thread_lock, self.env_step_limiter:
        # ★ env.reset 是容器启动的入口 ★
        # 对于容器化环境：触发 Sandbox.start() → K8s 拉取镜像 → 启动容器
        observation, info = self.env.reset(seed=seed)
        if observation is None:
            return None

    # 5. 记录初始观测到历史
    self.rollout_cache.history.append({
        "observation": observation,           # 初始游戏状态描述
        "actions_left": max_steps - step,     # 剩余步数
        "messages": None,                    # LLM 消息（下一步填充）
        **info,                               # 环境附加信息（tools, suffix 等）
    })
    return self.rollout_cache
```

**Seed 的一致性保证**：
- `group_seed` 在 `make_env_configs()` 中为每个 group 随机生成一次，同组所有 env 共享
- `episode_seed = group_seed + episode_id`：同组同 episode 的所有 env 使用完全相同的 seed → 相同初始状态
- 不同 episode 的 seed 不同 → 数据多样性

#### 6.3.5 step() — 执行智能体动作

```python
def step(self, llm_output: DataProto):
    # 1. 将 LLM 生成的 token_ids 解码为文本
    responses = self.tokenizer.batch_decode(
        llm_output.batch['responses'], skip_special_tokens=False
    )

    # 2. 加锁 + 并发限制后调用 env.step
    with self.thread_lock, self.env_step_limiter:
        # ★ env.step 是动作执行的入口 ★
        # 对于容器化环境：通过 run_in_session 在容器 bash 会话中执行
        observation, reward, terminated, truncated, info = \
            self.env.step(action=responses[0])

    # 3. 更新缓存状态
    self.rollout_cache.step += 1
    self.rollout_cache.terminated = terminated
    self.rollout_cache.truncated = truncated

    # 步数上限检查
    if self.rollout_cache.step >= self.env_config.max_steps:
        self.rollout_cache.terminated = True
        if not terminated:
            self.rollout_cache.truncated = True  # 区分"自然结束"和"被截断"

    # 4. 记录奖励和 LLM 响应到当前历史条目
    self.rollout_cache.history[-1]['reward'] = reward
    self.rollout_cache.history[-1]['llm_response'] = responses[0]
    self.rollout_cache.history[-1].update(info)

    # 5. 追加新的观测（作为下一轮 make_decision 的输入）
    self.rollout_cache.history.append({
        "observation": observation,
        "actions_left": max_steps - self.rollout_cache.step,
        "messages": None,
    })

    return self.rollout_cache
```

#### 6.3.6 make_decision() — LLM 推理

```python
def make_decision(self, rollout_cache):
    # 1. 将对话历史格式化为 LLM 输入
    lm_input = self.format_messages(rollout_cache)
    # → 构建 input_ids, attention_mask, position_ids
    # → 每轮都重新 tokenize 完整对话历史（非增量）

    # 2. 计算最大生成长度
    max_new_tokens = min(
        env_config_max_response_length,
        sequence_length - current_input_len
    )

    # 3. 调用推理集群（通过 RouterManager 路由到 vLLM/SGLang）
    lm_output = self.llm_proxy.generate(
        messages, lm_input, generation_config
    )
    # → 返回 DataProto(responses, infer_logprobs, stop_reason)

    # 4. 将生成结果存入历史
    rollout_cache.history[-1]["response_ids"] = lm_output.batch["responses"]
    rollout_cache.history[-1]["infer_logprobs"] = lm_output.batch.get("infer_logprobs")

    return lm_output
```

#### 6.3.7 format_messages() — 构建 LLM 输入

```python
def format_messages(self, rollout_cache):
    messages = []

    # 1. 第一轮：添加 system prompt
    if first_turn:
        messages.append({"role": "system", "content": agent_system_template})
        if "env_instruction" in history[0]:
            messages.append({"role": "system", "content": env_instruction})

    # 2. 遍历历史：交替添加 assistant/user 消息
    for i, entry in enumerate(history):
        if i == 0:
            # 第一轮的 observation 作为第一条 user 消息
            messages.append({"role": "user", "content": render(agent_template, observation, ...)})
        else:
            # 后续轮：先添加上一轮的 assistant 响应
            messages.append({"role": "assistant", "content": entry["llm_response"]})
            # 再添加本轮的 user 消息（新观测）
            messages.append({"role": "user", "content": render(agent_template, ...)})

    # 3. 使用 tokenizer 的 chat template 转换为 token_ids
    prompt_ids = custom_apply_chat_template(tokenizer, messages, tools=self.tools)

    # 4. 拼接历史 token_ids + 当前 prompt_ids
    input_ids = history_token_ids + prompt_ids

    return DataProto({"input_ids": input_ids, "attention_mask": ..., "position_ids": ...})
```

#### 6.3.8 formulate_rollouts() — 组装训练数据

```python
def formulate_rollouts(self, rollout_cache):
    """将一条完整轨迹转换为训练样本"""

    # 1. 丢弃最后一条（没有 response 的观测）
    history = rollout_cache.history[:-1]

    # 2. 计算轨迹总奖励
    scores = [i['reward'] for i in history]
    episode_score = sum(scores)

    # 3. 拼接所有轮次的 token 序列
    #    prompt_1 | response_1 | prompt_2 | response_2 | ... | prompt_N | response_N
    token_ids = []
    prompt_masks = []      # 1=prompt, 0=response
    response_masks = []    # 0=prompt, 1=response
    for entry in history:
        token_ids.extend(entry["prompt_ids"])
        token_ids.extend(entry["response_ids"])
        prompt_masks.extend([1]*len(prompt) + [0]*len(response))
        response_masks.extend([0]*len(prompt) + [1]*len(response))

    # 4. 构建张量
    input_ids = torch.tensor(token_ids).unsqueeze(0)
    response_mask = torch.tensor(response_masks).unsqueeze(0)  # 训练时只对 response 计算 loss

    # 5. ★ 奖励放在最后一个 token 位置 ★
    score_tensor = torch.zeros(len(token_ids))
    score_tensor[-1] = episode_score   # 轨迹级奖励

    # 6. Pad 到 sequence_length
    input_ids = pad_to_length(input_ids, sequence_length, pad_id)
    response_mask = pad_to_length(response_mask, sequence_length, 0)

    # 7. 附加非张量元数据
    return DataProto(
        batch={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,      # 哪些 token 是 response（训练目标）
            "prompt_mask": prompt_mask,          # 哪些 token 是 prompt
            "scores": score_tensor,              # 奖励信号
        },
        non_tensor_batch={
            "env_ids": env_id,                   # 环境标识
            "group_ids": group_id,               # 组标识（归一化用）
            "tags": tag,                         # 环境类型标签
            "step_scores": scores,               # 每步奖励列表
            "episode_scores": episode_score,     # 轨迹总奖励
        }
    )
```

#### 6.3.9 TrajEnvManager 与 GroupQueueManager 的协作关系

```
                    GroupQueueManager (Ray Actor)
                    ┌─────────────────────────────────┐
                    │  GroupQueue[0]  GroupQueue[1]  │
                    │  ┌──────────┐  ┌──────────┐    │
                    │  │Episode 0 │  │Episode 0 │    │
                    │  │  env_0✓  │  │  env_8✓  │    │
                    │  │  env_1⏳ │  │  env_9⏳ │    │
                    │  │  env_2   │  │  env_10  │    │
                    │  │  ...     │  │  ...     │    │
                    │  └──────────┘  └──────────┘    │
                    └──────┬──────────────┬──────────┘
                           │              │
              get_episode_id│              │put(rollout)
              (请求工作)    │              │(提交结果)
                           ▼              ▲
                    ┌──────────────────────────────────┐
                    │     EnvironmentWorker             │
                    │  ┌──────────────┐ ┌────────────┐ │
                    │  │TrajEnvManager│ │TrajEnvManager│ │
                    │  │   env_0      │ │   env_1     │ │
                    │  │              │ │             │ │
                    │  │ run_rollout  │ │ run_rollout │ │
                    │  │  loop()      │ │  loop()     │ │
                    │  │              │ │             │ │
                    │  │ reset()      │ │ reset()     │ │
                    │  │  ↓           │ │  ↓          │ │
                    │  │ env.reset()  │ │ env.reset() │ │
                    │  │  ↓           │ │  ↓          │ │
                    │  │ make_decision│ │ make_decision│ │
                    │  │  ↓           │ │  ↓          │ │
                    │  │ step()       │ │ step()      │ │
                    │  │  ↓           │ │  ↓          │ │
                    │  │ env.step()   │ │ env.step()  │ │
                    │  │  ↓           │ │  ↓          │ │
                    │  │ formulate    │ │ formulate   │ │
                    │  │ _rollouts()  │ │ _rollouts() │ │
                    │  │  ↓           │ │  ↓          │ │
                    │  │ put()────────┼─┼─────────────┼─┼──→ GroupQueueManager
                    │  │              │ │             │ │
                    │  │ reset()      │ │ reset()     │ │
                    │  │ (下一个      │ │ (下一个      │ │
                    │  │  episode)   │ │  episode)   │ │
                    │  └──────────────┘ └────────────┘ │
                    └──────────────────────────────────┘
```

**数据流总结**：
1. `GroupQueueManager` 通过 `advance_step` 为每个 group 创建新的 `GroupData`(episode)
2. `TrajEnvManager` 通过 `get_episode_id` 获取要跑的 episode
3. `TrajEnvManager` 执行完整 rollout 循环（reset → step × N）
4. `TrajEnvManager` 将结果通过 `put` 提交给 `GroupQueueManager`
5. 当同组所有 env 都完成后，`GroupQueueManager` 释放该组数据给 `get_batch`
6. Pipeline 收集足够多的组后，组成 training batch 进入优势估计和 PPO 训练

---

## 7. 奖励计算与优势估计

### 7.1 奖励信号来源

奖励来自环境 `env.step()` 返回的 `reward`：

- **SWE-Bench**: 运行 `run_tests.sh` → 测试通过率
- **Terminal-Bench**: 任务特定测试脚本
- **Sokoban**: 是否到达目标位置
- **自定义环境**: `env.step()` 返回的任意浮点值

### 7.2 compute_discounted_returns — 折扣回报（GiGPO 专用）

`roll/pipeline/agentic/utils.py:60-95`

```python
def compute_discounted_returns(batch, adv_estimator, gamma=1.0):
    """计算每步的折扣回报 R_t = r_t + γ * R_{t+1}"""
    if adv_estimator in ["gigpo", "step_reinforce"]:
        for traj_id, traj_batch in batch.group_by("traj_id"):
            # 按步序排序
            indices = torch.argsort(traj_batch.non_tensor_batch["step"])
            traj_batch.reorder(indices)

            rewards = traj_batch.non_tensor_batch["step_scores"]
            discounts = torch.empty_like(rewards)
            running_return = 0.0

            # 从后往前累加
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + gamma * running_return
                discounts[t] = running_return

            traj_batch.batch["step_rewards"] = discounts
```

### 7.3 compute_response_level_rewards — 响应级奖励组装

`roll/pipeline/agentic/utils.py:179-234`

| 算法 | 奖励公式 |
|------|---------|
| **GiGPO** | `R = w_ep · norm(episode_scores, traj_group) + w_step · norm(step_rewards, state_group)` |
| **Step REINFORCE** | `R = norm(step_rewards, grouping)` |
| **StarPO / GRPO** | `R = norm(episode_scores.sum(), grouping)` |

**GiGPO 的双层归一化**：

```python
# 外层组（轨迹级）：同一 traj_group_id 的多条轨迹归一化
episode_rewards_norm = agentic_reward_norm(
    episode_scores, grouping="traj_group_id"
)

# 内层组（步级）：同一 state_hash 的多个步样本归一化
batch = build_state_group(batch)  # 按 (traj_group_id, state_hash) 分组
step_rewards_norm = agentic_reward_norm(
    step_rewards, grouping="state_group_id"
)

# 组合
response_level_rewards = (
    episode_reward_weight * episode_rewards_norm +
    step_reward_weight * step_rewards_norm
)
```

### 7.4 agentic_reward_norm — 组内归一化

`roll/pipeline/agentic/utils.py:99-153`

```python
def agentic_reward_norm(batch, reward_normalization):
    """组内归一化: (score - mean) / (std + ε)"""
    for group_name, group_batch in batch.group_by(keys=grouping):
        scores = group_batch.batch["scores"]

        if norm_mean_type == "group":
            reward_mean = scores.float().mean()
        if norm_std_type == "group":
            reward_std = scores.float().std()

        normalized = (scores - reward_mean) / (reward_std + 1e-6)
```

### 7.5 agentic_compute_advantage — 最终优势

`roll/pipeline/agentic/utils.py:457-537`

```python
def agentic_compute_advantage(batch, cfg):
    if adv_estimator == "gae":
        # PPO + GAE: 使用 Critic 价值函数
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards, values, gamma, lambd)

    elif adv_estimator in ["reinforce", "grpo", "gigpo", "step_reinforce"]:
        # REINFORCE: 无 Critic，直接用 reward 作为优势
        advantages, returns = compute_reinforce_return(
            token_level_rewards, gamma, lambd)

    # 可选：优势白化
    if whiten_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 可选：优势裁剪
    if advantage_clip:
        advantages = advantages.clamp(-advantage_clip, advantage_clip)

    batch.batch["advantages"] = advantages * response_mask
    batch.batch["returns"] = returns
```

---

## 8. PPO 训练更新

### 8.1 Actor 损失函数

`roll/pipeline/agentic/agentic_actor_worker.py:10-147`

```python
def loss_func(data, output_tensor, cfg):
    # 1. 计算当前策略的 log_probs
    log_probs = compute_log_probs(logits, input_ids, attention_mask)

    # 2. 计算重要性采样比率 ratio = π_new / π_old
    if ratio_type == "segment":
        # 段级比率（多轮对话专用，避免跨步比率污染）
        log_ratio = compute_segment_masked_mean(log_probs - old_log_probs, mask)
        ratio = log_ratio.exp()
    else:
        ratio = (log_probs - old_log_probs).exp()

    # 3. PPO Clipped Surrogate Loss（非对称裁剪）
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - pg_clip_low, 1 + pg_clip_high) * advantages
    pg_loss = -torch.min(surr1, surr2)

    # 4. 可选：Dual-Clip Loss (DAPO 风格)
    if dual_clip_loss:
        dual_clip = -torch.max(-pg_loss, (1 + pg_clip * 2) * advantages)
        pg_loss = torch.where(advantages < 0, dual_clip, pg_loss)

    # 5. KL 散度惩罚
    kl_loss = compute_approx_kl(log_probs, ref_log_probs, mask, "k3")
    total_loss = pg_loss + kl_loss * kl_loss_coef

    # 6. 熵奖励
    total_loss -= entropy_loss * entropy_loss_coef

    return total_loss
```

### 8.2 Train-Infer 修正

由于推理集群（vLLM）和训练集群（Megatron/FSDP2）可能存在微小的数值差异，ROLL 引入 train-infer importance sampling 修正：

```python
train_infer_is_weight, filter_mask, _ = compute_train_infer_correction(
    cfg, response_mask, old_log_probs, infer_logprobs
)
response_mask = response_mask * filter_mask
pg_loss = pg_loss * train_infer_is_weight
```

---

## 9. 完整训练循环：14 个阶段

`roll/pipeline/agentic/agentic_pipeline.py:218-596`

每个 `global_step` 执行以下完整流程：

```
Phase 1:  Offload States
          critic.offload_states() + actor_train.offload_states(blocking=True)
          → 释放训练 GPU 上的优化器/梯度状态，为推理腾出空间

Phase 2:  Suspend Scheduler
          train_rollout_scheduler.suspend()
          → 暂停推理路由，中止所有正在进行的请求

Phase 3:  Model Update (On-Policy Enforcement)
          model_update: actor_train → actor_infer
          → 将最新训练权重同步到推理集群，保证 on-policy

Phase 4:  Load States
          actor_infer.load_states()
          → 加载 KV cache 和模型权重到推理 GPU

Phase 5:  Expand Sampler (Partial GPU Mode)
          train_rollout_scheduler.expand_sampler(target_gpus)
          → 恢复所有 DP rank 的推理路由

Phase 6:  Async Validation (每 eval_steps 步)
          executor.submit(self.val, ...)
          → 异步执行验证，不阻塞训练

Phase 7:  ★ ROLLOUT ★ (Agent-Environment Interaction)
          batch = train_rollout_scheduler.get_batch(batch, rollout_batch_size)
          │
          ├─ 每个 EnvironmentWorker:
          │   ├─ 每个 EnvManager (线程):
          │   │   ├─ env.reset(seed)  ← 启动 Docker 容器
          │   │   ├─ while not terminated:
          │   │   │   ├─ make_decision() ← LLM 生成动作
          │   │   │   └─ env.step(action) ← 容器内执行
          │   │   ├─ formulate_rollouts() ← 组装训练数据
          │   │   └─ output_queue.put(rollout)
          │   └─ GroupQueue 等待同组环境全部完成
          │
          └─ 返回 batch (DataProto)

Phase 8:  Stop Server (Sync Mode)
          scheduler.suspend() + actor_infer.offload_states()
          → 同步模式下暂停推理；异步模式下推理继续服务

Phase 9:  Shrink Sampler (Partial GPU Mode)
          shrink_sampler(target_gpus)
          → 释放推理占用的训练 GPU

Phase 10: Compute Discounted Returns (GiGPO)
          compute_discounted_returns(batch, adv_estimator, gamma)
          → R_t = r_t + γ·R_{t+1} (步级折扣回报)

Phase 11: Reference Log Probs
          reference.compute_log_probs(batch) [or actor_train with LoRA off]
          → 计算 KL 惩罚所需的参考模型对数概率

Phase 12: Old Log Probs & Values
          actor_train.compute_log_probs(batch)  [if enable_old_logprobs_recompute]
          critic.compute_values(batch)  [if adv_estimator == "gae"]
          → 计算旧策略对数概率（PPO 比率分母）和价值函数估计

Phase 13: ★ ADVANTAGE COMPUTATION ★
          compute_response_level_rewards(batch, cfg)
          → 组装响应级奖励（GiGPO 双层归一化 / StarPO 轨迹归一化）

          compute_token_reward(batch, cfg, kl_ctrl)
          → 扩展到 token 级 + 添加 KL 惩罚

          agentic_compute_advantage(batch, cfg)
          → 计算最终优势（REINFORCE / GAE / OPD）

Phase 14: ★ TRAINING ★
          if adv_estimator == "gae":
              critic.train_step(batch)  → Critic 梯度更新

          if critic_warmup <= global_step:
              actor_train.train_step(batch)  → Actor PPO 梯度更新
              → ( clipped surrogate loss + KL + entropy )

          compute_data_metrics(batch)  → 统计
          do_checkpoint(...)           → 保存检查点
          tracker.log(metrics, step)   → 日志记录
```

---

## 10. 两种信用分配范式：StarPO vs GiGPO

### 10.1 StarPO — 轨迹级信用分配

```yaml
# 配置
adv_estimator: "reinforce"
env_manager_cls: roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager
reward_normalization:
  grouping: traj_group_id
  method: mean
```

**数据流**：
```
一条轨迹 → ONE 训练样本
┌─────────────────────────────────────────────┐
│ prompt_1 │ response_1 │ prompt_2 │ response_2 │ ... │ prompt_N │ response_N │
│          │           │          │           │     │          │ (reward)   │
└─────────────────────────────────────────────┘
                                                              ↑
                                              episode_score = Σ(step_rewards)
                                              放在最后一个 token 位置
```

**优势计算**：
- `R = norm(episode_scores, grouping=traj_group_id)` — 组内均值归一化
- `advantages = R` 广播到所有 response tokens
- 无 Critic，无 bootstrap

**适用场景**：单轮或少轮交互，奖励稀疏

### 10.2 GiGPO — 步级信用分配（Group-in-Group）

```yaml
# 配置
adv_estimator: "gigpo"
env_manager_cls: roll.pipeline.agentic.env_manager.step_env_manager.StepEnvManager
episode_reward_weight: 1.0
step_reward_weight: 1.0
step_reward_gamma: 0.95
reward_normalization:
  grouping: traj_group_id
  method: mean
```

**数据流**：
```
一条 N 步轨迹 → N 个训练样本
┌──────────────┐  ┌──────────────┐       ┌──────────────┐
│ prompt_1     │  │ prompt_2     │  ...  │ prompt_N     │
│ response_1   │  │ response_2   │       │ response_N   │
│ (step_reward)│  │ (step_reward)│       │ (step_reward)│
│ state_hash=A │  │ state_hash=B │       │ state_hash=Z │
│ step=0       │  │ step=1       │       │ step=N-1     │
└──────────────┘  └──────────────┘       └──────────────┘
```

**双层归一化（Group-in-Group）**：

```
外层组 (traj_group_id):
  ┌─────────────────────────────────────┐
  │ Group 0 (seed=42, 8 条轨迹)          │
  │  traj_0: episode_score = 0.8        │
  │  traj_1: episode_score = 0.2        │  → norm: (score - mean) / std
  │  traj_2: episode_score = 0.5        │
  │  ...                                │
  └─────────────────────────────────────┘

内层组 (state_hash):
  ┌─────────────────────────────────────┐
  │ State A (state_hash=hash(obs_A))    │
  │  traj_0/step_3: step_reward = 0.1   │
  │  traj_2/step_5: step_reward = 0.3   │  → norm: (reward - mean) / std
  │  traj_5/step_2: step_reward = 0.0   │  (不同轨迹到达相同状态)
  └─────────────────────────────────────┘

最终奖励: R = w_ep * episode_norm + w_step * step_norm
```

**核心洞察**：
- **外层组** (traj_group_id)：评估轨迹整体质量 → 宏观信用分配
- **内层组** (state_hash)：评估在相同状态下不同动作的相对优劣 → 微观信用分配
- 组合两者 = 无 Critic 的细粒度信用分配

**适用场景**：多轮交互，步级动作有不同价值，需要细粒度信用分配

### 10.3 对比总结

| 维度 | StarPO | GiGPO |
|------|--------|-------|
| EnvManager | TrajEnvManager | StepEnvManager |
| 每条轨迹样本数 | 1 | N（每步一个） |
| 奖励粒度 | 轨迹级 (episode reward) | 轨迹级 + 步级 |
| 归一化分组 | traj_group_id | traj_group_id + state_hash |
| Critic | 不需要 | 不需要 |
| 信用分配 | 宏观 | 宏观 + 微观 |
| PPO ratio_type | token | segment |
| 论文 | [StarPO](https://arxiv.org/abs/2504.20073) | [GiGPO](https://arxiv.org/abs/2505.10978) |

---

## 附录：关键配置示例

```yaml
# examples/qwen2.5-7B-agentic_mooncake/agentic_val_sokoban_mooncake.yaml
pipeline_config:
  pipeline_type: agentic
  adv_estimator: gigpo              # 或 reinforce (StarPO)
  actor_rollout_ref:
    actor:
      strategy: megatron            # 或 fsdp2
      model:
        model_path: Qwen/Qwen2.5-7B-Instruct
      ppo_mini_batch_size: 128
      ppo_micro_batch_size: 4
      loss_agg_mode: token          # 或 segment
    rollout:
      strategy: vllm
      rollout_batch_size: 1024
      sequence_length: 8192
  custom_envs:
    SokobanTrain:
      env_type: sokoban_sandbox     # 容器化环境
      env_manager_cls: roll.pipeline.agentic.env_manager.step_env_manager.StepEnvManager
      env_config:
        mode: train
        max_steps: 30
        base_url: http://localhost:8080
  episode_reward_weight: 1.0
  step_reward_weight: 1.0
  step_reward_gamma: 0.95
```

---

## 附录：关键文件索引

| 功能 | 文件路径 |
|------|---------|
| 主 Pipeline | `roll/pipeline/agentic/agentic_pipeline.py` |
| 环境管理器基类 | `roll/pipeline/agentic/env_manager/base_env_manager.py` |
| 轨迹级管理器 (StarPO) | `roll/pipeline/agentic/env_manager/traj_env_manager.py` |
| 步级管理器 (GiGPO) | `roll/pipeline/agentic/env_manager/step_env_manager.py` |
| Native Agent 管理器 | `roll/pipeline/agentic/env_manager/agent_native_env_manager.py` |
| 环境工作进程 | `roll/pipeline/agentic/environment_worker.py` |
| Sandbox 管理器 | `roll/pipeline/agentic/env/rock/sanbox_manager.py` |
| Sandbox V2 | `roll/pipeline/agentic/env/rock/sandbox_manager_v2.py` |
| Sokoban 沙箱环境 | `roll/pipeline/agentic/env/sandbox/sokoban_sandbox_env.py` |
| SWE Native 环境 | `roll/pipeline/agentic/env/swe_env/swe_native_env.py` |
| Terminal Native 环境 | `roll/pipeline/agentic/env/terminal_env/terminal_native_env.py` |
| Rock TB Native 环境 | `roll/pipeline/agentic/env/terminal_env/rock_tb_native_env.py` |
| CLI 环境 | `roll/pipeline/agentic/env/cli_env/env.py` |
| 奖励/优势计算 | `roll/pipeline/agentic/utils.py` |
| PPO 损失 | `roll/pipeline/agentic/agentic_actor_worker.py` |
| Agentic 配置 | `roll/pipeline/agentic/agentic_config.py` |
| Rollout 调度器 | `roll/distributed/scheduler/rollout_scheduler.py` |
| 环境注册表 | `roll/pipeline/agentic/env/__init__.py` |
