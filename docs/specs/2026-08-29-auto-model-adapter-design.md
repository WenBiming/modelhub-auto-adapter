# 自动模型适配系统 — 架构设计与实现 Spec

> 状态：v0.1 已定稿骨架 / 待实现
> 上游产品文档：《自动模型适配系统设计文档》（ModelHub XC 开放平台）
> 本文档面向 Claude Code：按「里程碑」章节的顺序实现，每个模块的接口契约以本文为准。

---

## 1. 技术选型与总体决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 语言/运行时 | Python 3.11 | 平台官方基础镜像为 `python:3.11-slim`，生态匹配 |
| 进程模型 | 单进程：主调度循环（同步） + Flask 健康检查（后台线程） | 容器限额 1 CPU / 512Mi，只做"调度大脑"，无需并发框架 |
| HTTP 客户端 | `requests` | 简单、可测（配合 `responses` mock） |
| 持久化 | 抽象 `Storage` 接口；默认 SQLite（挂载卷），可切换 Redis | 满足"30 秒优雅停机 + 重启恢复"要求；接口隔离使后端可替换 |
| 配置 | 全部经环境变量注入（`settings.py` 集中读取） | `Xc-Token`/`STRATEGY_ID` 不落盘、不硬编码 |
| 打包 | `pyproject.toml` + `pip install -e .` | 标准化，Dockerfile 直接复用 |

**架构风格**：模块化单体。7 个业务层各自为一个模块，模块间只通过
数据类（`models.py`）和显式函数调用交互，不共享可变全局状态。
所有对平台/HF/MS 的网络访问收敛到两个客户端类中，业务模块不直接发 HTTP。

## 2. 运行时拓扑

```
main.py (entrypoint)
 ├─ health.py            Flask /health :8080（daemon 线程）
 ├─ signal handler        SIGTERM → 置 stop_event → 主循环收尾（<30s）
 └─ scheduler loop（主线程，每 tick 顺序执行）:
      1. discovery.run()        → List[CandidateModel] → storage
      2. eligibility.filter()   → 标注优先级/跳过 → 待提交队列(storage)
      3. submitter.drain()      → 限流提交 → TaskRecord 入库
      4. monitor.poll()         → 更新 TaskRecord 状态，失败任务拉日志
      5. failure.handle()       → 分类 → 重试入队 / 拉黑 / 终止超时任务
```

每个 tick 之间 sleep（可配置，默认 60s）；发现层内部另有更长的节流
（HF/MS 每小时增量拉取一次，悬赏列表每 tick 都查）。

## 3. 数据模型（`models.py`，全部为 dataclass）

### CandidateModel
| 字段 | 类型 | 说明 |
|---|---|---|
| source | `"huggingface" \| "modelscope" \| "bounty"` | 来源 |
| model_id | str | 如 `Qwen/Qwen2.5-7B-Instruct` |
| model_url | str | 提交时填入 `modelAddress` |
| pipeline_tag | str \| None | 缺失时由规则表兜底 |
| params_size | str \| None | 如 `"7B"`，驱动并行策略 |
| is_bounty | bool | |
| bounty_deadline | datetime \| None | |
| discovered_at | datetime | |

### TaskRecord（本地任务表，持久化）
| 字段 | 类型 | 说明 |
|---|---|---|
| task_id | int \| None | 平台返回（int64）；提交前为 None |
| model_id | str | |
| target_gpu | str | |
| framework | str | |
| status | `TaskStatus` | 见状态机 |
| priority | `Priority` | BOUNTY > NEW_MODEL > NEW_ADAPTATION |
| retry_count | int | |
| submit_time | datetime \| None | |
| bounty_deadline | datetime \| None | |
| config_params | str | 实际提交的 configParams（YAML 字符串，附录 A.1.1），重试时解析调整后重渲染 |

### 状态机（TaskStatus）

```
QUEUED ──submit──▶ PENDING ──▶ RUNNING ──▶ SUCCESS
   ▲                  │            │
   │                  │            ├──▶ ENGINE_FAILED ──retry<max──▶ QUEUED(调参后)
   │                  │            │         └─retry≥max─▶ BLACKLISTED
   └──────────────────┘            ├──▶ QUALITY_FAILED ──▶ NEEDS_HUMAN（不自动重试）
        (超时终止后可重排队)         └──▶ TIMEOUT ──stop API──▶ QUEUED 或 ABANDONED
```

**不变式**：同一 `(model_id, target_gpu)` 组合在
{QUEUED, PENDING, RUNNING} 中最多存在一条记录；BLACKLISTED 后永不再入队。
该不变式由 storage 层的唯一键保证（防呆幂等的最终防线）。

## 4. 模块契约

### 4.1 `platform_client.py` — ModelHub 开放平台客户端
唯一允许调用平台 API 的地方。构造时注入 `base_url` 与 `xc_token`。

| 方法 | 对应 API |
|---|---|
| `add_task(req: AddTaskRequest) -> int`（返回平台任务 id，int64） | `POST /api/adapt/task/add` |
| `list_my_tasks(current, page_size, **filters) -> Page`（固定 `onlyMine=true`） | `GET /api/adapt/task/page` |
| `get_task_log(task_id: int) -> str` | `GET /api/adapt/task/log?taskId=` |
| `search_model(model_id) -> ModelSearchResult` | `GET /api/computility/models/search-by-model-id?modelId=` |
| `stop_tasks(task_ids: list[int]) -> bool`（批量） | `PUT /api/async/task/stop-create-contest-task` |
| `list_bounties() -> list[BountyItem]` | 悬赏列表接口（开放平台 API 文档中无此接口，见 §9） |

所有响应为统一信封 `{code, message, data}`，`code == 0` 为成功；非 0 按错误码表
（附录 A.6）分类：40100/40101 为凭据问题（告警并打开 kill_switch），40400 视为资源
不存在（业务分支处理），50000/50001 按临时错误由下个 tick 自然重试。
网络错误/超时同样记日志跳过，不做进程内指数退避——下个 tick 就是退避。
精确请求/响应 schema 见附录 A（抓取自平台在线 API 文档，2026-08-29）。

### 4.2 `discovery/` — 模型发现层
- `base.py`：`class DiscoverySource(Protocol): def fetch(self) -> list[CandidateModel]`
- `huggingface.py` / `modelscope.py`：按 downloads/trending/pipeline_tag 拉取，节流 1h。
- `bounty.py`：包装 `platform_client.list_bounties()`，每 tick 执行，产出 `is_bounty=True` 的候选。
- 输出统一去重（同 model_id 保留最新），写入 storage 的候选表。

### 4.3 `eligibility.py` — 去重与准入（全系统最关键）
`def evaluate(candidate, storage, client) -> Decision`

双重校验，两道都过才入队：
1. **本地**：storage 中已有同 `(model_id, target_gpu)` 的活跃/成功/拉黑记录 → SKIP；
2. **平台**：`search_adaptations(model_id)` 按结果分类：

| 分类 | 条件 | 动作 |
|---|---|---|
| 新模型 | 平台无任何记录 | 入队，Priority.NEW_MODEL |
| 新适配 | 有记录但目标 GPU 不同 | 入队，Priority.NEW_ADAPTATION |
| 完全重复 | 同模型同 GPU 已适配 | **SKIP，绝不提交** |
| 悬赏 | 命中悬赏列表 | 入队，Priority.BOUNTY，携带 deadline |

平台查询失败时**保守处理：跳过本 tick**（宁可漏提交，不可重复提交）。

### 4.4 `config_gen.py` — 任务配置生成
`def build_request(candidate, target_gpu) -> AddTaskRequest`

- `taskType`：pipeline_tag 直映射；缺失时查 `rules.py` 规则表（模型名关键词 → taskType），仍无法判定 → 标记 NEEDS_HUMAN，不盲提。
- `targetGpu`：由 `gpu_selector` 选平台覆盖率最低的型号（覆盖率数据来自 search 接口聚合，缓存于 storage）。
- `framework`：架构在 vllm 支持列表（`rules.py` 维护）→ vllm；否则退化到备选框架。
- `configParams`：`templates/` 下 YAML 模板 + 参数量推导（如 tp_size：<14B→1，14–70B→2，>70B→4）。
- `strategyId`：读 `settings.strategy_id`。

### 4.5 `submitter.py` — 提交与调度
- 从 storage 取 QUEUED 记录，按 `(priority, bounty_deadline, discovered_at)` 排序；
- 限流：令牌桶——`max_submits_per_minute`（默认 2）与 `max_inflight`（默认 5，PENDING+RUNNING 总数）；
- 提交成功 → 写回 task_id、submit_time、status=PENDING；提交失败 → 保持 QUEUED 记录原因；
- 悬赏临近 deadline（剩余 < 预估适配时长 × 2）仍未能提交 → 告警日志并放弃（ABANDONED），避免无效占用。

### 4.6 `monitor.py` — 监控与日志
- 每 tick 调 `list_tasks` 对账：以平台状态为准更新本地 TaskRecord；
- 平台侧消失的任务（可能被违规清理）→ 标记 ABANDONED 并**触发最高级别告警**（说明去重逻辑可能出了问题，暂停 submitter 直到人工确认，通过 storage 中的 `kill_switch` 标志实现）；
- 失败任务拉日志存入 TaskRecord 关联字段，交给 failure 层；
- RUNNING/PENDING 超过 `task_timeout`（默认 6h）→ 标记 TIMEOUT。

### 4.7 `failure.py` — 失败分类与重试
`def classify(log_text) -> FailureKind`：基于日志关键词规则（OOM / CUDA error / 容器启动失败 → ENGINE；judge 未通过 → QUALITY）。

| 类型 | 动作 |
|---|---|
| ENGINE | `next_config(record)` 生成调整后的 configParams（降精度→降并行→换框架，按序尝试），retry_count+1，≤3 次重新入队；超限 → BLACKLISTED |
| QUALITY | 不重试，NEEDS_HUMAN + 写黑名单 |
| TIMEOUT | 调 stop API 释放资源；悬赏未过期可重排队一次，否则 ABANDONED |

### 4.8 `storage/` — 持久化
`base.py` 定义接口（候选表、任务表、黑名单、GPU 覆盖率缓存、kill_switch），
`sqlite.py` 为默认实现（唯一索引 `(model_id, target_gpu)` 保证 §3 不变式），
`redis_store.py` 留接口占位。路径来自 `STORAGE_PATH` 环境变量（须指向挂载卷）。

### 4.9 平台合规封装（仓库根目录）
- `Dockerfile`：基于 `modelhubxc-4pd.tencentcloudcr.com/xc_agent_platform/python:3.11-slim`，`EXPOSE 8080`；
- `health.py`：`GET /health → {"status":"ok"}, 200`；
- SIGTERM：`main.py` 注册 handler，置 `threading.Event`，主循环每步检查，当前步骤完成即退出（每步必须 < 30s，因此单 tick 内不做长阻塞调用，HTTP 超时 ≤ 10s）；
- 无任何本地推理/下载模型权重的代码路径（容器只有 512Mi）。

## 5. 配置项（环境变量）

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `XC_TOKEN` | ✅ | — | 平台鉴权 |
| `STRATEGY_ID` | ✅ | — | 平台注入的策略 ID |
| `MODELHUB_BASE_URL` | ✅ | — | 开放平台地址 |
| `STORAGE_PATH` | | `/data/agent.db` | SQLite 路径（挂载卷） |
| `TICK_SECONDS` | | 60 | 主循环间隔 |
| `MAX_SUBMITS_PER_MINUTE` | | 2 | 限流 |
| `MAX_INFLIGHT` | | 5 | 在途任务上限 |
| `MAX_RETRIES` | | 3 | 引擎失败重试上限 |
| `TASK_TIMEOUT_HOURS` | | 6 | 超时判定 |

## 6. 可观测性
`metrics.py` 维护进程内计数器（发现数/跳过数/提交数/各类失败数），
每 tick 以结构化 JSON 打到 stdout（平台日志即采集面）。
连续 N 次（默认 5）引擎失败 → ERROR 级告警日志 + 暂停提交（kill_switch）。

## 7. 测试策略
- 单测：eligibility（分类矩阵 4 案例 + 平台查询失败的保守路径）、config_gen（tp_size 推导、taskType 兜底）、failure（日志分类关键词）、storage（唯一键幂等）——全部纯函数/内存 SQLite，不打网络；
- 集成：`responses` mock 平台 API，跑通「发现→提交→失败→重试→拉黑」全链路一个 tick；
- 合规：SIGTERM 后 30s 内进程退出的冒烟测试。

## 8. 实现里程碑（供 Claude Code 按序执行）

> 详细任务级实现计划（含测试与实现代码）见
> `docs/superpowers/plans/2026-08-29-auto-model-adapter.md`，以计划文档为执行依据。

1. **M1 基座**：models.py / settings.py / storage(sqlite) / health / main 循环骨架 + SIGTERM，测试：storage 幂等、健康检查、优雅停机；
2. **M2 平台客户端**：platform_client 全部方法 + responses 单测；
3. **M3 准入链路**：discovery(bounty 优先) → eligibility → 入队，测试分类矩阵；
4. **M4 提交链路**：config_gen + submitter（限流/优先级），端到端 mock 提交；
5. **M5 闭环**：monitor + failure（分类/重试/黑名单/超时终止）；
6. **M6 合规打包**：Dockerfile、metrics、对账告警 kill_switch，全链路集成测试。

每个里程碑遵循 TDD：先写该模块契约测试，再实现。

## 9. 开放问题（实现前需向用户确认）
- ~~`POST /api/adapt/task/add` 的精确 schema~~ → 已确认，见附录 A（2026-08-29 抓取自在线文档）；
- ~~framework 与 targetGpu 的合法枚举~~ / ~~status 与 verifyResult 的枚举~~ →
  **已确认**，见附录 B（2026-08-29 实测自平台任务列表页）；
- 悬赏列表：开放平台 API 文档确认**没有**悬赏接口——悬赏来源需改为爬取
  「模型适配挑战」页面或人工配置列表（BountySource 实现方式待定）；
- `stage` 与 framework 的合法枚举仍未确认（framework 样例仅见 `vllm`，v0.1 只提交 vllm）。

---

## 附录 A：平台 API 契约（抓取自 https://modelhub.org.cn/other/api.html，2026-08-29）

统一约定：请求头 `Xc-Token`；响应信封 `{code: int32, message: str, data: T}`，`code == 0` 成功。

### A.1 POST /api/adapt/task/add — 创建验证任务

请求体（全部必填；**注意 `configParams` 是 YAML 字符串，不是 JSON 对象**）：

| 字段 | 类型 | 说明 |
|---|---|---|
| modelAddress | string | 模型仓库或模型文件地址 |
| taskType | string | 如 `text-generation` |
| targetGpu | string | 如 `MetaX_c-500` |
| framework | string | 如 `vllm` |
| strategyId | string | 验证策略 ID（UUID） |
| configParams | string | YAML 配置内容，结构见 A.1.1 |

响应 `data: AsyncTaskVO`：`{id: int64, name: str, status: str, createTime, updateTime}` —— **任务 id 是 int64**。

#### A.1.1 configParams YAML 结构（官方样例）

```yaml
framework: vllm
api: completion        # completion | (其他待确认)
lang: en
max_model_len: 4096
max_tokens: 1024
temperature: 0.7
repetition_penalty: 1.1
top_p: 0.9
sut_config:            # 被测系统（目标国产 GPU）启动配置
  gpu_num: 1
  values:
    command:
    - /opt/conda/bin/vllm
    - serve
    - /model
    - --port
    - '20644'
    - --served-model-name
    - llm
    - --max-model-len
    - '4096'
    - --gpu-memory-utilization
    - '0.9'
    - -tp
    - '1'
    - --enforce-eager
    - --trust-remote-code
ref_config:            # 参照系统启动配置（对照评测用）
  gpu_num: 1
  values:
    command:
    - vllm
    - serve
    - /model
    - --port
    - '80'
    - --served-model-name
    - llm
    - --max-model-len
    - '4096'
    - -tp
    - '1'
    - --enforce-eager
    - --trust-remote-code
```

调参含义：引擎失败重试时可调整 `gpu_num`/`-tp`（并行度）、`--gpu-memory-utilization`、
`max_model_len`（显存）等；`sut_config` 与 `ref_config` 的 `-tp`/`gpu_num` 需保持一致调整。

### A.2 GET /api/adapt/task/log — 获取任务日志

Query：`taskId: int64`（必填）。响应 `data: string`（日志全文）。

### A.3 GET /api/adapt/task/page — 获取验证任务列表

Query（全部可选）：`current`、`pageSize`（int64 分页）、`onlyMine: bool`（**对账时固定 true**）、
`modelId`、`gpuType`、`taskId`、`taskType`、`status`、`stage`、`verifyResult: int32`、
`searchText`、`creatorId`、`beginTime`、`endTime`（date-time）。

响应 `data: Page<AsyncModelVerifyTaskVO>`：`{current, pages, size, total, records[], ...}`，
`records[]` 关键字段：

| 字段 | 类型 | 用途 |
|---|---|---|
| taskId | int64 | 对账主键 |
| modelId | string | |
| gpuType | string | |
| status / statusText | string | 任务状态（枚举待确认） |
| verifyResult / verifyResultText | int32 / string | 验证结果（枚举待确认） |
| updateTime | date-time | 超时判定依据 |
| machineId/machineName, manufacturerId/manufacturerName, creatorId, userAccount, contributors[], modelTaskLevel(Id), modelAuthorName, userProfile | — | 本系统暂不使用 |

### A.4 GET /api/computility/models/search-by-model-id — 模型详情（去重依据）

Query：`modelId: string`（必填）。响应 `data: ModelSearchResultVO`：

| 字段 | 类型 | 说明 |
|---|---|---|
| isInDB | boolean | 模型是否已入库 |
| modelInfo | ModelBasicInfo | `{modelId, modelName, authorName, source, createTime}` |
| verifyResult | **Map<string, GpuVerifyResult>** | **按 GPU 型号分键的验证结果**——eligibility 分类的直接依据：无键=新模型；无当前目标 GPU 的键=新适配；有键且已通过=完全重复跳过（GpuVerifyResult 内部字段待实测确认） |

### A.5 PUT /api/async/task/stop-create-contest-task — 终止验证任务

请求体：`{"taskIds": [int64, ...]}`（**批量**）。响应 `data: boolean`。

### A.6 统一错误码

| code | 名称 | 含义 | 本系统处理 |
|---|---|---|---|
| 40100 | NOT_LOGIN_ERROR | 未登录/登录态失效 | 凭据告警 + kill_switch |
| 40101 | NO_AUTH_ERROR | 无权限 | 凭据告警 + kill_switch |
| 40400 | NOT_FOUND_ERROR | 资源不存在 | 按业务分支处理（如查无模型） |
| 50000 | SYSTEM_ERROR | 系统内部异常 | 记日志，下个 tick 重试 |
| 50001 | OPERATION_ERROR | 操作执行失败 | 记日志，检查参数后重试 |


---

## 附录 B：状态与 GPU 枚举（实测自平台任务列表页，2026-08-29）

来源：`https://modelhub.org.cn/#/profile` 的「我的适配任务」筛选控件与其发出的
`/api/adapt/task/page` 请求参数、以及响应记录的原始字段值。

### B.1 status 与 verifyResult 是正交的两个字段

**这是最容易踩的坑**：`status` 表示"任务作业是否执行完毕"，`verifyResult` 才表示
"适配是否通过"。UI 上的五个标签实际是两者的组合：

| UI 标签 | `status` | `verifyResult` | 本地映射 |
|---|---|---|---|
| 验证排队中 | `waiting` | 1 | `PENDING` |
| 验证进行中 | `running` | 1 | `RUNNING` |
| 验证异常 | `failed` | 1 | 拉日志分类（引擎层） |
| 验证通过 | `success` | **1** | `SUCCESS`（唯一的真正成功） |
| 验证失败 | `success` | **-1** | 拉日志分类（作业跑完但未通过） |

响应记录同时带 `statusText`（如"成功"）与 `verifyResultText`（如"验证失败"）——
注意"成功"指的是作业成功，不是适配成功。

**保守规则（`rules.map_platform_result`）**：只有 `status=success` 且
`verifyResult == 1` 才判定为 SUCCESS；`success` 配上任何其他 verifyResult（-1、
缺失、未知值）一律交给日志分类。宁可多分类一次，也绝不无凭据地宣布适配成功。

### B.2 targetGpu 合法枚举（10 项）

`Ascend_910-b4`、`MetaX_c-500`、`Cambricon_mlu-370-x4`、`Kunlunxin_p-800`、
`Kunlunxin_r-200-8f`、`Iluvatar_bi-150`、`Iluvatar_mrv-100`、`hygon_k100-ai`、
`Vastai_va16`、`Sunrise_pt-200-x1`

对应厂商信息可从任务记录的 `manufacturerName` / `machineName` 读到
（如 `Ascend_910-b4` → 厂商"昇腾"、机型"910B4"）。
