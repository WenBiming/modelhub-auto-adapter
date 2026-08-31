# modelhub-auto-adapter

自动模型适配智能体：从 HuggingFace / ModelScope / 平台悬赏列表发现候选模型，
经去重准入、配置生成后通过 ModelHub XC 开放平台 API 提交适配任务，
并监控、分类失败、自动重试，最终以平台合规容器（适配智能体）形态无人值守运行。

## 架构

```
discovery → eligibility → config_gen → submitter → [ModelHub API]
                                                        │
              failure ◀── monitor ◀─────────────────────┘
                 │
          retry / blacklist
```

详见 [`docs/specs/2026-08-29-auto-model-adapter-design.md`](docs/specs/2026-08-29-auto-model-adapter-design.md)。

## 开发

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## 运行（容器）

```bash
docker build -t auto-adapter .
docker run -p 8080:8080 \
  -e XC_TOKEN=... -e STRATEGY_ID=... -e MODELHUB_BASE_URL=... \
  -v $(pwd)/data:/data auto-adapter
```

健康检查：`GET :8080/health`（恒返回 200，平台合规要求）。
运行状态：`GET :8080/` 返回 `{status, config_error, dry_run}`——配置写错时看这里。

## 平台运行时契约

平台只注入两个环境变量（实测自官方 demo 仓库 `xc_agent_platform_demo` 与提交说明）：

| 变量 | 来源 | 说明 |
|---|---|---|
| `EXTERNAL_SERVICE_TOKEN` | 平台注入 | 平台凭据。本地开发可改用 `XC_TOKEN`，两者都接受 |
| `STRATEGY_ID` | 平台注入 | 策略 ID，提交任务时填入 `strategyId` |
| `MODELHUB_BASE_URL` | 可选 | 默认 `https://modelhub.org.cn` |

平台界面**不提供设置环境变量的入口**，所以其余可调参数（`DRY_RUN`、`STORAGE_PATH`
等）的默认值只能写在 Dockerfile 的 `ENV` 里——改完重新打 tag 即可生效。
**凭据绝不要写进 Dockerfile**：镜像层是明文且随仓库分发。

### 模型来源：平台内网连不上 HuggingFace

线上实测 `huggingface.co` connect timeout。默认来源因此改为 **ModelScope**
（`MODELSCOPE_DISCOVERY_ENABLED=true`，HF 默认关闭）——平台自己 API 文档的样例
用的也是 modelscope.cn 的地址。若你的环境有 HF 镜像，设 `HF_ENDPOINT` 再打开 HF。

### 鉴权：EXTERNAL_SERVICE_TOKEN 未必等于开放平台的 xcToken

线上实测：平台注入的 `EXTERNAL_SERVICE_TOKEN` 以 `Xc-Token` 头调开放平台 API 会被
**401 拒绝**。官方 demo 只检查该变量是否存在，从不拿它调 API，所以文档没有回答
"智能体该用什么令牌调开放平台"这个问题。

`AUTH_HEADER` 可在不改代码的情况下换鉴权头名（默认 `Xc-Token`）便于排查。
凭据无效时熔断闸会拉起且**不会自动解除**——重试一万次也修不好一个无效令牌。

### 存储是临时的

平台不挂载持久卷（默认 `/data` 在容器里根本不存在，线上会报
`unable to open database file`）。数据库落在镜像自带的 `/app/data`，**容器重启即丢**。

丢库的危险不在于丢数据，而在于：本地任务表清零后重新发现同一模型时，
**正在运行中的任务不会出现在 `search_model` 的已完成结果里**，去重会放行 →
同一 `(model_id, target_gpu)` 被提交第二次 → 触发平台的重复提交清号。

因此启动时会先从平台认领仍在途（waiting/running）的任务写回本地；若平台不可达
或只读到部分名单，**直接拉下熔断闸暂停提交**，宁可这一轮不干活。

配置缺失时进程**不会崩溃**：`/health` 先于配置校验启动并保持 200，错误原因写进
`/` 端点与日志。这是刻意的——平台 livenessProbe 连续失败会重启 Pod，重启三次后
标记为失败，崩溃退出等于把诊断信息一起丢掉。

## 首次上线：先跑演练（DRY_RUN）

设 `DRY_RUN=true` 后，智能体会完整执行发现 → 去重 → 配置生成 → 按限流挑选，
**唯独不调用 `add_task`**，把"本来要提交什么"以 JSON 打进日志：

```bash
docker run -p 8080:8080 \
  -e EXTERNAL_SERVICE_TOKEN=... -e STRATEGY_ID=... -e DRY_RUN=true \
  -v $(pwd)/data:/data auto-adapter
```

演练模式下记录保持 `QUEUED`、`task_id` 保持 `None`——绝不伪造 `PENDING`/`task_id`，
否则对账层会在平台侧找不到它，误判为"任务被平台清理"而拉下熔断闸。

首次上线务必先用它验证这几件从未接触过真实平台的假设：`search_model` 返回的
`verifyResult` 形状（去重的命脉）、`configParams` 能否被平台接受、选出来的模型
是否合理。确认无误再去掉 `DRY_RUN`。

默认基础镜像是平台 registry（本机通常不可达）。本地验证构建时用公共镜像覆盖：

```bash
docker build --build-arg BASE_IMAGE=python:3.11-slim -t auto-adapter .
```

**这一步是合并前的必过门禁**，不能只靠 `pytest`：测试跑在 editable 安装上，
`Path(__file__).parent` 会落到源码树，因此看不到"`templates/*.yaml` 没打进 wheel"
这类打包缺陷——而容器用的是 `pip install .`，缺了模板会在第一个候选渲染时抛
`FileNotFoundError`，整条流水线在 submitter 之前就中断，而 `/health` 照常返回 200。
`tests/test_packaging.py` 用 `importlib.resources` 守住了这一点，但真正的构建仍要跑一次。

## 运维：熔断开关（kill switch）

熔断开关是本系统唯一的安全刹车：一旦打开，`submitter.drain` 立即停止提交任何任务
（发现/对账仍继续）。它由这些情形自动打开，**不会自动关闭**：

| 触发 | 含义 |
|---|---|
| 平台返回 40100 / 40101 | 凭据失效或无权限（`XC_TOKEN` 过期） |
| 提交成功但 task_id 写盘失败 | 存储不可靠，有重复提交风险，须人工对账 |
| 在途任务从平台列表中消失 | 可能被平台按违规清理，最高级别告警 |
| 连续 5 次引擎失败 | 配置模板本身可能有问题 |

**查看状态**：每个 tick 的 metrics 日志行都带 `kill_switch` 字段，`/health` 不反映
它（平台合规要求恒返回 200）：

```json
{"metrics": {"submitted": 0}, "kill_switch": {"on": true, "reason": "task 42 vanished from platform (possible violation cleanup)"}}
```

也可以直接查库（`STORAGE_PATH`，容器内默认 `/data/agent.db`）：

```bash
sqlite3 /data/agent.db "SELECT value FROM kv WHERE key='kill_switch';"
```

**清除**：先按上表的 reason 排查并处理根因（换凭据、人工对账在途任务、修模板），
确认平台侧没有遗留的重复/在途任务后，再手动清零：

```bash
sqlite3 /data/agent.db \
  "UPDATE kv SET value='{\"on\": false, \"reason\": \"cleared by <操作人> <日期>\"}' WHERE key='kill_switch';"
```

进程会在下一个 tick 自动恢复提交，无需重启。**不要在未查清原因前清除**——刹车打开
的每一种情形都意味着"再提交一次就可能构成重复提交"，而平台对重复提交的处理是清空
账号下的全部任务。
