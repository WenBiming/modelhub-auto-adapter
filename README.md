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

健康检查：`GET :8080/health`。

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
