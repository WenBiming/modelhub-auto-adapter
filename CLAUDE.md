# modelhub-auto-adapter

ModelHub XC「适配智能体」：自动发现 HF/ModelScope 模型并通过开放平台 API
提交国产算力适配任务的无人值守调度系统。

## 权威文档
- 架构与模块契约（实现以此为准）：`docs/specs/2026-08-29-auto-model-adapter-design.md`
- 上游产品文档：《自动模型适配系统设计文档》（不在本仓库）

## 铁律（违反会触发平台清除所有任务）
1. **绝不重复提交**：同 `(model_id, target_gpu)` 已适配/在途/拉黑的组合一律跳过；
   平台查询失败时保守跳过，宁漏勿重。
2. **限流**：提交速率与在途上限只能调小，不能绕过。
3. 容器内**不做任何模型推理/权重下载**（限额 1 CPU / 512Mi，只当调度大脑）。

## 工程约定
- Python 3.11，src 布局：`src/auto_adapter/`；`pip install -e ".[dev]"` 后 `pytest`。
- 所有平台/外部 HTTP 调用只允许出现在 `platform_client.py` 和 `discovery/` 的
  source 实现里，业务模块不直接发请求。
- 凭据（`XC_TOKEN`、`STRATEGY_ID`）只从环境变量读取（`settings.py`），不硬编码、不写日志。
- 状态一律走 `storage/` 接口持久化，禁止业务模块自建内存态（30s 优雅停机要求）。
- 测试不打真实网络：平台 API 用 `responses` mock，storage 用内存 SQLite。

## 当前状态
M1–M6 均已实现：健康检查 + SIGTERM 优雅停机、HF/悬赏发现、去重与资格判定、
配置生成与入库、限流提交、对账监控与失败分类退避、全链路 tick 接线，均有测试
覆盖。测试：`.venv/bin/pytest`（全绿，无 skip 占位残留）。

上线前仍需人工核对（不属于代码任务，见 spec §9 及 DoD）：
- 用真实平台 `status`/`verifyResult` 枚举值回填 `rules.PLATFORM_STATUS_MAP`
  （当前为占位映射）；
- 按实际可用算力扩充 `rules.KNOWN_GPUS`（当前仅 `MetaX_c-500` 一项）；
- 准备生产环境的悬赏 JSON 配置文件（`BOUNTY_CONFIG_PATH` 指向的人工维护列表）。
