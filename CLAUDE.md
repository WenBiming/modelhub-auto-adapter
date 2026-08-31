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

## v0.1 明确的范围限制（改动前先读）
- **只提交 vllm 候选**：`config_gen.build_request` 对非 vllm 框架抛
  `UnresolvableCandidateError`，落 NEEDS_HUMAN 记录而不提交。理由见 spec §9：
  `templates/transformers.yaml` 的 command 是占位符，而重试梯子只调 tp/显存/上下文
  长度，改不了一条错误的启动命令——盲提只会白烧 3 次提交后被永久拉黑。平台确认真实
  框架列表与启动命令后解除。
- **一个模型只适配一张卡**：候选的 processed 标记按 `model_id` 记（不含 GPU）。
  `select_target_gpu` 按 `storage.gpu_coverage()` 选覆盖率最低的卡（覆盖率由
  `eligibility` 从平台 `verifyResult` 累积写入），扩充 `KNOWN_GPUS` 后多卡覆盖靠
  不同模型分流实现，而不是同一模型逐卡各提一次。要改成"每模型每卡各一次"必须同时
  改候选表主键。
- **提交结果不可知时不回滚**：传输层异常 / HTTPError / 50000 / 50001 一律让记录留在
  PENDING + `task_id=None`，由 `monitor.poll` 按 (modelId, gpuType) 认领回来；只有
  平台明确拒绝（其余业务码）才允许退回 QUEUED 重试。

上线前仍需人工核对（不属于代码任务，见 spec §9 及 DoD）：
- 用真实平台 `status`/`verifyResult` 枚举值回填 `rules.PLATFORM_STATUS_MAP`
  （当前为占位映射）；
- 按实际可用算力扩充 `rules.KNOWN_GPUS`（当前仅 `MetaX_c-500` 一项）；
- 准备生产环境的悬赏 JSON 配置文件（`BOUNTY_CONFIG_PATH` 指向的人工维护列表）。

熔断开关（kill switch）的触发条件、查看方式与人工清除步骤见 README「运维：熔断开关」。
