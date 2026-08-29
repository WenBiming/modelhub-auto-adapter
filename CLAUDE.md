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
骨架阶段：接口与数据模型已定，函数体多为 `NotImplementedError` 占位。
按 spec §8 的里程碑 M1→M6 顺序实现；动手前先读 spec §9 的开放问题。
