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
