"""共享 fixture：内存 SQLite storage、mock 平台客户端、样例候选模型。"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from auto_adapter.models import CandidateModel


@pytest.fixture
def candidate() -> CandidateModel:
    return CandidateModel(
        source="huggingface",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        pipeline_tag="text-generation",
        params_size="7B",
        is_bounty=False,
        bounty_deadline=None,
        discovered_at=datetime.now(timezone.utc),
    )
