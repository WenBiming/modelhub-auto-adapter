"""任务配置生成层（spec §4.4）。M4 实现。"""
from __future__ import annotations

from .models import AddTaskRequest, CandidateModel
from .storage import Storage


def resolve_task_type(candidate: CandidateModel) -> str | None:
    """pipeline_tag 直映射；缺失时查 rules.FALLBACK_TASK_TYPE_RULES；
    仍无法判定返回 None（调用方标记 NEEDS_HUMAN，不盲提）。"""
    raise NotImplementedError


def select_target_gpu(storage: Storage) -> str:
    """选平台覆盖率最低的 GPU 型号（覆盖率缓存于 storage）。"""
    raise NotImplementedError


def resolve_framework(candidate: CandidateModel) -> str:
    """架构在 vllm 支持列表 → vllm；否则退化到备选框架（rules.py 维护）。"""
    raise NotImplementedError


def resolve_tp_size(params_size: str | None) -> int:
    """参数量 → tensor 并行度：<14B→1，14–70B→2，>70B→4；未知按 1。"""
    raise NotImplementedError


def render_config_params(framework: str, tp_size: int, max_model_len: int = 4096,
                         gpu_mem_util: float = 0.9) -> str:
    """渲染 templates/{framework}.yaml 为 configParams YAML 字符串（spec 附录 A.1.1）。"""
    raise NotImplementedError


def build_request(
    candidate: CandidateModel, target_gpu: str, strategy_id: str
) -> AddTaskRequest:
    """组装完整请求体：上述推导结果 + render_config_params。"""
    raise NotImplementedError
