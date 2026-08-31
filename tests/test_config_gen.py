from dataclasses import replace

import pytest
import yaml

from auto_adapter import config_gen
from auto_adapter.storage.sqlite import SqliteStorage


def test_initial_tp_size_is_always_conservative():
    """gpu_num 同时意味着"向平台申请几张卡"。猜大了会因"机器没这么多卡"直接失败，
    而重试梯子的调整方向是加大 tp——它修不好这类失败。猜小导致的 OOM 反而是
    梯子能修的（rung 2 翻倍）。所以首次提交一律从 1 张卡起步。"""
    for size in ("7B", "13.5B", "14B", "70B", "72B", None):
        assert config_gen.resolve_tp_size(size) == 1


def test_task_type_from_pipeline_tag(candidate):
    assert config_gen.resolve_task_type(candidate) == "text-generation"


def test_task_type_fallback_rules(candidate):
    c = replace(candidate, pipeline_tag=None, model_id="org/awesome-chat-model")
    assert config_gen.resolve_task_type(c) == "text-generation"
    unknown = replace(candidate, pipeline_tag=None, model_id="org/mystery")
    assert config_gen.resolve_task_type(unknown) is None


def test_render_config_params_is_valid_yaml_with_consistent_tp():
    text = config_gen.render_config_params("vllm", tp_size=2, max_model_len=2048)
    cfg = yaml.safe_load(text)
    assert cfg["framework"] == "vllm"
    sut_cmd = cfg["sut_config"]["values"]["command"]
    ref_cmd = cfg["ref_config"]["values"]["command"]
    assert sut_cmd[sut_cmd.index("-tp") + 1] == "2"
    assert ref_cmd[ref_cmd.index("-tp") + 1] == "2"
    assert cfg["sut_config"]["gpu_num"] == 2
    assert cfg["max_model_len"] == 2048


def test_render_config_params_transformers_template():
    text = config_gen.render_config_params("transformers", tp_size=4, max_model_len=8192)
    cfg = yaml.safe_load(text)
    assert cfg["framework"] == "transformers"
    sut_cmd = cfg["sut_config"]["values"]["command"]
    ref_cmd = cfg["ref_config"]["values"]["command"]
    assert sut_cmd[sut_cmd.index("-tp") + 1] == "4"
    assert ref_cmd[ref_cmd.index("-tp") + 1] == "4"
    assert cfg["sut_config"]["gpu_num"] == 4
    assert cfg["max_model_len"] == 8192


def test_select_target_gpu_prefers_lowest_coverage(tmp_path):
    store = SqliteStorage(str(tmp_path / "t.db"))
    from auto_adapter import rules
    assert config_gen.select_target_gpu(store) == rules.KNOWN_GPUS[0]  # 空覆盖率时取首位


def test_build_request(candidate):
    req = config_gen.build_request(candidate, "MetaX_c-500", "uuid-1")
    assert req.model_address == candidate.model_url
    assert req.task_type == "text-generation" and req.strategy_id == "uuid-1"
    assert "vllm" in req.config_params


def test_build_request_unresolvable_raises(candidate):
    c = replace(candidate, pipeline_tag=None, model_id="org/mystery")
    with pytest.raises(ValueError):
        config_gen.build_request(c, "MetaX_c-500", "uuid-1")


def test_build_request_refuses_non_vllm_candidate(candidate):
    """Ruling: v0.1 must not submit non-vllm candidates. templates/transformers.yaml is an
    explicit placeholder ("command 字段结构为占位符，待平台确认"), and the retry ladder only
    tunes tp/memory/context — it cannot fix a wrong launch command. Such a candidate would
    burn 3 submissions and end up permanently blacklisted. Route it to a human instead."""
    c = replace(candidate, pipeline_tag="feature-extraction")
    assert config_gen.resolve_framework(c) != "vllm"  # precondition

    with pytest.raises(config_gen.UnresolvableCandidateError) as exc:
        config_gen.build_request(c, "MetaX_c-500", "uuid-1")
    assert exc.value.reason == "unsupported_framework"
    assert exc.value.framework == "transformers"


def test_build_request_unresolvable_task_type_reason(candidate):
    c = replace(candidate, pipeline_tag=None, model_id="org/mystery")
    with pytest.raises(config_gen.UnresolvableCandidateError) as exc:
        config_gen.build_request(c, "MetaX_c-500", "uuid-1")
    assert exc.value.reason == "unresolvable_task_type"


def test_vllm_candidate_still_builds(candidate):
    req = config_gen.build_request(candidate, "MetaX_c-500", "uuid-1")
    assert req.framework == "vllm"


def test_known_gpus_covers_platform_enum():
    """GPU 型号取自平台任务列表页的筛选下拉框（2026-08-29 实测）。
    选卡逻辑要有意义就必须是全集——只有一个型号时 select_target_gpu 恒定返回它。"""
    from auto_adapter import rules
    assert set(rules.KNOWN_GPUS) == {
        "Ascend_910-b4", "MetaX_c-500", "Cambricon_mlu-370-x4",
        "Kunlunxin_p-800", "Kunlunxin_r-200-8f", "Iluvatar_bi-150",
        "Iluvatar_mrv-100", "hygon_k100-ai", "Vastai_va16", "Sunrise_pt-200-x1",
    }


def test_select_target_gpu_picks_least_covered_among_all(tmp_path):
    store = SqliteStorage(str(tmp_path / "t.db"))
    from auto_adapter import rules
    # 除 hygon_k100-ai 外都有覆盖 → 应选中它
    store.set_gpu_coverage({g: 5 for g in rules.KNOWN_GPUS if g != "hygon_k100-ai"})
    assert config_gen.select_target_gpu(store) == "hygon_k100-ai"


def test_rendered_config_has_no_developer_comments():
    """开发注释随 configParams 发给平台没有意义；而且注释里的占位符也会被 format
    替换，线上演练里出现过 "# 占位符 2/4096/0.9" 这种被替换过的残句。"""
    text = config_gen.render_config_params("vllm", tp_size=2)
    assert "#" not in text
    assert "占位符" not in text
    cfg = yaml.safe_load(text)  # 剥注释后仍是合法 YAML
    assert cfg["framework"] == "vllm" and cfg["sut_config"]["gpu_num"] == 2
