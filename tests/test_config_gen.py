from dataclasses import replace

import pytest
import yaml

from auto_adapter import config_gen
from auto_adapter.storage.sqlite import SqliteStorage


def test_tp_size_by_params():
    assert config_gen.resolve_tp_size("7B") == 1
    assert config_gen.resolve_tp_size("13.5B") == 1
    assert config_gen.resolve_tp_size("14B") == 2
    assert config_gen.resolve_tp_size("70B") == 2
    assert config_gen.resolve_tp_size("72B") == 4
    assert config_gen.resolve_tp_size(None) == 1


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
    assert config_gen.select_target_gpu(store) == "MetaX_c-500"  # 空覆盖率时取 KNOWN_GPUS[0]


def test_build_request(candidate):
    req = config_gen.build_request(candidate, "MetaX_c-500", "uuid-1")
    assert req.model_address == candidate.model_url
    assert req.task_type == "text-generation" and req.strategy_id == "uuid-1"
    assert "vllm" in req.config_params


def test_build_request_unresolvable_raises(candidate):
    c = replace(candidate, pipeline_tag=None, model_id="org/mystery")
    with pytest.raises(ValueError):
        config_gen.build_request(c, "MetaX_c-500", "uuid-1")
