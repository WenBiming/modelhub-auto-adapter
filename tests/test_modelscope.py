"""ModelScope 发现源。平台内网连不上 huggingface.co，它是那里唯一可达的模型源。"""
import responses

from auto_adapter.discovery.modelscope import ModelScopeSource
from auto_adapter.storage.sqlite import SqliteStorage

_API = "https://www.modelscope.cn/api/v1/dolphin/models"


def _payload(*models):
    return {"Data": {"Model": {"Models": list(models)}, "TotalCount": len(models)}}


def _model(org, name, tasks, downloads=100):
    return {"Path": org, "Name": name, "Downloads": downloads,
            "Tasks": [{"Name": t} for t in tasks]}


@responses.activate
def test_filters_to_text_generation_and_builds_model_id(tmp_path):
    """按任务筛选走客户端过滤：SingleCriterion 的 schema 未公开，实测传猜测的形状
    会被静默忽略——"看起来生效其实没有"的筛选比不筛选更危险。"""
    responses.put(_API, json=_payload(
        _model("Qwen", "Qwen2.5-7B-Instruct", ["text-generation"], 7189402),
        _model("iic", "speech_fsmn_vad", ["voice-activity-detection"], 276241287),
        _model("OpenBMB", "cpm-bee-10b", ["text-generation"], 6508942),
    ))
    store = SqliteStorage(str(tmp_path / "t.db"))

    got = ModelScopeSource(store).fetch()

    assert [c.model_id for c in got] == ["Qwen/Qwen2.5-7B-Instruct", "OpenBMB/cpm-bee-10b"]
    assert got[0].model_url == "https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct"
    assert got[0].params_size == "7B" and got[1].params_size == "10B"
    assert got[0].source == "modelscope" and got[0].pipeline_tag == "text-generation"


@responses.activate
def test_throttle_survives_restart(tmp_path):
    """节流时间戳落盘：崩溃重启循环不能每次重启都再打一遍上游。"""
    responses.put(_API, json=_payload(_model("Qwen", "Qwen3-8B", ["text-generation"])))
    db = str(tmp_path / "t.db")

    assert len(ModelScopeSource(SqliteStorage(db)).fetch()) == 1
    # 换一个全新实例（模拟重启），仍应被节流挡住
    assert ModelScopeSource(SqliteStorage(db)).fetch() == []
    assert len(responses.calls) == 1


@responses.activate
def test_limit_is_respected(tmp_path):
    responses.put(_API, json=_payload(
        *[_model("org", f"m{i}-7B", ["text-generation"]) for i in range(10)]))
    store = SqliteStorage(str(tmp_path / "t.db"))
    assert len(ModelScopeSource(store, limit=3).fetch()) == 3
