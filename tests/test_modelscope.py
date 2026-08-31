"""ModelScope 发现源。平台内网连不上 huggingface.co，它是那里唯一可达的模型源。"""
import responses

from auto_adapter.discovery.modelscope import ModelScopeSource
from auto_adapter.storage.sqlite import SqliteStorage

_API = "https://www.modelscope.cn/api/v1/dolphin/models"


def _payload(*models):
    return {"Data": {"Model": {"Models": list(models)}, "TotalCount": len(models)}}


def _model(org, name, tasks, downloads=1000):
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


@responses.activate
def test_applies_download_thresholds_per_task_type(tmp_path):
    """text-generation >50，其他类型 >5（业务规则）。"""
    responses.put(_API, json=_payload(
        _model("org", "textgen-ok", ["text-generation"], downloads=51),
        _model("org", "textgen-too-few", ["text-generation"], downloads=50),
        _model("org", "image-ok", ["text-to-image-synthesis"], downloads=6),
        _model("org", "image-too-few", ["text-to-image-synthesis"], downloads=5),
    ))
    store = SqliteStorage(str(tmp_path / "t.db"))
    src = ModelScopeSource(store, task_types=("text-generation", "text-to-image-synthesis"))

    got = src.fetch()

    assert [c.model_id for c in got] == ["org/textgen-ok", "org/image-ok"]
    assert got[1].pipeline_tag == "text-to-image-synthesis"


@responses.activate
def test_sorts_by_newest_not_by_downloads(tmp_path):
    """新模型才是"平台从没见过"的那批（5 积分/个）；按下载量排只会拿到热门老模型，
    实测那批里多数卡位都已被适配。"""
    responses.put(_API, json=_payload(_model("org", "new-7B", ["text-generation"], 999)))
    ModelScopeSource(SqliteStorage(str(tmp_path / "t.db"))).fetch()
    import json as _json
    assert _json.loads(responses.calls[0].request.body)["SortBy"] == "GmtCreated"


@responses.activate
def test_only_configured_task_types_are_admitted(tmp_path):
    """v0.1 只提交 vllm（text-generation）。放开其他类型会让队列被无法提交的候选
    占满——每个候选都要花一次 10s 的平台查询，而单 tick 只评估 20 个。"""
    responses.put(_API, json=_payload(
        _model("org", "textgen", ["text-generation"], 999),
        _model("org", "image", ["text-to-image-synthesis"], 999),
    ))
    store = SqliteStorage(str(tmp_path / "t.db"))
    got = ModelScopeSource(store).fetch()  # 默认只收 text-generation
    assert [c.model_id for c in got] == ["org/textgen"]


@responses.activate
def test_pages_deeper_until_enough_candidates(tmp_path):
    """一页 100 条只筛出个位数候选（"最新"与下载量门槛互相拉扯），
    只看一页产出率太低，智能体大部分时间会空转。"""
    page1 = _payload(*[_model("org", f"junk{i}", ["text-to-image-synthesis"], 1)
                       for i in range(100)])
    page2 = _payload(*([_model("org", "good-7B", ["text-generation"], 999)]
                       + [_model("org", f"junk2-{i}", ["text-generation"], 1)
                          for i in range(99)]))
    responses.put(_API, json=page1)
    responses.put(_API, json=page2)
    store = SqliteStorage(str(tmp_path / "t.db"))

    got = ModelScopeSource(store, limit=1).fetch()

    assert [c.model_id for c in got] == ["org/good-7B"]
    assert len(responses.calls) == 2  # 第一页不够，翻到第二页；够了就收手


@responses.activate
def test_stops_paging_at_the_end_of_the_list(tmp_path):
    """返回不足一页说明翻到底了，不该继续空打上游。"""
    responses.put(_API, json=_payload(_model("org", "only-7B", ["text-generation"], 999)))
    store = SqliteStorage(str(tmp_path / "t.db"))
    assert len(ModelScopeSource(store, limit=50).fetch()) == 1
    assert len(responses.calls) == 1
