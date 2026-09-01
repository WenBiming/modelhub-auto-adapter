import pytest
import requests
import responses

from auto_adapter.models import AddTaskRequest
from auto_adapter.platform_client import PlatformClient, PlatformClientError

BASE = "https://modelhub.example"


@pytest.fixture
def client():
    return PlatformClient(BASE, xc_token="tok")


@responses.activate
def test_add_task_sends_camel_case_and_returns_id(client):
    responses.post(f"{BASE}/api/adapt/task/add", json={
        "code": 0, "message": "ok",
        "data": {"id": 123, "name": "t", "status": "PENDING",
                 "createTime": "2026-08-29T10:00:00Z", "updateTime": "2026-08-29T10:00:00Z"},
    })
    req = AddTaskRequest(
        model_address="https://huggingface.co/org/m", task_type="text-generation",
        target_gpu="MetaX_c-500", framework="vllm",
        config_params="framework: vllm\n", strategy_id="uuid-1",
    )
    assert client.add_task(req) == 123
    body = responses.calls[0].request
    assert body.headers["Xc-Token"] == "tok"
    import json as _json
    sent = _json.loads(body.body)
    assert sent == {
        "modelAddress": "https://huggingface.co/org/m", "taskType": "text-generation",
        "targetGpu": "MetaX_c-500", "framework": "vllm",
        "configParams": "framework: vllm\n", "strategyId": "uuid-1",
    }


@responses.activate
def test_nonzero_code_raises_with_credential_flag(client):
    responses.post(f"{BASE}/api/adapt/task/add",
                   json={"code": 40100, "message": "not login", "data": None})
    with pytest.raises(PlatformClientError) as ei:
        client.add_task(AddTaskRequest("a", "b", "c", "d", "e", "f"))
    assert ei.value.code == 40100 and ei.value.is_credential_error


@responses.activate
def test_search_model_maps_result(client):
    responses.get(
        f"{BASE}/api/computility/models/search-by-model-id",
        json={"code": 0, "message": "ok", "data": {
            "isInDB": True,
            "modelInfo": {"modelId": "org/m", "modelName": "m", "authorName": "org",
                          "source": "huggingface", "createTime": "2026-01-01T00:00:00Z"},
            "verifyResult": {"MetaX_c-500": {"passed": True}},
        }})
    r = client.search_model("org/m")
    assert r.is_in_db and "MetaX_c-500" in r.verify_result
    assert responses.calls[0].request.params["modelId"] == "org/m"


@responses.activate
def test_list_my_tasks_forces_only_mine(client):
    responses.get(f"{BASE}/api/adapt/task/page",
                  json={"code": 0, "message": "ok",
                        "data": {"records": [], "total": 0, "current": 1, "pages": 0, "size": 50}})
    page = client.list_my_tasks(current=2, page_size=20, status="RUNNING")
    p = responses.calls[0].request.params
    assert p["onlyMine"] == "true" and p["current"] == "2" and p["status"] == "RUNNING"
    assert page["records"] == []


@responses.activate
def test_get_task_log_and_stop_tasks(client):
    responses.get(f"{BASE}/api/adapt/task/log",
                  json={"code": 0, "message": "ok", "data": "CUDA out of memory"})
    assert client.get_task_log(123) == "CUDA out of memory"
    responses.put(f"{BASE}/api/async/task/stop-create-contest-task",
                  json={"code": 0, "message": "ok", "data": True})
    assert client.stop_tasks([1, 2]) is True
    import json as _json
    assert _json.loads(responses.calls[1].request.body) == {"taskIds": [1, 2]}


def test_http_401_counts_as_a_credential_failure():
    """线上实测：令牌无效时平台在 HTTP 层返回 401，raise_for_status 抛 HTTPError，
    走不到业务码 40100。只认业务码会让令牌过期这种最该拉闸的情形静默滑过去。"""
    from unittest.mock import Mock as _Mock

    from auto_adapter.platform_client import escalate_if_credential_error

    storage = _Mock()
    exc = requests.HTTPError("401 Client Error")
    exc.response = _Mock(status_code=401)

    assert escalate_if_credential_error(storage, exc) is True
    storage.set_kill_switch.assert_called_once()


def test_ordinary_http_error_is_not_a_credential_failure():
    from unittest.mock import Mock as _Mock

    from auto_adapter.platform_client import escalate_if_credential_error

    storage = _Mock()
    exc = requests.HTTPError("500 Server Error")
    exc.response = _Mock(status_code=500)

    assert escalate_if_credential_error(storage, exc) is False
    storage.set_kill_switch.assert_not_called()


@responses.activate
def test_sends_accept_json_like_the_documented_sample(client):
    """官方 API 文档示例带 Accept: application/json，照做以免内容协商上出现差异。"""
    responses.get(f"{BASE}/api/adapt/task/page",
                  json={"code": 0, "message": "ok", "data": {"records": []}})
    client.list_my_tasks()
    assert responses.calls[0].request.headers["Accept"] == "application/json"
    assert responses.calls[0].request.headers["Xc-Token"] == "tok"


@responses.activate
def test_auth_probe_adopts_the_scheme_that_works():
    """平台只注入 EXTERNAL_SERVICE_TOKEN（线上环境变量清单实证），而它用文档写明的
    Xc-Token 头被 401 拒绝。令牌应当是对的，差的是"怎么带"——逐个试标准方案，
    命中就换上，省掉一整个"改 Dockerfile 重新发版"的来回。"""
    from auto_adapter.platform_client import AUTH_SCHEMES

    def handler(request):
        if request.headers.get("Authorization") == "Bearer tok":
            return (200, {}, '{"code": 0, "message": "ok", "data": {"records": []}}')
        return (401, {}, "")

    responses.add_callback(responses.GET, f"{BASE}/api/adapt/task/page", callback=handler)
    client = PlatformClient(BASE, xc_token="tok")

    found = client.probe_auth()

    assert found == ("Authorization", "Bearer {token}")
    # 换上之后，后续正常调用应当通过
    assert client.list_my_tasks()["records"] == []
    assert ("Authorization", "Bearer {token}") in AUTH_SCHEMES


@responses.activate
def test_auth_probe_reports_failure_when_nothing_works():
    responses.get(f"{BASE}/api/adapt/task/page", status=401)
    assert PlatformClient(BASE, xc_token="tok").probe_auth() is None


@responses.activate
def test_auth_probe_never_creates_tasks():
    """探测只做只读 GET——绝不能在探路时建出任务来。"""
    responses.get(f"{BASE}/api/adapt/task/page", status=401)
    PlatformClient(BASE, xc_token="tok").probe_auth()
    assert all(c.request.method == "GET" for c in responses.calls)
