import pytest
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
