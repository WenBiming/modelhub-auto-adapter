# 自动模型适配智能体 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 ModelHub XC 适配智能体：自动发现模型→去重准入→生成配置→限流提交→监控→失败重试的完整闭环，可打包为平台合规容器。

**Architecture:** Python 3.11 模块化单体，单进程同步调度循环 + Flask 健康检查线程；所有状态经 `Storage` 接口持久化到 SQLite；平台访问收敛在 `PlatformClient`。

**Tech Stack:** Python 3.11、requests、Flask、PyYAML、SQLite (stdlib sqlite3)、pytest + responses。

**Spec:** `docs/specs/2026-08-29-auto-model-adapter-design.md`（含附录 A 平台 API 契约，实现以其为准）

## Global Constraints

- 同一 `(model_id, target_gpu)` 组合绝不重复提交；平台查询失败时保守跳过（宁漏勿重）。
- `configParams` 是 YAML **字符串**（spec 附录 A.1.1）；平台任务 id 是 **int64**；终止接口是批量的。
- 所有平台/HF HTTP 调用 timeout ≤ 10s；每个 tick 步骤 < 30s（SIGTERM 优雅停机）。
- 凭据只从环境变量读取，不落日志。
- 测试不打真实网络：平台 API 用 `responses` mock；storage 用临时文件 SQLite。
- 每个任务 TDD：先写失败测试，再实现，测试全绿后提交。
- 运行测试统一用：`cd ~/dev/modelhub-auto-adapter && .venv/bin/pytest`（Task 1 建 venv）。

**本版本明确不做（YAGNI）**：ModelScope 发现源（保留空实现文件）、基于模型架构的 framework 精细判断（用 pipeline_tag 规则代替）、质量失败自动调优。

---

### Task 1: 开发环境 + SQLite Storage

**Files:**
- Modify: `src/auto_adapter/storage/sqlite.py`（实现全部方法）
- Modify: `src/auto_adapter/models.py`（TaskRecord 增加 `model_url`/`task_type` 字段）
- Test: `tests/test_storage.py`（重写，去掉 skip 占位）

**Interfaces:**
- Produces: `SqliteStorage(path)` 实现 `storage/base.py` 的 `Storage` Protocol 全部方法；`insert_task` 对重复 `(model_id, target_gpu)` 抛 `DuplicateTaskError`；`TaskRecord` 新增字段 `model_url: str = ""`、`task_type: str = ""`；`Storage` Protocol 新增 `get_counter(key: str) -> int`（缺省 0）与 `set_counter(key: str, value: int) -> None`（Task 8/9 的连续失败熔断用，在 `storage/base.py` 中补充这两个方法声明）。

- [ ] **Step 1: 创建 venv 并安装依赖**

```bash
cd ~/dev/modelhub-auto-adapter
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

（本机无 3.11 时用 3.12 开发，容器内仍为 3.11。）

- [ ] **Step 2: 给 TaskRecord 增加字段**

在 `models.py` 的 `TaskRecord` 中 `framework: str` 之后加：

```python
    model_url: str = ""
    task_type: str = ""
```

注意：这两个字段有默认值，必须放在无默认值字段之后（dataclass 规则），直接加在类体末尾的 `last_log` 附近也可以。

- [ ] **Step 3: 重写失败测试**

`tests/test_storage.py` 整体替换为：

```python
"""M1：storage 幂等与状态流转（spec §3 不变式、§4.8）。"""
from datetime import datetime, timezone

import pytest

from auto_adapter.models import Priority, TaskRecord, TaskStatus
from auto_adapter.storage.base import DuplicateTaskError
from auto_adapter.storage.sqlite import SqliteStorage


@pytest.fixture
def store(tmp_path):
    return SqliteStorage(str(tmp_path / "test.db"))


def _record(status=TaskStatus.QUEUED):
    return TaskRecord(
        model_id="org/model-a", target_gpu="MetaX_c-500", framework="vllm",
        status=status, priority=Priority.NEW_MODEL,
        model_url="https://huggingface.co/org/model-a", task_type="text-generation",
    )


def test_duplicate_task_rejected(store):
    store.insert_task(_record())
    with pytest.raises(DuplicateTaskError):
        store.insert_task(_record())


def test_task_roundtrip_and_status_query(store):
    rec = _record()
    store.insert_task(rec)
    rec.status = TaskStatus.PENDING
    rec.task_id = 42
    rec.submit_time = datetime(2026, 8, 29, tzinfo=timezone.utc)
    store.update_task(rec)
    got = store.tasks_by_status(TaskStatus.PENDING)
    assert len(got) == 1 and got[0].task_id == 42
    assert got[0].submit_time == rec.submit_time
    assert store.tasks_by_status(TaskStatus.QUEUED) == []


def test_blacklist(store):
    rec = _record(TaskStatus.BLACKLISTED)
    store.insert_task(rec)
    assert store.is_blacklisted("org/model-a", "MetaX_c-500")
    assert not store.is_blacklisted("org/model-a", "other-gpu")


def test_kill_switch_roundtrip(store):
    assert store.kill_switch() is False
    store.set_kill_switch(True, "credential error")
    assert store.kill_switch() is True


def test_counter_roundtrip(store):
    assert store.get_counter("consecutive_engine_failures") == 0
    store.set_counter("consecutive_engine_failures", 3)
    assert store.get_counter("consecutive_engine_failures") == 3


def test_gpu_coverage_roundtrip(store):
    assert store.gpu_coverage() == {}
    store.set_gpu_coverage({"MetaX_c-500": 3})
    assert store.gpu_coverage() == {"MetaX_c-500": 3}


def test_candidate_flow(store, candidate):
    store.upsert_candidate(candidate)
    store.upsert_candidate(candidate)  # 幂等
    assert [c.model_id for c in store.pending_candidates()] == [candidate.model_id]
    store.mark_candidate_processed(candidate.model_id)
    assert store.pending_candidates() == []
    store.upsert_candidate(candidate)  # 已处理的候选再 upsert 不复活
    assert store.pending_candidates() == []
```

- [ ] **Step 4: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_storage.py -v`
Expected: FAIL（NotImplementedError）

- [ ] **Step 5: 实现 SqliteStorage**

`storage/sqlite.py` 中方法体替换为（保留文件头与 `_SCHEMA`）：

```python
import json
from dataclasses import asdict
from datetime import datetime

from ..models import CandidateModel, Priority, TaskRecord, TaskStatus
from .base import DuplicateTaskError

_DT_FIELDS_TASK = ("submit_time", "bounty_deadline")
_DT_FIELDS_CAND = ("bounty_deadline", "discovered_at")


def _dump(obj, dt_fields) -> str:
    d = asdict(obj)
    for f in dt_fields:
        if d.get(f) is not None:
            d[f] = d[f].isoformat()
    return json.dumps(d)


def _load_task(payload: str) -> TaskRecord:
    d = json.loads(payload)
    for f in _DT_FIELDS_TASK:
        if d.get(f) is not None:
            d[f] = datetime.fromisoformat(d[f])
    d["status"] = TaskStatus(d["status"])
    d["priority"] = Priority(d["priority"])
    return TaskRecord(**d)


def _load_candidate(payload: str) -> CandidateModel:
    d = json.loads(payload)
    for f in _DT_FIELDS_CAND:
        if d.get(f) is not None:
            d[f] = datetime.fromisoformat(d[f])
    return CandidateModel(**d)


class SqliteStorage:
    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)

    def upsert_candidate(self, candidate: CandidateModel) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO candidates (model_id, payload) VALUES (?, ?) "
                "ON CONFLICT(model_id) DO UPDATE SET payload = excluded.payload",
                (candidate.model_id, _dump(candidate, _DT_FIELDS_CAND)),
            )

    def pending_candidates(self) -> list[CandidateModel]:
        rows = self._conn.execute(
            "SELECT payload FROM candidates WHERE processed = 0").fetchall()
        return [_load_candidate(r[0]) for r in rows]

    def mark_candidate_processed(self, model_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE candidates SET processed = 1 WHERE model_id = ?", (model_id,))

    def insert_task(self, record: TaskRecord) -> None:
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO tasks (model_id, target_gpu, payload, status) "
                    "VALUES (?, ?, ?, ?)",
                    (record.model_id, record.target_gpu,
                     _dump(record, _DT_FIELDS_TASK), record.status.value),
                )
        except sqlite3.IntegrityError as e:
            raise DuplicateTaskError(
                f"{record.model_id} @ {record.target_gpu}") from e

    def update_task(self, record: TaskRecord) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE tasks SET payload = ?, status = ? "
                "WHERE model_id = ? AND target_gpu = ?",
                (_dump(record, _DT_FIELDS_TASK), record.status.value,
                 record.model_id, record.target_gpu),
            )

    def tasks_by_status(self, *statuses: TaskStatus) -> list[TaskRecord]:
        marks = ",".join("?" * len(statuses))
        rows = self._conn.execute(
            f"SELECT payload FROM tasks WHERE status IN ({marks})",
            [s.value for s in statuses]).fetchall()
        return [_load_task(r[0]) for r in rows]

    def get_task(self, model_id: str, target_gpu: str) -> TaskRecord | None:
        row = self._conn.execute(
            "SELECT payload FROM tasks WHERE model_id = ? AND target_gpu = ?",
            (model_id, target_gpu)).fetchone()
        return _load_task(row[0]) if row else None

    def is_blacklisted(self, model_id: str, target_gpu: str) -> bool:
        rec = self.get_task(model_id, target_gpu)
        return rec is not None and rec.status == TaskStatus.BLACKLISTED

    def _kv_get(self, key: str, default):
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _kv_set(self, key: str, value) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

    def get_counter(self, key: str) -> int:
        return int(self._kv_get(f"counter:{key}", 0))

    def set_counter(self, key: str, value: int) -> None:
        self._kv_set(f"counter:{key}", value)

    def gpu_coverage(self) -> dict[str, int]:
        return self._kv_get("gpu_coverage", {})

    def set_gpu_coverage(self, coverage: dict[str, int]) -> None:
        self._kv_set("gpu_coverage", coverage)

    def kill_switch(self) -> bool:
        return self._kv_get("kill_switch", {"on": False})["on"]

    def set_kill_switch(self, on: bool, reason: str) -> None:
        self._kv_set("kill_switch", {"on": on, "reason": reason})
```

- [ ] **Step 6: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_storage.py -v` → 全部 PASS

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(M1): implement SqliteStorage with idempotent task table"
```

---

### Task 2: 主循环 + SIGTERM 优雅停机

**Files:**
- Modify: `src/auto_adapter/main.py`
- Test: `tests/test_main_loop.py`（新建）

**Interfaces:**
- Produces: `main.run_loop(tick_fn, stop_event, interval_seconds)`——循环调用 `tick_fn()` 直到 stop_event；`tick_fn` 抛异常时记日志继续（循环永不因单 tick 崩溃）。`main.tick(deps)` 的接线留给 Task 10，本任务先让 `main()` 用空 tick 跑通。

- [ ] **Step 1: 写失败测试**

`tests/test_main_loop.py`：

```python
import threading

from auto_adapter.main import run_loop


def test_run_loop_stops_on_event():
    stop = threading.Event()
    calls = []

    def tick():
        calls.append(1)
        if len(calls) >= 3:
            stop.set()

    run_loop(tick, stop, interval_seconds=0)
    assert len(calls) == 3


def test_run_loop_survives_tick_exception():
    stop = threading.Event()
    calls = []

    def tick():
        calls.append(1)
        if len(calls) >= 2:
            stop.set()
        raise RuntimeError("boom")

    run_loop(tick, stop, interval_seconds=0)
    assert len(calls) == 2
```

- [ ] **Step 2: 确认失败** — `.venv/bin/pytest tests/test_main_loop.py -v` → ImportError/FAIL

- [ ] **Step 3: 实现**

`main.py` 替换 `main`/`tick` 为：

```python
import logging

logger = logging.getLogger(__name__)


def run_loop(tick_fn, stop_event: threading.Event, interval_seconds: float) -> None:
    while not stop_event.is_set():
        try:
            tick_fn()
        except Exception:
            logger.exception("tick failed; will retry next tick")
        stop_event.wait(interval_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    health.start_in_background(port=8080)
    deps = build_deps(settings)
    run_loop(lambda: tick(deps, stop_event), stop_event, settings.tick_seconds)


def build_deps(settings: Settings):
    """构造 storage/client/sources。Task 10 完成实现，这里先占位。"""
    raise NotImplementedError


def tick(deps, stop_event: threading.Event) -> None:
    """Task 10 接线。"""
    raise NotImplementedError
```

- [ ] **Step 4: 确认通过** — `.venv/bin/pytest tests/test_main_loop.py -v` → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(M1): resilient main loop with SIGTERM stop event"`

---

### Task 3: PlatformClient（M2）

**Files:**
- Modify: `src/auto_adapter/platform_client.py`
- Test: `tests/test_platform_client.py`（新建）

**Interfaces:**
- Consumes: `models.AddTaskRequest`、`models.ModelSearchResult`
- Produces: `PlatformClient(base_url, xc_token)`，方法签名照 `platform_client.py` 现有 docstring：`add_task→int`、`list_my_tasks→dict`、`get_task_log→str`、`search_model→ModelSearchResult`、`stop_tasks→bool`。业务码非 0 抛 `PlatformClientError(code, message)`。**删除 `list_bounties`**（悬赏改为 Task 6 的人工配置源）。

- [ ] **Step 1: 写失败测试**

`tests/test_platform_client.py`：

```python
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
```

- [ ] **Step 2: 确认失败** — `.venv/bin/pytest tests/test_platform_client.py -v`

- [ ] **Step 3: 实现**

`platform_client.py` 的 `PlatformClient` 方法体：

```python
    def _request(self, method: str, path: str, *, params=None, json_body=None):
        resp = self._session.request(
            method, self._base_url + path,
            params=params, json=json_body, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != CODE_OK:
            raise PlatformClientError(body.get("code", -1), body.get("message", ""))
        return body.get("data")

    def add_task(self, req: AddTaskRequest) -> int:
        data = self._request("POST", "/api/adapt/task/add", json_body={
            "modelAddress": req.model_address, "taskType": req.task_type,
            "targetGpu": req.target_gpu, "framework": req.framework,
            "configParams": req.config_params, "strategyId": req.strategy_id,
        })
        return int(data["id"])

    def list_my_tasks(self, current: int = 1, page_size: int = 50, **filters) -> dict:
        params = {"current": current, "pageSize": page_size, "onlyMine": "true", **filters}
        return self._request("GET", "/api/adapt/task/page", params=params)

    def get_task_log(self, task_id: int) -> str:
        return self._request("GET", "/api/adapt/task/log", params={"taskId": task_id})

    def search_model(self, model_id: str) -> ModelSearchResult:
        data = self._request("GET", "/api/computility/models/search-by-model-id",
                             params={"modelId": model_id})
        return ModelSearchResult(
            is_in_db=bool(data.get("isInDB")),
            model_info=data.get("modelInfo") or {},
            verify_result=data.get("verifyResult") or {},
        )

    def stop_tasks(self, task_ids: list[int]) -> bool:
        return bool(self._request("PUT", "/api/async/task/stop-create-contest-task",
                                  json_body={"taskIds": task_ids}))
```

同时删除 `list_bounties` 方法，并删除 `discovery/bounty.py` 对它的引用（Task 6 重写该文件）。

- [ ] **Step 4: 确认通过**，且全量 `.venv/bin/pytest` 无回归
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(M2): implement PlatformClient against appendix A contract"`

---

### Task 4: 去重与准入 eligibility（M3）

**Files:**
- Modify: `src/auto_adapter/eligibility.py`
- Test: `tests/test_eligibility.py`（重写）

**Interfaces:**
- Consumes: `Storage.get_task`、`PlatformClient.search_model`
- Produces: `evaluate(candidate, target_gpu, storage, client) -> Decision`（语义照现有 docstring 分类矩阵）

- [ ] **Step 1: 重写失败测试**

`tests/test_eligibility.py`：

```python
from dataclasses import replace
from unittest.mock import Mock

import pytest

from auto_adapter.eligibility import Verdict, evaluate
from auto_adapter.models import ModelSearchResult, Priority
from auto_adapter.storage.sqlite import SqliteStorage

GPU = "MetaX_c-500"


@pytest.fixture
def store(tmp_path):
    return SqliteStorage(str(tmp_path / "t.db"))


def _client(verify_result=None, error=False):
    client = Mock()
    if error:
        client.search_model.side_effect = ConnectionError("down")
    else:
        client.search_model.return_value = ModelSearchResult(
            is_in_db=bool(verify_result), model_info={}, verify_result=verify_result or {})
    return client


def test_new_model_enqueued(store, candidate):
    d = evaluate(candidate, GPU, store, _client({}))
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.NEW_MODEL


def test_new_adaptation_when_other_gpu_verified(store, candidate):
    d = evaluate(candidate, GPU, store, _client({"other-gpu": {"passed": True}}))
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.NEW_ADAPTATION


def test_same_gpu_verified_skipped(store, candidate):
    d = evaluate(candidate, GPU, store, _client({GPU: {"passed": True}}))
    assert d.verdict == Verdict.SKIP_DUPLICATE


def test_local_record_skipped_without_platform_query(store, candidate):
    from tests.test_storage import _record
    store.insert_task(_record())
    c = replace(candidate, model_id="org/model-a")
    client = _client({})
    d = evaluate(c, GPU, store, client)
    assert d.verdict == Verdict.SKIP_DUPLICATE
    client.search_model.assert_not_called()


def test_bounty_gets_top_priority(store, candidate):
    c = replace(candidate, is_bounty=True)
    d = evaluate(c, GPU, store, _client({"other-gpu": {}}))
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.BOUNTY


def test_platform_error_skips_conservatively(store, candidate):
    d = evaluate(candidate, GPU, store, _client(error=True))
    assert d.verdict == Verdict.SKIP_UNCERTAIN
```

- [ ] **Step 2: 确认失败**

- [ ] **Step 3: 实现 `evaluate`**

```python
def evaluate(candidate, target_gpu, storage, client) -> Decision:
    if storage.get_task(candidate.model_id, target_gpu) is not None:
        return Decision(Verdict.SKIP_DUPLICATE, reason="local record exists")
    try:
        result = client.search_model(candidate.model_id)
    except Exception as e:  # 平台不可知时宁漏勿重
        return Decision(Verdict.SKIP_UNCERTAIN, reason=f"platform query failed: {e}")
    if target_gpu in result.verify_result:
        # GpuVerifyResult 内部字段未确认前，键存在即视为已覆盖（保守）
        return Decision(Verdict.SKIP_DUPLICATE, reason=f"already verified on {target_gpu}")
    if candidate.is_bounty:
        return Decision(Verdict.ENQUEUE, Priority.BOUNTY, reason="bounty")
    if not result.verify_result:
        return Decision(Verdict.ENQUEUE, Priority.NEW_MODEL, reason="no adaptation record")
    return Decision(Verdict.ENQUEUE, Priority.NEW_ADAPTATION, reason="new gpu for model")
```

- [ ] **Step 4: 确认通过** - [ ] **Step 5: Commit** — `git commit -am "feat(M3): eligibility dedup with conservative fallback"`

---

### Task 5: 配置生成 config_gen（M4 前置）

**Files:**
- Modify: `src/auto_adapter/config_gen.py`、`src/auto_adapter/rules.py`
- Test: `tests/test_config_gen.py`（重写）

**Interfaces:**
- Produces: `resolve_task_type(candidate)->str|None`、`resolve_tp_size(params_size)->int`、`resolve_framework(candidate)->str`、`select_target_gpu(storage)->str`、`render_config_params(framework,tp_size,max_model_len=4096,gpu_mem_util=0.9)->str`、`build_request(candidate,target_gpu,strategy_id)->AddTaskRequest`（taskType 无法判定时抛 `ValueError`）。
- rules.py 新增：`KNOWN_GPUS = ["MetaX_c-500"]`（唯一已确认型号，后续人工扩充）。

- [ ] **Step 1: 重写失败测试**

`tests/test_config_gen.py`：

```python
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
    assert cfg["sut_config"]["gpu_num"] == "2"


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
```

- [ ] **Step 2: 确认失败**

- [ ] **Step 3: 实现**

`rules.py` 追加 `KNOWN_GPUS = ["MetaX_c-500"]`。`config_gen.py`：

```python
import re
from pathlib import Path

from . import rules

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def resolve_task_type(candidate) -> str | None:
    if candidate.model_id in rules.MANUAL_OVERRIDES:
        return rules.MANUAL_OVERRIDES[candidate.model_id][0]
    if candidate.pipeline_tag:
        return candidate.pipeline_tag
    lowered = candidate.model_id.lower()
    for keyword, task_type in rules.FALLBACK_TASK_TYPE_RULES:
        if keyword in lowered:
            return task_type
    return None


def resolve_tp_size(params_size: str | None) -> int:
    if not params_size:
        return 1
    m = re.match(r"([\d.]+)\s*B", params_size, re.IGNORECASE)
    if not m:
        return 1
    billions = float(m.group(1))
    if billions < 14:
        return 1
    if billions <= 70:
        return 2
    return 4


def resolve_framework(candidate) -> str:
    if candidate.model_id in rules.MANUAL_OVERRIDES:
        return rules.MANUAL_OVERRIDES[candidate.model_id][1]
    # v0.1：text-generation 走 vllm，其余退化（架构级判断见 spec §6 迭代方向）
    if resolve_task_type(candidate) == "text-generation":
        return "vllm"
    return rules.FALLBACK_FRAMEWORK


def select_target_gpu(storage) -> str:
    coverage = storage.gpu_coverage()
    return min(rules.KNOWN_GPUS, key=lambda g: coverage.get(g, 0))


def render_config_params(framework: str, tp_size: int,
                         max_model_len: int = 4096, gpu_mem_util: float = 0.9) -> str:
    template = (_TEMPLATE_DIR / f"{framework}.yaml").read_text()
    return template.format(tp_size=tp_size, max_model_len=max_model_len,
                           gpu_mem_util=gpu_mem_util)


def build_request(candidate, target_gpu, strategy_id):
    task_type = resolve_task_type(candidate)
    if task_type is None:
        raise ValueError(f"cannot resolve task type for {candidate.model_id}")
    framework = resolve_framework(candidate)
    config = render_config_params(framework, resolve_tp_size(candidate.params_size))
    from .models import AddTaskRequest
    return AddTaskRequest(
        model_address=candidate.model_url, task_type=task_type,
        target_gpu=target_gpu, framework=framework,
        config_params=config, strategy_id=strategy_id,
    )
```

注意：非 vllm 框架暂无模板文件，`render_config_params` 会抛 FileNotFoundError——v0.1 只有 vllm 模板，`FALLBACK_FRAMEWORK` 路径需在 rules 中把 fallback 也指向存在模板的框架，或复制 `vllm.yaml` 为 `transformers.yaml` 简化处理：本计划选择**复制模板并去掉 vllm 专属参数**（新建 `templates/transformers.yaml`，framework 字段改 transformers，command 留 vllm 同结构占位并在文件头注明待平台确认后修正）。

- [ ] **Step 4: 确认通过** - [ ] **Step 5: Commit** — `git commit -am "feat(M4): config generation with yaml template rendering"`

---

### Task 6: 发现层 discovery（M3）

**Files:**
- Modify: `src/auto_adapter/discovery/base.py`（实现 `run`）、`discovery/huggingface.py`、重写 `discovery/bounty.py` 为人工配置源
- Modify: `src/auto_adapter/settings.py`（新增 `bounty_config_path: str = ""`、`hf_fetch_limit: int = 50`）
- Test: `tests/test_discovery.py`（新建）

**Interfaces:**
- Produces: `discovery.base.run(sources, storage) -> int`；`HuggingFaceSource(limit, min_interval_seconds).fetch()`（HF API `GET https://huggingface.co/api/models?sort=downloads&direction=-1&limit=N&pipeline_tag=text-generation`，内部节流：距上次成功 fetch 不足 interval 时返回 `[]`）；`ManualBountySource(path).fetch()` 读 JSON 文件 `[{"model_id","model_url","deadline"}]`，文件不存在返回 `[]`。

- [ ] **Step 1: 写失败测试**

`tests/test_discovery.py`：

```python
import json
from datetime import datetime, timezone

import responses

from auto_adapter.discovery.base import run
from auto_adapter.discovery.bounty import ManualBountySource
from auto_adapter.discovery.huggingface import HuggingFaceSource
from auto_adapter.storage.sqlite import SqliteStorage


@responses.activate
def test_huggingface_fetch_and_throttle():
    responses.get("https://huggingface.co/api/models", json=[
        {"modelId": "Qwen/Qwen2.5-7B-Instruct", "pipeline_tag": "text-generation"},
        {"id": "org/other-13B", "pipeline_tag": "text-generation"},
    ])
    src = HuggingFaceSource(limit=2, min_interval_seconds=3600)
    got = src.fetch()
    assert [c.model_id for c in got] == ["Qwen/Qwen2.5-7B-Instruct", "org/other-13B"]
    assert got[0].params_size == "7B"
    assert got[0].model_url == "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"
    assert src.fetch() == []  # 节流：1h 内第二次返回空
    assert len(responses.calls) == 1


def test_manual_bounty_source(tmp_path):
    path = tmp_path / "bounty.json"
    path.write_text(json.dumps([{
        "model_id": "org/bounty-model",
        "model_url": "https://huggingface.co/org/bounty-model",
        "deadline": "2026-09-30T00:00:00+00:00",
    }]))
    got = ManualBountySource(str(path)).fetch()
    assert got[0].is_bounty and got[0].bounty_deadline == datetime(2026, 9, 30, tzinfo=timezone.utc)
    assert ManualBountySource(str(tmp_path / "missing.json")).fetch() == []


def test_run_dedups_and_persists(tmp_path, candidate):
    store = SqliteStorage(str(tmp_path / "t.db"))

    class Fake:
        name = "fake"
        def fetch(self):
            return [candidate, candidate]

    class Broken:
        name = "broken"
        def fetch(self):
            raise ConnectionError("down")

    assert run([Fake(), Broken()], store) == 1  # 去重 + 单源故障不影响整体
    assert len(store.pending_candidates()) == 1
```

- [ ] **Step 2: 确认失败**

- [ ] **Step 3: 实现**

`discovery/base.py` 的 `run`：

```python
import logging

logger = logging.getLogger(__name__)


def run(sources, storage) -> int:
    count, seen = 0, set()
    for src in sources:
        try:
            candidates = src.fetch()
        except Exception:
            logger.exception("discovery source %s failed", src.name)
            continue
        for c in candidates:
            if c.model_id in seen:
                continue
            seen.add(c.model_id)
            storage.upsert_candidate(c)
            count += 1
    return count
```

`discovery/huggingface.py`：

```python
import re
import time
from datetime import datetime, timezone

import requests

from ..models import CandidateModel

_API = "https://huggingface.co/api/models"
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)[bB]\b")


class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, limit: int = 50, min_interval_seconds: int = 3600) -> None:
        self._limit = limit
        self._min_interval = min_interval_seconds
        self._last_fetch = 0.0

    def fetch(self) -> list[CandidateModel]:
        if time.monotonic() - self._last_fetch < self._min_interval and self._last_fetch:
            return []
        resp = requests.get(_API, params={
            "sort": "downloads", "direction": -1,
            "limit": self._limit, "pipeline_tag": "text-generation",
        }, timeout=10)
        resp.raise_for_status()
        self._last_fetch = time.monotonic()
        out = []
        for item in resp.json():
            model_id = item.get("modelId") or item.get("id")
            if not model_id:
                continue
            m = _PARAMS_RE.search(model_id)
            out.append(CandidateModel(
                source="huggingface", model_id=model_id,
                model_url=f"https://huggingface.co/{model_id}",
                pipeline_tag=item.get("pipeline_tag"),
                params_size=f"{m.group(1)}B" if m else None,
                is_bounty=False, bounty_deadline=None,
                discovered_at=datetime.now(timezone.utc),
            ))
        return out
```

`discovery/bounty.py` 重写：

```python
"""悬赏来源：平台无悬赏 API（spec §9），v0.1 用人工维护的 JSON 配置文件。

文件格式：[{"model_id": str, "model_url": str, "deadline": ISO8601}]
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..models import CandidateModel

logger = logging.getLogger(__name__)


class ManualBountySource:
    name = "bounty"

    def __init__(self, path: str) -> None:
        self._path = Path(path) if path else None

    def fetch(self) -> list[CandidateModel]:
        if self._path is None or not self._path.exists():
            return []
        items = json.loads(self._path.read_text())
        return [CandidateModel(
            source="bounty", model_id=i["model_id"], model_url=i["model_url"],
            pipeline_tag=i.get("pipeline_tag"), params_size=i.get("params_size"),
            is_bounty=True,
            bounty_deadline=datetime.fromisoformat(i["deadline"]) if i.get("deadline") else None,
            discovered_at=datetime.now(timezone.utc),
        ) for i in items]
```

`settings.py` 增加两个字段（含 `from_env` 读取 `BOUNTY_CONFIG_PATH`、`HF_FETCH_LIMIT`）。`modelscope.py` 保持占位并在 docstring 标注「v0.1 范围外」。

- [ ] **Step 4: 确认通过** - [ ] **Step 5: Commit** — `git commit -am "feat(M3): HF + manual bounty discovery with dedup"`

---

### Task 7: 提交调度 submitter（M4）

**Files:**
- Modify: `src/auto_adapter/submitter.py`
- Test: `tests/test_submitter.py`（新建）

**Interfaces:**
- Consumes: `Storage.tasks_by_status/update_task`、`PlatformClient.add_task`、`Settings`
- Produces: `drain(storage, client, settings, now=None) -> int`。排序 `(priority, bounty_deadline↑, model_id)`；每 tick 最多提交 `max_submits_per_minute`（tick=60s 时即每分钟限流，`tick_seconds<60` 的部署被 Settings 校验拒绝——在 `Settings.from_env` 里加 `tick_seconds >= 60` 断言）；在途 ≥ `max_inflight` 停止；kill_switch 时返回 0；凭据错误触发 kill_switch；悬赏剩余 < `2 * EST_ADAPT_HOURS(=2h)` 标记 ABANDONED。

- [ ] **Step 1: 写失败测试**

`tests/test_submitter.py`：

```python
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from auto_adapter import submitter
from auto_adapter.models import Priority, TaskRecord, TaskStatus
from auto_adapter.platform_client import PlatformClientError
from auto_adapter.settings import Settings
from auto_adapter.storage.sqlite import SqliteStorage

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
SETTINGS = Settings(xc_token="t", strategy_id="s", base_url="https://x",
                    max_submits_per_minute=2, max_inflight=3)


@pytest.fixture
def store(tmp_path):
    return SqliteStorage(str(tmp_path / "t.db"))


def _queued(model_id, priority=Priority.NEW_MODEL, deadline=None):
    return TaskRecord(model_id=model_id, target_gpu="MetaX_c-500", framework="vllm",
                      status=TaskStatus.QUEUED, priority=priority,
                      model_url=f"https://huggingface.co/{model_id}",
                      task_type="text-generation", config_params="framework: vllm\n",
                      bounty_deadline=deadline)


def test_drain_respects_priority_and_rate_limit(store):
    store.insert_task(_queued("org/new"))
    store.insert_task(_queued("org/bounty", Priority.BOUNTY, NOW + timedelta(days=2)))
    store.insert_task(_queued("org/adapt", Priority.NEW_ADAPTATION))
    client = Mock()
    client.add_task.side_effect = [1, 2]
    assert submitter.drain(store, client, SETTINGS, now=NOW) == 2  # 限流=2
    submitted_types = [c.args[0].model_address for c in client.add_task.call_args_list]
    assert submitted_types == ["https://huggingface.co/org/bounty",
                               "https://huggingface.co/org/new"]
    pending = store.tasks_by_status(TaskStatus.PENDING)
    assert {r.task_id for r in pending} == {1, 2}
    assert all(r.submit_time == NOW for r in pending)


def test_drain_respects_inflight_cap(store):
    for i in range(3):
        rec = _queued(f"org/m{i}")
        rec.status = TaskStatus.RUNNING
        store.insert_task(rec)
    store.insert_task(_queued("org/new"))
    client = Mock()
    assert submitter.drain(store, client, SETTINGS, now=NOW) == 0
    client.add_task.assert_not_called()


def test_kill_switch_blocks(store):
    store.insert_task(_queued("org/new"))
    store.set_kill_switch(True, "test")
    assert submitter.drain(store, Mock(), SETTINGS, now=NOW) == 0


def test_credential_error_sets_kill_switch(store):
    store.insert_task(_queued("org/new"))
    client = Mock()
    client.add_task.side_effect = PlatformClientError(40100, "not login")
    assert submitter.drain(store, client, SETTINGS, now=NOW) == 0
    assert store.kill_switch() is True


def test_expiring_bounty_abandoned(store):
    store.insert_task(_queued("org/late", Priority.BOUNTY, NOW + timedelta(hours=1)))
    assert submitter.drain(store, Mock(), SETTINGS, now=NOW) == 0
    assert store.tasks_by_status(TaskStatus.ABANDONED)[0].model_id == "org/late"
```

- [ ] **Step 2: 确认失败**

- [ ] **Step 3: 实现 `drain`**

```python
import logging
from datetime import datetime, timedelta, timezone

from .models import ACTIVE_STATUSES, AddTaskRequest, TaskStatus
from .platform_client import PlatformClientError

logger = logging.getLogger(__name__)
EST_ADAPT_HOURS = 2
_MAX_DT = datetime.max.replace(tzinfo=timezone.utc)


def drain(storage, client, settings, now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    if storage.kill_switch():
        logger.warning("kill switch on; submission paused")
        return 0
    inflight = len(storage.tasks_by_status(TaskStatus.PENDING, TaskStatus.RUNNING))
    budget = min(settings.max_submits_per_minute, settings.max_inflight - inflight)
    if budget <= 0:
        return 0
    queued = sorted(storage.tasks_by_status(TaskStatus.QUEUED),
                    key=lambda r: (r.priority, r.bounty_deadline or _MAX_DT, r.model_id))
    submitted = 0
    for record in queued:
        if submitted >= budget:
            break
        if record.bounty_deadline is not None and \
                record.bounty_deadline - now < timedelta(hours=2 * EST_ADAPT_HOURS):
            record.status = TaskStatus.ABANDONED
            storage.update_task(record)
            logger.warning("bounty %s abandoned: deadline too close", record.model_id)
            continue
        req = AddTaskRequest(
            model_address=record.model_url, task_type=record.task_type,
            target_gpu=record.target_gpu, framework=record.framework,
            config_params=record.config_params, strategy_id=settings.strategy_id)
        try:
            record.task_id = client.add_task(req)
        except PlatformClientError as e:
            if e.is_credential_error:
                storage.set_kill_switch(True, str(e))
                return submitted
            logger.warning("submit failed for %s: %s", record.model_id, e)
            continue
        except Exception:
            logger.exception("submit failed for %s", record.model_id)
            continue
        record.submit_time = now
        record.status = TaskStatus.PENDING
        storage.update_task(record)
        submitted += 1
    return submitted
```

同时在 `Settings.from_env` 末尾（构造前）加校验：`tick_seconds` < 60 时抛 `ValueError("TICK_SECONDS must be >= 60 (rate limit assumes one drain per minute)")`。

- [ ] **Step 4: 确认通过** - [ ] **Step 5: Commit** — `git commit -am "feat(M4): priority-ordered rate-limited submitter"`

---

### Task 8: 监控对账 monitor（M5）

**Files:**
- Modify: `src/auto_adapter/monitor.py`、`src/auto_adapter/rules.py`（新增状态映射）
- Test: `tests/test_monitor.py`（新建）

**Interfaces:**
- Consumes: `PlatformClient.list_my_tasks/get_task_log`、`failure.classify`（Task 9 定义；本任务先在 monitor 内只把失败置为 `ENGINE_FAILED`/`QUALITY_FAILED` 之前的中间态——为避免前向依赖，映射到失败时直接调用 `rules.classify_failure_status`，见下）
- Produces: `poll(storage, client, settings, now=None)`；`rules.PLATFORM_STATUS_MAP: dict[str, TaskStatus|str]`，值 `"failed"` 表示需拉日志分类；`rules.map_platform_status(status_str) -> TaskStatus | "failed" | None`（None=未知，保持原状并 warning——枚举确认后补全，spec §9）。

- [ ] **Step 1: 写失败测试**

`tests/test_monitor.py`：

```python
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from auto_adapter import monitor
from auto_adapter.models import Priority, TaskRecord, TaskStatus
from auto_adapter.settings import Settings
from auto_adapter.storage.sqlite import SqliteStorage

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
SETTINGS = Settings(xc_token="t", strategy_id="s", base_url="https://x", task_timeout_hours=6)


@pytest.fixture
def store(tmp_path):
    return SqliteStorage(str(tmp_path / "t.db"))


def _pending(model_id, task_id, submit_time=NOW):
    return TaskRecord(model_id=model_id, target_gpu="MetaX_c-500", framework="vllm",
                      status=TaskStatus.PENDING, priority=Priority.NEW_MODEL,
                      task_id=task_id, submit_time=submit_time)


def _page(*rows):
    return {"records": list(rows), "total": len(rows), "current": 1, "pages": 1, "size": 100}


def test_status_sync_success_and_failure(store):
    store.insert_task(_pending("org/a", 1))
    store.insert_task(_pending("org/b", 2))
    client = Mock()
    client.list_my_tasks.return_value = _page(
        {"taskId": 1, "status": "SUCCESS"},
        {"taskId": 2, "status": "FAILED"},
    )
    client.get_task_log.return_value = "CUDA out of memory"
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.SUCCESS
    failed = store.get_task("org/b", "MetaX_c-500")
    assert failed.status == TaskStatus.ENGINE_FAILED
    assert failed.last_log == "CUDA out of memory"


def test_vanished_task_triggers_kill_switch(store):
    store.insert_task(_pending("org/a", 1))
    client = Mock()
    client.list_my_tasks.return_value = _page()
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.ABANDONED
    assert store.kill_switch() is True


def test_stuck_task_marked_timeout(store):
    store.insert_task(_pending("org/a", 1, submit_time=NOW - timedelta(hours=7)))
    client = Mock()
    client.list_my_tasks.return_value = _page({"taskId": 1, "status": "RUNNING"})
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.TIMEOUT


def test_unknown_status_left_unchanged(store):
    store.insert_task(_pending("org/a", 1))
    client = Mock()
    client.list_my_tasks.return_value = _page({"taskId": 1, "status": "WEIRD_STATE"})
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.PENDING
```

- [ ] **Step 2: 确认失败**

- [ ] **Step 3: 实现**

`rules.py` 追加（状态枚举未确认前的最优猜测 + 保守未知处理，spec §9）：

```python
from .models import TaskStatus

PLATFORM_STATUS_MAP: dict[str, object] = {
    "PENDING": TaskStatus.PENDING, "QUEUED": TaskStatus.PENDING,
    "RUNNING": TaskStatus.RUNNING,
    "SUCCESS": TaskStatus.SUCCESS, "SUCCEED": TaskStatus.SUCCESS,
    "FAILED": "failed", "FAIL": "failed", "ERROR": "failed",
}


def map_platform_status(status: str):
    return PLATFORM_STATUS_MAP.get((status or "").upper())
```

`monitor.py` 的 `poll`：

```python
import logging
from datetime import datetime, timedelta, timezone

from . import rules
from .models import TaskStatus

logger = logging.getLogger(__name__)


def poll(storage, client, settings, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    records = storage.tasks_by_status(TaskStatus.PENDING, TaskStatus.RUNNING)
    if not records:
        return
    try:
        page = client.list_my_tasks(page_size=100)
    except Exception:
        logger.exception("list_my_tasks failed; will retry next tick")
        return
    platform_rows = {row["taskId"]: row for row in page.get("records", [])}
    timeout = timedelta(hours=settings.task_timeout_hours)
    for rec in records:
        row = platform_rows.get(rec.task_id)
        if row is None:
            rec.status = TaskStatus.ABANDONED
            storage.update_task(rec)
            storage.set_kill_switch(
                True, f"task {rec.task_id} vanished from platform (possible violation cleanup)")
            logger.error("task %s vanished from platform; kill switch ON", rec.task_id)
            continue
        mapped = rules.map_platform_status(row.get("status"))
        if mapped is None:
            logger.warning("unknown platform status %r for task %s", row.get("status"), rec.task_id)
        elif mapped == "failed":
            try:
                rec.last_log = client.get_task_log(rec.task_id)
            except Exception:
                logger.exception("get_task_log failed for %s", rec.task_id)
                rec.last_log = ""
            from .failure import classify  # 局部导入避免模块环
            from .models import FailureKind
            kind = classify(rec.last_log)
            rec.status = (TaskStatus.QUALITY_FAILED if kind == FailureKind.QUALITY
                          else TaskStatus.ENGINE_FAILED)
            storage.update_task(rec)
            continue
        elif mapped != rec.status:
            if mapped == TaskStatus.SUCCESS:
                storage.set_counter("consecutive_engine_failures", 0)  # 成功即重置熔断计数
            rec.status = mapped
            storage.update_task(rec)
            continue
        if rec.submit_time is not None and now - rec.submit_time > timeout:
            rec.status = TaskStatus.TIMEOUT
            storage.update_task(rec)
```

注意 monitor 依赖 `failure.classify`——Task 8 与 Task 9 需同批实现；若严格按序，本任务先在 `failure.py` 只实现 `classify`（见 Task 9 Step 3 的 classify 代码，提前搬入），Task 9 补 `next_config`/`handle`。

- [ ] **Step 4: 确认通过** - [ ] **Step 5: Commit** — `git commit -am "feat(M5): reconciliation monitor with vanish alarm and timeout"`

---

### Task 9: 失败分类与重试 failure（M5）

**Files:**
- Modify: `src/auto_adapter/failure.py`
- Test: `tests/test_failure.py`（重写）

**Interfaces:**
- Produces: `classify(log)->FailureKind`（质量关键词优先，默认 ENGINE）、`next_config(record)->str|None`（YAML 调参梯子：retry 0→`gpu_mem_util` 提至 0.95；1→`max_model_len` 减半下限 2048；2→tp 翻倍上限 4；≥3→None）、`handle(storage, client, settings, now=None)`。
- 熔断（spec §6）：`handle` 每处理一条引擎失败给 `consecutive_engine_failures` 计数 +1，达到 `_STREAK_LIMIT = 5` 时 `set_kill_switch(True, ...)` 并 ERROR 日志（成功任务由 Task 8 的 monitor 清零该计数）。

- [ ] **Step 1: 重写失败测试**

`tests/test_failure.py`：

```python
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
import yaml

from auto_adapter import config_gen, failure
from auto_adapter.models import FailureKind, Priority, TaskRecord, TaskStatus
from auto_adapter.settings import Settings
from auto_adapter.storage.sqlite import SqliteStorage

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
SETTINGS = Settings(xc_token="t", strategy_id="s", base_url="https://x", max_retries=3)


@pytest.fixture
def store(tmp_path):
    return SqliteStorage(str(tmp_path / "t.db"))


def _failed(status=TaskStatus.ENGINE_FAILED, retry_count=0, deadline=None):
    return TaskRecord(model_id="org/m", target_gpu="MetaX_c-500", framework="vllm",
                      status=status, priority=Priority.NEW_MODEL, task_id=7,
                      retry_count=retry_count, bounty_deadline=deadline,
                      config_params=config_gen.render_config_params("vllm", 1))


def test_classify():
    assert failure.classify("CUDA out of memory on device") == FailureKind.ENGINE
    assert failure.classify("LLM judge score below threshold") == FailureKind.QUALITY
    assert failure.classify("some unknown garbage") == FailureKind.ENGINE


def test_next_config_ladder():
    rec = _failed(retry_count=0)
    cfg1 = yaml.safe_load(failure.next_config(rec))
    cmd = cfg1["sut_config"]["values"]["command"]
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.95"
    rec.retry_count = 1
    cfg2 = yaml.safe_load(failure.next_config(rec))
    assert cfg2["max_model_len"] == 2048
    rec.retry_count = 2
    cfg3 = yaml.safe_load(failure.next_config(rec))
    sut = cfg3["sut_config"]["values"]["command"]
    ref = cfg3["ref_config"]["values"]["command"]
    assert sut[sut.index("-tp") + 1] == "2" and ref[ref.index("-tp") + 1] == "2"
    rec.retry_count = 3
    assert failure.next_config(rec) is None


def test_handle_requeues_engine_failure_then_blacklists(store):
    store.insert_task(_failed(retry_count=0))
    failure.handle(store, Mock(), SETTINGS, now=NOW)
    rec = store.get_task("org/m", "MetaX_c-500")
    assert rec.status == TaskStatus.QUEUED and rec.retry_count == 1 and rec.task_id is None
    rec.status = TaskStatus.ENGINE_FAILED
    rec.retry_count = 3
    store.update_task(rec)
    failure.handle(store, Mock(), SETTINGS, now=NOW)
    assert store.is_blacklisted("org/m", "MetaX_c-500")


def test_handle_quality_failure_needs_human(store):
    store.insert_task(_failed(TaskStatus.QUALITY_FAILED))
    failure.handle(store, Mock(), SETTINGS, now=NOW)
    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.NEEDS_HUMAN


def test_handle_timeout_stops_and_requeues_bounty(store):
    store.insert_task(_failed(TaskStatus.TIMEOUT, deadline=NOW + timedelta(days=1)))
    client = Mock()
    failure.handle(store, client, SETTINGS, now=NOW)
    client.stop_tasks.assert_called_once_with([7])
    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.QUEUED


def test_handle_timeout_abandons_non_bounty(store):
    store.insert_task(_failed(TaskStatus.TIMEOUT))
    client = Mock()
    failure.handle(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.ABANDONED


def test_engine_failure_streak_triggers_kill_switch(store):
    for i in range(5):
        rec = _failed(retry_count=3)
        rec.model_id = f"org/m{i}"
        store.insert_task(rec)
    failure.handle(store, Mock(), SETTINGS, now=NOW)
    assert store.kill_switch() is True
    assert store.get_counter("consecutive_engine_failures") == 5
```

- [ ] **Step 2: 确认失败**

- [ ] **Step 3: 实现**

```python
import logging
from datetime import datetime, timezone

import yaml

from .models import FailureKind, TaskStatus

logger = logging.getLogger(__name__)

_QUALITY_PATTERNS = ("judge", "quality check failed", "score below")
_MIN_MODEL_LEN, _MAX_TP = 2048, 4
_STREAK_LIMIT = 5  # 连续引擎失败熔断阈值（spec §6）


def classify(log_text: str) -> FailureKind:
    lowered = (log_text or "").lower()
    if any(p in lowered for p in _QUALITY_PATTERNS):
        return FailureKind.QUALITY
    return FailureKind.ENGINE


def _set_flag(command: list, flag: str, value: str) -> None:
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command.extend([flag, value])


def next_config(record) -> str | None:
    cfg = yaml.safe_load(record.config_params)
    sut = cfg["sut_config"]["values"]["command"]
    ref = cfg["ref_config"]["values"]["command"]
    if record.retry_count == 0:
        _set_flag(sut, "--gpu-memory-utilization", "0.95")
    elif record.retry_count == 1:
        new_len = max(int(cfg.get("max_model_len", 4096)) // 2, _MIN_MODEL_LEN)
        cfg["max_model_len"] = new_len
        for cmd in (sut, ref):
            _set_flag(cmd, "--max-model-len", str(new_len))
    elif record.retry_count == 2:
        current = int(sut[sut.index("-tp") + 1]) if "-tp" in sut else 1
        new_tp = min(current * 2, _MAX_TP)
        for section, cmd in (("sut_config", sut), ("ref_config", ref)):
            _set_flag(cmd, "-tp", str(new_tp))
            cfg[section]["gpu_num"] = str(new_tp)
    else:
        return None
    return yaml.safe_dump(cfg, sort_keys=False)


def handle(storage, client, settings, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    engine_failed = storage.tasks_by_status(TaskStatus.ENGINE_FAILED)
    if engine_failed:
        streak = storage.get_counter("consecutive_engine_failures") + len(engine_failed)
        storage.set_counter("consecutive_engine_failures", streak)
        if streak >= _STREAK_LIMIT:
            storage.set_kill_switch(
                True, f"{streak} consecutive engine failures; config template may be broken")
            logger.error("engine failure streak %d >= %d; kill switch ON", streak, _STREAK_LIMIT)
    for rec in engine_failed:
        new_cfg = next_config(rec) if rec.retry_count < settings.max_retries else None
        if new_cfg is None:
            rec.status = TaskStatus.BLACKLISTED
            logger.warning("blacklisting %s@%s after %d retries",
                           rec.model_id, rec.target_gpu, rec.retry_count)
        else:
            rec.config_params = new_cfg
            rec.retry_count += 1
            rec.status = TaskStatus.QUEUED
            rec.task_id = None
            rec.submit_time = None
        storage.update_task(rec)
    for rec in storage.tasks_by_status(TaskStatus.QUALITY_FAILED):
        rec.status = TaskStatus.NEEDS_HUMAN
        storage.update_task(rec)
    for rec in storage.tasks_by_status(TaskStatus.TIMEOUT):
        if rec.task_id is not None:
            try:
                client.stop_tasks([rec.task_id])
            except Exception:
                logger.exception("stop_tasks failed for %s", rec.task_id)
        if rec.bounty_deadline is not None and rec.bounty_deadline > now and rec.retry_count == 0:
            rec.status = TaskStatus.QUEUED
            rec.retry_count += 1
            rec.task_id = None
            rec.submit_time = None
        else:
            rec.status = TaskStatus.ABANDONED
        storage.update_task(rec)
```

- [ ] **Step 4: 确认通过** - [ ] **Step 5: Commit** — `git commit -am "feat(M5): failure classification, retry ladder, blacklist"`

---

### Task 10: tick 接线 + metrics（M6）

**Files:**
- Modify: `src/auto_adapter/main.py`（实现 `build_deps`/`tick`）、`src/auto_adapter/metrics.py`（加 `reset()`）
- Test: `tests/test_tick_integration.py`（新建）

**Interfaces:**
- Consumes: 前面所有模块
- Produces: `Deps` dataclass（`settings, storage, client, sources`）；`tick(deps, stop_event)` 按 spec §2 顺序执行全链路。

- [ ] **Step 1: 写失败测试（全链路一个 tick，mock 平台 API）**

`tests/test_tick_integration.py`：

```python
import threading
from unittest.mock import Mock

import pytest

from auto_adapter.main import Deps, tick
from auto_adapter.models import ModelSearchResult, TaskStatus
from auto_adapter.settings import Settings
from auto_adapter.storage.sqlite import SqliteStorage


def test_tick_discovers_enqueues_and_submits(tmp_path, candidate):
    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.search_model.return_value = ModelSearchResult(False, {}, {})
    client.add_task.return_value = 99
    client.list_my_tasks.return_value = {"records": [{"taskId": 99, "status": "RUNNING"}]}

    class Src:
        name = "fake"
        def fetch(self):
            return [candidate]

    deps = Deps(
        settings=Settings(xc_token="t", strategy_id="uuid-1", base_url="https://x"),
        storage=storage, client=client, sources=[Src()],
    )
    tick(deps, threading.Event())
    rec = storage.get_task(candidate.model_id, "MetaX_c-500")
    assert rec is not None and rec.task_id == 99
    assert rec.status == TaskStatus.RUNNING  # submit 后同 tick 内 monitor 已对账
    assert storage.pending_candidates() == []  # 候选已消费


def test_tick_skips_duplicate_candidate(tmp_path, candidate):
    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.search_model.return_value = ModelSearchResult(
        True, {}, {"MetaX_c-500": {"passed": True}})
    client.list_my_tasks.return_value = {"records": []}

    class Src:
        name = "fake"
        def fetch(self):
            return [candidate]

    deps = Deps(settings=Settings(xc_token="t", strategy_id="s", base_url="https://x"),
                storage=storage, client=client, sources=[Src()])
    tick(deps, threading.Event())
    client.add_task.assert_not_called()
    assert storage.get_task(candidate.model_id, "MetaX_c-500") is None
```

- [ ] **Step 2: 确认失败**

- [ ] **Step 3: 实现**

`metrics.py` 加：

```python
def reset() -> None:
    _counters.clear()
```

`main.py`：

```python
from dataclasses import dataclass, field

from . import config_gen, eligibility, failure, health, metrics, monitor, submitter
from .discovery import base as discovery
from .discovery.bounty import ManualBountySource
from .discovery.huggingface import HuggingFaceSource
from .eligibility import Verdict
from .models import TaskRecord, TaskStatus
from .platform_client import PlatformClient
from .storage import SqliteStorage
from .storage.base import DuplicateTaskError


@dataclass
class Deps:
    settings: Settings
    storage: object
    client: object
    sources: list = field(default_factory=list)


def build_deps(settings: Settings) -> Deps:
    return Deps(
        settings=settings,
        storage=SqliteStorage(settings.storage_path),
        client=PlatformClient(settings.base_url, settings.xc_token),
        sources=[
            ManualBountySource(settings.bounty_config_path),
            HuggingFaceSource(limit=settings.hf_fetch_limit),
        ],
    )


def tick(deps: Deps, stop_event: threading.Event) -> None:
    s = deps
    metrics.incr("candidates_discovered", discovery.run(s.sources, s.storage))
    for cand in s.storage.pending_candidates():
        if stop_event.is_set():
            return
        target_gpu = config_gen.select_target_gpu(s.storage)
        decision = eligibility.evaluate(cand, target_gpu, s.storage, s.client)
        if decision.verdict == Verdict.SKIP_UNCERTAIN:
            metrics.incr("skipped_uncertain")
            continue  # 不标记 processed，下个 tick 重试
        if decision.verdict == Verdict.ENQUEUE:
            try:
                req = config_gen.build_request(cand, target_gpu, s.settings.strategy_id)
                s.storage.insert_task(TaskRecord(
                    model_id=cand.model_id, target_gpu=target_gpu,
                    framework=req.framework, status=TaskStatus.QUEUED,
                    priority=decision.priority, model_url=cand.model_url,
                    task_type=req.task_type, config_params=req.config_params,
                    bounty_deadline=cand.bounty_deadline))
                metrics.incr("enqueued")
            except ValueError:
                metrics.incr("unresolvable_task_type")
                logger.warning("cannot resolve task type for %s; needs human", cand.model_id)
            except DuplicateTaskError:
                metrics.incr("skipped_duplicate")
        else:
            metrics.incr("skipped_duplicate")
        s.storage.mark_candidate_processed(cand.model_id)
    if stop_event.is_set():
        return
    metrics.incr("submitted", submitter.drain(s.storage, s.client, s.settings))
    if stop_event.is_set():
        return
    monitor.poll(s.storage, s.client, s.settings)
    failure.handle(s.storage, s.client, s.settings)
    metrics.flush_tick_summary()
```

- [ ] **Step 4: 确认通过**，并全量 `.venv/bin/pytest` 无回归
- [ ] **Step 5: Commit** — `git commit -am "feat(M6): wire full pipeline into tick with metrics"`

---

### Task 11: 优雅停机冒烟测试 + 容器打包（M6）

**Files:**
- Modify: `Dockerfile`（基础镜像做成 ARG，便于本地无法访问平台 registry 时构建验证）
- Test: `tests/test_graceful_shutdown.py`（新建）

**Interfaces:**
- Consumes: `main.main()`、Task 10 的 `build_deps`

- [ ] **Step 1: 写冒烟测试**

`tests/test_graceful_shutdown.py`：

```python
import os
import signal
import subprocess
import sys
import time


def test_sigterm_exits_within_grace_period(tmp_path):
    env = os.environ | {
        "XC_TOKEN": "t", "STRATEGY_ID": "s",
        "MODELHUB_BASE_URL": "http://127.0.0.1:1",  # 打不通也不该崩（tick 容错）
        "STORAGE_PATH": str(tmp_path / "agent.db"),
        "TICK_SECONDS": "60",
    }
    proc = subprocess.Popen(
        [sys.executable, "-c", "from auto_adapter.main import main; main()"], env=env)
    time.sleep(2)
    assert proc.poll() is None, "process should be running"
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 0  # 30s 限额内（实际应秒级）
```

注意：HuggingFaceSource 首个 tick 会打真实 HF——为保持测试离线，给 `settings.py` 加
`hf_discovery_enabled: bool = True`（`from_env` 解析 `HF_DISCOVERY_ENABLED`，`"false"/"0"` 为
False），上面 env 里加 `"HF_DISCOVERY_ENABLED": "false"`，`build_deps` 中改为：

```python
    sources = [ManualBountySource(settings.bounty_config_path)]
    if settings.hf_discovery_enabled:
        sources.append(HuggingFaceSource(limit=settings.hf_fetch_limit))
```

`main()` 在 `run_loop` 返回后自然结束，退出码 0。

- [ ] **Step 2: 运行确认**（此测试直接对已实现代码，首跑可能就过；若失败按 systematic-debugging 排查信号处理）

- [ ] **Step 3: Dockerfile 参数化**

```dockerfile
ARG BASE_IMAGE=modelhubxc-4pd.tencentcloudcr.com/xc_agent_platform/python:3.11-slim
FROM ${BASE_IMAGE}
```

（其余保持不变。）本地验证：`docker build --build-arg BASE_IMAGE=python:3.11-slim -t auto-adapter .`（无 docker 环境则跳过，标注在 PR/提交信息中）。

- [ ] **Step 4: 全量测试** — `.venv/bin/pytest -v` 全绿
- [ ] **Step 5: Commit** — `git commit -am "feat(M6): graceful shutdown smoke test and parameterized Dockerfile"`

---

## 完成定义（Definition of Done）

- `.venv/bin/pytest` 全绿；无 skip 占位残留；
- `docker build` 可用公共基础镜像构建成功（或注明环境不可用）；
- CLAUDE.md「当前状态」段落更新为「M1–M6 已实现」；
- 上线前人工核对（不属于代码任务）：真实 `status`/`verifyResult` 枚举回填 `rules.PLATFORM_STATUS_MAP`、`KNOWN_GPUS` 扩充、悬赏 JSON 配置准备。
