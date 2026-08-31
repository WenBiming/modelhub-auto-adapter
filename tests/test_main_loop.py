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
