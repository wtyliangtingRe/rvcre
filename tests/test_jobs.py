import threading
import time

import pytest

from rvcre.jobs import JobManager


def wait_terminal(manager, identifier):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = manager.get(identifier)
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.01)
    pytest.fail("Job did not terminate")


def test_running_and_queued_cancellation_survives_reload(tmp_path):
    manager = JobManager(tmp_path)
    started = threading.Event()

    def blocking(directory, cancel):
        started.set()
        cancel.wait(3)
        return {"outputs": []}

    first = manager.submit("tts", {}, blocking)
    assert started.wait(3)
    second = manager.submit("tts", {}, lambda *_: {"outputs": []})
    assert manager.cancel(second["id"])["status"] == "cancelled"
    manager.cancel(first["id"])
    assert wait_terminal(manager, first["id"])["status"] == "cancelled"
    manager.close()
    reopened = JobManager(tmp_path)
    try:
        assert reopened.get(second["id"])["status"] == "cancelled"
    finally:
        reopened.close()


def test_reader_does_not_recover_another_active_queue(tmp_path):
    writer = JobManager(tmp_path)
    reader = JobManager(tmp_path)
    entered = threading.Event()

    def blocking(directory, cancel):
        entered.set()
        cancel.wait(3)
        return {"outputs": []}

    try:
        job = writer.submit("tts", {}, blocking)
        assert entered.wait(3)
        assert reader.get(job["id"])["status"] == "running"
        with pytest.raises(ValueError, match="另一个"):
            reader.submit("tts", {}, lambda *_: {"outputs": []})
        assert writer.get(job["id"])["status"] == "running"
        writer.cancel(job["id"])
    finally:
        writer.close()
        reader.close()


def test_closed_queue_cannot_leave_a_new_queued_record(tmp_path):
    manager = JobManager(tmp_path)
    manager.close()
    with pytest.raises(RuntimeError, match="关闭"):
        manager.submit("tts", {}, lambda *_: {"outputs": []})
    assert manager.list() == []
