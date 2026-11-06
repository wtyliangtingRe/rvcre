"""UI/service integration tests use the explicitly marked runner in conftest.

No model-generated speech is claimed by these tests. They exercise task state,
candidate selection and mode controls using real persisted service records.
"""

import time

import pytest

from rvcre.service import VoiceService

gr = pytest.importorskip("gradio")

# The optional UI module must only be imported after the Gradio availability check.
from rvcre.ui import _DEFAULT_SPEC, _UiActions, build_ui  # noqa: E402


def finish(service, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = service.get_job(job_id)
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Test task did not finish")


def test_design_candidates_are_saved_by_selected_index(service):
    actions = _UiActions(service)
    submitted = actions.start_design(None, "这是测试文本。", *_DEFAULT_SPEC, 3, 42, None)
    job_id = submitted[0]
    assert submitted[2]["active"] is True
    assert submitted[-1]["interactive"] is False
    completed = finish(service, job_id)
    assert completed["status"] == "succeeded"
    rendered = actions.poll_design(job_id)
    assert rendered[2]["active"] is False
    assert [player["value"] for player in rendered[5:8]] == [item["path"] for item in completed["outputs"]]
    saved = actions.save_candidate(job_id, 2, "第三个候选")
    profile = service.get_voice(saved[1]["value"])
    assert profile["generation"]["used_seed"] == 44
    assert profile["name"] == "第三个候选"
    assert saved[2]["value"] == profile["id"]  # TTS and VC's shared selection.
    assert (profile["name"], profile["id"]) in saved[3]["choices"]


def test_hifi_does_not_receive_hidden_expression_controls(service, tone):
    profile = service.import_voice("测试参考", str(tone), transcript="对应转写。")
    actions = _UiActions(service)
    submitted = actions.start_tts(None, profile["id"], "另一句测试文本。", "hifi", 1.4, "angry", "大声说", 91)
    job = finish(service, submitted[0])
    assert job["status"] == "succeeded"
    _, request = service.runner.requests[-1]
    assert request["mode"] == "hifi"
    assert request["instruction"] == ""
    assert request["performance"] == {"speed": 1.0, "emotion": "neutral", "instruction": ""}
    assert actions.poll_generation(job["id"])[-1]["value"] == job["outputs"][0]["path"]


def test_missing_backend_reports_failure_and_no_playable_result(tmp_path):
    service = VoiceService(tmp_path / "missing_backend", config_path=tmp_path / "absent.json")
    try:
        actions = _UiActions(service)
        job_id = actions.start_design(None, "测试。", *_DEFAULT_SPEC, 1, 42, None)[0]
        job = finish(service, job_id)
        assert job["status"] == "failed"
        view = actions.poll_design(job_id)
        assert "任务未完成" in view[1]
        assert view[2]["active"] is False
        assert all(player.get("value") is None and player["visible"] is False for player in view[5:8])
        assert view[-1]["interactive"] is False
        assert "尚未配置" in actions.backend_status()
    finally:
        service.close()


def test_cancel_is_forwarded_to_the_running_service_job(tmp_path):
    class WaitingRunner:
        def run(self, backend, request, directory, cancel_event):
            cancel_event.wait(3)
            return {"outputs": [], "metadata": {"test_double": True}}

    service = VoiceService(tmp_path / "cancellation", runner=WaitingRunner())
    try:
        actions = _UiActions(service)
        job_id = actions.start_design(None, "测试。", *_DEFAULT_SPEC, 1, 42, None)[0]
        with pytest.raises(gr.Error, match="当前任务"):
            actions.start_design(job_id, "不能覆盖当前任务。", *_DEFAULT_SPEC, 1, 42, None)
        actions.cancel_design(job_id)
        assert finish(service, job_id)["status"] == "cancelled"
        view = actions.poll_design(job_id)
        assert view[2]["active"] is False
        assert view[4]["interactive"] is True
        assert view[-1]["interactive"] is False
    finally:
        service.close()


def test_ui_builds_with_the_pinned_gradio_without_loading_models(service):
    app = build_ui(service)
    try:
        config = app.get_config_file()
        assert config["analytics_enabled"] is False
        timers = [component for component in config["components"] if component["type"] == "timer"]
        assert len(timers) == 2
        assert all(timer["props"]["active"] is False for timer in timers)
        callbacks = {fn.fn.__name__ for fn in app.fns.values()}
        assert {"start_design", "start_tts", "start_vc", "cancel_design", "cancel_generation"} <= callbacks
        assert service.runner.requests == []
    finally:
        app.close()
