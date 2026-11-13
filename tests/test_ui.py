"""UI/service integration tests use the explicitly marked runner in conftest.

No model-generated speech is claimed by these tests. They exercise task state,
candidate selection and mode controls using real persisted service records.
"""

import time

import pytest

from rvcre.service import VoiceService

gr = pytest.importorskip("gradio")

# The optional UI module must only be imported after the Gradio availability check.
from rvcre.ui import (  # noqa: E402
    _DEFAULT_SPEC, _UiActions, _generation_view, _job_message, _tts_text_summary, build_ui,
)


def finish(service, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = service.get_job(job_id)
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Test task did not finish")


@pytest.mark.parametrize(("transcript", "notes", "source_note", "expected"), [
    (None, None, None, ("", "", "")),
    ("", " \n ", "\t", ("", "", "")),
    ("  对应参考录音的转写。\n", None, None, ("对应参考录音的转写。", "", "")),
    ("  对应转写。  ", "  旁白用\n保留这行。  ", "  本地导入  ",
     ("对应转写。", "旁白用\n保留这行。", "本地导入")),
])
def test_import_voice_saves_empty_or_filled_optional_fields(
        service, tone, transcript, notes, source_note, expected):
    actions = _UiActions(service)
    saved = actions.import_voice("  新的参考声音  ", str(tone), transcript, notes, source_note)
    profile = service.get_voice(saved[1]["value"])
    assert profile["name"] == "新的参考声音"
    assert (profile["transcript"], profile["notes"], profile["source"]["note"]) == expected
    assert profile["source_transcript"] == expected[0]
    assert saved[2]["value"] == profile["id"]
    assert (profile["name"], profile["id"]) in saved[3]["choices"]
    assert service.runner.requests == []


@pytest.mark.parametrize("name", [None, "", " \n\t "])
def test_import_voice_requires_a_name_without_creating_a_profile(service, tone, name):
    with pytest.raises(gr.Error, match="请为声音填写名称"):
        _UiActions(service).import_voice(name, str(tone), None, None, None)
    assert service.list_voices() == []


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
    view = actions.poll_generation(job["id"])
    assert view[-2]["value"] == job["outputs"][0]["path"]
    assert view[-1]["visible"] is False


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


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_incomplete_long_text_keeps_downloads_without_presenting_a_complete_recording(status, tmp_path):
    clip = str(tmp_path / "audio-001.wav")
    text = str(tmp_path / "text-001.txt")
    job = {
        "id": "incomplete", "status": status, "error": "第二段生成失败。",
        "outputs": [{"path": clip, "kind": "segment"}], "files": [clip, text],
        "progress": {"total": 3, "completed": 1, "current": 2, "phase": "generating"},
    }
    view = _generation_view(job)
    assert "1/3" in view[1] and "下载" in view[1]
    assert view[2]["active"] is False
    assert view[4]["interactive"] is True
    assert view[-2]["value"] is None and view[-2]["visible"] is False
    assert view[-1]["visible"] is True
    assert view[-1]["value"] == [clip, text]


def test_completed_long_text_plays_merged_audio_and_offers_the_segment_archive(tmp_path):
    merged = str(tmp_path / "merged.wav")
    bundle = str(tmp_path / "segments.zip")
    job = {
        "id": "complete", "status": "succeeded",
        "outputs": [
            {"path": str(tmp_path / "audio-001.wav"), "kind": "segment"},
            {"path": merged, "kind": "merged"},
            {"path": str(tmp_path / "audio-002.wav"), "kind": "segment"},
        ],
        "files": [bundle], "progress": {"total": 2, "completed": 2, "phase": "complete"},
    }
    view = _generation_view(job)
    assert "2 段已合并" in view[1]
    assert view[-2]["value"] == merged
    assert view[-2]["label"] == "完整音频"
    assert view[-1]["value"] == [bundle]
    assert view[-1]["visible"] is True


def test_long_text_progress_reports_model_loading_generation_and_merge():
    job = {"id": "progress", "status": "running", "progress": {
        "total": 3, "completed": 0, "current": None, "phase": "loading",
    }}
    assert "加载本地模型" in _job_message(job)
    job["progress"].update(completed=2, current=3, phase="generating")
    message = _job_message(job)
    assert "2/3" in message and "第 3 段" in message
    job["progress"].update(completed=3, current=None, phase="merging")
    message = _job_message(job)
    assert "3/3" in message and "合并" in message


def test_a_new_generation_clears_previous_audio_and_downloads(service):
    view = _generation_view({"id": "new", "status": "queued", "outputs": [], "files": []}, clear=True)
    assert view[-2]["value"] is None and view[-2]["visible"] is False
    assert view[-1]["value"] is None and view[-1]["visible"] is False
    assert len(_UiActions(service).poll_generation(None)) == len(view) == 8


def test_text_preview_accepts_a_full_script_and_handles_cleared_input():
    assert "1 段音频" in _tts_text_summary("你好。")
    preview = _tts_text_summary("甲" * 600 + "。\n\n下一段。")
    assert "自动切分为 2 段" in preview
    assert "每段最多 600" in preview
    assert "生成后" in _tts_text_summary("  \n")


def test_ui_builds_with_the_pinned_gradio_without_loading_models(service):
    app = build_ui(service)
    try:
        config = app.get_config_file()
        assert config["analytics_enabled"] is False
        timers = [component for component in config["components"] if component["type"] == "timer"]
        assert len(timers) == 2
        assert all(timer["props"]["active"] is False for timer in timers)
        callbacks = {fn.fn.__name__ for fn in app.fns.values()}
        assert {"start_design", "start_tts", "start_vc", "cancel_design", "cancel_generation", "_tts_text_summary"} <= callbacks
        downloads = [component for component in config["components"] if component["type"] == "file"]
        assert len(downloads) == 1
        assert downloads[0]["props"]["file_count"] == "multiple"
        assert downloads[0]["props"]["visible"] is False
        for fn in app.fns.values():
            if fn.fn.__name__ in {"start_tts", "start_vc", "poll_generation", "cancel_generation"}:
                assert len(fn.outputs) == 8
        assert service.runner.requests == []
    finally:
        app.close()
