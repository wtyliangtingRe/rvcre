"""Verify narration options reach the service; no model inference is performed."""

import pytest
from fastapi.testclient import TestClient

from rvcre import cli
from rvcre.api import create_app


class SubmissionCapture:
    """Submission-only test double, with no audio-generation path."""

    def __init__(self):
        self.calls = []
        self.closed = False

    def submit_tts(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"id": "test-submission", "status": "queued", "outputs": [], "files": []}

    def close(self):
        self.closed = True


@pytest.mark.parametrize("enabled", [None, False, True])
def test_api_forwards_narration_choice_and_keeps_default_disabled(enabled):
    service = SubmissionCapture()
    body = {"voice_id": "test-voice", "text": "完整的旁白文稿。", "seed": 17}
    if enabled is not None:
        body["narration"] = enabled
    with TestClient(create_app(service=service)) as client:
        response = client.post("/api/tts", json=body)
    assert response.status_code == 202, response.text
    args, kwargs = service.calls[0]
    assert args[0:2] == ("test-voice", body["text"])
    assert args[4] == 17
    assert kwargs["narration"] is (enabled is True)


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_forwards_narration_and_full_utf8_script(tmp_path, monkeypatch, enabled):
    service = SubmissionCapture()
    monkeypatch.setattr(cli, "VoiceService", lambda *args: service)
    monkeypatch.setattr(cli, "_wait", lambda *_: 0)
    script = tmp_path / "script.txt"
    content = "第一段的内容。\n\n第二段带有年份：一九六七年。\n"
    script.write_text(content, encoding="utf-8")
    arguments = ["tts", "--voice", "test-voice", "--text-file", str(script), "--seed", "23"]
    if enabled:
        arguments.append("--narration")
    assert cli.main(arguments) == 0
    args, kwargs = service.calls[0]
    assert args[0:2] == ("test-voice", content)
    assert args[4] == 23
    assert kwargs["narration"] is enabled
    assert service.closed


def test_ui_preview_and_submission_apply_same_narration_choice():
    pytest.importorskip("gradio")
    from rvcre.text import NARRATION_ANCHOR_CHARS, NARRATION_SEGMENT_CHARS, plan_tts_segments
    from rvcre.ui import _UiActions, _tts_text_summary

    script = "这是用来观察朗读衔接的句子。\n\n" * 45
    segments = plan_tts_segments(script, narration=True)
    summary = _tts_text_summary(script, narration=True)
    assert f"自动切分为 {len(segments)} 段" in summary
    assert f"首段最多 {NARRATION_ANCHOR_CHARS}" in summary
    assert f"后续每段最多 {NARRATION_SEGMENT_CHARS}" in summary
    assert "每段最多 600" in _tts_text_summary(script)
    assert "生成后" in _tts_text_summary(None, narration=True)

    service = SubmissionCapture()
    # Hi-Fi must still drop the hidden expression fields when narration is on.
    result = _UiActions(service).start_tts(None, "test-voice", script, "hifi", 1.4,
                                          "angry", "不要传入 Hi-Fi", 31, narration=True)
    assert result[0] == "test-submission"
    _, kwargs = service.calls[0]
    assert kwargs["narration"] is True
    assert kwargs["performance"] is None
    assert kwargs["text"] == script


def test_ui_checkbox_starts_disabled_and_updates_summary(service):
    pytest.importorskip("gradio")
    from rvcre.ui import build_ui

    app = build_ui(service)
    try:
        config = app.get_config_file()
        choices = [component for component in config["components"]
                   if component["type"] == "checkbox"
                   and component["props"]["label"] == "长文首段定调（试用）"]
        assert len(choices) == 1
        assert choices[0]["props"]["value"] is False
        checkbox_id = choices[0]["id"]
        previews = [fn for fn in app.fns.values() if fn.fn.__name__ == "_tts_text_summary"]
        assert len(previews) == 2  # Typing and toggling both refresh the split preview.
        assert all(checkbox_id in [component._id for component in fn.inputs] for fn in previews)
        submit = next(fn for fn in app.fns.values() if fn.fn.__name__ == "start_tts")
        assert submit.inputs[-1]._id == checkbox_id
    finally:
        app.close()
