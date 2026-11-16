"""Long-form worker orchestration with a marked model double, not real inference."""

import builtins
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import types

import pytest
import soundfile as sf

from rvcre.workers import voxcpm as worker


@pytest.fixture
def batch_model(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"architecture":"voxcpm2"}', encoding="utf-8")
    reference = tmp_path / "reference.wav"
    reference.touch()  # Argument validation uses the path; the double does not read it.
    observed = {"loads": [], "calls": [], "snapshots": [], "fail_on": None}
    progress_path = tmp_path / "progress.json"

    class Device:
        type = "cpu"

        def __str__(self):
            return "cpu"

    class Model:
        tts_model = types.SimpleNamespace(sample_rate=24000, last_successful_seed=None)

        def generate(self, **kwargs):
            observed["calls"].append(kwargs)
            observed["snapshots"].append(json.loads(progress_path.read_text(encoding="utf-8")))
            index = len(observed["calls"])
            if index == observed["fail_on"]:
                raise RuntimeError("test-only segment failure")
            self.tts_model.last_successful_seed = (kwargs["seed"] + index) % (2**32)
            return [index / 10, 0, -index / 10]

    def load(*args, **kwargs):
        observed["loads"].append((args, kwargs))
        observed["loading_progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
        return Model()

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=lambda _: Device(), __version__="test"))
    monkeypatch.setitem(sys.modules, "voxcpm", types.SimpleNamespace(
        VoxCPM=types.SimpleNamespace(from_pretrained=load),
    ))
    # Loading compatibility has separate tests; these doubles exercise generation.
    monkeypatch.setattr(worker, "model_loading", lambda config: nullcontext({"backend": "test-fixture"}))
    config = {"model_dir": str(model_dir), "device": "cpu", "upstream_revision": "test-revision"}
    return config, reference, observed


@pytest.mark.parametrize("mode", ["controllable", "hifi"])
def test_batch_uses_one_model_and_reuses_voice_controls_for_every_segment(tmp_path, batch_model, mode):
    config, reference, observed = batch_model
    segments = [" 第一段。 ", "第二段继续。", "第三段结束。"]
    request = {
        "task": "tts", "text": "整篇文本不能直接送进一次模型调用。", "segments": segments,
        "reference_path": str(reference), "reference_text": "准确参考转写。",
        "mode": mode, "instruction": "平静，语速稍慢" if mode == "controllable" else "",
        "seed": 2**32 - 1, "candidates": 1,
        "settings": {"cfg_value": 1.8, "inference_timesteps": 8},
    }
    result = worker.run(config, request, tmp_path / "result.json")

    assert len(observed["loads"]) == 1
    assert observed["loads"][0][1]["local_files_only"] is True
    calls = observed["calls"]
    prefix = "(平静，语速稍慢)" if mode == "controllable" else ""
    assert [call["text"] for call in calls] == [prefix + text.strip() for text in segments]
    assert [call["seed"] for call in calls] == [2**32 - 1] * 3
    assert all(call["reference_wav_path"] == str(reference.resolve()) for call in calls)
    assert all(call["cfg_value"] == 1.8 and call["inference_timesteps"] == 8 for call in calls)
    if mode == "hifi":
        assert all(call["prompt_wav_path"] == str(reference.resolve()) for call in calls)
        assert all(call["prompt_text"] == request["reference_text"] for call in calls)
    else:
        assert all("prompt_wav_path" not in call and "prompt_text" not in call for call in calls)
    assert request["segments"] == segments  # Trimming must not mutate the saved request.

    assert observed["loading_progress"]["phase"] == "loading"
    assert observed["loading_progress"]["completed"] == 0
    assert observed["loading_progress"]["total"] == 3
    assert [snapshot["completed"] for snapshot in observed["snapshots"]] == [0, 1, 2]
    assert [snapshot["current"] for snapshot in observed["snapshots"]] == [1, 2, 3]
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "complete" and progress["current"] is None
    assert progress["completed"] == progress["total"] == 3
    assert progress["outputs"] == result["outputs"]
    assert progress["metadata"] == result["metadata"]
    assert result["metadata"]["mode"] == mode
    assert result["metadata"]["sample_rate"] == 24000
    assert [output["segment_index"] for output in result["outputs"]] == [1, 2, 3]
    assert [output["seed"] for output in result["outputs"]] == [0, 1, 2]
    assert [Path(output["path"]).name for output in result["outputs"]] == [
        "audio-001.wav", "audio-002.wav", "audio-003.wav",
    ]
    for index, output in enumerate(result["outputs"], start=1):
        samples, rate = sf.read(output["path"])
        assert rate == 24000
        assert samples == pytest.approx([index / 10, 0, -index / 10], abs=1e-6)
    assert not (tmp_path / "result.json").exists()  # Only main publishes the final result.


@pytest.mark.parametrize("narration", [False, True])
def test_completed_clips_survive_later_segment_failure(tmp_path, batch_model, narration):
    config, reference, observed = batch_model
    observed["fail_on"] = 2
    with pytest.raises(RuntimeError, match="test-only segment failure"):
        worker.run(config, {
            "task": "tts", "text": "全文", "segments": ["第一段。", "第二段。", "第三段。"],
            "reference_path": str(reference), "seed": 42,
            "narration": narration,
        }, tmp_path / "result.json")

    assert len(observed["loads"]) == 1
    assert len(observed["calls"]) == 2
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["total"] == 3 and progress["completed"] == 1 and progress["current"] == 2
    assert progress["phase"] == "generating"
    assert len(progress["outputs"]) == 1
    assert progress["outputs"][0]["segment_index"] == 1
    assert Path(progress["outputs"][0]["path"]).is_file()
    assert not (tmp_path / "audio-002.wav").exists()
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("overrides", [
    {"segments": []},
    {"segments": None},
    {"segments": "一段文本"},
    {"segments": [" "]},
    {"segments": ["第一段。", 23]},
    {"segments": ["甲" * 601]},
    {"segments": ["合法文本。"], "candidates": 2},
    {"segments": ["合法文本。"], "task": "design"},
    {"segments": ["第一段。", "甲" * 241], "narration": True},
    {"segments": ["甲" * 81, "第二段。"], "narration": True},
    {"segments": ["合法文本。"], "narration": "true"},
])
def test_invalid_batch_is_rejected_before_model_imports(tmp_path, monkeypatch, overrides):
    original_import = builtins.__import__

    def forbid_model_import(name, *args, **kwargs):
        if name in {"torch", "voxcpm"}:
            pytest.fail("An invalid batch tried to import a model library")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_model_import)
    with pytest.raises(ValueError):
        worker.run({}, {"task": "tts", "text": "全文", **overrides}, tmp_path / "result.json")
    assert not (tmp_path / "progress.json").exists()


@pytest.mark.parametrize("mode", ["controllable", "hifi"])
def test_narration_conditions_all_later_clips_on_fixed_opening(tmp_path, batch_model, mode):
    config, reference, observed = batch_model
    segments = ["这是用于定调的开场。\n\n接着同一段的下一句。", "第二段正文与第一段不同。", "第三段不应沿用第二段的偏差。"]
    instruction = "平静，语速稍慢" if mode == "controllable" else ""
    result = worker.run(config, {
        "task": "tts", "text": "".join(segments), "segments": segments,
        "reference_path": str(reference), "reference_text": "原始参考录音的文字。",
        "mode": mode, "instruction": instruction, "seed": 41, "narration": True,
    }, tmp_path / "result.json")

    calls = observed["calls"]
    assert len(observed["loads"]) == 1
    assert calls[0]["text"] == (f"({instruction})" if instruction else "") + segments[0]
    assert [call["text"] for call in calls[1:]] == segments[1:]
    assert all(call["reference_wav_path"] == str(reference.resolve()) for call in calls)
    opening_path = result["outputs"][0]["path"]
    assert all(call["prompt_wav_path"] == opening_path for call in calls[1:])
    assert all(call["prompt_text"] == "这是用于定调的开场。 接着同一段的下一句。" for call in calls[1:])
    if mode == "hifi":
        assert calls[0]["prompt_wav_path"] == str(reference.resolve())
        assert calls[0]["prompt_text"] == "原始参考录音的文字。"
    else:
        assert "prompt_text" not in calls[0]
    assert [call["seed"] for call in calls] == [41, 41, 41]
    assert [item["requested_seed"] for item in result["outputs"]] == [41, 41, 41]
    assert [item["seed"] for item in result["outputs"]] == [42, 43, 44]
    assert [item["prompt_source_segment"] for item in result["outputs"]] == [None, 1, 1]
    assert result["metadata"]["continuity"] == "fixed_opening"
    assert len(result["outputs"]) == len(segments)
    for output in result["outputs"]:
        info = sf.info(output["path"])
        assert output["duration_seconds"] == pytest.approx(info.duration)
