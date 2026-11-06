import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rvcre.audio import analyze_audio
from rvcre.schema import VoiceSpec, design_instruction
from rvcre.service import VoiceService


def finish(service, job):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = service.get_job(job["id"])
        if current["status"] in {"succeeded", "failed", "cancelled"}:
            assert current["status"] == "succeeded", current
            return current
        time.sleep(0.01)
    pytest.fail("Test-double generation did not finish")


def test_pitch_measurement_is_not_a_timbre_or_demographic_claim(tone):
    result = analyze_audio(tone)
    assert result["pitch_median_hz"] == pytest.approx(220, rel=0.03)
    assert result["voiced_frame_ratio"] > 0.8
    assert "age" not in result and "gender" not in result and "breathiness" not in result


def test_designed_identity_is_reused_for_both_renderers(service):
    spec = VoiceSpec(pitch="low", breathiness=0.4, description="克制自然").model_dump()
    design = finish(service, service.submit_design("这是固定的试听句。", spec, 2, 123))
    voice = service.save_candidate(design["id"], 1, "夜航")
    assert voice["generation"]["used_seed"] == 124
    assert voice["design_spec_origin"] == "requested"
    tts = finish(service, service.submit_tts(voice["id"], "新的台词。", {"emotion": "calm"}))
    vc = finish(service, service.submit_vc(voice["id"], voice["original_path"]))
    assert service.runner.requests[1][0] == "voxcpm"
    assert service.runner.requests[2][0] == "seedvc"
    for _, request in service.runner.requests[1:]:
        assert request["reference_path"] == voice["reference_path"]
        assert request["voice_id"] == voice["id"]
    assert Path(service.audio_path(tts["id"], 0)).is_file()
    assert vc["metadata"]["test_double"] is True
    variant_job = finish(service, service.submit_design("变体试听。", {"pitch": "high"}, 1, 9, voice["id"]))
    variant = service.save_candidate(variant_job["id"], 0, "夜航变体")
    assert variant["id"] != voice["id"]
    assert variant["parent_id"] == voice["id"] and variant["revision"] == 2
    assert service.get_voice(voice["id"])["revision"] == 1


def test_reference_crop_cannot_inherit_full_recording_transcript(service, tmp_path):
    sr = 16000
    source = tmp_path / "long.wav"
    sf.write(source, np.sin(2 * np.pi * 180 * np.arange(sr * 15) / sr) * 0.2, sr)
    voice = service.import_voice("长参考", str(source), transcript="这是整段转写。")
    assert voice["transcript"] == ""
    assert voice["source_transcript"] == "这是整段转写。"
    assert voice["analysis"]["duration_seconds"] == 12
    assert hashlib.sha256(source.read_bytes()).hexdigest() == voice["source"]["sha256"]
    with pytest.raises(ValueError, match="转写"):
        service.submit_tts(voice["id"], "你好", mode="hifi")


def test_hifi_cannot_silently_ignore_expression_controls(service, tone):
    voice = service.import_voice("测试", str(tone), transcript="准确参考文本")
    with pytest.raises(ValueError, match="表达控制"):
        service.submit_tts(voice["id"], "你好", {"emotion": "angry"}, mode="hifi")
    with pytest.raises(ValueError, match="长度"):
        service.submit_vc(voice["id"], str(tone), {"length_adjust": 1.2})
    assert not service.runner.requests


def test_voice_folder_can_move_without_changing_identity(service, tone, tmp_path):
    voice = service.import_voice("可迁移", str(tone))
    stored = json.loads((service.voices_dir / voice["id"] / "profile.json").read_text(encoding="utf-8"))
    assert stored["reference_path"] == "reference.wav"
    moved = tmp_path / "moved"
    shutil.copytree(service.workspace, moved)
    other = VoiceService(moved, runner=service.runner)
    try:
        loaded = other.get_voice(voice["id"])
        assert Path(loaded["reference_path"]).is_file()
        assert str(moved) in loaded["reference_path"]
        assert loaded["source"]["sha256"] == voice["source"]["sha256"]
    finally:
        other.close()


def test_semantic_controls_are_explicit_requests():
    prompt = design_instruction(VoiceSpec(brightness=0.8, breathiness=0.3, roughness=0.8))
    assert "非常明亮" in prompt and "轻微气声" in prompt and "明显的沙哑" in prompt
    with pytest.raises(ValueError):
        VoiceSpec(brightness=3)
