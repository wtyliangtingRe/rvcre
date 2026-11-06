from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml

from rvcre import dataset


def audio(path: Path, seconds: float = 1.2, sample_rate: int = 16000, stereo: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    waveform = (0.15 * np.sin(2 * np.pi * 190 * time)).astype(np.float32)
    if stereo:
        waveform = np.stack([waveform, waveform * 0.5], axis=1)
    sf.write(path, waveform, sample_rate)
    return path


def labelled(path: Path, seconds: float = 1.2, text: str = "今天的天气很好。", suffix: str = ".txt") -> Path:
    audio(path, seconds)
    path.with_suffix(suffix).write_text(text, encoding="utf-8")
    return path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_multi_speaker_split_references_and_portable_paths(tmp_path):
    source = tmp_path / "source"
    for speaker in ("角色甲", "speaker_b"):
        for index in range(5):
            labelled(source / speaker / f"clip{index}.wav", suffix=".lab" if index == 0 else ".txt")
    output = tmp_path / "prepared"
    summary = dataset.prepare_dataset(source, output, min_seconds=1, max_seconds=3, validation_ratio=0.4)
    assert summary["train_samples"] == 6
    assert summary["validation_samples"] == 4
    metadata = {item["audio"]: item for item in rows(output / "metadata.jsonl")}
    for split in ("train", "validation"):
        for item in rows(output / f"{split}.jsonl"):
            assert not Path(item["audio"]).is_absolute()
            assert (output / item["audio"]).is_file()
            assert sf.info(output / item["audio"]).samplerate == 16000
            assert item["ref_audio"] != item["audio"]
            assert metadata[item["ref_audio"]]["speaker"] == metadata[item["audio"]]["speaker"]
            assert metadata[item["ref_audio"]]["split"] == split
    moved = tmp_path / "moved"
    output.rename(moved)
    assert all((moved / item["audio"]).is_file() for item in rows(moved / "train.jsonl"))


def test_split_reproducibility_and_original_file_grouping_for_asr(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for index in range(4):
        audio(source / f"recording{index}.wav", seconds=8)
    calls = []

    class LocalASR:
        def transcribe(self, waveform, **kwargs):
            calls.append(kwargs)
            return iter([SimpleNamespace(start=float(i), end=float(i + 2), text=f"句子{i}。") for i in range(0, 8, 2)]), None

    monkeypatch.setattr(dataset, "_load_asr", lambda name: LocalASR())
    left, right = tmp_path / "left", tmp_path / "right"
    for output in (left, right):
        result = dataset.prepare_dataset(source, output, min_seconds=1, max_seconds=3, validation_ratio=0.5, seed=77, transcribe=True)
        assert result["asr_review_samples"] == 16
        assert result["train_samples"] == result["validation_samples"] == 8
    assert (left / "train.jsonl").read_bytes() == (right / "train.jsonl").read_bytes()
    splits = {}
    for item in rows(left / "metadata.jsonl"):
        splits.setdefault(item["source_id"], set()).add(item["split"])
    assert all(len(values) == 1 for values in splits.values())
    assert all(call["language"] == "zh" for call in calls)


def test_missing_text_and_long_unaligned_recording_are_reported_not_invented(tmp_path):
    source = tmp_path / "source"
    labelled(source / "valid.wav")
    audio(source / "missing.wav")
    labelled(source / "long.wav", seconds=5, text="一整篇文字，不能按秒数任意切开。")
    output = tmp_path / "prepared"
    result = dataset.prepare_dataset(source, output, min_seconds=1, max_seconds=3)
    assert result["accepted_samples"] == 1
    assert result["skipped_sources"] == 2
    assert {issue["code"] for issue in result["issues"]} == {"missing_transcript", "too_long_unaligned"}
    item = rows(output / "train.jsonl")[0]
    assert item["ref_audio"] is None
    assert not rows(output / "validation.jsonl")


def test_stereo_resampling_without_editing_source(tmp_path):
    source = tmp_path / "source"
    path = audio(source / "stereo.wav", sample_rate=22050, stereo=True)
    before = path.read_bytes()
    path.with_suffix(".txt").write_text("原文照录。", encoding="utf-8")
    output = tmp_path / "prepared"
    dataset.prepare_dataset(source, output, min_seconds=1, max_seconds=3)
    converted = sf.info(output / rows(output / "train.jsonl")[0]["audio"])
    assert converted.channels == 1
    assert converted.samplerate == 16000
    assert converted.duration == pytest.approx(1.2)
    assert path.read_bytes() == before


def test_asr_is_local_only_and_is_not_loaded_for_existing_transcripts(tmp_path, monkeypatch):
    source = tmp_path / "source"
    labelled(source / "a.wav")
    monkeypatch.setattr(dataset, "_load_asr", lambda name: pytest.fail("sidecar input must not load ASR"))
    dataset.prepare_dataset(source, tmp_path / "output", min_seconds=1, max_seconds=3, transcribe=True)

    import sys

    observed = {}

    def constructor(name, **kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=constructor))
    # Restore this specific implementation without importing a model.
    original = importlib.util.spec_from_file_location("dataset_for_local_asr_test", Path(dataset.__file__))
    module = importlib.util.module_from_spec(original)
    original.loader.exec_module(module)
    module._load_asr("local-model")
    assert observed["local_files_only"] is True
    assert observed["device"] == "cpu"


def test_unavailable_asr_preserves_labelled_data_and_reports_missing_model(tmp_path, monkeypatch):
    source = tmp_path / "source"
    labelled(source / "labelled.wav")
    audio(source / "missing.wav")

    def missing_model(_):
        raise RuntimeError("local model missing")

    monkeypatch.setattr(dataset, "_load_asr", missing_model)
    result = dataset.prepare_dataset(source, tmp_path / "output", min_seconds=1, max_seconds=3, transcribe=True)
    assert result["accepted_samples"] == 1
    assert any(issue["code"] == "asr_unavailable" for issue in result["issues"])


def test_ambiguous_layout_and_nonempty_output_require_explicit_choices(tmp_path):
    source = tmp_path / "source"
    labelled(source / "root.wav")
    labelled(source / "session" / "nested.wav")
    with pytest.raises(ValueError, match="说话人"):
        dataset.prepare_dataset(source, tmp_path / "output", min_seconds=1, max_seconds=3)
    dataset.prepare_dataset(source, tmp_path / "output", speaker="one speaker", min_seconds=1, max_seconds=3)
    with pytest.raises(FileExistsError):
        dataset.prepare_dataset(source, tmp_path / "output", speaker="one speaker")


@pytest.mark.parametrize("ratio", [-0.1, 1, float("nan")])
def test_invalid_split_ratio(tmp_path, ratio):
    source = tmp_path / "source"
    labelled(source / "a.wav")
    with pytest.raises(ValueError, match="validation_ratio"):
        dataset.prepare_dataset(source, tmp_path / "output", validation_ratio=ratio)


def test_timestamped_lab_is_not_trained_as_literal_words(tmp_path):
    source = tmp_path / "source"
    labelled(source / "a.wav", text="0 10000000 你好\n10000000 12000000 世界", suffix=".lab")
    result = dataset.prepare_dataset(source, tmp_path / "output", min_seconds=1, max_seconds=3)
    assert result["accepted_samples"] == 0
    assert result["issues"][0]["code"] == "invalid_source"


def test_asr_invalid_or_overlong_segments_do_not_get_fabricated_splits():
    clips, issues = dataset._asr_clips([
        {"start": 0, "end": 4, "text": "一个不能在内部可靠对齐的长片段。"},
        {"start": 4, "end": 6, "text": "有效片段。"},
        {"start": 5, "end": 7, "text": "重叠片段。"},
        {"start": 7, "end": 8, "text": None},
    ], duration=8, minimum=1, maximum=3)
    assert clips == [(4.0, 6.0, "有效片段。")]
    assert {issue["code"] for issue in issues} == {"asr_segment_too_long", "invalid_asr_segment"}


def test_lora_wrapper_uses_upstream_template_and_real_entrypoint(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/train_voxcpm.py"
    spec = importlib.util.spec_from_file_location("rvcre_train_test", script)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    monkeypatch.setattr(train, "_backend_settings", lambda: {})
    repo = tmp_path / "upstream"
    (repo / train.TEMPLATE).parent.mkdir(parents=True)
    (repo / train.TRAINER).parent.mkdir(parents=True)
    template = {
        "pretrained_path": "old", "train_manifest": "old", "val_manifest": None,
        "sample_rate": 16000, "out_sample_rate": 48000, "batch_size": 2,
        "grad_accum_steps": 8, "num_workers": 8, "preprocessing_num_workers": 8,
        "num_iters": 1000, "max_steps": 1000, "warmup_steps": 100,
        "save_interval": 500, "valid_interval": 500, "learning_rate": 0.0001,
        "save_path": "old", "tensorboard": "old", "lora": {"enable_lm": True, "enable_dit": True, "enable_proj": False, "r": 32, "alpha": 32},
    }
    (repo / train.TEMPLATE).write_text(yaml.safe_dump(template), encoding="utf-8")
    fields = ", ".join(f"{key}=None" for key in [*template, "distribute"])
    (repo / train.TRAINER).write_text(f"def train({fields}):\n    pass\n", encoding="utf-8")
    labelled(tmp_path / "raw" / "a.wav")
    dataset.prepare_dataset(tmp_path / "raw", tmp_path / "prepared", min_seconds=1, max_seconds=3)
    output = tmp_path / "run"
    args = train.parser().parse_args(["--data", str(tmp_path / "prepared"), "--output", str(output), "--repo", str(repo), "--model", str(tmp_path / "missing-model"), "--steps", "12", "--config-only"])
    plan = train.create_plan(args)
    actual = yaml.safe_load(Path(plan["config"]).read_text(encoding="utf-8"))
    assert actual["sample_rate"] == 16000
    assert actual["out_sample_rate"] == 48000
    assert actual["lora"]["enable_dit"] is True
    assert actual["num_iters"] == actual["max_steps"] == 12
    assert actual["val_manifest"] == ""
    assert plan["command"][-2:] == ["--config_path", str(output / "finetune_lora.yaml")]
    assert plan["environment"]["HF_HUB_OFFLINE"] == "1"
    assert plan["config_only"] is True


def test_training_uses_runner_relative_config_path_contract(tmp_path, monkeypatch):
    import shutil

    script = Path(__file__).resolve().parents[1] / "scripts/train_voxcpm.py"
    spec = importlib.util.spec_from_file_location("rvcre_train_paths_test", script)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    config = tmp_path / "config" / "backends.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"voxcpm": {"upstream_dir": "../third_party/VoxCPM", "model_dir": "../models/VoxCPM2", "python": "python"}}), encoding="utf-8")
    monkeypatch.setenv("RVCRE_BACKEND_CONFIG", str(config))
    settings = train._backend_settings()
    assert settings["upstream_dir"] == str(tmp_path / "third_party/VoxCPM")
    assert settings["model_dir"] == str(tmp_path / "models/VoxCPM2")
    assert settings["python"] == shutil.which("python")
