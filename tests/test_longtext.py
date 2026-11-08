"""Long-form TTS orchestration with synthetic PCM fixtures, never real inference.

The assertions cover text preservation, one backend job, recoverable segment
outputs, and sample-accurate concatenation. Acoustic quality is not tested here.
"""

from copy import deepcopy
import json
from pathlib import Path
import threading
import time
import zipfile

import numpy as np
import pytest
import soundfile as sf

from rvcre.audio import merge_audio_files
from rvcre.service import VoiceService
from rvcre.storage import write_json
from rvcre.text import split_tts_text


LONG_TEXT = "\n\n".join([
    "工厂重新开工之后，产品还需要卖到国外。" * 24,
    "进出口恢复的同时，政府也在调整货币政策。" * 23,
    "普通家庭的生活，随着收入和就业逐步改变。" * 22,
])


def without_space(text):
    return "".join(text.split())


def terminal(service, identifier):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = service.get_job(identifier)
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    pytest.fail("Synthetic long-text job did not terminate")


class BatchRunner:
    """Records the one whole-article request and emits short analytic signals."""

    def __init__(self, *, fail_after=None, wait_after=None, outside_path=None):
        self.requests = []
        self.fail_after = fail_after
        self.wait_after = wait_after
        self.outside_path = outside_path
        self.reached_pause = threading.Event()

    def run(self, backend, request, directory, cancel_event=None):
        self.requests.append((backend, deepcopy(request)))
        texts = request.get("segments") or [request["text"]]
        outputs = []
        for index, _ in enumerate(texts, 1):
            path = self.outside_path or directory / f"audio-{index:03d}.wav"
            sf.write(path, np.full(80, index / 16, dtype="float32"), 8000, subtype="PCM_24")
            outputs.append({"path": str(path), "seed": request["seed"], "segment_index": index})
            write_json(directory / "progress.json", {
                "total": len(texts), "completed": index,
                "current": index + 1 if index < len(texts) else None,
                "phase": "generating" if index < len(texts) else "complete",
                "outputs": outputs, "metadata": {"test_double": True, "sample_rate": 8000},
            })
            if index == self.fail_after:
                raise RuntimeError("Test backend failed on the following segment")
            if index == self.wait_after:
                self.reached_pause.set()
                if not cancel_event.wait(5):
                    raise RuntimeError("Test did not request cancellation in time")
                raise RuntimeError("Test backend was cancelled")
        return {"outputs": outputs, "metadata": {"test_double": True, "sample_rate": 8000}}


@pytest.fixture
def prepare_service(tmp_path, tone, monkeypatch):
    # Reference analysis has its own tests. Avoid running pYIN in each queue test.
    monkeypatch.setattr("rvcre.service.analyze_audio", lambda _: {"pitch_median_hz": None, "notes": []})
    services = []

    def create(runner=None):
        service = VoiceService(tmp_path / f"workspace-{len(services)}", runner=runner or BatchRunner())
        services.append(service)
        voice = service.import_voice("旁白参考", str(tone), transcript="与参考片段对应的文字。")
        return service, voice

    yield create
    for service in services:
        service.close()


def test_short_text_is_unchanged_and_blank_input_has_no_segments():
    assert split_tts_text(" \n\t ") == []
    assert split_tts_text("  第一段。\n\nSecond paragraph.\n") == ["第一段。\n\nSecond paragraph."]
    assert split_tts_text("字" * 600) == ["字" * 600]


def test_long_text_uses_paragraph_boundaries_and_keeps_all_content():
    paragraphs = LONG_TEXT.split("\n\n")
    chunks = split_tts_text(LONG_TEXT)
    assert chunks == paragraphs
    assert all(0 < len(chunk) <= 600 for chunk in chunks)
    assert without_space("".join(chunks)) == without_space(LONG_TEXT)


def test_sentences_keep_their_closing_quotes():
    sentence = "他说：“工厂开动以后，订单也要跟上。”"
    chunks = split_tts_text(sentence * 4, max_chars=len(sentence))
    assert chunks == [sentence] * 4


def test_english_words_and_decimal_numbers_stay_intact():
    sentence = "Dr. Klein measured 3.14 percent growth in 1950. "
    source = sentence * 8
    chunks = split_tts_text(source, max_chars=25)
    assert all(0 < len(chunk) <= 25 for chunk in chunks)
    assert " ".join(chunks).split() == source.split()


def test_unpunctuated_text_falls_back_to_bounded_chunks_without_loss():
    source = "无标点内容ABC123🔊" * 100
    chunks = split_tts_text(source, max_chars=97)
    assert all(0 < len(chunk) <= 97 for chunk in chunks)
    assert "".join(chunks) == source


def test_whole_article_is_one_job_with_merged_audio_and_an_editable_segment_archive(prepare_service):
    service, voice = prepare_service()
    performance = {"speed": 1.15, "emotion": "calm", "instruction": "清楚自然地讲述"}
    submitted = service.submit_tts(voice["id"], LONG_TEXT, performance, seed=1234)
    job = terminal(service, submitted["id"])
    assert job["status"] == "succeeded", job
    assert len(service.runner.requests) == 1
    backend, request = service.runner.requests[0]
    assert backend == "voxcpm"
    assert request["text"] == LONG_TEXT
    assert request["segments"] == LONG_TEXT.split("\n\n")
    assert request["reference_path"] == voice["reference_path"]
    assert request["reference_text"] == voice["transcript"]
    assert request["voice_revision"] == voice["revision"]
    assert request["performance"] == performance
    assert request["mode"] == "controllable"
    assert request["seed"] == 1234
    assert job["outputs"][0]["kind"] == "merged"
    segments = job["outputs"][1:]
    assert len(segments) == len(request["segments"])
    assert all(output["kind"] == "segment" for output in segments)
    assert service.audio_path(job["id"], 0).name == "merged.wav"
    assert job["progress"]["completed"] == job["progress"]["total"] == len(segments)

    merged, rate = sf.read(service.audio_path(job["id"], 0), dtype="float32")
    assert rate == 8000
    assert len(merged) == len(segments) * 80 + (len(segments) - 1) * 1600
    archive = next(Path(path) for path in job["files"] if Path(path).name == "segments.zip")
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        assert {"manifest.json", "full-text.txt"} <= names
        assert "merged.wav" not in names
        assert package.read("full-text.txt").decode("utf-8").strip() == LONG_TEXT
        manifest = json.loads(package.read("manifest.json"))
        assert manifest  # A readable manifest accompanies every exported batch.
        for index, (text, output) in enumerate(zip(request["segments"], segments), 1):
            assert package.read(f"text-{index:03d}.txt").decode("utf-8").strip() == text
            assert package.read(f"audio-{index:03d}.wav") == Path(output["path"]).read_bytes()


def test_merge_preserves_stereo_samples_and_inserts_only_internal_silence(tmp_path):
    first = np.array([[0.125, -0.25], [0.375, 0.5]], dtype="float32")
    second = np.array([[-0.625, 0.75], [0, -0.125], [0.25, 0.375]], dtype="float32")
    paths = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for path, samples in zip(paths, [first, second]):
        sf.write(path, samples, 8000, subtype="PCM_24")
    originals = [path.read_bytes() for path in paths]
    output = tmp_path / "merged.wav"
    metadata = merge_audio_files(paths, output, gap_seconds=0.001)
    actual, rate = sf.read(output, dtype="float32", always_2d=True)
    expected = np.concatenate([first, np.zeros((8, 2), dtype="float32"), second])
    assert rate == 8000
    np.testing.assert_array_equal(actual, expected)
    assert sf.info(output).subtype == "PCM_24"
    assert metadata["sample_rate"] == 8000 and metadata["channels"] == 2
    assert metadata["duration_seconds"] == pytest.approx(len(expected) / 8000)
    assert [path.read_bytes() for path in paths] == originals


@pytest.mark.parametrize("second_rate,second_channels", [(16000, 1), (8000, 2)])
def test_merge_rejects_incompatible_clip_formats(tmp_path, second_rate, second_channels):
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    sf.write(first, np.full(80, 0.125), 8000)
    sf.write(second, np.full((80, second_channels), 0.25), second_rate)
    with pytest.raises(ValueError):
        merge_audio_files([first, second], tmp_path / "merged.wav")


def test_failed_batch_retains_completed_audio_and_its_text(prepare_service):
    service, voice = prepare_service(BatchRunner(fail_after=1))
    submitted = service.submit_tts(voice["id"], LONG_TEXT)
    job = terminal(service, submitted["id"])
    assert job["status"] == "failed"
    assert "Test backend failed" in job["error"]
    assert job["progress"]["completed"] == 1
    assert job["progress"]["total"] == 3
    assert len(job["outputs"]) == 1
    assert all(output.get("kind") != "merged" for output in job["outputs"])
    clip = service.audio_path(job["id"], 0)
    assert clip.is_file()
    downloads = [Path(path) for path in job["files"]]
    assert clip in downloads
    assert any(path.name == "text-001.txt" and path.is_file() for path in downloads)
    assert not any(path.name == "merged.wav" for path in downloads)
    assert not (clip.parent / "merged.wav").exists()


def test_cancelled_batch_keeps_completed_segments_without_claiming_completion(prepare_service):
    runner = BatchRunner(wait_after=1)
    service, voice = prepare_service(runner)
    submitted = service.submit_tts(voice["id"], LONG_TEXT)
    assert runner.reached_pause.wait(3)
    active = service.get_job(submitted["id"])
    assert active["status"] == "running"
    assert active["progress"]["completed"] == 1
    service.cancel_job(submitted["id"])
    job = terminal(service, submitted["id"])
    assert job["status"] == "cancelled"
    assert len(job["outputs"]) == 1
    assert service.audio_path(job["id"], 0).is_file()
    assert job["progress"]["completed"] < job["progress"]["total"]
    assert all(Path(path).name != "merged.wav" for path in job["files"])


def test_backend_cannot_expose_or_archive_audio_outside_its_job(prepare_service, tmp_path):
    outside = tmp_path / "not-a-job-output.wav"
    service, voice = prepare_service(BatchRunner(outside_path=outside))
    submitted = service.submit_tts(voice["id"], LONG_TEXT)
    job = terminal(service, submitted["id"])
    assert job["status"] == "failed"
    assert job["outputs"] == []
    assert str(outside) not in job.get("files", [])
    with pytest.raises((FileNotFoundError, ValueError)):
        service.audio_path(job["id"], 0)
    assert not (service.workspace / "jobs" / job["id"] / "segments.zip").exists()


def test_api_accepts_entire_script_and_serves_the_completed_archive(prepare_service):
    from fastapi.testclient import TestClient

    from rvcre.api import create_app

    service, voice = prepare_service()
    with TestClient(create_app(service=service)) as client:
        response = client.post("/api/tts", json={"voice_id": voice["id"], "text": LONG_TEXT})
        assert response.status_code == 202, response.text
        job = terminal(service, response.json()["id"])
        assert job["status"] == "succeeded", job
        root = f"/api/jobs/{job['id']}"
        reported = client.get(root).json()
        assert reported["progress"]["completed"] == 3
        assert len(reported["files"]) == 1
        archive = client.get(root + "/files/0")
        assert archive.status_code == 200
        assert "segments.zip" in archive.headers["content-disposition"]
        assert archive.content == Path(job["files"][0]).read_bytes()
        merged = client.get(root + "/audio/0")
        assert merged.status_code == 200 and merged.content.startswith(b"RIFF")
        assert client.get(root + "/files/-1").status_code == 404
        assert client.get(root + "/files/1").status_code == 404
