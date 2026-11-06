"""Adapter contracts and real subprocess lifecycle, without downloading models.

Test doubles below exercise orchestration only; they are not inference fallbacks
and no acoustic quality or GPU compatibility is inferred from these tests.
"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

import pytest

from rvcre.backends import BackendCancelled, BackendRunner
from rvcre.backends import runner as runner_module
from rvcre.backends.runner import read_config
from rvcre.backends.specs import MODEL_ASSETS, REQUIRED_MODEL_FILES
from rvcre.workers.seedvc import conversion_arguments, local_checkpoint_loader
from rvcre.workers.voxcpm import generation_arguments


@pytest.fixture
def audio_paths(tmp_path):
    # These files are only path fixtures; the argument builders don't decode audio.
    source, reference = tmp_path / "source.wav", tmp_path / "reference.wav"
    source.write_bytes(b"path-fixture")
    reference.write_bytes(b"path-fixture")
    return source, reference


def test_voxcpm_controllable_cloning_uses_reference_without_transcript(audio_paths):
    _, reference = audio_paths
    args = generation_arguments({
        "task": "tts", "text": "你好。", "reference_path": str(reference),
        "instruction": "轻声，语速稍慢", "settings": {"cfg_value": 1.8},
    }, 71)
    assert args["text"] == "(轻声，语速稍慢)你好。"
    assert args["reference_wav_path"] == str(reference)
    assert "prompt_wav_path" not in args
    assert args["seed"] == 71
    assert args["cfg_value"] == 1.8


def test_voxcpm_hifi_combines_reference_and_continuation(audio_paths):
    _, reference = audio_paths
    request = {
        "task": "tts", "text": "新的一句。", "reference_path": str(reference),
        "reference_text": "参考音频内容。", "mode": "hifi",
    }
    args = generation_arguments(request, 2)
    assert args["prompt_wav_path"] == args["reference_wav_path"]
    assert args["prompt_text"] == request["reference_text"]
    with pytest.raises(ValueError, match="does not support style"):
        generation_arguments({**request, "instruction": "开心"}, 2)
    with pytest.raises(ValueError, match="exact reference_text"):
        generation_arguments({**request, "reference_text": ""}, 2)


def test_voxcpm_design_does_not_invent_reference_conditions():
    args = generation_arguments({"task": "design", "text": "Hello", "instruction": "warm alto"}, 1)
    assert args["text"] == "(warm alto)Hello"
    assert "reference_wav_path" not in args
    with pytest.raises(ValueError, match="not a voice-design mode"):
        generation_arguments({"task": "design", "text": "Hello", "mode": "hifi"}, 1)


def test_seed_v2_preserves_source_style_and_uses_real_api_spelling(audio_paths):
    source, reference = audio_paths
    request = {
        "task": "vc", "source_path": str(source), "reference_path": str(reference),
        "settings": {"intelligibility_cfg_rate": 0.6, "similarity_cfg_rate": 0.8},
    }
    args = conversion_arguments(request)
    assert args["convert_style"] is False
    assert args["anonymization_only"] is False
    assert args["intelligebility_cfg_rate"] == 0.6
    assert "intelligibility_cfg_rate" not in args
    assert args["length_adjust"] == 1.0
    with pytest.raises(ValueError, match="ignores length_adjust"):
        conversion_arguments({**request, "settings": {"length_adjust": 1.2}})
    with pytest.raises(ValueError, match="does not accept text"):
        conversion_arguments({**request, "instruction": "whisper"})


def test_seed_checkpoint_resolution_never_downloads(tmp_path):
    checkpoint = tmp_path / "astral" / "bsq32" / "bsq32_light.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint path fixture")
    resolve = local_checkpoint_loader(tmp_path)
    assert resolve("Plachta/ASTRAL-quantization", "bsq32/bsq32_light.pth") == str(checkpoint)
    with pytest.raises(FileNotFoundError, match="Missing local"):
        resolve("Plachta/Seed-VC", "v2/cfm_small.pth")


def test_voxcpm_candidates_load_once_and_record_actual_retry_seeds(tmp_path, monkeypatch):
    from rvcre.workers import voxcpm as worker

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"architecture":"voxcpm2"}')
    calls, loads = [], []

    class FakeDevice:
        type = "cpu"

        def __str__(self):
            return "cpu"

    class FakeModel:
        tts_model = types.SimpleNamespace(sample_rate=24000, last_successful_seed=None)

        def generate(self, **kwargs):
            calls.append(kwargs)
            self.tts_model.last_successful_seed = kwargs["seed"] + 1
            return [0.1, -0.1, 0.0]

    def from_pretrained(*args, **kwargs):
        loads.append((args, kwargs))
        return FakeModel()

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=lambda _: FakeDevice(), __version__="test"))
    monkeypatch.setitem(sys.modules, "voxcpm", types.SimpleNamespace(VoxCPM=types.SimpleNamespace(from_pretrained=from_pretrained)))
    result = worker.run(
        {"model_dir": str(model_dir), "device": "cpu", "lora_weights_path": str(tmp_path / "lora")},
        {"task": "design", "text": "测试句。", "seed": 30, "candidates": 3},
        tmp_path / "result.json",
    )
    assert len(loads) == 1
    assert loads[0][1]["local_files_only"] is True
    assert loads[0][1]["load_denoiser"] is False
    assert loads[0][1]["optimize"] is False
    assert loads[0][1]["device"] == "cpu"
    assert loads[0][1]["lora_weights_path"] == str(tmp_path / "lora")
    assert [call["seed"] for call in calls] == [30, 31, 32]
    assert [output["seed"] for output in result["outputs"]] == [31, 32, 33]
    assert all(Path(output["path"]).is_file() for output in result["outputs"])


def test_seed_worker_uses_local_assets_and_full_pcm_return(tmp_path, monkeypatch):
    from rvcre.workers import seedvc as worker

    monkeypatch.setattr(sys, "path", list(sys.path))
    upstream = tmp_path / "upstream"
    (upstream / "configs/v2").mkdir(parents=True)
    (upstream / "configs/v2/vc_wrapper.yaml").write_text(json.dumps({
        "content_extractor_narrow": {}, "content_extractor_wide": {}, "vocoder": {},
    }))
    reference, source = tmp_path / "ref.wav", tmp_path / "source.wav"
    reference.touch()
    source.touch()
    observed = {}

    class Device:
        type = "cpu"

        def __str__(self):
            return "cpu"

    class Model:
        def load_checkpoints(self, **kwargs):
            observed["checkpoints"] = kwargs

        def to(self, device):
            return self

        def eval(self):
            return self

        def setup_ar_caches(self, **kwargs):
            observed["cache"] = kwargs

        def convert_voice_with_streaming(self, **kwargs):
            observed["conversion"] = kwargs
            yield b"chunk fixture", None
            yield b"last chunk fixture", (22050, [0.1, 0.0, -0.1])

    def instantiate(config):
        observed["configuration"] = config
        return Model()

    module = types.SimpleNamespace()
    stubs = {
        "torch": types.SimpleNamespace(
            device=lambda _: Device(), float16="float16", float32="float32", __version__="test",
            manual_seed=lambda seed: observed.update(seed=seed),
        ),
        "imageio_ffmpeg": types.SimpleNamespace(get_ffmpeg_exe=lambda: "/test/ffmpeg"),
        "yaml": types.SimpleNamespace(safe_load=json.load),
        "hydra": types.ModuleType("hydra"),
        "hydra.utils": types.SimpleNamespace(instantiate=instantiate),
        "omegaconf": types.SimpleNamespace(DictConfig=lambda value: value),
        "pydub": types.SimpleNamespace(AudioSegment=types.SimpleNamespace()),
        "modules.v2.vc_wrapper": module,
    }
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    model_dir = tmp_path / "models"
    result = worker.run(
        {"model_dir": str(model_dir), "upstream_dir": str(upstream), "device": "cpu"},
        {"task": "vc", "source_path": str(source), "reference_path": str(reference), "seed": 17},
        tmp_path / "result.json",
    )
    assert observed["configuration"]["vocoder"]["pretrained_model_name_or_path"] == str(model_dir / "bigvgan")
    assert observed["configuration"]["content_extractor_wide"]["ssl_model_name"] == str(model_dir / "hubert")
    assert observed["conversion"]["convert_style"] is False
    assert observed["conversion"]["anonymization_only"] is False
    assert observed["seed"] == result["outputs"][0]["seed"] == 17
    assert result["metadata"]["sample_rate"] == 22050
    assert Path(result["outputs"][0]["path"]).is_file()
    assert callable(module.load_custom_model_from_hf)


@pytest.fixture
def configured_runner(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    (upstream / "src/voxcpm").mkdir(parents=True)
    (upstream / "src/voxcpm/core.py").write_text("# test checkout")
    model = tmp_path / "models"
    model.mkdir()
    for filename in REQUIRED_MODEL_FILES["voxcpm"]:
        (model / filename).write_bytes(b"path-fixture")
    configuration = tmp_path / "backends.json"
    configuration.write_text(json.dumps({"voxcpm": {
        "python": sys.executable, "upstream_dir": str(upstream), "model_dir": str(model), "device": "cpu",
    }}))
    monkeypatch.setattr(runner_module, "_probe_modules", lambda *args: [])
    return BackendRunner(configuration)


def test_status_is_lazy_and_reports_missing_paths(tmp_path):
    before = set(sys.modules)
    result = BackendRunner(tmp_path / "missing.json").status()
    assert [(item["name"], item["ready"]) for item in result] == [("voxcpm", False), ("seedvc", False)]
    assert "torch" not in set(sys.modules) - before
    assert not (tmp_path / "missing.json").exists()


def test_status_reports_missing_dependency(configured_runner, monkeypatch):
    monkeypatch.setattr(runner_module, "_probe_modules", lambda *args: ["voxcpm"])
    status = configured_runner.status()[0]
    assert status["ready"] is False
    assert "Missing Python dependencies: voxcpm" in status["details"]


def test_relative_configuration_paths_are_resolved_from_config_directory(tmp_path):
    directory = tmp_path / "config"
    directory.mkdir()
    path = directory / "backends.json"
    path.write_text(json.dumps({"seedvc": {"python": sys.executable, "model_dir": "../models", "upstream_dir": "../upstream"}}))
    config = read_config(path)["seedvc"]
    assert config["model_dir"] == str(tmp_path / "models")
    assert config["upstream_dir"] == str(tmp_path / "upstream")


def test_runner_collects_result_and_captures_logs(configured_runner, tmp_path, monkeypatch):
    # A real child process writes a tiny PCM fixture, testing only the worker protocol.
    code = (
        "import json,pathlib,sys,wave; p=pathlib.Path(sys.argv[1]); w=p.parent/'fixture.wav'; "
        "f=wave.open(str(w),'wb');f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);"
        "f.writeframes(b'\\0\\0'*16);f.close(); "
        "p.write_text(json.dumps({'outputs':[{'path':str(w),'seed':42}],'metadata':{'test_fixture':True}}));"
        "print('worker protocol test')"
    )
    monkeypatch.setattr(configured_runner, "_command", lambda backend, entry, request, result: [sys.executable, "-c", code, str(result)])
    result = configured_runner.run("voxcpm", {"task": "design", "text": "test"}, tmp_path / "job")
    assert result["outputs"][0]["seed"] == 42
    assert "worker protocol test" in (tmp_path / "job/backend.log").read_text()


def test_runner_propagates_failure_without_fallback(configured_runner, tmp_path, monkeypatch):
    monkeypatch.setattr(configured_runner, "_command", lambda *args: [sys.executable, "-c", "raise RuntimeError('model load failed')"])
    job = tmp_path / "failure"
    job.mkdir()
    (job / "result.json").write_text('{"outputs":[{"path":"stale.wav"}]}')
    with pytest.raises(RuntimeError, match="model load failed"):
        configured_runner.run("voxcpm", {"task": "design"}, job)
    assert not (job / "result.json").exists()
    assert not list(job.glob("*.wav"))


def test_runner_cancellation_terminates_worker(configured_runner, tmp_path, monkeypatch):
    worker_processes = []
    worker_command = [sys.executable, "-c", "import time;print('running',flush=True);time.sleep(30)"]
    original_popen = subprocess.Popen

    def capture_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        command = args[0] if args else kwargs.get("args")
        # Windows also starts taskkill to terminate the worker's process tree.
        if command == worker_command:
            worker_processes.append(process)
        return process

    monkeypatch.setattr(runner_module.subprocess, "Popen", capture_process)
    monkeypatch.setattr(configured_runner, "_command", lambda *args: worker_command)
    event = threading.Event()
    timer = threading.Timer(0.3, event.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(BackendCancelled):
            configured_runner.run("voxcpm", {"task": "design"}, tmp_path / "cancel", event)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 5
    assert len(worker_processes) == 1
    assert worker_processes[0].poll() is not None
    assert not (tmp_path / "cancel/result.json").exists()


def test_download_manifest_is_pinned_and_excludes_unneeded_whisper_weights():
    for assets in MODEL_ASSETS.values():
        for asset in assets:
            assert len(asset["revision"]) == 40
            int(asset["revision"], 16)
    whisper = next(asset for asset in MODEL_ASSETS["seedvc"] if asset["repo_id"] == "openai/whisper-small")
    assert not any(name.endswith((".bin", ".safetensors", ".pt")) for name in whisper["files"])


def test_setup_filters_conflicting_upstream_torch_pins():
    path = Path(__file__).resolve().parents[1] / "scripts/setup_backends.py"
    spec = importlib.util.spec_from_file_location("rvcre_setup_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.seed_requirements(
        "torch --pre --index-url nightly\ntorch==2.4.0\ntorchvision==0.19.0\ntorchaudio==2.4.0\nlibrosa==0.10.2\n"
    )
    assert "torch" not in result
    assert "librosa==0.10.2" in result
    assert "imageio-ffmpeg" in result
