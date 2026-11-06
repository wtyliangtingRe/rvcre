"""Seed-VC v2 timbre conversion using the actual upstream streaming interface."""

from __future__ import annotations

import importlib
from pathlib import Path
import random
import sys

from .common import actual_seed, arguments, checked_device, write_audio, write_result


def conversion_arguments(request: dict) -> dict:
    if request.get("task") != "vc":
        raise ValueError("Seed-VC accepts vc tasks only.")
    for field in ("source_path", "reference_path"):
        if not request.get(field) or not Path(request[field]).is_file():
            raise ValueError(f"An existing {field} is required for voice conversion.")
    settings = request.get("settings") or {}
    if float(settings.get("length_adjust", 1.0)) != 1.0:
        raise ValueError("Seed-VC v2 ignores length_adjust in timbre-only mode; use 1.0.")
    if (request.get("instruction") or "").strip():
        raise ValueError("Seed-VC timbre conversion does not accept text style instructions.")
    return {
        "source_audio_path": str(Path(request["source_path"]).resolve()),
        "target_audio_path": str(Path(request["reference_path"]).resolve()),
        "diffusion_steps": settings.get("diffusion_steps", 30),
        "length_adjust": 1.0,
        # This misspelling is part of the pinned upstream API.
        "intelligebility_cfg_rate": settings.get("intelligibility_cfg_rate", 0.7),
        "similarity_cfg_rate": settings.get("similarity_cfg_rate", 0.7),
        "convert_style": False, "anonymization_only": False,
        "stream_output": True,
    }


def local_checkpoint_loader(model_dir: Path):
    directories = {
        "Plachta/Seed-VC": "main",
        "Plachta/ASTRAL-quantization": "astral",
        "funasr/campplus": "campplus",
    }

    def resolve(repo_id, model_filename="pytorch_model.bin", config_filename=None):
        base = model_dir / directories[repo_id]
        model = base / model_filename
        if not model.is_file():
            raise FileNotFoundError(f"Missing local model asset: {model}")
        if config_filename is not None:
            configuration = base / config_filename
            if not configuration.is_file():
                raise FileNotFoundError(configuration)
            return str(model), str(configuration)
        return str(model)

    return resolve


def run(config: dict, request: dict, output_path: Path) -> dict:
    kwargs = conversion_arguments(request)
    upstream = Path(config["upstream_dir"])
    model_dir = Path(config["model_dir"])
    sys.path.insert(0, str(upstream))

    import imageio_ffmpeg
    import numpy as np
    import torch
    import yaml
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    from pydub import AudioSegment

    # Upstream yields final PCM alongside MP3 chunks. Its MP3 encoder needs FFmpeg;
    # use the platform binary supplied by imageio-ffmpeg, including on Windows.
    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
    device = checked_device(config, torch)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    with (upstream / "configs/v2/vc_wrapper.yaml").open(encoding="utf-8") as source:
        model_config = yaml.safe_load(source)
    for name in ("content_extractor_narrow", "content_extractor_wide"):
        model_config[name]["tokenizer_name"] = str(model_dir / "whisper-tokenizer")
        model_config[name]["ssl_model_name"] = str(model_dir / "hubert")
    model_config["vocoder"]["pretrained_model_name_or_path"] = str(model_dir / "bigvgan")
    wrapper_module = importlib.import_module("modules.v2.vc_wrapper")
    # Only replace the download resolver. Model architecture and inference stay upstream.
    wrapper_module.load_custom_model_from_hf = local_checkpoint_loader(model_dir)
    print("Loading local Seed-VC v2 and its content/style encoders.", flush=True)
    model = instantiate(DictConfig(model_config))
    model.load_checkpoints(
        ar_checkpoint_path=str(model_dir / "main/v2/ar_base.pth"),
        cfm_checkpoint_path=str(model_dir / "main/v2/cfm_small.pth"),
    )
    model.to(device).eval()
    model.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    seed = actual_seed(request.get("seed"))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    full_audio = None
    print(f"Converting timbre with seed {seed}; style conversion is disabled.", flush=True)
    for _, audio in model.convert_voice_with_streaming(**kwargs, device=device, dtype=dtype):
        if audio is not None:
            full_audio = audio
    if full_audio is None:
        raise RuntimeError("Seed-VC returned no complete waveform.")
    sample_rate, wav = full_audio
    path = output_path.parent / "audio-001.wav"
    write_audio(path, wav, sample_rate)
    return {
        "outputs": [{"path": str(path.resolve()), "seed": seed}],
        "metadata": {
            "backend": "seedvc", "model": "Plachta/Seed-VC v2",
            "upstream_revision": config.get("upstream_revision"),
            "model_revision": config.get("model_revision"),
            "asset_revisions": config.get("asset_revisions", {}),
            "sample_rate": sample_rate, "device": str(device), "torch_version": torch.__version__,
            "convert_style": False, "anonymization_only": False, "length_adjust": 1.0,
            "reference_max_seconds": 25,
        },
    }


def main() -> None:
    config, request, output = arguments()
    write_result(output, run(config, request, output))


if __name__ == "__main__":
    main()
