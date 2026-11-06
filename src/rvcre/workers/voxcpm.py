"""VoxCPM2 adapter for the pinned OpenBMB interface; no substitute audio path."""

from __future__ import annotations

import json
from pathlib import Path

from .common import actual_seed, arguments, checked_device, write_audio, write_result


def generation_arguments(request: dict, seed: int) -> dict:
    task = request.get("task")
    if task not in {"design", "tts"}:
        raise ValueError("VoxCPM2 accepts design and tts tasks only.")
    text = (request.get("text") or "").strip()
    if not text:
        raise ValueError("Text is required.")
    instruction = (request.get("instruction") or "").strip()
    mode = request.get("mode", "controllable")
    if mode not in {"controllable", "hifi"}:
        raise ValueError("VoxCPM2 mode must be controllable or hifi.")
    settings = request.get("settings") or {}
    kwargs = {
        "text": text, "seed": seed,
        "cfg_value": settings.get("cfg_value", 2.0),
        "inference_timesteps": settings.get("inference_timesteps", 10),
        "normalize": settings.get("normalize", False),
    }
    if task == "design":
        if request.get("reference_path"):
            raise ValueError("Voice design does not accept a reference; use tts for cloning.")
        if mode == "hifi":
            raise ValueError("Hi-Fi cloning needs a reference; it is not a voice-design mode.")
    else:
        reference = request.get("reference_path")
        if not reference or not Path(reference).is_file():
            raise ValueError("TTS requires an existing reference_path.")
        kwargs["reference_wav_path"] = str(Path(reference).resolve())
        if mode == "hifi":
            if instruction:
                raise ValueError("Hi-Fi mode does not support style instructions. Use controllable mode.")
            transcript = (request.get("reference_text") or "").strip()
            if not transcript:
                raise ValueError("Hi-Fi mode requires the exact reference_text transcript.")
            kwargs.update(prompt_wav_path=kwargs["reference_wav_path"], prompt_text=transcript)
    if instruction:
        kwargs["text"] = f"({instruction}){text}"
    return kwargs


def run(config: dict, request: dict, output_path: Path) -> dict:
    seed = actual_seed(request.get("seed"))
    first_args = generation_arguments(request, seed)
    candidates = int(request.get("candidates", 1))
    if candidates < 1:
        raise ValueError("candidates must be positive.")
    model_config = json.loads((Path(config["model_dir"]) / "config.json").read_text(encoding="utf-8"))
    if model_config.get("architecture", "").lower() != "voxcpm2":
        raise RuntimeError("This adapter requires VoxCPM2 weights.")

    import torch
    from voxcpm import VoxCPM

    device = checked_device(config, torch)
    print("Loading local VoxCPM2 once for this job.", flush=True)
    model = VoxCPM.from_pretrained(
        config["model_dir"], local_files_only=True, load_denoiser=False,
        optimize=False, device=str(device), lora_weights_path=config.get("lora_weights_path"),
    )
    outputs = []
    for index in range(candidates):
        requested_seed = (seed + index) % (2**32)
        kwargs = first_args if index == 0 else generation_arguments(request, requested_seed)
        print(f"Generating {index + 1}/{candidates}; requested seed {requested_seed}.", flush=True)
        wav = model.generate(**kwargs)
        path = output_path.parent / f"audio-{index + 1:03d}.wav"
        write_audio(path, wav, model.tts_model.sample_rate)
        used_seed = getattr(model.tts_model, "last_successful_seed", None)
        outputs.append({"path": str(path.resolve()), "seed": used_seed if used_seed is not None else requested_seed})
    return {
        "outputs": outputs,
        "metadata": {
            "backend": "voxcpm", "model": "openbmb/VoxCPM2",
            "upstream_revision": config.get("upstream_revision"),
            "model_revision": config.get("model_revision"),
            "asset_revisions": config.get("asset_revisions", {}),
            "mode": "design" if request["task"] == "design" else request.get("mode", "controllable"),
            "requested_seed": seed, "sample_rate": model.tts_model.sample_rate,
            "device": str(device), "torch_version": torch.__version__,
            "lora_weights_path": config.get("lora_weights_path"),
        },
    }


def main() -> None:
    config, request, output = arguments()
    write_result(output, run(config, request, output))


if __name__ == "__main__":
    main()
