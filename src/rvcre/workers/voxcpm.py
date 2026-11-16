"""VoxCPM2 adapter for the pinned OpenBMB interface; no substitute audio path."""

from __future__ import annotations

import json
from pathlib import Path

from ..storage import write_json
from ..text import NARRATION_ANCHOR_CHARS, NARRATION_SEGMENT_CHARS, TTS_SEGMENT_CHARS
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
    candidates = int(request.get("candidates", 1))
    if candidates < 1:
        raise ValueError("candidates must be positive.")
    narration = request.get("narration", False)
    if not isinstance(narration, bool):
        raise ValueError("narration must be a boolean.")
    if narration and request.get("task") != "tts":
        raise ValueError("Narration continuity is only available for tts tasks.")
    segments = None
    if "segments" in request:
        if request.get("task") != "tts":
            raise ValueError("Segmented generation is only available for tts tasks.")
        supplied = request["segments"]
        if not isinstance(supplied, list) or not supplied:
            raise ValueError("segments must be a non-empty list of text strings.")
        if any(not isinstance(segment, str) for segment in supplied):
            raise ValueError("Each segment must be a text string.")
        segments = [segment.strip() for segment in supplied]
        segment_limit = NARRATION_SEGMENT_CHARS if narration else TTS_SEGMENT_CHARS
        if any(not segment or len(segment) > segment_limit for segment in segments):
            raise ValueError(f"Each segment must contain 1–{segment_limit} characters after trimming whitespace.")
        if narration and len(segments) > 1 and len(segments[0]) > NARRATION_ANCHOR_CHARS:
            raise ValueError(f"The narration opening must contain at most {NARRATION_ANCHOR_CHARS} characters.")
        if candidates != 1:
            raise ValueError("Segmented tts requires exactly one candidate per segment.")
    first_request = request if segments is None else {**request, "text": segments[0]}
    first_args = generation_arguments(first_request, seed)
    model_config = json.loads((Path(config["model_dir"]) / "config.json").read_text(encoding="utf-8"))
    if model_config.get("architecture", "").lower() != "voxcpm2":
        raise RuntimeError("This adapter requires VoxCPM2 weights.")

    total = candidates if segments is None else len(segments)
    outputs = []
    metadata = {
        "backend": "voxcpm", "model": "openbmb/VoxCPM2",
        "upstream_revision": config.get("upstream_revision"),
        "model_revision": config.get("model_revision"),
        "asset_revisions": config.get("asset_revisions", {}),
        "mode": "design" if request["task"] == "design" else request.get("mode", "controllable"),
        "requested_seed": seed, "lora_weights_path": config.get("lora_weights_path"),
        "narration": narration,
        "continuity": "fixed_opening" if narration and segments and len(segments) > 1 else "independent",
    }

    def report(phase: str, current: int | None = None) -> None:
        if segments is not None:
            write_json(output_path.parent / "progress.json", {
                "total": total, "completed": len(outputs), "current": current,
                "phase": phase, "outputs": outputs, "metadata": metadata,
            })

    report("loading")
    import torch
    from voxcpm import VoxCPM

    device = checked_device(config, torch)
    metadata.update(device=str(device), torch_version=torch.__version__)
    print("Loading local VoxCPM2 once for this job.", flush=True)
    model = VoxCPM.from_pretrained(
        config["model_dir"], local_files_only=True, load_denoiser=False,
        optimize=False, device=str(device), lora_weights_path=config.get("lora_weights_path"),
    )
    metadata["sample_rate"] = model.tts_model.sample_rate
    for index in range(total):
        # TTS segments reuse the voice settings and requested seed. Design
        # candidates intentionally keep their existing incrementing seeds.
        requested_seed = (seed + index) % (2**32) if segments is None else seed
        segment_request = request if segments is None else {**request, "text": segments[index]}
        if narration and segments is not None and index > 0:
            # The opening WAV is a fixed continuation prompt for every later
            # clip. Keep the original reference identity as a separate input.
            # Do not feed later generations back: that would propagate drift.
            # Style instructions are not spoken, so they must not enter either
            # the prompt transcript or continuation target text.
            kwargs = generation_arguments({**segment_request, "instruction": ""}, requested_seed)
            # core.generate collapses target whitespace before speaking; the
            # continuation transcript must describe that same spoken content.
            kwargs.update(prompt_wav_path=outputs[0]["path"], prompt_text=" ".join(segments[0].split()))
        else:
            kwargs = first_args if index == 0 else generation_arguments(segment_request, requested_seed)
        report("generating", index + 1)
        print(f"Generating {index + 1}/{total}; requested seed {requested_seed}.", flush=True)
        wav = model.generate(**kwargs)
        path = output_path.parent / f"audio-{index + 1:03d}.wav"
        write_audio(path, wav, model.tts_model.sample_rate)
        used_seed = getattr(model.tts_model, "last_successful_seed", None)
        output = {
            "path": str(path.resolve()), "seed": used_seed if used_seed is not None else requested_seed,
            "requested_seed": requested_seed,
            "duration_seconds": len(wav) / model.tts_model.sample_rate,
            "prompt_source_segment": 1 if narration and segments is not None and index > 0 else None,
        }
        if segments is not None:
            output["segment_index"] = index + 1
        outputs.append(output)
        report("complete" if len(outputs) == total else "generating")
    return {"outputs": outputs, "metadata": metadata}


def main() -> None:
    config, request, output = arguments()
    write_result(output, run(config, request, output))


if __name__ == "__main__":
    main()
