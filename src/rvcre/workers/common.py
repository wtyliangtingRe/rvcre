"""Worker protocol and audio writing, without importing ML libraries at startup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets


def arguments() -> tuple[dict, dict, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    return payload["config"], payload["request"], args.output.resolve()


def actual_seed(value) -> int:
    return secrets.randbits(32) if value is None else int(value)


def write_result(path: Path, result: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_audio(path: Path, waveform, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    samples = np.asarray(waveform)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise RuntimeError("Model returned empty, non-mono or non-finite audio.")
    sf.write(str(path), samples, int(sample_rate), subtype="PCM_24")


def checked_device(config: dict, torch):
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Check the NVIDIA driver and CUDA PyTorch wheel, "
            "or explicitly configure device='cpu'."
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return device
