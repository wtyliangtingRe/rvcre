"""Small audio utilities; original recordings are preserved separately from analysis references."""

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"音频不存在：{path}")
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        with tempfile.TemporaryDirectory(prefix="rvcre-audio-") as tmp:
            decoded = Path(tmp) / "decoded.wav"
            try:
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(path), str(decoded)],
                    check=True, capture_output=True,
                )
            except FileNotFoundError as exc:
                raise ValueError("此音频需要 FFmpeg 解码，请安装 FFmpeg 或先转为 WAV。") from exc
            except subprocess.CalledProcessError as exc:
                raise ValueError("无法解码音频，请检查文件是否完整。") from exc
            audio, sr = sf.read(decoded, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("音频为空或包含无效采样。")
    return audio.mean(axis=1), int(sr)


def _resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return audio
    common = math.gcd(sr, target)
    return resample_poly(audio, target // common, sr // common).astype("float32")


def reference_audio(source: Path, destination: Path, max_seconds: float = 12.0) -> dict:
    """Select one energetic window, preserving internal pauses and avoiding denoising/timbre edits."""
    audio, sr = load_audio(source)
    duration = len(audio) / sr
    if duration < 3:
        raise ValueError("参考音频至少需要 3 秒；建议使用 5–15 秒清晰、单人干声。")
    if np.max(np.abs(audio)) < 1e-5:
        raise ValueError("音频接近静音，无法建立声音参考。")
    window = min(len(audio), round(max_seconds * sr))
    start = 0
    if len(audio) > window:
        # One-second bins avoid scanning every sample with a long sliding window.
        step = sr
        starts = list(range(0, len(audio) - window + 1, step))
        if starts[-1] != len(audio) - window:
            starts.append(len(audio) - window)
        energy = np.concatenate(([0.0], np.cumsum(audio.astype("float64") ** 2)))
        start = max(starts, key=lambda i: energy[i + window] - energy[i])
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio[start:start + window], sr, subtype="PCM_16")
    return {
        "source_duration_seconds": round(duration, 3),
        "reference_start_seconds": round(start / sr, 3),
        "reference_end_seconds": round((start + window) / sr, 3),
        "selection_method": "energy_window" if duration > max_seconds else "whole_clip",
    }


def analyze_audio(path: Path) -> dict:
    """Estimate voiced F0 with librosa pYIN, not a guessed age/gender/timbre score."""
    import librosa

    audio, sr = load_audio(path)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype("float64") ** 2)))
    sampled = _resample(audio, sr, 16000)
    f0, voiced, probabilities = librosa.pyin(
        sampled, sr=16000, fmin=50, fmax=800, frame_length=1024,
        hop_length=256, fill_na=np.nan,
    )
    valid = voiced & np.isfinite(f0) & (probabilities >= 0.5)
    pitch = f0[valid]
    frames = librosa.feature.rms(y=sampled, frame_length=1024, hop_length=256)[0]
    threshold = max(float(frames.max()) * 0.03, 1e-4)
    result = {
        "duration_seconds": round(len(audio) / sr, 3), "sample_rate": sr,
        "peak_dbfs": round(20 * math.log10(max(peak, 1e-10)), 2),
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-10)), 2),
        "clipped_sample_ratio": round(float(np.mean(np.abs(audio) >= 0.999)), 5),
        "silence_ratio": round(float(np.mean(frames < threshold)), 3),
        "voiced_frame_ratio": round(float(np.mean(valid)), 3),
        "pitch_method": "librosa.pyin; voiced probability >= 0.5; 50–800 Hz",
        "pitch_median_hz": None, "pitch_p10_hz": None, "pitch_p90_hz": None,
        "pitch_range_semitones": None,
        "notes": ["统计描述这段录音，不代表完整或唯一的声音身份。", "未推断年龄、性别、气声、沙哑或音源授权。"],
    }
    if len(pitch) >= 5:
        p10, median, p90 = np.percentile(pitch, [10, 50, 90])
        result.update(
            pitch_median_hz=round(float(median), 1), pitch_p10_hz=round(float(p10), 1),
            pitch_p90_hz=round(float(p90), 1), pitch_range_semitones=round(12 * math.log2(p90 / p10), 2),
        )
    else:
        result["notes"].append("可靠有声帧不足，音高统计留空。")
    return result
