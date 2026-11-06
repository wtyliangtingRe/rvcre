"""Local, auditable preparation of VoxCPM audio/text manifests.

No model or recording is downloaded here.  Splitting is done at original-file
level, before assigning references, so ASR slices cannot leak across splits.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aif", ".aiff", ".wma"}


def _component(value: str) -> str:
    readable = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._-")[:48] or "voice"
    return f"{readable}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"


def _duration(path: Path) -> float:
    try:
        duration = float(sf.info(path).duration)
    except Exception as exc:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise ValueError(f"无法读取音频；此格式可能需要 ffmpeg/ffprobe：{exc}") from exc
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        if result.returncode:
            raise ValueError(f"ffprobe 无法读取音频：{result.stderr.strip()[:500]}")
        info = json.loads(result.stdout)
        raw = info.get("format", {}).get("duration")
        if raw in (None, "N/A"):
            raw = next((stream.get("duration") for stream in info.get("streams", []) if stream.get("duration") not in (None, "N/A")), None)
        duration = float(raw)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("音频时长为空或无效")
    return duration


def _read_audio(path: Path) -> np.ndarray:
    """Decode and resample; do not denoise, trim, normalize or change pitch."""
    try:
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError(f"soundfile 不支持此音频，且未找到 ffmpeg：{exc}") from exc
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
            capture_output=True, check=False,
        )
        if result.returncode:
            raise ValueError(f"ffmpeg 解码失败：{result.stderr.decode('utf-8', errors='replace')[:500]}")
        waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
        sample_rate = SAMPLE_RATE
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("音频为空或含有 NaN/Inf")
    if sample_rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        factor = math.gcd(int(sample_rate), SAMPLE_RATE)
        waveform = resample_poly(waveform, SAMPLE_RATE // factor, int(sample_rate) // factor)
    waveform = np.asarray(waveform, dtype=np.float32)
    if float(np.max(np.abs(waveform))) < 1e-7:
        raise ValueError("音频为静音；请检查输入文件")
    return waveform


def _transcript(path: Path) -> tuple[str | None, str | None]:
    candidates = [path.with_suffix(suffix) for suffix in (".txt", ".lab", ".TXT", ".LAB")]
    readings: list[tuple[str, str]] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        raw = candidate.read_text(encoding="utf-8-sig").strip()
        if not raw:
            continue
        # Plain .lab is supported; timestamped HTK labels require alignment
        # handling and must never become the literal training transcript.
        if candidate.suffix.lower() == ".lab" and re.search(r"(?m)^\s*\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s+\S", raw):
            raise ValueError(f"{candidate.name} 含时间戳；本工具的 .lab 输入须为纯转写文本")
        readings.append((" ".join(raw.split()), str(candidate)))
    if not readings:
        return None, None
    if len({text for text, _ in readings}) > 1:
        raise ValueError("同名 .txt/.lab 的内容不一致，请保留或统一正确转写")
    return readings[0]


def _load_asr(model: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("可选转写需要 faster-whisper；请安装 rvcre[asr] 并事先准备本地模型") from exc
    try:
        return WhisperModel(model, device="cpu", compute_type="int8", local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"无法打开本地 ASR 模型 {model!r}；不会自动下载。可将 asr_model 指向本地 CTranslate2 Whisper 目录：{exc}") from exc


def _asr_clips(segments, duration: float, minimum: float, maximum: float) -> tuple[list[tuple[float, float, str]], list[dict[str, Any]]]:
    """Combine timestamped segments without inventing word-to-time alignment."""
    clips: list[tuple[float, float, str]] = []
    issues: list[dict[str, Any]] = []
    pending: list[tuple[float, float, str]] = []
    previous_end = 0.0

    def flush() -> None:
        if not pending:
            return
        start, end = pending[0][0], pending[-1][1]
        if end - start >= minimum:
            clips.append((start, end, " ".join(item[2] for item in pending)))
        else:
            issues.append({"code": "asr_chunk_too_short", "start": start, "end": end, "message": "ASR 对齐片段短于最小时长，未补写或挪用相邻文本"})
        pending.clear()

    for segment in segments:
        get = segment.get if isinstance(segment, dict) else lambda key: getattr(segment, key)
        start, end = float(get("start")), float(get("end"))
        raw_text = get("text")
        text = " ".join(raw_text.split()) if isinstance(raw_text, str) else ""
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or end > duration + 0.1 or start < previous_end - 0.05 or not text:
            flush()
            issues.append({"code": "invalid_asr_segment", "message": "ASR 时间戳/顺序/文本无效，片段未采用"})
            continue
        start, end = max(start, previous_end), min(end, duration)
        previous_end = end
        if end - start > maximum:
            flush()
            issues.append({"code": "asr_segment_too_long", "start": start, "end": end, "message": "单个 ASR 片段过长；没有可靠的内部对齐，未等分文本"})
            continue
        if pending and end - pending[0][0] > maximum:
            flush()
        pending.append((start, end, text))
    flush()
    return clips, issues


def _assign_splits(records: list[dict[str, Any]], ratio: float, seed: int, warnings: list[str]) -> None:
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[record["speaker"]].add(record["source_id"])
    validation: set[tuple[str, str]] = set()
    for speaker, source_ids in sorted(groups.items()):
        ordered = sorted(source_ids)
        random.Random(f"rvcre:{seed}:{speaker}").shuffle(ordered)
        count = 0
        if ratio > 0 and len(ordered) > 1:
            count = max(1, min(len(ordered) - 1, round(len(ordered) * ratio)))
        elif ratio > 0:
            warnings.append(f"{speaker} 只有一个有效源文件：全部保留在训练集，无法建立独立验证集。")
        validation.update((speaker, source_id) for source_id in ordered[:count])
    for record in records:
        record["split"] = "validation" if (record["speaker"], record["source_id"]) in validation else "train"


def _assign_references(records: list[dict[str, Any]], seed: int, warnings: list[str]) -> None:
    buckets: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        buckets[(record["speaker"], record["split"])][record["source_id"]].append(record)
    for (speaker, split), groups in sorted(buckets.items()):
        ids = sorted(groups)
        for group_index, source_id in enumerate(ids):
            items = sorted(groups[source_id], key=lambda item: item["audio"])
            for index, record in enumerate(items):
                candidates = groups[ids[(group_index + 1) % len(ids)]] if len(ids) > 1 else [item for item in items if item is not record]
                if candidates:
                    reference = random.Random(f"ref:{seed}:{record['audio']}").choice(candidates)
                    record["ref_audio"] = reference["audio"]
                    record["ref_duration"] = reference["duration"]
                else:
                    record["ref_audio"] = None
                    record["ref_duration"] = None
                    warnings.append(f"{speaker}/{split} 只有一个片段：ref_audio=null，避免把目标音频本身当参考。")


def prepare_dataset(
    source: Path,
    output: Path,
    speaker: str | None = None,
    language: str = "zh",
    validation_ratio: float = 0.1,
    seed: int = 42,
    min_seconds: float = 3,
    max_seconds: float = 30,
    transcribe: bool = False,
    asr_model: str = "small",
) -> dict[str, Any]:
    """Prepare a new dataset directory and return its JSON-serializable summary.

    Accepted layouts: one speaker directly in ``source``; one first-level
    subdirectory per speaker; or any nested layout with an explicit ``speaker``
    assigning every source file to that speaker.  The output must be new/empty.

    Audio paths in JSONL are relative to the JSONL's directory, as supported by
    the pinned upstream VoxCPM data loader.  ASR is opt-in and strictly local.
    All automatically transcribed text is flagged for human review, but review
    is not an artificial authorization gate to training.
    """
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"音源目录不存在：{source}")
    if output == source or source.is_relative_to(output):
        raise ValueError("输出目录不能等于或包含音源目录")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio 须满足 0 <= ratio < 1")
    if not math.isfinite(min_seconds) or not math.isfinite(max_seconds) or min_seconds <= 0 or max_seconds < min_seconds:
        raise ValueError("时长参数须满足 0 < min_seconds <= max_seconds")
    if speaker is not None and not speaker.strip():
        raise ValueError("speaker 不得为空")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"输出目录必须为空，避免覆盖已准备的数据：{output}")
    files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES and not path.is_relative_to(output))
    if not files:
        raise ValueError(f"目录中没有支持的音频：{source}")
    direct = [path for path in files if path.parent == source]
    nested = [path for path in files if path.parent != source]
    if speaker is None and direct and nested:
        raise ValueError("音源同时包含根目录音频和子目录音频，无法可靠推断说话人；若确为同一人请显式指定 speaker，否则按说话人分目录")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-prepare-", dir=output.parent))
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    warnings: list[str] = []
    asr = None
    asr_error: str | None = None
    accepted_sources: set[str] = set()
    try:
        for path in files:
            relative = path.relative_to(source)
            speaker_name = speaker.strip() if speaker else (source.name if direct else relative.parts[0])
            source_id = hashlib.sha256(f"{speaker_name}\0{relative.as_posix()}".encode("utf-8")).hexdigest()[:24]
            context = {"source": relative.as_posix(), "speaker": speaker_name, "source_id": source_id}

            def note(code: str, message: str, **extra: Any) -> None:
                issues.append({**context, "code": code, "message": message, **extra})

            try:
                duration = _duration(path)
                text, transcript_path = _transcript(path)
            except Exception as exc:
                note("invalid_source", str(exc))
                continue
            if duration < min_seconds:
                note("too_short", "音频短于最小时长", duration=duration)
                continue
            need_asr = not text or duration > max_seconds
            if need_asr and not transcribe:
                if not text:
                    note("missing_transcript", "缺少非空 UTF-8 同名 .txt/.lab；补充文字或使用可选本地 ASR", duration=duration)
                if duration > max_seconds:
                    note("too_long_unaligned", "音频过长且无可靠分段对齐；未按时长平均分摊文本", duration=duration)
                continue
            try:
                waveform = _read_audio(path)
            except Exception as exc:
                note("decode_error", str(exc))
                continue
            duration = len(waveform) / SAMPLE_RATE
            if need_asr:
                if asr is None and asr_error is None:
                    try:
                        asr = _load_asr(asr_model)
                    except Exception as exc:
                        asr_error = str(exc)
                        warnings.append(asr_error)
                if asr_error:
                    note("asr_unavailable", asr_error)
                    continue
                try:
                    segments, _ = asr.transcribe(waveform, language=None if language in ("auto", "") else language, beam_size=5, vad_filter=True)
                    clips, segment_issues = _asr_clips(segments, duration, min_seconds, max_seconds)
                except Exception as exc:
                    note("asr_error", str(exc))
                    continue
                for issue in segment_issues:
                    issues.append({**context, **issue})
                if transcript_path:
                    note("long_transcript_not_force_aligned", "长音频使用 ASR 自带时间戳和转写；原始整篇文字保留在源目录，未伪造强制对齐")
                if not clips:
                    note("no_aligned_segments", "ASR 未产生符合时长条件的片段")
            else:
                clips = [(0.0, duration, text)]
            for index, (start, end, clip_text) in enumerate(clips):
                first, last = round(start * SAMPLE_RATE), min(len(waveform), round(end * SAMPLE_RATE))
                clip = waveform[first:last]
                clip_duration = len(clip) / SAMPLE_RATE
                if clip_duration < min_seconds or clip_duration > max_seconds:
                    note("decoded_duration_out_of_range", "解码后片段时长不在范围内，未截断对应文本", start=start, end=end)
                    continue
                audio_relative = Path("audio") / _component(speaker_name) / f"{source_id}_{index:04d}.wav"
                destination = staging / audio_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(destination, clip, SAMPLE_RATE, subtype="FLOAT")
                records.append({
                    **context, "audio": audio_relative.as_posix(), "text": clip_text,
                    "duration": clip_duration, "start": first / SAMPLE_RATE, "end": last / SAMPLE_RATE,
                    "transcript_origin": "asr" if need_asr else "sidecar",
                    "transcript_file": str(Path(transcript_path).relative_to(source)) if transcript_path else None,
                    "needs_text_review": need_asr,
                })
                accepted_sources.add(source_id)
        _assign_splits(records, validation_ratio, seed, warnings)
        _assign_references(records, seed, warnings)
        speakers = sorted({record["speaker"] for record in records})
        speaker_ids = {name: index for index, name in enumerate(speakers)}
        for split, name in (("train", "train.jsonl"), ("validation", "validation.jsonl")):
            with (staging / name).open("w", encoding="utf-8", newline="\n") as stream:
                for record in records:
                    if record["split"] == split:
                        item = {key: record[key] for key in ("audio", "text", "ref_audio", "duration", "ref_duration")}
                        item["dataset_id"] = speaker_ids[record["speaker"]]
                        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        with (staging / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        summary: dict[str, Any] = {
            "schema_version": 1, "source": str(source), "output": str(output),
            "train_manifest": str(output / "train.jsonl"), "validation_manifest": str(output / "validation.jsonl"),
            "metadata": str(output / "metadata.jsonl"), "report": str(output / "preparation_report.json"),
            "sample_rate": SAMPLE_RATE, "audio_path_base": "manifest_directory", "language": language,
            "seed": seed, "validation_ratio": validation_ratio, "min_seconds": min_seconds, "max_seconds": max_seconds,
            "sources_total": len(files), "accepted_sources": len(accepted_sources), "skipped_sources": len(files) - len(accepted_sources),
            "accepted_samples": len(records), "train_samples": sum(record["split"] == "train" for record in records),
            "validation_samples": sum(record["split"] == "validation" for record in records),
            "asr_review_samples": sum(record["needs_text_review"] for record in records),
            "speakers": {name: {"dataset_id": speaker_ids[name], "samples": sum(record["speaker"] == name for record in records)} for name in speakers},
            "warnings": list(dict.fromkeys(warnings)), "issues": issues,
        }
        if summary["asr_review_samples"]:
            summary["warnings"].append("ASR 文本和切分只是初稿：训练前请听读复核，特别是人名、语气词、口音、笑声及句首句尾。")
        if not records:
            summary["warnings"].append("没有可训练样本；JSONL 为空，请根据 issues 补充文字或修正音频后使用新输出目录重试。")
        (staging / "preparation_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if output.exists():
            output.rmdir()  # Only the explicitly checked empty directory.
        staging.rename(output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
