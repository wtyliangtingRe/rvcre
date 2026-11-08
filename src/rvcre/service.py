"""Application workflow shared by the CLI, HTTP API and Gradio UI."""

import hashlib
import shutil
import uuid
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .audio import analyze_audio, merge_audio_files, reference_audio
from .jobs import JobManager
from .schema import Performance, VoiceSpec, design_instruction, performance_instruction
from .storage import item_path, read_json, timestamp, write_json
from .text import TTS_SEGMENT_CHARS, split_tts_text


class VoiceService:
    def __init__(self, workspace: Path, config_path: Path | None = None, *, runner=None):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.voices_dir = self.workspace / "voices"
        self.voices_dir.mkdir(exist_ok=True)
        self.jobs = JobManager(self.workspace / "jobs")
        if runner is None:
            from .backends import BackendRunner
            runner = BackendRunner(config_path)
        self.runner = runner

    def close(self) -> None:
        self.jobs.close()

    def backend_status(self) -> list[dict]:
        return self.runner.status()

    def get_voice(self, voice_id: str) -> dict:
        path = item_path(self.voices_dir, voice_id, "profile.json")
        profile = read_json(path)
        for key in ("reference_path", "original_path"):
            profile[key] = str((path.parent / profile[key]).resolve())
        return profile

    def list_voices(self) -> list[dict]:
        profiles = [self.get_voice(p.parent.name) for p in self.voices_dir.glob("*/profile.json")]
        return sorted(profiles, key=lambda p: p["created_at"], reverse=True)

    def import_voice(self, name: str, audio_path: str, transcript: str = "", notes: str = "",
                     source_kind: str = "reference", source_note: str = "", *,
                     design_spec: dict | None = None, parent_voice_id: str | None = None,
                     generation: dict | None = None) -> dict:
        if not name.strip():
            raise ValueError("请为声音填写名称。")
        source = Path(audio_path).resolve()
        parent = self.get_voice(parent_voice_id) if parent_voice_id else None
        identifier = uuid.uuid4().hex[:12]
        directory = self.voices_dir / identifier
        directory.mkdir()
        try:
            selection = reference_audio(source, directory / "reference.wav")
            analysis = analyze_audio(directory / "reference.wav")
            original = directory / ("original" + source.suffix.lower())
            shutil.copyfile(source, original)
            with original.open("rb") as stream:
                source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            cropped = selection["selection_method"] != "whole_clip"
            if cropped and transcript.strip():
                analysis["notes"].append("参考片段已截取，未沿用整段转写。Hi-Fi 请重新导入已剪好且转写对应的片段。")
            initial_spec = VoiceSpec.model_validate(design_spec or {})
            spec_origin = "requested" if design_spec else "defaults_not_measured"
            if design_spec is None and analysis["pitch_median_hz"] is not None:
                median = analysis["pitch_median_hz"]
                initial_spec.pitch = "low" if median < 150 else "high" if median > 240 else "medium"
                spec_origin = "pitch_hint_from_measurement"
                analysis["notes"].append("设计初值仅按本应用的音高区间给出粗略提示；其余音色旋钮未从录音反演。")
            profile = {
                "schema_version": 1, "id": identifier, "name": name.strip(),
                "revision": parent["revision"] + 1 if parent else 1,
                "parent_id": parent_voice_id, "created_at": timestamp(),
                "reference_path": "reference.wav", "original_path": original.name,
                "transcript": transcript.strip() if not cropped else "",
                "source_transcript": transcript.strip(), "notes": notes.strip(),
                "source": {"kind": source_kind, "note": source_note.strip(), "sha256": source_hash},
                "selection": selection, "analysis": analysis,
                "design_spec": initial_spec.model_dump(), "design_spec_origin": spec_origin,
                "generation": generation or {},
            }
            write_json(directory / "profile.json", profile)
        except Exception:
            shutil.rmtree(directory)
            raise
        return self.get_voice(identifier)

    @staticmethod
    def _text(text: str, max_length: int | None = 600) -> str:
        text = text.strip()
        if not text:
            raise ValueError("请输入要合成的文本。")
        if max_length is not None and len(text) > max_length:
            raise ValueError(f"当前单次任务最多 {max_length} 个字符，请将长篇文本按段提交。")
        return text

    @staticmethod
    def _seed(seed: int) -> int:
        seed = int(seed)
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("随机种子应在 0 到 4294967295 之间。")
        return seed

    def _submit(self, backend: str, request: dict) -> dict:
        def generate(directory, event):
            result = self.runner.run(backend, request, directory, cancel_event=event)
            if not event.is_set():
                for output in result["outputs"]:
                    path = Path(output["path"]).resolve()
                    if not path.is_file() or not path.is_relative_to(directory.resolve()):
                        raise RuntimeError("后端未生成有效的任务音频文件。")
                    output["path"] = str(path)
            return result
        return self.jobs.submit(request["task"], request, generate)

    @staticmethod
    def _checked_outputs(outputs: list[dict], directory: Path) -> list[dict]:
        checked = []
        for index, output in enumerate(outputs, 1):
            path = Path(output["path"]).resolve()
            if not path.is_file() or not path.is_relative_to(directory.resolve()):
                raise RuntimeError("后端未生成有效的任务音频文件。")
            checked.append({**output, "path": str(path), "kind": "segment", "segment_index": index})
        return checked

    @staticmethod
    def _read_progress(directory: Path) -> dict:
        try:
            progress = read_json(directory / "progress.json")
            return progress if isinstance(progress, dict) else {}
        except (OSError, ValueError):
            return {}

    def _submit_long_tts(self, request: dict) -> dict:
        """Run one model process for the whole script, then assemble its clips."""
        def generate(directory, event):
            segments = request["segments"]
            total = len(segments)
            (directory / "full-text.txt").write_text(request["text"], encoding="utf-8")
            for index, text in enumerate(segments, 1):
                (directory / f"text-{index:03d}.txt").write_text(text, encoding="utf-8")
            result = self.runner.run("voxcpm", request, directory, cancel_event=event)
            if event.is_set():
                return result
            outputs = self._checked_outputs(result["outputs"], directory)
            if len(outputs) != total:
                raise RuntimeError(f"分段生成不完整：应有 {total} 段，实际返回 {len(outputs)} 段。")
            progress = {
                "total": total, "completed": total, "current": None, "phase": "merging",
                "outputs": outputs, "metadata": result.get("metadata", {}),
            }
            write_json(directory / "progress.json", progress)
            merged = directory / "merged.wav"
            audio_info = merge_audio_files([Path(item["path"]) for item in outputs], merged,
                                          gap_seconds=request["gap_seconds"], cancel_event=event)
            # Text and audio share an index; originals are copied byte-for-byte into
            # the archive so individual paragraphs can be regenerated or edited.
            manifest = {
                "schema_version": 1, "voice_id": request["voice_id"],
                "voice_revision": request["voice_revision"], "mode": request["mode"],
                "performance": request["performance"], "requested_seed": request["seed"],
                "segment_count": total, "segment_limit": TTS_SEGMENT_CHARS,
                "gap_seconds": request["gap_seconds"], "merged_audio": audio_info,
                "segments": [{"index": index, "audio": f"audio-{index:03d}.wav",
                              "text": f"text-{index:03d}.txt", "characters": len(segments[index - 1]),
                              "seed": item.get("seed")}
                             for index, item in enumerate(outputs, 1)],
            }
            write_json(directory / "manifest.json", manifest)
            bundle = directory / "segments.zip"
            temporary = directory / "segments.tmp.zip"
            try:
                with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
                    for filename in ("full-text.txt", "manifest.json"):
                        archive.write(directory / filename, filename)
                    for index, item in enumerate(outputs, 1):
                        if event.is_set():
                            raise InterruptedError("分段打包已取消，已生成的分段保留。")
                        archive.write(item["path"], f"audio-{index:03d}.wav")
                        archive.write(directory / f"text-{index:03d}.txt", f"text-{index:03d}.txt")
                if event.is_set():
                    raise InterruptedError("分段打包已取消，已生成的分段保留。")
                temporary.replace(bundle)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            write_json(directory / "progress.json", {**progress, "phase": "complete"})
            return {
                "outputs": [{"path": str(merged), "kind": "merged", **audio_info}, *outputs],
                "metadata": {**result.get("metadata", {}), "segment_count": total,
                             "gap_seconds": request["gap_seconds"], "bundle_path": str(bundle)},
            }

        job = self.jobs.submit("tts", request, generate)
        return self.get_job(job["id"])

    def submit_design(self, text: str, spec: dict, candidates: int = 3, seed: int = 42,
                      parent_voice_id: str | None = None) -> dict:
        spec = VoiceSpec.model_validate(spec)
        if not 1 <= candidates <= 6:
            raise ValueError("每次请生成 1–6 个声音候选。")
        if parent_voice_id:
            self.get_voice(parent_voice_id)
        return self._submit("voxcpm", {
            "task": "design", "text": self._text(text, 240), "design_spec": spec.model_dump(),
            "instruction": design_instruction(spec), "seed": self._seed(seed),
            "candidates": candidates, "parent_voice_id": parent_voice_id,
            "mode": "controllable", "settings": {},
        })

    def save_candidate(self, job_id: str, index: int, name: str) -> dict:
        job = self.get_job(job_id)
        if job["task"] != "design" or job["status"] != "succeeded":
            raise ValueError("只有成功完成的设计任务可以保存候选。")
        if not 0 <= index < len(job["outputs"]):
            raise ValueError("候选编号超出范围。")
        request = read_json(item_path(self.workspace / "jobs", job_id, "request.json"))
        output = job["outputs"][index]
        return self.import_voice(
            name, output["path"], transcript=request["text"], source_kind="designed",
            source_note=f"RVCre 设计任务 {job_id}，候选 {index + 1}",
            design_spec=request["design_spec"], parent_voice_id=request.get("parent_voice_id"),
            generation={"job_id": job_id, "requested_seed": request["seed"],
                        "used_seed": output.get("seed"), "backend": "voxcpm", **job["metadata"]},
        )

    def submit_tts(self, voice_id: str, text: str, performance: dict | None = None,
                   mode: str = "controllable", seed: int = 42) -> dict:
        voice = self.get_voice(voice_id)
        controls = Performance.model_validate(performance or {})
        if mode not in ("controllable", "hifi"):
            raise ValueError("TTS 模式应为 controllable 或 hifi。")
        if mode == "hifi":
            if not voice["transcript"]:
                raise ValueError("Hi-Fi 模式需要与保存的参考片段准确对应的转写。")
            if controls != Performance():
                raise ValueError("Hi-Fi 模式不支持额外表达控制，请恢复默认设置或使用可控模式。")
        text = self._text(text, max_length=None)
        segments = split_tts_text(text)
        request = {
            "task": "tts", "text": text, "voice_id": voice_id,
            "voice_revision": voice["revision"], "reference_path": voice["reference_path"],
            "reference_text": voice["transcript"], "instruction": performance_instruction(controls),
            "performance": controls.model_dump(), "mode": mode, "seed": self._seed(seed),
            "candidates": 1, "settings": {},
        }
        if len(segments) > 1:
            return self._submit_long_tts({**request, "segments": segments, "gap_seconds": 0.2})
        return self._submit("voxcpm", {**request, "text": segments[0]})

    def submit_vc(self, voice_id: str, source_path: str, settings: dict | None = None) -> dict:
        voice = self.get_voice(voice_id)
        settings = settings or {}
        allowed = {"diffusion_steps", "similarity_cfg_rate", "intelligibility_cfg_rate", "length_adjust"}
        if set(settings) - allowed:
            raise ValueError("不支持的 VC 参数：" + ", ".join(sorted(set(settings) - allowed)))
        if settings.get("length_adjust", 1) != 1:
            raise ValueError("首版纯音色转换不支持长度调整；length_adjust 必须为 1。")
        if not 1 <= int(settings.get("diffusion_steps", 30)) <= 100:
            raise ValueError("扩散步数应在 1–100 之间。")
        for field in ("similarity_cfg_rate", "intelligibility_cfg_rate"):
            if not 0 <= float(settings.get(field, 0.7)) <= 1:
                raise ValueError(f"{field} 应在 0–1 之间。")
        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"源音频不存在：{source}")
        staged = self.workspace / "uploads" / (uuid.uuid4().hex + source.suffix.lower())
        staged.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, staged)
        return self._submit("seedvc", {
            "task": "vc", "voice_id": voice_id, "voice_revision": voice["revision"],
            "reference_path": voice["reference_path"], "source_path": str(staged),
            "settings": {"length_adjust": 1.0, **settings},
        })

    def get_job(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        job["files"] = []
        if job["task"] != "tts":
            return job
        directory = self.workspace / "jobs" / job_id
        request = read_json(directory / "request.json")
        segments = request.get("segments")
        if not segments:
            return job
        total = len(segments)
        progress = self._read_progress(directory)
        completed = []
        # Each progress snapshot is atomic. A killed worker can therefore expose
        # its last completed clips without advertising an unfinished WAV.
        for output in (progress.get("outputs") or [])[:total]:
            try:
                completed.extend(self._checked_outputs([output], directory))
            except (OSError, KeyError, TypeError, RuntimeError):
                break
        for index, output in enumerate(completed, 1):
            output["segment_index"] = index
        phase = progress.get("phase", "loading")
        if job["status"] == "succeeded":
            phase = "complete"
            bundle = directory / "segments.zip"
            if bundle.is_file() and bundle.resolve().is_relative_to(directory.resolve()):
                job["files"] = [str(bundle)]
        else:
            job["outputs"] = completed
            for index, output in enumerate(completed, 1):
                job["files"].append(output["path"])
                text_path = directory / f"text-{index:03d}.txt"
                if text_path.is_file() and text_path.resolve().is_relative_to(directory.resolve()):
                    job["files"].append(str(text_path))
        job["progress"] = {
            "total": total, "completed": total if job["status"] == "succeeded" else len(completed),
            "current": progress.get("current"), "phase": phase,
        }
        return job

    def cancel_job(self, job_id: str) -> dict:
        self.jobs.cancel(job_id)
        return self.get_job(job_id)

    def audio_path(self, job_id: str, index: int) -> Path:
        job = self.get_job(job_id)
        if not 0 <= index < len(job["outputs"]):
            raise FileNotFoundError("此任务没有对应的音频输出。")
        path = Path(job["outputs"][index]["path"]).resolve()
        if not path.is_relative_to(self.workspace / "jobs" / job_id):
            raise ValueError("任务音频路径无效。")
        return path

    def file_path(self, job_id: str, index: int) -> Path:
        files = self.get_job(job_id)["files"]
        if not 0 <= index < len(files):
            raise FileNotFoundError("此任务没有对应的下载文件。")
        path = Path(files[index]).resolve()
        if not path.is_file() or not path.is_relative_to(self.workspace / "jobs" / job_id):
            raise FileNotFoundError("任务文件已被移动或删除。")
        return path
