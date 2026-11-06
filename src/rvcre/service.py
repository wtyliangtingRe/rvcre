"""Application workflow shared by the CLI, HTTP API and Gradio UI."""

import hashlib
import shutil
import uuid
from pathlib import Path

from .audio import analyze_audio, reference_audio
from .jobs import JobManager
from .schema import Performance, VoiceSpec, design_instruction, performance_instruction
from .storage import item_path, read_json, timestamp, write_json


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
    def _text(text: str, max_length: int = 600) -> str:
        text = text.strip()
        if not text:
            raise ValueError("请输入要合成的文本。")
        if len(text) > max_length:
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
        return self._submit("voxcpm", {
            "task": "tts", "text": self._text(text), "voice_id": voice_id,
            "voice_revision": voice["revision"], "reference_path": voice["reference_path"],
            "reference_text": voice["transcript"], "instruction": performance_instruction(controls),
            "performance": controls.model_dump(), "mode": mode, "seed": self._seed(seed),
            "candidates": 1, "settings": {},
        })

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
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> dict:
        return self.jobs.cancel(job_id)

    def audio_path(self, job_id: str, index: int) -> Path:
        job = self.get_job(job_id)
        if job["status"] != "succeeded" or not 0 <= index < len(job["outputs"]):
            raise FileNotFoundError("此任务没有对应的音频输出。")
        path = Path(job["outputs"][index]["path"]).resolve()
        if not path.is_relative_to(self.workspace / "jobs" / job_id):
            raise ValueError("任务音频路径无效。")
        return path
