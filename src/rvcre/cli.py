import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .service import VoiceService


def _common(parser):
    parser.add_argument("--workspace", type=Path, default=Path("data"), help="声音与任务目录（默认 data）")
    parser.add_argument("--config", type=Path, default=None, help="后端配置（默认 config/backends.json）")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="rvcre", description="RVCre · 声构，本地声音设计与生成")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="检查应用与模型配置，不下载权重")
    _common(doctor)
    for command, port in (("ui", 7860), ("serve", 8000)):
        action = commands.add_parser(command, help="启动界面" if command == "ui" else "启动本地 HTTP API")
        _common(action)
        action.add_argument("--host", default="127.0.0.1")
        action.add_argument("--port", type=int, default=port)
        if command == "ui":
            action.add_argument("--no-browser", action="store_true")
    voices = commands.add_parser("voices", help="声音档案")
    actions = voices.add_subparsers(dest="action", required=True)
    for action in ("list", "show", "import", "save-candidate"):
        sub = actions.add_parser(action)
        _common(sub)
        if action == "show":
            sub.add_argument("voice_id")
        elif action == "import":
            sub.add_argument("audio", type=Path)
            sub.add_argument("--name", required=True)
            text = sub.add_mutually_exclusive_group()
            text.add_argument("--transcript", default="")
            text.add_argument("--transcript-file", type=Path)
            sub.add_argument("--notes", default="")
            sub.add_argument("--source-note", default="")
        elif action == "save-candidate":
            sub.add_argument("job_id")
            sub.add_argument("--index", type=int, default=0, help="候选下标，从 0 开始")
            sub.add_argument("--name", required=True)
    design = commands.add_parser("design", help="生成声音候选，等待任务完成")
    _common(design)
    design.add_argument("--text", default="你好，欢迎来到声构。我们可以一起设计一个声音，让它讲述新的故事。")
    design.add_argument("--spec", type=Path, help="声音参数 JSON 文件")
    design.add_argument("--candidates", type=int, default=3)
    design.add_argument("--seed", type=int, default=42)
    design.add_argument("--parent-voice", default=None)
    tts = commands.add_parser("tts", help="文字合成，等待任务完成")
    _common(tts)
    tts.add_argument("--voice", required=True)
    content = tts.add_mutually_exclusive_group(required=True)
    content.add_argument("--text")
    content.add_argument("--text-file", type=Path)
    tts.add_argument("--mode", choices=["controllable", "hifi"], default="controllable")
    tts.add_argument("--emotion", default="neutral", choices=["neutral", "happy", "sad", "angry", "calm", "excited"])
    tts.add_argument("--speed", type=float, default=1.0, help="语速倾向：模型指令控制，不是精确倍率")
    tts.add_argument("--instruction", default="")
    tts.add_argument("--seed", type=int, default=42)
    vc = commands.add_parser("vc", help="音色转换，等待任务完成")
    _common(vc)
    vc.add_argument("--voice", required=True)
    vc.add_argument("--source", type=Path, required=True)
    vc.add_argument("--steps", type=int, default=30)
    vc.add_argument("--similarity", type=float, default=0.7)
    vc.add_argument("--clarity", type=float, default=0.7)
    jobs = commands.add_parser("jobs", help="查看已保存的任务")
    actions = jobs.add_subparsers(dest="action", required=True)
    for action in ("list", "show", "cancel"):
        sub = actions.add_parser(action)
        _common(sub)
        if action != "list":
            sub.add_argument("job_id")
    dataset = commands.add_parser("dataset", help="整理本地训练音源")
    prepare = dataset.add_subparsers(dest="action", required=True).add_parser("prepare")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--speaker", default=None)
    prepare.add_argument("--language", default="zh")
    prepare.add_argument("--validation-ratio", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--min-seconds", type=float, default=3.0)
    prepare.add_argument("--max-seconds", type=float, default=30.0)
    prepare.add_argument("--transcribe", action="store_true")
    prepare.add_argument("--asr-model", default="small")
    return root


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _wait(service: VoiceService, job: dict):
    print(f"任务 {job['id']} 已提交。", file=sys.stderr)
    last = None
    while True:
        job = service.get_job(job["id"])
        if job["status"] != last:
            print(f"状态：{job['status']}", file=sys.stderr)
            last = job["status"]
        if job["status"] in ("succeeded", "failed", "cancelled"):
            _print(job)
            return 0 if job["status"] == "succeeded" else 1
        time.sleep(0.25)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "ui":
            try:
                from .ui import launch_ui
            except ModuleNotFoundError as exc:
                if exc.name == "gradio":
                    raise ValueError('界面依赖未安装，请运行 python -m pip install -e ".[ui]"。') from exc
                raise
            launch_ui(args.workspace, args.config, args.host, args.port, not args.no_browser)
            return 0
        if args.command == "serve":
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(args.workspace, args.config), host=args.host, port=args.port)
            return 0
        if args.command == "dataset":
            from .dataset import prepare_dataset
            _print(prepare_dataset(
                args.source, args.output, speaker=args.speaker, language=args.language,
                validation_ratio=args.validation_ratio, seed=args.seed, min_seconds=args.min_seconds,
                max_seconds=args.max_seconds, transcribe=args.transcribe, asr_model=args.asr_model,
            ))
            return 0
        service = VoiceService(args.workspace, args.config)
        try:
            if args.command == "doctor":
                _print({"version": __version__, "python": sys.version.split()[0],
                        "workspace": str(service.workspace), "ffmpeg": shutil.which("ffmpeg"),
                        "backends": service.backend_status()})
            elif args.command == "voices":
                if args.action == "list":
                    _print(service.list_voices())
                elif args.action == "show":
                    _print(service.get_voice(args.voice_id))
                elif args.action == "import":
                    transcript = args.transcript_file.read_text(encoding="utf-8") if args.transcript_file else args.transcript
                    _print(service.import_voice(args.name, str(args.audio), transcript, args.notes,
                                                source_note=args.source_note))
                else:
                    _print(service.save_candidate(args.job_id, args.index, args.name))
            elif args.command == "jobs":
                if args.action == "list":
                    _print(service.jobs.list())
                elif args.action == "show":
                    _print(service.get_job(args.job_id))
                else:
                    _print(service.cancel_job(args.job_id))
            elif args.command == "design":
                spec = json.loads(args.spec.read_text(encoding="utf-8")) if args.spec else {}
                return _wait(service, service.submit_design(args.text, spec, args.candidates, args.seed, args.parent_voice))
            elif args.command == "tts":
                text = args.text_file.read_text(encoding="utf-8") if args.text_file else args.text
                return _wait(service, service.submit_tts(args.voice, text, {
                    "speed": args.speed, "emotion": args.emotion, "instruction": args.instruction,
                }, args.mode, args.seed))
            elif args.command == "vc":
                return _wait(service, service.submit_vc(args.voice, str(args.source), {
                    "diffusion_steps": args.steps, "similarity_cfg_rate": args.similarity,
                    "intelligibility_cfg_rate": args.clarity,
                }))
        finally:
            service.close()
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"RVCre：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
