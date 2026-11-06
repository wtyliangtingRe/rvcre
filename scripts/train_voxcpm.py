#!/usr/bin/env python3
"""Configure and launch the upstream VoxCPM2 LoRA trainer using local assets.

This is a wrapper around OpenBMB's trainer, not another training implementation.
Use --config-only to inspect a plan without loading a neural model or training.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
UPSTREAM_COMMIT = "4b8dd4440d33396b3f429201f8af0529b4d2c601"
TEMPLATE = Path("conf/voxcpm_v2/voxcpm_finetune_lora.yaml")
TRAINER = Path("scripts/train_voxcpm_finetune.py")


def _manifest_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} 不是有效 JSON") from exc
            if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
                raise ValueError(f"{path}:{number} 缺少非空 text")
            for field in ("audio", "ref_audio"):
                value = item.get(field)
                if field == "ref_audio" and not value:
                    continue
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{path}:{number} 缺少有效 {field}")
                audio_path = Path(value).expanduser()
                if not audio_path.is_absolute():
                    audio_path = path.parent / audio_path
                if not audio_path.is_file():
                    raise FileNotFoundError(f"{path}:{number} {field} 文件不存在：{audio_path}")
            count += 1
    return count


def _backend_settings() -> dict[str, Any]:
    from rvcre.backends.runner import default_config_path, read_config

    path = default_config_path().expanduser().resolve()
    if not path.is_file():
        return {}
    return read_config(path).get("voxcpm", {})


def _training_keys(trainer: Path) -> set[str]:
    module = ast.parse(trainer.read_text(encoding="utf-8"), filename=str(trainer))
    function = next((node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "train"), None)
    if function is None:
        raise ValueError("上游脚本未包含预期的 train()；请使用 setup_backends.py 配置的固定版本")
    return {argument.arg for argument in (*function.args.args, *function.args.kwonlyargs)}


def create_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Validate manifests, use the real upstream schema, and write a run plan."""
    settings = _backend_settings()
    repo = Path(args.repo or settings.get("upstream_dir") or PROJECT_ROOT / "third_party/VoxCPM").expanduser().resolve()
    model = Path(args.model or settings.get("model_dir") or PROJECT_ROOT / "models/VoxCPM2").expanduser().resolve()
    environment_python = PROJECT_ROOT / ".venvs/voxcpm" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    executable = str(args.python or settings.get("python") or environment_python)
    python = Path(shutil.which(executable) or executable).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    trainer, template = repo / TRAINER, repo / TEMPLATE
    if not trainer.is_file() or not template.is_file():
        raise FileNotFoundError(f"缺少固定版本的 VoxCPM2 上游训练脚本/模板：{repo}。请先运行 scripts/setup_backends.py --backend voxcpm。")
    train_manifest = data / "train.jsonl" if data.is_dir() else data
    val_manifest = Path(args.validation).expanduser().resolve() if args.validation else train_manifest.parent / "validation.jsonl"
    train_count = _manifest_count(train_manifest)
    if train_count == 0:
        raise ValueError("训练 JSONL 为空；请先检查 preparation_report.json")
    if args.validation and not val_manifest.is_file():
        raise FileNotFoundError(f"指定的验证清单不存在：{val_manifest}")
    val_count = _manifest_count(val_manifest) if val_manifest.is_file() else 0
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("lora"), dict):
        raise ValueError("上游模板不是预期的 VoxCPM2 LoRA 配置")
    for name in ("steps", "batch_size", "grad_accum_steps", "lora_rank", "lora_alpha", "learning_rate"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} 必须大于 0")
    if args.num_workers < 0 or args.preprocessing_workers < 1:
        raise ValueError("num_workers 必须 >= 0，preprocessing_workers 必须 >= 1")
    config.update({
        "pretrained_path": str(model), "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest) if val_count else "",
        "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps,
        "num_workers": args.num_workers, "preprocessing_num_workers": args.preprocessing_workers,
        "num_iters": args.steps, "max_steps": args.steps,
        "warmup_steps": min(int(config.get("warmup_steps", 100)), max(0, args.steps - 1)),
        "save_interval": min(int(config.get("save_interval", 500)), args.steps),
        "valid_interval": min(int(config.get("valid_interval", 500)), args.steps),
        "learning_rate": args.learning_rate, "save_path": str(output / "checkpoints"),
        "tensorboard": str(output / "tensorboard"), "distribute": False,
    })
    config["lora"].update({"r": args.lora_rank, "alpha": args.lora_alpha})
    unsupported = set(config) - _training_keys(trainer)
    if unsupported:
        raise ValueError(f"配置与实际上游 train() 不匹配，未知字段：{', '.join(sorted(unsupported))}")
    warnings: list[str] = []
    if not val_count:
        warnings.append("没有独立验证样本；此次训练不会提供可用于判断泛化的验证结果。")
    model_config = model / "config.json"
    if model_config.is_file():
        model_info = json.loads(model_config.read_text(encoding="utf-8"))
        if model_info.get("architecture", "").lower() != "voxcpm2":
            raise ValueError(f"{model_config} 不是 VoxCPM2；不能套用 v2 LoRA 配置")
        encoder_rate = model_info.get("audio_vae_config", {}).get("sample_rate")
        if encoder_rate is not None and int(config.get("sample_rate", 0)) != int(encoder_rate):
            raise ValueError("训练采样率与本地 AudioVAE encoder 配置不匹配")
    elif args.config_only:
        warnings.append(f"本地模型尚未准备，当前只生成配置；正式训练前需要 {model_config} 和完整权重。")
    else:
        raise FileNotFoundError(f"本地模型尚未准备：{model_config}。此脚本不会下载权重；请先运行 scripts/download_models.py --backend voxcpm。")
    if not args.config_only and not python.is_file():
        raise FileNotFoundError(f"VoxCPM 独立环境 Python 不存在：{python}；请先 setup_backends 或显式 --python。")
    if not args.config_only and not any(model.glob("*.safetensors")) and not any(model.glob("*.bin")):
        raise FileNotFoundError(f"模型目录中没有模型权重：{model}；请先完成本地模型下载。")
    report_path = train_manifest.parent / "preparation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("asr_review_samples", 0):
            warnings.append(f"数据含 {report['asr_review_samples']} 条 ASR 初稿；请复核，不会为此添加审批步骤。")
    if (output / "checkpoints/latest").exists():
        warnings.append("检测到同一输出目录的 latest 检查点；上游训练器会尝试续训。全新实验请使用新输出目录。")
    actual_commit = None
    try:
        git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if git.returncode == 0:
            actual_commit = git.stdout.strip()
    except FileNotFoundError:
        pass
    if actual_commit and actual_commit != UPSTREAM_COMMIT:
        warnings.append(f"上游版本不同于已核实版本 {UPSTREAM_COMMIT}；配置字段已做静态检查，运行兼容性需重新验证。")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "finetune_lora.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    command = [str(python), "-u", str(trainer), "--config_path", str(config_path)]
    plan: dict[str, Any] = {
        "backend": "voxcpm", "architecture": "voxcpm2", "mode": "lora",
        "upstream_repo": str(repo), "verified_upstream_commit": UPSTREAM_COMMIT, "actual_upstream_commit": actual_commit,
        "template": str(template), "config": str(config_path), "model": str(model),
        "train_manifest": str(train_manifest), "validation_manifest": str(val_manifest) if val_count else None,
        "train_samples": train_count, "validation_samples": val_count,
        "command": command, "cwd": str(repo), "config_only": args.config_only,
        "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
        "verification": "config_and_local_paths_only; neural_training_not_run",
        "warnings": warnings,
    }
    (output / "training_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="用本地模型和 JSONL 启动官方 VoxCPM2 LoRA 训练；--config-only 只生成可复核配置")
    command.add_argument("--data", required=True, help="prepare_dataset 输出目录或 train.jsonl")
    command.add_argument("--output", required=True, help="本次训练输出目录，例如 runs/voice-a-lora")
    command.add_argument("--repo", help="上游 VoxCPM checkout；默认 third_party/VoxCPM")
    command.add_argument("--model", help="本地 VoxCPM2 权重目录；默认 models/VoxCPM2")
    command.add_argument("--python", help="VoxCPM 独立环境的 Python 路径")
    command.add_argument("--validation", help="可选验证 JSONL；默认同目录 validation.jsonl")
    command.add_argument("--steps", type=int, default=1000)
    command.add_argument("--batch-size", type=int, default=2)
    command.add_argument("--grad-accum-steps", type=int, default=8)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--lora-rank", type=int, default=32)
    command.add_argument("--lora-alpha", type=int, default=32)
    command.add_argument("--num-workers", type=int, default=0, help="默认0，便于Windows与小数据集")
    command.add_argument("--preprocessing-workers", type=int, default=1)
    command.add_argument("--config-only", action="store_true", help="生成配置和调用计划，绝不载入权重或启动训练")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        plan = create_plan(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"训练准备失败：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.config_only:
        return 0
    environment = {**os.environ, **plan["environment"]}
    try:
        return subprocess.run(plan["command"], cwd=plan["cwd"], env=environment, check=False).returncode
    except OSError as exc:
        print(f"无法启动上游训练器：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
