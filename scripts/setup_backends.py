#!/usr/bin/env python3
"""Create isolated Python 3.11 environments for explicitly selected backends.

Windows: py -3.11 scripts/setup_backends.py --backend voxcpm
Linux:   python3.11 scripts/setup_backends.py --backend voxcpm
Pass --device cpu for an intentional CPU installation. No models are downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvcre.backends.runner import default_config_path, read_config  # noqa: E402
from rvcre.backends.specs import MODEL_ASSETS, UPSTREAMS  # noqa: E402

TORCH_PROFILES = {
    "2.4.0": {"torchvision": "0.19.0", "index": "cu124", "torchcodec": None},
    "2.8.0": {"torchvision": "0.23.0", "index": "cu128", "torchcodec": "0.7.0"},
}


def python_command(explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    if sys.version_info[:2] == (3, 11):
        return [sys.executable]
    if os.name == "nt":
        return ["py", "-3.11"]
    return ["python3.11"]


def execute(command: list[str], *, dry_run: bool, cwd: Path | None = None) -> None:
    print(subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def seed_requirements(text: str) -> str:
    """Keep upstream pins while removing its conflicting Torch/nightly entries."""
    lines = [
        line for line in text.splitlines()
        if not re.match(r"^\s*(torch|torchvision|torchaudio)(?=[\s<>=!~\[]|$)", line)
    ]
    return "\n".join(lines) + "\nimageio-ffmpeg==0.6.0\n"


def setup_one(name: str, args, configuration: dict) -> dict:
    source = UPSTREAMS[name]
    existing = configuration.get(name, {})
    upstream = Path(existing.get("upstream_dir", ROOT / "third_party" / source["directory"])).resolve()
    model_dir = Path(existing.get("model_dir", ROOT / "models" / ("VoxCPM2" if name == "voxcpm" else "SeedVC"))).resolve()
    environment = ROOT / ".venvs" / name
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    version = args.torch_version or ("2.8.0" if name == "voxcpm" else "2.4.0")
    if name == "voxcpm" and version == "2.4.0":
        raise ValueError("VoxCPM2 requires torch>=2.5; select the 2.8.0 profile.")
    profile = TORCH_PROFILES[version]
    index = args.torch_index or ("cpu" if args.device == "cpu" else profile["index"])
    if args.device == "cuda" and index == "cpu":
        raise ValueError("--device cuda cannot use the CPU wheel index.")
    if args.device == "cpu" and index != "cpu":
        raise ValueError("--device cpu must use the CPU wheel index.")

    if not args.dry_run:
        upstream.parent.mkdir(parents=True, exist_ok=True)
        environment.parent.mkdir(parents=True, exist_ok=True)
        if upstream.exists():
            if not (upstream / ".git").exists():
                raise RuntimeError(f"{upstream} exists but is not a Git checkout.")
            changes = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True)
            if changes.strip():
                raise RuntimeError(f"{upstream} has local changes; keep them before rerunning setup.")
    if not upstream.exists():
        execute(["git", "clone", "--filter=blob:none", "--no-checkout", source["url"], str(upstream)], dry_run=args.dry_run)
    execute(["git", "-C", str(upstream), "fetch", "--depth", "1", "origin", source["revision"]], dry_run=args.dry_run)
    execute(["git", "-C", str(upstream), "checkout", "--detach", source["revision"]], dry_run=args.dry_run)
    if not executable.is_file():
        execute([*python_command(args.python), "-m", "venv", str(environment)], dry_run=args.dry_run)
    execute([str(executable), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], dry_run=args.dry_run)
    execute(
        [str(executable), "-m", "pip", "install", f"torch=={version}", f"torchaudio=={version}",
         f"torchvision=={profile['torchvision']}", "--index-url", f"https://download.pytorch.org/whl/{index}"],
        dry_run=args.dry_run,
    )
    constraints = environment / "rvcre-constraints.txt"
    pins = [f"torch=={version}", f"torchaudio=={version}", f"torchvision=={profile['torchvision']}", "numpy<2"]
    if name == "voxcpm":
        pins += [f"torchcodec=={profile['torchcodec']}", "transformers<5", "huggingface-hub<1"]
    if not args.dry_run:
        constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
    if name == "voxcpm":
        execute(
            [str(executable), "-m", "pip", "install", "-c", str(constraints), "-e", str(upstream), "tensorboardX"],
            dry_run=args.dry_run,
        )
    else:
        requirements = environment / "rvcre-seed-requirements.txt"
        if not args.dry_run:
            requirements.write_text(seed_requirements((upstream / "requirements.txt").read_text(encoding="utf-8")), encoding="utf-8")
        execute(
            [str(executable), "-m", "pip", "install", "-c", str(constraints), "-r", str(requirements)],
            dry_run=args.dry_run,
        )
    verify = (
        "import sys,torch,torchaudio; "
        "assert sys.version_info[:2] == (3,11), 'Backend requires Python 3.11'; "
        "print('PyTorch:',torch.__version__,'CUDA build:',torch.version.cuda,'CUDA available:',torch.cuda.is_available()); "
    )
    if args.device == "cuda":
        verify += "assert torch.version.cuda is not None, 'A CPU-only PyTorch wheel was installed'; "
    if name == "voxcpm":
        verify += "from voxcpm import VoxCPM; print('VoxCPM import OK; no weights loaded')"
    else:
        verify += (
            "from modules.v2.vc_wrapper import VoiceConversionWrapper; "
            "import imageio_ffmpeg; print('Seed-VC import OK; FFmpeg:',imageio_ffmpeg.get_ffmpeg_exe())"
        )
    execute([str(executable), "-m", "pip", "check"], dry_run=args.dry_run)
    execute([str(executable), "-c", verify], dry_run=args.dry_run, cwd=upstream)
    entry = dict(existing)
    entry.update({
        "python": str(executable), "upstream_dir": str(upstream), "model_dir": str(model_dir),
        "device": args.device, "upstream_revision": source["revision"],
        "model_revision": MODEL_ASSETS[name][0]["revision"],
        "asset_revisions": {asset["repo_id"]: asset["revision"] for asset in MODEL_ASSETS[name]},
    })
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=UPSTREAMS, action="append", required=True, help="Repeat to install both backends.")
    parser.add_argument("--python", help="Path to Python 3.11; Windows defaults to py -3.11.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--torch-version", choices=TORCH_PROFILES, help="Override the backend's default Torch profile.")
    parser.add_argument("--torch-index", choices=("cu124", "cu126", "cu128", "cpu"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without installing or writing configuration.")
    args = parser.parse_args()
    args.config = args.config.resolve()
    if not args.dry_run:
        if not shutil.which("git"):
            parser.error("Git is required to install the pinned upstream sources.")
        try:
            version = subprocess.check_output([*python_command(args.python), "-c", "import sys;print('%d.%d'%sys.version_info[:2])"], text=True).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            parser.error(f"Python 3.11 was not found; use --python with its executable path. {exc}")
        if version != "3.11":
            parser.error(f"Backend environments use Python 3.11; selected interpreter is {version}.")
    configuration = read_config(args.config) if args.config.exists() else {}
    try:
        for name in dict.fromkeys(args.backend):
            configuration[name] = setup_one(name, args, configuration)
            if not args.dry_run:
                args.config.parent.mkdir(parents=True, exist_ok=True)
                args.config.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Setup failed: {exc}\n")
    print("No model weights were downloaded. Next: python scripts/download_models.py --backend <name>")


if __name__ == "__main__":
    main()
