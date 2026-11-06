#!/usr/bin/env python3
"""Download only a selected backend's pinned model assets using huggingface_hub."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvcre.backends.runner import default_config_path, read_config  # noqa: E402
from rvcre.backends.specs import MODEL_ASSETS  # noqa: E402


def download(name: str, entry: dict, *, dry_run: bool = False) -> None:
    model_dir = Path(entry["model_dir"])
    total = sum(asset["size_bytes"] for asset in MODEL_ASSETS[name])
    print(f"{name}: selected model files total {total / 1e9:.2f} GB ({total / 2**30:.2f} GiB); excludes Python dependencies.")
    receipts = []
    for asset in MODEL_ASSETS[name]:
        destination = model_dir / asset["directory"]
        print(f"{asset['repo_id']} @ {asset['revision']} -> {destination}", flush=True)
        print("  " + ", ".join(asset["files"]), flush=True)
        if not dry_run:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=asset["repo_id"], revision=asset["revision"],
                allow_patterns=asset["files"], local_dir=str(destination),
            )
            missing = [name for name in asset["files"] if not (destination / name).is_file()]
            if missing:
                raise RuntimeError(f"Download incomplete for {asset['repo_id']}: {', '.join(missing)}")
        receipts.append({"repo_id": asset["repo_id"], "revision": asset["revision"], "path": str(destination)})
    if not dry_run:
        receipt = {"backend": name, "downloaded_at": datetime.now(timezone.utc).isoformat(), "models": receipts}
        (model_dir / "rvcre-models.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=MODEL_ASSETS, action="append", required=True)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--dry-run", action="store_true", help="List pinned downloads without downloading anything.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.config = args.config.resolve()
    configuration = read_config(args.config) if args.config.is_file() else {}
    for name in dict.fromkeys(args.backend):
        entry = configuration.get(name)
        if not entry:
            if args.dry_run:
                entry = {"model_dir": str(ROOT / "models" / ("VoxCPM2" if name == "voxcpm" else "SeedVC"))}
            else:
                parser.error(f"Configure {name} first: python scripts/setup_backends.py --backend {name}")
        try:
            if args.worker or args.dry_run:
                # Downloading is the explicit online step; workers doing inference stay offline.
                for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
                    os.environ.pop(variable, None)
                download(name, entry, dry_run=args.dry_run)
            else:
                subprocess.run(
                    [entry["python"], str(Path(__file__).resolve()), "--backend", name,
                     "--config", str(args.config), "--worker"], check=True,
                )
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            parser.exit(1, f"Model download failed: {exc}\n")
    if not args.dry_run:
        print("Pinned model assets are ready. Inference workers use local files only.")


if __name__ == "__main__":
    main()
