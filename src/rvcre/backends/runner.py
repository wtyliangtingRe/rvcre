"""Small subprocess runner; the web application never imports model libraries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading

from .specs import REQUIRED_MODEL_FILES, REQUIRED_MODULES, UPSTREAMS

WORKERS = {"voxcpm": "rvcre.workers.voxcpm", "seedvc": "rvcre.workers.seedvc"}
ROOT = Path(__file__).resolve().parents[3]


class BackendCancelled(RuntimeError):
    """A running backend was stopped by the user."""


def default_config_path() -> Path:
    return Path(os.environ.get("RVCRE_BACKEND_CONFIG", ROOT / "config" / "backends.json"))


def read_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Backend configuration must be a JSON object.")
    for name, entry in data.items():
        if name not in WORKERS:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Backend configuration for {name} must be an object.")
        for key in ("upstream_dir", "model_dir", "lora_weights_path"):
            if entry.get(key):
                value = Path(entry[key]).expanduser()
                entry[key] = str((path.parent / value).resolve() if not value.is_absolute() else value)
        executable = entry.get("python", "")
        if executable:
            found = shutil.which(executable)
            if found:
                entry["python"] = found
            else:
                value = Path(executable).expanduser()
                entry["python"] = str((path.parent / value).resolve() if not value.is_absolute() else value)
    return data


def _probe_modules(python: str, modules: list[str]) -> list[str]:
    # find_spec checks installation without importing torch or loading any model.
    code = (
        "import importlib.util,json,sys; "
        "print(json.dumps([m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]))"
    )
    result = subprocess.run(
        [python, "-c", code, *modules], capture_output=True, text=True, timeout=15,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Python dependency check failed.")
    return json.loads(result.stdout.strip())


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Include a possible FFmpeg child used by upstream Seed-VC's streaming API.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)


class BackendRunner:
    def __init__(self, config_path: Path | None = None):
        self.config_path = (Path(config_path) if config_path else default_config_path()).resolve()

    def _configuration(self) -> dict:
        if not self.config_path.is_file():
            return {}
        return read_config(self.config_path)

    @staticmethod
    def _check(name: str, entry: dict | None) -> dict:
        details = []
        if not entry:
            return {
                "name": name, "ready": False, "configured": False,
                "details": [f"Backend is not configured. Run scripts/setup_backends.py --backend {name}."],
            }
        python = entry.get("python", "")
        if not python or not Path(python).is_file():
            details.append(f"Python executable is missing: {python or '(not configured)'}")
        else:
            try:
                missing = _probe_modules(python, REQUIRED_MODULES[name])
                if missing:
                    details.append("Missing Python dependencies: " + ", ".join(missing))
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
                details.append(f"Cannot check Python dependencies: {exc}")
        upstream = Path(entry.get("upstream_dir", ""))
        marker = "src/voxcpm/core.py" if name == "voxcpm" else "configs/v2/vc_wrapper.yaml"
        if not entry.get("upstream_dir") or not (upstream / marker).is_file():
            details.append(f"Upstream checkout is missing: {upstream / marker}")
        model_dir = Path(entry.get("model_dir", ""))
        missing_files = [f for f in REQUIRED_MODEL_FILES[name] if not (model_dir / f).is_file()]
        if not entry.get("model_dir") or missing_files:
            details.append("Missing model files: " + ", ".join(missing_files))
            details.append(f"Run scripts/download_models.py --backend {name}.")
        if entry.get("lora_weights_path") and not Path(entry["lora_weights_path"]).exists():
            details.append("Configured LoRA checkpoint does not exist.")
        return {
            "name": name, "ready": not details, "configured": True,
            "details": details or ["Paths and packages are present; model inference has not been run by this check."],
            "device": entry.get("device", "cuda"),
            "upstream_revision": entry.get("upstream_revision", UPSTREAMS[name]["revision"]),
            "model_dir": str(model_dir),
        }

    def status(self) -> list[dict]:
        try:
            configuration = self._configuration()
        except (OSError, ValueError) as exc:
            return [{"name": name, "ready": False, "details": [str(exc)]} for name in WORKERS]
        return [self._check(name, configuration.get(name)) for name in WORKERS]

    @staticmethod
    def _command(backend: str, entry: dict, request_path: Path, result_path: Path) -> list[str]:
        return [entry["python"], "-m", WORKERS[backend], "--input", str(request_path), "--output", str(result_path)]

    def run(
        self, backend: str, request: dict, job_dir: Path,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        if backend not in WORKERS:
            raise RuntimeError(f"Unknown backend: {backend}")
        job_dir = Path(job_dir).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        result_path = job_dir / "result.json"
        log_path = job_dir / "backend.log"
        if result_path.exists():
            result_path.unlink()
        try:
            entry = self._configuration().get(backend)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot read backend configuration: {exc}") from exc
        check = self._check(backend, entry)
        if not check["ready"]:
            message = "\n".join(check["details"])
            log_path.write_text(message + "\n", encoding="utf-8")
            raise RuntimeError(message)
        if cancel_event is not None and cancel_event.is_set():
            raise BackendCancelled("Job cancelled before model startup.")
        request_path = job_dir / "backend-request.json"
        request_path.write_text(
            json.dumps({"config": entry, "request": request}, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH", "")]))
        with log_path.open("w", encoding="utf-8") as log:
            try:
                process = subprocess.Popen(
                    self._command(backend, entry, request_path, result_path),
                    stdout=log, stderr=subprocess.STDOUT, env=env,
                    cwd=entry["upstream_dir"], start_new_session=os.name != "nt",
                )
            except OSError as exc:
                raise RuntimeError(f"Cannot start {backend}: {exc}") from exc
            try:
                while process.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        _stop_process(process)
                        raise BackendCancelled("Job cancelled; backend process stopped.")
                    try:
                        process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    _stop_process(process)
        if cancel_event is not None and cancel_event.is_set():
            raise BackendCancelled("Job cancelled.")
        if process.returncode:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
            raise RuntimeError(f"{backend} failed (exit {process.returncode}). See {log_path}\n{tail}")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict) or not result.get("outputs"):
                raise ValueError("Worker returned no audio outputs.")
            for output in result["outputs"]:
                path = Path(output["path"])
                if not path.is_absolute() or not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Worker output is missing: {path}")
                if not path.is_relative_to(job_dir):
                    raise ValueError("Worker output must be inside the job directory.")
            return result
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Invalid {backend} result. See {log_path}: {exc}") from exc
