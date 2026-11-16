"""Scoped Windows weight loading and native-crash breadcrumbs for VoxCPM2.

No model code is imported until model_loading() is entered inside the worker.
The upstream checkout and the process-wide safetensors module are not modified.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from functools import wraps
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys


def log_runtime(torch) -> dict:
    details = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
    }
    for package in ("safetensors", "transformers", "numpy"):
        try:
            details[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            details[package] = "not installed"
    print("[VoxCPM runtime] " + json.dumps(details, ensure_ascii=True), flush=True)
    return details


def select_backend(requested: str, load_file, platform_name: str) -> tuple[str, bool]:
    if not isinstance(requested, str) or requested not in {"auto", "mmap", "pread"}:
        raise ValueError("safetensors_backend must be auto, mmap or pread.")
    supports_backend = "backend" in inspect.signature(load_file).parameters
    selected = "pread" if requested == "auto" and platform_name == "win32" and supports_backend else requested
    if selected == "auto":
        selected = "mmap"
    if selected == "pread" and not supports_backend:
        raise RuntimeError(
            "The installed safetensors does not support backend='pread'. "
            "Update safetensors to 0.8.0 or newer in the VoxCPM backend environment."
        )
    return selected, supports_backend


@contextmanager
def _temporary_attribute(owner, name: str, value):
    # Remember ownership as well as the value: inherited nn.Module methods must
    # become inherited again, rather than leave shadows on VoxCPM2Model.
    existed = name in vars(owner)
    previous = vars(owner).get(name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if existed:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


def _traced_method(original, label: str):
    @wraps(original)
    def traced(self, *args, **kwargs):
        detail = ""
        if label == "model.to":
            target = args[0] if args else kwargs.get("device", kwargs.get("dtype", "unspecified"))
            detail = f" target={target}"
        print(f"[VoxCPM load] {label} begin{detail}", flush=True)
        result = original(self, *args, **kwargs)
        print(f"[VoxCPM load] {label} done", flush=True)
        return result
    return traced


@contextmanager
def model_loading(config: dict):
    module = importlib.import_module("voxcpm.model.voxcpm2")
    original = module.load_file
    requested = os.environ.get("RVCRE_SAFETENSORS_BACKEND", config.get("safetensors_backend", "auto"))
    selected, supports_backend = select_backend(requested, original, sys.platform)
    print(f"[VoxCPM load] safetensors backend={selected} requested={requested}", flush=True)
    if requested == "auto" and sys.platform == "win32" and not supports_backend:
        print("[VoxCPM load] Legacy safetensors: pread unavailable; keeping the existing mmap loader. "
              "For Windows pread support, update safetensors in .venvs/voxcpm.", flush=True)

    @wraps(original)
    def read_weights(filename, *args, **kwargs):
        if supports_backend:
            kwargs["backend"] = selected
        print(f"[VoxCPM load] weights.read begin backend={selected}", flush=True)
        result = original(filename, *args, **kwargs)
        print(f"[VoxCPM load] weights.read done tensors={len(result)}", flush=True)
        return result

    model_class = module.VoxCPM2Model
    with ExitStack() as stack:
        stack.enter_context(_temporary_attribute(module, "load_file", read_weights))
        for name in ("load_state_dict", "to"):
            stack.enter_context(_temporary_attribute(
                model_class, name, _traced_method(getattr(model_class, name), f"model.{name}"),
            ))
        yield {"requested": requested, "backend": selected}
