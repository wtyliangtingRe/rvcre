"""Loader forwarding, scope and failure behavior; no GPU/acoustic claims."""

import sys
import types

import pytest

from rvcre.workers import voxcpm_loading as loading


def modern_loader(filename, device="cpu", *, backend="mmap"):
    return {"path": filename, "device": device, "backend": backend}


def legacy_loader(filename, device="cpu"):
    return {"path": filename, "device": device}


@pytest.mark.parametrize("requested,platform_name,expected", [
    ("auto", "win32", "pread"), ("auto", "linux", "mmap"), ("auto", "darwin", "mmap"),
    ("mmap", "win32", "mmap"), ("pread", "linux", "pread"),
])
def test_loader_selection_is_platform_scoped_and_explicitly_overridable(requested, platform_name, expected):
    assert loading.select_backend(requested, modern_loader, platform_name) == (expected, True)


def test_old_safetensors_keeps_its_compatible_call_signature():
    assert loading.select_backend("auto", legacy_loader, "win32") == ("mmap", False)
    with pytest.raises(RuntimeError, match="backend='pread'"):
        loading.select_backend("pread", legacy_loader, "win32")
    with pytest.raises(ValueError, match="safetensors_backend"):
        loading.select_backend("typo", modern_loader, "win32")


@pytest.fixture
def upstream(monkeypatch):
    calls = []

    class Base:
        def load_state_dict(self, state, strict=True):
            calls.append(("load_state_dict", state, strict))
            return "state-loaded"

        def to(self, *args, **kwargs):
            calls.append(("to", args, kwargs))
            return self

    class Model(Base):
        pass

    module = types.SimpleNamespace(load_file=modern_loader, VoxCPM2Model=Model)
    monkeypatch.setitem(sys.modules, "voxcpm.model.voxcpm2", module)
    monkeypatch.setattr(loading.sys, "platform", "win32")
    monkeypatch.delenv("RVCRE_SAFETENSORS_BACKEND", raising=False)
    return module, Model, Base, calls


def test_real_loader_signature_and_model_methods_are_preserved_then_restored(upstream, capsys):
    module, Model, Base, calls = upstream
    with loading.model_loading({}) as metadata:
        result = module.load_file("weights.safetensors", "cpu")
        assert result == {"path": "weights.safetensors", "device": "cpu", "backend": "pread"}
        assert metadata == {"requested": "auto", "backend": "pread"}
        model = Model()
        assert model.load_state_dict(result, strict=False) == "state-loaded"
        assert model.to(device="cuda", dtype="unchanged") is model
    assert calls == [("load_state_dict", result, False), ("to", (), {"device": "cuda", "dtype": "unchanged"})]
    assert module.load_file is modern_loader
    assert "load_state_dict" not in vars(Model) and "to" not in vars(Model)
    assert Model.to is Base.to and Model.load_state_dict is Base.load_state_dict
    log = capsys.readouterr().out
    for stage in ("weights.read", "model.load_state_dict", "model.to"):
        assert f"{stage} begin" in log and f"{stage} done" in log


@pytest.mark.parametrize("failure_stage", ["weights", "state", "device"])
def test_loading_failure_propagates_and_restores_owned_and_inherited_attributes(upstream, failure_stage):
    module, Model, Base, _ = upstream

    def failed_load(filename, device="cpu", *, backend="mmap"):
        raise OSError("checkpoint read failed")

    def failed_method(self, *args, **kwargs):
        raise RuntimeError("native-stage test double")

    if failure_stage == "weights":
        module.load_file = failed_load
    else:
        setattr(Model, "load_state_dict" if failure_stage == "state" else "to", failed_method)
    old_dict = dict(vars(Model))
    original_loader = module.load_file
    with pytest.raises((OSError, RuntimeError)):
        with loading.model_loading({}):
            state = module.load_file("weights")
            model = Model()
            model.load_state_dict(state)
            model.to("cuda")
    assert module.load_file is original_loader
    assert dict(vars(Model)) == old_dict


def test_old_version_and_environment_override_forward_without_unexpected_keywords(upstream, monkeypatch):
    module, _, _, _ = upstream
    module.load_file = legacy_loader
    with loading.model_loading({}) as info:
        assert info["backend"] == "mmap"
        assert module.load_file("weights", device="cpu") == {"path": "weights", "device": "cpu"}
    module.load_file = modern_loader
    monkeypatch.setenv("RVCRE_SAFETENSORS_BACKEND", "mmap")
    with loading.model_loading({"safetensors_backend": "pread"}) as info:
        assert info["backend"] == "mmap"
        assert module.load_file("weights")["backend"] == "mmap"
