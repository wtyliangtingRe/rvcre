"""Atomic record updates tolerate brief Windows reader locks without hiding errors."""

from pathlib import Path

import pytest

from rvcre import storage


def test_atomic_json_replacement_retries_a_transient_reader_lock(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    previous = {"completed": 1, "text": "已经完成的第一段。"}
    updated = {"completed": 2, "text": "第二段也已经完成。"}
    storage.write_json(target, previous)
    original_bytes = target.read_bytes()
    real_replace = Path.replace
    attempts, delays = [], []

    def locked_replace(source, destination):
        attempts.append((source, destination))
        assert target.read_bytes() == original_bytes
        assert storage.read_json(source) == updated
        if len(attempts) < 3:
            raise PermissionError("test-only Windows reader lock")
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr(storage.time, "sleep", delays.append)
    storage.write_json(target, updated)

    assert len(attempts) == 3
    assert len(delays) == 2 and 0 < sum(delays) <= 0.5
    assert storage.read_json(target) == updated
    assert not target.with_suffix(".json.tmp").exists()


def test_persistent_replace_failure_is_bounded_and_preserves_original_json(tmp_path, monkeypatch):
    target = tmp_path / "job.json"
    original = {"status": "running", "completed": 1}
    storage.write_json(target, original)
    original_bytes = target.read_bytes()
    attempts, delays = [], []

    def locked_replace(source, destination):
        attempts.append((source, destination))
        assert target.read_bytes() == original_bytes
        raise PermissionError("test-only persistent file lock")

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr(storage.time, "sleep", delays.append)
    with pytest.raises(PermissionError, match="persistent file lock"):
        storage.write_json(target, {"status": "succeeded", "completed": 2})

    assert 1 < len(attempts) <= 10
    assert len(delays) == len(attempts) - 1 and 0 < sum(delays) <= 0.5
    assert target.read_bytes() == original_bytes
    assert storage.read_json(target) == original
