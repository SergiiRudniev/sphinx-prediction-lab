import importlib
from pathlib import Path
from typing import Any

from scripts.run_h010_policy_replay import _atomic_torch_save


def test_atomic_replay_checkpoint_retries_transient_reader_lock(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runner: Any = importlib.import_module("scripts.run_h010_policy_replay")

    real_replace = runner.os.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("reader lock")
        real_replace(source, target)

    monkeypatch.setattr(runner.os, "replace", flaky_replace)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    target = tmp_path / "checkpoint.pt"

    _atomic_torch_save(target, {"epoch": 3})

    assert target.is_file()
    assert attempts == 3
