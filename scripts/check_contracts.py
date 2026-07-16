"""Validate repository JSON contracts without downloading data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: object = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Contract root must be an object: {path}")
    return value


def main() -> None:
    files = sorted((ROOT / "configs").rglob("*.json")) + sorted((ROOT / "schemas").rglob("*.json"))
    if not files:
        raise RuntimeError("No JSON contracts found")
    for path in files:
        load_object(path)
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
