"""JSON contract loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object and reject non-object roots."""

    with Path(path).open(encoding="utf-8") as handle:
        value: object = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value
