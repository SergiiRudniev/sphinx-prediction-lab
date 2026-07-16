from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zstandard

_PART_PATTERN = re.compile(r"^pulse-(?P<hour>\d{2})-part-(?P<part>\d{5})\.jsonl\.zst$")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class HourlyZstdStore:
    """Append complete zstd frames to bounded UTC-hour files."""

    def __init__(
        self,
        root: Path,
        *,
        max_part_bytes: int = 512 * 1024 * 1024,
        compression_level: int = 6,
    ) -> None:
        self.root = root.resolve()
        self.max_part_bytes = int(max_part_bytes)
        self.compressor = zstandard.ZstdCompressor(
            level=int(compression_level),
            write_checksum=True,
        )

    @staticmethod
    def _partition(timestamp_ms: int) -> tuple[str, str]:
        value = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
        return value.date().isoformat(), f"{value.hour:02d}"

    def _target(self, date: str, hour: str, estimated_bytes: int) -> Path:
        directory = self.root / f"date={date}" / f"hour={hour}"
        directory.mkdir(parents=True, exist_ok=True)
        candidates: list[tuple[int, Path]] = []
        for path in directory.glob("pulse-*.jsonl.zst"):
            match = _PART_PATTERN.match(path.name)
            if match and match.group("hour") == hour:
                candidates.append((int(match.group("part")), path))
        if candidates:
            part, latest = max(candidates)
            if latest.stat().st_size + estimated_bytes <= self.max_part_bytes:
                return latest
            part += 1
        else:
            part = 0
        return directory / f"pulse-{hour}-part-{part:05d}.jsonl.zst"

    def append(self, records: Iterable[dict[str, Any]]) -> tuple[int, int]:
        grouped: dict[tuple[str, str], list[bytes]] = {}
        count = 0
        for record in records:
            timestamp_ms = int(record["received_at_ms"])
            key = self._partition(timestamp_ms)
            encoded = json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode()
            grouped.setdefault(key, []).append(encoded + b"\n")
            count += 1

        written = 0
        for (date, hour), rows in grouped.items():
            raw = b"".join(rows)
            frame = self.compressor.compress(raw)
            target = self._target(date, hour, len(frame))
            with target.open("ab") as handle:
                handle.write(frame)
                handle.flush()
            written += len(frame)
        return count, written


def count_zstd_jsonl(path: Path) -> int:
    count = 0
    with (
        path.open("rb") as source,
        zstandard.ZstdDecompressor().stream_reader(source) as reader,
    ):
        while chunk := reader.read(1024 * 1024):
            count += chunk.count(b"\n")
    return count
