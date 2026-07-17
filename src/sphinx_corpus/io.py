from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import zstandard


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _replace(temporary, path)


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _compress(raw: bytes) -> bytes:
    return zstandard.ZstdCompressor(level=6, write_checksum=True).compress(raw)


def write_json_zst(path: Path, payload: Any) -> int:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    encoded = _compress(raw)
    _atomic_bytes(path, encoded)
    return len(encoded)


def read_json_zst(path: Path) -> Any:
    with path.open("rb") as source:
        raw = zstandard.ZstdDecompressor().stream_reader(source).read()
    return json.loads(raw)


def write_jsonl_zst(path: Path, records: Iterable[dict[str, Any]]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("wb") as handle:
        with zstandard.ZstdCompressor(
            level=6,
            write_checksum=True,
        ).stream_writer(handle, closefd=False) as writer:
            for record in records:
                writer.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode()
                    + b"\n"
                )
                count += 1
        handle.flush()
        os.fsync(handle.fileno())
    _replace(temporary, path)
    return count, path.stat().st_size


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
    with (
        path.open("rb") as source,
        zstandard.ZstdDecompressor().stream_reader(source) as reader,
    ):
        buffered = b""
        while chunk := reader.read(1024 * 1024):
            buffered += chunk
            lines = buffered.split(b"\n")
            buffered = lines.pop()
            for line in lines:
                if line:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise TypeError(f"Expected JSONL object in {path}")
                    yield value
        if buffered:
            value = json.loads(buffered)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSONL object in {path}")
            yield value


def count_jsonl_zst(path: Path) -> int:
    count = 0
    has_data = False
    last_byte = 0
    with (
        path.open("rb") as source,
        zstandard.ZstdDecompressor().stream_reader(source) as reader,
    ):
        while chunk := reader.read(8 * 1024 * 1024):
            has_data = True
            count += chunk.count(b"\n")
            last_byte = chunk[-1]
    return count + int(has_data and last_byte != ord("\n"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _replace(temporary, path)


def _replace(source: Path, target: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(1.0, 0.025 * (2**attempt)))
    raise PermissionError(f"Could not atomically replace {target}: {last_error}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_disk_reserve(path: Path, minimum_free_gib: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    minimum = int(float(minimum_free_gib) * 1024**3)
    if free < minimum:
        raise RuntimeError(
            f"Sphinx Corpus disk reserve reached: {free / 1024**3:.2f} GiB free, "
            f"{minimum_free_gib:.2f} GiB required"
        )


def now_utc() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def build_manifest(
    root: Path,
    *,
    corpus_id: str,
    version: str,
    research_id: str,
    source_config: dict[str, Any],
    workers: int = 1,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("manifest workers must be positive")
    ignored = tuple(str(value) for value in source_config["storage"].get("ignored_prefixes", []))
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp") or path.name == "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(ignored):
            continue
        paths.append(path)
    entry_builder = partial(_manifest_entry, root=root)
    files: list[dict[str, Any]] = []
    if workers == 1:
        files = [entry_builder(path) for path in paths]
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="sphinx-manifest",
        ) as executor:
            for offset in range(0, len(paths), 4096):
                files.extend(executor.map(entry_builder, paths[offset : offset + 4096]))
    manifest = {
        "dataset_id": corpus_id,
        "version": version,
        "research_id": research_id,
        "generated_at": now_utc(),
        "window": source_config["window"],
        "protocols": sorted(
            {item["protocol"] for item in source_config["sources"]["ledger"]["contracts"]}
        ),
        "sources": source_config["sources"],
        "known_gaps": source_config["known_gaps"],
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "row_count": sum(int(item.get("rows", 0)) for item in files),
    }
    atomic_json(root / "manifest.json", manifest)
    return manifest


def _manifest_entry(path: Path, *, root: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.name.endswith(".jsonl.zst"):
        entry["rows"] = count_jsonl_zst(path)
    return entry
