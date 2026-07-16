from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from sphinx_pulse import SCHEMA_VERSION
from sphinx_pulse.storage import atomic_json, count_zstd_jsonl

GITHUB_API_VERSION = "2026-03-10"
DAY_PATTERN = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetSpec:
    path: Path
    name: str
    size: int
    digest: str
    rows: int | None


class ReleaseClient(Protocol):
    def ensure_release(self, day: str) -> dict[str, Any]: ...

    def upload_verified(self, release: dict[str, Any], asset: AssetSpec) -> dict[str, Any]: ...

    def verify_assets(self, release_id: int, assets: Iterable[AssetSpec]) -> None: ...


class GitHubReleaseClient:
    def __init__(self, repository: str, token: str) -> None:
        self.repository = repository
        self.client = httpx.Client(
            base_url="https://api.github.com",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
                "User-Agent": "sphinx-pulse/1.0",
            },
            timeout=httpx.Timeout(120.0, connect=30.0),
        )

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _object(response: httpx.Response) -> dict[str, Any]:
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("GitHub response is not an object")
        return payload

    def verify_access(self) -> dict[str, Any]:
        response = self.client.get(f"/repos/{self.repository}")
        response.raise_for_status()
        payload = self._object(response)
        if payload.get("full_name") != self.repository:
            raise RuntimeError("GitHub repository identity mismatch")
        if not payload.get("permissions", {}).get("push"):
            raise RuntimeError("GitHub token does not have repository write access")
        return payload

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        delay = 1.0
        response: httpx.Response | None = None
        for attempt in range(6):
            response = self.client.request(method, url, **kwargs)
            if response.status_code != 429 and response.status_code < 500:
                return response
            if attempt < 5:
                time.sleep(delay)
                delay = min(30.0, delay * 2.0)
        if response is None:
            raise RuntimeError("GitHub request produced no response")
        return response

    def ensure_release(self, day: str) -> dict[str, Any]:
        tag = f"pulse-{day}"
        existing = self._request("GET", f"/repos/{self.repository}/releases/tags/{tag}")
        if existing.status_code == 200:
            return self._object(existing)
        if existing.status_code != 404:
            existing.raise_for_status()
        created = self._request(
            "POST",
            f"/repos/{self.repository}/releases",
            json={
                "tag_name": tag,
                "target_commitish": "main",
                "name": f"Sphinx Pulse {day}",
                "body": (
                    f"UTC market-channel capture for {day}. "
                    "See the manifest asset for provenance and SHA-256 digests."
                ),
                "draft": False,
                "prerelease": False,
            },
        )
        created.raise_for_status()
        return self._object(created)

    def _assets(self, release_id: int) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        page = 1
        while True:
            response = self._request(
                "GET",
                f"/repos/{self.repository}/releases/{release_id}/assets",
                params={"per_page": 100, "page": page},
            )
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list):
                raise RuntimeError("GitHub release assets response is not a list")
            assets.extend(item for item in batch if isinstance(item, dict))
            if len(batch) < 100:
                return assets
            page += 1

    @staticmethod
    def _matches(remote: dict[str, Any], local: AssetSpec) -> bool:
        return (
            remote.get("state") == "uploaded"
            and int(remote.get("size", -1)) == local.size
            and remote.get("digest") == f"sha256:{local.digest}"
        )

    def upload_verified(self, release: dict[str, Any], asset: AssetSpec) -> dict[str, Any]:
        release_id = int(release["id"])
        existing = {item.get("name"): item for item in self._assets(release_id)}.get(asset.name)
        if existing is not None:
            if self._matches(existing, asset):
                return existing
            raise RuntimeError(f"Remote asset differs from local file: {asset.name}")

        upload_url = str(release["upload_url"]).split("{", 1)[0]
        content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        delay = 1.0
        response: httpx.Response | None = None
        for attempt in range(6):
            with asset.path.open("rb") as handle:
                response = self.client.post(
                    upload_url,
                    params={"name": asset.name},
                    headers={"Content-Type": content_type},
                    content=handle,
                    timeout=httpx.Timeout(7200.0, connect=30.0),
                )
            if response.is_success:
                break
            if (
                response.status_code == 422
                or response.status_code == 429
                or response.status_code >= 500
            ):
                current = {item.get("name"): item for item in self._assets(release_id)}.get(
                    asset.name
                )
                if current is not None and self._matches(current, asset):
                    return current
                if current is not None and current.get("state") == "starter":
                    deleted = self._request(
                        "DELETE",
                        f"/repos/{self.repository}/releases/assets/{int(current['id'])}",
                    )
                    if deleted.status_code != 204:
                        deleted.raise_for_status()
                elif response.status_code == 422:
                    raise RuntimeError(f"Remote asset differs from local file: {asset.name}")
            if attempt < 5:
                time.sleep(delay)
                delay = min(30.0, delay * 2.0)
        if response is None:
            raise RuntimeError("GitHub upload produced no response")
        response.raise_for_status()
        remote = self._object(response)
        if not self._matches(remote, asset):
            raise RuntimeError(f"GitHub checksum verification failed: {asset.name}")
        return remote

    def verify_assets(self, release_id: int, assets: Iterable[AssetSpec]) -> None:
        remote = {item.get("name"): item for item in self._assets(release_id)}
        for asset in assets:
            current = remote.get(asset.name)
            if current is None or not self._matches(current, asset):
                raise RuntimeError(f"Remote release verification failed: {asset.name}")


class DayPublisher:
    def __init__(
        self,
        data_dir: Path,
        client: ReleaseClient,
        *,
        maximum_asset_bytes: int = 2_000_000_000,
        stable_age_seconds: int = 300,
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.raw_dir = self.data_dir / "raw"
        self.staging_dir = self.data_dir / "staging"
        self.receipts_dir = self.data_dir / "receipts"
        self.client = client
        self.maximum_asset_bytes = int(maximum_asset_bytes)
        self.stable_age_seconds = int(stable_age_seconds)

    def eligible_days(self, today: date | None = None) -> list[Path]:
        current = datetime.now(tz=UTC).date() if today is None else today
        if not self.raw_dir.is_dir():
            return []
        eligible: list[Path] = []
        now = time.time()
        for path in self.raw_dir.iterdir():
            match = DAY_PATTERN.match(path.name)
            if not match or not path.is_dir() or date.fromisoformat(match.group(1)) >= current:
                continue
            files = [item for item in path.rglob("*") if item.is_file()]
            latest_mtime = max((item.stat().st_mtime for item in files), default=now)
            if files and latest_mtime <= now - self.stable_age_seconds:
                eligible.append(path)
        return sorted(eligible)

    @staticmethod
    def _asset_name(day: str, path: Path, root: Path) -> str:
        relative = path.relative_to(root).as_posix().replace("/", "--")
        return f"sphinx-pulse-{day}--{relative}"

    def _specs(self, day_dir: Path, day: str) -> tuple[list[AssetSpec], dict[str, Any]]:
        specs: list[AssetSpec] = []
        total_rows = 0
        paths = sorted(item for item in day_dir.rglob("*") if item.is_file())
        for path in paths:
            size = path.stat().st_size
            if size >= self.maximum_asset_bytes:
                raise RuntimeError(f"Pulse asset exceeds GitHub release limit: {path}")
            rows = count_zstd_jsonl(path) if path.name.endswith(".jsonl.zst") else None
            total_rows += rows or 0
            specs.append(
                AssetSpec(
                    path=path,
                    name=self._asset_name(day, path, day_dir),
                    size=size,
                    digest=sha256(path),
                    rows=rows,
                )
            )
        if not specs:
            raise RuntimeError(f"Pulse day has no files: {day}")
        manifest = {
            "dataset_id": f"sphinx-pulse-{day}",
            "schema_version": SCHEMA_VERSION,
            "source": {
                "catalog": "https://gamma-api.polymarket.com/markets/keyset",
                "stream": "wss://ws-subscriptions-clob.polymarket.com/ws/market",
            },
            "utc_date": day,
            "created_at": datetime.fromtimestamp(
                max(path.stat().st_mtime for path in paths),
                tz=UTC,
            ).isoformat(),
            "file_count": len(specs),
            "row_count": total_rows,
            "files": [
                {
                    "asset": spec.name,
                    "bytes": spec.size,
                    "sha256": spec.digest,
                    "rows": spec.rows,
                }
                for spec in specs
            ],
            "known_gaps": [],
            "redistribution": "Public Polymarket market-channel observations; no ownership claim.",
        }
        return specs, manifest

    def _delete_verified_day(self, day_dir: Path) -> None:
        resolved = day_dir.resolve()
        if resolved.parent != self.raw_dir or DAY_PATTERN.match(resolved.name) is None:
            raise RuntimeError(f"Refusing unsafe Pulse deletion: {resolved}")
        shutil.rmtree(resolved)

    def publish_day(self, day_dir: Path) -> dict[str, Any]:
        match = DAY_PATTERN.match(day_dir.name)
        if match is None:
            raise ValueError(f"Invalid Pulse day directory: {day_dir}")
        day = match.group(1)
        specs, manifest = self._specs(day_dir, day)
        staging = self.staging_dir / day
        staging.mkdir(parents=True, exist_ok=True)
        manifest_path = staging / f"sphinx-pulse-{day}-manifest.json"
        atomic_json(manifest_path, manifest)
        manifest_spec = AssetSpec(
            path=manifest_path,
            name=manifest_path.name,
            size=manifest_path.stat().st_size,
            digest=sha256(manifest_path),
            rows=None,
        )
        release = self.client.ensure_release(day)
        for spec in [*specs, manifest_spec]:
            self.client.upload_verified(release, spec)
        self.client.verify_assets(int(release["id"]), [*specs, manifest_spec])
        receipt = {
            "schema_version": 1,
            "dataset_id": manifest["dataset_id"],
            "release_id": int(release["id"]),
            "release_url": release.get("html_url"),
            "uploaded_at": datetime.now(tz=UTC).isoformat(),
            "assets": len(specs) + 1,
            "manifest_sha256": manifest_spec.digest,
        }
        atomic_json(self.receipts_dir / f"{day}.json", receipt)
        self._delete_verified_day(day_dir)
        shutil.rmtree(staging, ignore_errors=True)
        return receipt


def _read_token(path: Path) -> str:
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("GitHub token file is empty")
    return token


def run_loop(
    publisher: DayPublisher,
    status_path: Path,
    *,
    interval_seconds: int,
    once: bool,
) -> None:
    while True:
        try:
            published = [publisher.publish_day(day) for day in publisher.eligible_days()]
            atomic_json(
                status_path,
                {
                    "heartbeat_ms": int(time.time() * 1000),
                    "status": "ok",
                    "published": published,
                },
            )
            print(json.dumps({"published": len(published)}), flush=True)
        except Exception as exc:
            atomic_json(
                status_path,
                {
                    "heartbeat_ms": int(time.time() * 1000),
                    "status": "error",
                    "error": repr(exc),
                },
            )
            print(json.dumps({"publisher_error": repr(exc)}), flush=True)
        if once:
            return
        time.sleep(max(60, interval_seconds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--verify-auth", action="store_true")
    args = parser.parse_args()
    client = GitHubReleaseClient(args.repository, _read_token(args.token_file))
    try:
        if args.verify_auth:
            payload = client.verify_access()
            print(json.dumps({"repository": payload["full_name"], "push": True}))
            return
        publisher = DayPublisher(args.data_dir, client)
        run_loop(
            publisher,
            args.data_dir / "status" / "publisher.json",
            interval_seconds=args.interval_seconds,
            once=args.once,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
