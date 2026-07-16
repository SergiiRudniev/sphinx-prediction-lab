from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include a timezone: {value}")
    return parsed.astimezone(UTC)


def utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    def intersect(self, other: Window) -> Window | None:
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return Window(start, end) if start < end else None


@dataclass(frozen=True)
class ExchangeContract:
    id: str
    protocol: str
    market_type: str
    address: str
    active: Window
    event_signature: str


@dataclass(frozen=True)
class CorpusConfig:
    path: Path
    payload: dict[str, Any]
    id: str
    version: str
    research_id: str
    window: Window
    data_dir: Path
    chain_id: int
    rpc_env: str
    contracts: tuple[ExchangeContract, ...]

    @classmethod
    def load(cls, path: Path, data_dir: Path | None = None) -> CorpusConfig:
        resolved = path.resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Corpus config must be an object")
        window_payload = payload["window"]
        window = Window(
            parse_utc(str(window_payload["start"])),
            parse_utc(str(window_payload["end_exclusive"])),
        )
        if window.start >= window.end:
            raise ValueError("Corpus window must be non-empty")
        network = payload["network"]
        contracts: list[ExchangeContract] = []
        for item in payload["sources"]["ledger"]["contracts"]:
            active = Window(
                parse_utc(str(item["active_from"])),
                parse_utc(str(item["active_until_exclusive"])),
            )
            contracts.append(
                ExchangeContract(
                    id=str(item["id"]),
                    protocol=str(item["protocol"]),
                    market_type=str(item["market_type"]),
                    address=str(item["address"]),
                    active=active,
                    event_signature=str(item["event_signature"]),
                )
            )
        configured_dir = Path(str(payload["storage"]["default_data_dir"]))
        return cls(
            path=resolved,
            payload=payload,
            id=str(payload["id"]),
            version=str(payload["version"]),
            research_id=str(payload["research_id"]),
            window=window,
            data_dir=(data_dir or configured_dir).resolve(),
            chain_id=int(network["chain_id"]),
            rpc_env=str(network["rpc_env"]),
            contracts=tuple(contracts),
        )
