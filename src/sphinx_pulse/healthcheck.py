from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def status_is_healthy(path: Path, maximum_age_seconds: float) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        heartbeat_ms = int(payload["heartbeat_ms"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    age_ms = max(0, int(time.time() * 1000) - heartbeat_ms)
    return age_ms <= maximum_age_seconds * 1000 and payload.get("status") != "error"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: healthcheck STATUS_PATH MAXIMUM_AGE_SECONDS")
    healthy = status_is_healthy(Path(sys.argv[1]), float(sys.argv[2]))
    raise SystemExit(0 if healthy else 1)


if __name__ == "__main__":
    main()
