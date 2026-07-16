# Sphinx Pulse Operations

Sphinx Pulse records the public Polymarket market WebSocket and hourly Gamma
catalog snapshots. UTC-hour zstd JSONL shards are grouped into one GitHub Release
per UTC day.

The default resource guard tracks the 1,000 highest-volume active CLOB markets
and refreshes the selection every five minutes. Change `selection.max_markets`
only after observing daily volume, memory and upload duration.

## Storage contract

```text
/srv/sphinx-pulse/data/
|-- raw/date=YYYY-MM-DD/hour=HH/*.jsonl.zst
|-- receipts/YYYY-MM-DD.json
|-- staging/
`-- status/{collector,publisher}.json
```

The publisher uploads every completed day, verifies the GitHub asset size and
SHA-256 digest, uploads the manifest last, verifies the complete remote asset
set, writes a local receipt, and only then removes the local `date=...` folder.
Partial or failed uploads never trigger deletion.

GitHub credentials are read from `/run/secrets/github_token`; they are not passed
through the container environment or written to logs.

## Commands

```bash
sudo docker compose --env-file deploy/pulse/.env \
  -f deploy/pulse/compose.yaml up -d --build

sudo docker compose -f deploy/pulse/compose.yaml ps
sudo docker compose -f deploy/pulse/compose.yaml logs -f --tail 100 collector

cat /srv/sphinx-pulse/data/status/collector.json
cat /srv/sphinx-pulse/data/status/publisher.json
```

## Restore

Download every asset from the matching `pulse-YYYY-MM-DD` release and verify it
against `sphinx-pulse-YYYY-MM-DD-manifest.json`. Each `.jsonl.zst` file contains
newline-delimited JSON and may contain multiple concatenated zstd frames.
