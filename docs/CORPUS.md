# Sphinx Corpus

Sphinx Corpus is the data system for Sphinx Prediction Lab. It separates immutable
source evidence from derived point-in-time features and evaluation episodes.

## Dataset Map

| Dataset | Ownership |
| --- | --- |
| **Sphinx Atlas** | Market, event, token, outcome and resolution metadata |
| **Sphinx Ledger** | Executed trades, positions and wallet activity |
| **Sphinx Depth** | Price history, spread and collected orderbook state |
| **Sphinx Web** | Time-indexed wallet, funding and market relationships |
| **Sphinx Chronicle** | Model-ready point-in-time rows and labels |
| **Sphinx Replay** | Stateful execution episodes and fills |
| **Sphinx Pulse** | Append-only live WebSocket and chain ingestion |

## Data Flow

```mermaid
flowchart LR
    API["Public APIs"] --> RAW["Immutable raw snapshots"]
    CHAIN["Polygon events"] --> RAW
    WS["Live WebSocket"] --> PULSE["Sphinx Pulse"]
    RAW --> ATLAS["Atlas"]
    RAW --> LEDGER["Ledger"]
    PULSE --> DEPTH["Depth"]
    LEDGER --> WEB["Web"]
    ATLAS --> CHRONICLE["Chronicle"]
    LEDGER --> CHRONICLE
    DEPTH --> CHRONICLE
    WEB --> CHRONICLE
    CHRONICLE --> REPLAY["Replay"]
```

## Point-in-Time Contract

For every decision timestamp `t`:

1. Every feature must have `published_at <= t`.
2. Wallet performance may use only markets resolved before `t`.
3. Graph edges must have occurred before `t`.
4. Resolution and markout labels are joined only after features are frozen.
5. Rows with uncertain publication time are excluded from causal evaluation.
6. Splits are chronological and grouped by event, never random by trade row.

Current positions and leaderboard aggregates cannot be backfilled as historical
features unless their state at `t` is reconstructed from underlying events.

## Required Manifest

Every snapshot must declare:

```text
dataset_id
schema_version
source endpoints and contract addresses
source cursors or block range
minimum and maximum event time
collection time
row count
content hashes
protocol version
known gaps
license and redistribution constraints
```

## Storage

- Raw responses: compressed immutable objects, partitioned by source and date.
- Normalized facts: Parquet with UTC timestamps and explicit source provenance.
- Graph edges: Parquet edge tables plus point-in-time neighborhood indices.
- Training rows: frozen Sphinx Chronicle snapshots.
- Large data and credentials: never committed to Git.

## Historical Limits

Executed trades, resolution and on-chain activity can be backfilled. Complete
historical off-chain order placement, cancellation and L2 depth generally cannot be
reconstructed. Sphinx Depth therefore distinguishes:

- `historical_price`: backfilled price observations;
- `collected_l2`: full snapshots collected by Sphinx Pulse;
- `synthetic_depth`: prohibited for accepted execution evidence.

## Versioning

Datasets and models version independently:

```text
Model: Sphinx Trace S0
Training data: Sphinx Chronicle v1.0
Graph data: Sphinx Web v1.0
Execution data: Sphinx Replay v1.0
Snapshot cutoff: YYYY-MM-DDTHH:MM:SSZ
```

Schema changes require a major dataset version when they alter row meaning,
causal availability or label construction.
