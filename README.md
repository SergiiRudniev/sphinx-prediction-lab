<div align="center">

![Sphinx Prediction Lab](assets/sphinx-prediction-lab-hero.png)

<img src="assets/sphinx-icon.png" alt="Sphinx icon" width="112">

# Sphinx Prediction Lab

**Causal wallet-flow intelligence for prediction markets**

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-1f2328?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/model-Sphinx%20Trace%20S0-b08d57)](docs/ARCHITECTURE.md)
[![Corpus](https://img.shields.io/badge/data-Sphinx%20Corpus-23a9c9)](docs/CORPUS.md)
[![Status](https://img.shields.io/badge/status-H021%20development--qualified-d6c6a5)](#current-status)
[![CI](https://github.com/SergiiRudniev/sphinx-prediction-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/SergiiRudniev/sphinx-prediction-lab/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-1f2328)](LICENSE)

</div>

Sphinx Prediction Lab is a research system for prediction-market intelligence.
It studies market state, participant histories, wallet relationships and causal
capital flow without using news or natural language in the model core.

The first model family, **Sphinx Trace**, is designed to estimate fair outcome
probabilities, detect potentially informed activity and manage positions through
entry, reduction, exit or settlement.

> [!IMPORTANT]
> The repository contains the S0 architecture, causal data contracts, trained
> development checkpoints and exact replay evidence. It has no untouched-test or
> accepted live trading result.

> [!WARNING]
> This is research software, not financial advice. Automated execution must remain
> disabled until a jurisdiction check, locked paper-forward evaluation and
> deterministic capital controls have passed.

## Research Map

```mermaid
flowchart TB
    LAB["Sphinx Prediction Lab"]
    LAB --> CORPUS["Sphinx Corpus<br/>causal market and wallet data"]
    LAB --> TRACE["Sphinx Trace<br/>wallet-flow model family"]
    CORPUS --> ATLAS["Atlas<br/>markets and outcomes"]
    CORPUS --> LEDGER["Ledger<br/>fills and activity"]
    CORPUS --> DEPTH["Depth<br/>prices and orderbook"]
    CORPUS --> WEB["Web<br/>wallet graph"]
    ATLAS --> CHRONICLE["Chronicle<br/>point-in-time training data"]
    LEDGER --> CHRONICLE
    DEPTH --> CHRONICLE
    WEB --> CHRONICLE
    CHRONICLE --> S0["Sphinx Trace S0"]
    S0 --> SIGNAL["Probability · Flow · Edge · Uncertainty"]
    SIGNAL --> POLICY["Learned CALL · SKIP · sizing policy"]
```

## Model Family

| Family | Specialization | Primary objective | Current state |
| --- | --- | --- | --- |
| **Sphinx Trace** | Wallet-flow prediction-market intelligence | Detect informed activity, estimate fair probabilities and manage positions for maximum net edge | H021 incumbent; H022 replay pending; test unopened |

![Sphinx Trace S0](assets/sphinx-trace-s0-banner.png)

### Sphinx Trace S0

The H008 S0 direction is a stateful full-universe causal architecture:

1. **Market-state encoder** — prices, outcomes, activity and structure.
2. **Streaming wallet memory** — every valid participant without a hard wallet cap.
3. **Temporal graph encoder** — wallet, funding, market and event relationships.
4. **Market and event memory** — recurrent state across the complete lifecycle.
5. **Universe memory** — relative capital flow and opportunity across Polymarket.
6. **Prediction and Position Books** — prior beliefs, calls, balance and positions.
7. **Opportunity policy** — terminal outcome, uncertainty, `SKIP` and learned sizing.

The policy learns outcome selection, abstention and balance-conditioned sizing.
The simulator separately enforces physical cash, liquidity, cost and causal-data
constraints.

See [S0 Architecture](docs/ARCHITECTURE.md) and the machine-readable
[`sphinx_trace_research_mandate_v1.json`](configs/trace/sphinx_trace_research_mandate_v1.json).

## Sphinx Corpus

| Dataset | Contract |
| --- | --- |
| **Sphinx Atlas** | Markets, events, outcomes, categories and resolution metadata |
| **Sphinx Ledger** | Executed trades, positions and wallet activity |
| **Sphinx Depth** | Price, spread and historical/live orderbook state |
| **Sphinx Web** | Temporal `wallet ↔ wallet ↔ market` graph |
| **Sphinx Chronicle** | Point-in-time model training dataset |
| **Sphinx Replay** | Stateful backtest and execution episodes |
| **Sphinx Pulse** | Append-only live market and wallet stream |

Raw datasets and credentials are never committed. Every usable snapshot requires
source cursors, hashes, schema versions and a frozen UTC cutoff.

See [Sphinx Corpus](docs/CORPUS.md) and
[`sphinx_chronicle_v1.json`](configs/corpus/sphinx_chronicle_v1.json).

## Causal Research Standard

Sphinx research must preserve:

- features published no later than the decision timestamp;
- wallet reputation computed only from outcomes already resolved at that time;
- chronological, event-grouped splits with target purging;
- validation-only model, threshold and policy selection;
- one-time untouched test opening after source and configuration hashes are locked;
- executable bid/ask, depth, latency, spread, fees and slippage;
- paper-forward evidence before any automated capital allocation;
- complete accounting of accepted, rejected and invalidated hypotheses.

## Current Status

| Item | Status |
| --- | --- |
| Lab and naming | Locked |
| Sphinx Corpus taxonomy | Locked |
| Sphinx Trace S0 contract | Design registered as `SPH-T-H000` |
| Historical backfill | Full `SPH-T-H001` in development; fast S0 `SPH-T-H002` qualified |
| Full-universe research mandate | `SPH-T-H008` registered |
| Development tape | 47,252,399 causal rows; 809,614 validation decisions |
| Protocol-exact fees | `SPH-T-H016` receipt-qualified schedule |
| Sphinx Pulse collector | Implemented; passive collection only |
| Development incumbent | `SPH-T-H021`; 65,854,783 parameters |
| Conditional net-edge candidate | `SPH-T-H022` seed 17; 77,005,637 combined neural parameters; replay pending |
| Exact development replay | `+21.53%` at 1.0x fees; `+20.49%` at 2.0x fees |
| Untouched test | Unopened |
| Accepted backtest | None — development evidence only |
| Accepted forward result | None |
| Live execution | Disabled |

## Repository Map

```text
.
|-- assets/                 Generated Sphinx brand assets
|-- configs/                Frozen data and model contracts
|-- docs/                   Architecture, corpus and research protocol
|-- schemas/                Point-in-time JSON schemas
|-- deploy/pulse/           Live collector and verified daily publisher
|-- scripts/                Local Corpus backfill and contract entrypoints
|-- src/sphinx_trace/       Runtime contracts and deterministic policy boundary
|-- tests/                  Data-free contract tests
|-- .github/                CI, issue forms and repository governance
`-- pyproject.toml          Python package and development tooling
```

## Development

```bash
git clone https://github.com/SergiiRudniev/sphinx-prediction-lab.git
cd sphinx-prediction-lab

python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,pulse,corpus,research]"

python scripts/check_contracts.py
python -m pytest
ruff check .
mypy src tests scripts
```

For research-only environments:

```bash
python -m pip install -e ".[research,dev]"
```

## Documentation

- [Sphinx Trace S0 Architecture](docs/ARCHITECTURE.md)
- [Sphinx Trace Research Mandate](configs/trace/sphinx_trace_research_mandate_v1.json)
- [Sphinx Corpus](docs/CORPUS.md)
- [Data Sources](docs/DATA_SOURCES.md)
- [Evaluation Protocol](docs/EVALUATION_PROTOCOL.md)
- [Research Journal](docs/RESEARCH.md)
- [Polymarket Edge Bot Review](docs/POLYMARKET_EDGE_BOT_REVIEW.md)
- [Roadmap](docs/ROADMAP.md)
- [Sphinx Pulse Operations](deploy/pulse/README.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
