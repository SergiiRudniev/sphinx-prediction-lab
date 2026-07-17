# Evaluation Protocol

Sphinx Trace is evaluated as a causal decision system, not as a random-row
classification benchmark.

## Split Discipline

The first frozen campaign must contain four chronological segments:

| Segment | Allowed use |
| --- | --- |
| Train | Model fitting and train-only normalization |
| Validation | Checkpoint and architecture selection |
| Calibration | Probability, uncertainty and selective-policy calibration |
| Test | Open exactly once after hashes are locked |

Markets belonging to the same event must not cross segments. Targets that extend
past a boundary are purged or embargoed.

### Trial T0 frozen split

`SPH-T-H005` assigns an event group by the latest eligible market resolution time.
Only decisions inside the assigned segment are retained, and every markout must
also be observed before that segment ends.

| Segment | UTC interval | Development access |
| --- | --- | --- |
| Train | `[2025-07-16, 2026-02-16)` | Features and labels |
| Embargo | `[2026-02-16, 2026-02-23)` | Purged |
| Validation | `[2026-02-23, 2026-04-20)` | Features and labels |
| Embargo | `[2026-04-20, 2026-04-27)` | Purged |
| Calibration | `[2026-04-27, 2026-05-25)` | Features and labels |
| Embargo | `[2026-05-25, 2026-06-01)` | Purged |
| Test | `[2026-06-01, 2026-07-16)` | Labels remain unopened |

The development builder does not emit test rows. Opening test requires locked model,
feature, checkpoint, calibration and deterministic-policy hashes.

## Baselines

S0 must be compared with:

- current executable market probability;
- market-only model without wallet inputs;
- wallet features without graph message passing;
- graph model without cross-attention;
- deterministic follow-flow and no-trade policies;
- wallet-only and prior-event wallet-control models;
- frequency-matched random and simple fractional-Kelly policies.

Capacity and feature ablations use matched seeds and identical rows.

## Metrics

### Probability

- Brier score and log loss against resolved outcomes;
- calibration error and reliability by probability band;
- paired improvement over executable market probability.

### Wallet and flow

- post-trade markout at fixed horizons;
- resolved net edge with Bayesian shrinkage;
- breadth across wallets, events and categories;
- cluster and single-market concentration.

### Trading

- net PnL under bid/ask, depth, latency, fees and slippage;
- profit factor, drawdown and break-even cost buffer;
- turnover, fill rate and rejected-order rate;
- block-bootstrap confidence interval;
- results after additional cost and latency stress.

## Promotion Sequence

```text
Historical development
→ untouched historical test
→ live calls
→ paper execution
→ user-confirmed execution
→ bounded automation
```

Skipping a stage invalidates promotion. Emergency limits and jurisdiction checks
remain deterministic at every stage.

## Required S0 Gates

Exact numeric thresholds are registered before training. At minimum, promotion
requires:

- zero detected causal leakage;
- probability improvement over the same-window market baseline;
- positive net edge after costs on untouched data;
- positive lower confidence bound under block bootstrap;
- breadth across independent markets and wallets;
- positive paper-forward result under executable fills;
- no threshold or model selection using test or forward labels.

Until these gates are frozen and passed, repository status remains `design` or
`development`, never `released`.

## H008 Stateful Simulator Mandate

`SPH-T-H008` makes simulator net profit after executable costs the primary system
metric. Every learned policy and baseline must use the same point-in-time calls,
orderbook liquidity, balance, orders, fills, fees, latency, slippage, partial
fills, positions and resolutions. `SKIP` and position size are learned decisions;
cash, liquidity and data-integrity enforcement are physical simulator rules.

Historical promotion requires at least 1,000 calls across 1,000 independent
events, a positive lower 95% block-bootstrap net-profit bound, improvement over
the strongest registered baseline and a target median frequency of at least three
calls per week. These gates do not authorize test opening or production by
themselves. Model, policy, simulator, calibration and source hashes must first be
locked. Production consideration remains downstream of at least 90 paper-forward
days, 100 calls and positive net profit after all costs.
