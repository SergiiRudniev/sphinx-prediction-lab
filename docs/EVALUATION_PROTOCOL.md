# Evaluation Protocol

Sphinx Trace is evaluated as a causal decision system, not as a random-row
classification benchmark.

## Split Discipline

The first frozen campaign must contain four chronological segments:

| Segment | Allowed use |
| --- | --- |
| Train | Model fitting and train-only normalization |
| Validation | Checkpoint and architecture selection |
| Calibration | Threshold, probability and deterministic policy selection |
| Test | Open exactly once after hashes are locked |

Markets belonging to the same event must not cross segments. Targets that extend
past a boundary are purged or embargoed.

## Baselines

S0 must be compared with:

- current executable market probability;
- market-only model without wallet inputs;
- wallet features without graph message passing;
- graph model without cross-attention;
- deterministic follow-flow and no-trade policies.

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
