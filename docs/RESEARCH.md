# Sphinx Trace Research Journal

This ledger records every Sphinx Trace hypothesis, including rejected, invalidated
and unavailable results.

## Rules

Every hypothesis must record:

- stable `SPH-T-H###` identifier before execution;
- question and controlled change;
- dataset snapshot, schema and content hashes;
- chronological split and target embargo;
- seeds, training budget and checkpoint rule;
- frozen acceptance and falsification criteria;
- whether calibration, test or forward labels were opened;
- obtained result, decision and next action.

Valid statuses:

```text
design
registered
development
diagnostic
rejected
invalidated
promoted
result_unavailable
not_run
```

## SPH-T-H000: Sphinx Trace S0 Architecture

**Status:** `design`
**Registered:** 2026-07-16

**Question.** Can a point-in-time temporal graph model combine market state,
behavioral wallet history and wallet relationships to estimate fair probability and
executable net edge better than the same-window market probability?

**Architecture.** Causal market encoder, behavioral wallet encoder, heterogeneous
temporal graph attention, market-query cross-attention fusion and five calibrated
heads. The deterministic position manager owns execution.

**Expected evidence.** Improvement over market-only and graph-free controls on an
untouched chronological test, followed by positive paper-forward net edge under
executable fills.

**Falsification.** Reject S0 promotion if probability gains disappear after market
grouping, wallet information does not improve matched controls, graph attention
collapses, execution loses after costs, or the result depends on future wallet
performance.

**Result.** Not run. No checkpoint or performance claim exists.

**Next action.** Freeze Sphinx Chronicle v1 source boundaries and register the first
historical backfill hypothesis.
