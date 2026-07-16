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

## SPH-T-H001: Sphinx Corpus v1 Historical Backfill

**Status:** `development`
**Registered:** 2026-07-16

**Question.** Can a complete, restart-safe one-year market and public-trade
corpus be reconstructed without offset truncation and without treating technical
exchange fills as model-ready user trades?

**Registered window.** `[2025-07-16T00:00:00Z, 2026-07-16T00:00:00Z)`.

**Source boundary.** Gamma keyset pages define Atlas. Bounded Data API trade
windows define the primary Ledger. Polygon CLOB v1/v2 standard and Neg Risk
`OrderFilled` events are retained for sampled reconciliation and forensic
backfills. Hourly CLOB price history defines the first historical Depth tier.

**Acceptance.** Every selected market must have a completed receipt; no leaf
time window may remain saturated at the maximum offset; timestamps must remain
inside the registered market and corpus windows; normalized identifiers must be
stable across restart; sampled API trades must reconcile to chain transactions.

**Falsification.** Reject the snapshot if any pagination leaf is unresolved,
protocol migration coverage is missing, restart changes normalized row identity,
or chain reconciliation reveals unexplained systematic loss.

**Development evidence.** A six-point block-density probe across four exchange
contracts estimated approximately 1.45 billion technical order-fill events for
the year. This invalidated full duplicated chain JSON as the primary local
dataset. Real v1/v2 decoding produced 694 valid pilot rows with prices in
`[0.001, 0.999]`. The Data API accepted `limit=10000` but returned 1000 rows,
and rejected offsets above 3000 in the pilot. The registered implementation
therefore uses observed runtime limits and recursive bounded-window saturation
checks instead of trusting nominal parameter maxima.

A bounded Data API pilot completed one closed market with 74 public trades and
no saturated leaf. One sampled trade reconciled to a successful Polygon
transaction receipt containing three exchange logs; its public wallet address
was present in the indexed event topics. A high-volume market stopped cleanly at
the configured request budget and remained incomplete for restart, rather than
being misreported as a finished snapshot.

A two-market request-group pilot completed 653 rows in one bounded API request
with no gap. This validates batched low-volume discovery without weakening the
same leaf-saturation rule used for individual high-volume markets.

**Result.** In progress. No Chronicle snapshot is accepted yet.

**Next action.** Complete Atlas and a multi-market Ledger pilot on the local
training machine, reconcile sampled transactions, then start the restart-safe
yearly Ledger run.
