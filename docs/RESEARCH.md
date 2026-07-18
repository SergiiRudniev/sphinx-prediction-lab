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

## SPH-T-H002: Sphinx Corpus S0 Fast Ledger

**Status:** `qualified`
**Registered:** 2026-07-16

**Question.** Can a training-ready one-year Ledger be collected in approximately
three to four hours without discarding more than ten percent of observed cash
notional?

**Controlled change.** Reuse the `SPH-T-H001` Atlas and time window, select
markets with at least 25 USD reported volume, request public trades with the Data
API `CASH >= 25` filter, group low-volume markets up to 1 million USD reported
volume and run 24 workers behind a shared 18 request/second token bucket. The
unfiltered `SPH-T-H001` contract remains unchanged and uses a separate storage
namespace.

**Acceptance.** Every selected group must have a completed receipt, every raw
request must contain the registered cash filter, no leaf may remain saturated,
and stratified pilots must retain at least 90 percent of unfiltered cash notional.
Elapsed time is diagnostic and does not override completeness checks.

**Falsification.** Reject the fast tier if retained notional falls below 90
percent, parallel collection changes normalized identity, group receipts collide,
rate limiting causes persistent failure, or wallet coverage becomes unsuitable
for the registered S0 comparisons.

**Development evidence.** In a completed high-volume market window, the 25 USD
cash filter reduced 3,650 public rows to 925 while retaining 291,666.45 USD of
311,180.94 USD unfiltered notional (93.7 percent). The official Data API limit is
200 `/trades` requests per ten seconds; the registered collector stays below it
at 18 requests per second. A bounded concurrent pilot issued 100 requests in
6.15 seconds and received 97,361 response rows without a request failure. Every
saved raw request contained the registered cash filter. A closed-market pilot
completed eight filtered rows with no gap, and its immediate restart skipped the
completed group with zero network requests.

**Result.** Qualified as the S0 fast Ledger snapshot. The yearly run completed all
57,828 registered request groups spanning 1,189,164 selected markets and produced
176,119,673 normalized public-trade rows. Receipt audit found zero incomplete
groups, zero unresolved pagination gaps, zero duplicate condition assignments and
zero group budget mismatches. The exact row total independently reconstructed from
123,443 compressed Ledger partitions matches the group-receipt total.

The frozen corpus manifest covers 1,008,442 files, 27,853,609,624 source bytes and
181,945,996 normalized rows across all included namespaces. Its SHA-256 is
`be7d5c384fe6a5199e7e575f4dcf16dbbc36b6ee1d0b65de3f654c03eebbb4d5`.
A deterministic spread sample of 2,048 Ledger partitions parsed 2,914,572 rows,
18,549 condition IDs and 265,783 wallets. It found no missing required values,
duplicate trade IDs, malformed identifiers, invalid prices or sizes, inconsistent
notional arithmetic, out-of-window timestamps or within-file ordering violations.

The sample contains 6,757 rows (0.232 percent) whose normalized `size * price` is
between 21.4036 and 25 USD even though the request used `filterType=CASH` and
`filterAmount=25`. Every one has `size >= 25`; the current Data API documentation
requires both filter parameters but does not define the server-side CASH formula.
This is retained as a source-semantics caveat, not treated as corruption or silently
removed. One stale page-receipt `.tmp` from an interrupted attempt remains outside
the manifest; its group later completed successfully and its final receipt is valid.

**Next action.** Freeze the causal target schema and chronological split over this
snapshot, then run Sphinx Trace S0 Trial T0 before allocating the full token budget.

## SPH-T-H003: Sphinx Trace S0 Throughput Qualification

**Status:** `qualified`
**Registered:** 2026-07-17

**Question.** What end-to-end training throughput can a roughly 50 million
parameter Sphinx Trace S0-shaped model sustain on the local RTX 5070 when fed
fixed-shape tensors packed from real Sphinx Ledger rows?

**Controlled change.** Pack a deterministic subset of completed fast Ledger
partitions into fixed 224-token sequences containing 128 trade tokens, 32 wallet
tokens and 64 market/graph/time tokens. Benchmark a ten-layer, 640-width,
ten-head SDPA backbone with BF16 autocast, fused AdamW, fixed shapes, pinned
prefetch and optional `torch.compile`. Keep the source cutoff and benchmark
configuration in the run receipt.

**Acceptance.** The run must report the exact parameter count, source paths and
row count, packed sequence count, measured rather than assumed token throughput,
step latency, peak allocated and reserved VRAM, PyTorch/CUDA versions, compile
status and extrapolated wall times. At least 100 measured optimizer steps must
follow warmup, unless a recorded platform limitation prevents the configured
path from running.

**Falsification.** Do not use the result if synthetic tensors replace the real
Ledger loader, dynamic input shapes cause recompilation during measurement,
the model is outside 48–52 million parameters, fewer than 100 optimizer steps
are measured, or CUDA memory/compile failures are omitted from the receipt.

**Evidence boundary.** This qualification measures engineering throughput only.
Its proxy targets and loss are not model-quality, trading, calibration or
promotion evidence.

**Result.** Qualified as engineering evidence. The deterministic pack contains
8,192 fixed 224-token sequences built from 564,534 real normalized Ledger rows
across 366 used source files. The measured model has 50,213,128 parameters.
Preflight eager runs sustained 97,230 tokens/second at batch 16, 97,714 at
batch 32 and 92,170 at batch 64. Batch 32 with BF16, Flash SDPA, fused AdamW
and active `torch.compile`/Triton sustained 128,320 tokens/second across 1,000
measured optimizer steps and 7,168,000 tokens. Mean, p50 and p95 step times were
55.57, 55.45 and 56.21 milliseconds. Steady device usage was 4.15 GB including
the CUDA context; PyTorch reserved 2.84 GB. A cold compile took 53.70 seconds,
while the cached final run compiled and executed its first step in 2.93 seconds.

At measured core-loop throughput, 1.0, 2.5 and 4.0 billion tokens require 2.16,
5.41 and 8.66 hours respectively. These estimates exclude validation,
checkpointing, final target construction and future architectural changes. The
proxy loss decrease is not evidence of predictive or trading quality.

**Next action.** Freeze the causal target schema and temporal train/validation/
test split, then run S0 Trial T0 before committing the full token budget.

## SPH-T-H004: Fast Ledger Worker Scaling Qualification

**Status:** `qualified`
**Registered:** 2026-07-17

**Question.** Can the restart-safe fast Ledger sustain approximately 15
successful Data API responses per second on the local collection machine?

**Controlled change.** Resume the unchanged `SPH-T-H002` Atlas, market groups,
18 request/second global gate and storage namespace with 56 workers instead of
24. Preserve every completed receipt and do not change filtering, normalization
or recursive completeness rules.

**Acceptance.** The resumed process must expose the configured worker pool,
keep the error log empty, remain below 4 GB resident memory and sustain at least
12 successful newly persisted raw responses per second across two consecutive
30-second measurements. Completed group receipts must be monotonic across the
restart.

**Falsification.** Reject 56 workers if successful persistence remains below 10
responses per second, memory exceeds 4 GB, the process reports an unhandled
error, completed receipts regress or rate limiting causes persistent failure.

**Result.** Qualified as an operational throughput change. The restart preserved
all 38,439 completed group receipts. The resumed process exposed 56 established
HTTPS connections, remained error-free and used at most 1.34 GB resident memory
during qualification. A first post-warmup interval persisted 10.93 successful
raw responses per second; the next two consecutive intervals sustained 13.00
and 12.08 responses per second. The accepted intervals averaged 12.54 responses
per second, compared with 6.79 across the two 24-worker preflight intervals.
Completed-group throughput rose from 39-49 to 76-86 groups per minute during
the sampled windows. The measured path improved materially but did not sustain
the nominal 15-response target.

**Next action.** Keep the 56-worker, 18 request/second run active and use the
fixed 57,828-group denominator for completion reporting.

## SPH-T-H005: Sphinx Trace S0 Trial T0 Target Contract

**Status:** `qualified`
**Registered:** 2026-07-17

**Question.** Can the qualified fast Ledger and terminal binary Atlas markets
produce a causal, event-grouped target index for the first Sphinx Trace S0 trial
without opening the untouched test labels?

**Controlled change.** Restrict the first target contract to resolved two-outcome
markets whose ordered outcomes are exactly `Yes` and `No` and whose terminal
prices form a one-hot result. Normalize every Ledger trade price into the
equivalent YES probability. Anchor a decision one second after an observed trade,
use only events strictly before that decision as features, and derive future
trade-price markouts at 5 minutes, 1 hour and 1 day. Treat these markouts and the
fixed-cost edge as development proxies rather than executable-price evidence.

Assign the entire event group to one chronological segment using the latest
resolution time among its eligible markets. Keep only decisions inside that same
segment, require each markout observation to remain before the segment end and
leave seven-day embargoes between train, validation, calibration and test. The
test segment remains label-withheld during development.

**Acceptance.** Every emitted row must have `feature_max_event_time < decision_time`;
event IDs must be disjoint across segments; no decision or target observation may
fall in an embargo; future observations must satisfy the registered horizon and
tolerance; terminal resolution labels must be binary; restart must preserve row
identity and split assignment; no test target value or test label statistic may be
written by the development builder.

**Falsification.** Reject the contract if any event group crosses segments,
same-second ordering enters the future feature set, target lookup crosses a split
boundary, outcome order is inferred from display text, test labels are exposed,
or the pilot requires current wallet aggregates that cannot be reconstructed at
the decision time.

**Evidence boundary.** Historical public trade prices are not executable bid/ask
or L2 depth. Trial T0 may qualify target construction and model learning behavior,
but cannot establish executable net edge or trading performance.

**Result.** Qualified as the Trial T0 target and split contract. Atlas contained
547,220 eligible terminal binary `Yes/No` markets across 82,939 event groups after
excluding malformed resolution times and markets without exactly one event ID.
The frozen event-resolution split assigns 19,890 groups to train, 16,185 to
validation, 14,205 to calibration and 24,628 to the still-unopened test; 8,031
groups fall in embargoes or outside the registered one-year window.

The bounded real-data pilot uniformly selected 1,024 of 123,443 Ledger partitions
and read 1,477,551 public trades. It emitted 3,212 target rows: 2,128 train, 773
validation and 311 calibration examples across 359 event groups. No event crossed
segments, no feature timestamp reached or exceeded its decision timestamp, no
markout violated its registered horizon or lag, and every resolution-edge and
fixed-cost proxy recomputed exactly. The development builder withheld 153,762
source rows assigned to test and emitted zero test rows or label statistics.

An immediate restart reproduced identical compressed output hashes for all three
development splits. Five-minute, one-hour and one-day markouts were available for
2,509, 2,955 and 1,097 examples respectively. This confirms the contract and also
shows that the one-day public-trade target is materially sparser than shorter
horizons. No model has been trained and no performance claim exists.

**Next action.** Build the full development Chronicle feature pack with causal
market and wallet histories, register Trial T0 training gates and run the first
train/validation/calibration-only learning test without opening historical test.

## SPH-T-H006: Sphinx Trace S0 Trial T0 Learning Preflight

**Status:** `qualified learning preflight; no promotion`
**Registered:** 2026-07-17

**Question.** Can the 50 million parameter Sphinx Trace S0-shaped backbone learn
the registered multi-head Trial T0 targets from causal market and wallet histories
without consuming test rows or relying on raw wallet identity?

**Controlled change.** Pack the qualified 3,212-row `SPH-T-H005` development
target index into 224-token sequences: 128 causal market-trade tokens, 32 causal
wallet-history tokens and 64 market context tokens. Wallet history may use all
events in the same 1,024-partition deterministic source sample strictly before
the decision, but raw wallet identifiers and future resolved performance are
excluded. Train the 48-52 million parameter throughput-qualified backbone with
eight registered outputs, masked continuous labels and a binary resolution head.

Use train for optimization, validation for checkpoint selection and calibration
only for a single frozen probability temperature. Do not emit, load or summarize
test rows. Compare the probability head with the decision-time market probability
and compare markout heads with a zero-change baseline.

**Acceptance.** The pack must report zero feature-time violations, zero cross-split
event overlap, zero consumed test rows, exact target hashes and deterministic row
identity. The model must remain inside 48-52 million trainable parameters, all
losses must remain finite, train loss must decline and validation plus calibration
baseline comparisons must be recorded. This preflight cannot promote the model
regardless of metric direction.

**Falsification.** Invalidate the run if any feature timestamp reaches the decision
time, raw wallet identity enters a feature, missing labels contribute to loss,
checkpoint selection observes calibration or test, event groups cross segments,
or a failed/negative learning result is omitted.

**Evidence boundary.** The wallet history covers a deterministic development
sample rather than the complete Ledger. Public-trade markouts and fixed costs are
not executable fills. This run tests the pipeline and learning behavior only.

**Result.** Qualified as pipeline and learning-behavior evidence, not as model
promotion evidence. The pack reproduced all 3,212 registered development rows
(2,128 train, 773 validation and 311 calibration) from 1,477,551 Ledger trades.
It indexed 172,762 wallet histories and reported zero missing histories, feature-
time violations, event overlaps, raw wallet identity features or consumed test
rows. Test labels remain unopened.

The measured backbone has 50,213,128 parameters. BF16 training with fused AdamW
and active `torch.compile` selected epoch 4 by validation loss and stopped after
epoch 7. Train loss fell from 0.702622 to 0.260836; the best validation loss was
0.412560. The final successful run took 71.36 seconds and PyTorch reported 2.70
GB peak allocated and 2.90 GB peak reserved memory. Windows used
`max-autotune-no-cudagraphs` after the registered CUDA Graph mode failed during
an earlier diagnostic attempt; the effective mode and failure are not hidden.

The resolution head showed a real development signal. On validation, Brier score
was 0.121416 versus 0.133705 for the decision-time market price, and log loss was
0.365398 versus 0.400153. After fitting one temperature on calibration, its
in-sample calibration Brier was 0.129079 versus 0.136632 for market and log loss
was 0.382724 versus 0.403911. This does not establish out-of-sample test gain.

The multi-task result is negative: none of the seven markout or net-edge heads
beat the registered zero-change baseline on validation or calibration. Validation
also worsened after epoch 4 while train loss continued falling. The checkpoint is
therefore retained only as a diagnostic artifact and cannot drive calls, paper
orders or live orders.

**Next action.** Register an ablation that separates the resolution objective
from markout/edge objectives, adds market-only baselines and fixes target scaling
and imbalance before expanding Chronicle coverage. Keep the untouched historical
test closed.

## SPH-T-H007: Terminal-Outcome Wallet-History Ablation

**Status:** `qualified ablation; wallet signal inconclusive; no promotion`
**Registered:** 2026-07-17

**Amended before training:** 2026-07-17. The product objective was clarified as
contract selection followed by holding to terminal resolution. No H007 run had
started. Short-horizon markout prediction was removed from the experiment before
observing any H007 result.

**Question.** Does causal cross-market wallet history add terminal-outcome signal
beyond the same within-market flow and context?

**Controlled change.** Reuse the frozen H006 feature pack, backbone, seed,
optimizer, temporal splits and validation checkpoint rule. Train three matched
variants: resolution without the 32 wallet-history tokens, resolution with the
original causal wallet-history tokens and resolution with wallet tokens taken
from the latest non-future example belonging to a different event. Every variant
has one terminal `resolved_yes` output. The market and context tokens remain
unchanged, so this isolates incremental cross-market wallet history rather than
every within-market participant-flow statistic.

The prior-event control may only use a donor whose decision time is no later than
the recipient and whose event ID differs. The earliest rows without an eligible
donor receive zero wallet tokens. No raw wallet identifier or test row may enter
any variant.

**Acceptance.** Every variant must satisfy the H006 causal and finite-training
gates, consume zero test rows and save hashed validation predictions. Backbone
parameter counts may differ by at most 5,000 because the resolution-only output
layer is narrower. The control must report zero future-time and same-event donor
violations. Compare uncalibrated validation log loss as the primary metric using
5,000 deterministic event-group bootstrap samples. Support incremental wallet
signal only if the upper 95% delta bound is below zero against both the no-wallet
and prior-event controls. Promotion remains forbidden regardless of direction.

**Falsification.** Reject the comparison if example order differs between runs,
the control draws from a later decision or the same event, a variant changes the
backbone or training schedule, row-level bootstrap replaces event-group sampling,
predictions are not bound by hash, or any test label is opened.

Price is an input cost, not a forecast target. For descriptive contract-selection
diagnostics, rank the fixed validation rows by the absolute gap between terminal
probability and the decision-time YES price, choose YES for a positive gap and NO
otherwise, and compute terminal share PnL at fixed top-score fractions. These
public-trade prices exclude executable bid/ask, fees, depth and slippage, so the
diagnostic cannot qualify a betting policy.

**Evidence boundary.** This is a bounded development ablation over sampled public
trades. It cannot establish executable edge, hidden-insider detection, full-corpus
generalization or trading performance.

**Result.** Qualified as a controlled negative/inconclusive development result.
All three 50,208,641-parameter runs completed with finite declining train loss,
identical validation row order, hashed predictions and zero consumed test rows.
The prior-event control used 2,122 of 2,128 train rows, 772 of 773 validation
rows and 306 of 311 calibration rows as causal donors, with zero future-time or
same-event violations.

On 773 validation examples across 96 event groups, the decision-time market
baseline had Brier 0.133705 and log loss 0.400153. Resolution without dedicated
wallet-history tokens reached Brier 0.121999 and log loss 0.393260. Original
causal wallet history reached Brier 0.127541 and log loss 0.392836. The causal
prior-event wallet control reached Brier 0.130432 and log loss 0.394623.

The causal-wallet point estimate improved log loss by only 0.000424 against no
wallet history and 0.001786 against the prior-event control. The registered
5,000-sample event-group bootstrap intervals were [-0.082541, 0.077149] and
[-0.108588, 0.094235]. Both cross zero by a wide margin, so the registered
wallet-signal criterion failed. This pilot does not demonstrate incremental
cross-market wallet value.

The descriptive hold-to-resolution ranking is unstable. The causal-wallet model
lost 0.033 shares before costs in its top 1% and gained 4.221 shares on 31.779
public-price cost in its top 10%. Its top 25% gained 25.339 shares on 95.661 cost,
but the no-wallet model produced a larger top-25% diagnostic. These validation-
only figures omit executable bid/ask, depth, fees and slippage and do not qualify
thresholds, stake sizing or a betting policy.

**Next action.** Build the full outcome-only Chronicle from the complete Ledger
instead of the 1,024-file pilot, increase independent event coverage and preserve
the same wallet/no-wallet/prior-event controls. Only then decide whether wallet
history belongs in full S0 training. Keep the historical test closed.

## SPH-T-H008: Full-Universe Stateful Research Mandate

**Status:** `registered design mandate; no performance evidence`
**Registered:** 2026-07-17

**Objective.** Maximize net profit after executable costs by selecting a concrete
Polymarket event outcome and learned position size, or by learning to `SKIP`.
Compute cost, training time and minimal parameter count are explicitly not model
objectives. Wrong CALLs are more costly than missed opportunities, while the
selected policy must retain useful call frequency.

**Scope decision.** Include binary, multi-outcome and neg-risk structures across
all categories and resolution horizons. Exclude only invalid, cancelled,
ambiguous or non-replayable records. Permit market question/rules semantics but
exclude external news. Require point-in-time Polygon funding/transfer graphs and
permit other-protocol wallet activity with reproducible provenance.

**Architecture decision.** Remove the participant cap through streaming wallet
memory and chunked latent aggregation. Build the hierarchy `wallet -> market ->
event -> graph -> universe -> opportunity ranker`, conditioned by separate
Prediction and Position Books. Preserve the complete prediction trajectory. A
CALL may update, change side or be cancelled; prior predictions remain internal
state and cannot become self-confirming market evidence.

New and dust wallets remain visible through novelty, uncertainty, robust
aggregation and manipulation features. `SKIP`, outcome selection and balance-
conditioned sizing are learned. Analytical thresholds, category exposure caps and
position-count caps are not substituted for the model in research simulation.
The replay engine still enforces cash, point-in-time liquidity, fees, latency,
slippage, partial fills and data integrity as physical constraints.

**Training decision.** Allow semantic, wallet, graph and market self-supervised
pretraining; terminal-outcome supervision; cross-sectional ranking; and stateful
policy learning in a full Polymarket simulator. Compare approximately 50M, 100M
and 150M candidates and ensembles. Select by locked evidence rather than compute
cost. Any run longer than two hours must checkpoint complete training and
simulator state at least every 15 minutes and support graceful exact resume.

**Debugging decision.** Treat prediction debugging as a first-class deliverable.
Every CALL or `SKIP` must replay from hashed inputs and expose the observed market,
wallet, graph, semantic, universe and position state; prediction changes; top
positive, negative and conflicting evidence; graph paths; and counterfactual
with/without-module results. Attention weights alone are not explanations.

**Evaluation decision.** Primary metric is simulator net profit after executable
costs. Run every model and baseline through the same simulator. Historical
promotion requires at least 1,000 calls across 1,000 independent events, a
positive lower 95% block-bootstrap net-profit bound and improvement over the
strongest registered baseline. Target at least three median calls per week. Keep
test labels closed until model, policy, simulator, calibration and source hashes
are locked. Subsequent production consideration requires at least 90 days and 100
positive-net-profit paper-forward calls.

**Evidence boundary.** This is a registered research and architecture mandate,
not evidence that any model is profitable, detects insiders or is ready for test,
paper execution or production.

**Next action.** Freeze the H008 full Chronicle episode and simulator schemas,
then profile the complete historical source coverage before choosing the first
50M/100M/150M pretraining and outcome-training campaign.

## SPH-T-H009: Full Outcome Chronicle

**Status:** `full Ledger structurally qualified; Polygon graph pending`
**Registered:** 2026-07-17

**Question.** Can the complete qualified Atlas and Fast Ledger be transformed
into a causal, restartable full-universe outcome corpus that preserves every
public trade and every participant, represents multi-market and neg-risk events,
keeps the historical test terminal labels physically closed and admits a
point-in-time Polygon funding and transfer graph?

**Controlled construction.** Treat a Gamma market as a binary condition and
join markets through shared event IDs into connected event components. Assign an
entire component to one chronological segment using its latest market close;
never split linked markets across train, validation, calibration or test. Retain
all valid Ledger rows in one globally event-time-ordered daily stream. Build a
reconstructable decision cursor at early powers of two, every 128 later component
trades and after six hours without a decision. This cursor is an index, not a
trade or wallet sampling rule.

The collected Atlas snapshot is provenance, not historical state. Mutable
snapshot values, including current prices, liquidity and post-collection
metadata, are masked from causal features. Terminal payout vectors may be opened
for train, validation and calibration only after component assignment. The
development builder must decide that a component is test before reading its
terminal price field. Test rows may exist in the unlabeled catalog and trade
stream, but their terminal values may not be read, emitted or summarized.

Collect Polygon ERC-20 collateral and ERC-1155 conditional-token transfers for
all unique Ledger participants with block timestamps and transaction/log
provenance. The graph is mandatory for full qualification. A structural pilot may
carry an explicit unavailable mask when an archive RPC is absent, but this cannot
be reported as a completed graph or a fully qualified Chronicle.

**Registered sources.** Atlas contains 1,718,409 markets, 671,186 events and
3,436,728 tokens. The qualified Fast Ledger contains 176,119,673 public trades
covering 1,189,164 markets in 57,828 complete scope groups and 123,443 compressed
files. Exact source and receipt hashes are frozen in
`configs/corpus/sphinx_chronicle_h009_v1.json`.

**Acceptance.** Preserve exactly 176,119,673 Ledger rows with stable identity and
global ordering; retain an uncapped participant set; produce linked multi-market
and neg-risk episodes; report zero component overlap, future-feature violations
and test-terminal-field access; reproduce completed artifact hashes after a
restart; checkpoint at atomic scope-run or daily-shard boundaries no more than 15
minutes apart; and complete the registered Polygon graph backfill before full
qualification.

**Falsification.** Reject the build if a wallet or trade is sampled away, a
connected component crosses splits, current Atlas market state is treated as a
historical feature, any test payout is accessed during development, source rows
are lost or duplicated, restart changes an artifact, a missing graph channel is
silently represented as zero activity, or an incomplete build is described as
model or profitability evidence.

**Full Ledger result.** The completed global merge contains exactly 176,119,673
rows across 365 daily shards and 1,368,360 unique participant wallets. The
adaptive cursor emitted 4,483,489 post-trade component decisions. The catalog
contains 1,718,409 markets and 671,192 connected components, including 111,955
multi-market and 40,501 neg-risk components. The full deep-hash validation read
the catalog, episodes, participant index, all stream shards and all decision
shards in 602.02 seconds; every hash matched, all causal/label checks passed and
the violation list was empty. Test terminal labels remained zero.

This qualifies the full Ledger replay structure, not all of H009. The receipt is
deliberately `fully_qualified=false` because the required Polygon transfer graph
is not complete. Decision rows therefore carry graph availability as false.

**Evidence boundary.** H009 qualifies data structure and causal replay only. It
does not show that wallet flow predicts outcomes, identify an insider, establish
executable fills or costs, open historical test, train a model or demonstrate
profit.

**Next action.** Complete the indexed Polygon transfer backfill and rerun graph-
required qualification. Model work may proceed with the graph channel explicitly
masked, but cannot claim the temporal-graph variant.

## SPH-T-H010: Stateful Polymarket Simulator

**Status:** `registered; mechanics implemented; corpus integration pending`

**Registered:** 2026-07-17, before simulator or model-profit results were
observed.

**Question.** Can every Sphinx candidate and baseline be trained and compared in
one causal replay system that carries prediction memory, orders, fills, cash,
positions and resolutions without inventing executable liquidity?

**Frozen mechanics.** The simulator accepts learned CALL, update, cancel, hold,
reduce, close and `SKIP` decisions. Its deterministic boundary enforces available
cash and shares, latency, marketable limits, expiry, adverse price movement,
fees, partial fills, shared liquidity consumption and terminal payouts. There is
no strategy position, category or correlation cap. Those choices remain model
outputs; only physical impossibilities are rejected.

Prediction and position state are separate. Each prediction binds to a hashed
causal input. Orders cannot fill from the decision's own evidence trade, inputs
must be globally non-decreasing in event time and a liquidity event cannot be
consumed twice. The full state has a deterministic checkpoint hash and restores
orders, positions, fills, processed liquidity, marks, cash, PnL and prediction
memory exactly.

**Two evidence tiers.** The development replay uses later public trades as a
conservative liquidity proxy with a registered latency, duplication haircut,
participation fraction, adverse tick, fee and cost stress. It never imputes a
fill when no eligible later trade exists. This proxy is useful for policy
iteration but cannot qualify executable historical profit. Simulator
qualification requires point-in-time orderbook depth from Sphinx Depth or an
equivalent archive, with real level consumption and missing depth causing order
rejection.

**Initial result.** Unit replay covers delayed and partial buying, self-fill
prevention, cash rejection, share reservation, selling, fee and cost-basis
accounting, expiry, duplicate-liquidity rejection, resolution and exact
checkpoint restoration. This is an engineering result only. H009 stream and
terminal labels have not yet been integrated into a full H010 run.

**Next action.** Finish the complete H009 stream, construct event-time simulator
episodes, register market and wallet baselines, and run development-only
trade-tape replay before any model campaign. Keep test labels closed.

**Evidence boundary.** H010 mechanics do not establish historical depth,
executable fills, model quality, profit, untouched-test, paper-forward or
production evidence.

## SPH-T-H011: Uncapped Causal Flow Campaign

**Status:** `full causal pack verified; first 50M outcome training in progress`

**Registered:** 2026-07-17, before full-corpus model metrics were observed.

**Question.** Does processing every causal trade and participant improve
terminal outcome selection and development simulator profit beyond the current
market probability? Separate contemporaneous wallet-flow value from prior
resolved wallet performance and temporal graph value.

**Controlled variants.** Train market-only, uncapped wallet-flow, causal prior
resolved wallet-performance and temporal-graph variants on identical H009
decisions. Do not learn raw wallet identity. Every valid trade updates compact
wallet, market and event state; no wallet, trade, market or outcome-count cap is
allowed. Apply terminal wallet-performance updates only after the registered
public resolution time. All supervised rows remain grouped by connected event
component and test terminal fields remain unopened.

The feature pack retains every decision and uses component/lifecycle-balanced
loss weights rather than deleting frequent-event rows. Model components are a
streaming wallet encoder, linear-cost wallet-to-market latent aggregator,
multi-scale market memory, linked-market event encoder, chunked universe memory,
opportunity ranker and probability, uncertainty, sufficiency and sizing heads.
Candidate capacities are approximately 50M, 100M and 150M parameters, beginning
with 50M. Compute cost is not a selection criterion.

**Acceptance.** A wallet feature is supported only when the paired
component-block-bootstrap upper 95% log-loss delta is below zero against
market-only and its H010 simulator profit delta is positive. Profit must survive
the registered cost stress. The first model campaign cannot open test.

**Implemented causal pack.** The registered pack processes the full H009 stream
with no wallet, market or trade cap. Its 128 named features contain clock,
evidence-market, connected-component, evidence-wallet, all-wallet recurrent
DeepSet and universe blocks. Every trade updates state; raw wallet identity is a
state key and never a learned input. Approximate cardinality uses an uncapped
HyperLogLog update for every row. Decision materialization requires exact
`stream_row` and evidence-trade-ID agreement. Daily tensors, debug provenance,
train-only median/IQR statistics and the complete recurrent state have atomic,
source-bound resume receipts.

A separate resolution ledger computes winner-oriented directional edge and an
observed-trade PnL proxy per wallet/market, excludes post-resolution trades and
applies each update strictly after its public resolution second. This proxy is
explicitly incomplete because the 25 USD source filter and token transfers can
hide inventory. It is useful as a causal feature, not exact realized PnL.

The inspectable group-latent backbone has terminal-outcome, uncertainty, CALL
sufficiency, expected-edge and beta sizing heads. Exact candidate sizes are
50,248,198, 99,789,446 and 158,990,598 parameters. Debug mode records per-layer
group/latent attention; integrated-gradient and module-ablation inputs remain
available because all 128 features are named. Attention alone is not treated as
an explanation.

**Compute preflight.** On the RTX 5070, the 50M bf16 forward/backward benchmark
processed 6,446 examples/s at batch 512 with 3.53 GiB peak allocated VRAM. Batch
1024 processed 6,309 examples/s at 6.18 GiB. Batch 2048 reached the 12 GiB memory
cliff, falling to 498 examples/s at 11.60 GiB. The first run therefore uses batch
512. These are synthetic compute measurements, not full-pack training time or
model quality.

**Determinism preflight.** A full-scale one-day build allocated all 1,368,360
wallet states, 1,189,164 market states and 530,974 component states, processed
69,380 source trades and materialized 4,256 exact decisions without non-finite
features or test labels. Replaying the same build after an atomic checkpoint
produced the same completed-output hashes. The repeated run finished in 71.47
seconds and reported `output_hashes_match_previous=true`. This validates the
builder and resume boundary, not predictive quality.

**Source-price anomaly rule.** Before full-pack or model results were observed,
the strict replay encountered a structurally complete public trade whose source
price was 1.1140588235. The Ledger remains immutable and the row, size and
notional remain in recurrent state. For model probability fields only, finite
source prices outside `[0, 1]` are clamped to that physical interval and counted
per day; no trade is dropped. Non-finite prices and structurally invalid rows
still fail the build. Daily receipts and checkpoints are now bound to a digest
of the builder, feature, kernel and source-index implementations so a resume
cannot mix code versions. This rule was frozen before training metrics.

**Full feature-pack result.** The resolution-backed build processed all 365
H009 days and exactly 176,119,673 Ledger rows into 4,483,489 causal decision
examples. It retained all 1,368,360 wallet states, 1,189,164 market states and
530,974 component states without a hard wallet, market or trade cap. The build
reported zero non-finite features, zero dropped trades and eight finite source-
price anomalies handled by the registered clamp-and-count rule. Test labels
remained physically unopened and the resolved-wallet schedule was complete and
unmasked.

An exact cached replay reproduced the recurrent checkpoint, train-only
normalization and every daily feature-shard digest. The manifest reports
`previous_comparable_manifest_found=true` and
`output_hashes_match_previous=true`; its checkpoint digest is
`8a8edd3d267b46a0f17addb7c50ac6e5947a7b692fc05f79423021c393b12a96`
and its daily-receipt digest is
`e87f7fe5eeb120e863d720fc43e9e49dda4c34da3939ae51942d95dd06af8239`.
The first complete build took 1,155.72 seconds and the verification replay took
452.72 seconds. These receipts qualify the training input, not model quality or
profit.

**Resolved-performance result.** The complete causal ledger read exactly
141,033,306 selected source trades across 51,683 scope groups, excluded 363,279
post-resolution rows and emitted 30,058,318 wallet/market resolution updates in
310 daily shards. A cached full replay finished in 77.32 seconds and reproduced
the same shard-receipt digest
`124b6091b2fa0e2604fa0b54637966d22b1b7d86282a369819ebe088c8a5084f`.
The manifest is valid and complete with test terminal access remaining false.

The public CryptoHouse actor source exposed a shared 120-query/hour quota and a
976.56 KiB response limit. The collector begins with 32 hash partitions,
preserves every completed partition and deterministically splits only oversized
buckets into disjoint 64-way leaves, recursively if required. Result-limit
failures are no longer retried unchanged. The collector recognizes the quota
reset time, checkpoints the adaptive leaf plan and waits without discarding
completed actors. This changes query shape only; no actor is capped or sampled.

**Next action.** Complete the active 50M market-only run, then train the matched
uncapped-wallet-flow and causal resolved-performance variants on the identical
verified pack. Finish the quota-aware actor context in parallel and add its
matched variant; keep the Polygon graph variant blocked until its source is
complete.

**Evidence boundary.** H011 validation and conservative trade-tape replay cannot
establish untouched-test, historical orderbook depth, executable profit,
paper-forward performance, insider attribution or production readiness.

## SPH-T-H012: Portfolio-Aware Selective Policy

**Status:** `registered; implementation pending H011 outcome evidence`

**Registered:** 2026-07-17, before full H011 outcome or profit metrics were
observed.

**Question.** Can one learned, portfolio-aware policy turn calibrated H011
outcome state into profitable `CALL_YES`, `CALL_NO` and `SKIP` decisions, revise
earlier calls and size multiple concurrent positions without fixed confidence,
edge, category or position limits?

**Controlled construction.** Warm-start the selected H011 backbone, add causal
portfolio and prior-prediction state, and train categorical action plus beta-size
outputs against the exact H010 development replay. The model sees current cash
and equity. The only hard rejections are physical cash, share, time and observed-
liquidity constraints. There is no call-frequency bonus, skip penalty, fixed
threshold or fixed bet size. Reward is the change in net liquidation value after
registered latency, fill, adverse-price and fee costs.

Fit policy only on the early component-time block of validation, select on its
later disjoint block and use calibration as the final development audit. Keep
test physically closed. All model and baseline candidates must consume identical
liquidity events through H010.

**Acceptance.** Promotion requires positive calibration net profit under cost
stress, a positive component-block-bootstrap lower 95% profit bound and
outperformance of both current market probability and the strongest learned
baseline. CALL frequency is reported, not hardcoded; the policy must earn useful
weekly profit by choosing when evidence is sufficient.

**Next action.** Complete the H011 outcome ablations, implement the H009/H010
event-time adapter and train the first portfolio-aware selective policy without
opening test.

**Evidence boundary.** Trade-tape development profit cannot establish historical
orderbook executability, untouched-test performance, paper-forward profit,
insider attribution or production safety.
