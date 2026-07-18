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

**Status:** `bounded adapter and audit sink implemented; full replay pending`

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

**H009 event adapter.** The bounded adapter consumes an exact source-bound
`shard_ordinal/row_ordinal` cursor, applies public liquidity first and only then
applies calls whose evidence-trade ID, timestamp and condition match that row.
It maps the learned outcome index to the actual catalog token, derives the
binary complement reference without inventing a fill, cancels stale pending
orders, and supports call, update, hold, reduce, close and resolution actions.
Portfolio and prediction-memory tensors are derived from the same simulator
state used for fills.

Full-tape mode no longer retains one Python object for every irrelevant trade or
one equity point for every unchanged portfolio mark. It keeps a monotonic source
cursor, counts consumed liquidity, retains marks only for exposed or fillable
tokens and records equity only when portfolio value can change. Prediction and
liquidity retention remain available for small audit runs. Exact adapter plus
simulator restoration is hash-stable in unit replay. Active orders are indexed
by token and condition and expire through a deterministic priority queue, so one
new public trade no longer scans the complete historical order ledger. The
append-only disk audit
stores compact decision, order, fill and resolution records in immutable atomic
daily Zstandard shards. Each decision points back to its exact feature date and
row instead of duplicating 128 inputs, while retaining action logits, physical
mask, portfolio state, prediction memory, catalog outcomes and the causal-input
digest. Shard receipts bind source, policy and implementation hashes; their
ordered hashes form a verified replay manifest.

The H011 binder accepts validation or calibration artifacts only, rejects any
test array, requires valid closed-test pack and model receipts, and joins each
logit back to the exact feature shard and row. It verifies timestamp, market and
component state IDs against both tensors and debug provenance before emitting a
stable input digest bound to the train-only normalization artifact. This removes
positional assumptions between model output and simulator input. It additionally
binds the model's source digest to the pack manifest, normalization and every
daily receipt, and verifies the prediction artifact hash before any row is read.

The closed-development catalog selector loads only explicitly requested
validation/calibration conditions through a temporary indexed join. It rejects
test conditions, absent or non-replayable markets, missing close/payout state,
non-binary payouts and any catalog whose closed-test metadata or physical test-
label count is invalid. Resolutions are emitted in deterministic event time with
their exact catalog outcome and token mapping.

A registered reusable development-tape builder discovers every strictly
qualified validation/calibration condition directly from daily feature masks,
validates it against the closed catalog, and scans all 176,119,673 H009 rows.
For each market it retains the complete public trade interval from its first
eligible decision through the public close second, without market or trade caps.
Daily atomic receipts support resume and bind the result to the deep-hash-valid
H009 stream, qualified pack and deterministic condition digest. This removes the
need to parse the entire annual source tape on every policy epoch while keeping
test physically closed.

Sequential policy replay now has an explicit post-evidence inference boundary:
H010 first consumes the evidence trade and any fills it causes, then H012 reads
the updated portfolio and prediction memory, and only then is its action applied.
The qualified decision index exposes validation/calibration rows only, verifies
all tensor/example state IDs, and binds each action to the raw feature digest,
market anchor, portfolio, memory, previous action and physical mask. Normalized
feature access uses a bounded shard LRU rather than loading the annual pack into
RAM.

The full replay runner loads hash-bound direct or residual outcome checkpoints
and the selected H012 checkpoint, merges policy decisions, filtered public
liquidity and public resolutions in causal event time, and writes daily decision,
order, fill and resolution audits. It resumes from an atomic adapter checkpoint,
supports registered cost multipliers, reports CALL precision and weekly net
profit, and refuses any unconsumed qualified decision. After each immutable audit
shard, terminal orders, fills and closed-PnL rows are compacted into exact metric
aggregates while live orders, positions, prediction state and checkpoint hashes
remain restorable. This bounds memory even for a poorly initialized high-CALL
policy without discarding debug evidence.

Profit promotion now has a separate preregistered uncertainty audit. Weekly net
profit uses a four-week circular moving-block bootstrap to retain short serial
dependence; realized profit across called event components uses an equal-
component bootstrap. Both lower 95% bounds must be positive, with at least 1,000
resolved calls and 1,000 independent called components. Per-condition realized
PnL now includes both early sell/close PnL and terminal settlement, is written at
resolution, then compacted from live state. Baseline outperformance and cost
stress remain separate mandatory gates; a positive point estimate cannot promote
a policy.

**Full development tape result.** H010 scanned all 176,119,673 immutable H009
trade rows across 365 daily shards and retained 47,252,399 events from the first
qualified decision through public close for every development market. The tape
contains 288,350 resolved binary conditions: 153,006 validation and 135,344
calibration, linked to 809,614 and 578,176 qualified decisions respectively. Its
compressed payload is 4,713,675,472 bytes. Independent condition-loader checks
recovered both exact split counts from the source-bound artifact. The manifest is
valid, test rows consumed are zero and test labels remain unopened.

**Next action.** Bind completed H011 prediction rows and catalog resolutions to
the adapter and run the first development-only trade-tape replay with test
physically closed.

**Evidence boundary.** H010 mechanics do not establish historical depth,
executable fills, model quality, profit, untouched-test, paper-forward or
production evidence.

## SPH-T-H011: Uncapped Causal Flow Campaign

**Status:** `direct wallet rejected; 50M market-anchored wallet residual running`

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

**Calibration protocol.** Before the first full 50M result was opened, H011
registered a positive affine-logit Platt transform fitted on validation only.
Calibration remains a disjoint development audit. Outcome lift is paired against
the contemporaneous market probability per row, averaged equally within each
connected event component and measured with 5,000 deterministic component
bootstrap replicates. Promotion requires the upper 95% component-bootstrap
log-loss delta below zero; test arrays remain forbidden.

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

**First 50M market-only result.** Seed 17 trained 50,248,198 parameters for five
epochs before registered early stopping, with epoch 1 retained as best. The run
took 1,604.09 seconds over the complete pack and kept test rows at zero. On
811,726 validation decisions, calibrated log loss was 0.456481 against the
market's 0.452098. The equal-component delta was +0.005211 with 95% interval
[+0.004885, +0.005552], so the candidate clearly failed the validation lift
gate.

On the later 579,252-row calibration audit it moved in the opposite direction:
calibrated log loss was 0.445097 against 0.450778, and the equal-component delta
was -0.007515 with 95% interval [-0.008529, -0.006495]. This is real held-out
development evidence of temporal regime sensitivity, not a stable promotion.
Market-only is therefore retained as a control and not promoted. The matched
uncapped-wallet-flow run started immediately under the same seed, source and
training contract.

**Post-result timing audit and withdrawal.** A subsequently registered horizon
audit joined every labeled feature row to the same catalog `closed_at` used by
the causal resolution ledger. It found 4,040 development decisions at or after
public close: 852 train, 2,112 validation and 1,076 calibration. The maximum lag
was 46,722 seconds. Although this is only 0.14% of labeled rows and cannot explain
the broad result by itself, zero leakage is the contract. The market-only result
above is therefore retained as diagnostic history but withdrawn from model
selection. The active wallet run was paused atomically at epoch 2, shard 198,
batch 4 before opening any test data.

**Qualified pack.** A source-bound derivative view preserves all 176,119,673
stream rows, 4,483,489 decisions and immutable feature tensors through 3,288
NTFS hardlinks, while rebuilding only daily label masks. It removed exactly the
4,040 ineligible rows, leaving 1,397,374 train, 809,614 validation and 578,176
calibration labels. A second full horizon audit reports zero labeled decisions
at or after close, zero missing close times and zero test labels. New training
and calibration v2 contracts depend on this qualified view; old checkpoints
cannot resume under the new source digest.

**Qualified direct wallet result.** The matched 50,248,198-parameter uncapped-
wallet-flow model trained for six epochs and stopped after 1,869.65 seconds,
selecting epoch 2. It remained worse than the contemporaneous market on both
strictly pre-close development blocks. Validation log loss was 0.459994 versus
0.453239 (`+0.006755`); calibration was 0.453494 versus 0.451444
(`+0.002050`). Platt scaling fitted only on validation did not rescue the result:
calibration delta became `+0.003649`. The equal-component mean delta was
`+0.003562` with 95% bootstrap interval [`+0.002579`, `+0.004548`] across
91,840 components. The entire interval is worse than market. This rejects direct
full-probability relearning from the current wallet-flow representation; it does
not reject incremental wallet signal around the market anchor. Test remained
physically closed.

**Actor-context completion.** The uncapped public maker/taker context collector
completed all 451 adaptive partitions: 2,016,833 actor rows and 136,967,269
compressed bytes, with 32/64/128-way result-limit splits and a bound receipt
digest. Availability ends at 2026-01-06 and every monthly block remains masked
until its public window closes. This source is complete for its declared window,
not for the later half of H009, and contains no funding-transfer or insider
labels. It can now support an honest availability-masked actor ablation.

**Next action.** Complete the running 50M market-anchored uncapped-wallet-flow
residual. Build the availability-masked actor feature pack and its matched
residual after the direct causal comparison. Keep the Polygon graph variant
blocked until its source is complete.

**Evidence boundary.** H011 validation and conservative trade-tape replay cannot
establish untouched-test, historical orderbook depth, executable profit,
paper-forward performance, insider attribution or production readiness.

## SPH-T-H012: Portfolio-Aware Selective Policy

**Status:** `H012-v2 exact validation complete; profitable but rejected by robustness gates`

**Registered:** 2026-07-17, before full H011 outcome or profit metrics were
observed.

**Question.** Can one learned, portfolio-aware policy turn calibrated H011
outcome state into profitable `CALL_OUTCOME_0`, `CALL_OUTCOME_1` and `SKIP`
decisions, revise earlier calls and size multiple concurrent positions without
fixed confidence, edge, category or position limits? Outcome indices render as
`YES`/`NO` for yes/no markets and as the actual catalog labels elsewhere.

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

**Implemented policy backbone.** H012 warm-starts an H011 outcome backbone and
fuses its causal latent with a nine-field portfolio token and a recurrent
prediction-memory token. Previous action is categorical; probability, size,
elapsed time, position existence, position fraction, average entry and current
mark are numeric. Four learned policy latents pass through four inspectable
fusion blocks and emit seven action logits, beta-distributed equity sizing and a
state-value estimate. A physical action mask can reject impossible operations
but cannot impose confidence, edge, category, correlation or position-count
limits. Debug output retains market-group attention, policy attention, portfolio
and prediction-memory tokens for group ablation and attribution audits.

H012 now accepts either a direct H011 backbone or the market-anchored H013
wrapper. In the latter case the contemporaneous market probability is a required
input and the policy receives the selected anchor-plus-residual terminal logit;
the residual signal can no longer be silently lost at the outcome-to-policy
boundary.

Before any H012 profit result was observed, a two-stage optimization contract was
registered. Stage one vectorizes a learned CALL-outcome-0/CALL-outcome-1/SKIP and
beta-size warm start on the early validation component-time block. It maximizes
realized log growth after the registered adverse tick and fee, with no confidence
threshold, edge threshold, fixed size, call bonus or skip penalty. Whole
components are assigned to disjoint chronological fit and selection blocks.
Stage two remains recurrent clipped policy-gradient fine-tuning against exact
H010 state transitions and cost stress. Warm-start utility is diagnostic and
cannot replace replay profit evidence.

The stage-one trainer is implemented for both direct and residual checkpoints.
It verifies the selected outcome artifact hash, preserves the registered feature
mask, derives the component-time partition from all qualified validation rows,
balances fit rows by inverse square-root component frequency and keeps
calibration out of model selection. Initial portfolio and memory state are
explicit tensors; only physically possible initial actions are exposed. The
50M+ policy checkpoint binds configs, outcome result, source pack, partition,
implementation, optimizer, scheduler and all RNG states, and supports an atomic
`PAUSE` boundary. Reported warm-start utility, CALL rate and precision remain
diagnostic until their calls pass the stateful H010 replay.

**First wallet-flow utility warm-start result.** The 64,583,696-parameter H012
policy trained for four epochs and 866.28 seconds over 602,201 early-validation
fit rows, with 207,413 later-validation selection rows and 578,176 calibration
rows. Epoch 0 became the selected checkpoint by choosing `SKIP` for every row.
Later epochs did explore: epoch 1 made 1,377 calls at 68.12% side precision and
epoch 2 made 2,435 at 65.87%, but their mean realized log utilities were
`-0.0002362` and `-0.0003546`. Epoch 3 became unstable and called on all 207,413
rows at 47.71% precision, with `-0.0015015` mean log utility. The selected model
therefore makes zero calls on both selection and calibration and is not
promoted. This also demonstrates why CALL precision is only a diagnostic: the
apparently accurate sparse epochs concentrated on prices whose payout did not
cover error and registered costs.

**H012-v2 hypothesis.** Direct expected-utility gradients can escape into either
zero-size/all-SKIP or indiscriminate CALL without first learning the conditional
value of each side. The next warm-start will regress counterfactual realized
log-utility for CALL-0, CALL-1 and SKIP at a registered reference exposure, then
use the learned action values to optimize a separate state-dependent size. It
does not impose a CALL count, confidence threshold or fixed production stake;
SKIP remains the zero-utility action and the model must learn a positive value to
beat it.

**H012-v2 static result.** The same 64,583,696-parameter policy trained for seven
epochs and 1,293.61 seconds. Counterfactual action-value regression selected
epoch 3 rather than the safe all-SKIP anchor. On 207,413 later-validation rows it
made 2,893 calls (1.395%), achieved 76.08% side precision and positive mean
chosen log utility `+0.00000412` at a learned mean balance fraction of 0.436%.
The untouched development calibration block remained positive: 10,310 calls
(1.783%), 79.45% precision and `+0.00000338` mean chosen log utility at 0.436%
mean size. Fit, selection and calibration are all positive under the registered
static entry-cost proxy. This is the first H012 candidate eligible for exact
H010 replay, not a profit promotion: repeated calls, shared cash, fills,
liquidity, exits and component dependence are not represented by the static
metric.

**Exact-replay implementation audit.** The first replay attempt was invalidated
before policy inference because the qualified example JSON does not duplicate
every canonical `component_state_id`; the loader now reads the bound numeric
array and only verifies an optional JSON duplicate. A second attempt exposed a
simulator compaction defect: expired order IDs remained in the expiry heap after
their order records had been compacted. The simulator now discards stale heap
entries and rebuilds the heap from open orders at every history compaction, with
a regression test. Neither failed attempt is profit evidence.

The corrected batch-one replay was deliberately stopped after development day
182 because repeatedly executing the complete 64.6M-parameter market backbone
for each decision made the audit needlessly slow. Its checkpoint at 2026-01-13
had equity `$10,148.83` from `$10,000`, observed maximum drawdown `2.264%`,
16,278 `CALL_OUTCOME_1`, 2,837 `SKIP`, 3,260 resolved calls and 3,110 correct
calls. This trajectory is explicitly incomplete and non-promotable. It also
revealed an important distribution shift: sequential portfolio and prediction
memory caused far more calls than the static initial-state evaluation. Exact
full-period replay, rather than static metrics or this partial path, must decide
the hypothesis.

**Source-bound encoding-cache registration.** Before observing a completed
replay result, H012 registered a no-feature-removal throughput optimization.
The fine-tuned market backbone is independent of sequential portfolio state, so
its 512-wide latent, terminal logit and uncertainty output are computed once for
all 1,387,790 qualified validation/calibration decisions in large GPU batches.
Float32 retains the actual mixed-precision output without an additional latent
quantization step. Daily arrays bind source row indices, the qualified-pack
manifest and receipt, selected policy result, implementation digest and closed-
test contract. Sequential replay still recomputes the portfolio token,
prediction memory, physical action mask, policy fusion, position size and every
simulator transition in event-time order. A full-forward versus cached-forward
identity test protects the decomposition.

The complete cache contains 1,387,790 rows: 809,614 validation and 578,176
calibration decisions across 365 source-bound daily shards, with zero test rows.
It built on the RTX 5070 in 160.35 seconds. A real-checkpoint GPU audit sampled
42 rows across the time range. Cached and batch-one paths chose the same action
on all 42. Because BF16 matrix kernels can select different accumulation paths
at batch 4,096 and batch one, the comparison is not bitwise: maximum observed
absolute differences were `0.01283` in the 512-wide latent, `0.0078125` in the
terminal logit and `0.0002441` in an action logit. Float32 storage adds no second
quantization step, but the cache is still a fixed canonical inference
microbatch, not proof of bitwise batch-shape invariance. This numerical boundary
must remain visible in promotion evidence.

The first cached replay reached day 183 before a deliberate throughput pause.
It showed that two device-to-host input-validation synchronizations and eager
batch-one policy fusion had become the new bottleneck. H012 split validation
from a tensor-only trusted replay core and compiled only the four stateful fusion
layers with full-graph reduced-overhead CUDA execution. On the same GPU, the
policy-only benchmark improved from 339.99 to 3,555.84 decisions/second
(`10.46x`). Public model entry points retain their shape, categorical-range and
physical-mask validation; only source-validated replay uses the unchecked tensor
core. A second runtime pass packed the 512-wide latent, portfolio, memory,
previous action and physical mask into one CPU-to-GPU tensor and returned logits
plus both size parameters in one synchronization. The complete inference and
audit-object construction benchmark, including feature/input digests and
`Decimal` policy output, reached 1,903.39 decisions/second. The paused
trajectories remain non-evidence and will not be resumed across changed
implementation digests.

A profile of the first dense replay state then identified an algorithmic, not
GPU, bottleneck. With 426 open positions and 169,786 recorded equity points,
`portfolio_features()` rescanned every position and the full equity curve for
every new decision. One profiled day spent 14.97 seconds on 677 portfolio-state
reads and executed more than 110 million peak-equity generator steps. H010 now
maintains cost basis, marked exposure, peak equity and condition-to-position
membership incrementally on every fill, mark, sale and resolution. Every daily
snapshot still performs a full independent recomputation, rebases Decimal
associativity dust below a relative `1e-18`, and refuses to save if any aggregate
or index differs materially. The first invariant smoke run correctly stopped on
`1e-25`-scale addition-order dust before a checkpoint existed; the bounded
rebase was then registered and a clean daily snapshot passed. Restoring the profiled state produced
bit-identical features; 100,000 state reads completed in 0.300 seconds
(`333,826/s`). The shared Zstandard reader also moved from standard-library JSON
to the already-declared `orjson` dependency; the same 94,997-row tape shard
improved from 0.324 to 0.153 seconds while preserving row objects and order.

**Completed exact validation result.** H012-v2 processed all 809,614 qualified
validation decisions and ended with `$10,568.83` from `$10,000`, for net profit
`+$568.83` (`+5.688%`) after `$326.41` in fees. It made 38,702 resolved calls
over 11,628 conditions, all on outcome 1, with 34,813 correct calls (`89.95%`
precision). Maximum drawdown was `6.682%` and profit factor was `1.278`. All
positions were resolved and no test row or test label was consumed.

The path is not robust enough for promotion. Across 53 reported weeks, mean
profit was `+$10.73`, but only `50.94%` were positive; the worst week lost
`$495.69`. The registered four-week circular block bootstrap gave a 95% interval
of [`-$14.83`, `+$37.36`] for mean weekly profit. Across 6,772 independent
called components, the equal-component mean was `+$0.0840`, but its 95%
interval was [`-$0.0188`, `+$0.1981`]. Both required lower-positive gates fail.
The result is therefore rejected for promotion despite positive aggregate PnL.

Exact replay also confirms the distribution-shift trigger: the static selection
view called on `1.395%` of rows, while sequential replay called on `4.780%`.
Portfolio and prediction-memory tokens were trained only at their synthetic
initial state, so their real trajectory moved the learned policy substantially
off its training distribution. The immutable 1,009,433-row action/state audit is
now a source-bound teacher for H014; no fixed CALL threshold will be tuned from
this path.

**Next action.** Build H014 on the exact H012 replay states, fit only the original
early validation component block, select on the disjoint late block and rerun
the exact simulator. Keep calibration and test closed until H014 passes both
registered validation bootstrap gates and improves on H012.

**Evidence boundary.** Trade-tape development profit cannot establish historical
orderbook executability, untouched-test performance, paper-forward profit,
insider attribution or production safety.

## SPH-T-H013: Market-Anchored Residual Outcome Model

**Status:** `registered; residual wrapper and resumable trainer implemented`

**Registered:** 2026-07-18, after the direct H011 market-only temporal
inconsistency was measured and before any wallet-variant result was observed.

**Trigger.** Direct 50M outcome learning was worse than market on validation but
better on calibration. Relearning the full probability can discard a strong
causal prior and lets small regime shifts dominate the network output.

The later horizon audit withdrew that source run from evidentiary model
selection. H013 remains a registered architectural hypothesis motivated by the
market anchor, but its triggering metrics are diagnostic only; every H013 run
must use the strictly pre-close qualified pack.

**Hypothesis.** Start the terminal logit exactly at the contemporaneous market
logit and make the network learn only an additive residual. Initialize the
residual head to zero, apply no hard residual cap, and add a small registered L2
penalty. At step zero the model is exactly the market baseline; every learned
deviation must be justified by causal market, component, wallet and universe
state.

**Controlled comparison.** Train matched market-residual, uncapped-wallet-flow
residual, causal resolved-performance residual and actor-context residual
variants with the same H011 pack, candidate sizes, seed, optimizer and split
contract. Compare each both to its direct H011 counterpart and to the market.

**Implementation.** The wrapper consumes the exact causal market probability as
a separate anchor tensor, clips only for a finite logit transform and adds the
H011 outcome head as an unrestricted residual. The final residual projection is
initialized to exact zeros, so unit tests verify that initial output equals the
market bit-for-bit before learning. H011 information-group masks are reused
unchanged for matched direct-versus-residual ablations.

The H013 trainer evaluates and stores the exact zero-residual market anchor as
epoch -1 before the first optimizer step. A learned checkpoint replaces it only
when validation log loss improves, so the selected residual candidate cannot be
worse than the anchor merely because training ran. Checkpoints bind all three
configs, source receipts, implementation files, optimizer, scheduler and RNG
state; the lightweight progress receipt includes epoch history and supports the
same `PAUSE` resume boundary as H011.

**First qualified residual result.** The 50,248,198-parameter uncapped-wallet-
flow residual selected epoch 1 after five epochs and 1,617.38 seconds. Its raw
validation log loss improved the anchor by only `0.00000686`; raw calibration
was worse by `0.00003193`. Validation-fitted Platt scaling produced row-weighted
deltas of `-0.00000722` on validation and `-0.00004310` on calibration. The
equal-component view is deliberately stricter: validation mean delta was
`+0.00001701`, 95% interval [`-0.00000031`, `+0.00003407`], while calibration
was `-0.00007497`, interval [`-0.00009318`, `-0.00005637`]. Thus calibration
contains a statistically stable but extremely small incremental signal, whereas
validation does not meet the registered upper-bound-below-zero gate. The model
is not promoted. The exact market anchor successfully prevented the severe
degradation seen in direct wallet relearning.

**Causal resolved-performance residual result.** The matched 50,248,198-
parameter run stopped after three stale epochs and 1,012.49 seconds. No learned
epoch beat the exact market anchor, so the selected checkpoint remained epoch
`-1`: zero residual, not a learned wallet-performance correction. Validation-
fitted Platt scaling improved row-weighted log loss by `0.00000656` on validation
and `0.00004270` on calibration, but this is calibration of the market anchor,
not evidence for the masked causal wallet channel. Equal-component validation
delta was `+0.00001714`, 95% interval [`-0.00000024`, `+0.00003424`]; calibration
delta was `-0.00007468`, interval [`-0.00009290`, `-0.00005613`]. The registered
two-block gate fails and the variant is not promoted. This rejects the present
aggregate resolved-wallet-performance representation, not the wider actor-
sequence hypothesis.

**Acceptance.** Both validation and calibration must have component-bootstrap
upper 95% log-loss delta below zero. Test remains closed. Passing outcome lift
does not imply profitable selection; H010/H012 remain required.

**Next action.** Test whether the selected residual representation can identify
rare profitable subsets under H012/H010, then build the actor-sequence pack for
the wider wallet-attention hypothesis.

**Evidence boundary.** H013 development lift cannot establish selective-call
profit, untouched-test performance, executable profit or paper-forward profit.

## SPH-T-H014: Replay-State Policy Distillation

**Status:** `exact validation complete; rejected for lower profit and failed robustness`

**Registered:** 2026-07-18, after the complete H012-v2 exact validation and
bootstrap results were observed, before any H014 corpus, training or profit
metric existed.

**Trigger.** H012-v2 earned `+$568.83`, but its weekly and independent-component
bootstrap lower bounds were negative. Its exact replay CALL rate was `4.780%`,
more than three times the `1.395%` static selection rate. The market backbone was
trained causally, but the portfolio and prediction-memory fusion had seen only
synthetic initial states during optimization.

**Hypothesis.** Initialize from H012-v2 and train the state-dependent policy on
the exact causal portfolio, prediction-memory, previous-action and physical-mask
states it actually encountered. Preserve the immutable market latent and the
learned `CALL_OUTCOME_0`, `CALL_OUTCOME_1`, `SKIP` and balance-dependent size
outputs. Correcting state-distribution shift should reduce wrong clustered calls
and improve weekly risk-adjusted profit without a confidence threshold, edge
threshold, frequency target or fixed stake.

**Controlled construction.** Materialize one source-bound row for each of the
809,614 validation decisions in the H012 audit. Bind each row to the exact
qualified-pack row, cached market encoding, terminal outcome, causal market
price, component and timestamp. The recorded H012 action is provenance only and
is not a training target. Counterfactual CALL-0, CALL-1 and zero SKIP utilities
remain the targets. Freeze the market/outcome backbone and train the portfolio
encoder, memory encoder, previous-action embedding, fusion blocks, action head,
size heads and value head.

Reuse the exact whole-component chronological partition registered for H012:
602,201 early fit rows and 207,413 disjoint late selection rows are expected.
Calibration and test are excluded from corpus construction, optimization and
selection. Checkpoints bind every source manifest, implementation file,
partition, optimizer, scheduler and RNG state and support exact pause/resume.

**Replay-state corpus result.** The source-bound pack contains all 809,614 exact
validation decisions across 365 daily shards and occupies 92,416,562 bytes.
Construction took 50.50 seconds. Every row joins its original qualified feature
row to the corresponding immutable 512-wide market encoding and stores the exact
nine-field portfolio state, seven-field prediction memory, previous action and
seven-action physical mask. The teacher action is deliberately absent. The
original partition was reproduced exactly: 602,201 fit rows over 74,370
components and 207,413 selection rows over 31,874 components, with partition
digest `d3c2c16b...5608f3bf`. Calibration and test consumption are both zero.

The H014 trainer starts from the complete H012-v2 checkpoint, freezes and hashes
the market/outcome backbone, and optimizes only the state encoders, recurrent
fusion, action, beta-size and value heads. Batches join cached market latents to
logged causal states by source-row identity and refuse offset drift. Selection
uses learned counterfactual economic utility only, with no CALL-count target.
The initial H012 checkpoint is stored as epoch `-1`, so training cannot silently
replace it with a worse static selection candidate. Model, optimizer, scheduler,
all RNG states and sources are atomically checkpointed for exact pause/resume.

**First replay-state training result.** The 64,583,696-parameter model exposed
14,335,498 trainable state-policy parameters and completed five epochs in 249.18
seconds on the RTX 5070 before registered early stopping. Before optimization,
the H012 checkpoint reproduced 3,048 selection calls at 77.03% precision and
`+0.00000462` mean chosen log utility on the logged exact states. Epoch 0 was
selected with 9,556 calls (`4.607%`), 87.35% precision, `+0.00000503` mean
chosen log utility and a learned mean balance fraction of `0.393%`. Fit had
72,593 calls at 90.76% precision and `+0.00002272` utility.

Later epochs did not replace epoch 0. They oscillated between 10.32% and 87.11%
selection CALL rates and all had non-positive chosen utility, demonstrating that
lower fit loss alone does not identify a useful selective policy. The immutable
epoch-0 checkpoint preserves the frozen market-backbone digest and is selected
for a fresh exact H010 validation replay. This logged-state result is not profit
evidence because H014 will create a different recurrent portfolio trajectory.

**First exact-replay attempt invalidated.** H014 created a much denser recurrent
path than its logged-state evaluation. The run stopped on development day 223
before producing a result when an affordable BUY fill exceeded remaining cash
by bounded `Decimal` division/fee-rounding dust. H010 already capped shares by
cash and reserved every pending order, so this was not a strategy overdraft.
The simulator now distinguishes material overspend from relative `1e-18`
arithmetic dust, rebases only the separately rounded fee at that boundary and
retains a hard failure for any material excess. A regression test reproduces the
cash-boundary fill. The partial trajectory is invalid and will not be used as
profit evidence or resumed across the changed implementation digest.

**Completed exact validation result.** The corrected fresh replay processed all
809,614 decisions. H014 made 76,444 resolved calls over 32,394 conditions, with
69,112 correct calls (`90.41%` precision). Despite higher precision than H012,
it earned only `+$71.14` (`+0.711%`) from `$10,000`, paid `$1,178.13` in fees,
reached `10.38%` maximum drawdown and had profit factor `1.010`. H012 had earned
`+$568.83` with `$326.41` fees and `6.68%` drawdown under the same simulator.

Only `49.06%` of 53 weeks were positive; mean weekly profit was `+$1.34`, the
worst week lost `$587.65`, and the four-week block-bootstrap 95% interval was
[`-$33.06`, `+$34.22`]. Across 22,371 called components, equal-component mean
profit was only `+$0.00318`, interval [`-$0.04980`, `+$0.05404`]. Both lower-
positive gates fail and H014 does not outperform H012. The candidate is rejected
without opening calibration or test.

The failure is informative: rowwise state training increased precision by 0.46
percentage points but nearly doubled calls, added `$851.72` in fees and ignored
the joint cost of many correlated/repeated opportunities. One pass over H012
states reduced but did not eliminate recurrent state-distribution shift.

**Acceptance.** H014 must first improve static selection utility, then beat the
H012 exact replay in the same H010 simulator. Promotion still requires positive
lower 95% weekly and independent-component profit bounds, at least 1,000 calls
and 1,000 independent components, positive profit at registered cost stress,
and later untouched calibration, test and paper-forward evidence.

**Evidence boundary.** Logged-state counterfactual training is off-policy
development evidence. Only a fresh exact replay can measure its shared-cash,
liquidity and recurrent behavior, and neither can establish historical
orderbook executability or forward profit.

## SPH-T-H015: On-Policy Portfolio Advantage Aggregation

**Status:** `complete; unchanged model rejected; proxy replay failed replication and profit gates`

**Registered:** 2026-07-18, after the complete H014 exact replay and bootstrap
were observed, before any H015 corpus, training or replay metric existed.

**Trigger.** H014 improved side precision but converted that improvement into
more repeated calls, five times the H012 fee burden, higher drawdown and 87.5%
less net profit. The H014 loss weighted every decision row and supervised
terminal standalone utility, so highly active markets could dominate fit even
when the portfolio already carried similar risk or the order would not fill.

**Hypothesis.** Aggregate exact states from both H012 and H014 so the policy sees
successive on-policy distributions. Give every market equal total fit weight
within each behavior policy. In addition to CALL-0/CALL-1/SKIP counterfactual
terminal utility, regress the logged behavior action to its exact realized
fill-, fee- and resolution-aware value. This should teach causal fillability and
portfolio opportunity cost without a fixed CALL frequency, confidence threshold
or exposure limit.

**Controlled construction.** Build 1,619,228 state rows: one copy of every
validation decision from each behavior replay. Join decisions to orders, fills
and terminal payouts by immutable audit IDs. A filled BUY receives the exact
reference-size log utility implied by realized payout versus fill cost; an
unfilled order and SKIP receive zero logged execution value. The behavior action
is used only to select which action-value logit receives that target, never as
an imitation label. Whole-component fit/selection boundaries remain unchanged;
calibration and test remain closed.

Initialize from H014 epoch 0, freeze the identical market backbone and use a
lower state-policy learning rate. Select by equal-market counterfactual utility,
then require a fresh exact replay. A failed fresh trajectory may be added only
through a newly registered aggregation iteration; no replay result may be
converted into an after-the-fact CALL threshold.

**Acceptance.** H015 must beat H012 in net profit and maximum drawdown, pass both
positive lower-95% bootstrap gates, retain at least 1,000 calls/components and
remain profitable under registered cost stress before calibration can open.

**Evidence boundary.** Iterative development replay is not untouched-test,
historical-orderbook executable or paper-forward profit evidence.

**Completed corpus.** The source-bound H015 pack contains 1,619,228 decision
states: all 809,614 validation decisions from each H012-v2 and H014-epoch0
trajectory. Its 730 daily behavior shards preserve the exact portfolio state,
prediction memory, physical action mask and common frozen market encoding, and
add market identity, logged action, execution fraction, fill cost and marginal
fill-to-resolution PnL. The whole-component split remains 1,204,402 fit and
414,826 selection rows across 74,370 and 31,874 components respectively.

The independent attribution audit joined 36,590 H012 orders/10,223 fills and
72,751 H014 orders/44,735 fills. Summed decision PnL reproduces the source exact
replays at Decimal precision: `+$568.828209791737626...` and
`+$71.143302291975189...`. Only 4,538 H012 decisions and 12,999 H014 decisions
received fills, directly exposing the execution-selection failure that terminal
outcome supervision alone cannot represent. The behavior action is stored only
to select its action-value regression target; it is not an imitation label.

The 247,536,941-byte artifact is bound by manifest
`9b5f6414d826fde6bc0fd3ff49c3de59a227aabbf39f80ee6bca602f9c3e2e89`
and contract
`4793aee9b48d6f4ac6969764ec4c1cb3c358c8cb2da32e88438eb286f94e71da`.
Calibration and test consumption remain zero.

**Trainer contract.** H015 initializes the H014 epoch-0 policy and keeps its
64.6M-parameter market/outcome backbone frozen. For fit and selection
independently, every market receives equal total weight within each behavior
trajectory and both behavior trajectories receive equal total weight. The loss
combines three-action terminal counterfactual value with smooth-L1 regression of
only the logged action to its execution-fraction-adjusted realized value. This
does not imitate the logged action and introduces no CALL-rate, confidence,
edge, bet-size or portfolio threshold. Checkpoints preserve optimizer,
scheduler and all RNG states for exact epoch-boundary resume.

**Numerical qualification.** The first H015 training trajectory was stopped
after epoch 0 because its evaluation loss was infinite. The model outputs were
finite, but counterfactual smooth-L1 evaluated physically masked action logits
represented by the simulator sentinel `-3.39e38`; the resulting overflow made
the checkpoint invalid. The loss now accepts the causal physical-action mask,
excludes unavailable actions from counterfactual regression and asserts that
the logged behavior action was available. Regression tests reproduce the
sentinel case and require finite loss plus zero masked-action gradient. The
partial trajectory is not resumed or used as evidence; a fresh output contract
is required.

**Completed training.** The corrected fresh run trained 14,335,498 policy
parameters for three epochs over 1,204,402 fit rows before registered patience
stopped it. Epoch selection utilities were `-5.205e-6`, `-3.012e-6` and
`-8.988e-6`, all below the initial `-1.534e-6`. CALL rate expanded from 4.67%
at initialization to 9.59%, 14.26% and 33.72%, while unweighted precision fell
from 87.45% to 94.17%, 89.17% and 79.79% respectively. Higher first-epoch
precision did not compensate its negative equal-market utility.

Early stopping therefore selected epoch `-1`: the untouched H014
initialization. An independent named-tensor digest is identical for H014 and
the selected H015 policy:
`a3912a309df25c45b48b04081618e8a5665e46707d48be1c521138e8f04d4b81`.
This is a valid negative H015 training result, not an improved model. The result
is bound by contract
`f947dd172f7c0c81f2df31b57cea4989e05ce2db47d6286bbf62f038930dac6f`;
a fresh exact replay remains required by the registered protocol.

**Fresh proxy replay.** The selected H015 checkpoint was replayed over all
809,614 validation decisions under the registered legacy flat-100-bps cost.
It produced 76,555 calls, 44,915 fills, 90.400% precision, `+$131.39` proxy net
profit, 9.87% maximum drawdown and `$1,184.28` proxy fees. The lower 95% weekly
and component profit bounds remained negative at `-$31.60` and `-$0.0468`.
H015 therefore fails both robustness gates, trails H012 by `$437.43` and has
higher drawdown. Under H016 these economics are additionally classified as
`flat_fee_proxy_only`, not real-cost evidence.

**Identity-control failure.** H015 selected epoch `-1`, and every one of its 266
state-dict tensors is bit-identical to H014, so a correct deterministic replay
must reproduce the H014 trajectory. It did not: H015 made 111 more calls and
reported `$60.25` more proxy profit. The first action divergence occurred at
decision `cc6ec431...` on 2026-02-23: H014 selected CALL-1 while H015 was forced
to SKIP because both CALL actions became physically unavailable. Model logits,
portfolio features and prediction memory were identical immediately before the
state mask diverged.

The simulator computes reserved cash and shares by reducing Python sets of order
IDs. Set iteration order changes across processes, while finite-precision Decimal
addition is order-dependent. Near zero available cash this changes the physical
action mask and then cascades through the stateful trajectory. H015 consequently
fails the identity-replication gate; its apparent improvement over H014 is not a
model effect. Sorted deterministic reductions and a bit-identical fresh-process
replication test are now mandatory in H016. H015 is rejected for promotion.

## SPH-T-H016: Protocol-Exact Polymarket Fees

**Status:** `in progress; exact schedule and 1.0x baselines qualified; 1.5x/2.0x stress and fee-dependent retraining pending`

**Registered:** 2026-07-18, while the pre-registered H015 fresh replay was still
running and before its result was observed. The replay is allowed to finish only
as the final comparable flat-cost trajectory; it cannot qualify real net profit.

**Trigger.** H010 currently subtracts a flat 100 bps of notional from every BUY
and SELL fill. That is not Polymarket's fee model. The official rules make fees
dependent on protocol era, per-market schedule, price and liquidity role. Makers
pay no platform fee. Current takers pay a nonlinear USDC fee, while legacy CLOB
V1 settled BUY fees in outcome-token proceeds and SELL fees in collateral.
Consequently, every H010/H012/H014/H015 fee total and net-profit result observed
so far is explicitly reclassified as `flat_fee_proxy_only`; none is real-cost
qualified and none can support promotion.

**Official chronology.** The dated Polymarket changelog first enabled taker fees
for 15-minute crypto markets on 2026-01-05, added 5-minute crypto on 2026-02-12,
NCAAB and Serie A on 2026-02-18, and new all-duration crypto markets after
2026-03-06. The category fee structure changed on 2026-03-30 and the REST source
of truth became each market's `feeSchedule` on 2026-03-31. CLOB V2 went live on
2026-04-28 at approximately 11:00 UTC, replacing signed-order V1 fees with
operator-set per-market fees at match time. These creation-time and execution-
time boundaries intersect the development tape, which spans 2025-07-16 through
2026-07-15, so a present-day category rate cannot be projected backward.

**Registered implementation.** Build an immutable condition-and-time fee
schedule from historically observed market/order parameters, pinned official
contract and SDK behavior, and dated rollout rules in that priority order. Every
fill must record protocol version, schedule source, maker/taker role, fee asset
and exact amount. The present marketable-limit simulator is registered as a
taker because it crosses observed liquidity after latency; it cannot claim maker
status without a separate passive-order queue model. No builder code, maker
rebate or taker rebate is assumed.

V1 arithmetic must reproduce the pinned contract's integer rounding and preserve
the distinction between shares deducted from a BUY and collateral deducted from
a SELL. V2 must query the per-market fee details and reproduce the official
price-dependent formula and five-decimal minimum-fee behavior. Current reference
rates are recorded for regression tests, not historical imputation. Any fill
whose applicable historical schedule cannot be established fails closed and
rejects real-fee qualification; there is no zero-fee, category or 100-bps
fallback.

**Official implementation audit.** The research pins CLOB V1 contract commit
`ed5c7708...`, V1 FeeModule `1a3c31c4...`, CLOB V2 contract `ccc05960...` and
the V2 Python SDK `215fc63a...`. The V2 SDK implements the generalized schedule
`shares * rate * (price * (1-price)) ** exponent`; the current public table is
the exponent-one case. The V2 contract receives an absolute operator fee in
collateral, charges it in addition to BUY collateral and deducts it from SELL
proceeds. The V1 contract instead charges the output asset. Its FeeModule may
refund the signed exchange fee down to the operator's intended amount, so V1
qualification must join `OrderFilled` with `FeeRefunded` by order hash rather
than treating the gross exchange event as the amount actually paid.

The tape already retains a transaction hash for every public liquidity event.
That permits a stronger historical source than present-day market metadata: fetch
the contemporaneous Polygon receipt, identify the active taker event, decode any
V1 refund and bind the resulting schedule evidence to the liquidity ID. As a
proof vector, transaction `0x1907...c63d3` on 2026-05-15 contains a V2 taker BUY
of 150 shares at 0.51 and an exact 2.62395 collateral fee; maker events charge
zero. This reconciles exactly as `150 * 0.07 * 0.51 * 0.49 = 2.62395`.

**Determinism repair.** Every reservation reduction now sorts order IDs before
Decimal accumulation. Daily full-position validation, checkpoint serialization
and resume-time aggregate reconstruction also use canonical token/order order.
The regression suite forces identical positive Decimal reservations through two
opposite set iteration orders at a 28-digit precision boundary that previously
produced different available cash; both now return exactly the same state.
Fresh-process full replay identity remains an H016 acceptance gate and is checked
together with the real-fee baselines below.

**Protocol fee engine.** The simulator now has an immutable, manifest-bound and
fail-closed fee-schedule book keyed by the causal liquidity event. It implements
the generalized V2 price curve, taker-only role handling, five-decimal half-up
rounding and rate stress. It separately implements the V1 symmetric minimum-
price curve with integer-style downward rounding. A V1 BUY deducts the fee from
received outcome shares without spending extra collateral; V1 SELL and both V2
sides settle the fee in collateral. Orders reserve cash against their evidence
event's schedule, exact fills rebind to the later transaction/condition/time,
and checkpoint resume rejects a missing or changed schedule manifest.

Per-fill audit rows now preserve gross shares, position shares, collateral fee,
outcome-share fee, USD fee value, protocol, schedule ID, fee asset and explicit
taker role. BUY sizing solves against the active protocol fee instead of the old
flat-bps denominator. Legacy proxy replay remains available only when no H016
schedule artifact is supplied, so prior results remain reproducible and cannot
silently acquire a real-cost label. The complete regression suite includes cash/
position reconciliation vectors for V1 and V2, fail-closed schedule binding and
the Decimal affordability boundary that previously rejected a valid all-cash
fill by one terminal digit. All 183 tests pass.

**Historical receipt qualification.** On-chain smoke qualification exposed two
distinct V1 operator settlement curves that the gross exchange contract alone
cannot represent. Before the category rollout, a crypto receipt for 1,590.06
shares at 0.59 retained 23.260830 outcome shares, matching the rate-0.25,
exponent-2 curve applied directly in the output asset up to constituent-fill
rounding. After the rollout, a 200-share BUY at 0.04 retained 13.824 shares,
matching a rate-0.072 exponent-1 USD curve converted back to BUY outcome shares.
V1 post-rollout operator amounts settle at five decimals; V2 remains five-
decimal collateral settlement. H016 now models both V1 formulas explicitly and
selects between them from the contemporaneous active-taker receipt, never from
the present category label. March 30 and March 31 are separate registered
boundaries because the category announcement and `feeSchedule` source-of-truth
change were dated separately.

The completed validation Schedule Corpus covers all 809,614 decisions, including
SKIP states, so a future policy cannot select a validation market outside the
qualified fee universe. It contains 163,707 non-overlapping condition-time
intervals over all 153,006 validation conditions and binds 854,610 liquidity or
decision-evidence IDs. Evidence consists of 162,839 Polygon receipt proofs, 308
market-wide trade/receipt proofs and eight receipt-reconciled market-info proofs.
No interval is unresolved. The compressed schedule data digest is
`77762080...40ed`; the manifest digest is `3fe07cad...3cc`.

The qualified formulas are 86,427 generalized Polymarket USD curves, 34,268 V1
output-asset curves and 43,012 zero-fee intervals. All validation intervals are
CLOB V1 because the last validation decision precedes the V2 cutover. The
builder never infers a positive tariff from category alone: 120,456 intervals
come from direct positive active-taker receipts, 42,067 from direct zero-fee
receipts, 868 from the dated official pre-fee boundary, 308 from market-wide
trade receipts and eight from receipt-reconciled market metadata. Receipt cache,
pause/resume state and large artifacts remain on `E:`; only code and immutable
result contracts are committed.

**Real-fee 1.0x baselines.** H012-v2 completes all 809,614 validation decisions
with 38,682 resolved calls and 10,291 fills. It earns `$950.229106` after
`$2.841145` execution-time fee value, for `9.5023%` return, `6.1537%` maximum
drawdown, `1.5270` profit factor and `89.9514%` call precision. Across the 40
non-zero activity weeks its mean is `$23.76`, median `$10.65`, positive fraction
`70.0%` and worst week `-$433.79`. Its registered moving-block weekly lower 95%
bound is `-$7.63`, so H012 still fails weekly robustness.

H014 earns `$1,186.530763` after `$95.442343` execution-time fee value, for
`11.8653%` return, `5.8156%` maximum drawdown, `1.1890` profit factor and
`90.4032%` precision over 76,359 resolved calls and 45,655 fills. Its 40 active
weeks average `$29.66`, have a `$10.41` median, are positive `67.5%` of the time
and have a `-$350.94` worst week. The weekly moving-block lower 95% bound is
`+$0.30`, but the independent-component lower mean bound remains slightly
negative at `-$0.000757`; H014 is therefore the current real-fee development
leader, not a promoted model.

Every replay fill has schedule ID, protocol, taker role, fee asset, collateral
fee, outcome-share fee and position-share fields. Summing `fee_usd` directly
over the audit reproduces each result total within `1e-12`. H014 and the selected
H015 checkpoint have identical model tensors; their independently generated 365
daily compressed audit shards are byte-identical and every economic metric is
equal. The H015 identity-replication control now passes. Component bootstrap
inputs are also sorted canonically after this comparison exposed a second
process-order dependency in the evaluator.

**Acceptance.** Official fee examples, V1/V2 contract vectors, role handling,
rounding, creation-time rollout rules and the cutover boundary must pass
regression tests. Per-fill fee audit must reconcile exactly to cash and token
positions. Then H012, H014 and H015 must be replayed under 1.0x, 1.5x and 2.0x
authoritative rate stress, and fee-dependent targets must be rebuilt and
retrained. Calibration and test remain closed. No new model-profit claim may be
made until the real-fee baselines are committed.

**Evidence boundary.** Correct platform fees qualify only the cost side of the
simulator. The trade tape still does not reconstruct historical orderbook depth,
queue position or forward executable profit.
