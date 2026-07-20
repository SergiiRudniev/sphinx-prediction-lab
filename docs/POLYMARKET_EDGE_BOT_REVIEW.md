# Polymarket Edge Bot Review

**Reviewed:** 2026-07-20

**Artifact:** [jc-builds/polymarket-edge-bot](https://huggingface.co/jc-builds/polymarket-edge-bot)

## What it is

The released system is an 800-tree LightGBM classifier over 424 features: a
384-dimensional frozen MiniLM question embedding and 40 tabular market/event
features. Its calibrated output estimates `P(YES)` at market open and the
trading rule compares that estimate with opening price. It is not a wallet-flow,
participant-graph or continuous stateful policy model.

The model card reports 219,680 training rows and 38,768 calibration rows, with a
chronological boundary at 2026-02-16 and calibration through 2026-03-19. Its
constrained three-month backtest reports 79 bets and `+7.7%` from `$1,000`, but
uses a flat 2% fee assumption and does not model order-book fills or slippage.
Those numbers are therefore not comparable to the Sphinx receipt-qualified,
stateful simulator.

## Reproducible artifact audit

Files were read from the public Hugging Face repository. Recorded SHA-256:

- `feature_spec.json`: `8c6605238cadaac63f7e0759e8611dce0a8acbfb8258ee2ef621cbaca0a15e94`
- `training_meta.json`: `b0ba3beabc5515cf9d30da69aa6ef475538a36b4370b4e467822880c19f04843`
- `inference_example.py`: `271c8250c44e951266dffcb45aa7881838d355477179ea6571e69ac7909092fb`
- `lightgbm_model.txt`: `52621d533afa7acc32deae12b3224eb50d7cdf5e0c821a73abcf2a8d8630af81`

The largest parsed LightGBM gain features are `first_yes_price`,
`price_squared`, `price_log_odds`, `this_over_event_max`, `log_total_usd`,
`event_open_sum`, `price_distance_half`, `event_open_max`, `log_n_trades`, and
`event_open_mean`. Tabular features contribute 57.43% of total tree gain and
text-embedding dimensions 42.57%. Gain is not causal attribution, but the
ranking is strong architectural evidence that price geometry, calibration and
event-relative context deserve explicit representation.

## What Sphinx can use

H021 adopts the transferable ideas without copying its decisions:

- explicit candidate entry price for both outcomes;
- calibrated outcome probability and exact break-even probability;
- event-relative context alongside Sphinx wallet, flow, portfolio and memory
  representations;
- a selective policy evaluated by deployable portfolio replay rather than by
  aggregate classification accuracy.

The released weights are prohibited in Sphinx historical validation. Their
training/calibration period overlaps the Sphinx development replay, so using
them would leak later resolutions. A future ablation may retrain a LightGBM plus
isotonic baseline only on the Sphinx fit interval, then compare it with H021 and
with wallet features shuffled or removed. This is an architecture/control
hypothesis, not evidence that the external bot is profitable.

## Sources

- [Model card](https://huggingface.co/jc-builds/polymarket-edge-bot)
- [Feature specification](https://huggingface.co/jc-builds/polymarket-edge-bot/blob/main/feature_spec.json)
- [Training metadata](https://huggingface.co/jc-builds/polymarket-edge-bot/blob/main/training_meta.json)
- [Inference example](https://huggingface.co/jc-builds/polymarket-edge-bot/blob/main/inference_example.py)
- [Repository files](https://huggingface.co/jc-builds/polymarket-edge-bot/tree/main)
