# Implemented study protocol

## Research questions

1. **RQ1:** How do nine pre-specified ML model families compare in short-horizon
   directional prediction and net-of-cost trading performance?
2. **RQ2:** Do pre-specified all-nine ensemble methods improve net-of-cost performance
   over the best eligible single model?
3. **RQ3:** Do sentiment features improve performance over a market-only control?
4. **RQ4:** Does host-controlled LLM routing improve BTC net-of-cost performance over
   fixed controls?
5. **RQ5:** Do channel or volatility signals improve BTC net-of-cost performance over
   matched controls?

## Markets and chronology

| Component | Registered boundary |
|---|---|
| Instruments | BTCUSDT, USA500 and USATECH |
| Main grid | M15 prediction grid; selected BTC/channel experiments also use causal M1, M5 and H1 inputs |
| Model selection | 2024 chronological blocking cross-validation |
| Policy calibration | January-June 2025 |
| Forward evaluation | July 2025-March 2026 |
| Final interval | Q2 2026, half-open `[2026-04-01, 2026-07-01)` |
| Channel branch | BTC development evidence through 30 June 2025 only |

BTC is the sole confirmatory Q2 stream. USA500 and USATECH provide descriptive transport
checks under separately frozen market policies.

## Models and inputs

The nine pre-specified families are logistic regression, decision tree, random forest,
linear SVM, XGBoost, CatBoost, MLP, LSTM and GRU. All predict down/flat/up under the same
chronological protocol; class balancing is fitted on training folds only.

The primary input arms are price/order-flow, Binance positioning, VIX for indices,
DeBERTa sentiment and structured LLM sentiment. News/event features are joined as-of so
only information available before the decision timestamp is used.

## Ensemble and agent comparisons

- RQ2 compares all-nine probability averaging, directional voting and a causal logistic
  meta-model with the best eligible single-model control.
- The BTC Qualified Union is a separate two-member LSTM/SVM candidate with an
  opposite-signal veto.
- RQ4 compares host-bounded LLM routing with no-memory, shuffled-memory and deterministic
  controls. The host fixes the available actions, calculations, chronology and risk gates.
- RQ5 compares channel-conditioned activation with matched channel-blind
  volatility/opportunity controls.

## Evaluation rules

- No random time shuffling for model selection or evaluation.
- Symmetric trading mapping: down to short, flat to no position and up to long.
- Net results deduct 10 bps BTC, 2 bps USA500 and 3 bps USATECH per round trip.
- Predictive reporting includes macro-F1, balanced accuracy and class-level diagnostics.
- Economic reporting includes net return, trade count, long/short coverage, Sharpe,
  Sortino and maximum drawdown.
- Q2 was opened once after candidates and controls were frozen; it is not used for
  post-hoc selection.

The executed evidence is organised in [code/notebooks/README.md](code/notebooks/README.md),
and the data/rebuild boundary is defined in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
