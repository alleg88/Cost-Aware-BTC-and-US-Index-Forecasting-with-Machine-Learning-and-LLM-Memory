# Evaluation protocol

The experiments test whether added forecasting methods improve trading returns
after costs. Prediction scores and trading returns are reported separately.

## Markets and inputs

| Market | Data |
|---|---|
| BTCUSDT | Binance spot candles and order flow; derivatives funding, open interest and long–short ratios |
| USA500 and USATECH | Dukascopy index-proxy bid/ask quotes, aggregated into 15-minute bars |
| Volatility | Dukascopy VOLIDXUSD, used as a VIX proxy |
| Sentiment and events | GDELT news via BigQuery, FRED releases, Crypto Fear and Greed, FOMC material and a frozen public Truth Social archive |

Inputs use information available at the decision time. The main prediction grid
is 15 minutes; one-minute bars support trading replay. The channel experiments
use completed hourly structures and five-minute decisions.

## Comparisons

The nine model families are logistic regression, decision tree, random forest,
linear SVM, XGBoost, CatBoost, MLP, LSTM and GRU. Targets classify the next
movement as down, flat or up, with a threshold around zero for the flat class.

| Addition | Control |
|---|---|
| All-nine ensembles | Best eligible single model |
| Qualified Union, a separate LSTM/SVM ensemble | Eligible LSTM |
| Sentiment features | Market-only inputs under matched settings |
| LLM outcome memory | Absent and shuffled memory; deterministic Hedge weighting |
| Model-based direction | Channel direction at the same entry times |
| Channel entry restriction | Channel-blind selection with matched entry counts |

Each comparison holds its relevant data, execution and evaluation settings fixed.
The LLM memory comparison is exploratory; the channel studies use development data.

## Chronology and costs

- **2024:** core model comparison using five chronological folds.
- **January–June 2025:** policy calibration and eligibility checks.
- **July 2025–March 2026:** later historical replay with frozen selected policies.
- **April–June 2026:** separate final test of six previously selected policies.

The final BTC comparison is LSTM versus Qualified Union. The index policies
provide supplementary checks; their final results contain only one to three trades.
The final quarter is not used to select or tune models. Earlier channel inputs
extend back to 2021, with the channel studies ending in June 2025.

Executed returns subtract 10 basis points per round trip for BTC, 2 for USA500
and 3 for USATECH. One basis point is 0.01%. These are fixed modelling assumptions;
variable slippage, market impact and live delays are not simulated.

Prediction reporting includes macro-F1. Trading reporting includes net return,
trade count, Sharpe, Sortino and drawdown. The [notebooks](code/notebooks/README.md)
contain the results; the [running guide](REPRODUCIBILITY.md) explains reconstruction.
