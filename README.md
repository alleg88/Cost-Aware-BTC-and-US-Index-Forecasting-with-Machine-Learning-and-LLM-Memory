# Cost-aware BTC and US index forecasting

Do better forecasts produce better trading returns after costs? This project
compares machine learning models, ensembles, news sentiment, LLM memory and
price-channel signals on Bitcoin and two US equity-index proxies.

**3 markets · 9 model families · 22 notebooks with saved results**

[Explore the notebooks](code/notebooks/README.md) ·
[Download](https://github.com/alleg88/Cost-Aware-BTC-and-US-Index-Forecasting-with-Machine-Learning-and-LLM-Memory/releases/latest) ·
[Run the code](REPRODUCIBILITY.md)

## Start here

Open the [first notebook](code/notebooks/01_RQ1_A_BTC_data_labels_baseline.ipynb)
for data and labels, or the [final test](code/notebooks/22_Lockbox_Q2_2026.ipynb)
for the held-out Q2 2026 results. Tables, charts and explanations are saved in
each notebook; viewing them requires no installation, account or data download.

| Notebooks | Comparison |
|---|---|
| 01–05 | Individual models, market positioning and volatility inputs |
| 06–11 | Single models versus ensembles |
| 12–17 | Market-only inputs versus news sentiment |
| 18 | LLM ensemble weights with real, absent and shuffled outcome memory |
| 19–21 | Price-channel entry timing and trade direction |
| 22 | Previously selected policies on an unseen quarter |

## What the results show

Prediction quality and trading returns can rank models differently. The added
methods did not establish a reliable benefit after costs under the tested conditions.
The two-model LSTM/SVM ensemble's earlier advantage did not persist in the final
Q2 test: both BTC policies lost money. The US index results were mixed and based
on very few final-test trades.

## Data and evaluation

BTCUSDT candles and positioning come from Binance; the US index and volatility
proxies come from Dukascopy. Additional inputs include timestamped news and
economic events. Training precedes testing, and each addition has a defined control.
Trading returns deduct fixed round-trip costs of 10 basis points for BTC,
2 for USA500 and 3 for USATECH. Variable slippage, market impact and live delays
were not simulated. See the [evaluation protocol](METHODOLOGY.md).

## Download or rerun

- **Release-Client.zip:** a small download with Python code and saved notebook results.
- **Release-Rebuild.zip:** code and additional inputs for recalculation; larger runs
  need internet access, several GB of disk space and hours of computation.

Use the [running guide](REPRODUCIBILITY.md) for local or Colab setup and the
[input guide](code/notebooks/DATA.md) for individual notebook reruns.
