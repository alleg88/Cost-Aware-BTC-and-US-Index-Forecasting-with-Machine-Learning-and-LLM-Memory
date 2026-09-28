# Python code

The code builds market features, fits forecasting models and evaluates their
trading signals after transaction costs.

| Folder | Purpose |
|---|---|
| `notebooks/` | 22 analyses with saved tables, charts and explanations |
| `data/`, `features/` | Data loaders, features and target labels |
| `models/`, `ensemble/` | Forecasting models and ensemble methods |
| `sentiment/`, `reflection_agent/` | News scoring and LLM decisions |
| `evaluation/` | Chronological splits, prediction metrics and trading replays |
| `experiments/`, `configs/` | Experiment commands and settings |
| `tests/` | Checks for calculations, data timing and reproducibility |

Start with the [notebook list](notebooks/README.md). Viewing saved results needs
no installation. For calculations, follow the [running guide](../REPRODUCIBILITY.md)
and the [input instructions](notebooks/DATA.md).

In Colab, open the selected Rebuild notebook, choose **Runtime > Run all**, and
upload **Release-Rebuild.zip** when prompted. Notebook 18 downloads its verified
published inputs directly. Local setup needs Python 3.11 or newer and Git;
larger calculations can take several hours.
