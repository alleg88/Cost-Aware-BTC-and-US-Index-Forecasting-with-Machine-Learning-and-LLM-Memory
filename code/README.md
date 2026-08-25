# Python implementation

This directory contains the package, experiment runners, tests and 27 executed
notebooks. Run commands here with Python 3.12.10. The
[running guide](../REPRODUCIBILITY.md) distinguishes verification from
full numerical reconstruction.

## Installation and verification

Create `python -m venv .venv`, then activate it with `.venv\Scripts\activate`
(Windows) or `source .venv/bin/activate` (macOS/Linux). Git is required for reference checks;
ZIP users first run `python prepare_project.py` from the project root.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-repro.txt
python -m experiments.reproduce_tracked
python -m experiments.reproduce_source --audit-only
```

`reproduce_tracked` runs the dependency check, release audit and registered
`pytest --clean-clone` suite. The audit-only source command validates 59 tasks and 184
declared outputs without downloading data or fitting models.

## Full numerical reconstruction

Extract the source-data ZIP from [Releases](https://github.com/alleg88/Cost-Aware-BTC-and-US-Index-Forecasting-with-Machine-Learning-and-LLM-Memory/releases/tag/v1.0.0)
alongside the code ZIP. For a clone, copy `code/.source_evidence/` into this directory.
Then verify the data and rebuild:

```bash
python -m experiments.source_evidence verify --source .source_evidence --manifest .source_evidence/source_evidence_manifest.json
python -m experiments.reproduce_source
python -m experiments.reproduce_notebooks
```

The bundle supplies the registered Dukascopy/JForex index and VIX exports, English
source-whitelisted BigQuery news, FRED, Fear & Greed, Truth Social/direct-event snapshots,
frozen provider responses and channel reference inputs. Public Binance archives are
downloaded automatically. Generated features, fits and caches are rebuilt.

Fresh LLM scoring or Reflection Agent calls require an Ollama Cloud key; never store it in
the repository:

```powershell
$env:OLLAMA_API_KEY = "your-key"
$env:OLLAMA_HOST = "https://ollama.com"
```

Exact reconstruction uses frozen provider evidence because a fresh response may differ.

A bundle stored elsewhere can be imported with
`python -m experiments.source_evidence restore --source /path/to/bundle --target .source_evidence`.

## Module map

| Path | Responsibility |
|---|---|
| `configs/` | Registered study, policy and rebuild protocols |
| `data/` | Binance and Dukascopy/JForex loading and normalisation |
| `features/` | Causal price, positioning, sentiment, index and channel features |
| `models/` | Nine-model registry and deep-model wrappers |
| `ensemble/` | Chronological stacking primitives |
| `evaluation/` | Splits, trading ledger, costs and metrics |
| `sentiment/` | GDELT/direct-event preparation and DeBERTa/LLM scoring |
| `memory/`, `reflection_agent/` | Bounded memory controls and BTC/index LLM routers |
| `experiments/` | Rebuild graph, model runs, audits and notebook builders |
| `tests/` | Leakage, protocol, packaging and result contracts |
| `notebooks/` | Executed experimental readers and one launcher |

The notebook order and concise conclusions are in
[`notebooks/README.md`](notebooks/README.md).
