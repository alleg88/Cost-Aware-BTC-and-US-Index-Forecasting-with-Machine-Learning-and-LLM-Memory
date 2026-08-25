# Cost-Aware BTC and US Index Forecasting with Machine Learning and LLM Memory

[Read notebooks](code/notebooks/README.md) · [Download code and data](https://github.com/alleg88/Cost-Aware-BTC-and-US-Index-Forecasting-with-Machine-Learning-and-LLM-Memory/releases/tag/v1.0.0) · [Reproduce the study](REPRODUCIBILITY.md)

**3 markets · 9 model families · 27 executed notebooks**

Reproducible research code for short-horizon directional prediction on BTCUSDT,
USA500 and USATECH. The implementation compares nine machine-learning families,
sentiment features, ensembles, channel/volatility signals and host-controlled LLM
routing under chronological evaluation and transaction costs.

## Study boundary

- Core predictions use three classes: short, flat and long.
- Model selection uses 2024 chronological out-of-fold evidence.
- Policy calibration uses January-June 2025; the forward period is July 2025-March 2026.
- The final Q2 interval is `[2026-04-01, 2026-07-01)` and was opened once under a frozen protocol.
- BTC is the confirmatory stream; USA500 and USATECH are descriptive cross-market checks.
- Reported trading results are net of the registered costs: 10 bps BTC, 2 bps USA500
  and 3 bps USATECH per round trip.

The exact research questions and evidence boundaries are in
[PROJECT-PLAN.md](PROJECT-PLAN.md).

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Python package, experiment runners, tests and executed notebooks |
| `PROJECT-PLAN.md` | Implemented study protocol and research questions |
| `REPRODUCIBILITY.md` | Installation, data, verification and rerun instructions |

## Quick start

**Start in Google Colab:** open [00_run_in_colab.ipynb](code/notebooks/00_run_in_colab.ipynb),
the first notebook in `code/notebooks`. Extract it from the code ZIP and
open it in Colab via **File → Upload notebook**. Select **CPU**, then **Runtime → Run all**.
When **Choose files** appears, select the **code ZIP**, not the notebook or `run_zip.py`.
Wait for the upload to finish; setup continues automatically.
It uses Colab's own Python and installs only missing packages.
Rerunning cells reuses the ZIP; changing or deleting the runtime removes uploaded files.
**Check installation** works without the source ZIP.
For the full calculation, upload both ZIPs and choose **Rebuild results**. No API key is
needed for the supplied data and frozen LLM responses.
Colab versions can differ from the fixed local environment; recalculated values may differ.

**Local setup:**

Requirements: Python 3.12.10 and Git. Download the named code ZIP from
[v1.0.0](https://github.com/alleg88/Cost-Aware-BTC-and-US-Index-Forecasting-with-Machine-Learning-and-LLM-Memory/releases/tag/v1.0.0),
extract it, then run `python prepare_project.py` from the extracted project root.
The source ZIP is optional for checks; extract it into the same folder for a full rebuild.
Use the prepared code ZIP, not GitHub's automatic Source code
archive. A Git clone can proceed directly to environment setup.

From the project root, create an environment:

```bash
cd code
python -m venv .venv
```

Windows: `.venv\Scripts\activate`; macOS/Linux: `source .venv/bin/activate`.
Then install and verify:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-repro.txt
python -m experiments.reproduce_tracked
python -m experiments.reproduce_source --audit-only
```

`reproduce_tracked` checks the environment, release evidence and the registered
`pytest --clean-clone` suite. The source audit validates the dependency graph without
downloading data or fitting models. See the
[running guide](REPRODUCIBILITY.md) for data setup and additional options.

## Rebuild calculations

Numerical reconstruction needs the hash-verified source bundle in
`code/.source_evidence/` (included in the data ZIP). Run:

```bash
python -m experiments.reproduce_source
python -m experiments.reproduce_notebooks
```

Public Binance archives are downloaded automatically. The portable bundle supplies the
registered index/VIX exports, news and direct-event snapshots, and frozen provider
responses that cannot be recreated byte-for-byte. Fresh LLM calls require
`OLLAMA_API_KEY`; no secret is stored in the repository.

The saved notebooks can be read immediately; refitting the full study needs substantial
runtime, disk space and internet access for public data and model downloads.
