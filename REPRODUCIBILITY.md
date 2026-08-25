# How to run the code

The repository separates tracked verification, immutable source evidence and generated
calculations.

| Level | Reproduced from | Additional requirement |
|---|---|---|
| Tracked code, installation and tests | Git checkout, pinned Python requirements, release audit and `--clean-clone` tests | Package-download network access |
| Saved executed notebooks | 27 canonical readers with saved tables and figures | None for inspection |
| Public, checksum-verified downloads | Binance BTCUSDT candles, futures metrics and funding | Access to `data.binance.vision` |
| Portable source-evidence bundle | Dukascopy/JForex USA500, USATECH and VIX; English approved-source GDELT GKG via BigQuery; FRED, Fear & Greed, Truth Social/direct-event snapshots; frozen provider evidence and channel handoffs | Restore the separate bundle into `code/.source_evidence/` |
| Generated calculations | Normalised data, features, fits, predictions, ensembles, channels and result tables | Run the registered source-to-results graph |

The manifest registers 4,842 logical inputs. The portable export contains 426 files
(0.92 GB uncompressed); public Binance downloads add about 2.3 GB before extraction.
Small scientific references are retained for comparison; environments, bulk calculated
caches, credentials and working notes are not included.

## Download and setup

For Google Colab, open `code/notebooks/00_run_in_colab.ipynb` from the code ZIP and select
**CPU**, then **Runtime → Run all**. When **Choose files** appears, select the code ZIP,
not the notebook or `run_zip.py`. Setup continues automatically after the upload.
Rerunning cells reuses the ZIP; changing or deleting the runtime removes uploaded files.
The source ZIP is optional for installation
checks and required for **Rebuild results**. The launcher safely unpacks the supplied files,
uses Colab's native Python, installs only missing packages and verifies imports, files
and the full clean-clone test suite. Existing Colab libraries are preserved; their versions
are recorded in `code/run.log`. The fixed local requirements remain unchanged; numerical
equivalence is not assumed when Colab versions differ. **Rebuild results** additionally runs the full calculation; it can exceed
Colab's memory, disk or session limits. Download rebuilt notebooks before the runtime ends.
To switch a code-only run to a full rebuild, choose **Rebuild results** and upload the source ZIP when prompted; the code ZIP is reused.

For a local installation:

Download and extract the code ZIP from [v1.0.0](https://github.com/alleg88/Cost-Aware-BTC-and-US-Index-Forecasting-with-Machine-Learning-and-LLM-Memory/releases/tag/v1.0.0).
Add the source ZIP to the same folder only for data verification or a full rebuild. From the extracted project
root, run `python prepare_project.py` once with Python 3.12.10 and Git installed.
It checks file integrity and prepares the project for verification. Git clones can skip
this step. The data ZIP supplies `code/.source_evidence/` directly.
The 27 saved notebooks can be read without running Python.

## 1. Install and verify

From the project root, create an environment:

```bash
cd code
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate`
on macOS/Linux, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-repro.txt
python -m experiments.reproduce_tracked
python -m experiments.reproduce_source --audit-only
```

The tracked command performs dependency checks, the repository release audit and pytest
with `--clean-clone`; checks requiring unavailable local inputs are skipped.
The source audit verifies 59 tasks, 184 declared outputs and all four
notebook sequences without reading external data.

## 2. Get and verify source evidence

The [source-data ZIP](https://github.com/alleg88/Cost-Aware-BTC-and-US-Index-Forecasting-with-Machine-Learning-and-LLM-Memory/releases/download/v1.0.0/cost-aware-market-forecasting-sources.zip)
contains the registered bundle. For a Git clone, copy its `code/.source_evidence/`
directory into the checkout's `code/` directory. For the prepared code ZIP, extracting
both archives together already provides this layout. From `code/`, verify:

```bash
python -m experiments.source_evidence verify --source .source_evidence --manifest .source_evidence/source_evidence_manifest.json
```

Verification fails on a missing file or a size/hash mismatch.

A bundle stored elsewhere can be imported with
`python -m experiments.source_evidence restore --source /path/to/bundle --target .source_evidence`.

## 3. Rebuild results and notebooks

```bash
python -m experiments.reproduce_source
python -m experiments.reproduce_notebooks
```

The first command downloads public Binance inputs, stages the immutable snapshots, runs the
59-task graph and compares declared outputs with tracked references. It resumes through
`.rebuild/state`; use `--task TASK_ID` for one dependency closure or `--audit-only` for a
structural check. The second command executes the 27 canonical readers in the order defined
by [`code/notebooks/README.md`](code/notebooks/README.md) and reruns the release audit.

## 4. Fresh acquisitions and cloud calls

Exact comparison uses the frozen bundle because databases and model providers change. A new
news acquisition requires the `acquisition` extra, Google Application Default Credentials
and the tracked English source whitelist. Fresh index exports must preserve the registered
instrument, Bid/Ask, UTC and date contracts.

Fresh sentiment or Reflection Agent calls additionally require Ollama Cloud credentials:

```powershell
$env:OLLAMA_API_KEY = "your-key"
$env:OLLAMA_HOST = "https://ollama.com"
```

The experiment configuration selects the model and prompt. A fresh provider response is a
new sensitivity run, not a byte-identical reconstruction.

Q2 2026 was opened once under the registered frozen protocol. The final readers verify the
completed artifacts; no model, threshold or policy is selected from Q2.
