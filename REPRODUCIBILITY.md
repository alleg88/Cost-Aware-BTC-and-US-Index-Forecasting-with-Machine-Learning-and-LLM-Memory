# How to run the code

The repository separates tracked verification, immutable source evidence and generated
calculations.

| Level | Reproduced from | Additional requirement |
|---|---|---|
| Tracked code, installation and tests | Git checkout, pinned Python requirements, release audit and `--clean-clone` tests | Package-download network access |
| Saved executed notebooks | 22 canonical readers with saved tables and figures | None for inspection |
| Public, checksum-verified downloads | Binance BTCUSDT candles, futures metrics and funding | Access to `data.binance.vision` |
| Portable source-evidence bundle | Dukascopy/JForex USA500, USATECH and VIX; English approved-source GDELT GKG via BigQuery; FRED, Fear & Greed, Truth Social/direct-event snapshots; frozen provider evidence | Supplied inside Rebuild, or restore a matching bundle into `code/.source_evidence/` |
| Generated calculations | Normalised data, features, fits, predictions, ensembles, channels and result tables | Run the registered source-to-results graph |

Rebuild includes source snapshots, frozen provider responses, prepared inputs for
the first notebook and completed final-test evidence. It regenerates prediction
and result caches; additional Binance downloads need several GB of disk space.
The smaller Client contains code and saved notebook outputs, without raw data,
calculated caches or developer tests.

## Download and setup

Download the latest [Release-Client.zip](https://github.com/alleg88/Cost-Aware-BTC-and-US-Index-Forecasting-with-Machine-Learning-and-LLM-Memory/releases/latest/download/Release-Client.zip)
to browse code and results, or [Release-Rebuild.zip](https://github.com/alleg88/Cost-Aware-BTC-and-US-Index-Forecasting-with-Machine-Learning-and-LLM-Memory/releases/latest/download/Release-Rebuild.zip)
to recalculate experiments. The [release page](https://github.com/alleg88/Cost-Aware-BTC-and-US-Index-Forecasting-with-Machine-Learning-and-LLM-Memory/releases/latest)
also provides SHA-256 checksums.

Extract `Release-Client.zip` and open the notebooks in `code/notebooks/` to inspect
their code and saved results. No starter notebook, runtime or data download is needed
for viewing. Google Colab can display an uploaded `.ipynb` without running its cells.

For a rerun, use Python 3.11 or newer and supply the files listed in
[`code/notebooks/DATA.md`](code/notebooks/DATA.md). Select **Run all** in the research
notebook. Its first cell checks the inputs before installing missing libraries.
In a new Colab session, it explicitly asks for `Release-Client.zip`, then any missing
data files by name. Choose the requested files to continue; result folders use the
code-relative `Notebook-inputs.zip` layout described in the input guide. Colab files do not persist after the
runtime is deleted. Direct Client reruns preserve existing package versions, so
numerical equivalence with the pinned reference environment is not assumed.

For calculation, choose an ordinary notebook from **Rebuild/notebooks/** or from
the extracted **Release-Rebuild.zip**, and select **Run all**. No separate starter
or mode selection is needed. In Colab, the first cell explicitly requests
**Release-Rebuild.zip** (project code and rebuild inputs); choose it to continue.
The first cell verifies the archive, installs missing packages, and runs only the
producers needed by that notebook and their upstream dependencies. Stage names and
process output are visible there. New notebook tables and figures appear in the
following cells, using the same notebook kernel. Run all does not run CI/tests.

The first notebook uses the two supplied normalised data files and fits its models
in its ordinary cells; it does not reconstruct those inputs. Later notebooks
automatically build the inputs they read, which may require public downloads and
hours of computation. Valid upstream receipts are reused only when inputs, outputs,
registered task settings and package versions match. Requested producer commands
run again, but some experiment recipes also resume their own internal checkpoints.
Q2 readers only verify completed frozen evidence: they do not reopen the lockbox.

Notebook 18 has a separate Colab setup that downloads verified public files
directly and reconstructs its saved LLM weighting comparison without fresh API calls.

Compact Rebuild preserves the original market/news snapshots and frozen LLM/agent
responses. It recomputes the channel development chain J, M, N, O, P, Q, R, U, V and W
as needed, instead of shipping their large tables. New fits use the same periods,
features, model settings and chronological rules. Derived run identities and trade
counts may change; raw-data hashes, causal joins, artifact integrity and Q2 seals
remain checked. Internal consistency between stages of the new run is still required.

Direct Rebuild execution preserves the host kernel's installed packages, including
Colab's native Python. Actual versions and input-preparation status are written to
`code/.rebuild/notebooks/<notebook>/preparation.json`; detailed stage logs are in
`code/.rebuild/state/`. `INPUTS_READY` means preparation finished, not that all
notebook cells succeeded. Completion of every cell must be checked in the notebook.
Neither this status nor successful execution proves numerical equivalence. For the
fixed reference environment, use Python 3.12.10 with `requirements-repro.txt` below.
A full rebuild can exceed Colab's resource limits.

The commands below are for a full Git checkout. Rebuild notebook calculations use
the **Run all** steps above. Existing code/source ZIP exports and Git clones remain
supported. For a legacy code ZIP, run
`python prepare_project.py` once from the extracted project root; a Git clone skips this.
The rebuild ZIP validates all supplied bytes but excludes raw source inputs from the
local Git reference snapshot.

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
The source audit reports the registered task/output counts and checks all four
notebook sequences without reading external data. The historical full-checkout
graph stages frozen channel handoffs; compact Rebuild registers their producers.

## 2. Get and verify source evidence

The [source-data ZIP](https://github.com/alleg88/Cost-Aware-BTC-and-US-Index-Forecasting-with-Machine-Learning-and-LLM-Memory/releases/download/v1.0.0/cost-aware-market-forecasting-sources.zip)
contains the historical full-checkout bundle. Compact Rebuild already contains its
matching reduced source bundle; do not replace its manifest with this older one.
For a Git clone, copy the downloaded bundle's `code/.source_evidence/`
directory into the checkout's `code/` directory. Also copy the current
`code/experiments/cache/reflection_ensemble_v5/` folder and
`code/sentiment/raw/index_deepseek_identity.json` from Release-Rebuild.zip for the
latest LLM comparisons. For the prepared code ZIP, extracting
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

The first command downloads public Binance inputs, stages the immutable snapshots and
runs the registered graph. The full-checkout graph compares outputs with tracked
references; compact Rebuild explicitly omits that historical-result comparison.
Neither successful execution nor an omitted comparison proves numerical equivalence.
It resumes through
`.rebuild/state`; use `--task TASK_ID` for one dependency closure or `--audit-only` for a
structural check. The second command executes the 22 canonical readers in the order defined
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
