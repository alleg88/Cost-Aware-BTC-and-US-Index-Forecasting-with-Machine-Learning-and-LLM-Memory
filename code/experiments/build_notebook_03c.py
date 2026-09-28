"""Build and execute the artifact-only qualified Union v1 notebook."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "08_RQ2_C_BTC_qualified_union_ensemble.ipynb"
KERNEL = {"display_name": "MSC Code", "language": "python", "name": "msc-code"}
COLAB_SETUP = """# Google Colab / local setup
import os, sys, subprocess
from pathlib import Path
if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    CODE_ROOT = Path("/content/drive/MyDrive/msc project/code")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(CODE_ROOT), "catboost==1.2.10", "rapidfuzz==3.14.3"])
else:
    CODE_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").exists())
os.chdir(CODE_ROOT); sys.path.insert(0, str(CODE_ROOT)); CODE = CODE_ROOT
"""


def build_notebook(path: Path = NOTEBOOK) -> Path:
    cells = [
        nbf.v4.new_code_cell(COLAB_SETUP),
        nbf.v4.new_markdown_cell(
            """# 03c — Qualified Union v1

## Immutable baseline

This notebook is the reader for the frozen **qualified Union v1** handoff. The
ensemble contains no-sentiment LSTM DZ55 and Linear SVM DZ75, uses a union with
an opposite-signal veto, and executes TP200/SL100 for one M15 bar at 5 bps per
side. All fitting, member screening and execution were completed by the frozen
runner; this notebook only validates and presents its hashed artifacts.

The development-forward result is **74 trades** from July 2025 through March
2026. It remains development-forward evidence because the underlying models
were inspected earlier. The **2026-Q2 lockbox remains sealed**."""
        ),
        nbf.v4.new_code_cell(
            """import hashlib
import json

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

CACHE = CODE_ROOT / 'experiments' / 'cache' / 'qualified_union_v1'
LOCKBOX = pd.Timestamp('2026-04-01', tz='UTC')

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

protocol = json.loads((CACHE / 'protocol.json').read_text(encoding='utf-8'))
manifest = json.loads((CACHE / 'manifest.json').read_text(encoding='utf-8'))
for filename, expected in manifest['artifact_hashes'].items():
    assert sha256(CACHE / filename) == expected, filename
assert protocol['protocol_version'] == 'qualified-union-v1'
assert manifest['forward_replay_count'] == 1
assert protocol['lockbox_2026_q2_used'] is False
print('Validated hashed Union v1 artifacts; Q2 was not used.')"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Frozen construction

The two members keep their Notebook 02e gates. Temperature scaling or later
calibration cannot revise these baseline signals. In particular, SVM has
`tau=0`, so its class-preserving temperature experiment cannot change Union v1
trade count or economics."""
        ),
        nbf.v4.new_code_cell(
            """membership = pd.DataFrame(protocol['members'])
execution = pd.DataFrame([protocol['execution']])
print('Table 1 — immutable membership')
display(membership)
print('Table 2 — immutable execution')
display(execution)"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. H1 construction evidence

H1-2025 is retained as development evidence for later admission-gate selection.
It is not mixed with the July-2025–March-2026 replay."""
        ),
        nbf.v4.new_code_cell(
            """summary = pd.read_csv(CACHE / 'summary.csv').set_index('phase')
monthly = pd.read_csv(CACHE / 'monthly.csv')
h1 = summary.loc[['h1']]
display(h1.round(4))
assert int(h1.iloc[0]['trades']) == 88"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Frozen development-forward result

Union v1 combines complementary rare signals without learned weights. The
published result is 74 trades: 32 LONG and 42 SHORT, +6.315% net, Sortino 2.106
and Sharpe 1.165 after the 10 bps round trip."""
        ),
        nbf.v4.new_code_cell(
            """forward = summary.loc[['forward']]
display(forward.round(4))
row = forward.iloc[0]
assert int(row['trades']) == 74
assert int(row['long_trades']) == 32 and int(row['short_trades']) == 42
assert abs(float(row['net_return']) - 0.06315) < 5e-5

forward_monthly = monthly.loc[monthly['phase'].eq('forward')].copy()
fig, ax = plt.subplots(figsize=(8.0, 3.5), layout='constrained')
ax.bar(forward_monthly['month'], forward_monthly['net_return'], color='#457b9d')
ax.axhline(0, color='black', linewidth=.8)
ax.set_title('Qualified Union v1 monthly net return')
ax.set_ylabel('Net return')
ax.tick_params(axis='x', rotation=45)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Frozen handoff to 04a and 04b

Notebook 04a may calibrate SVM probabilities but cannot rewrite this baseline.
Notebook 04b tests whether XGBoost can add trades only when calibrated strong-
move and conditional-direction probabilities pass independent H1 gates.

**XGBoost remains a challenger** until it increases frequency and improves
net-of-cost quality without selecting any rule from forward. Failure leaves
qualified Union v1 unchanged. The Reflection Agent receives only the final
promoted manifest; the 2026-Q2 lockbox remains sealed."""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = KERNEL
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    return path


def execute_notebook(path: Path = NOTEBOOK) -> Path:
    notebook = nbf.read(path, as_version=4)
    executed = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="msc-code",
        resources={"metadata": {"path": str(CODE_ROOT)}},
    ).execute()
    nbf.write(executed, path)
    return path


def main() -> int:
    execute_notebook(build_notebook())
    print(NOTEBOOK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
