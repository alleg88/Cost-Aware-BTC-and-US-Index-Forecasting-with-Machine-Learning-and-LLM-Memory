"""Build and execute Notebook 04b, the XGBoost admission artifact reader."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "04b_xgboost_strong_move_admission.ipynb"
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
            """# 04b — Calibrated XGBoost Strong-Move Admission

## Objective and frozen protocol

Test whether XGBoost DZ65 can increase the number of trades without changing
any existing Qualified Union v1 decision. Two natural-prevalence sigmoid maps
were fitted once on 2024 OOF: **calibrated P(strong move)** and calibrated
**conditional direction** given a move. Policy selection uses 2025-Q1 only;
April–June confirmation and the forward replay are conditional on the preceding
gate. The 2026-Q2 lockbox remains sealed.

This notebook is an artifact reader. It does not fit a model, calibrator, or
threshold."""
        ),
        nbf.v4.new_code_cell(
            """import hashlib
import json

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

CACHE = CODE_ROOT / 'experiments' / 'cache' / 'xgb_strong_move_admission'

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

summary = json.loads((CACHE / 'summary.json').read_text(encoding='utf-8'))
manifest = json.loads((CACHE / 'manifest.json').read_text(encoding='utf-8'))
for filename, expected in manifest['artifact_hashes'].items():
    assert sha256(CACHE / filename) == expected, filename
assert summary['lockbox_2026_q2_used'] is False
assert manifest['lockbox_2026_q2_used'] is False
print('Validated 04b artifacts and the staged access contract.')"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. What calibration changed

The original three-class probabilities came from a class-balanced XGBoost.
They are decomposed into an opportunity head and a direction-given-opportunity
head, then mapped back to natural prevalence using disjoint 2024 OOF data.
H1 is confirmation only."""
        ),
        nbf.v4.new_code_cell(
            """calibration = pd.read_csv(CACHE / 'calibration_metrics.csv')
display(calibration.round(6))

move = calibration.loc[calibration['target'].eq('move')].copy()
plot = move.pivot(index='period', columns='arm', values='mean_probability')
truth = move.drop_duplicates('period').set_index('period')['prevalence']
plot.insert(0, 'observed prevalence', truth)
ax = plot.plot.bar(figsize=(8, 3.8), color=['#264653', '#e76f51', '#2a9d8f'])
ax.set_title('Strong-move probability versus observed natural prevalence')
ax.set_ylabel('Probability')
ax.set_xlabel('')
ax.grid(axis='y', alpha=.25)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

display(pd.DataFrame(summary['calibrators']).T.rename_axis('head').round(6))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. Union-flat add-on rule

XGBoost is never allowed to vote against or replace Union v1. A new trade is
permitted only when Union is flat, there is no member conflict, LSTM DZ55 and
Linear SVM DZ75 have the same latent side, XGBoost supports that side, and both
calibrated thresholds pass. The full registered grid is shown below; costs are
5 bps per side under the frozen TP200/SL100/one-bar execution protocol."""
        ),
        nbf.v4.new_code_cell(
            """grid = pd.read_csv(CACHE / 'h1_selection_grid.csv')
columns = [
    'move_threshold', 'direction_threshold', 'addon_signal_bars',
    'addon_trades', 'addon_long_trades', 'addon_short_trades',
    'addon_gross_bps_per_trade', 'addon_net',
    'combined_trades', 'combined_net', 'combined_sortino', 'eligible',
]
display(grid[columns].round(6))

active = grid.loc[grid['addon_trades'].gt(0)].copy()
fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
for threshold, group in active.groupby('direction_threshold'):
    ax.scatter(
        group['addon_trades'], group['addon_gross_bps_per_trade'],
        label=f'direction ≥ {threshold:g}', s=55,
    )
ax.axhline(10.0, color='#2a9d8f', linestyle='--', label='round-trip cost')
ax.axhline(0.0, color='#6c757d', linewidth=.8)
ax.set_title('More XGBoost trades did not preserve gross edge')
ax.set_xlabel('Add-on trades in 2025-Q1')
ax.set_ylabel('Gross bps per add-on trade')
ax.grid(alpha=.25)
ax.legend()
plt.show()

permissive = grid.loc[
    grid['move_threshold'].eq(0.10) & grid['direction_threshold'].eq(0.60),
    columns,
]
display(permissive.round(6))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Gate decision and forward replay

Selection required at least ten add-on trades, both sides represented, gross
edge above round-trip cost, positive add-on net return, and improvement in the
combined net return without reducing Sortino. Only a policy passing selection
could enter April–June confirmation; only a policy passing full H1 could load
the development-forward predictions."""
        ),
        nbf.v4.new_code_cell(
            """decision = pd.DataFrame({
    'eligible_Q1_policies': [summary['eligible_selection_policies']],
    'selected_policy': [summary['selected_policy']],
    'H1_pass': [summary['h1_pass']],
    'forward_loaded': [summary['forward_loaded']],
    'forward_promoted': [summary['forward_promoted']],
    'decision': [summary['decision']],
    'final_ensemble': [summary['final_ensemble']],
})
display(decision)

assert summary['eligible_selection_policies'] == 0
assert summary['selected_policy'] is None
assert summary['h1_pass'] is False
assert summary['forward_loaded'] is False
assert summary['forward_promoted'] is False
assert summary['decision'] == 'reject_xgboost_keep_union_v1'
assert summary['final_ensemble'] == 'qualified_union_v1'
assert not (CACHE / 'forward_combined_per_bar.parquet').exists()
print('Forward replay was not opened because the H1 gate failed.')"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Final ensemble handoff

Calibration corrected probability levels but could not manufacture an
economically useful high-confidence tail. At the most permissive registered
pair (`P(move) ≥ 0.10`, direction confidence `≥ 0.60`), XGBoost produced 80
add-on trades but averaged **−8.01 gross bps per trade** before costs and
**−14.41% net**. At direction confidence `≥ 0.70`, it produced no candidates.

Therefore XGBoost is rejected as a trading member, no forward data was loaded,
and **Qualified Union v1 remains the frozen final ensemble** for the Reflection
Agent handoff. The 2026-Q2 lockbox remains sealed."""
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
