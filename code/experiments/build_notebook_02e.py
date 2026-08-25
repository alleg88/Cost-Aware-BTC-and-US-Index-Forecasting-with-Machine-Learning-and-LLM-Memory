"""Build Notebook 02e: all-model sentiment policy calibration."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "02e_all_model_sentiment_policy.ipynb"


def build_notebook(path: Path = NOTEBOOK) -> Path:
    cells = [
        nbf.v4.new_markdown_cell(
            """# 02e — All-model sentiment policy calibration

## Methodology overview

This notebook continues Notebook 02d and uses the same nine model families, four arms (**No sentiment**, **DeBERTa**, **LLM-matched** and **LLM-full**), DZ55/DZ65/DZ75, 180-day history, fixed 15-minute horizon and 5 bps per side. It does not repeat the 2024 blocking-fold or raw-sentiment experiment.

All three sentiment arms see the same sources; they differ only in the scorer and in how each weights what it reads. **DeBERTa against LLM-full** is the layer-versus-layer comparison, and **LLM-matched against LLM-full** isolates how much of any difference is the scalar rather than the structure.

For each January–June 2025 month, each Model/Arm/DZ is fitted only on the preceding 180 days and predicts that month. The six causal prediction blocks evaluate 33 pre-declared policies: 11 confidence thresholds crossed with TP/SL 150/75, 150/100 and 200/100 bps; hold is one M15 bar. Selection ranks economically adequate policies (at least 50 trades, 15 long, 15 short and four positive months) by Sortino, then Sharpe and net return.

The selected policy is frozen and applied to the exact Notebook 02d July-2025 fit and July-2025–March-2026 predictions. One-minute candles are used only to resolve TP/SL execution after the M15 signal. The forward span cannot change the policy, and 2026 Q2 remains sealed."""
        ),
        nbf.v4.new_markdown_cell(
            """### 1. Load and validate artifacts
The checks require every Model/Arm root and one unique policy and forward row per Model/Arm/DZ."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys
import pandas as pd
from IPython.display import display

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)

CODE_ROOT = Path.cwd().resolve()
if CODE_ROOT.name == 'notebooks':
    CODE_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiments.all_model_sentiment_policy import ARMS, DEFAULT_ROOT, WIDTHS, validate_policy_model_artifacts
from experiments.all_model_sentiment_policy_scoreboard import build_scoreboards
from experiments.all_model_sentiment_scoreboard import build_scoreboards as build_raw_scoreboards
from experiments.raw_hold_control import MODEL_LABELS, MODEL_NAMES

ROOT = DEFAULT_ROOT
for arm in ARMS:
    for model_name in MODEL_NAMES:
        validate_policy_model_artifacts(ROOT / arm / model_name, model_name=model_name)

tables = build_scoreboards(ROOT)
policies = tables['policies']
economics = tables['economics']
raw_economics = build_raw_scoreboards()['economics']
keys = ['sentiment_arm', 'model_name', 'width_bps']
expected = len(ARMS) * 9 * len(WIDTHS)
assert len(policies) == expected and policies[keys].drop_duplicates().shape[0] == expected
assert len(economics) == expected and economics[keys].drop_duplicates().shape[0] == expected
assert len(raw_economics) == expected and raw_economics[keys].drop_duplicates().shape[0] == expected
assert set(policies['width_bps']) == set(WIDTHS)
print(f'Validated: {expected} raw rows, {len(ARMS) * 9} policy roots, {expected} selected policies '
      f'and {expected} calibrated forward rows across arms {ARMS}.')"""
        ),
        nbf.v4.new_markdown_cell(
            """### 2. H1 2025 policy selection
Each row is the policy selected from six causal monthly prediction blocks; these are calibration results, not forward results."""
        ),
        nbf.v4.new_code_cell(
            """h1 = policies[[
    'model_name', 'Arm', 'width_bps', 'tau', 'tp_bps', 'sl_bps',
    'hold_minutes', 'trades', 'pooled_net', 'pooled_sortino',
    'pooled_sharpe', 'positive_segments',
]].copy()
h1['model_name'] = h1['model_name'].map(MODEL_LABELS)
h1.columns = [
    'Model', 'Sentiment', 'DZ', 'Threshold', 'TP', 'SL', 'Hold (min)',
    'Trades', 'Net return', 'Sortino', 'Sharpe', 'Positive months',
]
h1 = h1.sort_values(
    ['Sortino', 'Sharpe', 'Net return'], ascending=False, ignore_index=True
)
print('Table 1 — Full H1 2025 selected-policy comparison')
display(h1.round(3))"""
        ),
        nbf.v4.new_markdown_cell(
            """### 3. Frozen-forward economics
The H1-selected threshold and TP/SL are replayed unchanged on July 2025–March 2026; no row is removed because the final research decision is manual."""
        ),
        nbf.v4.new_code_cell(
            """forward = economics[[
    'model_name', 'Arm', 'width_bps', 'tau', 'tp_bps', 'sl_bps',
    'max_hold', 'trades', 'net_return', 'sortino', 'sharpe', 'positive_months',
]].copy()
forward['model_name'] = forward['model_name'].map(MODEL_LABELS)
forward['max_hold'] = forward['max_hold'] * 15
forward.columns = [
    'Model', 'Sentiment', 'DZ', 'Threshold', 'TP', 'SL', 'Hold (min)',
    'Trades', 'Net return', 'Sortino', 'Sharpe', 'Positive months',
]
forward = forward.sort_values(['Sortino', 'Sharpe', 'Net return'], ascending=False).reset_index(drop=True)
print('Table 2 — Full frozen-forward calibrated economic comparison')
display(forward.round(3))"""
        ),
        nbf.v4.new_markdown_cell(
            """### 4. Five uncalibrated model-family leaders
This is the policy-free Notebook 02d control: each family keeps its best Sentiment/DZ row by raw-forward Sortino, then Sharpe and net return."""
        ),
        nbf.v4.new_code_cell(
            """raw_leaders = raw_economics.sort_values(
    ['sortino', 'sharpe', 'net_return'], ascending=False
).drop_duplicates('model_name').head(5).copy()
raw_leaders['model_name'] = raw_leaders['model_name'].map(MODEL_LABELS)
raw_leaders = raw_leaders[[
    'model_name', 'Arm', 'width_bps', 'hold_minutes', 'trades',
    'net_return', 'sortino', 'sharpe', 'positive_months',
]]
raw_leaders.columns = [
    'Model', 'Sentiment', 'DZ', 'Hold (min)', 'Trades', 'Net return',
    'Sortino', 'Sharpe', 'Positive months',
]
print('Table 3 — Five uncalibrated model-family leaders')
display(raw_leaders.round(3))"""
        ),
        nbf.v4.new_markdown_cell(
            """### 5. Five calibrated model-family leaders
Each family keeps its best Arm/DZ/policy row; the table then reports the top five families by forward Sortino, with Sharpe and net return as tie-breakers. Because a family shortlist can only show one row per model, the per-arm leaders are reported separately below — otherwise an arm whose best row shares a family with another arm's would be invisible."""
        ),
        nbf.v4.new_code_cell(
            """leaders = economics.sort_values(
    ['sortino', 'sharpe', 'net_return'], ascending=False
).drop_duplicates('model_name').head(5).copy()
leaders['model_name'] = leaders['model_name'].map(MODEL_LABELS)
leaders['max_hold'] = leaders['max_hold'] * 15
leaders = leaders[[
    'model_name', 'Arm', 'width_bps', 'tau', 'tp_bps', 'sl_bps',
    'max_hold', 'trades', 'net_return', 'sortino', 'sharpe', 'positive_months',
]]
leaders.columns = [
    'Model', 'Sentiment', 'DZ', 'Threshold', 'TP', 'SL', 'Hold (min)',
    'Trades', 'Net return', 'Sortino', 'Sharpe', 'Positive months',
]
print('Table 4 — Five calibrated model-family leaders')
display(leaders.round(3))

per_arm = economics.sort_values(
    ['sortino', 'sharpe', 'net_return'], ascending=False
).drop_duplicates('sentiment_arm').copy()
per_arm['model_name'] = per_arm['model_name'].map(MODEL_LABELS)
per_arm['max_hold'] = per_arm['max_hold'] * 15
per_arm['gross_bps'] = per_arm['gross_return'] / per_arm['trades'].clip(lower=1) * 1e4
per_arm = per_arm[[
    'Arm', 'model_name', 'width_bps', 'tau', 'tp_bps', 'sl_bps', 'max_hold',
    'trades', 'net_return', 'sortino', 'sharpe', 'positive_months', 'gross_bps',
]]
per_arm.columns = [
    'Arm', 'Model', 'DZ', 'Threshold', 'TP', 'SL', 'Hold (min)', 'Trades',
    'Net return', 'Sortino', 'Sharpe', 'Positive months', 'Gross bps/trade',
]
print('\\nTable 5 — Best calibrated row of each arm')
display(per_arm.round(3).reset_index(drop=True))

paired = economics.pivot_table(
    index=['model_name', 'width_bps'], columns='sentiment_arm', values='net_return')
cells = len(paired)
print('\\nCalibrated deltas against the no-sentiment control, over '
      f'{cells} matched Model/DZ cells:')
for arm, label in (('classic', 'DeBERTa'), ('llm', 'LLM-matched'), ('llm_full', 'LLM-full')):
    if arm not in paired:
        continue
    delta = paired[arm] - paired['none']
    print(f'  {label:12s} better in {int((delta > 0).sum()):2d}/{cells} cells, '
          f'median delta {delta.median():+.2%}')
if {'llm_full', 'classic'} <= set(paired.columns):
    delta = paired['llm_full'] - paired['classic']
    print(f'  LLM-full vs DeBERTa (layer vs layer): better in '
          f'{int((delta > 0).sum())}/{cells}, median delta {delta.median():+.2%}')

raw_by_model = raw_economics.sort_values(
    ['sortino', 'sharpe', 'net_return'], ascending=False
).drop_duplicates('model_name').set_index('model_name')
calibrated_by_model = economics.sort_values(
    ['sortino', 'sharpe', 'net_return'], ascending=False
).drop_duplicates('model_name').set_index('model_name')
improvement = calibrated_by_model[['sortino', 'net_return']].join(
    raw_by_model[['sortino', 'net_return']],
    how='inner',
    lsuffix='_calibrated',
    rsuffix='_raw',
)
assert len(improvement) == 9
sortino_improvements = int(
    (improvement['sortino_calibrated'] > improvement['sortino_raw']).sum()
)
net_improvements = int(
    (improvement['net_return_calibrated'] > improvement['net_return_raw']).sum()
)
print(
    f'Calibration improves the best-family Sortino in {sortino_improvements}/9 '
    f'models and best-family net return in {net_improvements}/9 models.'
)"""
        ),
        nbf.v4.new_markdown_cell(
            """### 6. Handoff
The five rows above are forward-development candidates, not final lockbox winners. They may proceed to ensemble testing, while 2026 Q2 stays untouched until the final specification is frozen."""
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (msc-code venv)",
                "language": "python",
                "name": "msc-code",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    return path


if __name__ == "__main__":
    print(build_notebook())
