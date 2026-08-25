"""Build Notebook 03 (H1 selection) and 03a (frozen forward)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_03 = CODE_ROOT / "notebooks" / "03_all_model_stacking.ipynb"
NOTEBOOK_03A = CODE_ROOT / "notebooks" / "03a_stacking_forward.ipynb"
KERNEL = {"display_name": "MSC Code", "language": "python", "name": "msc-code"}


def _notebook(cells: list) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = KERNEL
    notebook.metadata.language_info = {"name": "python", "version": "3"}
    return notebook


def build_notebook_03(path: Path = NOTEBOOK_03) -> Path:
    cells = [
        nbf.v4.new_markdown_cell(
            """# 03 — All-nine-model Logistic Regression stacking

## Methodology overview

This notebook tests whether a Logistic Regression meta-learner can combine all nine fixed model families from Notebook 02e. No model is removed. Separate stacks are built for No sentiment, DeBERTa and LLM, and for DZ55, DZ65 and DZ75; different dead-zone labels are never mixed.

The meta-learner receives `P(short)` and `P(long)` from every base model: 18 independent inputs, with `P(flat)` implied. It is a fixed `StandardScaler + L2 LogisticRegression(C=0.1, class_weight='balanced')`; no Optuna is used. The base models keep the 180-day history and 15-minute horizon.

Five 2024 BlockingTimeSeriesSplit OOF blocks train the first meta-model. January 2025 is predicted from 2024 OOF only; each completed H1 month is then added before predicting the next month. H1 evaluates 33 fixed confidence/TP/SL policies for each ensemble/DZ. Adequacy requires 50 trades, 15 long, 15 short and four positive months; candidates are ranked by adequacy, robust score, Sortino, net return and trades. H1 freezes one DZ and policy per sentiment/variant. Forward results are not opened in this notebook, and 2026 Q2 remains sealed.

The design follows Nti et al. for Logistic Regression stacking; Sebastião and Godinho supply the unanimity and net-cost controls; Derbentsev et al. motivate tree-family diversity; Lu et al. and Wang et al. motivate inclusion of deep temporal models."""
        ),
        nbf.v4.new_markdown_cell(
            """### 1. Architecture and literature controls
The primary model is the all-nine stack; best single, equal soft vote and unanimity consensus are matched controls."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)

CODE_ROOT = Path.cwd().resolve()
if CODE_ROOT.name == 'notebooks':
    CODE_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiments.all_model_stacking import DEFAULT_ROOT, MODEL_LABELS, MODEL_NAMES, VARIANTS

ROOT = DEFAULT_ROOT
manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
per_dz = pd.read_parquet(ROOT / 'h1_selected_per_dz.parquet')
candidates = pd.read_parquet(ROOT / 'h1_selected_candidates.parquet')
coefficients = pd.read_parquet(ROOT / 'meta_coefficients.parquet')

assert len(MODEL_NAMES) == 9
assert len(per_dz) == 36 and len(candidates) == 12 and len(coefficients) == 486
assert manifest['lockbox_2026_q2_used'] is False
assert set(candidates['variant']) == set(VARIANTS)
print('Validated: 9 base models, 3 sentiment arms, 3 separate DZs, 12 H1-frozen candidates; forward not loaded.')"""
        ),
        nbf.v4.new_code_cell(
            """architecture = pd.DataFrame({
    'Level': ['Base models', 'Meta inputs', 'Meta-learner', 'Controls'],
    'Definition': [
        ', '.join(MODEL_LABELS[name] for name in MODEL_NAMES),
        'P(short) and P(long) from each model — 18 features',
        'StandardScaler + L2 Logistic Regression, C=0.1, balanced',
        'Best single; equal soft vote; unanimity consensus',
    ],
})
print('Table 1 — Fixed stacking architecture')
display(architecture)"""
        ),
        nbf.v4.new_markdown_cell(
            """### 2. H1 policy selection for every DZ
Each row is selected from 33 policies using causal January-June 2025 predictions. Failed adequacy guards are shown, not removed."""
        ),
        nbf.v4.new_code_cell(
            """h1_table = per_dz[[
    'Arm', 'variant', 'width_bps', 'selected_base_model', 'tau', 'tp_bps', 'sl_bps',
    'hold_minutes', 'trades', 'pooled_net', 'pooled_sortino', 'pooled_sharpe',
    'positive_segments', 'n_long', 'n_short'
]].copy()
h1_table['selected_base_model'] = h1_table['selected_base_model'].map(MODEL_LABELS).fillna('All nine')
h1_table['Pass guards'] = (
    (h1_table['trades'] >= 50) & (h1_table['n_long'] >= 15) &
    (h1_table['n_short'] >= 15) & (h1_table['positive_segments'] >= 4)
)
h1_table.columns = [
    'Sentiment', 'Variant', 'DZ', 'Model set', 'Threshold', 'TP', 'SL', 'Hold (min)',
    'Trades', 'Net return', 'Sortino', 'Sharpe', 'Positive months', 'Long', 'Short', 'Pass guards'
]
h1_table = h1_table.sort_values(['Sortino', 'Sharpe', 'Net return'], ascending=False, ignore_index=True)
print('Table 2 — Full H1 selected-policy comparison by DZ')
display(h1_table.round(3))"""
        ),
        nbf.v4.new_markdown_cell(
            """### 3. H1-frozen candidates passed to Notebook 03a
One DZ and policy are frozen for each sentiment/variant using the same adequacy-first economic rule."""
        ),
        nbf.v4.new_code_cell(
            """candidate_table = candidates[[
    'Arm', 'variant', 'width_bps', 'selected_base_model', 'tau', 'tp_bps', 'sl_bps',
    'hold_minutes', 'trades', 'pooled_net', 'pooled_sortino', 'pooled_sharpe',
    'positive_segments', 'n_long', 'n_short'
]].copy()
candidate_table['selected_base_model'] = candidate_table['selected_base_model'].map(MODEL_LABELS).fillna('All nine')
candidate_table['Pass guards'] = (
    (candidate_table['trades'] >= 50) & (candidate_table['n_long'] >= 15) &
    (candidate_table['n_short'] >= 15) & (candidate_table['positive_segments'] >= 4)
)
candidate_table.columns = [
    'Sentiment', 'Variant', 'DZ', 'Model set', 'Threshold', 'TP', 'SL', 'Hold (min)',
    'Trades', 'Net return', 'Sortino', 'Sharpe', 'Positive months', 'Long', 'Short', 'Pass guards'
]
candidate_table = candidate_table.sort_values(['Sortino', 'Sharpe', 'Net return'], ascending=False, ignore_index=True)
print('Table 3 — Twelve H1-frozen candidates')
display(candidate_table.round(3))"""
        ),
        nbf.v4.new_markdown_cell(
            """### 4. Meta-learner coefficients
The heatmaps show the final pre-forward Logistic Regression weights for the H1-selected stack DZ in each sentiment arm."""
        ),
        nbf.v4.new_code_cell(
            """stack_choices = candidates.loc[candidates['variant'] == 'stack', ['sentiment_arm', 'Arm', 'width_bps']]
chosen = coefficients.merge(stack_choices, on=['sentiment_arm', 'Arm', 'width_bps'], how='inner')
panels = []
for arm, group in chosen.groupby('sentiment_arm', sort=False):
    matrix = group.pivot(index='feature', columns='class_label', values='coefficient').reindex(columns=['short', 'flat', 'long'])
    panels.append((group['Arm'].iloc[0], int(group['width_bps'].iloc[0]), matrix))
limit = max(abs(matrix.to_numpy()).max() for _, _, matrix in panels)
fig, axes = plt.subplots(1, 3, figsize=(14, 8), constrained_layout=True)
image = None
for axis, (label, width, matrix) in zip(axes, panels):
    image = axis.imshow(matrix, cmap='coolwarm', vmin=-limit, vmax=limit, aspect='auto')
    axis.set_title(f'{label} — DZ{width}')
    axis.set_xticks(range(3), ['short', 'flat', 'long'])
    axis.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
fig.colorbar(image, ax=axes, shrink=0.7, label='Standardised LR coefficient')
fig.suptitle('All-nine Logistic Regression meta-learner coefficients', fontsize=14)
artifact_dir = CODE_ROOT / 'notebooks' / 'artifacts'
artifact_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(artifact_dir / '03_stacking_coefficients.png', dpi=180, bbox_inches='tight')
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Handoff to Notebook 03a

Notebook 03a may evaluate only the 12 rows in Table 3. It cannot change the base-model set, DZ, Logistic Regression, threshold, TP, SL or hold after seeing forward results. Guard failures remain visible rather than being silently excluded."""
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(_notebook(cells), path)
    return path


def build_notebook_03a(path: Path = NOTEBOOK_03A) -> Path:
    cells = [
        nbf.v4.new_markdown_cell(
            """# 03a — Frozen-forward stacking evaluation

## Methodology overview

This notebook continues Notebook 03 and evaluates only its 12 H1-frozen candidates: best single, all-nine soft vote, all-nine unanimity consensus and all-nine Logistic Regression stack for each sentiment arm. The selected DZ, threshold, TP, SL and 15-minute hold cannot change here.

Each July 2025 base model uses the preceding 180 days. The Logistic Regression meta-learner is fitted on 2024 OOF plus completed H1 predictions, then frozen. July 2025-March 2026 is a development-forward evaluation with 5 bps per side and one-minute TP/SL resolution; it is not used for reselection. Sortino is primary, with Sharpe, net return, trades, long/short counts and positive months reported. The final 2026-Q2 lockbox remains sealed."""
        ),
        nbf.v4.new_markdown_cell(
            """### 1. Validate the Notebook 03 freeze
The forward table must match every H1-selected DZ and policy field exactly."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', None)

CODE_ROOT = Path.cwd().resolve()
if CODE_ROOT.name == 'notebooks':
    CODE_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiments.all_model_stacking import DEFAULT_ROOT, MODEL_LABELS, VARIANTS

ROOT = DEFAULT_ROOT
manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
selected = pd.read_parquet(ROOT / 'h1_selected_candidates.parquet')
forward = pd.read_parquet(ROOT / 'forward_summary.parquet')
monthly = pd.read_parquet(ROOT / 'forward_monthly.parquet')
keys = ['sentiment_arm', 'variant']
for field in ['width_bps', 'policy_id', 'tau', 'tp_bps', 'sl_bps', 'max_hold']:
    assert selected.set_index(keys)[field].sort_index().equals(forward.set_index(keys)[field].sort_index())
assert len(forward) == 12 and len(monthly) == 108
assert pd.to_datetime(forward['period_end'], utc=True).max() == pd.Timestamp('2026-04-01', tz='UTC')
assert manifest['lockbox_2026_q2_used'] is False
print('Validated: 12 exact frozen replays, 9 months each; 2026 Q2 sealed.')"""
        ),
        nbf.v4.new_markdown_cell(
            """### 2. Full frozen-forward results
Every candidate is reported, including sparse or economically inadequate rows; no forward row is used to change the experiment."""
        ),
        nbf.v4.new_code_cell(
            """forward_table = forward[[
    'Arm', 'variant', 'width_bps', 'selected_base_model', 'tau', 'tp_bps', 'sl_bps',
    'max_hold', 'trades', 'net_return', 'sortino', 'sharpe', 'positive_months', 'n_long', 'n_short'
]].copy()
forward_table['selected_base_model'] = forward_table['selected_base_model'].map(MODEL_LABELS).fillna('All nine')
forward_table['max_hold'] *= 15
forward_table['Pass forward guards'] = (
    (forward_table['trades'] >= 50) & (forward_table['n_long'] >= 15) &
    (forward_table['n_short'] >= 15) & (forward_table['positive_months'] >= 6)
)
forward_table.columns = [
    'Sentiment', 'Variant', 'DZ', 'Model set', 'Threshold', 'TP', 'SL', 'Hold (min)',
    'Trades', 'Net return', 'Sortino', 'Sharpe', 'Positive months', 'Long', 'Short', 'Pass forward guards'
]
forward_table = forward_table.sort_values(['Sortino', 'Sharpe', 'Net return'], ascending=False, ignore_index=True)
print('Table 1 — Full frozen-forward comparison')
display(forward_table.round(3))"""
        ),
        nbf.v4.new_markdown_cell(
            """### 3. Does stacking beat the best single model?
The comparison is paired within each sentiment arm using candidates frozen in Notebook 03."""
        ),
        nbf.v4.new_code_cell(
            """comparison = []
for arm, group in forward.groupby('sentiment_arm', sort=False):
    single = group.loc[group['variant'] == 'best_single'].iloc[0]
    stack = group.loc[group['variant'] == 'stack'].iloc[0]
    comparison.append({
        'Sentiment': single['Arm'],
        'Single model': MODEL_LABELS[single['selected_base_model']],
        'Single DZ': int(single['width_bps']),
        'Single Sortino': single['sortino'],
        'Single Sharpe': single['sharpe'],
        'Single net': single['net_return'],
        'Stack DZ': int(stack['width_bps']),
        'Stack Sortino': stack['sortino'],
        'Stack Sharpe': stack['sharpe'],
        'Stack net': stack['net_return'],
        'Sortino delta': stack['sortino'] - single['sortino'],
        'Net delta': stack['net_return'] - single['net_return'],
    })
comparison = pd.DataFrame(comparison).sort_values('Stack Sortino', ascending=False, ignore_index=True)
print('Table 2 — All-nine stack versus H1-frozen single control')
display(comparison.round(3))"""
        ),
        nbf.v4.new_markdown_cell(
            """### 4. Monthly stability
The panels show whether a forward result is broad or driven by one isolated month."""
        ),
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True, constrained_layout=True)
for axis, (arm, group) in zip(axes, monthly.groupby('sentiment_arm', sort=False)):
    for variant in VARIANTS:
        series = group.loc[group['variant'] == variant].sort_values('period')
        axis.plot(series['period'], 100 * series['net_return'], marker='o', label=variant)
    axis.axhline(0, color='black', linewidth=0.8)
    axis.set_title(group['Arm'].iloc[0])
    axis.tick_params(axis='x', rotation=55, labelsize=8)
    axis.set_ylabel('Monthly net return (%)')
axes[-1].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
fig.suptitle('Frozen-forward monthly stability')
artifact_dir = CODE_ROOT / 'notebooks' / 'artifacts'
artifact_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(artifact_dir / '03a_stacking_monthly.png', dpi=180, bbox_inches='tight')
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """### 5. Decision
Promotion requires the stack to improve the frozen single control after costs without relying on a sparse trade count."""
        ),
        nbf.v4.new_code_cell(
            """best = forward.sort_values(['sortino', 'sharpe', 'net_return'], ascending=False).iloc[0]
stack_wins = int((comparison['Sortino delta'] > 0).sum())
print(f"Best forward row: {best['Arm']} / {best['variant']} / DZ{int(best['width_bps'])}; "
      f"Sortino {best['sortino']:.3f}, Sharpe {best['sharpe']:.3f}, "
      f"net {best['net_return']:.2%}, trades {int(best['trades'])}.")
print(f'Stacking improves Sortino versus the frozen single control in {stack_wins}/3 sentiment arms.')
print('Decision: reject the all-nine Logistic Regression stack for promotion; retain the best frozen single-model control.')"""
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(_notebook(cells), path)
    return path


def build_all() -> tuple[Path, Path]:
    return build_notebook_03(), build_notebook_03a()


if __name__ == "__main__":
    print(build_all())
