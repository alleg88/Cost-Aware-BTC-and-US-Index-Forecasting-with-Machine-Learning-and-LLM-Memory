"""Reader-facing notebook structure, independent of experiment calculations."""
import json
from pathlib import Path
import re

import pytest


NOTEBOOKS = sorted((Path(__file__).parents[1] / 'notebooks').glob('*.ipynb'))


def _text(value):
    return ''.join(value) if isinstance(value, list) else value


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: p.stem[:8])
def test_each_notebook_opens_with_methods_and_ends_with_results(path):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    prose = [cell for cell in notebook['cells'] if cell['cell_type'] == 'markdown']
    opening = _text(prose[0]['source'])
    closing = _text(notebook['cells'][-1]['source'])
    assert '## Methodology' in opening
    assert len(opening.split()) <= 210
    assert notebook['cells'][-1]['cell_type'] == 'markdown'
    assert re.search(r'^## Results\s*$', closing, re.M)
    assert re.search(r'(?:^|\n)Takeaway: \S.*[.!?]\s*$', closing, re.S)


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: p.stem[:8])
def test_visible_prose_has_one_takeaway_and_no_internal_next_steps(path):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    visible = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            visible.append(_text(cell['source']))
        for output in cell.get('outputs', []):
            data = output.get('data', {})
            if 'text/markdown' in data:
                visible.append(_text(data['text/markdown']))
            if output.get('output_type') == 'stream':
                visible.append(_text(output['text']))
    text = '\n'.join(visible)
    assert len(re.findall(r'Takeaway:', text, re.I)) == 1
    assert not re.search(r'what follows|next steps|Q2 remains unused|move to the evidence matrix|then run.{0,60}lockbox', text, re.I)


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: p.stem[:8])
def test_every_displayed_table_has_a_preceding_explanation(path):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    explanation = ''
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            # Headings alone are not an explanation.
            explanation = '\n'.join(line for line in _text(cell['source']).splitlines() if not line.startswith('#'))
        for output in cell.get('outputs', []):
            data = output.get('data', {})
            if 'text/markdown' in data:
                explanation = _text(data['text/markdown'])
            if '<table' in _text(data.get('text/html', '')).lower():
                assert len(explanation.split()) >= 8, (path.name, cell['id'], 'missing table explanation')
                explanation = ''
