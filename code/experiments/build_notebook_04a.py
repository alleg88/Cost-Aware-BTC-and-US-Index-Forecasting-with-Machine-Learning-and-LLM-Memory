"""Build and execute Notebook 04a, the SVM temperature artifact reader."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "04a_svm_temperature_calibration.ipynb"
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
            """# 04a — SVM Temperature Calibration

## Objective and frozen hypothesis

Test one **class-preserving temperature** for the frozen Linear SVM DZ75
probabilities. The six-value grid is selected by natural-prevalence multiclass
log loss on 2024 OOF and confirmed on H1-2025 without retuning. This experiment
cannot change SVM classes, and SVM's frozen `tau=0` means **Union v1 economics
are unchanged** whatever class-preserving temperature is selected.

The notebook is an artifact reader: selection and confirmation were completed
outside notebook cells. The 2026-Q2 lockbox remains sealed."""
        ),
        nbf.v4.new_code_cell(
            """import hashlib
import json

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

CACHE = CODE_ROOT / 'experiments' / 'cache' / 'svm_temperature_calibration'

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
assert summary['forward_loaded'] is False
assert summary['lockbox_2026_q2_used'] is False
print('Validated 04a artifacts; no forward or Q2 data was loaded.')"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. 2024 OOF selection

`T<1` sharpens and `T>1` softens every class through the same monotone
transformation. The fixed grid below reports every candidate; the selector is
multiclass log loss, with the smaller temperature only as an exact-tie rule."""
        ),
        nbf.v4.new_code_cell(
            """grid = pd.read_csv(CACHE / 'temperature_grid.csv')
display(grid.round(6))
selected = float(summary['selected_temperature'])
assert grid['changed_classes'].eq(0).all()

fig, ax = plt.subplots(figsize=(7.5, 3.5), layout='constrained')
ax.plot(grid['temperature'], grid['log_loss'], marker='o', color='#457b9d')
ax.axvline(selected, color='#e76f51', linestyle='--', label=f'selected T={selected:g}')
ax.set_title('2024 OOF natural-prevalence log loss')
ax.set_xlabel('Temperature')
ax.set_ylabel('Multiclass log loss')
ax.grid(alpha=.25)
ax.legend()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. H1 confirmation

The selected temperature is applied unchanged to the six H1 monthly-fit
panels. Confirmation checks log loss, Brier score and exact class invariance;
it does not choose another value from H1."""
        ),
        nbf.v4.new_code_cell(
            """h1 = pd.read_csv(CACHE / 'h1_confirmation.csv').set_index('arm')
display(h1.round(6))
decision = pd.DataFrame({
    'selected_temperature': [selected],
    'changed_classes_2024': [summary['changed_classes_2024']],
    'changed_classes_h1': [summary['changed_classes_h1']],
    'h1_log_loss_improvement': [summary['h1_log_loss_improvement']],
    'h1_brier_improvement': [summary['h1_brier_improvement']],
    'confirmation_pass': [summary['h1_confirmation_pass']],
})
display(decision.round(8))
assert selected == 1.0
assert summary['changed_classes_2024'] == summary['changed_classes_h1'] == 0"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Decision and XGBoost handoff

The registered grid selected the identity control, **T=1.0**. SVM calibration
therefore neither improves nor degrades H1 probabilities, and all class
decisions remain exact. This closes the SVM temperature experiment without
inventing a new threshold.

**Union v1 economics are unchanged.** Notebook 04b now evaluates XGBoost on a
different question: calibrated `P(strong move)` and calibrated conditional
LONG/SHORT direction. XGBoost may add a trade only when Union is flat and the
two qualified experts independently support its side. The 2026-Q2 lockbox
remains sealed."""
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
