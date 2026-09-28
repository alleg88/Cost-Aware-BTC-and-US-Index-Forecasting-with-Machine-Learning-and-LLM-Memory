# Notebook RQ5_C reproducibility

Notebook RQ5_C is a deterministic BTC development experiment. It reads the frozen Notebook RQ5_B
activations and bounded inputs ending before 1 July 2025; later rows cannot enter its
identity.

## Environment and artifact audit

From `code/`:

```powershell
py -3.12 -m venv .venv-v
.venv-v\Scripts\python.exe -m pip install -r requirements-v-repro.txt
.venv-v\Scripts\python.exe -m pip install --no-deps -e .
.venv-v\Scripts\python.exe -m experiments.audit_notebook_v_reproducibility
```

The audit checks tracked artifacts, manifests, frozen handoffs and environment identity.

## Exact input audit

Restore these registered files in one data directory:

- `btcusdt_1m_2021_2026.parquet`
- `btcusdt_5min_2021_2026.parquet`
- `btcusdt_1h_2021_2026.parquet`
- `btcusdt_positioning_15min_2021_2026.parquet`

```powershell
.venv-v\Scripts\python.exe -m experiments.audit_notebook_v_reproducibility `
  --verify-inputs --data-root data
```

The bounded aggregate input SHA-256 is
`f69c6402e8a0b0af1ce3d2c8d12c1df175f3c668c64a8907e8a66d2870b2ef6a`.

## Rerun

```powershell
.venv-v\Scripts\python.exe -m experiments.run_event_window_direction_head `
  --stage dev --data-root data
.venv-v\Scripts\python.exe -m experiments.build_notebook_v
```
