# Data for rerunning

**Rebuild:** open the selected Rebuild notebook and choose **Run all**. In Colab,
upload **Release-Rebuild.zip** when requested. The first cell prepares that notebook's
inputs automatically; its ordinary cells then calculate/display the results inline.
No starter notebook or separate input upload is needed. Later dependency chains can
take hours. The following manual input-transfer instructions concern **Client**.

Compact Rebuild keeps the same source snapshots and frozen provider responses, while
regenerating channel handoffs and prediction caches. New results need not match the
saved reference row for row. Input hashes, chronological rules and completed Q2
verification remain strict.

Saved notebook outputs require no data download. Run all repeats code and may need
the additional files below; raw data and calculated caches are omitted to keep Client small.
All paths are relative to `code/`. Supply complete result folders, including their manifests.

In a new Colab session, first choose the explicitly requested `Release-Client.zip`
for the Python code. A separate **Choose files** prompt then names the missing data
files and their destination paths. You may select several named files together.
Correct uploads are placed automatically and execution continues; wrong or incomplete
uploads stop before calculation. If the browser picker is cancelled but the cell is
still waiting, stop that cell and rerun it when the required files are ready.

When a complete result folder is required, the prompt asks for **Notebook-inputs.zip**.
This is your input transfer file, not a third software distribution. Include only the
requested files/folders, preserving their paths relative to `code/`. For example,
`experiments/cache/qualified_union_v1/summary.json` must have that path in the ZIP
(a leading `code/` is also accepted). Generate these calculation outputs with Rebuild;
the small Client and the raw-source bundle do not contain all fitted results.
Existing files, unrelated paths and unsafe ZIP members are not overwritten.

## First notebook

`01_RQ1_A_BTC_data_labels_baseline.ipynb` needs:

- `data/btcusdt_m15_2024_2025.parquet`
- `data/btcusdt_positioning_m15_2024_2026.parquet`

These two prepared files are included under `code/data/` in `Release-Rebuild.zip`.
They are shared BTC market/positioning inputs, not data exclusive to the first notebook.
Extract them on your computer and select both when Client asks for them in Colab.
Alternatively, open the first notebook from Rebuild and select **Run all**. It uses
these inputs directly and fits models in the notebook's current kernel. This does
not rebuild the normalised input files themselves. For source reconstruction, generate the
files from public Binance archives instead.
With Git and the packages in `requirements-repro.txt` installed, run from `code/`:

```bash
python -m experiments.reproduce_source --task market.binance.build_positioning --no-compare
```

This also builds its registered market-data prerequisites, including 1-minute history.
Downloads and extraction require several GB of disk space and internet access.
The command constructs inputs; it does not compare them with Git references.

## Later notebooks

Most later scientific cells read experiment outputs. In Rebuild, the first cell
runs their producer commands before those cells; in Client, supply the named result
folders separately. Producer commands may resume verified internal checkpoints.
Raw source files alone do not replace fitted predictions and derived result tables.

Public BTC inputs come from Binance. Registered index/VIX exports, news/direct-event
snapshots and frozen provider responses are supplied in the separate source bundle.
Fresh provider calls are not required to inspect saved results. See the
[reconstruction guide](../../REPRODUCIBILITY.md) for acquisition and full regeneration.

## Inputs by notebook

The first cell reports which of these paths are missing. The scientific cells retain
their own format, manifest and integrity checks; file presence alone does not establish
that an input is correct. A `*` denotes the matching filename, not an arbitrary replacement.

### RQ1

**01_RQ1_A_BTC_data_labels_baseline.ipynb**

- `data/btcusdt_m15_2024_2025.parquet`
- `data/btcusdt_positioning_m15_2024_2026.parquet`

**02_RQ1_B_BTC_positioning_ablation.ipynb**

- `data/btcusdt_m15_2024_2025.parquet`
- `data/btcusdt_positioning_m15_2024_2026.parquet`
- `experiments/cache/walkforward/btc_of-base_dz55_devwf.parquet`
- `experiments/cache/walkforward/btc_of-pos_dz55_devwf.parquet`
- `experiments/cache/walkforward/btc_of-placebo_dz55_devwf.parquet`
- `experiments/cache/walkforward/btc_of-base_dz60_devwf.parquet`
- `experiments/cache/walkforward/btc_of-pos_dz60_devwf.parquet`
- `experiments/cache/walkforward/btc_of-placebo_dz60_devwf.parquet`
- `experiments/cache/walkforward/btc_of-base_dz65_devwf.parquet`
- `experiments/cache/walkforward/btc_of-pos_dz65_devwf.parquet`
- `experiments/cache/walkforward/btc_of-placebo_dz65_devwf.parquet`
- `experiments/cache/walkforward/btc_of-base_dz75_devwf.parquet`
- `experiments/cache/walkforward/btc_of-pos_dz75_devwf.parquet`
- `experiments/cache/walkforward/btc_of-placebo_dz75_devwf.parquet`

**03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb**

- `experiments/cache/tuning/notebook01_handoff/selected_widths.parquet`
- `experiments/cache/tuning/notebook02_no_sentiment/handoff.json`
- `experiments/cache/tuning/notebook02b_handoff/handoff.json`
- `experiments/cache/tuning/notebook02_no_sentiment/matched_catboost_monthly_h1`

**04_RQ1_E_indices_nine_model_benchmark.ipynb**

- `experiments/cache/index_replication/usa500`
- `experiments/cache/index_all_model_forward/usa500`
- `experiments/cache/index_replication/usatech`
- `experiments/cache/index_all_model_forward/usatech`

**05_RQ1_F_indices_VIX_ablation.ipynb**

- `experiments/cache/index_replication/usa500/vix_admission.json`
- `experiments/cache/index_replication/usa500/vix_gate_paired_2024.parquet`
- `experiments/cache/index_replication/usa500/vix_gate_classification_2024.parquet`
- `experiments/cache/index_replication/usatech/vix_admission.json`
- `experiments/cache/index_replication/usatech/vix_gate_paired_2024.parquet`
- `experiments/cache/index_replication/usatech/vix_gate_classification_2024.parquet`

### RQ2

**06_RQ2_A_BTC_all_model_stacking.ipynb**

- `experiments/cache/tuning/all_model_stacking`

**07_RQ2_B_BTC_stacking_forward_validation.ipynb**

- `experiments/cache/tuning/all_model_stacking`

**08_RQ2_C_BTC_qualified_union_ensemble.ipynb**

- `experiments/cache/qualified_union_v1`

**09_RQ2_F_BTC_LSTM_GMADL_shadow.ipynb**

- `experiments/cache/qualified_union_v1`
- `experiments/cache/unified_expected_net_ensemble`
- `experiments/cache/lstm_gmadl_shadow`

**10_RQ2_H_indices_all_model_ensemble.ipynb**

- `experiments/cache/index_all_model_ensemble/usa500`
- `experiments/cache/index_all_model_ensemble/usatech`
- `experiments/cache/index_replication/usa500/vix_admission.json`
- `experiments/cache/index_replication/usatech/vix_admission.json`

**11_RQ2_I_indices_policy_comparison.ipynb**

- `experiments/cache/index_all_model_forward/usa500`
- `experiments/cache/index_trade_coverage/usa500`
- `experiments/cache/index_side_calibration/usa500`
- `experiments/cache/index_all_model_ensemble/usa500`
- `experiments/cache/index_channel_replication/usa500`
- `experiments/cache/index_all_model_forward/usatech`
- `experiments/cache/index_trade_coverage/usatech`
- `experiments/cache/index_side_calibration/usatech`
- `experiments/cache/index_all_model_ensemble/usatech`
- `experiments/cache/index_channel_replication/usatech`
- `experiments/cache/index_replication/usa500`
- `experiments/cache/index_replication/usatech`

### RQ3

**12_RQ3_A_BTC_sentiment_data_methodology.ipynb**

- `data/btcusdt_m15_2024_2025.parquet`
- `sentiment/raw/scores_btc.parquet`
- `sentiment/raw/scores_direct_events_btc.parquet`
- `sentiment/raw/scores_llm_btc.parquet`
- `sentiment/raw/scores_llm_direct_events_btc.parquet`

**13_RQ3_B_BTC_nine_model_sentiment_ablation.ipynb**

- `experiments/cache/tuning/notebook02b_handoff/handoff.json`
- `experiments/cache/tuning/all_model_sentiment_raw_180d_fixed15_v3`

**14_RQ3_C_BTC_sentiment_policy_ablation.ipynb**

- `experiments/cache/tuning/all_model_sentiment_policy_180d_fixed15_monthly_h1_v3`

**15_RQ3_D_indices_DeBERTa_sentiment.ipynb**

- `experiments/cache/index_replication/usa500`
- `experiments/cache/index_replication/usatech`
- `experiments/cache/index_all_model_forward/usa500`
- `sentiment/raw/scores_usa500.manifest.json`
- `sentiment/raw/scores_direct_events_usa500.manifest.json`
- `experiments/cache/index_all_model_forward/usatech`
- `sentiment/raw/scores_usatech.manifest.json`
- `sentiment/raw/scores_direct_events_usatech.manifest.json`

**16_RQ3_E_indices_LLM_sentiment.ipynb**

- `experiments/cache/index_replication/usa500`
- `experiments/cache/index_replication/usatech`
- `experiments/cache/index_all_model_forward/usa500`
- `sentiment/raw/scores_llm_usa500.manifest.json`
- `sentiment/raw/scores_llm_direct_events_usa500.manifest.json`
- `experiments/cache/index_all_model_forward/usatech`
- `sentiment/raw/scores_llm_usatech.manifest.json`
- `sentiment/raw/scores_llm_direct_events_usatech.manifest.json`
- `sentiment/raw/index_*_identity.json`

**17_RQ3_F_indices_Q2_sentiment_sensitivity.ipynb**

- `experiments/cache/q2_sentiment_sensitivity`

### RQ4

**18_RQ4_A_BTC_LLM_policy_router.ipynb**

- `experiments/cache/reflection_ensemble_v5/reader_manifest.json` and its individually hashed files: nine-model forecasts, execution paths, realised expert outcomes, LLM requests/responses, weight decisions and result tables.
- `experiments/rq4_ensemble_reader.py`: independently reconstructs memory inputs, weighted trades, net returns and paired intervals from the saved inputs.
- `configs/rq4_colab_publication.json`: pins the public Google Drive loader and individual-file manifest used by notebook18 in Colab.

Run All loads these inputs from Google Drive and verifies the replay. The separate producers are `python -m experiments.rq4_nine_model_data` and `python -m experiments.run_reflection_ensemble`; a fresh LLM run needs the configured Ollama Cloud model. Replaying saved LLM answers does not constitute new LLM calls.

### RQ5

**19_RQ5_B_BTC_volatility_feature_consolidation.ipynb**

- `experiments/cache/event_window_feature_consolidation`

**20_RQ5_C_BTC_economic_direction_head.ipynb**

- `experiments/cache/event_window_direction_head`

**21_RQ5_D_BTC_channel_vs_volatility_ablation.ipynb**

- `experiments/cache/channel_vs_volatility_ablation`

### Lockbox

**22_Lockbox_Q2_2026.ipynb**

- `experiments/cache/final_q2_lockbox/8185dee25cc468ed9be8b79aad59eb33eeda874f4cb245c31883ef127f205312`
- `experiments/cache/final_q2_lockbox/OPENED.json`
