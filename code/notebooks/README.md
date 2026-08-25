# Notebook reading order

The repository contains three independent development sequences followed by a final confirmation reader and one explicitly supplementary sensitivity. Each notebook is an executed, reader-facing experiment: methodology appears before results and heavy computation lives in `../experiments/`.

For Google Colab, start with [00_run_in_colab.ipynb](00_run_in_colab.ipynb), the first notebook in this folder. Select **CPU**, then **Runtime → Run all**. Choose the code ZIP when prompted; setup uses Colab's native Python and reruns reuse the upload. **Check installation** works with code alone; **Rebuild results** requires both ZIPs. Locally, run each result notebook from its first cell after setup.

## Bitcoin

The numbered BTC sequence progresses from data and single models to sentiment, ensembles and Reflection Agents.

| Notebook | Purpose and current conclusion |
|---|---|
| `01_data_labels_and_baseline.ipynb` | M15 data, causal price/order-flow/positioning features, blocking CV and the frozen dead-zone handoff. |
| `01b_positioning_ablation.ipynb` | Paired funding/open-interest ablation; weak positive input diagnostic, not a standalone edge. |
| `02b_catboost_economic_optuna.ipynb` | Matched CatBoost objectives, H1 policy calibration and frozen-forward economics. |
| `02c_sentiment_data_and_methodology.ipynb` | Sentiment provenance, causal matching and the exact DeBERTa/LLM feature contract. |
| `02d_all_model_sentiment.ipynb` | Raw nine-model price/DeBERTa/LLM comparison under one time-series protocol. |
| `02e_all_model_sentiment_policy.ipynb` | H1 confidence/TP/SL calibration followed by frozen-forward policy comparison. |
| `03_all_model_stacking.ipynb` | All-nine Logistic Regression stack construction and H1 selection. |
| `03a_stacking_forward.ipynb` | Frozen-forward stack evaluation; the stack is rejected against the best single model. |
| `03c_qualified_union_ensemble.ipynb` | Frozen Union v1: LSTM DZ55 plus Linear SVM DZ75 with conflict veto. |
| `04a_svm_temperature_calibration.ipynb` | Six-temperature SVM check; identity temperature is retained. |
| `04b_xgboost_strong_move_admission.ipynb` | Guarded XGBoost satellite; no H1 policy qualifies. |
| `04d_unified_2021_ensemble.ipynb` | Shared 2021 XGBoost/LSTM/SVM protocol; profitable high-volume candidate fails two-sided stability. |
| `04g_lstm_gmadl_shadow.ipynb` | Paired GMADL LSTM shadow; frequency rises but economics worsen. |
| `04h_union_v1_episode_reentry.ipynb` | Same-side episode re-entry; trade count rises and incremental trades lose money. |
| `05c_causal_policy_router_agent.ipynb` | Final weekly policy router; real memory passes H1 gates but fails development value/volume promotion. |

## Indices

The index replication applies one 2024 OOF → H1 2025 calibration → July 2025–March 2026 descriptive forward protocol separately to USA500 and USATECH. The April–June 2026 run is complete: BTC is the sole confirmatory stream and the index rows are descriptive transport checks.

| Notebook | Purpose and current conclusion |
|---|---|
| `06a_index_nine_models.ipynb` | Nine price models; USA500 Decision Tree leads the descriptive forward table. |
| `06b_index_vix.ipynb` | Numeric price-only versus price-plus-VIX admission; relative gains do not imply absolute profitability. |
| `06c_index_deberta.ipynb` | Matched DeBERTa features with H1 and forward results for nine individual models. |
| `06d_index_llm.ipynb` | LLM matched/full features, exact batch-10 prompt/schema and results for nine individual models. |
| `06g_index_all_model_ensemble.ipynb` | Nine-base-model probability average, 5/9 directional vote and causal logistic meta-model; neither best-ranked H1 ensemble exceeds its eligible single-model control. |
| `06i_index_comparison.ipynb` | Net-ranked synthesis of original, side-calibrated, coverage-first, nine-model ensemble and channel policies. |

## Channels

Notebook A defines the causal channel construction; U, V and W retain the final volatility/timing, direction and matched ablation evidence. The decision grid is 5 minutes with native 1-minute execution; no retained channel policy establishes robust positive net-of-cost economics.

| Notebook | Purpose and current conclusion |
|---|---|
| `A_channel_strategy.ipynb` | Causal hourly channel construction and mechanical entry feasibility; no promotion. |
| `U_volatility_timing_feature_consolidation.ipynb` | Exact 28-feature volatility/timing contract; rejected against the frozen control. |
| `V_economic_direction_head.ipynb` | Forced LONG/SHORT direction at every activation; economically non-viable. |
| `W_channel_vs_volatility_ablation.ipynb` | Final matched channel-versus-volatility comparison; both net arms are negative. |

## Final confirmation and supplementary sensitivity

Notebook 07 reports the frozen six-policy Q2 2026 evaluation. Notebook 07a then changes only fresh sentiment-feed availability while retaining the same frozen index policies, prices, VIX and costs.

| Notebook | Purpose and current conclusion |
|---|---|
| `07_final_q2_lockbox.ipynb` | Hash-verified Q2 confirmation: BTC Qualified Union does not outperform its frozen LSTM control; the USA500 all-nine vote and USATECH LLM LSTM are positive descriptive transport results. |
| `07a_q2_sentiment_sensitivity.ipynb` | Supplementary completed-feed replay: USA500 ensemble improves by 0.028 percentage points and adds one short trade, USATECH LLM-LSTM loses 0.728 percentage points, and the other two frozen policies are unchanged. |
