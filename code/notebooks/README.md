# Notebook instructions

Open the 22 notebooks in numerical order (01-22), in Jupyter, VS Code or Google Colab. Saved tables and figures appear when a notebook opens.

Each notebook explains its method, displays its results and ends with a takeaway.
Choose a topic below or follow the numerical order. The filename labels RQ1–RQ5
identify the five comparison groups.

## Run a notebook

1. For calculations, open the matching notebook from **Rebuild**.
2. In Colab, select **File > Upload notebook**, then **Runtime > Run all**. Upload **Release-Rebuild.zip** when prompted.
3. Locally, open `code/notebooks/` from the extracted ZIP, or keep the supplied `notebooks/` folder beside the ZIP. Select **Run all**.
4. Follow the progress in the first cell. Save the completed notebook to keep the new results.

Local requirements: Python 3.11 or newer, Git and internet access. Allow several hours and several GB of free space for larger experiments.

For Client reruns, supply the files listed in [DATA.md](DATA.md).

## RQ1: Individual models

- [`01_RQ1_A_BTC_data_labels_baseline.ipynb`](01_RQ1_A_BTC_data_labels_baseline.ipynb): BTC data, labels and baseline.
- [`02_RQ1_B_BTC_positioning_ablation.ipynb`](02_RQ1_B_BTC_positioning_ablation.ipynb): BTC positioning ablation.
- [`03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb`](03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb): BTC CatBoost economic objectives.
- [`04_RQ1_E_indices_nine_model_benchmark.ipynb`](04_RQ1_E_indices_nine_model_benchmark.ipynb): Index nine-model benchmark.
- [`05_RQ1_F_indices_VIX_ablation.ipynb`](05_RQ1_F_indices_VIX_ablation.ipynb): Index VIX ablation.

## RQ2: Ensembles

- [`06_RQ2_A_BTC_all_model_stacking.ipynb`](06_RQ2_A_BTC_all_model_stacking.ipynb): BTC all-model stacking.
- [`07_RQ2_B_BTC_stacking_forward_validation.ipynb`](07_RQ2_B_BTC_stacking_forward_validation.ipynb): BTC stacking forward validation.
- [`08_RQ2_C_BTC_qualified_union_ensemble.ipynb`](08_RQ2_C_BTC_qualified_union_ensemble.ipynb): BTC Qualified Union ensemble.
- [`09_RQ2_F_BTC_LSTM_GMADL_shadow.ipynb`](09_RQ2_F_BTC_LSTM_GMADL_shadow.ipynb): BTC LSTM GMADL ensemble shadow.
- [`10_RQ2_H_indices_all_model_ensemble.ipynb`](10_RQ2_H_indices_all_model_ensemble.ipynb): Index all-model ensemble.
- [`11_RQ2_I_indices_policy_comparison.ipynb`](11_RQ2_I_indices_policy_comparison.ipynb): Index policy comparison.

## RQ3: Sentiment

- [`12_RQ3_A_BTC_sentiment_data_methodology.ipynb`](12_RQ3_A_BTC_sentiment_data_methodology.ipynb): BTC sentiment data and methodology.
- [`13_RQ3_B_BTC_nine_model_sentiment_ablation.ipynb`](13_RQ3_B_BTC_nine_model_sentiment_ablation.ipynb): BTC nine-model sentiment ablation.
- [`14_RQ3_C_BTC_sentiment_policy_ablation.ipynb`](14_RQ3_C_BTC_sentiment_policy_ablation.ipynb): BTC sentiment policy ablation.
- [`15_RQ3_D_indices_DeBERTa_sentiment.ipynb`](15_RQ3_D_indices_DeBERTa_sentiment.ipynb): Index DeBERTa sentiment.
- [`16_RQ3_E_indices_LLM_sentiment.ipynb`](16_RQ3_E_indices_LLM_sentiment.ipynb): Index LLM sentiment.
- [`17_RQ3_F_indices_Q2_sentiment_sensitivity.ipynb`](17_RQ3_F_indices_Q2_sentiment_sensitivity.ipynb): Index Q2 sentiment-feed sensitivity.

## RQ4: LLM ensemble weights and memory

- [`18_RQ4_A_BTC_LLM_policy_router.ipynb`](18_RQ4_A_BTC_LLM_policy_router.ipynb): weekly weights over nine BTC models using real, absent or shuffled four-week outcome memory, compared with Hedge and a fixed LSTM. In Colab, Run All downloads verified published inputs directly. See [DATA.md](DATA.md) for rerunning details.

## RQ5: Channels and volatility

- [`19_RQ5_B_BTC_volatility_feature_consolidation.ipynb`](19_RQ5_B_BTC_volatility_feature_consolidation.ipynb): BTC volatility and timing feature consolidation.
- [`20_RQ5_C_BTC_economic_direction_head.ipynb`](20_RQ5_C_BTC_economic_direction_head.ipynb): BTC economic direction head.
- [`21_RQ5_D_BTC_channel_vs_volatility_ablation.ipynb`](21_RQ5_D_BTC_channel_vs_volatility_ablation.ipynb): BTC channel-versus-volatility ablation.

## Lockbox: Q2 2026 results

- [`22_Lockbox_Q2_2026.ipynb`](22_Lockbox_Q2_2026.ipynb): Q2 2026 lockbox confirmation.
