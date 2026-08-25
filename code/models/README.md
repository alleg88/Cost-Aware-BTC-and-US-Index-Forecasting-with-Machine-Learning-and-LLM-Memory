# Model registry

All nine pre-specified models expose a common `fit`, `predict` and `predict_proba`
interface and predict down/flat/up.

| Family | Registry name |
|---|---|
| Logistic regression | `logreg` |
| Decision tree | `decision_tree` |
| Random forest | `random_forest` |
| Linear SVM | `svm_linear` |
| XGBoost | `xgboost_balanced` |
| CatBoost | `catboost_balanced` |
| MLP | `mlp` |
| LSTM | `lstm` |
| GRU | `gru` |

Class weights are fitted within training folds. Scale-sensitive estimators use training-fold
statistics only; sequence windows contain no future rows. Dummy controls and stacking
factories are registered separately from the nine-model comparison.
