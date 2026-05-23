# Prediction-Level Ensemble

Folds: `fold1, fold2, fold3`.
Grid step: `0.01`.
Weights are tuned on validation per fold, then applied to that fold's test set. Thresholds are calibrated on validation by balanced accuracy.

## Selected Models

| Model | Metrics Dir | Folds | Selected |
| --- | --- | --- | --- |
| `cross_attention_auc` | `outputs/metrics/retrain_clean_dolos_three_stream_auc_soft_temporal_penalty` | `fold1,fold2,fold3` | yes |
| `gated_prior_kl` | `outputs/metrics/retrain_clean_dolos_three_stream_gated_logits_prior_kl_temporal_mask` | `fold1,fold2,fold3` | yes |

## Mean Across Folds

| Method | Mean Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 62.24 | 62.20 | 61.98 | 57.22 | 48.36 | 52.30 |
| `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 63.32 | 62.66 | 64.38 | 59.39 | 60.27 | 59.26 |
| `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.65, gated_prior_kl=0.35 | 65.37 | 65.02 | 65.43 | 60.54 | 61.39 | 60.13 |
| `ensemble_raw_auc_roc` | cross_attention_auc=0.62, gated_prior_kl=0.38 | 65.46 | 64.16 | 65.20 | 60.00 | 59.06 | 59.53 |
| `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.65, gated_prior_kl=0.35 | 65.35 | 65.02 | 65.40 | 60.47 | 61.35 | 60.05 |
| `ensemble_logit_auc_roc` | cross_attention_auc=0.62, gated_prior_kl=0.38 | 65.46 | 64.16 | 65.19 | 60.00 | 59.06 | 59.53 |

## Per-Fold Results

| Fold | Method | Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro | Test CM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fold1 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 56.78 | 57.69 | 56.86 | 51.46 | 19.02 | 40.88 | `[[208, 19], [228, 29]]` |
| fold1 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 64.58 | 64.85 | 65.51 | 60.02 | 63.50 | 60.03 | `[[125, 102], [90, 167]]` |
| fold1 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.85, gated_prior_kl=0.15 | 64.96 | 64.85 | 65.41 | 59.63 | 62.98 | 59.64 | `[[125, 102], [92, 165]]` |
| fold1 | `ensemble_raw_auc_roc` | cross_attention_auc=0.85, gated_prior_kl=0.15 | 64.96 | 64.85 | 65.41 | 59.63 | 62.98 | 59.64 | `[[125, 102], [92, 165]]` |
| fold1 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.86, gated_prior_kl=0.14 | 64.96 | 64.85 | 65.39 | 59.63 | 62.98 | 59.64 | `[[125, 102], [92, 165]]` |
| fold1 | `ensemble_logit_auc_roc` | cross_attention_auc=0.86, gated_prior_kl=0.14 | 64.96 | 64.85 | 65.39 | 59.63 | 62.98 | 59.64 | `[[125, 102], [92, 165]]` |
| fold2 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 69.78 | 67.06 | 64.73 | 61.27 | 55.81 | 60.12 | `[[172, 57], [133, 120]]` |
| fold2 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 63.75 | 62.54 | 62.17 | 59.07 | 58.92 | 58.92 | `[[142, 87], [111, 142]]` |
| fold2 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 69.78 | 67.06 | 64.73 | 61.27 | 55.81 | 60.12 | `[[172, 57], [133, 120]]` |
| fold2 | `ensemble_raw_auc_roc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 69.78 | 67.06 | 64.73 | 61.27 | 55.81 | 60.12 | `[[172, 57], [133, 120]]` |
| fold2 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 69.78 | 67.06 | 64.73 | 61.27 | 55.81 | 60.12 | `[[172, 57], [133, 120]]` |
| fold2 | `ensemble_logit_auc_roc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 69.78 | 67.06 | 64.73 | 61.27 | 55.81 | 60.12 | `[[172, 57], [133, 120]]` |
| fold3 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 60.17 | 61.86 | 64.34 | 58.92 | 70.24 | 55.89 | `[[65, 154], [29, 216]]` |
| fold3 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 61.64 | 60.58 | 65.46 | 59.08 | 58.39 | 58.83 | `[[139, 80], [111, 134]]` |
| fold3 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.11, gated_prior_kl=0.89 | 61.37 | 63.14 | 66.14 | 60.72 | 65.38 | 60.63 | `[[114, 105], [75, 170]]` |
| fold3 | `ensemble_raw_auc_roc` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 61.64 | 60.58 | 65.46 | 59.08 | 58.39 | 58.83 | `[[139, 80], [111, 134]]` |
| fold3 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.10, gated_prior_kl=0.90 | 61.32 | 63.14 | 66.07 | 60.49 | 65.26 | 60.39 | `[[113, 106], [75, 170]]` |
| fold3 | `ensemble_logit_auc_roc` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 61.64 | 60.58 | 65.46 | 59.08 | 58.39 | 58.83 | `[[139, 80], [111, 134]]` |

## Interpretation

- This report only compares models that have both validation and test predictions for the selected folds.
- The main comparison target is `gated_prior_kl`, because it is the current primary model.
- A useful ensemble should improve mean test AUC or mean calibrated BA without relying on a single fold spike.
