# Prediction-Level Ensemble

Folds: `fold1, fold2, fold3`.
Grid step: `0.01`.
Weights are tuned on validation per fold, then applied to that fold's test set. Thresholds are calibrated on validation by balanced accuracy.

## Selected Models

| Model | Metrics Dir | Folds | Selected |
| --- | --- | --- | --- |
| `cross_attention_auc` | `outputs/metrics/retrain_clean_dolos_three_stream_auc` | `fold1,fold2,fold3` | yes |
| `gated_prior_kl` | `outputs/metrics/retrain_clean_dolos_three_stream_gated_logits_prior_kl_temporal_mask` | `fold1,fold2,fold3` | yes |

## Mean Across Folds

| Method | Mean Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 59.61 | 60.88 | 60.59 | 56.70 | 48.06 | 53.02 |
| `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 63.32 | 62.66 | 64.38 | 59.39 | 60.27 | 59.26 |
| `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.09, gated_prior_kl=0.91 | 64.13 | 64.25 | 65.27 | 59.44 | 62.10 | 59.29 |
| `ensemble_raw_auc_roc` | cross_attention_auc=0.41, gated_prior_kl=0.59 | 64.21 | 63.16 | 65.32 | 59.61 | 58.94 | 59.30 |
| `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.09, gated_prior_kl=0.91 | 64.12 | 64.25 | 65.27 | 59.44 | 62.10 | 59.29 |
| `ensemble_logit_auc_roc` | cross_attention_auc=0.33, gated_prior_kl=0.67 | 64.19 | 63.80 | 65.51 | 59.72 | 61.30 | 59.64 |

## Per-Fold Results

| Fold | Method | Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro | Test CM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fold1 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 58.33 | 59.29 | 59.12 | 56.42 | 62.04 | 56.26 | `[[106, 121], [87, 170]]` |
| fold1 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 64.58 | 64.85 | 65.51 | 60.02 | 63.50 | 60.03 | `[[125, 102], [90, 167]]` |
| fold1 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.05, gated_prior_kl=0.95 | 64.90 | 65.81 | 65.41 | 60.02 | 63.50 | 60.03 | `[[125, 102], [90, 167]]` |
| fold1 | `ensemble_raw_auc_roc` | cross_attention_auc=0.26, gated_prior_kl=0.74 | 64.96 | 64.21 | 64.84 | 59.24 | 56.33 | 58.56 | `[[155, 72], [128, 129]]` |
| fold1 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.05, gated_prior_kl=0.95 | 64.90 | 65.81 | 65.41 | 60.02 | 63.50 | 60.03 | `[[125, 102], [90, 167]]` |
| fold1 | `ensemble_logit_auc_roc` | cross_attention_auc=0.05, gated_prior_kl=0.95 | 64.90 | 65.81 | 65.41 | 60.02 | 63.50 | 60.03 | `[[125, 102], [90, 167]]` |
| fold2 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 65.97 | 64.68 | 65.73 | 60.94 | 62.55 | 60.93 | `[[137, 92], [96, 157]]` |
| fold2 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 63.75 | 62.54 | 62.17 | 59.07 | 58.92 | 58.92 | `[[142, 87], [111, 142]]` |
| fold2 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.17, gated_prior_kl=0.83 | 65.78 | 66.35 | 64.98 | 59.65 | 64.84 | 59.45 | `[[113, 116], [76, 177]]` |
| fold2 | `ensemble_raw_auc_roc` | cross_attention_auc=0.94, gated_prior_kl=0.06 | 65.97 | 64.68 | 65.72 | 60.94 | 62.55 | 60.93 | `[[137, 92], [96, 157]]` |
| fold2 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.17, gated_prior_kl=0.83 | 65.75 | 66.35 | 64.98 | 59.65 | 64.84 | 59.45 | `[[113, 116], [76, 177]]` |
| fold2 | `ensemble_logit_auc_roc` | cross_attention_auc=0.91, gated_prior_kl=0.09 | 65.97 | 65.00 | 65.70 | 60.48 | 62.45 | 60.48 | `[[134, 95], [95, 158]]` |
| fold3 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 54.54 | 58.65 | 56.92 | 52.75 | 19.58 | 41.88 | `[[206, 13], [217, 28]]` |
| fold3 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 61.64 | 60.58 | 65.46 | 59.08 | 58.39 | 58.83 | `[[139, 80], [111, 134]]` |
| fold3 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.04, gated_prior_kl=0.96 | 61.71 | 60.58 | 65.41 | 58.65 | 57.95 | 58.40 | `[[138, 81], [112, 133]]` |
| fold3 | `ensemble_raw_auc_roc` | cross_attention_auc=0.04, gated_prior_kl=0.96 | 61.71 | 60.58 | 65.41 | 58.65 | 57.95 | 58.40 | `[[138, 81], [112, 133]]` |
| fold3 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.04, gated_prior_kl=0.96 | 61.71 | 60.58 | 65.41 | 58.65 | 57.95 | 58.40 | `[[138, 81], [112, 133]]` |
| fold3 | `ensemble_logit_auc_roc` | cross_attention_auc=0.04, gated_prior_kl=0.96 | 61.71 | 60.58 | 65.41 | 58.65 | 57.95 | 58.40 | `[[138, 81], [112, 133]]` |

## Interpretation

- This report only compares models that have both validation and test predictions for the selected folds.
- The main comparison target is `gated_prior_kl`, because it is the current primary model.
- A useful ensemble should improve mean test AUC or mean calibrated BA without relying on a single fold spike.
