# Prediction-Level Ensemble

Folds: `fold1, fold2, fold3`.
Grid step: `0.01`.
Weights are tuned on validation per fold, then applied to that fold's test set. Thresholds are calibrated on validation by balanced accuracy.

## Selected Models

| Model | Metrics Dir | Folds | Selected |
| --- | --- | --- | --- |
| `cross_attention_auc` | `outputs/metrics/retrain_clean_dolos_three_stream_auc` | `fold1,fold2,fold3` | yes |
| `gated_prior_kl` | `outputs/metrics/retrain_clean_dolos_three_stream_gated_logits_prior_kl_soft_face_valid` | `fold1,fold2,fold3` | yes |

## Mean Across Folds

| Method | Mean Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 59.61 | 60.88 | 60.59 | 56.70 | 48.06 | 53.02 |
| `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 62.16 | 62.54 | 60.81 | 56.78 | 58.51 | 56.64 |
| `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.07, gated_prior_kl=0.93 | 62.68 | 63.78 | 61.72 | 57.57 | 63.09 | 56.69 |
| `ensemble_raw_auc_roc` | cross_attention_auc=0.37, gated_prior_kl=0.63 | 63.33 | 62.30 | 61.94 | 58.10 | 59.89 | 57.95 |
| `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.07, gated_prior_kl=0.93 | 62.70 | 63.78 | 61.71 | 57.50 | 63.05 | 56.61 |
| `ensemble_logit_auc_roc` | cross_attention_auc=0.36, gated_prior_kl=0.64 | 63.31 | 62.30 | 61.94 | 58.10 | 59.89 | 57.95 |

## Per-Fold Results

| Fold | Method | Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro | Test CM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fold1 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 58.33 | 59.29 | 59.12 | 56.42 | 62.04 | 56.26 | `[[106, 121], [87, 170]]` |
| fold1 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 58.17 | 62.50 | 59.59 | 55.60 | 56.91 | 55.54 | `[[127, 100], [115, 142]]` |
| fold1 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.01, gated_prior_kl=0.99 | 58.23 | 62.50 | 59.62 | 55.04 | 55.87 | 54.94 | `[[128, 99], [119, 138]]` |
| fold1 | `ensemble_raw_auc_roc` | cross_attention_auc=0.16, gated_prior_kl=0.84 | 59.35 | 60.04 | 59.54 | 57.15 | 56.00 | 56.80 | `[[142, 85], [124, 133]]` |
| fold1 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.01, gated_prior_kl=0.99 | 58.23 | 62.50 | 59.62 | 55.04 | 55.87 | 54.94 | `[[128, 99], [119, 138]]` |
| fold1 | `ensemble_logit_auc_roc` | cross_attention_auc=0.15, gated_prior_kl=0.85 | 59.29 | 60.04 | 59.54 | 57.15 | 56.00 | 56.80 | `[[142, 85], [124, 133]]` |
| fold2 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 65.97 | 64.68 | 65.73 | 60.94 | 62.55 | 60.93 | `[[137, 92], [96, 157]]` |
| fold2 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 63.65 | 62.94 | 62.25 | 58.54 | 57.51 | 58.28 | `[[145, 84], [117, 136]]` |
| fold2 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.16, gated_prior_kl=0.84 | 65.56 | 66.35 | 64.93 | 60.09 | 65.07 | 59.92 | `[[115, 114], [76, 177]]` |
| fold2 | `ensemble_raw_auc_roc` | cross_attention_auc=0.94, gated_prior_kl=0.06 | 65.97 | 64.68 | 65.72 | 60.94 | 62.55 | 60.93 | `[[137, 92], [96, 157]]` |
| fold2 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.15, gated_prior_kl=0.85 | 65.62 | 66.35 | 64.91 | 59.87 | 64.95 | 59.68 | `[[114, 115], [76, 177]]` |
| fold2 | `ensemble_logit_auc_roc` | cross_attention_auc=0.94, gated_prior_kl=0.06 | 65.97 | 64.68 | 65.72 | 60.94 | 62.55 | 60.93 | `[[137, 92], [96, 157]]` |
| fold3 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 54.54 | 58.65 | 56.92 | 52.75 | 19.58 | 41.88 | `[[206, 13], [217, 28]]` |
| fold3 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 64.67 | 62.18 | 60.57 | 56.22 | 61.12 | 56.11 | `[[105, 114], [87, 158]]` |
| fold3 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.05, gated_prior_kl=0.95 | 64.25 | 62.50 | 60.60 | 57.59 | 68.33 | 55.20 | `[[69, 150], [40, 205]]` |
| fold3 | `ensemble_raw_auc_roc` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 64.67 | 62.18 | 60.57 | 56.22 | 61.12 | 56.11 | `[[105, 114], [87, 158]]` |
| fold3 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.05, gated_prior_kl=0.95 | 64.25 | 62.50 | 60.60 | 57.59 | 68.33 | 55.20 | `[[69, 150], [40, 205]]` |
| fold3 | `ensemble_logit_auc_roc` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 64.67 | 62.18 | 60.57 | 56.22 | 61.12 | 56.11 | `[[105, 114], [87, 158]]` |

## Interpretation

- This report only compares models that have both validation and test predictions for the selected folds.
- The main comparison target is `gated_prior_kl`, because it is the current primary model.
- A useful ensemble should improve mean test AUC or mean calibrated BA without relying on a single fold spike.
