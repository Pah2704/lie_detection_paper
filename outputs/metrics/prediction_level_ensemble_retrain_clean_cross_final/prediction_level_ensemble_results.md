# Prediction-Level Ensemble

Folds: `fold1, fold2, fold3`.
Grid step: `0.01`.
Weights are tuned on validation per fold, then applied to that fold's test set. Thresholds are calibrated on validation by balanced accuracy.

## Selected Models

| Model | Metrics Dir | Folds | Selected |
| --- | --- | --- | --- |
| `cross_attention_auc` | `outputs/metrics/retrain_clean_dolos_three_stream_auc` | `fold1,fold2,fold3` | yes |
| `gated_prior_kl` | `outputs/metrics/retrain_clean_dolos_three_stream_gated_logits_prior_kl` | `fold1,fold2,fold3` | yes |

## Mean Across Folds

| Method | Mean Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 59.61 | 60.88 | 60.59 | 56.70 | 48.06 | 53.02 |
| `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 62.73 | 64.67 | 61.42 | 57.51 | 56.50 | 57.08 |
| `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.16, gated_prior_kl=0.84 | 62.89 | 66.29 | 61.97 | 57.92 | 57.05 | 57.61 |
| `ensemble_raw_auc_roc` | cross_attention_auc=0.40, gated_prior_kl=0.60 | 63.80 | 64.01 | 61.30 | 57.21 | 58.96 | 57.08 |
| `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.16, gated_prior_kl=0.84 | 62.89 | 66.29 | 61.97 | 57.92 | 57.05 | 57.61 |
| `ensemble_logit_auc_roc` | cross_attention_auc=0.40, gated_prior_kl=0.60 | 63.80 | 64.01 | 61.29 | 57.21 | 58.96 | 57.08 |

## Per-Fold Results

| Fold | Method | Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro | Test CM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fold1 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 58.33 | 59.29 | 59.12 | 56.42 | 62.04 | 56.26 | `[[106, 121], [87, 170]]` |
| fold1 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 59.94 | 61.97 | 61.20 | 55.26 | 55.98 | 55.15 | `[[129, 98], [119, 138]]` |
| fold1 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 59.94 | 61.97 | 61.20 | 55.26 | 55.98 | 55.15 | `[[129, 98], [119, 138]]` |
| fold1 | `ensemble_raw_auc_roc` | cross_attention_auc=0.12, gated_prior_kl=0.88 | 61.32 | 61.65 | 60.92 | 56.75 | 61.60 | 56.67 | `[[111, 116], [91, 166]]` |
| fold1 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 59.94 | 61.97 | 61.20 | 55.26 | 55.98 | 55.15 | `[[129, 98], [119, 138]]` |
| fold1 | `ensemble_logit_auc_roc` | cross_attention_auc=0.12, gated_prior_kl=0.88 | 61.32 | 61.65 | 60.88 | 56.75 | 61.60 | 56.67 | `[[111, 116], [91, 166]]` |
| fold2 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 65.97 | 64.68 | 65.73 | 60.94 | 62.55 | 60.93 | `[[137, 92], [96, 157]]` |
| fold2 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 64.41 | 67.30 | 66.06 | 61.47 | 56.15 | 60.34 | `[[172, 57], [132, 121]]` |
| fold2 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.21, gated_prior_kl=0.79 | 64.89 | 68.97 | 67.01 | 62.92 | 59.87 | 62.29 | `[[166, 63], [118, 135]]` |
| fold2 | `ensemble_raw_auc_roc` | cross_attention_auc=0.99, gated_prior_kl=0.01 | 65.97 | 64.68 | 65.75 | 60.94 | 62.55 | 60.93 | `[[137, 92], [96, 157]]` |
| fold2 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.21, gated_prior_kl=0.79 | 64.89 | 68.97 | 67.01 | 62.92 | 59.87 | 62.29 | `[[166, 63], [118, 135]]` |
| fold2 | `ensemble_logit_auc_roc` | cross_attention_auc=0.99, gated_prior_kl=0.01 | 65.97 | 64.68 | 65.75 | 60.94 | 62.55 | 60.93 | `[[137, 92], [96, 157]]` |
| fold3 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 54.54 | 58.65 | 56.92 | 52.75 | 19.58 | 41.88 | `[[206, 13], [217, 28]]` |
| fold3 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 63.86 | 64.74 | 57.00 | 55.79 | 57.38 | 55.76 | `[[121, 98], [107, 138]]` |
| fold3 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.26, gated_prior_kl=0.74 | 63.86 | 67.95 | 57.70 | 55.57 | 55.29 | 55.39 | `[[129, 90], [117, 128]]` |
| fold3 | `ensemble_raw_auc_roc` | cross_attention_auc=0.10, gated_prior_kl=0.90 | 64.10 | 65.71 | 57.24 | 53.94 | 52.75 | 53.65 | `[[129, 90], [125, 120]]` |
| fold3 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.26, gated_prior_kl=0.74 | 63.86 | 67.95 | 57.71 | 55.57 | 55.29 | 55.39 | `[[129, 90], [117, 128]]` |
| fold3 | `ensemble_logit_auc_roc` | cross_attention_auc=0.10, gated_prior_kl=0.90 | 64.10 | 65.71 | 57.24 | 53.94 | 52.75 | 53.65 | `[[129, 90], [125, 120]]` |

## Interpretation

- This report only compares models that have both validation and test predictions for the selected folds.
- The main comparison target is `gated_prior_kl`, because it is the current primary model.
- A useful ensemble should improve mean test AUC or mean calibrated BA without relying on a single fold spike.
