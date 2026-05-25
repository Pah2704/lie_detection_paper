# Prediction-Level Ensemble

Folds: `fold1, fold2, fold3`.
Grid step: `0.01`.
Weights are tuned on validation per fold, then applied to that fold's test set. Thresholds are calibrated on validation by balanced accuracy.

## Selected Models

| Model | Metrics Dir | Folds | Selected |
| --- | --- | --- | --- |
| `cross_attention_auc` | `outputs/metrics/dolos_three_stream_auc` | `fold1,fold2,fold3` | yes |
| `gated_prior_kl` | `outputs/metrics/dolos_three_stream_gated_logits_prior_kl` | `fold1,fold2,fold3` | yes |
| `spatial_only` | `outputs/metrics/dolos_fold3_spatial_only` | `fold1,fold2,fold3` | yes |

## Mean Across Folds

| Method | Mean Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, spatial_only=0.00 | 61.04 | 62.39 | 61.39 | 56.35 | 48.30 | 54.26 |
| `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, spatial_only=0.00 | 61.75 | 61.36 | 63.94 | 59.18 | 55.53 | 58.14 |
| `spatial_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, spatial_only=1.00 | 60.47 | 63.20 | 63.12 | 58.76 | 48.19 | 55.80 |
| `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.61, gated_prior_kl=0.15, spatial_only=0.24 | 62.91 | 66.73 | 63.86 | 59.03 | 50.25 | 56.54 |
| `ensemble_raw_auc_roc` | cross_attention_auc=0.62, gated_prior_kl=0.09, spatial_only=0.29 | 63.74 | 64.13 | 63.96 | 58.26 | 45.37 | 54.64 |
| `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.61, gated_prior_kl=0.15, spatial_only=0.25 | 62.85 | 66.73 | 63.88 | 58.74 | 50.02 | 56.27 |
| `ensemble_logit_auc_roc` | cross_attention_auc=0.60, gated_prior_kl=0.15, spatial_only=0.25 | 63.75 | 63.83 | 64.32 | 58.53 | 45.40 | 54.87 |

## Per-Fold Results

| Fold | Method | Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro | Test CM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fold1 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, spatial_only=0.00 | 64.85 | 66.88 | 65.28 | 58.21 | 51.29 | 56.59 | `[[168, 59], [148, 109]]` |
| fold1 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, spatial_only=0.00 | 61.54 | 62.82 | 61.77 | 57.45 | 55.56 | 56.98 | `[[146, 81], [127, 130]]` |
| fold1 | `spatial_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, spatial_only=1.00 | 58.65 | 62.61 | 66.57 | 59.79 | 53.83 | 58.39 | `[[169, 58], [141, 116]]` |
| fold1 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.75, gated_prior_kl=0.24, spatial_only=0.01 | 64.37 | 69.98 | 65.40 | 59.76 | 56.00 | 58.89 | `[[160, 67], [131, 126]]` |
| fold1 | `ensemble_raw_auc_roc` | cross_attention_auc=1.00, gated_prior_kl=0.00, spatial_only=0.00 | 64.85 | 66.88 | 65.28 | 58.21 | 51.29 | 56.59 | `[[168, 59], [148, 109]]` |
| fold1 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.74, gated_prior_kl=0.24, spatial_only=0.02 | 64.21 | 69.98 | 65.45 | 58.90 | 55.31 | 58.08 | `[[157, 70], [132, 125]]` |
| fold1 | `ensemble_logit_auc_roc` | cross_attention_auc=1.00, gated_prior_kl=0.00, spatial_only=0.00 | 64.85 | 66.88 | 65.28 | 58.21 | 51.29 | 56.59 | `[[168, 59], [148, 109]]` |
| fold2 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, spatial_only=0.00 | 64.79 | 64.84 | 60.95 | 58.26 | 57.92 | 58.09 | `[[141, 88], [114, 139]]` |
| fold2 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, spatial_only=0.00 | 64.70 | 62.94 | 67.73 | 60.44 | 49.74 | 57.80 | `[[189, 40], [156, 97]]` |
| fold2 | `spatial_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, spatial_only=1.00 | 64.35 | 66.75 | 67.00 | 62.43 | 60.13 | 61.95 | `[[161, 68], [115, 138]]` |
| fold2 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.79, gated_prior_kl=0.13, spatial_only=0.08 | 65.71 | 67.06 | 67.45 | 62.70 | 61.73 | 62.44 | `[[155, 74], [107, 146]]` |
| fold2 | `ensemble_raw_auc_roc` | cross_attention_auc=0.85, gated_prior_kl=0.14, spatial_only=0.01 | 66.44 | 64.29 | 67.84 | 61.97 | 55.79 | 60.61 | `[[177, 52], [135, 118]]` |
| fold2 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.79, gated_prior_kl=0.13, spatial_only=0.08 | 65.65 | 67.06 | 67.45 | 62.70 | 61.73 | 62.44 | `[[155, 74], [107, 146]]` |
| fold2 | `ensemble_logit_auc_roc` | cross_attention_auc=0.79, gated_prior_kl=0.21, spatial_only=0.00 | 66.48 | 64.37 | 67.54 | 61.93 | 53.69 | 60.00 | `[[185, 44], [144, 109]]` |
| fold3 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, spatial_only=0.00 | 53.48 | 55.45 | 57.93 | 52.58 | 35.69 | 48.11 | `[[174, 45], [182, 63]]` |
| fold3 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, spatial_only=0.00 | 59.02 | 58.33 | 62.33 | 59.66 | 61.28 | 59.63 | `[[129, 90], [97, 148]]` |
| fold3 | `spatial_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, spatial_only=1.00 | 58.41 | 60.26 | 55.80 | 54.06 | 30.63 | 47.06 | `[[193, 26], [196, 49]]` |
| fold3 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.29, gated_prior_kl=0.07, spatial_only=0.64 | 58.65 | 63.14 | 58.72 | 54.63 | 33.03 | 48.29 | `[[191, 28], [191, 54]]` |
| fold3 | `ensemble_raw_auc_roc` | cross_attention_auc=0.00, gated_prior_kl=0.14, spatial_only=0.86 | 59.91 | 61.22 | 58.75 | 54.62 | 29.03 | 46.72 | `[[199, 20], [200, 45]]` |
| fold3 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.29, gated_prior_kl=0.07, spatial_only=0.64 | 58.68 | 63.14 | 58.73 | 54.63 | 33.03 | 48.29 | `[[191, 28], [191, 54]]` |
| fold3 | `ensemble_logit_auc_roc` | cross_attention_auc=0.00, gated_prior_kl=0.24, spatial_only=0.76 | 59.94 | 60.26 | 60.14 | 55.43 | 31.21 | 48.02 | `[[199, 20], [196, 49]]` |

## Interpretation

- This report only compares models that have both validation and test predictions for the selected folds.
- The main comparison target is `gated_prior_kl`, because it is the current primary model.
- A useful ensemble should improve mean test AUC or mean calibrated BA without relying on a single fold spike.
