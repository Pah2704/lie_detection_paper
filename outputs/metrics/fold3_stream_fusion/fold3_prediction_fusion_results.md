# Fold3 Prediction Fusion

Grid step: `0.01`. Thresholds are selected on validation by balanced accuracy.

| Method | Weights S/F/A | Thr | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro | Test CM | Test score mean/std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| late_raw_balanced_accuracy | 0.05/0.47/0.48 | 0.5290 | 70.32 | 69.87 | 58.10 | 54.31 | 44.22 | 51.98 | `[[161, 58], [159, 86]]` | 0.5265/0.0064 |
| late_raw_auc_roc | 0.09/0.1/0.81 | 0.5360 | 70.56 | 68.59 | 57.58 | 53.13 | 41.16 | 50.27 | `[[163, 56], [167, 78]]` | 0.5305/0.0107 |
| late_logit_balanced_accuracy | 0.05/0.47/0.48 | 0.5290 | 70.32 | 69.87 | 58.09 | 54.31 | 44.22 | 51.98 | `[[161, 58], [159, 86]]` | 0.5265/0.0064 |
| late_logit_auc_roc | 0.09/0.11/0.8 | 0.5357 | 70.56 | 68.59 | 57.61 | 53.34 | 41.58 | 50.53 | `[[163, 56], [166, 79]]` | 0.5303/0.0106 |
| gated_logreg | - | 0.6193 | 70.29 | 68.59 | 57.80 | 54.01 | 44.95 | 51.99 | `[[157, 62], [156, 89]]` | 0.5310/0.1919 |

## Interpretation

- The val-tuned fusion improves validation BA, but fold3 test BA remains modest.
- If best weights collapse to one stream, the other streams are not adding useful validation signal under this objective.
- Compare test AUC with single-stream and full-model summaries before running full 3-fold.
