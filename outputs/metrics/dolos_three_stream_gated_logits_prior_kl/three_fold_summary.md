# Three-fold DOLOS Gated Logit Prior-KL Summary

## Per Fold

| Fold | Best epoch | Val AUC | ACC | BA | F1-Lie | Macro F1 | AUC | Cal Thr | Cal BA | Cal F1 | Confusion | Cal Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| fold1 | 13 | 61.54 | 56.82 | 54.63 | 68.85 | 49.24 | 61.77 | 0.6055 | 57.45 | 55.56 | `[[44, 183], [26, 231]]` | `[[146, 81], [127, 130]]` |
| fold2 | 12 | 64.70 | 60.17 | 58.87 | 69.13 | 56.50 | 67.73 | 0.5356 | 60.44 | 49.74 | `[[75, 154], [38, 215]]` | `[[189, 40], [156, 97]]` |
| fold3 | 3 | 59.02 | 59.70 | 59.15 | 64.38 | 58.99 | 62.33 | 0.5114 | 59.66 | 61.28 | `[[108, 111], [76, 169]]` | `[[129, 90], [97, 148]]` |

## Mean Across Folds

| Setting | ACC | BA | F1-Lie | Macro F1 | AUC | AUC-PR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default_threshold_0.5 | 58.89 | 57.55 | 67.46 | 54.91 | 63.94 | 65.61 |
| calibrated_threshold | 58.69 | 59.18 | 55.53 | 58.14 | 63.94 | 65.61 |

## Combined Counts

- Threshold 0.5: ACC 58.88, BA 57.54, F1-Lie 67.66, AUC 61.35, confusion `[[227, 448], [140, 615]]`
- Per-fold calibrated: ACC 58.67, BA 59.20, F1-Lie 55.93, AUC 61.35, confusion `[[464, 211], [380, 375]]`

## Baseline Comparison

| Model | Mean AUC | Mean BA@0.5 | Mean Cal BA | Combined AUC | Combined Cal BA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-attn AUC baseline | 61.39 | 54.33 | 56.35 | 59.38 | 56.37 |
| Gated logits prior-KL | 63.94 | 57.55 | 59.18 | 61.35 | 59.20 |

## Notes

- Fold2 summary is eval-only from the best checkpoint because the original process was interrupted after `/tmp` filled due to system syslog growth.
- Fold3 gated-logit prior-KL beat the fold3 late-fusion target: AUC 62.33 vs 58.10 and calibrated BA 59.66 vs 54.31.
