# Soft face-valid fold1/fold3 comparison

| Variant | Fold | Best ep | Val AUC | Test AUC | Test BA@0.5 | Test cal BA | Test F1@0.5 | Test cal F1 |
|---|---|---|---|---|---|---|---|---|
| hard_mask | fold1 | 5 | 59.94 | 61.20 | 57.18 | 55.26 | 65.53 | 55.98 |
| hard_mask | fold3 | 2 | 63.86 | 57.00 | 55.45 | 55.79 | 56.24 | 57.38 |
| soft_face_valid | fold1 | 2 | 58.17 | 59.59 | 55.64 | 55.60 | 64.76 | 56.91 |
| soft_face_valid | fold3 | 2 | 64.67 | 60.57 | 55.66 | 56.22 | 38.97 | 61.12 |
| hard_mask | mean_fold1_fold3 |  | 61.90 | 59.10 | 56.32 | 55.53 | 60.88 | 56.68 |
| soft_face_valid | mean_fold1_fold3 |  | 61.42 | 60.08 | 55.65 | 55.91 | 51.86 | 59.02 |

## Takeaway

Across fold1+fold3, soft weighting changes mean AUC from 59.10 to 60.08 and mean calibrated BA from 55.53 to 55.91.
This supports replacing the hard visual mask with soft face_valid_ratio if we want to keep a clean DOLOS-only protocol.
