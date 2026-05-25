# Soft face-valid 3-fold comparison

| Variant | Fold | Best ep | Val AUC | Test AUC | Test BA@0.5 | Test cal BA | Test F1@0.5 | Test cal F1 |
|---|---|---|---|---|---|---|---|---|
| hard_mask | fold1 | 5 | 59.94 | 61.20 | 57.18 | 55.26 | 65.53 | 55.98 |
| hard_mask | fold2 | 5 | 64.41 | 66.06 | 62.55 | 61.47 | 66.03 | 56.15 |
| hard_mask | fold3 | 2 | 63.86 | 57.00 | 55.45 | 55.79 | 56.24 | 57.38 |
| soft_face_valid | fold1 | 2 | 58.17 | 59.59 | 55.64 | 55.60 | 64.76 | 56.91 |
| soft_face_valid | fold2 | 1 | 63.65 | 62.25 | 52.43 | 58.54 | 68.76 | 57.51 |
| soft_face_valid | fold3 | 2 | 64.67 | 60.57 | 55.66 | 56.22 | 38.97 | 61.12 |
| hard_mask | mean_3fold |  | 62.73 | 61.42 | 58.39 | 57.51 | 62.60 | 56.50 |
| soft_face_valid | mean_3fold |  | 62.16 | 60.81 | 54.58 | 56.78 | 57.49 | 58.51 |

## Takeaway

Soft face-valid changes 3-fold gated prior-KL mean AUC from 61.42 to 60.81, and calibrated BA from 57.51 to 56.78.
