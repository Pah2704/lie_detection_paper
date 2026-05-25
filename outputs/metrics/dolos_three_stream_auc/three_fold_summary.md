# Three-fold DOLOS Summary

## Per fold, threshold 0.5

| Fold | Best epoch | ACC | BA | F1-Lie | Macro F1 | AUC | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fold1 | 12 | 62.81 | 61.82 | 68.97 | 61.29 | 65.28 | [[104, 123], [57, 200]] |
| fold2 | 2 | 52.49 | 50.00 | 68.84 | 34.42 | 60.95 | [[0, 229], [0, 253]] |
| fold3 | 1 | 48.92 | 51.17 | 18.56 | 40.68 | 57.93 | [[200, 19], [218, 27]] |

## Mean ± std

| Setting | ACC | BA | F1-Lie | Macro F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| threshold 0.5 | 54.74 ± 7.21 | 54.33 ± 6.51 | 52.12 ± 29.07 | 45.46 ± 14.06 | 61.39 ± 3.69 |
| calibrated | 55.47 ± 3.83 | 56.35 ± 3.26 | 48.30 ± 11.41 | 54.26 ± 5.38 | 61.39 ± 3.69 |

## Combined out-of-fold

- Threshold 0.5: ACC 54.83, F1-Lie 59.78, AUC 59.38, confusion [[304, 371], [275, 480]]
- Per-fold calibrated: ACC 55.52, F1-Lie 49.44, AUC 59.38, confusion [[483, 192], [444, 311]]
