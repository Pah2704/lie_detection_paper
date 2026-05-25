# DOLOS Gated Logit Prior-KL Final Error Analysis

## Final Position

- Gated logits prior-KL improves mean AUC from 61.39 to 63.94.
- Mean calibrated balanced accuracy improves from 56.35 to 59.18.
- Combined calibrated BA improves from 56.37 to 59.20.
- This should be the main baseline for the report, replacing the earlier cross-attention run.

## Final Tables

| Model | Mean AUC | Mean BA@0.5 | Mean Cal BA | Combined AUC | Combined Cal BA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-attn AUC baseline | 61.39 | 54.33 | 56.35 | 59.38 | 56.37 |
| Gated logits prior-KL | 63.94 | 57.55 | 59.18 | 61.35 | 59.20 |

| Fold | AUC | BA@0.5 | Cal BA | F1@0.5 | Cal F1 | Cal Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fold1 | 61.77 | 54.63 | 57.45 | 68.85 | 55.56 | `[[146, 81], [127, 130]]` |
| fold2 | 67.73 | 58.87 | 60.44 | 69.13 | 49.74 | `[[189, 40], [156, 97]]` |
| fold3 | 62.33 | 59.15 | 59.66 | 64.38 | 61.28 | `[[129, 90], [97, 148]]` |

## Figures

- `plots/final_baseline_comparison.png`: final model vs cross-attention baseline.
- `plots/per_fold_metrics.png`: fold-level AUC, BA, calibrated BA, F1.
- `plots/calibrated_confusion_matrices.png`: calibrated confusion matrices per fold.
- `plots/roc_pr_curves.png`: ROC and PR curves per fold.
- `plots/score_distributions_by_fold.png`: score overlap by label and fold.
- `plots/host_calibrated_ba.png`: host-level calibrated BA.
- `plots/calibrated_error_counts.png`: TP/TN/FP/FN counts under calibrated threshold.
- `plots/window_instability_by_correctness.png`: window-score variance for correct vs wrong clips.

## Error Patterns

### Worst Hosts By Calibrated BA (n >= 20)

| Host | n | Lie | Truth | Cal BA | Acc | Pred Lie Rate | AUC | CM tn/fp/fn/tp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SB | 183 | 101 | 82 | 50.18 | 49.18 | 40.44 | 47.27 | 49/33/60/41 |
| BRI | 269 | 111 | 158 | 55.14 | 59.11 | 26.39 | 62.85 | 123/35/75/36 |
| YW | 338 | 185 | 153 | 59.25 | 59.17 | 50.00 | 60.41 | 92/61/77/108 |
| AN | 266 | 150 | 116 | 60.74 | 59.02 | 37.97 | 65.63 | 86/30/79/71 |
| LS | 374 | 208 | 166 | 62.94 | 62.30 | 45.72 | 62.03 | 114/52/89/119 |

### Best Hosts By Calibrated BA (n >= 20)

| Host | n | Lie | Truth | Cal BA | Acc | Pred Lie Rate | AUC | CM tn/fp/fn/tp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LS | 374 | 208 | 166 | 62.94 | 62.30 | 45.72 | 62.03 | 114/52/89/119 |
| AN | 266 | 150 | 116 | 60.74 | 59.02 | 37.97 | 65.63 | 86/30/79/71 |
| YW | 338 | 185 | 153 | 59.25 | 59.17 | 50.00 | 60.41 | 92/61/77/108 |
| BRI | 269 | 111 | 158 | 55.14 | 59.11 | 26.39 | 62.85 | 123/35/75/36 |
| SB | 183 | 101 | 82 | 50.18 | 49.18 | 40.44 | 47.27 | 49/33/60/41 |

### Worst Episodes By Calibrated BA (n >= 8)

| Episode | n | Lie | Truth | Cal BA | Acc | Pred Lie Rate | AUC | CM tn/fp/fn/tp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LS_EP6 | 17 | 7 | 10 | 20.00 | 23.53 | 35.29 | 28.57 | 4/6/7/0 |
| AN_EP24 | 17 | 14 | 3 | 20.24 | 11.76 | 17.65 | 40.48 | 1/2/13/1 |
| SB_EP32 | 13 | 10 | 3 | 25.00 | 38.46 | 61.54 | 26.67 | 0/3/5/5 |
| SB_EP35 | 19 | 8 | 11 | 30.68 | 31.58 | 47.37 | 10.23 | 4/7/6/2 |
| SB_EP33 | 9 | 3 | 6 | 33.33 | 44.44 | 22.22 | 38.89 | 4/2/3/0 |
| YW_EP46 | 18 | 8 | 10 | 33.75 | 33.33 | 55.56 | 23.75 | 3/7/5/3 |
| AN_EP18 | 23 | 15 | 8 | 35.83 | 39.13 | 56.52 | 41.67 | 2/6/8/7 |
| SB_EP34 | 15 | 11 | 4 | 38.64 | 33.33 | 33.33 | 27.27 | 2/2/8/3 |
| YW_EP52 | 19 | 8 | 11 | 39.77 | 42.11 | 36.84 | 63.64 | 6/5/6/2 |
| YW_EP50 | 30 | 15 | 15 | 40.00 | 40.00 | 43.33 | 55.11 | 7/8/10/5 |
| YW_EP49 | 22 | 8 | 14 | 40.18 | 40.91 | 50.00 | 34.82 | 6/8/5/3 |
| SB_EP31 | 15 | 7 | 8 | 40.18 | 40.00 | 53.33 | 42.86 | 3/5/4/3 |

### Most Confident False Positives (Calibrated)

| Fold | Video | Host | Episode | Score | Thr | Margin | Window Std |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| fold1 | `YW_WILTY_EP49_truth14` | YW | YW_EP49 | 0.8236 | 0.6055 | 0.2182 | 0.0192 |
| fold1 | `YW_WILTY_EP49_truth13` | YW | YW_EP49 | 0.8020 | 0.6055 | 0.1966 | 0.0058 |
| fold1 | `YW_WILTY_EP47_truth4` | YW | YW_EP47 | 0.7829 | 0.6055 | 0.1774 | 0.0196 |
| fold1 | `LS_WILTY_EP1_truth2` | LS | LS_EP1 | 0.7824 | 0.6055 | 0.1769 | 0.0180 |
| fold1 | `YW_WILTY_EP46_truth4` | YW | YW_EP46 | 0.7794 | 0.6055 | 0.1739 | 0.0364 |
| fold1 | `AN_WILTY_EP18_truth19` | AN | AN_EP18 | 0.7678 | 0.6055 | 0.1623 | 0.0314 |
| fold1 | `SB_WILTY_EP32_truth3` | SB | SB_EP32 | 0.7578 | 0.6055 | 0.1523 | 0.0322 |
| fold1 | `YW_WILTY_EP46_truth9` | YW | YW_EP46 | 0.7544 | 0.6055 | 0.1489 | 0.0770 |
| fold1 | `AN_WILTY_EP16_truth25` | AN | AN_EP16 | 0.7467 | 0.6055 | 0.1412 | 0.0483 |
| fold1 | `AN_WILTY_EP15_truth7` | AN | AN_EP15 | 0.7463 | 0.6055 | 0.1409 | 0.0245 |

### Most Confident False Negatives (Calibrated)

| Fold | Video | Host | Episode | Score | Thr | Margin | Window Std |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| fold2 | `BRI_WILTY_EP63_lie_30` | BRI | BRI_EP63 | 0.2326 | 0.5356 | 0.3030 | 0.0386 |
| fold2 | `YW_WILTY_EP46_lie6` | YW | YW_EP46 | 0.2356 | 0.5356 | 0.3000 | 0.0639 |
| fold2 | `AN_WILTY_EP24_lie7` | AN | AN_EP24 | 0.2451 | 0.5356 | 0.2905 | 0.0861 |
| fold2 | `SB_WILTY_EP29_lie8` | SB | SB_EP29 | 0.2747 | 0.5356 | 0.2609 | 0.1164 |
| fold2 | `BRI_WILTY_EP59_lie_10` | BRI | BRI_EP59 | 0.2783 | 0.5356 | 0.2573 | 0.0718 |
| fold2 | `LS_WILTY_EP14_lie6` | LS | LS_EP14 | 0.2902 | 0.5356 | 0.2454 | 0.1068 |
| fold2 | `SB_WILTY_EP29_lie9` | SB | SB_EP29 | 0.3018 | 0.5356 | 0.2338 | 0.0895 |
| fold2 | `SB_WILTY_EP35_lie2` | SB | SB_EP35 | 0.3096 | 0.5356 | 0.2260 | 0.1128 |
| fold1 | `LS_WILTY_EP6_lie1` | LS | LS_EP6 | 0.3933 | 0.6055 | 0.2121 | 0.0191 |
| fold1 | `LS_WILTY_EP72_lie4` | LS | LS_EP72 | 0.4049 | 0.6055 | 0.2006 | 0.0546 |

## Interpretation For Next Step

- Fold2 is the largest gain: AUC 60.95 -> 67.73, calibrated BA 58.26 -> 60.44.
- Fold3 also improves: AUC 57.93 -> 62.33, calibrated BA 52.58 -> 59.66.
- Fold1 loses AUC against the old cross-attention run: 65.28 -> 61.77, although calibrated BA remains close: 58.21 vs 57.45.
- The model still has threshold/score-distribution sensitivity: F1@0.5 is high because the default threshold predicts lie often, while calibrated threshold improves BA but reduces lie recall on fold2.
- This supports a small stabilization ablation, but only after keeping gated prior-KL as the report baseline.

## Recommended Small Ablation

- Do not reduce the spatial backbone yet.
- Try one small variant: normalize or temperature-scale each stream logits before gated summation, and set aux weights to spatial 0.05, flow 0.20, audio 0.20.
- Run only fold1 and fold3 first. Success criteria: improve fold1 AUC without losing fold3 calibrated BA; do not run full 3-fold unless both are non-regressive.
