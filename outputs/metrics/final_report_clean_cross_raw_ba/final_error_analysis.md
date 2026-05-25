# Final Ensemble Error Analysis

- Clips: 1430
- Accuracy: 57.69
- Balanced accuracy: 57.96
- Error rate: 42.31

## Error Counts

| error_type | n |
| --- | --- |
| TN | 424 |
| TP | 401 |
| FN | 354 |
| FP | 251 |

## Worst Hosts

| host | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 50.86 | 50.82 | 53.51 | 42 | 40 | 50 | 51 |
| BRI | 269 | 54.05 | 56.88 | 60.14 | 111 | 47 | 69 | 42 |
| LS | 374 | 58.12 | 58.02 | 59.14 | 98 | 68 | 89 | 119 |
| YW | 338 | 59.81 | 59.47 | 59.64 | 97 | 56 | 81 | 104 |
| AN | 266 | 61.09 | 60.53 | 70.90 | 76 | 40 | 65 | 85 |

## Worst Episodes

| episode_group_id | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB_EP36 | 15 | 22.22 | 26.67 | 29.63 | 0 | 6 | 5 | 4 |
| SB_EP32 | 13 | 25.00 | 38.46 | 33.33 | 0 | 3 | 5 | 5 |
| LS_EP3 | 20 | 26.19 | 30.00 | 32.14 | 1 | 5 | 9 | 5 |
| YW_EP46 | 18 | 26.25 | 27.78 | 25.00 | 4 | 6 | 7 | 1 |
| SB_EP31 | 15 | 26.79 | 26.67 | 26.79 | 2 | 6 | 5 | 2 |
| AN_EP24 | 17 | 27.38 | 23.53 | 11.90 | 1 | 2 | 11 | 3 |
| LS_EP6 | 17 | 30.00 | 35.29 | 48.57 | 6 | 4 | 7 | 0 |
| SB_EP35 | 19 | 32.39 | 31.58 | 44.32 | 3 | 8 | 5 | 3 |
| YW_EP52 | 19 | 33.52 | 36.84 | 48.86 | 6 | 5 | 7 | 1 |
| YW_EP49 | 22 | 33.93 | 36.36 | 39.29 | 6 | 8 | 6 | 2 |
| AN_EP19 | 21 | 35.00 | 42.86 | 52.22 | 1 | 5 | 7 | 8 |
| BRI_EP66 | 19 | 37.50 | 63.16 | 37.50 | 12 | 4 | 3 | 0 |

## Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 180 | 60.07 | 60.00 | 61.07 | 40.00 | 49.61 |
| True | 1250 | 57.59 | 57.36 | 61.46 | 42.64 | 49.34 |

## Most Confident False Positives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold1 | YW_WILTY_EP49_truth14 | YW | YW_EP49 | 0.5803 | 0.5165 | 0.0637 |
| fold1 | YW_WILTY_EP49_truth13 | YW | YW_EP49 | 0.5730 | 0.5165 | 0.0565 |
| fold1 | LS_WILTY_EP72_true9 | LS | LS_EP72 | 0.5686 | 0.5165 | 0.0520 |
| fold1 | YW_WILTY_EP49_truth11 | YW | YW_EP49 | 0.5670 | 0.5165 | 0.0505 |
| fold1 | YW_WILTY_EP46_truth4 | YW | YW_EP46 | 0.5633 | 0.5165 | 0.0467 |
| fold1 | YW_WILTY_EP46_truth3 | YW | YW_EP46 | 0.5611 | 0.5165 | 0.0446 |
| fold1 | BRI_WILTY_EP57_truth_28 | BRI | BRI_EP57 | 0.5586 | 0.5165 | 0.0421 |
| fold1 | YW_WILTY_EP51_truth5 | YW | YW_EP51 | 0.5567 | 0.5165 | 0.0402 |
| fold1 | SB_WILTY_EP36_truth4 | SB | SB_EP36 | 0.5558 | 0.5165 | 0.0392 |
| fold1 | LS_WILTY_EP10_truth7 | LS | LS_EP10 | 0.5553 | 0.5165 | 0.0388 |
| fold1 | YW_WILTY_EP49_truth10 | YW | YW_EP49 | 0.5543 | 0.5165 | 0.0378 |
| fold1 | LS_WILTY_EP4_truth2 | LS | LS_EP4 | 0.5537 | 0.5165 | 0.0372 |

## Most Confident False Negatives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold2 | AN_WILTY_EP24_lie7 | AN | AN_EP24 | 0.3280 | 0.5147 | 0.1868 |
| fold2 | YW_WILTY_EP46_lie6 | YW | YW_EP46 | 0.3353 | 0.5147 | 0.1794 |
| fold2 | BRI_WILTY_EP63_lie_30 | BRI | BRI_EP63 | 0.3531 | 0.5147 | 0.1617 |
| fold2 | YW_WILTY_EP56_lie10 | YW | YW_EP56 | 0.3556 | 0.5147 | 0.1591 |
| fold2 | SB_WILTY_EP35_lie2 | SB | SB_EP35 | 0.3561 | 0.5147 | 0.1586 |
| fold2 | SB_WILTY_EP29_lie8 | SB | SB_EP29 | 0.3618 | 0.5147 | 0.1529 |
| fold2 | YW_WILTY_EP46_lie1 | YW | YW_EP46 | 0.3626 | 0.5147 | 0.1521 |
| fold2 | BRI_WILTY_EP59_lie_15 | BRI | BRI_EP59 | 0.3628 | 0.5147 | 0.1519 |
| fold2 | YW_WILTY_EP56_lie1 | YW | YW_EP56 | 0.3711 | 0.5147 | 0.1436 |
| fold2 | LS_WILTY_EP14_lie6 | LS | LS_EP14 | 0.3806 | 0.5147 | 0.1342 |
| fold2 | BRI_WILTY_EP60_lie_14 | BRI | BRI_EP60 | 0.3830 | 0.5147 | 0.1317 |
| fold2 | BRI_WILTY_EP59_lie_10 | BRI | BRI_EP59 | 0.3846 | 0.5147 | 0.1301 |
