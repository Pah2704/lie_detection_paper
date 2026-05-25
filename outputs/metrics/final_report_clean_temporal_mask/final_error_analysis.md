# Final Ensemble Error Analysis

- Clips: 1430
- Accuracy: 59.37
- Balanced accuracy: 59.60
- Error rate: 40.63

## Error Counts

| error_type | n |
| --- | --- |
| TN | 430 |
| TP | 419 |
| FN | 336 |
| FP | 245 |

## Worst Hosts

| host | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 53.30 | 53.01 | 52.00 | 46 | 36 | 50 | 51 |
| YW | 338 | 58.00 | 57.99 | 61.38 | 89 | 64 | 78 | 107 |
| LS | 374 | 58.67 | 58.29 | 63.18 | 103 | 63 | 93 | 115 |
| BRI | 269 | 60.72 | 62.83 | 64.65 | 115 | 43 | 57 | 54 |
| AN | 266 | 63.86 | 63.53 | 66.51 | 77 | 39 | 58 | 92 |

## Worst Episodes

| episode_group_id | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AN_EP24 | 17 | 10.71 | 17.65 | 42.86 | 0 | 3 | 11 | 3 |
| YW_EP56 | 25 | 21.93 | 16.00 | 54.39 | 2 | 4 | 17 | 2 |
| SB_EP36 | 15 | 22.22 | 26.67 | 33.33 | 0 | 6 | 5 | 4 |
| LS_EP1 | 17 | 29.17 | 29.41 | 38.89 | 2 | 6 | 6 | 3 |
| SB_EP33 | 9 | 33.33 | 44.44 | 27.78 | 4 | 2 | 3 | 0 |
| YW_EP46 | 18 | 35.00 | 33.33 | 25.00 | 2 | 8 | 4 | 4 |
| SB_EP32 | 13 | 35.00 | 53.85 | 43.33 | 0 | 3 | 3 | 7 |
| LS_EP3 | 20 | 36.90 | 45.00 | 26.19 | 1 | 5 | 6 | 8 |
| SB_EP35 | 19 | 38.64 | 36.84 | 17.05 | 3 | 8 | 4 | 4 |
| YW_EP52 | 19 | 38.64 | 36.84 | 38.64 | 3 | 8 | 4 | 4 |
| YW_EP49 | 22 | 40.18 | 40.91 | 27.68 | 6 | 8 | 5 | 3 |
| LS_EP72 | 25 | 43.40 | 40.00 | 54.17 | 5 | 4 | 11 | 5 |

## Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 180 | 60.32 | 61.11 | 66.07 | 38.89 | 52.71 |
| True | 1250 | 59.35 | 59.12 | 62.00 | 40.88 | 51.59 |

## Most Confident False Positives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold1 | YW_WILTY_EP49_truth14 | YW | YW_EP49 | 0.7102 | 0.5524 | 0.1578 |
| fold1 | LS_WILTY_EP72_true9 | LS | LS_EP72 | 0.6966 | 0.5524 | 0.1442 |
| fold1 | YW_WILTY_EP47_truth7 | YW | YW_EP47 | 0.6864 | 0.5524 | 0.1340 |
| fold1 | YW_WILTY_EP49_truth13 | YW | YW_EP49 | 0.6849 | 0.5524 | 0.1325 |
| fold1 | LS_WILTY_EP3_truth1 | LS | LS_EP3 | 0.6750 | 0.5524 | 0.1227 |
| fold1 | SB_WILTY_EP32_truth3 | SB | SB_EP32 | 0.6747 | 0.5524 | 0.1223 |
| fold1 | YW_WILTY_EP49_truth11 | YW | YW_EP49 | 0.6698 | 0.5524 | 0.1174 |
| fold1 | SB_WILTY_EP36_truth4 | SB | SB_EP36 | 0.6647 | 0.5524 | 0.1123 |
| fold1 | LS_WILTY_EP10_truth7 | LS | LS_EP10 | 0.6640 | 0.5524 | 0.1117 |
| fold1 | YW_WILTY_EP51_truth5 | YW | YW_EP51 | 0.6611 | 0.5524 | 0.1087 |
| fold1 | LS_WILTY_EP3_truth4 | LS | LS_EP3 | 0.6456 | 0.5524 | 0.0933 |
| fold1 | YW_WILTY_EP49_truth10 | YW | YW_EP49 | 0.6415 | 0.5524 | 0.0891 |

## Most Confident False Negatives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold1 | BRI_WILTY_EP62_lie_30 | BRI | BRI_EP62 | 0.3414 | 0.5524 | 0.2109 |
| fold1 | BRI_WILTY_EP62_lie_29 | BRI | BRI_EP62 | 0.3548 | 0.5524 | 0.1976 |
| fold1 | YW_WILTY_EP70_lie17 | YW | YW_EP70 | 0.3686 | 0.5524 | 0.1838 |
| fold1 | YW_WILTY_EP53_lie1 | YW | YW_EP53 | 0.3855 | 0.5524 | 0.1669 |
| fold1 | YW_WILTY_EP54_lie3 | YW | YW_EP54 | 0.3929 | 0.5524 | 0.1594 |
| fold1 | YW_WILTY_EP54_lie4 | YW | YW_EP54 | 0.3952 | 0.5524 | 0.1572 |
| fold1 | SB_WILTY_EP29_lie2 | SB | SB_EP29 | 0.3985 | 0.5524 | 0.1538 |
| fold1 | SB_WILTY_EP34_lie7 | SB | SB_EP34 | 0.4119 | 0.5524 | 0.1405 |
| fold1 | AN_WILTY_EP22_lie5 | AN | AN_EP22 | 0.4193 | 0.5524 | 0.1330 |
| fold1 | YW_WILTY_EP48_lie3 | YW | YW_EP48 | 0.4235 | 0.5524 | 0.1289 |
| fold1 | YW_WILTY_EP49_lie5 | YW | YW_EP49 | 0.4274 | 0.5524 | 0.1249 |
| fold1 | LS_WILTY_EP10_lie1 | LS | LS_EP10 | 0.4279 | 0.5524 | 0.1245 |
