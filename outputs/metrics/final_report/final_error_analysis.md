# Final Ensemble Error Analysis

- Clips: 1430
- Accuracy: 59.44
- Balanced accuracy: 60.17
- Error rate: 40.56

## Error Counts

| error_type | n |
| --- | --- |
| TN | 494 |
| FN | 399 |
| TP | 356 |
| FP | 181 |

## Worst Hosts

| host | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 51.85 | 50.27 | 48.36 | 55 | 27 | 64 | 37 |
| BRI | 269 | 55.74 | 60.59 | 65.49 | 132 | 26 | 80 | 31 |
| YW | 338 | 59.83 | 58.88 | 62.11 | 107 | 46 | 93 | 92 |
| LS | 374 | 62.76 | 62.03 | 62.45 | 115 | 51 | 91 | 117 |
| AN | 266 | 62.97 | 61.65 | 68.16 | 85 | 31 | 71 | 79 |

## Worst Episodes

| episode_group_id | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB_EP32 | 13 | 20.00 | 30.77 | 46.67 | 0 | 3 | 6 | 4 |
| LS_EP6 | 17 | 25.00 | 29.41 | 40.00 | 5 | 5 | 7 | 0 |
| YW_EP46 | 18 | 27.50 | 27.78 | 21.25 | 3 | 7 | 6 | 2 |
| SB_EP35 | 19 | 28.98 | 31.58 | 9.09 | 5 | 6 | 7 | 1 |
| BRI_EP64 | 23 | 36.25 | 43.48 | 36.67 | 9 | 6 | 7 | 1 |
| YW_EP52 | 19 | 36.36 | 42.11 | 45.45 | 8 | 3 | 8 | 0 |
| LS_EP3 | 20 | 36.90 | 45.00 | 20.24 | 1 | 5 | 6 | 8 |
| AN_EP24 | 17 | 36.90 | 17.65 | 19.05 | 2 | 1 | 13 | 1 |
| SB_EP37 | 21 | 38.46 | 47.62 | 53.85 | 10 | 3 | 8 | 0 |
| SB_EP31 | 15 | 40.18 | 40.00 | 42.86 | 3 | 5 | 4 | 3 |
| SB_EP33 | 9 | 41.67 | 55.56 | 38.89 | 5 | 1 | 3 | 0 |
| SB_EP34 | 15 | 43.18 | 40.00 | 25.00 | 2 | 2 | 7 | 4 |

## Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 179 | 66.78 | 67.04 | 66.95 | 32.96 | 53.65 |
| True | 1251 | 59.02 | 58.35 | 62.55 | 41.65 | 51.75 |

## Most Confident False Positives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold1 | LS_WILTY_EP10_truth7 | LS | LS_EP10 | 0.7224 | 0.6781 | 0.0443 |
| fold1 | YW_WILTY_EP47_truth7 | YW | YW_EP47 | 0.7210 | 0.6781 | 0.0430 |
| fold1 | BRI_WILTY_EP64_truth_19 | BRI | BRI_EP64 | 0.7207 | 0.6781 | 0.0426 |
| fold1 | YW_WILTY_EP49_truth14 | YW | YW_EP49 | 0.7204 | 0.6781 | 0.0424 |
| fold1 | YW_WILTY_EP49_truth13 | YW | YW_EP49 | 0.7202 | 0.6781 | 0.0422 |
| fold1 | YW_WILTY_EP49_truth11 | YW | YW_EP49 | 0.7190 | 0.6781 | 0.0409 |
| fold1 | LS_WILTY_EP76_lie1 | LS | LS_EP76 | 0.7189 | 0.6781 | 0.0408 |
| fold1 | BRI_WILTY_EP57_truth_28 | BRI | BRI_EP57 | 0.7184 | 0.6781 | 0.0403 |
| fold1 | AN_WILTY_EP16_truth25 | AN | AN_EP16 | 0.7183 | 0.6781 | 0.0403 |
| fold1 | YW_WILTY_EP51_truth5 | YW | YW_EP51 | 0.7168 | 0.6781 | 0.0387 |
| fold1 | YW_WILTY_EP47_truth4 | YW | YW_EP47 | 0.7163 | 0.6781 | 0.0383 |
| fold1 | BRI_WILTY_EP64_truth_18 | BRI | BRI_EP64 | 0.7137 | 0.6781 | 0.0356 |

## Most Confident False Negatives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold1 | YW_WILTY_EP49_lie5 | YW | YW_EP49 | 0.2726 | 0.6781 | 0.4055 |
| fold1 | YW_WILTY_EP54_lie4 | YW | YW_EP54 | 0.2742 | 0.6781 | 0.4039 |
| fold1 | BRI_WILTY_EP59_lie_14 | BRI | BRI_EP59 | 0.2754 | 0.6781 | 0.4027 |
| fold1 | SB_WILTY_EP29_lie2 | SB | SB_EP29 | 0.2782 | 0.6781 | 0.3999 |
| fold1 | BRI_WILTY_EP59_deception_23 | BRI | BRI_EP59 | 0.2788 | 0.6781 | 0.3993 |
| fold1 | BRI_WILTY_EP59_deception_21 | BRI | BRI_EP59 | 0.2791 | 0.6781 | 0.3990 |
| fold1 | AN_WILTY_EP24_lie1 | AN | AN_EP24 | 0.2795 | 0.6781 | 0.3986 |
| fold1 | YW_WILTY_EP49_lie2 | YW | YW_EP49 | 0.2804 | 0.6781 | 0.3977 |
| fold1 | YW_WILTY_EP54_lie3 | YW | YW_EP54 | 0.2807 | 0.6781 | 0.3974 |
| fold1 | SB_WILTY_EP37_lie2 | SB | SB_EP37 | 0.2814 | 0.6781 | 0.3967 |
| fold1 | YW_WILTY_EP49_lie1 | YW | YW_EP49 | 0.2823 | 0.6781 | 0.3958 |
| fold1 | YW_WILTY_EP53_lie1 | YW | YW_EP53 | 0.2834 | 0.6781 | 0.3946 |
