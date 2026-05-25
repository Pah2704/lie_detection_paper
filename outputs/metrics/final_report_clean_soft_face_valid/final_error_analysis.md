# Final Ensemble Error Analysis

- Clips: 1430
- Accuracy: 58.18
- Balanced accuracy: 58.11
- Error rate: 41.82

## Error Counts

| error_type | n |
| --- | --- |
| TP | 448 |
| TN | 384 |
| FN | 307 |
| FP | 291 |

## Worst Hosts

| host | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 50.51 | 50.82 | 52.64 | 39 | 43 | 47 | 54 |
| YW | 338 | 55.86 | 56.51 | 59.42 | 75 | 78 | 69 | 116 |
| LS | 374 | 57.58 | 57.75 | 57.72 | 93 | 73 | 85 | 123 |
| BRI | 269 | 59.76 | 60.59 | 59.84 | 102 | 56 | 50 | 61 |
| AN | 266 | 63.66 | 63.53 | 68.49 | 75 | 41 | 56 | 94 |

## Worst Episodes

| episode_group_id | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AN_EP24 | 17 | 10.71 | 17.65 | 14.29 | 0 | 3 | 11 | 3 |
| SB_EP36 | 15 | 22.22 | 26.67 | 40.74 | 0 | 6 | 5 | 4 |
| YW_EP46 | 18 | 28.75 | 27.78 | 30.00 | 2 | 8 | 5 | 3 |
| YW_EP49 | 22 | 29.46 | 27.27 | 35.71 | 3 | 11 | 5 | 3 |
| SB_EP32 | 13 | 30.00 | 46.15 | 36.67 | 0 | 3 | 4 | 6 |
| YW_EP52 | 19 | 32.39 | 31.58 | 56.82 | 3 | 8 | 5 | 3 |
| SB_EP33 | 9 | 33.33 | 33.33 | 33.33 | 2 | 4 | 2 | 1 |
| YW_EP56 | 25 | 35.09 | 36.00 | 36.84 | 2 | 4 | 12 | 7 |
| BRI_EP64 | 23 | 35.42 | 34.78 | 16.67 | 5 | 10 | 5 | 3 |
| LS_EP3 | 20 | 36.90 | 45.00 | 28.57 | 1 | 5 | 6 | 8 |
| BRI_EP63 | 30 | 38.46 | 66.67 | 52.88 | 20 | 6 | 4 | 0 |
| SB_EP35 | 19 | 38.64 | 36.84 | 34.09 | 3 | 8 | 4 | 4 |

## Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 179 | 60.29 | 61.45 | 62.71 | 38.55 | 50.11 |
| True | 1251 | 57.71 | 57.71 | 59.82 | 42.29 | 49.53 |

## Most Confident False Positives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold2 | LS_WILTY_EP9_truth2 | LS | LS_EP9 | 0.5663 | 0.4838 | 0.0825 |
| fold2 | BRI_WILTY_EP57_truth_25 | BRI | BRI_EP57 | 0.5657 | 0.4838 | 0.0819 |
| fold2 | BRI_WILTY_EP57_truth_27 | BRI | BRI_EP57 | 0.5654 | 0.4838 | 0.0816 |
| fold2 | BRI_WILTY_EP57_truth_24 | BRI | BRI_EP57 | 0.5640 | 0.4838 | 0.0802 |
| fold2 | LS_WILTY_EP5_truth13 | LS | LS_EP5 | 0.5617 | 0.4838 | 0.0779 |
| fold2 | BRI_WILTY_EP65_truth_16 | BRI | BRI_EP65 | 0.5587 | 0.4838 | 0.0749 |
| fold2 | LS_WILTY_EP1_truth5 | LS | LS_EP1 | 0.5572 | 0.4838 | 0.0734 |
| fold2 | LS_WILTY_EP5_truth6 | LS | LS_EP5 | 0.5566 | 0.4838 | 0.0728 |
| fold2 | BRI_WILTY_EP62_truth_10 | BRI | BRI_EP62 | 0.5557 | 0.4838 | 0.0719 |
| fold2 | BRI_WILTY_EP64_truth_9 | BRI | BRI_EP64 | 0.5520 | 0.4838 | 0.0683 |
| fold2 | BRI_WILTY_EP57_truth_29 | BRI | BRI_EP57 | 0.5501 | 0.4838 | 0.0663 |
| fold2 | BRI_WILTY_EP57_truth_26 | BRI | BRI_EP57 | 0.5494 | 0.4838 | 0.0656 |

## Most Confident False Negatives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold1 | LS_WILTY_EP73_ lie6 | LS | LS_EP73 | 0.4235 | 0.5151 | 0.0916 |
| fold1 | BRI_WILTY_EP62_lie_29 | BRI | BRI_EP62 | 0.4257 | 0.5151 | 0.0894 |
| fold1 | YW_WILTY_EP70_lie17 | YW | YW_EP70 | 0.4311 | 0.5151 | 0.0839 |
| fold1 | BRI_WILTY_EP62_lie_30 | BRI | BRI_EP62 | 0.4312 | 0.5151 | 0.0839 |
| fold1 | BRI_WILTY_EP60_lie_13 | BRI | BRI_EP60 | 0.4440 | 0.5151 | 0.0710 |
| fold2 | YW_WILTY_EP56_lie10 | YW | YW_EP56 | 0.4137 | 0.4838 | 0.0701 |
| fold2 | BRI_WILTY_EP63_lie_30 | BRI | BRI_EP63 | 0.4147 | 0.4838 | 0.0691 |
| fold2 | YW_WILTY_EP46_lie6 | YW | YW_EP46 | 0.4149 | 0.4838 | 0.0689 |
| fold1 | YW_WILTY_EP50_lie9 | YW | YW_EP50 | 0.4463 | 0.5151 | 0.0688 |
| fold1 | YW_WILTY_EP53_lie1 | YW | YW_EP53 | 0.4463 | 0.5151 | 0.0687 |
| fold2 | BRI_WILTY_EP59_lie_15 | BRI | BRI_EP59 | 0.4152 | 0.4838 | 0.0686 |
| fold2 | BRI_WILTY_EP59_lie_10 | BRI | BRI_EP59 | 0.4155 | 0.4838 | 0.0683 |
