# Final Ensemble Error Analysis

- Clips: 1430
- Accuracy: 57.34
- Balanced accuracy: 57.26
- Error rate: 42.66

## Error Counts

| error_type | n |
| --- | --- |
| TP | 443 |
| TN | 377 |
| FN | 312 |
| FP | 298 |

## Worst Hosts

| host | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 51.62 | 51.91 | 54.53 | 40 | 42 | 46 | 55 |
| LS | 374 | 55.41 | 55.88 | 58.43 | 85 | 81 | 84 | 124 |
| YW | 338 | 56.74 | 57.10 | 59.23 | 81 | 72 | 73 | 112 |
| BRI | 269 | 57.64 | 58.74 | 60.23 | 101 | 57 | 54 | 57 |
| AN | 266 | 61.84 | 62.03 | 69.80 | 70 | 46 | 55 | 95 |

## Worst Episodes

| episode_group_id | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AN_EP24 | 17 | 14.29 | 23.53 | 21.43 | 0 | 3 | 10 | 4 |
| SB_EP36 | 15 | 22.22 | 26.67 | 33.33 | 0 | 6 | 5 | 4 |
| YW_EP49 | 22 | 26.79 | 27.27 | 34.82 | 4 | 10 | 6 | 2 |
| YW_EP46 | 18 | 28.75 | 27.78 | 27.50 | 2 | 8 | 5 | 3 |
| SB_EP35 | 19 | 32.39 | 31.58 | 37.50 | 3 | 8 | 5 | 3 |
| SB_EP32 | 13 | 35.00 | 53.85 | 23.33 | 0 | 3 | 3 | 7 |
| LS_EP3 | 20 | 36.90 | 45.00 | 28.57 | 1 | 5 | 6 | 8 |
| LS_EP10 | 24 | 38.89 | 45.83 | 31.11 | 1 | 8 | 5 | 10 |
| LS_EP6 | 17 | 39.29 | 41.18 | 41.43 | 5 | 5 | 5 | 2 |
| SB_EP29 | 17 | 39.29 | 41.18 | 55.71 | 2 | 5 | 5 | 5 |
| LS_EP12 | 22 | 39.58 | 50.00 | 56.25 | 1 | 5 | 6 | 10 |
| AN_EP28 | 17 | 40.15 | 47.06 | 62.12 | 7 | 4 | 5 | 1 |

## Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 180 | 59.36 | 60.00 | 62.27 | 40.00 | 49.72 |
| True | 1250 | 56.92 | 56.96 | 60.66 | 43.04 | 49.21 |

## Most Confident False Positives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold2 | LS_WILTY_EP9_truth2 | LS | LS_EP9 | 0.5689 | 0.4819 | 0.0871 |
| fold2 | BRI_WILTY_EP57_truth_25 | BRI | BRI_EP57 | 0.5684 | 0.4819 | 0.0865 |
| fold2 | BRI_WILTY_EP57_truth_27 | BRI | BRI_EP57 | 0.5682 | 0.4819 | 0.0863 |
| fold2 | BRI_WILTY_EP57_truth_24 | BRI | BRI_EP57 | 0.5664 | 0.4819 | 0.0845 |
| fold2 | LS_WILTY_EP5_truth13 | LS | LS_EP5 | 0.5641 | 0.4819 | 0.0822 |
| fold2 | BRI_WILTY_EP65_truth_16 | BRI | BRI_EP65 | 0.5609 | 0.4819 | 0.0791 |
| fold2 | LS_WILTY_EP1_truth5 | LS | LS_EP1 | 0.5593 | 0.4819 | 0.0774 |
| fold2 | LS_WILTY_EP5_truth6 | LS | LS_EP5 | 0.5588 | 0.4819 | 0.0769 |
| fold2 | BRI_WILTY_EP62_truth_10 | BRI | BRI_EP62 | 0.5578 | 0.4819 | 0.0759 |
| fold1 | YW_WILTY_EP49_truth14 | YW | YW_EP49 | 0.5794 | 0.5064 | 0.0730 |
| fold2 | BRI_WILTY_EP64_truth_9 | BRI | BRI_EP64 | 0.5538 | 0.4819 | 0.0719 |
| fold2 | BRI_WILTY_EP57_truth_29 | BRI | BRI_EP57 | 0.5518 | 0.4819 | 0.0699 |

## Most Confident False Negatives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold1 | BRI_WILTY_EP62_lie_30 | BRI | BRI_EP62 | 0.4077 | 0.5064 | 0.0987 |
| fold3 | LS_WILTY_EP2_lie1 | LS | LS_EP2 | 0.4030 | 0.4992 | 0.0963 |
| fold1 | BRI_WILTY_EP62_lie_29 | BRI | BRI_EP62 | 0.4108 | 0.5064 | 0.0955 |
| fold3 | BRI_WILTY_EP57_lie_16 | BRI | BRI_EP57 | 0.4048 | 0.4992 | 0.0945 |
| fold3 | BRI_WILTY_EP60_lie_8 | BRI | BRI_EP60 | 0.4109 | 0.4992 | 0.0883 |
| fold1 | YW_WILTY_EP70_lie17 | YW | YW_EP70 | 0.4198 | 0.5064 | 0.0865 |
| fold3 | LS_WILTY_EP72_lie7 | LS | LS_EP72 | 0.4190 | 0.4992 | 0.0802 |
| fold3 | LS_WILTY_EP9_lie2 | LS | LS_EP9 | 0.4204 | 0.4992 | 0.0789 |
| fold3 | YW_WILTY_EP70_lie20 | YW | YW_EP70 | 0.4218 | 0.4992 | 0.0774 |
| fold3 | LS_WILTY_EP1_lie5 | LS | LS_EP1 | 0.4228 | 0.4992 | 0.0764 |
| fold3 | YW_WILTY_EP70_lie19 | YW | YW_EP70 | 0.4237 | 0.4992 | 0.0756 |
| fold3 | SB_WILTY_EP34_lie10 | SB | SB_EP34 | 0.4241 | 0.4992 | 0.0752 |
