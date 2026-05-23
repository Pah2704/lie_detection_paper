# Final Ensemble Error Analysis

- Clips: 1430
- Accuracy: 60.56
- Balanced accuracy: 60.58
- Error rate: 39.44

## Error Counts

| error_type | n |
| --- | --- |
| TP | 455 |
| TN | 411 |
| FN | 300 |
| FP | 264 |

## Worst Hosts

| host | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 51.70 | 51.37 | 54.06 | 45 | 37 | 52 | 49 |
| YW | 338 | 57.72 | 57.99 | 59.66 | 84 | 69 | 73 | 112 |
| AN | 266 | 62.02 | 62.78 | 60.80 | 65 | 51 | 48 | 102 |
| BRI | 269 | 62.92 | 64.31 | 62.42 | 112 | 46 | 50 | 61 |
| LS | 374 | 63.12 | 63.10 | 62.43 | 105 | 61 | 77 | 131 |

## Worst Episodes

| episode_group_id | n | balanced_accuracy | accuracy | auc_roc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB_EP35 | 19 | 26.14 | 26.32 | 53.41 | 3 | 8 | 6 | 2 |
| YW_EP46 | 18 | 27.50 | 27.78 | 22.50 | 3 | 7 | 6 | 2 |
| SB_EP36 | 15 | 30.56 | 33.33 | 40.74 | 1 | 5 | 5 | 4 |
| YW_EP50 | 30 | 33.33 | 33.33 | 35.11 | 2 | 13 | 7 | 8 |
| BRI_EP64 | 23 | 35.42 | 34.78 | 35.00 | 5 | 10 | 5 | 3 |
| YW_EP49 | 22 | 36.61 | 36.36 | 31.25 | 5 | 9 | 5 | 3 |
| LS_EP3 | 20 | 36.90 | 45.00 | 46.43 | 1 | 5 | 6 | 8 |
| BRI_EP66 | 19 | 37.50 | 63.16 | 64.58 | 12 | 4 | 3 | 0 |
| YW_EP52 | 19 | 38.64 | 36.84 | 23.86 | 3 | 8 | 4 | 4 |
| LS_EP4 | 25 | 39.24 | 44.00 | 70.14 | 2 | 7 | 7 | 9 |
| LS_EP6 | 17 | 39.29 | 41.18 | 17.14 | 5 | 5 | 5 | 2 |
| AN_EP19 | 21 | 40.00 | 57.14 | 34.44 | 0 | 6 | 3 | 12 |

## Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 180 | 64.17 | 65.56 | 60.55 | 34.44 | 51.99 |
| True | 1250 | 59.91 | 59.84 | 60.86 | 40.16 | 51.56 |

## Most Confident False Positives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold3 | AN_WILTY_EP17_truth5 | AN | AN_EP17 | 0.5915 | 0.5323 | 0.0592 |
| fold3 | LS_WILTY_EP14_truth8 | LS | LS_EP14 | 0.5837 | 0.5323 | 0.0514 |
| fold3 | SB_WILTY_EP29_truth7 | SB | SB_EP29 | 0.5815 | 0.5323 | 0.0492 |
| fold3 | BRI_WILTY_EP63_truth_31 | BRI | BRI_EP63 | 0.5783 | 0.5323 | 0.0460 |
| fold3 | YW_WILTY_EP47_truth6 | YW | YW_EP47 | 0.5776 | 0.5323 | 0.0453 |
| fold3 | AN_WILTY_EP16_truth23 | AN | AN_EP16 | 0.5758 | 0.5323 | 0.0435 |
| fold3 | AN_WILTY_EP16_truth24 | AN | AN_EP16 | 0.5755 | 0.5323 | 0.0432 |
| fold3 | YW_WILTY_EP45_truth7 | YW | YW_EP45 | 0.5746 | 0.5323 | 0.0423 |
| fold3 | LS_WILTY_EP9_truth10 | LS | LS_EP9 | 0.5696 | 0.5323 | 0.0373 |
| fold3 | BRI_WILTY_EP59_truth_12 | BRI | BRI_EP59 | 0.5686 | 0.5323 | 0.0363 |
| fold3 | YW_WILTY_EP46_truth6 | YW | YW_EP46 | 0.5680 | 0.5323 | 0.0357 |
| fold3 | LS_WILTY_EP10_truth9 | LS | LS_EP10 | 0.5677 | 0.5323 | 0.0354 |

## Most Confident False Negatives

| fold | video_id | host | episode_group_id | score_lie | threshold | margin |
| --- | --- | --- | --- | --- | --- | --- |
| fold3 | BRI_WILTY_EP60_lie_8 | BRI | BRI_EP60 | 0.4806 | 0.5323 | 0.0517 |
| fold3 | AN_WILTY_EP24_lie14 | AN | AN_EP24 | 0.4814 | 0.5323 | 0.0509 |
| fold3 | LS_WILTY_EP2_lie1 | LS | LS_EP2 | 0.4906 | 0.5323 | 0.0417 |
| fold3 | LS_WILTY_EP13_lie1 | LS | LS_EP13 | 0.4916 | 0.5323 | 0.0407 |
| fold3 | SB_WILTY_EP34_lie10 | SB | SB_EP34 | 0.4945 | 0.5323 | 0.0378 |
| fold3 | LS_WILTY_EP75_lie8 | LS | LS_EP75 | 0.4946 | 0.5323 | 0.0377 |
| fold3 | SB_WILTY_EP33_lie4 | SB | SB_EP33 | 0.4952 | 0.5323 | 0.0371 |
| fold3 | LS_WILTY_EP10_lie15 | LS | LS_EP10 | 0.4954 | 0.5323 | 0.0369 |
| fold3 | SB_WILTY_EP41_lie15 | SB | SB_EP41 | 0.4981 | 0.5323 | 0.0342 |
| fold3 | SB_WILTY_EP42_lie10 | SB | SB_EP42 | 0.4989 | 0.5323 | 0.0334 |
| fold3 | AN_WILTY_EP17_lie13 | AN | AN_EP17 | 0.5012 | 0.5323 | 0.0311 |
| fold3 | AN_WILTY_EP26_lie18 | AN | AN_EP26 | 0.5026 | 0.5323 | 0.0297 |
