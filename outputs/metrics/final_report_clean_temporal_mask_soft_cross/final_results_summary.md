# Final DOLOS Results

## Final Decision

- Final method: `ensemble_raw_balanced_accuracy`.
- It is a validation-tuned prediction-level ensemble of `cross_attention_auc` and `gated_prior_kl`.
- Mean AUC changes from 64.38 to 65.43 (+1.05 points).
- Mean calibrated BA changes from 59.39 to 60.54 (+1.15 points).

## Mean 3-Fold Metrics

| method_label | auc_roc | balanced_accuracy_at_0p5 | calibrated_balanced_accuracy | calibrated_f1_lie | calibrated_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| Cross-attention AUC baseline | 61.98 | 53.31 | 57.22 | 48.36 | 52.30 |
| Final ensemble raw-AUC | 65.20 | 54.25 | 60.00 | 59.06 | 59.53 |
| Final ensemble raw-BA | 65.43 | 54.90 | 60.54 | 61.39 | 60.13 |
| Gated logits prior-KL | 64.38 | 55.71 | 59.39 | 60.27 | 59.26 |

## Final Ensemble Per Fold

| fold | weights | threshold | auc_roc | balanced_accuracy_at_0p5 | calibrated_balanced_accuracy | calibrated_f1_lie | calibrated_confusion_matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold1 | cross_attention_auc=0.85;gated_prior_kl=0.15 | 0.5001 | 65.41 | 59.83 | 59.63 | 62.98 | [[125, 102], [92, 165]] |
| fold2 | cross_attention_auc=1.00;gated_prior_kl=0.00 | 0.5116 | 64.73 | 49.82 | 61.27 | 55.81 | [[172, 57], [133, 120]] |
| fold3 | cross_attention_auc=0.11;gated_prior_kl=0.89 | 0.5323 | 66.14 | 55.04 | 60.72 | 65.38 | [[114, 105], [75, 170]] |

## DOLOS Paper Comparison

Paper source: https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Audio-Visual_Deception_Detection_DOLOS_Dataset_and_Parameter-Efficient_Crossmodal_Learning_ICCV_2023_paper.pdf

The DOLOS paper reports ACC, F1, and AUC for the official 3-fold average. BA is not reported in the paper, so BA below is only for our models.

| source_group | method | acc | balanced_accuracy | f1_lie | auc_roc | notes |
| --- | --- | --- | --- | --- | --- | --- |
| paper | DOLOS paper Visual | 61.44 |  | 69.42 | 58.89 | Table 2 |
| paper | DOLOS paper Audio | 59.19 |  | 73.46 | 52.54 | Table 2 |
| paper | DOLOS paper Concatenation | 61.62 |  | 70.20 | 60.50 | Table 2 |
| paper | DOLOS paper PAVF | 64.75 |  | 71.20 | 62.71 | Table 2 |
| paper | DOLOS paper PAVF + Multi-task | 66.84 |  | 73.35 | 64.58 | Table 2 |
| ours | Cross-attention AUC baseline | 56.70 | 57.22 | 48.36 | 61.98 | mean over fold calibrated thresholds |
| ours | Gated logits prior-KL | 59.36 | 59.39 | 60.27 | 64.38 | mean over fold calibrated thresholds |
| ours | Final ensemble raw-BA | 60.57 | 60.54 | 61.39 | 65.43 | mean over fold calibrated thresholds |

## Error Analysis Highlights

### Worst Hosts By Calibrated BA

| host | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 82 | 101 | 51.70 | 51.37 | 54.06 | 46.99 | 45 | 37 | 52 | 49 |
| YW | 338 | 153 | 185 | 57.72 | 57.99 | 59.66 | 53.55 | 84 | 69 | 73 | 112 |
| AN | 266 | 116 | 150 | 62.02 | 62.78 | 60.80 | 57.52 | 65 | 51 | 48 | 102 |
| BRI | 269 | 158 | 111 | 62.92 | 64.31 | 62.42 | 39.78 | 112 | 46 | 50 | 61 |
| LS | 374 | 166 | 208 | 63.12 | 63.10 | 62.43 | 51.34 | 105 | 61 | 77 | 131 |

### Worst Episodes By Calibrated BA

| episode_group_id | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB_EP35 | 19 | 11 | 8 | 26.14 | 26.32 | 53.41 | 52.63 | 3 | 8 | 6 | 2 |
| YW_EP46 | 18 | 10 | 8 | 27.50 | 27.78 | 22.50 | 50.00 | 3 | 7 | 6 | 2 |
| SB_EP36 | 15 | 6 | 9 | 30.56 | 33.33 | 40.74 | 60.00 | 1 | 5 | 5 | 4 |
| YW_EP50 | 30 | 15 | 15 | 33.33 | 33.33 | 35.11 | 70.00 | 2 | 13 | 7 | 8 |
| BRI_EP64 | 23 | 15 | 8 | 35.42 | 34.78 | 35.00 | 56.52 | 5 | 10 | 5 | 3 |
| YW_EP49 | 22 | 14 | 8 | 36.61 | 36.36 | 31.25 | 54.55 | 5 | 9 | 5 | 3 |
| LS_EP3 | 20 | 6 | 14 | 36.90 | 45.00 | 46.43 | 65.00 | 1 | 5 | 6 | 8 |
| BRI_EP66 | 19 | 16 | 3 | 37.50 | 63.16 | 64.58 | 21.05 | 12 | 4 | 3 | 0 |
| YW_EP52 | 19 | 11 | 8 | 38.64 | 36.84 | 23.86 | 63.16 | 3 | 8 | 4 | 4 |
| LS_EP4 | 25 | 9 | 16 | 39.24 | 44.00 | 70.14 | 64.00 | 2 | 7 | 7 | 9 |
| LS_EP6 | 17 | 10 | 7 | 39.29 | 41.18 | 17.14 | 41.18 | 5 | 5 | 5 | 2 |
| AN_EP19 | 21 | 6 | 15 | 40.00 | 57.14 | 34.44 | 85.71 | 0 | 6 | 3 | 12 |

### Face Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 180 | 64.17 | 65.56 | 60.55 | 34.44 | 51.99 |
| True | 1250 | 59.91 | 59.84 | 60.86 | 40.16 | 51.56 |

## Interpretation

- The final ensemble beats both constituent models on mean AUC and calibrated BA.
- The strongest single model remains `gated_prior_kl`.
- Adding standalone audio/spatial predictions was tested separately and rejected because validation tuning did not generalize as well as the 2-model ensemble.
- Error concentration by host/episode remains material, so report conclusions should emphasize protocol-level average performance and residual group sensitivity.
