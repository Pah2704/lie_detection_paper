# Final DOLOS Results

## Final Decision

- Final method: `ensemble_raw_auc_roc`.
- It is a validation-tuned prediction-level ensemble of `cross_attention_auc` and `gated_prior_kl`.
- Mean AUC changes from 64.38 to 65.32 (+0.95 points).
- Mean calibrated BA changes from 59.39 to 59.61 (+0.22 points).

## Mean 3-Fold Metrics

| method_label | auc_roc | balanced_accuracy_at_0p5 | calibrated_balanced_accuracy | calibrated_f1_lie | calibrated_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| Cross-attention AUC baseline | 60.59 | 55.24 | 56.70 | 48.06 | 53.02 |
| Final ensemble raw-AUC | 65.32 | 56.87 | 59.61 | 58.94 | 59.30 |
| Final ensemble raw-BA | 65.27 | 57.24 | 59.44 | 62.10 | 59.29 |
| Gated logits prior-KL | 64.38 | 55.71 | 59.39 | 60.27 | 59.26 |

## Final Ensemble Per Fold

| fold | weights | threshold | auc_roc | balanced_accuracy_at_0p5 | calibrated_balanced_accuracy | calibrated_f1_lie | calibrated_confusion_matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold1 | cross_attention_auc=0.26;gated_prior_kl=0.74 | 0.5524 | 64.84 | 59.05 | 59.24 | 56.33 | [[155, 72], [128, 129]] |
| fold2 | cross_attention_auc=0.94;gated_prior_kl=0.06 | 0.4838 | 65.72 | 58.23 | 60.94 | 62.55 | [[137, 92], [96, 157]] |
| fold3 | cross_attention_auc=0.04;gated_prior_kl=0.96 | 0.5430 | 65.41 | 53.32 | 58.65 | 57.95 | [[138, 81], [112, 133]] |

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
| ours | Cross-attention AUC baseline | 56.15 | 56.70 | 48.06 | 60.59 | mean over fold calibrated thresholds |
| ours | Gated logits prior-KL | 59.36 | 59.39 | 60.27 | 64.38 | mean over fold calibrated thresholds |
| ours | Final ensemble raw-AUC | 59.36 | 59.61 | 58.94 | 65.32 | mean over fold calibrated thresholds |

## Error Analysis Highlights

### Worst Hosts By Calibrated BA

| host | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 82 | 101 | 53.30 | 53.01 | 52.00 | 47.54 | 46 | 36 | 50 | 51 |
| YW | 338 | 153 | 185 | 58.00 | 57.99 | 61.38 | 50.59 | 89 | 64 | 78 | 107 |
| LS | 374 | 166 | 208 | 58.67 | 58.29 | 63.18 | 47.59 | 103 | 63 | 93 | 115 |
| BRI | 269 | 158 | 111 | 60.72 | 62.83 | 64.65 | 36.06 | 115 | 43 | 57 | 54 |
| AN | 266 | 116 | 150 | 63.86 | 63.53 | 66.51 | 49.25 | 77 | 39 | 58 | 92 |

### Worst Episodes By Calibrated BA

| episode_group_id | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AN_EP24 | 17 | 3 | 14 | 10.71 | 17.65 | 42.86 | 35.29 | 0 | 3 | 11 | 3 |
| YW_EP56 | 25 | 6 | 19 | 21.93 | 16.00 | 54.39 | 24.00 | 2 | 4 | 17 | 2 |
| SB_EP36 | 15 | 6 | 9 | 22.22 | 26.67 | 33.33 | 66.67 | 0 | 6 | 5 | 4 |
| LS_EP1 | 17 | 8 | 9 | 29.17 | 29.41 | 38.89 | 52.94 | 2 | 6 | 6 | 3 |
| SB_EP33 | 9 | 6 | 3 | 33.33 | 44.44 | 27.78 | 22.22 | 4 | 2 | 3 | 0 |
| YW_EP46 | 18 | 10 | 8 | 35.00 | 33.33 | 25.00 | 66.67 | 2 | 8 | 4 | 4 |
| SB_EP32 | 13 | 3 | 10 | 35.00 | 53.85 | 43.33 | 76.92 | 0 | 3 | 3 | 7 |
| LS_EP3 | 20 | 6 | 14 | 36.90 | 45.00 | 26.19 | 65.00 | 1 | 5 | 6 | 8 |
| SB_EP35 | 19 | 11 | 8 | 38.64 | 36.84 | 17.05 | 63.16 | 3 | 8 | 4 | 4 |
| YW_EP52 | 19 | 11 | 8 | 38.64 | 36.84 | 38.64 | 63.16 | 3 | 8 | 4 | 4 |
| YW_EP49 | 22 | 14 | 8 | 40.18 | 40.91 | 27.68 | 50.00 | 6 | 8 | 5 | 3 |
| LS_EP72 | 25 | 9 | 16 | 43.40 | 40.00 | 54.17 | 36.00 | 5 | 4 | 11 | 5 |

### Face Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 180 | 60.32 | 61.11 | 66.07 | 38.89 | 52.71 |
| True | 1250 | 59.35 | 59.12 | 62.00 | 40.88 | 51.59 |

## Interpretation

- The final ensemble beats both constituent models on mean AUC and calibrated BA.
- The strongest single model remains `gated_prior_kl`.
- Adding standalone audio/spatial predictions was tested separately and rejected because validation tuning did not generalize as well as the 2-model ensemble.
- Error concentration by host/episode remains material, so report conclusions should emphasize protocol-level average performance and residual group sensitivity.
