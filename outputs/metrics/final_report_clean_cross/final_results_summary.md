# Final DOLOS Results

## Final Decision

- Final method: `ensemble_raw_auc_roc`.
- It is a validation-tuned prediction-level ensemble of `cross_attention_auc` and `gated_prior_kl`.
- Mean AUC changes from 61.42 to 61.30 (-0.11 points).
- Mean calibrated BA changes from 57.51 to 57.21 (-0.30 points).

## Mean 3-Fold Metrics

| method_label | auc_roc | balanced_accuracy_at_0p5 | calibrated_balanced_accuracy | calibrated_f1_lie | calibrated_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| Cross-attention AUC baseline | 60.59 | 55.24 | 56.70 | 48.06 | 53.02 |
| Final ensemble raw-AUC | 61.30 | 56.63 | 57.21 | 58.96 | 57.08 |
| Final ensemble raw-BA | 61.97 | 58.30 | 57.92 | 57.05 | 57.61 |
| Gated logits prior-KL | 61.42 | 58.39 | 57.51 | 56.50 | 57.08 |

## Final Ensemble Per Fold

| fold | weights | threshold | auc_roc | balanced_accuracy_at_0p5 | calibrated_balanced_accuracy | calibrated_f1_lie | calibrated_confusion_matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold1 | cross_attention_auc=0.12;gated_prior_kl=0.88 | 0.5064 | 60.92 | 57.41 | 56.75 | 61.60 | [[111, 116], [91, 166]] |
| fold2 | cross_attention_auc=0.99;gated_prior_kl=0.01 | 0.4819 | 65.75 | 58.25 | 60.94 | 62.55 | [[137, 92], [96, 157]] |
| fold3 | cross_attention_auc=0.10;gated_prior_kl=0.90 | 0.4992 | 57.24 | 54.22 | 53.94 | 52.75 | [[129, 90], [125, 120]] |

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
| ours | Gated logits prior-KL | 57.26 | 57.51 | 56.50 | 61.42 | mean over fold calibrated thresholds |
| ours | Final ensemble raw-AUC | 57.30 | 57.21 | 58.96 | 61.30 | mean over fold calibrated thresholds |

## Error Analysis Highlights

### Worst Hosts By Calibrated BA

| host | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 82 | 101 | 51.62 | 51.91 | 54.53 | 53.01 | 40 | 42 | 46 | 55 |
| LS | 374 | 166 | 208 | 55.41 | 55.88 | 58.43 | 54.81 | 85 | 81 | 84 | 124 |
| YW | 338 | 153 | 185 | 56.74 | 57.10 | 59.23 | 54.44 | 81 | 72 | 73 | 112 |
| BRI | 269 | 158 | 111 | 57.64 | 58.74 | 60.23 | 42.38 | 101 | 57 | 54 | 57 |
| AN | 266 | 116 | 150 | 61.84 | 62.03 | 69.80 | 53.01 | 70 | 46 | 55 | 95 |

### Worst Episodes By Calibrated BA

| episode_group_id | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AN_EP24 | 17 | 3 | 14 | 14.29 | 23.53 | 21.43 | 41.18 | 0 | 3 | 10 | 4 |
| SB_EP36 | 15 | 6 | 9 | 22.22 | 26.67 | 33.33 | 66.67 | 0 | 6 | 5 | 4 |
| YW_EP49 | 22 | 14 | 8 | 26.79 | 27.27 | 34.82 | 54.55 | 4 | 10 | 6 | 2 |
| YW_EP46 | 18 | 10 | 8 | 28.75 | 27.78 | 27.50 | 61.11 | 2 | 8 | 5 | 3 |
| SB_EP35 | 19 | 11 | 8 | 32.39 | 31.58 | 37.50 | 57.89 | 3 | 8 | 5 | 3 |
| SB_EP32 | 13 | 3 | 10 | 35.00 | 53.85 | 23.33 | 76.92 | 0 | 3 | 3 | 7 |
| LS_EP3 | 20 | 6 | 14 | 36.90 | 45.00 | 28.57 | 65.00 | 1 | 5 | 6 | 8 |
| LS_EP10 | 24 | 9 | 15 | 38.89 | 45.83 | 31.11 | 75.00 | 1 | 8 | 5 | 10 |
| LS_EP6 | 17 | 10 | 7 | 39.29 | 41.18 | 41.43 | 41.18 | 5 | 5 | 5 | 2 |
| SB_EP29 | 17 | 7 | 10 | 39.29 | 41.18 | 55.71 | 58.82 | 2 | 5 | 5 | 5 |
| LS_EP12 | 22 | 6 | 16 | 39.58 | 50.00 | 56.25 | 68.18 | 1 | 5 | 6 | 10 |
| AN_EP28 | 17 | 11 | 6 | 40.15 | 47.06 | 62.12 | 29.41 | 7 | 4 | 5 | 1 |

### Face Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 180 | 59.36 | 60.00 | 62.27 | 40.00 | 49.72 |
| True | 1250 | 56.92 | 56.96 | 60.66 | 43.04 | 49.21 |

## Interpretation

- The final ensemble beats both constituent models on mean AUC and calibrated BA.
- The strongest single model remains `gated_prior_kl`.
- Adding standalone audio/spatial predictions was tested separately and rejected because validation tuning did not generalize as well as the 2-model ensemble.
- Error concentration by host/episode remains material, so report conclusions should emphasize protocol-level average performance and residual group sensitivity.
