# Final DOLOS Results

## Final Decision

- Final method: `ensemble_raw_auc_roc`.
- It is a validation-tuned prediction-level ensemble of `cross_attention_auc` and `gated_prior_kl`.
- Mean AUC changes from 60.81 to 61.94 (+1.14 points).
- Mean calibrated BA changes from 56.78 to 58.10 (+1.32 points).

## Mean 3-Fold Metrics

| method_label | auc_roc | balanced_accuracy_at_0p5 | calibrated_balanced_accuracy | calibrated_f1_lie | calibrated_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| Cross-attention AUC baseline | 60.59 | 55.24 | 56.70 | 48.06 | 53.02 |
| Final ensemble raw-AUC | 61.94 | 56.11 | 58.10 | 59.89 | 57.95 |
| Final ensemble raw-BA | 61.72 | 56.26 | 57.57 | 63.09 | 56.69 |
| Gated logits prior-KL | 60.81 | 54.58 | 56.78 | 58.51 | 56.64 |

## Final Ensemble Per Fold

| fold | weights | threshold | auc_roc | balanced_accuracy_at_0p5 | calibrated_balanced_accuracy | calibrated_f1_lie | calibrated_confusion_matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold1 | cross_attention_auc=0.16;gated_prior_kl=0.84 | 0.5151 | 59.54 | 54.44 | 57.15 | 56.00 | [[142, 85], [124, 133]] |
| fold2 | cross_attention_auc=0.94;gated_prior_kl=0.06 | 0.4838 | 65.72 | 58.23 | 60.94 | 62.55 | [[137, 92], [96, 157]] |
| fold3 | cross_attention_auc=0.00;gated_prior_kl=1.00 | 0.4959 | 60.57 | 55.66 | 56.22 | 61.12 | [[105, 114], [87, 158]] |

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
| ours | Gated logits prior-KL | 56.85 | 56.78 | 58.51 | 60.81 | mean over fold calibrated thresholds |
| ours | Final ensemble raw-AUC | 58.17 | 58.10 | 59.89 | 61.94 | mean over fold calibrated thresholds |

## Error Analysis Highlights

### Worst Hosts By Calibrated BA

| host | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 82 | 101 | 50.51 | 50.82 | 52.64 | 53.01 | 39 | 43 | 47 | 54 |
| YW | 338 | 153 | 185 | 55.86 | 56.51 | 59.42 | 57.40 | 75 | 78 | 69 | 116 |
| LS | 374 | 166 | 208 | 57.58 | 57.75 | 57.72 | 52.41 | 93 | 73 | 85 | 123 |
| BRI | 269 | 158 | 111 | 59.76 | 60.59 | 59.84 | 43.49 | 102 | 56 | 50 | 61 |
| AN | 266 | 116 | 150 | 63.66 | 63.53 | 68.49 | 50.75 | 75 | 41 | 56 | 94 |

### Worst Episodes By Calibrated BA

| episode_group_id | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AN_EP24 | 17 | 3 | 14 | 10.71 | 17.65 | 14.29 | 35.29 | 0 | 3 | 11 | 3 |
| SB_EP36 | 15 | 6 | 9 | 22.22 | 26.67 | 40.74 | 66.67 | 0 | 6 | 5 | 4 |
| YW_EP46 | 18 | 10 | 8 | 28.75 | 27.78 | 30.00 | 61.11 | 2 | 8 | 5 | 3 |
| YW_EP49 | 22 | 14 | 8 | 29.46 | 27.27 | 35.71 | 63.64 | 3 | 11 | 5 | 3 |
| SB_EP32 | 13 | 3 | 10 | 30.00 | 46.15 | 36.67 | 69.23 | 0 | 3 | 4 | 6 |
| YW_EP52 | 19 | 11 | 8 | 32.39 | 31.58 | 56.82 | 57.89 | 3 | 8 | 5 | 3 |
| SB_EP33 | 9 | 6 | 3 | 33.33 | 33.33 | 33.33 | 55.56 | 2 | 4 | 2 | 1 |
| YW_EP56 | 25 | 6 | 19 | 35.09 | 36.00 | 36.84 | 44.00 | 2 | 4 | 12 | 7 |
| BRI_EP64 | 23 | 15 | 8 | 35.42 | 34.78 | 16.67 | 56.52 | 5 | 10 | 5 | 3 |
| LS_EP3 | 20 | 6 | 14 | 36.90 | 45.00 | 28.57 | 65.00 | 1 | 5 | 6 | 8 |
| BRI_EP63 | 30 | 26 | 4 | 38.46 | 66.67 | 52.88 | 20.00 | 20 | 6 | 4 | 0 |
| SB_EP35 | 19 | 11 | 8 | 38.64 | 36.84 | 34.09 | 63.16 | 3 | 8 | 4 | 4 |

### Face Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 179 | 60.29 | 61.45 | 62.71 | 38.55 | 50.11 |
| True | 1251 | 57.71 | 57.71 | 59.82 | 42.29 | 49.53 |

## Interpretation

- The final ensemble beats both constituent models on mean AUC and calibrated BA.
- The strongest single model remains `gated_prior_kl`.
- Adding standalone audio/spatial predictions was tested separately and rejected because validation tuning did not generalize as well as the 2-model ensemble.
- Error concentration by host/episode remains material, so report conclusions should emphasize protocol-level average performance and residual group sensitivity.
