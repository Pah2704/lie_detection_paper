# Final DOLOS Results

## Final Decision

- Final method: `ensemble_raw_balanced_accuracy`.
- It is a validation-tuned prediction-level ensemble of `cross_attention_auc` and `gated_prior_kl`.
- Mean AUC changes from 61.42 to 61.97 (+0.55 points).
- Mean calibrated BA changes from 57.51 to 57.92 (+0.41 points).

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
| fold1 | cross_attention_auc=0.00;gated_prior_kl=1.00 | 0.5165 | 61.20 | 57.18 | 55.26 | 55.98 | [[129, 98], [119, 138]] |
| fold2 | cross_attention_auc=0.21;gated_prior_kl=0.79 | 0.5147 | 67.01 | 62.50 | 62.92 | 59.87 | [[166, 63], [118, 135]] |
| fold3 | cross_attention_auc=0.26;gated_prior_kl=0.74 | 0.4939 | 57.70 | 55.24 | 55.57 | 55.29 | [[129, 90], [117, 128]] |

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
| ours | Final ensemble raw-BA | 57.67 | 57.92 | 57.05 | 61.97 | mean over fold calibrated thresholds |

## Error Analysis Highlights

### Worst Hosts By Calibrated BA

| host | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 82 | 101 | 50.86 | 50.82 | 53.51 | 49.73 | 42 | 40 | 50 | 51 |
| BRI | 269 | 158 | 111 | 54.05 | 56.88 | 60.14 | 33.09 | 111 | 47 | 69 | 42 |
| LS | 374 | 166 | 208 | 58.12 | 58.02 | 59.14 | 50.00 | 98 | 68 | 89 | 119 |
| YW | 338 | 153 | 185 | 59.81 | 59.47 | 59.64 | 47.34 | 97 | 56 | 81 | 104 |
| AN | 266 | 116 | 150 | 61.09 | 60.53 | 70.90 | 46.99 | 76 | 40 | 65 | 85 |

### Worst Episodes By Calibrated BA

| episode_group_id | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB_EP36 | 15 | 6 | 9 | 22.22 | 26.67 | 29.63 | 66.67 | 0 | 6 | 5 | 4 |
| SB_EP32 | 13 | 3 | 10 | 25.00 | 38.46 | 33.33 | 61.54 | 0 | 3 | 5 | 5 |
| LS_EP3 | 20 | 6 | 14 | 26.19 | 30.00 | 32.14 | 50.00 | 1 | 5 | 9 | 5 |
| YW_EP46 | 18 | 10 | 8 | 26.25 | 27.78 | 25.00 | 38.89 | 4 | 6 | 7 | 1 |
| SB_EP31 | 15 | 8 | 7 | 26.79 | 26.67 | 26.79 | 53.33 | 2 | 6 | 5 | 2 |
| AN_EP24 | 17 | 3 | 14 | 27.38 | 23.53 | 11.90 | 29.41 | 1 | 2 | 11 | 3 |
| LS_EP6 | 17 | 10 | 7 | 30.00 | 35.29 | 48.57 | 23.53 | 6 | 4 | 7 | 0 |
| SB_EP35 | 19 | 11 | 8 | 32.39 | 31.58 | 44.32 | 57.89 | 3 | 8 | 5 | 3 |
| YW_EP52 | 19 | 11 | 8 | 33.52 | 36.84 | 48.86 | 31.58 | 6 | 5 | 7 | 1 |
| YW_EP49 | 22 | 14 | 8 | 33.93 | 36.36 | 39.29 | 45.45 | 6 | 8 | 6 | 2 |
| AN_EP19 | 21 | 6 | 15 | 35.00 | 42.86 | 52.22 | 61.90 | 1 | 5 | 7 | 8 |
| BRI_EP66 | 19 | 16 | 3 | 37.50 | 63.16 | 37.50 | 21.05 | 12 | 4 | 3 | 0 |

### Face Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 180 | 60.07 | 60.00 | 61.07 | 40.00 | 49.61 |
| True | 1250 | 57.59 | 57.36 | 61.46 | 42.64 | 49.34 |

## Interpretation

- The final ensemble beats both constituent models on mean AUC and calibrated BA.
- The strongest single model remains `gated_prior_kl`.
- Adding standalone audio/spatial predictions was tested separately and rejected because validation tuning did not generalize as well as the 2-model ensemble.
- Error concentration by host/episode remains material, so report conclusions should emphasize protocol-level average performance and residual group sensitivity.
