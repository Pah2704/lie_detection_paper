# Final DOLOS Results

## Final Decision

- Final method: `ensemble_raw_auc_roc`.
- It is a validation-tuned prediction-level ensemble of `cross_attention_auc` and `gated_prior_kl`.
- Mean AUC improves from 63.94 to 65.05.
- Mean calibrated BA improves from 59.18 to 60.17.

## Mean 3-Fold Metrics

| method_label | auc_roc | balanced_accuracy_at_0p5 | calibrated_balanced_accuracy | calibrated_f1_lie | calibrated_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| Cross-attention AUC baseline | 61.39 | 54.33 | 56.35 | 48.30 | 54.26 |
| Final ensemble raw-AUC | 65.05 | 58.81 | 60.17 | 54.95 | 58.91 |
| Final ensemble raw-BA | 64.87 | 58.21 | 60.08 | 58.62 | 58.91 |
| Gated logits prior-KL | 63.94 | 57.55 | 59.18 | 55.53 | 58.14 |

## Final Ensemble Per Fold

| fold | weights | threshold | auc_roc | balanced_accuracy_at_0p5 | calibrated_balanced_accuracy | calibrated_f1_lie | calibrated_confusion_matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold1 | cross_attention_auc=1.00;gated_prior_kl=0.00 | 0.6781 | 65.28 | 61.82 | 58.21 | 51.29 | [[168, 59], [148, 109]] |
| fold2 | cross_attention_auc=0.79;gated_prior_kl=0.21 | 0.5154 | 67.54 | 55.65 | 61.93 | 53.69 | [[185, 44], [144, 109]] |
| fold3 | cross_attention_auc=0.04;gated_prior_kl=0.96 | 0.5139 | 62.35 | 58.94 | 60.36 | 59.87 | [[141, 78], [107, 138]] |

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
| ours | Cross-attention AUC baseline | 55.47 | 56.35 | 48.30 | 61.39 | mean over fold calibrated thresholds |
| ours | Gated logits prior-KL | 58.69 | 59.18 | 55.53 | 63.94 | mean over fold calibrated thresholds |
| ours | Final ensemble raw-AUC | 59.45 | 60.17 | 54.95 | 65.05 | mean over fold calibrated thresholds |

## Error Analysis Highlights

### Worst Hosts By Calibrated BA

| host | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | 183 | 82 | 101 | 51.85 | 50.27 | 48.36 | 34.97 | 55 | 27 | 64 | 37 |
| BRI | 269 | 158 | 111 | 55.74 | 60.59 | 65.49 | 21.19 | 132 | 26 | 80 | 31 |
| YW | 338 | 153 | 185 | 59.83 | 58.88 | 62.11 | 40.83 | 107 | 46 | 93 | 92 |
| LS | 374 | 166 | 208 | 62.76 | 62.03 | 62.45 | 44.92 | 115 | 51 | 91 | 117 |
| AN | 266 | 116 | 150 | 62.97 | 61.65 | 68.16 | 41.35 | 85 | 31 | 71 | 79 |

### Worst Episodes By Calibrated BA

| episode_group_id | n | truth | lie | balanced_accuracy | accuracy | auc_roc | pred_lie_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB_EP32 | 13 | 3 | 10 | 20.00 | 30.77 | 46.67 | 53.85 | 0 | 3 | 6 | 4 |
| LS_EP6 | 17 | 10 | 7 | 25.00 | 29.41 | 40.00 | 29.41 | 5 | 5 | 7 | 0 |
| YW_EP46 | 18 | 10 | 8 | 27.50 | 27.78 | 21.25 | 50.00 | 3 | 7 | 6 | 2 |
| SB_EP35 | 19 | 11 | 8 | 28.98 | 31.58 | 9.09 | 36.84 | 5 | 6 | 7 | 1 |
| BRI_EP64 | 23 | 15 | 8 | 36.25 | 43.48 | 36.67 | 30.43 | 9 | 6 | 7 | 1 |
| YW_EP52 | 19 | 11 | 8 | 36.36 | 42.11 | 45.45 | 15.79 | 8 | 3 | 8 | 0 |
| LS_EP3 | 20 | 6 | 14 | 36.90 | 45.00 | 20.24 | 65.00 | 1 | 5 | 6 | 8 |
| AN_EP24 | 17 | 3 | 14 | 36.90 | 17.65 | 19.05 | 11.76 | 2 | 1 | 13 | 1 |
| SB_EP37 | 21 | 13 | 8 | 38.46 | 47.62 | 53.85 | 14.29 | 10 | 3 | 8 | 0 |
| SB_EP31 | 15 | 8 | 7 | 40.18 | 40.00 | 42.86 | 53.33 | 3 | 5 | 4 | 3 |
| SB_EP33 | 9 | 6 | 3 | 41.67 | 55.56 | 38.89 | 11.11 | 5 | 1 | 3 | 0 |
| SB_EP34 | 15 | 4 | 11 | 43.18 | 40.00 | 25.00 | 40.00 | 2 | 2 | 7 | 4 |

### Face Contamination Heuristic

| clip_suspect_contamination | n | balanced_accuracy | accuracy | auc_roc | error_rate | mean_score_lie |
| --- | --- | --- | --- | --- | --- | --- |
| False | 179 | 66.78 | 67.04 | 66.95 | 32.96 | 53.65 |
| True | 1251 | 59.02 | 58.35 | 62.55 | 41.65 | 51.75 |

## Interpretation

- The final ensemble beats both constituent models on mean AUC and calibrated BA.
- The strongest single model remains `gated_prior_kl`.
- Adding standalone audio/spatial predictions was tested separately and rejected because validation tuning did not generalize as well as the 2-model ensemble.
- Error concentration by host/episode remains material, so report conclusions should emphasize protocol-level average performance and residual group sensitivity.
