# Final Decision

## Selected Result

Use `ensemble_raw_auc_roc` as the final DOLOS result.

This is a validation-tuned prediction-level ensemble of:

- `cross_attention_auc`
- `gated_prior_kl`

Mean validation-tuned weights across folds:

- `cross_attention_auc`: 0.61
- `gated_prior_kl`: 0.39

## Final Metrics

| Method | Mean AUC | Mean BA@0.5 | Mean Cal. BA | Mean Cal. F1 Lie | Mean Cal. Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cross_attention_auc` | 61.39 | 54.33 | 56.35 | 48.30 | 54.26 |
| `gated_prior_kl` | 63.94 | 57.55 | 59.18 | 55.53 | 58.14 |
| `ensemble_raw_auc_roc` | 65.05 | 58.81 | 60.17 | 54.95 | 58.91 |

Compared with `gated_prior_kl`, the final ensemble improves:

- AUC: +1.11 points
- Calibrated BA: +0.99 points
- Calibrated macro F1: +0.77 points

## DOLOS Paper Position

The DOLOS paper reports PAVF + multi-task at:

- Accuracy: 66.84
- F1: 73.35
- AUC: 64.58

Our final ensemble:

- Calibrated accuracy: 59.45
- Calibrated F1 lie: 54.95
- AUC: 65.05
- Calibrated BA: 60.17

Interpretation:

- Our final AUC is slightly above the paper's reported PAVF + multi-task AUC.
- Our accuracy and F1 are lower, so the result should not be framed as globally better than DOLOS.
- The most defensible claim is: our validation-tuned ensemble improves our strongest single-model baseline and reaches comparable AUC to the DOLOS paper, but remains weaker on classification-threshold metrics.

Paper source: https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Audio-Visual_Deception_Detection_DOLOS_Dataset_and_Parameter-Efficient_Crossmodal_Learning_ICCV_2023_paper.pdf

## Final Artifacts

- `final_results_table.csv`: final mean metrics.
- `final_per_fold_metrics.csv`: per-fold metrics.
- `dolos_paper_comparison.csv`: paper-vs-ours comparison.
- `final_ensemble_predictions_with_errors.csv`: final predictions with metadata and error labels.
- `final_error_analysis.md`: final error analysis.
- `final_results_summary.md`: report-ready final summary.
