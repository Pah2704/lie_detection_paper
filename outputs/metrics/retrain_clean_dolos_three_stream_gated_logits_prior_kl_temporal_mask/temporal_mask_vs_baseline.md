# Temporal Face-Valid Mask Ablation

Protocol: DOLOS-only clean cache, 3-fold, seed 42.

Change tested:

- Use frame-level `face_valid_frames` as a temporal mask for spatial/flow pooling and attention.
- Do not multiply visual features directly by validity weights.
- Downweight visual gates with `log(face_valid_ratio + eps)`.
- If all visual frames are invalid, force visual gates near zero and keep the audio gate available.

## Mean 3-Fold Comparison

| Method | Mean AUC | Raw BA @ 0.5 | Calibrated BA |
| --- | ---: | ---: | ---: |
| gated prior-KL clean | 61.42 | 58.39 | 57.51 |
| gated prior-KL + temporal mask | 64.38 | 55.71 | 59.39 |
| cross-attention clean | 60.59 | 55.24 | 56.70 |
| ensemble raw-AUC: cross-attention + temporal-mask gated | 65.32 | 56.87 | 59.61 |
| ensemble logit-AUC: cross-attention + temporal-mask gated | 65.51 | 57.15 | 59.72 |

## Per-Fold Temporal-Mask Gated Model

| Fold | Best epoch | Val AUC | Test AUC | Raw BA @ 0.5 | Calibrated BA |
| --- | ---: | ---: | ---: | ---: | ---: |
| fold1 | 6 | 64.58 | 65.51 | 60.70 | 60.02 |
| fold2 | 1 | 63.75 | 62.17 | 53.35 | 59.07 |
| fold3 | 12 | 61.64 | 65.46 | 53.09 | 59.08 |

## Decision

Use `ensemble_raw_auc_roc` from `outputs/metrics/prediction_level_ensemble_retrain_clean_temporal_mask/` as the current report result because it follows the existing final-report protocol and improves over both clean cross-attention and clean gated prior-KL.

Generated final report:

- `outputs/metrics/final_report_clean_temporal_mask/final_results_summary.md`
- `outputs/metrics/final_report_clean_temporal_mask/final_error_analysis.md`
