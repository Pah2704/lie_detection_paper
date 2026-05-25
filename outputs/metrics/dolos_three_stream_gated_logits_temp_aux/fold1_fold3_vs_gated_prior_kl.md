# Fold1/Fold3 Logit Temperature Ablation

Ablation config: `configs/multimodal_dolos_gated_logits_temp_aux.yaml`.

- Stream logit temperature: `[2.0, 1.0, 1.0]` for spatial/flow/audio.
- Aux loss weights: `[0.05, 0.20, 0.20]` for spatial/flow/audio.
- Backbone unchanged; no full 3-fold run was triggered.

## Result

| Fold | Model | Best epoch | AUC | BA@0.5 | Cal BA | Cal Macro F1 | Cal confusion [[TN,FP],[FN,TP]] |
|---|---:|---:|---:|---:|---:|---:|---|
| fold1 | gated_prior_kl | 13 | 61.77 | 54.63 | 57.45 | 56.98 | `[[146, 81], [127, 130]]` |
| fold1 | temp_aux_ablation | 2 | 56.99 | 52.29 | 54.09 | 54.01 | `[[106, 121], [99, 158]]` |
| fold3 | gated_prior_kl | 3 | 62.33 | 59.15 | 59.66 | 59.63 | `[[129, 90], [97, 148]]` |
| fold3 | temp_aux_ablation | 12 | 62.21 | 51.86 | 59.54 | 54.04 | `[[200, 19], [177, 68]]` |

## Delta vs Gated Prior-KL

| Fold | Delta AUC | Delta Cal BA | Delta BA@0.5 | Decision |
|---|---:|---:|---:|---|
| fold1 | -4.78 | -3.36 | -2.34 | reject: non-regressive criteria failed |
| fold3 | -0.12 | -0.12 | -7.28 | reject: non-regressive criteria failed |

## Decision

Do not run the temp/aux ablation as a full 3-fold experiment.

Fold1 regressed heavily: AUC -4.78 points and calibrated BA -3.36 points. Fold3 was effectively flat to slightly worse: AUC -0.12 points and calibrated BA -0.12 points. The feature norm logs also show that large logit spikes can move between streams across epochs, so a fixed spatial-only temperature does not solve the scale-instability problem.

Keep `dolos_three_stream_gated_logits_prior_kl` as the final trained model for the report. If future work continues, the next technically cleaner ablation would be dynamic per-sample logit normalization or a learned temperature constrained across streams, but it should not be run before finalizing the current report baseline.

Plot: `outputs/metrics/dolos_three_stream_gated_logits_temp_aux/plots/fold1_fold3_ablation_vs_baseline.png`
CSV: `outputs/metrics/dolos_three_stream_gated_logits_temp_aux/fold1_fold3_vs_gated_prior_kl.csv`
