# Fold3 Local Cross-Attention Ablation

Protocol: DOLOS-only clean cache, fold3, seed 42.

Change tested:

- Add shared sinusoidal temporal positional encoding before cross-attention.
- Restrict audio-to-visual cross-attention to a local temporal radius of 2 tokens.
- Keep frame-level `face_valid` handling and all-invalid fallback to avoid NaN.

## Fold3 Results

| Method | Best epoch | Val AUC | Test AUC | Raw BA @ 0.5 | Calibrated BA | Calibrated F1 lie |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cross-attention clean baseline | 1 | 54.54 | 56.92 | 50.00 | 52.75 | 19.58 |
| cross-attention + local temporal mask | 1 | 52.38 | 55.60 | 50.00 | 51.64 | 32.16 |
| gated prior-KL + temporal face-valid mask | 12 | 61.64 | 65.46 | 53.09 | 59.08 | 58.39 |

## Decision

Reject this local cross-attention ablation.

The local attention constraint did not improve fold3. It reduced validation AUC and test AUC compared with the clean cross-attention baseline, and it remains far below the gated temporal-mask model. The most likely reason is that the 16 audio tokens are produced by learned attention pooling over wav2vec states, so they are not guaranteed to align frame-by-frame with the visual tokens. A hard local window can therefore remove useful cross-modal correspondences instead of removing only noise.

Keep the current main result unchanged:

- `outputs/metrics/final_report_clean_temporal_mask/final_results_summary.md`
- final method: `ensemble_raw_auc_roc`
- mean AUC: 65.32
- mean calibrated BA: 59.61
