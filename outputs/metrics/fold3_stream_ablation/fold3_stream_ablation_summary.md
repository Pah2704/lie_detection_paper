# Fold3 Stream Ablation Summary

All runs use fold3, seed 42, checkpoint selection by validation AUC-ROC. Metrics are clip-level mean-window aggregation.

| Run | Stream | Best epoch | Val AUC | Test AUC | BA@0.5 | F1@0.5 | Macro@0.5 | Cal thr | Cal BA | Cal F1 | Test CM@0.5 | Cal CM | Score mean/std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| full_auc | all | 1 | 53.48 | 57.93 | 51.17 | 18.56 | 40.68 | 0.4895 | 52.58 | 35.69 | `[[200, 19], [218, 27]]` | `[[174, 45], [182, 63]]` | 0.4813/0.0130 |
| fusion_residual | all_residual | 1 | 54.34 | 55.58 | 50.00 | 0.00 | 32.06 | 0.4819 | 50.87 | 20.13 | `[[219, 0], [245, 0]]` | `[[196, 23], [215, 30]]` | 0.4672/0.0116 |
| spatial_only | spatial | 2 | 58.41 | 55.80 | 51.46 | 60.04 | 50.55 | 0.5299 | 54.06 | 30.63 | `[[77, 142], [79, 166]]` | `[[193, 26], [196, 49]]` | 0.5096/0.0186 |
| flow_only | flow | 2 | 56.34 | 57.01 | 50.00 | 69.11 | 34.56 | 0.5199 | 54.21 | 62.72 | `[[0, 219], [0, 245]]` | `[[81, 138], [70, 175]]` | 0.5207/0.0017 |
| audio_only | audio | 28 | 69.77 | 56.66 | 50.00 | 69.11 | 34.56 | 0.5380 | 53.91 | 46.04 | `[[0, 219], [0, 245]]` | `[[153, 66], [152, 93]]` | 0.5340/0.0127 |

## Notes

- Audio-only has the strongest validation AUC, but its fold3 test AUC is close to spatial/flow and below the old full model test AUC.
- At threshold 0.5, flow/audio collapse to all-lie on test. Calibrated thresholds recover non-trivial confusion matrices but only modest balanced accuracy.
- Spatial-only has slightly lower test AUC than flow/audio, but less severe threshold collapse at 0.5.
- Feature norms for single streams are non-zero at best epochs; the observed issue is score calibration/rank transfer, not missing or zero features.

CSV: `outputs/metrics/fold3_stream_ablation/fold3_stream_ablation_summary.csv`
JSON: `outputs/metrics/fold3_stream_ablation/fold3_stream_ablation_summary.json`
