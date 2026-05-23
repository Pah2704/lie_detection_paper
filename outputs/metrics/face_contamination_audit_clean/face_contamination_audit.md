# Face Contamination Audit

Predictions: `outputs/metrics/retrain_clean_dolos_three_stream_gated_logits_prior_kl`
Faces: `data/processed/faces_224_clean/dolos`
Embedding model: `LaurenGurgiolo/vit-micro-facial-expressions`
Distance threshold: `0.35`; min dominant ratio: `0.75`

Important: this is a heuristic audit from cached face crops. It flags likely identity switches by embedding distance/clustering, but it is not manually verified identity ground truth.

## Counts

- Clips audited: 1430
- Windows audited: 7135
- Suspect clips, any window: 1250 (87.41%)
- Suspect clips, window 0: 996 (69.65%)
- Suspect windows: 4214 (59.06%)
- Suspect window 0 rows: 996 (69.65%)

## Error Rate By Contamination

### Clip Any Suspect Window

| clip_suspect_contamination | n | default_error_rate | calibrated_error_rate | mean_score_lie |
| --- | --- | --- | --- | --- |
| False | 180 | 0.3556 | 0.4000 | 0.4968 |
| True | 1250 | 0.4192 | 0.4312 | 0.4949 |

### Clip First Window Suspect

| first_window_suspect | n | default_error_rate | calibrated_error_rate | mean_score_lie |
| --- | --- | --- | --- | --- |
| False | 434 | 0.4032 | 0.4355 | 0.4974 |
| True | 996 | 0.4147 | 0.4237 | 0.4941 |

### Window-Level Suspect

| suspect_contamination | n | default_error_rate | calibrated_error_rate | mean_score_lie |
| --- | --- | --- | --- | --- |
| False | 2921 | 0.4053 | 0.4036 | 0.4966 |
| True | 4214 | 0.4411 | 0.4528 | 0.4930 |

## Outputs

- `clip_predictions_with_face_audit.csv`
- `window_predictions_with_face_audit.csv`
- `top_suspect_clips.csv`
- `top_suspect_window0.csv`
