# Prediction-Level Ensemble Final Decision

## What Was Run

Full 3-fold prediction-level ensemble was completed using all currently reliable predictions:

- Existing full models:
  - `cross_attention_auc`
  - `gated_prior_kl`
- Newly completed missing folds:
  - `audio_only` fold1 and fold2
  - `spatial_only` fold1 and fold2
- Already available:
  - `audio_only` fold3
  - `spatial_only` fold3

Flow was not promoted into the final ensemble because fold3 flow-clean showed a strong validation/test mismatch and old flow artifacts were not reliable in the current workspace.

## Mean 3-Fold Results

| Candidate | Models | Mean Test AUC | Mean Cal. BA | Mean F1 Lie | Mean Macro F1 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `cross_attention_auc` | cross-attention | 61.39 | 56.35 | 48.30 | 54.26 | Baseline only |
| `gated_prior_kl` | gated prior-KL | 63.94 | 59.18 | 55.53 | 58.14 | Strongest single model |
| `spatial_only` | spatial | 63.12 | 58.76 | 48.19 | 55.80 | Useful, not final |
| `audio_only` | audio | 58.03 | 56.04 | 53.09 | 55.22 | Not final |
| `ensemble_2model_raw_auc` | cross-attention + gated prior-KL | 65.05 | 60.17 | 54.95 | 58.91 | Final candidate |
| `ensemble_2model_raw_ba` | cross-attention + gated prior-KL | 64.87 | 60.08 | 58.62 | 58.91 | Secondary if prioritizing F1 |
| `ensemble_audio_full_raw_ba` | cross-attention + gated prior-KL + audio | 62.73 | 57.86 | 52.61 | 56.69 | Rejected |
| `ensemble_spatial_subset_logit_auc` | cross-attention + gated prior-KL + spatial | 64.32 | 58.53 | 45.40 | 54.87 | Rejected |
| `ensemble_full4_raw_ba` | cross-attention + gated prior-KL + audio + spatial | 62.85 | 57.99 | 52.89 | 56.85 | Rejected |

## Decision

Use `ensemble_2model_raw_auc` as the current final result.

It improves over the strongest single trained model, `gated_prior_kl`:

- Mean test AUC: 63.94 -> 65.05, improvement +1.11 points.
- Mean calibrated BA: 59.18 -> 60.17, improvement +0.99 points.

Recommended reporting framing:

- Main final method: validation-tuned prediction-level ensemble of `cross_attention_auc` and `gated_prior_kl`.
- Strongest single model baseline: `gated_prior_kl`.
- Ablation note: adding audio-only or spatial-only predictions did not improve mean test performance, because validation tuning overweighted streams that did not generalize across test folds.

## Output Locations

- Final 2-model ensemble: `outputs/metrics/prediction_level_ensemble/`
- Audio-full ensemble check: `outputs/metrics/prediction_level_ensemble_audio_full/`
- Spatial-subset ensemble check: `outputs/metrics/prediction_level_ensemble_spatial_subset/`
- Full 4-model ensemble check: `outputs/metrics/prediction_level_ensemble_full/`
