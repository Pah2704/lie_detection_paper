# Fold3 Gated-Logit Result

| Method | Val AUC | Test AUC | BA@0.5 | F1@0.5 | Macro@0.5 | Cal BA | Cal F1 | Cal Macro | CM@0.5 | Cal CM | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| full_auc_fold3 | 53.48% | 57.94% | 51.17% | 18.56% | 40.68% | 52.58% | 35.69% | - | `[[200,19],[218,27]]` | `[[174,45],[182,63]]` | old full cross-attention |
| late_fusion_best | 70.32% | 58.10% | 54.31% | 44.22% | 51.98% | 54.31% | 44.22% | 51.98% | `[[161,58],[159,86]]` | `[[161,58],[159,86]]` | prediction fusion val-BA tuned |
| gated_logits_prior_kl | 59.02% | 62.33% | 59.15% | 64.38% | 58.99% | 59.66% | 61.28% | 59.63% | `[[108, 111], [76, 169]]` | `[[129, 90], [97, 148]]` | trained gated-logit fusion |

## Decision

- Fold3 gated-logit prior-KL beats the target late-fusion baseline on both AUC and calibrated balanced accuracy.
- It is now worth running full 3-fold for this gated configuration, while keeping prediction late fusion as fallback baseline.
- The useful pieces were gate prior/KL and auxiliary stream losses; unconstrained random gate drifted toward spatial and underperformed.

Summary JSON: `outputs/metrics/dolos_three_stream_gated_logits_prior_kl/fold3_seed42_summary.json`
