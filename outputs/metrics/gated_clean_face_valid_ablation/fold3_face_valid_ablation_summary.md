# Fold3 face-valid ablation

| Run | Best ep | Val AUC | Val cal BA | Test AUC | Test BA@0.5 | Test cal BA | Test F1@0.5 | Test cal F1 |
|---|---|---|---|---|---|---|---|---|
| gated_prior_kl_clean_hard_mask | 2 | 63.86 | 64.74 | 57.00 | 55.45 | 55.79 | 56.24 | 57.38 |
| gated_prior_kl_clean_soft_face_valid | 2 | 64.67 | 62.18 | 60.57 | 55.66 | 56.22 | 38.97 | 61.12 |
| gated_prior_kl_clean_no_visual_mask | 2 | 58.48 | 59.29 | 59.37 | 51.85 | 55.42 | 69.81 | 65.52 |
| gated_prior_kl_mixed_original | 3 | 59.02 | 58.33 | 62.33 | 59.15 | 59.66 | 64.38 | 61.28 |

## Takeaway

Soft face-valid weighting is the best clean-cache fold3 variant among the two new ablations: it raises test AUC from 56.99 to 60.57 and calibrated BA from 55.79 to 56.22 versus the hard clean mask.
No visual mask improves AUC over hard mask but lowers calibrated BA, so it is not the preferred direction for the main protocol.
The original mixed-cache run still has higher fold3 calibrated BA, but it should not be used for the clean main table because it mixes preprocessing protocols.
