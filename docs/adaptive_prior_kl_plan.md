# Adaptive Prior-KL Plan

## Motivation

The fixed prior `[static=0.10, flow=0.45, audio=0.45]` encodes a strong design bias against the static stream. The Fold1-Fold2 single-stream ablation shows that the static stream is competitive, so a fixed low static prior can underuse useful facial evidence when faces are reliable.

## Method

Use a face-valid-conditioned prior for gated logit fusion:

- Low face validity prior: `[static=0.05, flow=0.45, audio=0.50]`
- High face validity prior: `[static=0.50, flow=0.25, audio=0.25]`
- Interpolation: `prior(x) = prior_low + face_valid(x) * (prior_high - prior_low)`

The prior is fixed by configuration, but the gate remains learned per sample. The KL term regularizes the learned gate toward the sample-specific prior:

`loss = CE + aux_loss + lambda * KL(gate_weights || adaptive_prior(face_valid))`

Unlike fixed Prior-KL, the adaptive KL should not be weighted by `face_valid`, because low-face-valid samples also need a low-static prior.

## Pilot Run

Start with a 3-fold, 1-seed pilot:

```bash
FOLDS="fold1 fold2 fold3" SEEDS="42" scripts/run_adaptive_prior_ablation.sh
```

If the pilot improves over both `Gated neutral no-prior` and fixed `Gated Prior-KL` on AUC/Cal. BA/Cal. F1 Lie, expand to full robustness:

```bash
FOLDS="fold1 fold2 fold3" SEEDS="42 123 2025" scripts/run_adaptive_prior_ablation.sh
```

## Pilot Result, Seed 42

Current adaptive configuration:

- Gate initialization: `[0.333333, 0.333333, 0.333334]`
- KL weight: `0.1`
- Low face prior: `[0.05, 0.45, 0.50]`
- High face prior: `[0.50, 0.25, 0.25]`

| Run | Mean AUC | Mean BA@0.5 | Mean Cal. BA | Mean Cal. F1 Lie | Mean Cal. Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adaptive Prior-KL | 0.6248 | 0.5644 | 0.5909 | 0.5587 | 0.5828 |
| Fixed Prior-KL | 0.6472 | 0.6012 | 0.5910 | 0.6276 | 0.5885 |
| Neutral no-prior | 0.6323 | 0.5735 | 0.5851 | 0.5744 | 0.5670 |
| Prior-init no-prior | 0.6794 | 0.6393 | 0.6265 | 0.6371 | 0.6167 |

The current adaptive prior should not be expanded to 3 seeds yet. It is roughly tied with fixed Prior-KL on calibrated balanced accuracy, but it is worse on AUC and calibrated F1 Lie. The strongest seed-42 run is `Prior-init no-prior`, which keeps the prior only as gate initialization and removes the KL regularizer.

Face-valid distribution also limits the current adaptive mechanism:

| Split group | Mean face_valid_ratio range | frac(face_valid_ratio == 1.0) |
| --- | ---: | ---: |
| Train folds | 0.968-0.972 | 0.881-0.885 |
| Val folds | 0.942-0.988 | 0.845-0.944 |
| Test folds | 0.962-0.981 | 0.874-0.893 |

Most windows have fully valid faces, so `face_valid` does not provide enough low-quality variation. In practice, this adaptive prior behaves close to a high-static fixed prior for most samples.

Recommended next step:

- Do not claim current Adaptive Prior-KL as the main contribution.
- Use `Prior-init no-prior` and `Neutral no-prior` to separate the effect of gate initialization from KL regularization.
- If adaptive regularization remains desired, use a richer reliability score than binary/window face validity, for example detection confidence, landmark stability, face crop size, blur, occlusion, or per-stream validation uncertainty.

## Paper Claim

If results support it, describe this as `face-valid-conditioned Prior-KL`, a less arbitrary alternative to a fixed domain prior. The claim should be framed as adaptive regularization rather than learning the prior itself.
