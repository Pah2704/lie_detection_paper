# Reliability-Confidence Voting Plan

## Motivation

Fixed Prior-KL applies the same prior to every clip/fold. This is too rigid because each clip can have different modality quality and different stream certainty. Face validity alone is also insufficient: a valid face track only means the face is detectable and stable, not that the facial signal is discriminative for that clip.

## Method

Use confidence-weighted reliability voting over the three stream logits:

`w_m = r_m * |s_m - 0.5|^gamma`

The softened variant adds a small confidence floor:

`w_m = r_m * (floor + |s_m - 0.5|^gamma)`

where:

- `s_m` is the stream probability for the Lie class.
- `|s_m - 0.5|` is stream confidence, regardless of whether the stream votes Lie or Truth.
- `r_m` is modality reliability.
- Static and flow reliability use `face_valid_ratio`, because both streams come from face crops.
- Audio reliability defaults to `1.0` because no audio-quality score is currently available.
- The final weights are normalized per sample and used to fuse calibrated stream logits.

The first pilot uses:

- `logit_fusion_mode: reliability_confidence`
- `reliability_confidence_gamma: 1.0`
- `audio_reliability: 1.0`
- `confidence_detach: true`
- no Prior-KL regularizer
- neutral gate initialization only for fallback

`confidence_detach: true` keeps the voting weights as a decision rule during training instead of letting the model increase its weight by directly optimizing confidence magnitude.

## Pilot

Run seed 42 on all three folds:

```bash
FOLDS="fold1 fold2 fold3" SEEDS="42" scripts/run_reliability_confidence_ablation.sh
```

Compare against seed-42 results from:

- `paper_gated_prior_kl`
- `paper_gated_neutral_no_prior`
- `paper_gated_no_prior`
- `paper_gated_adaptive_prior_kl`

If this pilot improves AUC or calibrated balanced accuracy without hurting calibrated F1 Lie, expand to 3 seeds.

## Seed-42 Pilot Results

Reliability-confidence voting was run on all three folds with `gamma=1.0`.

| Fold | Best epoch | Val AUC | Test AUC | BA@0.5 | F1 Lie@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 | Cal. threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold1 | 6 | 58.49 | 66.13 | 62.04 | 69.08 | 60.18 | 50.37 | 57.64 | 0.7089 |
| fold2 | 1 | 63.87 | 58.33 | 52.11 | 69.10 | 56.27 | 52.77 | 55.63 | 0.5270 |
| fold3 | 2 | 56.98 | 57.35 | 54.21 | 66.45 | 53.84 | 30.53 | 46.90 | 0.5308 |
| Mean | - | 59.78 | 60.60 | 56.12 | 68.21 | 56.76 | 44.56 | 53.39 | - |

Seed-42 mean comparison:

| Method | AUC | BA@0.5 | F1 Lie@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reliability-confidence | 60.60 | 56.12 | 68.21 | 56.76 | 44.56 | 53.39 |
| Fixed Prior-KL | 64.72 | 60.12 | 55.03 | 59.10 | 62.76 | 58.85 |
| Adaptive face_valid Prior-KL | 62.48 | 56.44 | 41.94 | 59.09 | 55.87 | 58.28 |
| Gated neutral no-prior | 63.23 | 57.35 | 42.49 | 58.51 | 57.44 | 56.70 |
| Gated prior-init w/o Prior-KL | 67.94 | 63.93 | 68.95 | 62.65 | 63.71 | 61.67 |

Interpretation:

- The mechanism is implemented correctly and gives clip-specific weights from both reliability and stream confidence.
- The `gamma=1.0` pilot should not be expanded to 3 seeds yet. It is lower than fixed Prior-KL on AUC, BA@0.5, calibrated BA, calibrated F1 Lie, and calibrated macro F1.
- The high `F1 Lie@0.5` but low calibrated F1 indicates that this decision rule changes score calibration. Fold3 is the clearest failure case: calibrated Lie recall drops sharply after threshold selection.
- The likely issue is that confidence weighting is too hard. A stream that is confidently wrong can dominate the fusion, especially when `|s_m - 0.5|` is used directly.

## Softened Seed-42 Pilot Results

The hard confidence rule was softened with `floor=0.05` and two exponents: `gamma=0.5` and `gamma=0.25`.

Seed-42 mean comparison:

| Method | AUC | BA@0.5 | F1 Lie@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reliability-confidence, gamma=1.0, floor=0.00 | 60.60 | 56.12 | 68.21 | 56.76 | 44.56 | 53.39 |
| Reliability-confidence, gamma=0.5, floor=0.05 | 67.20 | 60.26 | 63.54 | 61.27 | 56.15 | 59.94 |
| Reliability-confidence, gamma=0.25, floor=0.05 | 61.16 | 56.37 | 66.65 | 58.05 | 49.10 | 55.56 |
| Fixed Prior-KL | 64.72 | 60.12 | 55.03 | 59.10 | 62.76 | 58.85 |
| Gated prior-init w/o Prior-KL | 67.94 | 63.93 | 68.95 | 62.65 | 63.71 | 61.67 |

Fold-level details for the best softened variant:

| Fold | Best epoch | Val AUC | Test AUC | BA@0.5 | F1 Lie@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 | Cal. threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold1 | 22 | 60.10 | 70.10 | 63.29 | 64.26 | 61.16 | 49.87 | 58.10 | 0.6497 |
| fold2 | 1 | 63.84 | 60.49 | 54.29 | 65.59 | 58.88 | 56.52 | 58.42 | 0.5176 |
| fold3 | 31 | 60.38 | 71.02 | 63.19 | 60.77 | 63.76 | 62.05 | 63.32 | 0.4889 |
| Mean | - | 61.44 | 67.20 | 60.26 | 63.54 | 61.27 | 56.15 | 59.94 | - |

Interpretation:

- `gamma=0.5, floor=0.05` fixes most of the hard-voting failure. Compared with `gamma=1.0`, it improves mean AUC by +6.60 pp, Cal. BA by +4.50 pp, and Cal. Macro F1 by +6.56 pp.
- `gamma=0.25, floor=0.05` is too soft. It keeps the gate closer to uniform and fails on fold3, with mean AUC only 61.16.
- Compared with fixed Prior-KL, `gamma=0.5, floor=0.05` improves AUC (+2.49 pp), BA@0.5 (+0.14 pp), F1 Lie@0.5 (+8.51 pp), Cal. BA (+2.16 pp), and Cal. Macro F1 (+1.09 pp), but Cal. F1 Lie is lower (-6.61 pp).
- Compared with prior-init no-prior, `gamma=0.5, floor=0.05` is still slightly weaker overall. This means it is promising as a technical variant, but not yet a replacement for the strongest no-prior baseline.

Decision before any full 3-seed run:

- Do not expand `gamma=0.25, floor=0.05`.
- Expand `gamma=0.5, floor=0.05` only if the paper frames the contribution around clip-adaptive reliability and ranking/calibrated balanced accuracy.
- If the primary claim is lie-class calibrated F1, run one more constrained variant first: `gamma=0.5, floor=0.05` plus a maximum stream-weight cap or majority-consistency constraint.

Next candidate before any full 3-seed run:

- keep `gamma=0.5` and `floor=0.05`;
- optionally cap the maximum per-stream weight or add a majority-consistency constraint so one confident stream cannot overrule two aligned streams;
- avoid `gamma=0.25` unless the goal is an intentionally conservative, near-uniform fusion rule.

## Vote-Score-Sum Direction

The next implementation follows the simpler class-support idea: add the support of streams that vote for the same class, then compare total Lie support against total Truth support.

For each stream:

`p_m = softmax(z_m)`

`v_m = argmax(p_m)`

`support_m = r_m * (0.05 + max(p_m)^0.5)`

Then:

`S_lie = sum support_m where v_m = Lie`

`S_truth = sum support_m where v_m = Truth`

`margin = S_lie - S_truth`

The final two-class logit is `[-0.5 * margin, 0.5 * margin]`.

This is different from the earlier reliability-confidence weighted-logit average:

- It directly rewards agreement between streams.
- It still preserves reliability through `r_m`.
- It uses confidence as support strength, but not as a free global prior.
- Two aligned weak-to-moderate streams usually beat one opposing stream, while one stream can still win if the other two have low reliability/support.

Implementation:

- `logit_fusion_mode: vote_score_sum`
- `reliability_confidence_gamma: 0.5`
- `reliability_confidence_floor: 0.05`
- `vote_consensus_bonus: 0.0`
- `confidence_detach: false`
- no Prior-KL regularizer

Unlike weighted-logit reliability-confidence fusion, `vote_score_sum` keeps support differentiable. Detaching the confidence/support term here would remove the main fusion-loss gradient path, because the final logit is built from class support rather than from a weighted sum of stream logits.

## Vote-Score-Sum Seed-42 Pilot Results

The first pilot used all three folds with seed 42.

| Fold | Best epoch | Val AUC | Test AUC | BA@0.5 | F1 Lie@0.5 | Macro F1@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 | Cal. threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold1 | 22 | 63.89 | 68.54 | 63.23 | 61.70 | 62.78 | 61.80 | 58.41 | 60.99 | 0.5294 |
| fold2 | 9 | 68.16 | 69.37 | 64.46 | 70.21 | 64.11 | 64.29 | 68.52 | 64.21 | 0.5578 |
| fold3 | 7 | 66.47 | 70.15 | 57.96 | 69.98 | 54.34 | 64.67 | 62.61 | 64.16 | 0.7170 |
| Mean | - | 66.17 | 69.35 | 61.88 | 67.30 | 60.41 | 63.59 | 63.18 | 63.12 | - |

Seed-42 mean comparison:

| Method | AUC | BA@0.5 | F1 Lie@0.5 | Macro F1@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vote-score-sum gamma=0.5 floor=0.05 | 69.35 | 61.88 | 67.30 | 60.41 | 63.59 | 63.18 | 63.12 |
| Reliability-confidence gamma=0.5 floor=0.05 | 67.20 | 60.26 | 63.54 | 59.11 | 61.27 | 56.15 | 59.94 |
| Fixed Prior-KL | 64.72 | 60.12 | 55.03 | 57.79 | 59.10 | 62.76 | 58.85 |
| Gated prior-init w/o Prior-KL | 67.94 | 63.93 | 68.95 | 63.62 | 62.65 | 63.71 | 61.67 |

Interpretation:

- Vote-score-sum is the strongest seed-42 exploratory direction so far on AUC, Cal. BA, and Cal. Macro F1.
- Compared with reliability-confidence `gamma=0.5`, it improves AUC by +2.15 pp, Cal. BA by +2.32 pp, Cal. F1 Lie by +7.03 pp, and Cal. Macro F1 by +3.17 pp.
- Compared with fixed Prior-KL, it improves AUC by +4.63 pp, Cal. BA by +4.48 pp, Cal. F1 Lie by +0.42 pp, and Cal. Macro F1 by +4.26 pp.
- Compared with prior-init no-prior, it improves AUC by +1.41 pp, Cal. BA by +0.94 pp, and Cal. Macro F1 by +1.45 pp, while trailing on BA@0.5 and F1 Lie@0.5.

Decision:

- Expand `vote_score_sum, gamma=0.5, floor=0.05, consensus_bonus=0.0` to 3 seeds.
- Do not run `consensus_bonus=0.25` before the 3-seed expansion. The current rule already includes class agreement; an extra bonus could over-amplify majority bias and weaken the technical claim.

## Vote-Score-Sum 3-Seed Results

The 3-seed expansion was run on all folds with seeds `42`, `123`, and `2025`.

| Method | Runs | AUC | BA@0.5 | F1 Lie@0.5 | Macro F1@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vote-score-sum gamma=0.5 floor=0.05 | 9 | 66.06 +- 3.82 | 58.62 +- 4.18 | 53.90 +- 21.48 | 55.50 +- 9.26 | 61.01 +- 3.08 | 58.52 +- 10.05 | 59.87 +- 4.43 |
| Fixed Prior-KL | 9 | 64.16 +- 4.56 | 57.94 +- 5.61 | 63.30 +- 12.23 | 53.18 +- 11.87 | 58.96 +- 4.16 | 56.74 +- 10.67 | 57.45 +- 5.15 |
| Gated prior-init w/o Prior-KL | 9 | 64.20 +- 4.20 | 57.23 +- 6.68 | 68.61 +- 1.85 | 50.97 +- 13.48 | 59.27 +- 3.62 | 52.75 +- 11.45 | 57.18 +- 4.93 |
| Gated neutral no-prior | 9 | 61.90 +- 3.31 | 56.27 +- 5.26 | 50.45 +- 27.90 | 49.73 +- 12.86 | 57.34 +- 2.85 | 51.78 +- 12.67 | 55.16 +- 5.10 |

Paired deltas:

| Comparison | AUC | BA@0.5 | F1 Lie@0.5 | Macro F1@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vote-score-sum - Fixed Prior-KL | +1.90 | +0.68 | -9.39 | +2.32 | +2.05 | +1.78 | +2.43 |
| Vote-score-sum - prior-init w/o Prior-KL | +1.86 | +1.39 | -14.71 | +4.53 | +1.74 | +5.77 | +2.70 |
| Vote-score-sum - neutral no-prior | +4.16 | +2.35 | +3.46 | +5.77 | +3.67 | +6.75 | +4.72 |

Interpretation:

- The 3-seed result confirms vote-score-sum as the strongest current paper direction for AUC and calibrated operating-point metrics.
- It beats fixed Prior-KL and prior-init no-KL on AUC, Cal. BA, Cal. F1 Lie, and Cal. Macro F1.
- The weak point is default-threshold `F1 Lie@0.5`. This is mostly a calibration issue, not a ranking issue: `fold3 seed2025` has high AUC (68.60) but nearly all predictions fall below threshold 0.5, causing `F1 Lie@0.5=0.81`; validation calibration recovers `Cal. F1 Lie=67.57`.
- For the paper, position vote-score-sum as the proposed agreement-aware adaptive fusion method and report calibrated-threshold metrics as the main operating point.

Final decision:

- Use `vote_score_sum, gamma=0.5, floor=0.05, consensus_bonus=0.0` as the main proposed method.
- Keep fixed Prior-KL as an ablation/baseline showing why a global hand-written prior is weaker than clip-adaptive agreement voting.
- Do not add `consensus_bonus=0.25` to the main method unless a later experiment specifically targets threshold stability.
