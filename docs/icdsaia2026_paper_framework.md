# ICDSAIA 2026 Paper Framework

Working title:

**Validation-Tuned Multimodal Fusion for Audio-Visual Deception Detection on DOLOS**

Alternative titles:

1. **A Clean Multimodal Pipeline for Audio-Visual Deception Detection on DOLOS**
2. **Robust Prediction-Level Fusion for DOLOS Deception Detection with Face-Validity Aware Modeling**
3. **Audio-Visual Deception Detection with Temporal Face Masking and Validation-Tuned Fusion**

Recommended target venue framing:

- Venue: ICDSAIA 2026, International Conference on Data Science, Artificial Intelligence and Applications.
- Fit: Computer Vision and Image Processing; Pattern Recognition and Pattern Analysis; Machine Learning and Deep Learning; Emerging and Innovative Applications of AI and Data Science.
- Format: Springer LNCS/CCIS style, single column.
- Limit: maximum 15 pages including references at submission.
- Review: double blind, so remove author names, affiliations, self-identifying repository links, and acknowledgements from the submitted version.
- Submission deadline currently shown by ICDSAIA 2026: 15 June 2026.

Main claim to use:

> We present a reproducible DOLOS-only multimodal pipeline for audio-visual deception detection that combines clean face tracking, face-validity-aware temporal modeling, and validation-tuned prediction-level fusion. On the official 3-fold DOLOS protocol, the best ensemble reaches 65.43% mean AUC, 60.54% calibrated balanced accuracy, and 61.39% lie-class F1, improving over its strongest single-model component while remaining transparent about threshold-level limitations against the original DOLOS benchmark.

Revised main claim after switching the paper focus to Gated Prior-KL:

> We present a reproducible DOLOS-only multimodal pipeline for audio-visual deception detection centered on a gated logit fusion model with face-validity-aware temporal masking and Prior-KL regularization. On the official 3-fold DOLOS protocol, the Gated Prior-KL model reaches 64.38% mean AUC, 59.39% calibrated balanced accuracy, and 60.27% lie-class F1. The paper will use single-stream ablations, a no-prior gated baseline, and three-seed robustness analysis to isolate the contribution of the Prior-KL regularizer.

Final experiment source:

- Use `outputs/metrics/final_report_clean_temporal_mask_soft_cross/` as the authoritative result folder for the ICDSAIA 2026 paper.
- Main method for the paper: `gated_prior_kl`.
- Secondary result: `ensemble_raw_balanced_accuracy`, used only as an optional ensemble analysis.
- Main 3-fold mean metrics: AUC 64.38, calibrated balanced accuracy 59.39, calibrated F1 Lie 60.27, calibrated macro F1 59.26.
- Secondary ensemble metrics: AUC 65.43, calibrated balanced accuracy 60.54, calibrated F1 Lie 61.39, calibrated macro F1 60.13.

Important caution:

- Do not claim the method is globally better than the original DOLOS paper.
- The defensible comparison is: AUC is comparable/slightly higher than the reported PAVF + multi-task AUC, while accuracy/F1 remain lower than the original DOLOS method.

---

## Proposed Paper Skeleton

### Title Page

Use only the title and abstract in the double-blind submission. Do not include author names or affiliations.

**Title**

Validation-Tuned Multimodal Fusion for Audio-Visual Deception Detection on DOLOS

**Short running title**

Multimodal Fusion for DOLOS Deception Detection

---

## Abstract

Audio-visual deception detection is challenging because behavioral cues associated with truthful and deceptive statements are weak, sparse, and highly sensitive to speaker identity, scene composition, and signal quality. This paper presents a DOLOS-only multimodal pipeline for video-level truth/lie classification using three complementary streams: facial appearance, facial motion, and speech audio. The pipeline applies clean face preprocessing, optical-flow extraction, face-validity masking, temporal aggregation, and gated logit fusion with Prior-KL regularization. The main model combines stream logits while using a weak prior to discourage unstable modality weighting under noisy visual evidence. Experiments follow the official 3-fold DOLOS protocol. The Gated Prior-KL model reaches 64.38% mean AUC, 59.39% calibrated balanced accuracy, and 60.27% lie-class F1. The paper further evaluates single-stream ablations, an otherwise identical no-prior gated baseline, and three-seed robustness. Error analysis shows that performance remains sensitive to host, episode, and suspected face-track contamination, highlighting the importance of protocol-level reporting and group-aware diagnostics in deception detection.

**Keywords**

Audio-visual deception detection; DOLOS; multimodal learning; micro-expression; face validity; prediction-level fusion; pattern recognition.

---

## 1. Introduction

### 1.1 Motivation

Write 2-3 paragraphs:

- Deception detection from video is a difficult pattern recognition problem because the target concept is not directly observable.
- Facial expressions, micro-expressions, head motion, and speech prosody may contain useful cues, but each cue is noisy and context-dependent.
- Real-world video introduces additional instability: multiple faces, camera cuts, pose changes, lighting variation, background laughter, and host/speaker bias.
- Therefore, the problem is well suited to multimodal AI but requires careful evaluation and cautious interpretation.

Suggested paragraph:

> Detecting deception from short video clips is substantially different from recognizing visible objects or explicit actions. The model must infer a latent behavioral state from subtle, temporally sparse, and often ambiguous cues. Facial appearance can capture brief affective changes, optical flow can describe local motion around the face and head, and speech audio can reflect prosody or hesitation. However, each modality is unreliable in isolation: facial crops may follow the wrong person, motion may reflect camera edits rather than facial behavior, and audio may encode speaker or content bias rather than deception. This motivates a controlled multimodal framework that explicitly accounts for signal quality and validates fusion decisions on held-out data.

### 1.2 Research Gap

Emphasize:

- Existing DOLOS work reports strong audio-visual results with parameter-efficient crossmodal learning and multi-task supervision.
- A practical follow-up question remains: how far can a reproducible pipeline go using directly extracted face, motion, and audio signals without relying on additional task labels?
- Many works report mean metrics but provide limited group/error diagnostics.
- DOLOS contains internal variation by host and episode, so error analysis is important.

### 1.3 Contributions

Use exactly 3-4 contributions:

1. We build a DOLOS-only three-stream pipeline combining facial appearance, facial optical flow, and speech audio under the official 3-fold protocol.
2. We introduce face-validity-aware temporal modeling to reduce the impact of unstable or contaminated facial tracks during video-level prediction.
3. We compare Gated Prior-KL against an otherwise identical no-prior gated baseline and single-stream static, flow, and audio ablations.
4. We provide random-seed and group-level diagnostics by fold, host, episode, and suspected face-track contamination to characterize residual failure modes.

### 1.4 Paper Organization

Short paragraph:

> Section 2 reviews audio-visual deception detection and multimodal fusion. Section 3 describes the DOLOS dataset and evaluation protocol. Section 4 presents the preprocessing pipeline and proposed fusion models. Section 5 reports experimental settings. Section 6 presents quantitative results and error analysis. Section 7 discusses limitations and ethical considerations, and Section 8 concludes the paper.

---

## 2. Related Work

### 2.1 Deception Detection and Micro-Expression Analysis

Points to cover:

- Deception cues are indirect and noisy.
- Micro-expressions can reveal affective leakage, but they are not deterministic indicators of lying.
- In unconstrained video, face quality and temporal alignment matter.

References to include:

- Micro-expression survey or classic micro-expression paper.
- Facial action coding / expression recognition reference if needed.

### 2.2 Audio-Visual Learning for Deception Detection

Points to cover:

- Audio cues: prosody, pitch, pauses, stress, rhythm.
- Visual cues: facial expression, head pose, eye/mouth movement, temporal dynamics.
- Multimodal fusion can be early, intermediate, late, attention-based, or ensemble-based.

### 2.3 The DOLOS Dataset and Benchmark

Points to cover:

- DOLOS provides audio-visual deception clips from "Would I Lie To You?"
- It supports official 3-fold evaluation.
- The original DOLOS paper reports Visual, Audio, Concatenation, PAVF, and PAVF + multi-task baselines.

Use this comparison table later in Results, not necessarily in Related Work:

| Source | Method | ACC | F1 Lie | AUC |
| --- | ---: | ---: | ---: | ---: |
| DOLOS paper | Visual | 61.44 | 69.42 | 58.89 |
| DOLOS paper | Audio | 59.19 | 73.46 | 52.54 |
| DOLOS paper | Concatenation | 61.62 | 70.20 | 60.50 |
| DOLOS paper | PAVF | 64.75 | 71.20 | 62.71 |
| DOLOS paper | PAVF + Multi-task | 66.84 | 73.35 | 64.58 |

### 2.4 Multimodal Fusion and Calibration

Points to cover:

- Cross-attention learns interactions across modalities.
- Gating controls modality contribution and is useful when signal quality varies.
- Prediction-level ensembling can be robust when individual models specialize differently across folds.
- Threshold calibration matters when class balance and score distributions differ by fold.

---

## 3. Dataset and Evaluation Protocol

### 3.1 Dataset

Write:

- Dataset: DOLOS.
- Task: binary video-level classification, truth vs lie.
- Unit: short clip.
- Modalities used: face RGB, optical flow from face crops, audio waveform.
- Splits: official 3-fold train/validation/test.

Table to include:

**Table 1. DOLOS 3-fold split statistics.**

| Fold | Split | Clips | Truth | Lie | Hosts | Episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fold1 | train | 857 | 411 | 446 | 5 | 59 |
| fold1 | val | 88 | 36 | 52 | 4 | 8 |
| fold1 | test | 484 | 227 | 257 | 5 | 67 |
| fold2 | train | 832 | 400 | 432 | 5 | 60 |
| fold2 | val | 115 | 45 | 70 | 5 | 7 |
| fold2 | test | 482 | 229 | 253 | 5 | 66 |
| fold3 | train | 835 | 403 | 432 | 5 | 58 |
| fold3 | val | 130 | 52 | 78 | 4 | 9 |
| fold3 | test | 464 | 219 | 245 | 5 | 66 |

Recommended figure:

- `docs/figures/report/graph01_dolos_label_distribution.png`

### 3.2 Evaluation Protocol

Write:

- Train model parameters on train split.
- Select checkpoints using validation AUC.
- Search fusion weights on validation split only.
- Calibrate threshold on validation split only.
- Report metrics on test split.
- Average metrics over three folds.

Metrics:

- AUC-ROC: main ranking metric.
- Balanced accuracy: main threshold metric due to class imbalance.
- Lie-class F1: deception-class performance.
- Macro F1: threshold-level aggregate.
- Confusion matrix: interpretability.

### 3.3 Data Usage and Ethics

Write:

- DOLOS is used for academic/non-commercial research.
- Raw videos, annotations, face crops, audio files, optical flow, and derived datasets are not redistributed.
- The model should not be presented as a forensic lie detector.
- Predictions are probabilistic pattern-recognition outputs and should not be used for high-stakes decisions without domain validation.

---

## 4. Method

### 4.1 Overview

Recommended figure:

- `docs/figures/report/fig01_problem_pipeline.png`
- `docs/figures/report/fig04_preprocessing_pipeline.png`
- `docs/figures/report/fig08_three_stream_architecture.png`

Suggested text:

> Given a video clip, the pipeline extracts three synchronized representations: facial appearance frames, facial optical flow, and audio waveform segments. Each clip is divided into fixed-length temporal windows. The model produces window-level deception scores, which are aggregated into clip-level scores. Fusion is performed either inside the network through cross-attention or gated logit combination, or after inference through validation-tuned prediction-level ensembling.

### 4.2 Preprocessing

#### 4.2.1 Face Cropping and Face Validity

Content:

- Use MediaPipe-based face detection/cropping.
- Resize face crops to 224x224.
- Apply clean face track selection to reduce wrong-person crops.
- Compute `face_valid` masks.
- Use `face_valid` during temporal modeling and/or gating.

Suggested concise formula:

Let \(x_t^v\) be the visual feature at time \(t\), and \(m_t \in \{0,1\}\) be the face-validity indicator. Temporal pooling uses masked attention:

\[
\alpha_t = \frac{\exp(q^\top x_t^v) m_t}{\sum_j \exp(q^\top x_j^v) m_j + \epsilon}.
\]

#### 4.2.2 Optical Flow

Content:

- Compute flow on clean face crops.
- Use 2-channel flow input.
- Extract motion features with a modified ResNet18.
- Missing flow uses zero fallback and validity-aware handling.

#### 4.2.3 Audio

Content:

- Extract mono 16 kHz waveform.
- Use Wav2Vec2-base as audio feature extractor.
- Temporal pooling compresses audio sequence into tokens aligned with visual windows.

### 4.3 Three-Stream Feature Encoders

Table:

**Table 2. Stream encoders.**

| Stream | Input | Backbone | Output | Role |
| --- | --- | --- | ---: | --- |
| Spatial | 224x224 face RGB | ViT facial-expression model | 256 | appearance/expression |
| Flow | 2-channel optical flow | ResNet18 | 256 | facial/head motion |
| Audio | 16 kHz waveform | Wav2Vec2-base | 256 | speech/prosody |

### 4.4 Cross-Attention Fusion

Content:

- Define visual tokens and audio tokens.
- Use cross-attention to model audio-visual interaction.
- Include soft temporal penalty to prefer temporally nearby interactions while allowing long-range links.

Suggested formula:

\[
A_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} - \lambda |t_i - t_j|,
\quad
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}(A)V.
\]

Explain:

- \(t_i\) and \(t_j\) are token timestamps.
- \(\lambda\) controls temporal penalty.
- This discourages unrealistic cross-time matching without hard blocking cross-modal context.

Recommended figure:

- `docs/figures/report/fig09_cross_attention_block.png`

### 4.5 Gated Logit Fusion with Prior Regularization

Content:

- Each stream/model produces logits.
- A learned or validation-conditioned gate combines logits.
- Prior-KL regularization prevents unstable over-reliance on a noisy modality.

Suggested formula:

\[
z = \sum_{k=1}^{K} g_k z_k,
\quad
g=\mathrm{softmax}(h),
\quad
\mathcal{L} = \mathcal{L}_{CE}(y,z) + \beta D_{KL}(g || p).
\]

Where:

- \(z_k\): modality or model logit.
- \(g_k\): learned gate weight.
- \(p\): prior distribution over modalities.

Recommended figure:

- `docs/figures/report/fig10_gated_logit_fusion.png`

### 4.6 Prediction-Level Ensemble

Content:

- The final method combines the clip-level scores of cross-attention and gated prior-KL models.
- Weights are searched on validation folds.
- Thresholds are calibrated on validation folds.
- Test is untouched until final reporting.

Formula:

\[
s = w s_{cross} + (1-w) s_{gated},
\quad
w^\* = \arg\max_{w \in [0,1]} M_{val}(w).
\]

For the secondary ensemble result, \(M_{val}\) is balanced accuracy for `ensemble_raw_balanced_accuracy`. This ensemble is not the main method after the paper is reframed around Gated Prior-KL.

Recommended figure:

- `docs/figures/report/fig11_prediction_level_ensemble.png`

---

## 5. Experiments

### 5.1 Implementation Details

Content:

- Framework: PyTorch.
- Face preprocessing: MediaPipe.
- Spatial backbone: pretrained ViT facial-expression model.
- Audio backbone: Wav2Vec2-base.
- Flow backbone: ResNet18 with 2-channel first convolution.
- Window length: 2 seconds.
- Main evaluation: clip-level predictions.

Table:

**Table 3. Training and evaluation settings.**

| Setting | Value |
| --- | --- |
| Protocol | DOLOS official 3-fold |
| Window length | 2 s |
| Main checkpoint metric | validation AUC |
| Threshold calibration | validation balanced accuracy |
| Ensemble search | validation grid search |
| Test metric aggregation | mean over 3 folds |

### 5.2 Baselines and Variants

Include:

1. Cross-attention AUC baseline.
2. Gated logits prior-KL model.
3. Final prediction-level ensemble raw-AUC.
4. Final prediction-level ensemble raw-BA.
5. Optional ablations: audio only, spatial only, flow only, soft face-validity, temporal mask, clean face protocol.

### 5.3 Main Evaluation Metrics

Define:

- AUC-ROC.
- Accuracy.
- Balanced accuracy:

\[
BA = \frac{1}{2}\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right).
\]

- Lie-class F1.
- Macro F1.

Explain why AUC and BA matter:

- AUC measures ranking quality across thresholds.
- BA reduces bias under uneven truth/lie proportions.
- Lie-class F1 measures detection quality for the target deception class.

---

## 6. Results and Analysis

### 6.1 Main Results

Use this as the current main results table. The authoritative source is `outputs/metrics/final_report_clean_temporal_mask_soft_cross/final_results_summary.md`, with Gated Prior-KL treated as the main method and ensemble rows treated as secondary.

**Table 4. Mean 3-fold DOLOS results.**

| Method | AUC | BA@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-attention AUC baseline | 61.98 | 53.31 | 57.22 | 48.36 | 52.30 |
| Gated logits Prior-KL | **64.38** | **55.71** | **59.39** | **60.27** | **59.26** |
| Final ensemble raw-AUC | 65.20 | 54.25 | 60.00 | 59.06 | 59.53 |
| Final ensemble raw-BA | **65.43** | 54.90 | **60.54** | **61.39** | **60.13** |

Suggested result paragraph:

> The Gated Prior-KL model is used as the main method, reaching 64.38% AUC, 59.39% calibrated balanced accuracy, 60.27% lie-class F1, and 59.26% macro F1. The validation-tuned ensemble is reported only as a secondary analysis: it improves AUC to 65.43% and calibrated balanced accuracy to 60.54%, indicating that cross-attention and gated fusion provide complementary predictions. The core contribution, however, is evaluated through no-prior and single-stream ablations around the gated model.

Recommended figure:

- `docs/figures/report/graph02_dolos_method_comparison.png`

### 6.2 Per-Fold Results

Table:

**Table 5. Main Gated Prior-KL per-fold results.**

| Fold | Threshold | AUC | BA@0.5 | Cal. BA | Cal. F1 Lie | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fold1 | 0.5262 | 65.51 | 60.70 | 60.02 | 63.50 | [[125,102],[90,167]] |
| fold2 | 0.5140 | 62.17 | 53.35 | 59.07 | 58.92 | [[142,87],[111,142]] |
| fold3 | 0.5460 | 65.46 | 53.09 | 59.08 | 58.39 | [[139,80],[111,134]] |

Suggested paragraph:

> Gated Prior-KL performance varies across folds, especially in threshold-level metrics. Fold 1 has the strongest calibrated F1 for the lie class, while Fold 2 shows lower AUC and balanced accuracy. This variance motivates both no-prior comparison and random-seed analysis before making strong claims about the regularizer.

Recommended figure:

- `docs/figures/report/graph03_dolos_per_fold_auc_ba.png`

### 6.3 Comparison with the DOLOS Paper

Table:

**Table 6. Comparison with published DOLOS baselines.**

| Source | Method | ACC | BA | F1 Lie | AUC | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| DOLOS paper | Visual | 61.44 | - | 69.42 | 58.89 | reported in paper |
| DOLOS paper | Audio | 59.19 | - | 73.46 | 52.54 | reported in paper |
| DOLOS paper | Concatenation | 61.62 | - | 70.20 | 60.50 | reported in paper |
| DOLOS paper | PAVF | 64.75 | - | 71.20 | 62.71 | reported in paper |
| DOLOS paper | PAVF + Multi-task | 66.84 | - | 73.35 | 64.58 | reported in paper |
| Ours | Cross-attention baseline | 56.70 | 57.22 | 48.36 | 61.98 | calibrated threshold |
| Ours | Gated logits Prior-KL | 59.36 | 59.39 | 60.27 | 64.38 | main method, calibrated threshold |
| Ours | Final ensemble raw-BA | 60.57 | 60.54 | 61.39 | 65.43 | secondary ensemble analysis |

Suggested comparison paragraph:

> The main Gated Prior-KL model reaches a mean AUC close to the PAVF + multi-task result reported in the original DOLOS paper (64.38% vs. 64.58%), while its accuracy and lie-class F1 remain lower. Therefore, the method should be interpreted as a reproducible gated-fusion study with controlled ablations, not as a complete replacement for the original multi-task DOLOS system.

Recommended figure:

- `docs/figures/report/graph04_ours_vs_paper_auc.png`

### 6.4 Ablation Study

Suggested ablation table:

**Table 7. Ablation summary.**

| Variant | AUC | Cal. BA | Cal. F1 Lie | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Cross-attention baseline | 61.98 | 57.22 | 48.36 | useful ranking, weak lie threshold |
| Gated Prior-KL | 64.38 | 59.39 | 60.27 | main model |
| Gated no-prior | TBD | TBD | TBD | direct regularizer baseline |
| Temporal mask ensemble | 65.32 | 59.61 | 58.94 | improves AUC |
| Soft-cross + temporal-mask ensemble | 65.43 | 60.54 | 61.39 | best selected setting |

Write:

- Temporal masking improves robustness when face validity is unstable.
- Soft temporal cross-attention provides complementary behavior.
- Adding extra single-stream predictions should only be retained if validation and test behavior are consistent.

Recommended figure:

- `docs/figures/report/graph05_stream_ablation.png`

### 6.5 Error Analysis by Host and Episode

Use:

- `outputs/metrics/final_report_clean_temporal_mask_soft_cross/final_error_by_host.csv`
- `outputs/metrics/final_report_clean_temporal_mask_soft_cross/final_error_by_episode.csv`
- If those are not present, use the closest final report folder and regenerate.

Suggested paragraph:

> Error analysis shows that performance is not uniformly distributed across hosts and episodes. This is expected in audio-visual deception detection because identity, speaking style, camera framing, and episode-specific editing can influence all three modalities. Reporting only mean AUC can therefore hide important group-level failure modes.

Recommended figures:

- `docs/figures/report/graph06_error_by_host.png`
- `docs/figures/report/graph07_episode_error_heatmap.png`

### 6.6 Face-Track Contamination Analysis

Use table like:

| Suspected contamination | n | BA | Accuracy | AUC | Error rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| False | fill from final artifact | fill | fill | fill | fill |
| True | fill from final artifact | fill | fill | fill | fill |

Write:

> Clips marked by the contamination heuristic are harder than clean clips, suggesting that face tracking quality materially affects the visual stream. This supports the use of face-validity masks and motivates future work on identity-consistent tracking.

---

## 7. Discussion

### 7.1 What the Results Show

Write:

- Multimodal fusion can improve DOLOS ranking performance.
- Validation-tuned prediction-level fusion is useful because different models generalize differently across folds.
- Face-validity-aware handling is important for video deception detection.

### 7.2 What the Results Do Not Show

Be explicit:

- The system is not a reliable real-world lie detector.
- The model is trained and evaluated on DOLOS only.
- It may learn dataset-specific cues such as host, speaker, episode style, or editing patterns.
- It does not prove that micro-expressions alone determine deception.

### 7.3 Practical Implications

Write:

- For research: use group/error diagnostics, not only aggregate metrics.
- For system design: maintain modality quality checks and conservative threshold calibration.
- For deployment: avoid high-stakes use without independent validation and human oversight.

---

## 8. Limitations and Future Work

Limitations:

1. DOLOS-only evaluation limits external validity.
2. Threshold metrics remain weaker than original DOLOS PAVF + multi-task results.
3. Face tracking and clip contamination affect performance.
4. Validation-tuned fold-specific weights may overfit small validation splits.
5. The method does not explicitly model speech content or transcript semantics.

Future work:

1. Evaluate cross-dataset generalization.
2. Add identity-consistent multi-face tracking.
3. Integrate transcript/text features with leakage controls.
4. Use uncertainty estimation and abstention for low-quality clips.
5. Study fairness and group robustness by speaker attributes when ethically and legally appropriate.

---

## 9. Conclusion

Draft conclusion:

> This paper presented a DOLOS-only multimodal framework for audio-visual deception detection using facial appearance, facial motion, and speech audio. The proposed pipeline combines clean face preprocessing, face-validity-aware temporal modeling, and gated logit fusion with Prior-KL regularization. Under the official 3-fold DOLOS protocol, the main Gated Prior-KL model achieved 64.38% mean AUC, 59.39% calibrated balanced accuracy, and 60.27% lie-class F1. The model reaches comparable AUC to the original DOLOS benchmark while remaining weaker on accuracy and F1. Planned no-prior, single-stream, and multi-seed ablations are used to isolate the contribution and stability of the Prior-KL regularizer. Detailed error analysis further shows that host, episode, and face-track quality remain important sources of variation.

---

## Figures and Tables Checklist

Target: 6-8 figures/tables total for a 15-page LNCS paper.

Recommended figures:

1. `fig01_problem_pipeline.png`: Problem and pipeline overview.
2. `fig04_preprocessing_pipeline.png`: Preprocessing flow.
3. `fig08_three_stream_architecture.png`: Three-stream architecture.
4. `fig11_prediction_level_ensemble.png`: Ensemble strategy.
5. `graph02_dolos_method_comparison.png`: Main results.
6. `graph03_dolos_per_fold_auc_ba.png`: Per-fold metrics.
7. `graph06_error_by_host.png`: Host-level errors.
8. `graph07_episode_error_heatmap.png`: Episode-level errors.

Recommended tables:

1. Dataset statistics.
2. Stream encoders and settings.
3. Main 3-fold results.
4. Per-fold final ensemble results.
5. Comparison with DOLOS paper.
6. Ablation/error analysis table.

If space is tight:

- Keep figures 1, 3, 5, and 6.
- Keep tables 1, 3, and 5.
- Move detailed error tables to appendix or supplementary material if allowed.

---

## Page Budget for 15-Page Springer Submission

| Section | Target pages |
| --- | ---: |
| Abstract + keywords | 0.5 |
| Introduction | 1.5 |
| Related work | 1.5 |
| Dataset/protocol | 1.5 |
| Method | 3.0 |
| Experiments | 1.5 |
| Results/error analysis | 3.0 |
| Discussion/limitations/conclusion | 1.5 |
| References | 1.0-1.5 |

---

## Reference Seed List

Add exact BibTeX before writing the LaTeX version.

Core references:

1. Guo et al., "Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient Crossmodal Learning", ICCV 2023.
2. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
3. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", NeurIPS 2020.
4. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021.
5. Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
6. MediaPipe face mesh / face detection reference.
7. A micro-expression survey or deception detection survey.
8. A multimodal fusion survey.

Venue/submission references:

- ICDSAIA 2026 Call for Papers: `https://icdsaia2026.org/authors/call-for-paper`
- ICDSAIA 2026 Paper Submission: `https://icdsaia2026.org/authors/paper-submission`
- Springer LNCS guidelines: `https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines`

---

## Double-Blind Submission Checklist

Before creating the PDF:

- Remove author names and affiliations.
- Remove acknowledgements.
- Do not include GitHub URL if it identifies the authors.
- Avoid phrases like "our previous work" unless anonymized.
- If citing own prior work, cite in third person.
- Ensure file metadata does not contain author identity.
- Do not include local paths from the machine.
- Do not include DOLOS raw frames unless licensing and fair-use constraints are checked.
- Keep the paper within 15 pages including references.
- Submit a single PDF via CMT.

---

## Finalization TODO

1. Use `outputs/metrics/final_report_clean_temporal_mask_soft_cross/` as the single source of final experimental numbers.
2. Regenerate all final figures from the same source folder if any figure was produced from an older result folder.
3. Verify that every number in the paper matches the final CSV/JSON artifacts.
4. Convert this framework to `paper/main.tex` using Springer LNCS.
5. Add BibTeX entries and compile.
6. Run a final double-blind audit.
7. Submit before 15 June 2026.
