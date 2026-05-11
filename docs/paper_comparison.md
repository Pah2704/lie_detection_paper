# Comparison With Original Real-life Trial Dataset Paper

Source:

- Pérez-Rosas, V., Abouelenien, M., Mihalcea, R., and Burzo, M. "Deception Detection using Real-life Trial Data." ICMI 2015.
- PDF: https://web.eecs.umich.edu/~mihalcea/papers/perezrosas.icmi15.pdf

## Original Paper Summary

The original paper introduced the Real-life Trial Dataset, consisting of 121 video clips collected from public court trial recordings. The paper studied deception detection using verbal and non-verbal modalities.

The original paper used:

- Verbal features from transcripts, such as unigrams and bigrams.
- Manually annotated non-verbal cues, including facial displays, hand gestures, gaze, head movement, and related behaviors.
- Decision Tree and Random Forest classifiers.
- Leave-one-out cross-validation.
- Accuracy as the primary reported metric.

Reported results from Table 3:

| Feature Set | Decision Tree Accuracy | Random Forest Accuracy |
|---|---:|---:|
| Unigrams | 60.33% | 56.19% |
| Bigrams | 53.71% | 51.20% |
| Facial displays | 70.24% | 76.03% |
| Hand gestures | 61.98% | 62.80% |
| Unigrams + Facial displays | 66.94% | 57.02% |
| All verbal | 60.33% | 50.41% |
| All non-verbal | 68.59% | 73.55% |
| All features | 75.20% | 50.41% |

Reported system results from Table 6:

| Modality | System Accuracy |
|---|---:|
| Text | 60.33% |
| Silent video | 68.59% |
| Full video | 75.20% |

The paper also reports that human annotators performed poorly, with agreement and detection accuracy only slightly above chance depending on modality.

## Current Project Summary

This project uses the same dataset family but a different experimental protocol:

- Group-disjoint train/validation/test split.
- Face crops generated automatically with MediaPipe.
- Learned visual models:
  - ResNet18 frame baseline.
  - R3D-18 RGB video baseline.
- Transcript TF-IDF baselines.
- Metadata baselines.
- Late fusion between visual ensemble and text LR.
- Metrics: AUC-ROC, AUC-PR, EER, Macro F1, Lie Recall, confusion matrix, bootstrap CI.

Main current results:

| Model | Modality | Test AUC-ROC | Macro F1 | Lie Recall |
|---|---|---:|---:|---:|
| Late fusion | Visual + Text | 0.678 | 0.619 | 0.462 |
| TF-IDF Logistic Regression | Text | 0.643 | 0.571 | 0.385 |
| ResNet18 frame | Visual | 0.622 +/- 0.086 | 0.571 +/- 0.075 | 0.487 +/- 0.247 |
| R3D-18 RGB | Visual | 0.476 | 0.486 | 0.308 |

## Key Protocol Differences

The results are not directly comparable.

Reasons:

- The original paper used leave-one-out cross-validation; this project uses a held-out group-disjoint test split.
- The original paper used manually annotated gesture and facial-display features; this project uses automatically learned visual features from face crops.
- The original paper reports primarily accuracy; this project emphasizes AUC-ROC, AUC-PR, EER, Macro F1, and Lie Recall.
- This project includes metadata and transcript baselines to expose possible shortcuts.
- The test set here has only 24 videos, so confidence intervals are wide.

## Fair Interpretation

The original paper's best reported result, 75.20% accuracy using all features with a Decision Tree, relies on manually annotated non-verbal features and a leave-one-out protocol. Under the stricter group-disjoint split used in this project, learned visual-only models are less stable. The best visual-only model is the ResNet18 frame ensemble, while the best overall result comes from late fusion with transcript features.

The R3D-18 result is a useful negative result: adding temporal 3D-CNN modeling did not improve performance on this small dataset. This suggests that model complexity alone is not enough; group/trial domain shift and small sample size remain the dominant issues.

## Suggested Report Wording

Compared with the original Real-life Trial Dataset paper, the present work uses a stricter group-disjoint split and automatic learned representations rather than manually annotated behavioral cues. Therefore, the reported numbers should be interpreted as a complementary evaluation rather than a direct reproduction. The late-fusion model achieved the best AUC-ROC in this project, while the visual-only models showed substantial variance and weaker generalization, indicating that trial-level domain shift is a central challenge for deception detection on this dataset.
