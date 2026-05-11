# Report Notes

## 1. Dataset and Protocol

Dataset used: Real-life Trial Deception Detection Dataset.

Current local dataset status:

- Total videos: 121.
- Labels: 61 deceptive, 60 truthful.
- Split protocol: group-disjoint split using `group_id` as the practical grouping key.
- Split sizes:
  - Train: 73 videos.
  - Validation: 24 videos.
  - Test: 24 videos.
- Main model-selection metric: validation AUC-ROC.
- Main reporting metrics: AUC-ROC, AUC-PR, Macro F1, Lie Recall, EER, confusion matrix.

Important methodology note:

- `group_id` is a proxy for subject/trial identity, not a perfect subject identifier.
- Test metrics are not used to choose model architecture, threshold, augmentation, or fusion weights.
- Threshold tuning is performed on validation only and then applied once to test predictions.

## 2. Preprocessing

Face preprocessing:

- Face crops were generated using MediaPipe.
- Output face crop size: 128x128.
- Detection interval: every 3 frames, with last-bounding-box reuse.
- Fallback: center crop.

Face crop result:

- 121/121 videos processed successfully.
- Mean face crop rate: 0.9975.
- Median face crop rate: 1.0.
- Videos below 70% face crop rate: 0.
- Videos below 95% face crop rate: 2.

Manual QA notes:

- `trial_truth_041.mp4`: occasional hand crop.
- `trial_lie_053.mp4`: low quality, low FPS, two visible faces.
- In the test-set error analysis, face crop quality does not explain most errors: both correct and incorrect late-fusion predictions have mean `face_crop_rate = 1.0`.

## 3. Main Results

Source table:

- `outputs/metrics/final_report/final_results_table.csv`
- `outputs/metrics/final_report/final_results_summary.md`

Main model results:

| Model | Modality | Protocol | Test AUC-ROC | AUC-PR | Macro F1 | Lie Recall | EER |
|---|---|---|---:|---:|---:|---:|---:|
| Late fusion | Visual ensemble + Text LR | group-disjoint | 0.678 | 0.690 | 0.619 | 0.462 | 0.252 |
| TF-IDF Logistic Regression | Text | group-disjoint | 0.643 | 0.667 | 0.571 | 0.385 | 0.458 |
| TF-IDF Linear SVM | Text | group-disjoint | 0.622 | 0.627 | 0.344 | 0.077 | 0.336 |
| ResNet18 frame | Visual | group-disjoint, 3 seeds | 0.622 +/- 0.086 | 0.669 +/- 0.064 | 0.571 +/- 0.075 | 0.487 +/- 0.247 | 0.402 +/- 0.097 |
| Metadata SVM | Metadata | group-disjoint | 0.601 | 0.699 | 0.497 | 0.538 | 0.503 |
| Metadata Logistic Regression | Metadata | group-disjoint | 0.503 | 0.532 | 0.497 | 0.538 | 0.458 |
| R3D-18 RGB | Visual | group-disjoint | 0.476 | 0.593 | 0.486 | 0.308 | 0.542 |

Interpretation:

- Best overall model: late fusion, AUC-ROC 0.678.
- Best visual-only result: ResNet18 frame ensemble, AUC-ROC 0.622 +/- 0.086.
- Text-only TF-IDF Logistic Regression performs strongly, AUC-ROC 0.643.
- R3D-18 pretrained is a negative result: it does not improve over frame-based visual modeling.

Sanity baseline warning:

- Random stratified baseline has high single-seed test AUC-ROC because the test set has only 24 videos.
- It should be treated as a sanity check, not as a meaningful model.

## 4. Threshold Tuning

Source files:

- `outputs/metrics/threshold_tuning/threshold_tuning_macro_f1.csv`
- `outputs/metrics/threshold_tuning/threshold_tuning_macro_f1.md`

Threshold tuning protocol:

- Choose threshold on validation by Macro F1.
- Apply selected threshold once to test.

Key findings:

| Model | Threshold | Test Macro F1 default | Test Macro F1 tuned | Lie Recall default | Lie Recall tuned |
|---|---:|---:|---:|---:|---:|
| ResNet18 seed42 | 0.589 | 0.657 | 0.667 | 0.769 | 0.615 |
| Late fusion | 0.488 | 0.619 | 0.624 | 0.462 | 0.538 |
| TF-IDF Linear SVM | 0.031 | 0.344 | 0.571 | 0.077 | 0.385 |
| TF-IDF Logistic Regression | 0.485 | 0.571 | 0.541 | 0.385 | 0.462 |
| ResNet18 frame ensemble | 0.581 | 0.580 | 0.467 | 0.462 | 0.231 |
| R3D-18 RGB | 0.375 | 0.486 | 0.314 | 0.308 | 0.846 |

Interpretation:

- Threshold tuning improves late fusion slightly.
- It substantially improves TF-IDF SVM Macro F1 because the default SVM decision threshold gives very low Lie Recall.
- It hurts ResNet18 ensemble and R3D-18, indicating that validation threshold selection is unstable on this small validation set.
- Threshold tuning should be reported as an ablation, not as a replacement for all default-threshold results.

## 5. Error Analysis

Source files:

- `outputs/metrics/error_analysis/late_fusion_error_report.csv`
- `outputs/metrics/error_analysis/late_fusion_errors_only.csv`
- `outputs/metrics/error_analysis/late_fusion_error_summary.json`
- `outputs/metrics/error_analysis/resnet18_seed42_error_report.csv`
- `outputs/metrics/error_analysis/rgb_r3d18_error_report.csv`

Late-fusion confusion breakdown:

- True negative: 9.
- True positive: 6.
- False positive: 2.
- False negative: 7.

Main error pattern:

- Errors concentrate in three test groups:
  - `Andrea Sneiderman`: 6 errors.
  - `Crystal Mangum`: 2 errors.
  - `Jamie Hood`: 1 error.
- Most errors are from `Defendant` role videos.
- Face crop quality is not the main explanation:
  - Correct mean face crop rate: 1.0.
  - Error mean face crop rate: 1.0.
  - Center crop frames in test errors: 0.

Interpretation:

- The primary failure mode appears to be group/trial domain shift, not preprocessing failure.
- Late fusion reduces some false positives compared with visual-only seed 42, but increases false negatives because transcript scores pull several deceptive examples below 0.5.
- Visual model seed 42 produces false positives mainly on truthful `Andrea Sneiderman` videos.

Visualization artifacts:

- `outputs/figures/report_examples/visual_examples_contact_sheet.jpg`
- `outputs/figures/report_examples/visual_examples.csv`
- `outputs/metrics/final_report/visual_examples.md`

The visualization sheet contains representative TP/TN/FP/FN examples from the late-fusion test predictions. Several false positives and false negatives come from `Andrea Sneiderman`, matching the group-level error analysis.

## 6. Comparison with Original Paper

Original paper:

- Pérez-Rosas, Abouelenien, Mihalcea, and Burzo, "Deception Detection using Real-life Trial Data", ICMI 2015.
- Paper URL: https://web.eecs.umich.edu/~mihalcea/papers/perezrosas.icmi15.pdf

Relevant details from the paper:

- The paper introduced 121 real-life trial video clips.
- It used verbal features from transcripts and non-verbal gesture/facial-display annotations.
- It reported a baseline of 50.4% due to near-balanced labels.
- It used Decision Trees and Random Forest classifiers with leave-one-out cross-validation.
- Reported accuracies include:
  - Unigrams: 60.33% DT, 56.19% RF.
  - Facial displays: 70.24% DT, 76.03% RF.
  - All non-verbal: 68.59% DT, 73.55% RF.
  - All features: 75.20% DT, 50.41% RF.
- The paper's human-study system row reports:
  - Text: 60.33%.
  - Silent video: 68.59%.
  - Full video: 75.20%.

Protocol differences:

- Original paper used hand-annotated verbal and non-verbal cue features and leave-one-out cross-validation.
- This project uses group-disjoint train/validation/test split to reduce trial/group leakage.
- This project uses learned visual features from face crops, transcript TF-IDF, metadata baselines, and late fusion.
- This project reports AUC-ROC, AUC-PR, EER, Macro F1, Lie Recall, and bootstrap CIs instead of accuracy only.
- Results are not directly comparable because the evaluation protocol and feature sets differ.

Fair comparison statement:

- The original paper reports higher accuracy for annotated non-verbal and multimodal features, but those results rely on manual gesture/facial-display annotations and leave-one-out evaluation.
- Under the stricter group-disjoint split used here, learned face-crop models are less stable, and the best AUC-ROC comes from late fusion with transcript features.

## 7. Negative Results

R3D-18:

- Validation AUC-ROC: 0.681.
- Test AUC-ROC: 0.476.
- Macro F1: 0.486.
- Lie Recall: 0.308.

Interpretation:

- Temporal 3D-CNN does not improve over frame-based ResNet18.
- The dataset is likely too small for stable fine-tuning of a 3D-CNN.
- Short 8-frame clips may not capture reliable deception-related temporal cues.
- The strong group/trial shift can dominate subtle temporal signals.

## 8. Recommended Report Conclusion

The project successfully builds a reproducible deception-detection pipeline on the Real-life Trial dataset using group-disjoint splits, face preprocessing, visual models, transcript models, metadata baselines, late fusion, threshold tuning, and error analysis. The best overall test AUC-ROC is obtained by late fusion, while the best visual-only approach is a ResNet18 frame ensemble. The R3D-18 experiment is a negative result, suggesting that temporal modeling is not automatically beneficial on this small and domain-shifted dataset. Error analysis indicates that remaining failures are driven more by group/trial shift than by face crop quality.
