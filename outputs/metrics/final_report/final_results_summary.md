# Final Results

## Main Models

| model | modality | split_protocol | val_auc_roc | test_auc_roc | test_auc_pr | test_macro_f1 | test_lie_recall | test_eer | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Late fusion | Visual ensemble + Text LR | group-disjoint | 0.722 | 0.678 | 0.690 | 0.619 | 0.462 | 0.252 | alpha_visual=0.15 |
| TF-IDF Logistic Regression | Text | group-disjoint | 0.701 | 0.643 | 0.667 | 0.571 | 0.385 | 0.458 | Transcript may contain trial/source shortcuts |
| TF-IDF Linear SVM | Text | group-disjoint | 0.694 | 0.622 | 0.627 | 0.344 | 0.077 | 0.336 | Default threshold has low Lie recall |
| ResNet18 frame | Visual | group-disjoint, 3 seeds | 0.498 +/- 0.295 | 0.622 +/- 0.086 | 0.669 +/- 0.064 | 0.571 +/- 0.075 | 0.487 +/- 0.247 | 0.402 +/- 0.097 | Mean +/- std over seeds 42, 123, 2025 |
| Metadata SVM | Metadata | group-disjoint | 0.729 | 0.601 | 0.699 | 0.497 | 0.538 | 0.503 |  |
| Metadata Logistic Regression | Metadata | group-disjoint | 0.604 | 0.503 | 0.532 | 0.497 | 0.538 | 0.458 |  |
| R3D-18 RGB | Visual | group-disjoint | 0.681 | 0.476 | 0.593 | 0.486 | 0.308 | 0.542 | Negative result |

## Sanity Baselines

| model | modality | split_protocol | val_auc_roc | test_auc_roc | test_auc_pr | test_macro_f1 | test_lie_recall | test_eer | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random stratified | Label only | group-disjoint | 0.667 | 0.717 | 0.701 | 0.708 | 0.615 | 0.283 | Random baseline is one seed only |
| Majority baseline | Label only | group-disjoint | 0.500 | 0.500 | 0.542 | 0.314 | 0.000 | 0.500 |  |

Notes:

- Main model ranking uses test AUC-ROC and excludes label-only sanity baselines.
- ResNet18 frame is reported as mean +/- std over 3 seeds.
- Random stratified is retained as a sanity baseline, not as a meaningful model; its high score reflects small-test-set variance from one random seed.
- Text and fusion results may include trial/source-content shortcuts and should be interpreted separately from behavioral visual models.
