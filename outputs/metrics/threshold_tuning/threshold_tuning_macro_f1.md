# Threshold Tuning

Optimized on validation: `macro_f1`

| model | selected_threshold | val_selection_score | test_auc_roc | test_macro_f1_default | test_macro_f1_tuned | test_lie_recall_default | test_lie_recall_tuned | test_accuracy_tuned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet18_seed42 | 0.589 | 0.497 | 0.720 | 0.657 | 0.667 | 0.769 | 0.615 | 0.667 |
| late_fusion | 0.488 | 0.708 | 0.678 | 0.619 | 0.624 | 0.462 | 0.538 | 0.625 |
| tfidf_linear_svm | 0.031 | 0.695 | 0.622 | 0.344 | 0.571 | 0.077 | 0.385 | 0.583 |
| tfidf_logistic_regression | 0.485 | 0.667 | 0.643 | 0.571 | 0.541 | 0.385 | 0.462 | 0.542 |
| resnet18_frame_ensemble | 0.581 | 0.556 | 0.622 | 0.580 | 0.467 | 0.462 | 0.231 | 0.500 |
| r3d18_rgb | 0.375 | 0.664 | 0.476 | 0.486 | 0.314 | 0.308 | 0.846 | 0.458 |

Notes:

- Threshold is selected on validation only, then applied once to test predictions.

- AUC-ROC and EER do not depend on the classification threshold; threshold tuning affects accuracy, F1, recall, precision, and confusion matrix.

