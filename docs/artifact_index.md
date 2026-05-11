# Artifact Index

## Use These For Report

Final result tables:

- `outputs/metrics/final_report/final_results_table.csv`
- `outputs/metrics/final_report/final_results_summary.md`

Threshold tuning:

- `outputs/metrics/threshold_tuning/threshold_tuning_macro_f1.csv`
- `outputs/metrics/threshold_tuning/threshold_tuning_macro_f1.md`

Error analysis:

- `outputs/metrics/error_analysis/late_fusion_error_report.csv`
- `outputs/metrics/error_analysis/late_fusion_errors_only.csv`
- `outputs/metrics/error_analysis/late_fusion_error_summary.json`
- `outputs/metrics/error_analysis/resnet18_seed42_error_report.csv`
- `outputs/metrics/error_analysis/rgb_r3d18_error_report.csv`

Visualization:

- `outputs/figures/real_life_face_previews/contact_sheet.jpg`
- `outputs/figures/report_examples/visual_examples_contact_sheet.jpg`
- `outputs/metrics/final_report/visual_examples.md`

Experiment summaries:

- `outputs/metrics/baseline_frame_resnet18_multiseed/summary.json`
- `outputs/metrics/real_life_text_baselines/text_baselines.json`
- `outputs/metrics/real_life_late_fusion/summary.json`
- `outputs/metrics/rgb_r3d18/summary.json`
- `outputs/metrics/real_life_baselines/non_learning_baselines.json`

Dataset and preprocessing:

- `outputs/metrics/real_life_data_check/data_check_summary.json`
- `outputs/metrics/real_life_data_check/metadata.csv`
- `data/processed/splits/real_life/split_summary.json`
- `outputs/metrics/real_life_face_check/face_summary.json`
- `outputs/metrics/real_life_face_check/face_metadata.csv`

Report writing notes:

- `docs/report_notes.md`
- `docs/paper_comparison.md`
- `docs/progress.md`
- `docs/plan.md`

## Keep But Do Not Cite In Main Report

Smoke-test outputs were moved here:

- `outputs/archive/smoke/metrics/smoke_frame_resnet18/`
- `outputs/archive/smoke/metrics/smoke_rgb_r3d18/`
- `outputs/archive/smoke/checkpoints/smoke_frame_resnet18/`
- `outputs/archive/smoke/checkpoints/smoke_rgb_r3d18/`

These confirm that the training loops worked, but they are not real experiments and should not be included in the result table.

## Checkpoints

Main checkpoints:

- `outputs/checkpoints/baseline_frame_resnet18/best.pt`
- `outputs/checkpoints/baseline_frame_resnet18_seed123/best.pt`
- `outputs/checkpoints/baseline_frame_resnet18_seed2025/best.pt`
- `outputs/checkpoints/rgb_r3d18/best.pt`

Use checkpoints only if additional visualization such as Grad-CAM is needed.
