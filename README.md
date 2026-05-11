# Lie Detection on Real-life Trial Videos

This project builds a reproducible deception-detection pipeline for the Real-life Trial Deception Detection dataset.

The current implementation includes:

- dataset validation and group-disjoint splitting
- non-learning and metadata baselines
- MediaPipe face preprocessing
- frame-based ResNet18 visual baseline
- R3D-18 RGB video baseline
- transcript TF-IDF baselines
- late fusion between visual and text predictions
- threshold tuning on validation predictions
- error analysis and report-ready result tables

## Repository Contents

- `src/`: data processing, training, evaluation, reporting scripts
- `configs/`: experiment configs
- `docs/`: plan, progress log, report notes, paper comparison, artifact index
- `data/processed/splits/real_life/`: group-disjoint split files
- `outputs/metrics/`: reportable metrics, predictions, error analysis, final tables

Raw videos, derived face-crop videos, checkpoints, and visual contact-sheet images are intentionally not committed because they are large and may be restricted by dataset terms.

## Environment

The project was run in a Conda environment named `ai_env`.

Key package versions used:

- Python 3.10
- PyTorch 2.5.1 + CUDA 12.1
- torchvision 0.20.1
- OpenCV
- MediaPipe 0.10.14
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Main Commands

Data checks and splits:

```bash
conda run -n ai_env python -m src.data.data_check
conda run -n ai_env python -m src.data.make_splits --balanced-search
conda run -n ai_env python -m src.data.run_baselines
```

Face preprocessing:

```bash
conda run -n ai_env python -m src.data.preprocess_faces \
  --metadata outputs/metrics/real_life_data_check/metadata.csv \
  --out-dir data/processed/faces/real_life \
  --report-dir outputs/metrics/real_life_face_check \
  --preview-dir outputs/figures/real_life_face_previews
```

Visual and text baselines:

```bash
conda run -n ai_env python -m src.train --config configs/baseline_frame.yaml
conda run -n ai_env python -m src.train_3dcnn --config configs/rgb_3dcnn.yaml
conda run -n ai_env python -m src.data.run_text_baselines
conda run -n ai_env python -m src.data.run_late_fusion
```

Reporting:

```bash
conda run -n ai_env python -m src.data.analyze_errors \
  --predictions outputs/metrics/real_life_late_fusion/test_predictions.csv \
  --model-name late_fusion \
  --out-dir outputs/metrics/error_analysis

conda run -n ai_env python -m src.data.collect_results
conda run -n ai_env python -m src.data.tune_thresholds --optimize-metric macro_f1
```

## Current Best Result

The best main model is late fusion between the visual ensemble and transcript TF-IDF logistic regression:

- Test AUC-ROC: `0.678`
- Test AUC-PR: `0.690`
- Macro F1: `0.619`
- Lie Recall: `0.462`
- EER: `0.252`

The best visual-only result is the ResNet18 frame ensemble:

- Test AUC-ROC: `0.622 +/- 0.086`

See:

- `docs/report_notes.md`
- `outputs/metrics/final_report/final_results_summary.md`
- `docs/artifact_index.md`

## Notes

The original Real-life Trial Dataset paper used leave-one-out cross-validation and manually annotated behavioral features. This project uses a group-disjoint split and learned visual/text representations, so results are not directly comparable.
