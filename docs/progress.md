# Tien trinh thuc hien du an

## Thong tin hien tai

- Dataset dang dung: Real-life Deception Detection Dataset 2016.
- Archive: `data/RealLifeDeceptionDetection.2016.zip`.
- Raw data: `data/raw/Real-life_Deception_Detection_2016/`.
- Moi truong chay: `ai_env`.
- GPU local: NVIDIA RTX 3060 12GB.

## Moi truong

Da xac nhan trong `ai_env`:

```text
Python 3.10.20
torch 2.5.1+cu121
torchvision 0.20.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
mediapipe 0.10.14
opencv 4.13.0
scikit-learn 1.7.2
```

Lenh chay mac dinh:

```bash
conda run -n ai_env python -m <module>
```

## Da hoan thanh

### 1. Chuyen dataset

Ly do:

- Khong xin duoc quyen su dung Bag-of-Lies.
- Chuyen sang Real-life Deception Detection Dataset 2016.

Ket qua:

- Giai nen thanh cong.
- Khong can password.
- Co video, transcript, annotation gesture.

### 2. Data check

Lenh da chay:

```bash
conda run -n ai_env python -m src.data.data_check \
  --data-root data/raw/Real-life_Deception_Detection_2016 \
  --zip data/RealLifeDeceptionDetection.2016.zip \
  --annotation 'data/raw/Real-life_Deception_Detection_2016/Annotation/All_Gestures_Deceptive and Truthful.csv' \
  --out-dir outputs/metrics/real_life_data_check
```

Ket qua:

- 121 video.
- 121/121 video doc duoc.
- 121 transcript.
- 61 deceptive.
- 60 truthful.
- 32 `group_id` suy ra tu README.

Output:

- `outputs/metrics/real_life_data_check/metadata.csv`
- `outputs/metrics/real_life_data_check/data_check_summary.json`
- `outputs/metrics/real_life_data_check/annotation_preview.csv`
- `outputs/metrics/real_life_data_check/subject_label_counts.csv`

### 3. Group-disjoint split

Do dataset khong co `subject_id`, da dung `group_id` theo trial/person name lam proxy de giam leakage.

Lenh da chay:

```bash
conda run -n ai_env python -m src.data.make_splits \
  --metadata outputs/metrics/real_life_data_check/metadata.csv \
  --subject-column group_id \
  --train-subjects 22 \
  --val-subjects 5 \
  --test-subjects 5 \
  --balanced-search \
  --out-dir data/processed/splits/real_life
```

Ket qua split:

| Split | Videos | Groups | Deceptive | Truthful |
|---|---:|---:|---:|---:|
| train | 73 | 22 | 36 | 37 |
| val | 24 | 5 | 12 | 12 |
| test | 24 | 5 | 13 | 11 |

Output:

- `data/processed/splits/real_life/train.csv`
- `data/processed/splits/real_life/val.csv`
- `data/processed/splits/real_life/test.csv`
- `data/processed/splits/real_life/subjects.json`
- `data/processed/splits/real_life/split_summary.json`
- `data/processed/splits/real_life/subject_stats.csv`

Ghi chu terminology:

- Trong bao cao se goi la `group-disjoint split`.
- File `subjects.json` la ten cu cua script; noi dung trong file nay la danh sach `group_id`, khong phai subject ID chuan.
- Known fix: README bi lap `trial_truth_008.mp4` va thieu `trial_truth_009.mp4`; pipeline da gan `trial_truth_009.mp4` vao group `Jodi Arias` va video nay nam trong train split.

### 4. Non-learning va metadata baselines

Lenh da chay:

```bash
conda run -n ai_env python -m src.data.run_baselines \
  --train data/processed/splits/real_life/train.csv \
  --val data/processed/splits/real_life/val.csv \
  --test data/processed/splits/real_life/test.csv \
  --out-dir outputs/metrics/real_life_baselines
```

Ket qua test:

| Model | AUC-ROC | Macro F1 | Lie Recall | EER |
|---|---:|---:|---:|---:|
| Majority | 0.500 | 0.314 | 0.000 | 0.500 |
| Random stratified | 0.717 | 0.708 | 0.615 | 0.283 |
| Metadata Logistic Regression | 0.503 | 0.497 | 0.538 | 0.458 |
| Metadata SVM | 0.601 | 0.497 | 0.538 | 0.503 |

Output:

- `outputs/metrics/real_life_baselines/non_learning_baselines.json`

Luu y:

- Metadata baseline co the hoc bias tu duration/resolution/file size.
- Chi dung lam moc tham khao, khong phai model deception chinh.
- Metadata SVM sau khi sua split dat test AUC-ROC `0.601`; neu visual model khong vuot duoc moc nay, can phan tich kha nang metadata dang hoc shortcut/technical bias.
- Random baseline AUC cao bat thuong do test set nho va mot seed ngau nhien; khong xem day la moc model that.

### 5. Face crop bang MediaPipe

Lenh da chay:

```bash
conda run -n ai_env python -m src.data.preprocess_faces \
  --metadata outputs/metrics/real_life_data_check/metadata.csv \
  --out-dir data/processed/faces/real_life \
  --report-dir outputs/metrics/real_life_face_check \
  --preview-dir outputs/figures/real_life_face_previews \
  --overwrite
```

Ket qua:

- 121/121 video crop thanh cong.
- Mean face crop rate: `0.9975`.
- Median face crop rate: `1.0`.
- Video duoi 70%: `0`.
- Video duoi 95%: `2`.
- Mean new detection rate: `0.9628`.

Output:

- `data/processed/faces/real_life/`
- `outputs/metrics/real_life_face_check/face_metadata.csv`
- `outputs/metrics/real_life_face_check/face_summary.json`
- `outputs/metrics/real_life_face_check/low_face_crop_rate.csv`
- `outputs/figures/real_life_face_previews/contact_sheet.jpg`

Luu y:

- Face crop khong con la blocker lon theo metric hien tai.
- Nen xem nhanh `contact_sheet.jpg` va 2 video co face crop rate thap nhat truoc khi train visual baseline.
- QA thu cong can chu y:
  - `trial_truth_041.mp4`: mot so frame crop vao tay thay vi mat.
  - `trial_lie_053.mp4`: chat luong kem, fps thap, co 2 khuon mat.

### 6. Frame-based CNN baseline tren face crops

Da thay placeholder `src/train.py` bang training entry point that:

- Doc config tu `configs/baseline_frame.yaml`.
- Dung `outputs/metrics/real_life_face_check/face_metadata.csv` de lay face crop path.
- Train ResNet18 ImageNet pretrained tren 8 frame/video.
- Aggregate frame probabilities thanh video-level probability bang average.
- Chon checkpoint theo validation AUC-ROC.
- Xuat prediction va metrics cho validation/test.
- Bootstrap 95% CI tren test predictions.

Lenh smoke test da chay thanh cong:

```bash
conda run -n ai_env python -c "... train smoke_frame_resnet18 ..."
```

Lenh train baseline:

```bash
conda run -n ai_env python -m src.train --config configs/baseline_frame.yaml
```

Ket qua:

- Device: `cuda` tren RTX 3060 12GB.
- Best epoch: `1`.
- Best validation AUC-ROC: `0.382`.
- Test AUC-ROC: `0.720`.
- Test AUC-PR: `0.738`.
- Test Macro F1: `0.657`.
- Test Lie Recall: `0.769`.
- Test EER: `0.290`.
- Test confusion matrix: `[[6, 5], [3, 10]]`.
- Bootstrap 95% CI AUC-ROC: `[0.467, 0.924]`.

Output:

- `outputs/checkpoints/baseline_frame_resnet18/best.pt`
- `outputs/metrics/baseline_frame_resnet18/train_log.csv`
- `outputs/metrics/baseline_frame_resnet18/val_predictions.csv`
- `outputs/metrics/baseline_frame_resnet18/val_metrics.json`
- `outputs/metrics/baseline_frame_resnet18/test_predictions.csv`
- `outputs/metrics/baseline_frame_resnet18/test_metrics.json`
- `outputs/metrics/baseline_frame_resnet18/summary.json`

Luu y:

- Test AUC-ROC vuot metadata SVM (`0.720` vs `0.601`), nhung validation AUC-ROC thap (`0.382`), nen chua dien giai manh tu mot seed don le.
- Dataset nho va split group-disjoint khac biet domain manh; can chay them multiple seeds `[42, 123, 2025]`.
- Can lam error analysis tren `test_predictions.csv`, dac biet cac false positive/false negative va video crop chat luong kem.

### 7. Multiple seeds cho frame-based CNN

Da chay them seed `123` va `2025` bang cung config, chi doi seed va output dir.

Ket qua tung seed:

| Seed | Best val AUC-ROC | Test AUC-ROC | Test AUC-PR | Macro F1 | Lie Recall | EER |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.382 | 0.720 | 0.738 | 0.657 | 0.769 | 0.290 |
| 123 | 0.278 | 0.559 | 0.660 | 0.521 | 0.308 | 0.458 |
| 2025 | 0.833 | 0.587 | 0.610 | 0.534 | 0.385 | 0.458 |

Trung binh 3 seeds:

- Test AUC-ROC: `0.622 +/- 0.086`.
- Test AUC-PR: `0.669 +/- 0.064`.
- Test Macro F1: `0.571 +/- 0.075`.
- Test Lie Recall: `0.487 +/- 0.247`.
- Test EER: `0.402 +/- 0.097`.

Output:

- `outputs/metrics/baseline_frame_resnet18_multiseed/seed_results.csv`
- `outputs/metrics/baseline_frame_resnet18_multiseed/summary.json`

Nhan xet:

- Seed 42 cho ket qua tot nhat, mean 3 seeds vuot nhe Metadata SVM AUC-ROC (`0.622` vs `0.601`).
- Variance rat lon, dac biet Lie Recall; can bao cao mean/std thay vi mot seed don le.
- Best validation AUC-ROC khong on dinh voi test AUC-ROC, cho thay validation set 24 video qua nho va co domain shift theo group.

### 8. Transcript TF-IDF baselines

Da them `src/data/run_text_baselines.py`.

Lenh da chay:

```bash
conda run -n ai_env python -m src.data.run_text_baselines
```

Ket qua:

| Model | Val AUC-ROC | Test AUC-ROC | Test AUC-PR | Macro F1 | Lie Recall | EER |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF Logistic Regression | 0.701 | 0.643 | 0.667 | 0.571 | 0.385 | 0.458 |
| TF-IDF Linear SVM | 0.694 | 0.622 | 0.627 | 0.344 | 0.077 | 0.336 |

Output:

- `outputs/metrics/real_life_text_baselines/text_baselines.json`
- `outputs/metrics/real_life_text_baselines/*_predictions.csv`
- `outputs/metrics/real_life_text_baselines/*_metrics.json`

Nhan xet:

- Text Logistic Regression test AUC-ROC (`0.643`) cao hon visual ResNet18 mean 3 seeds (`0.622`).
- Text baseline co nguy co hoc shortcut theo trial/source/noi dung vu an, khong nen dien giai nhu tin hieu hanh vi noi doi.
- TF-IDF SVM co AUC kha nhung threshold mac dinh cho Lie Recall rat thap (`0.077`), can can nhac threshold tuning tren validation neu dung cho bao cao precision/recall.

### 9. Late fusion baseline

Da them `src/data/run_late_fusion.py`.

Lenh da chay:

```bash
conda run -n ai_env python -m src.data.run_late_fusion
```

Cach lam:

- Visual score = average probability cua 3 ResNet18 frame seeds.
- Text score = TF-IDF Logistic Regression probability.
- Chon alpha tren validation AUC-ROC voi grid `0.0, 0.05, ..., 1.0`.
- Fusion score = `alpha * visual + (1 - alpha) * text`.

Ket qua:

- Best alpha visual: `0.15`.
- Best validation AUC-ROC: `0.722`.
- Test AUC-ROC: `0.678`.
- Test AUC-PR: `0.690`.
- Test Macro F1: `0.619`.
- Test Lie Recall: `0.462`.
- Test EER: `0.252`.
- Confusion matrix: `[[9, 2], [7, 6]]`.
- Bootstrap 95% CI AUC-ROC: `[0.420, 0.905]`.

Output:

- `outputs/metrics/real_life_late_fusion/alpha_search.csv`
- `outputs/metrics/real_life_late_fusion/test_predictions.csv`
- `outputs/metrics/real_life_late_fusion/summary.json`

Nhan xet:

- Fusion cai thien AUC-ROC so voi visual mean (`0.678` vs `0.622`) va text LR (`0.678` vs `0.643`).
- Alpha visual chi `0.15`, cho thay text signal manh hon visual trong split nay.
- Can ghi ro text/fusion co nguy co hoc shortcut theo noi dung transcript/source, khong thay the cho behavioral visual model.

### 10. Error analysis

Da them `src/data/analyze_errors.py`.

Lenh da chay:

```bash
conda run -n ai_env python -m src.data.analyze_errors \
  --predictions outputs/metrics/real_life_late_fusion/test_predictions.csv \
  --model-name late_fusion \
  --out-dir outputs/metrics/error_analysis

conda run -n ai_env python -m src.data.analyze_errors \
  --predictions outputs/metrics/baseline_frame_resnet18/test_predictions.csv \
  --model-name resnet18_seed42 \
  --out-dir outputs/metrics/error_analysis
```

Output:

- `outputs/metrics/error_analysis/late_fusion_error_report.csv`
- `outputs/metrics/error_analysis/late_fusion_errors_only.csv`
- `outputs/metrics/error_analysis/late_fusion_error_summary.json`
- `outputs/metrics/error_analysis/resnet18_seed42_error_report.csv`
- `outputs/metrics/error_analysis/resnet18_seed42_errors_only.csv`
- `outputs/metrics/error_analysis/resnet18_seed42_error_summary.json`

Ket qua chinh:

- Late fusion: `TN=9`, `TP=6`, `FP=2`, `FN=7`.
- ResNet18 seed 42: `TN=6`, `TP=10`, `FP=5`, `FN=3`.
- Loi tap trung vao 3 group test: `Andrea Sneiderman`, `Crystal Mangum`, `Jamie Hood`.
- Face crop khong phai nguyen nhan chinh: test error va correct deu co mean `face_crop_rate = 1.0`, center crop frames = `0`.
- ResNet18 seed 42 false positive toan bo nam o truthful `Andrea Sneiderman`.
- Late fusion giam false positive nhung tang false negative vi text score keo nhieu deceptive clip xuong duoi threshold 0.5.

Nhan xet bao cao:

- Loi co dau hieu domain/group shift theo trial hon la loi tien xu ly face crop.
- Nen trinh bay ket qua theo group va neu co thoi gian thu threshold tuning tren validation thay vi threshold mac dinh 0.5.

### 11. RGB 3D-CNN baseline

Da them `src/train_3dcnn.py` va cap nhat `configs/rgb_3dcnn.yaml`.

Cau hinh:

- Model: `r3d_18` pretrained Kinetics-400.
- Input: face crop clip 8 frames, `112x112`.
- Batch size: `4`.
- Freeze backbone 2 epoch dau, sau do fine-tune voi LR backbone `1e-5`, head `1e-4`.
- AMP bat tren CUDA.
- Early stopping theo validation AUC-ROC.

Lenh smoke test da chay thanh cong:

```bash
conda run -n ai_env python -c "... run smoke_rgb_r3d18 ..."
```

Lenh train baseline:

```bash
conda run -n ai_env python -m src.train_3dcnn --config configs/rgb_3dcnn.yaml
```

Ket qua:

- Device: `cuda`.
- Best epoch: `2`.
- Best validation AUC-ROC: `0.681`.
- Test AUC-ROC: `0.476`.
- Test AUC-PR: `0.593`.
- Test Macro F1: `0.486`.
- Test Lie Recall: `0.308`.
- Test EER: `0.542`.
- Confusion matrix: `[[8, 3], [9, 4]]`.
- Bootstrap 95% CI AUC-ROC: `[0.237, 0.729]`.

Output:

- `outputs/checkpoints/rgb_r3d18/best.pt`
- `outputs/metrics/rgb_r3d18/train_log.csv`
- `outputs/metrics/rgb_r3d18/val_predictions.csv`
- `outputs/metrics/rgb_r3d18/test_predictions.csv`
- `outputs/metrics/rgb_r3d18/summary.json`
- `outputs/metrics/error_analysis/rgb_r3d18_error_report.csv`
- `outputs/metrics/error_analysis/rgb_r3d18_error_summary.json`

Nhan xet:

- R3D-18 khong cai thien so voi frame-based ResNet18; day la negative result can bao cao.
- Loi R3D-18 van tap trung theo group/trial, dac biet `Andrea Sneiderman`.
- Face crop tiep tuc khong phai nguyen nhan chinh: error mean `face_crop_rate = 1.0`.
- Co kha nang dataset qua nho de fine-tune 3D-CNN on dinh; temporal signal 8 frames khong du hoac khong lien quan manh.

### 12. Final result collection

Da them `src/data/collect_results.py`.

Lenh da chay:

```bash
conda run -n ai_env python -m src.data.collect_results
```

Output:

- `outputs/metrics/final_report/final_results_table.csv`
- `outputs/metrics/final_report/final_results_summary.md`

Noi dung:

- Gom Majority, Random, Metadata LR/SVM, ResNet18 frame 3 seeds, R3D-18, TF-IDF LR/SVM, Late fusion.
- Markdown tach rieng `Main Models` va `Sanity Baselines` de tranh hieu nham random baseline la model tot.
- Main model tot nhat theo test AUC-ROC hien tai: Late fusion, AUC-ROC `0.678`.
- Visual-only tot nhat: ResNet18 frame 3 seeds, AUC-ROC `0.622 +/- 0.086`.
- R3D-18 la negative result, AUC-ROC `0.476`.

### 13. Threshold tuning

Da them `src/data/tune_thresholds.py`.

Lenh da chay:

```bash
conda run -n ai_env python -m src.data.tune_thresholds --optimize-metric macro_f1
```

Nguyen tac:

- Chon threshold tren validation theo Macro F1.
- Ap dung threshold da chon len test.
- Khong dung test de chon threshold.

Output:

- `outputs/metrics/threshold_tuning/threshold_tuning_macro_f1.csv`
- `outputs/metrics/threshold_tuning/threshold_tuning_macro_f1.json`
- `outputs/metrics/threshold_tuning/threshold_tuning_macro_f1.md`
- `outputs/metrics/threshold_tuning/*_test_predictions_tuned.csv`

Ket qua chinh:

| Model | Threshold | Test Macro F1 default | Test Macro F1 tuned | Lie Recall default | Lie Recall tuned |
|---|---:|---:|---:|---:|---:|
| ResNet18 seed42 | 0.589 | 0.657 | 0.667 | 0.769 | 0.615 |
| Late fusion | 0.488 | 0.619 | 0.624 | 0.462 | 0.538 |
| TF-IDF Linear SVM | 0.031 | 0.344 | 0.571 | 0.077 | 0.385 |
| TF-IDF Logistic Regression | 0.485 | 0.571 | 0.541 | 0.385 | 0.462 |
| ResNet18 frame ensemble | 0.581 | 0.580 | 0.467 | 0.462 | 0.231 |
| R3D-18 RGB | 0.375 | 0.486 | 0.314 | 0.308 | 0.846 |

Nhan xet:

- Threshold tuning huu ich cho TF-IDF SVM va cai thien nhe cho Late fusion.
- Threshold tuning lam giam ResNet18 ensemble va R3D-18 tren test, cho thay validation set nho khong on dinh.
- Bao cao nen trinh bay threshold tuning nhu ablation phu, khong thay the bang ket qua mac dinh cho tat ca model.

### 14. Report notes, visualization, paper comparison, artifact cleanup

Da tao cac file ho tro viet bao cao:

- `docs/report_notes.md`
- `docs/paper_comparison.md`
- `docs/artifact_index.md`

Visualization:

- Da them `src/data/make_visual_examples.py`.
- Da chay contact sheet cho late-fusion TP/TN/FP/FN examples.
- Output:
  - `outputs/figures/report_examples/visual_examples_contact_sheet.jpg`
  - `outputs/figures/report_examples/visual_examples.csv`
  - `outputs/metrics/final_report/visual_examples.md`

So sanh paper goc:

- Da tom tat paper Perez-Rosas et al., ICMI 2015.
- Da ghi protocol khac nhau:
  - paper goc dung leave-one-out CV va manual gesture/facial-display annotations.
  - du an hien tai dung group-disjoint split va learned face/text/metadata/fusion models.
  - khong so sanh truc tiep vi metric va protocol khac nhau.

Artifact cleanup:

- Khong xoa smoke outputs.
- Da chuyen smoke artifacts vao:
  - `outputs/archive/smoke/metrics/`
  - `outputs/archive/smoke/checkpoints/`
- Main report artifacts duoc liet ke trong `docs/artifact_index.md`.

## Dang lam

- San sang viet bao cao chinh thuc.

## Viec tiep theo

1. Viet bao cao chinh thuc tu `docs/report_notes.md`.
2. Chen bang tu `outputs/metrics/final_report/final_results_summary.md`.
3. Chen visualization tu `outputs/figures/report_examples/visual_examples_contact_sheet.jpg`.
4. Neu con thoi gian, them Grad-CAM that cho ResNet18 seed42.
5. Kiem tra lai format citation/reference.

## Rui ro hien tai

- Dataset nho: 121 video, can than overfit.
- `group_id` chi la proxy, khong phai subject ID chuan.
- Video courtroom co the kho face crop do goc quay/blur/occlusion, nhung lan face crop dau tien dat rate tot.
- Metadata baseline co nguy co shortcut theo tinh chat file/video.
- Van can xem preview thu cong de tranh crop dung metric nhung sai ve mat ngu nghia.

## Trang thai tong quat

Du an da qua buoc chuan bi du lieu, co baseline toi thieu, da tao face crops cho toan bo dataset, da co frame-based visual baseline 3 seeds, transcript TF-IDF baseline, late fusion baseline, error analysis, RGB 3D-CNN baseline, final result table, threshold tuning, visualization examples, paper comparison va artifact index. Pipeline san sang viet bao cao chinh thuc.
