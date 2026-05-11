# Ke hoach du an: Nhan dang noi doi tren Real-life Trial Dataset

## 0. Muc tieu va protocol chinh

Dataset chinh da chuyen tu Bag-of-Lies sang **Real-life Deception Detection Dataset 2016**.

Muc tieu:

- Phan loai video thanh 2 lop:
  - `0`: Truthful / noi that
  - `1`: Deceptive / noi doi
- Danh gia o muc **video-level**, khong chi clip-level.
- Bao cao theo protocol ro rang, tranh data leakage va test contamination.

Thong tin dataset hien tai:

- 121 video.
- 61 deceptive, 60 truthful.
- 121 transcript.
- 32 `group_id` suy ra tu README theo trial/person name.
- Khong co `subject_id` rieng trong annotation, nen dung `group_id` de chia group-disjoint split.

Protocol chinh:

- Split: **group-disjoint** theo `group_id`.
- Split hien tai:
  - train: 73 video, 36 lie / 37 truth, 22 groups
  - val: 24 video, 12 lie / 12 truth, 5 groups
  - test: 24 video, 13 lie / 11 truth, 5 groups
- Metric chon model: validation video-level AUC-ROC.
- Metric bao cao chinh: test video-level AUC-ROC, Macro F1, Lie Recall, EER.
- Model chinh: visual model tren face/video, uu tien pretrained model.
- Test set chi dung sau khi da chon model bang validation.

Quy tac chong test contamination:

- Khong dung test set de chon model, backbone, augmentation, clip length, scheduler, threshold.
- Khong xem test metric lap lai trong qua trinh tuning.
- Neu bat buoc chay test nhieu lan, phai ghi ro so lan va ly do trong bao cao.

## 1. Moi truong va tai nguyen

Moi truong chinh:

- Conda env: `ai_env`
- Python: 3.10.20
- GPU local: **NVIDIA RTX 3060 12GB**
- CUDA trong PyTorch: CUDA 12.1
- Torch: `2.5.1+cu121`
- Torchvision: `0.20.1+cu121`
- OpenCV: `4.13.0`
- MediaPipe: `0.10.14`
- Scikit-learn: `1.7.2`

Nguyen tac chay:

- Chay local tren RTX 3060 12GB la mac dinh.
- Dung AMP/mixed precision khi train deep learning.
- Neu 3D-CNN bi thieu VRAM:
  - giam batch size
  - dung gradient accumulation `steps=2-4`
  - giam clip length tu 16 xuong 8
  - dung frame-based CNN truoc
- Colab A100 chi la tuy chon neu can train 3D-CNN nhieu cau hinh hoac batch lon.

Lenh chay mac dinh:

```bash
conda run -n ai_env python -m <module>
```

## 2. Cau truc thu muc

```text
lie_nonlie/
  data/
    RealLifeDeceptionDetection.2016.zip
    raw/
      Real-life_Deception_Detection_2016/
    processed/
      splits/real_life/
      faces/real_life/
      flow/real_life/
  docs/
    plan.md
    progress.md
  configs/
    baseline_frame.yaml
    rgb_3dcnn.yaml
    flow_3dcnn.yaml
    audio_baseline.yaml
  src/
    data/
      data_check.py
      extract_archive.py
      make_splits.py
      run_baselines.py
      preprocess_faces.py
      dataset.py
      audio_features.py
    models/
      video_models.py
      audio_models.py
    evaluate.py
    train.py
    utils.py
  outputs/
    metrics/
    checkpoints/
    figures/
  requirements.txt
```

## 3. Giai doan 1: Du lieu, metadata va split

Trang thai: **da hoan thanh buoc nen**.

Viec da lam:

1. Giai nen `data/RealLifeDeceptionDetection.2016.zip` vao `data/raw/`.
2. Doc annotation:
   - `Annotation/All_Gestures_Deceptive and Truthful.csv`
3. Tao metadata:
   - `video_id`
   - `video_path`
   - `label`
   - `transcript_path`
   - `duration_sec`
   - `fps`
   - `num_frames`
   - `width`, `height`
   - `group_id`
4. Suy ra `group_id` tu README theo trial/person name.
5. Tao group-disjoint split can bang nhan.

File ket qua:

- `outputs/metrics/real_life_data_check/metadata.csv`
- `outputs/metrics/real_life_data_check/data_check_summary.json`
- `data/processed/splits/real_life/train.csv`
- `data/processed/splits/real_life/val.csv`
- `data/processed/splits/real_life/test.csv`
- `data/processed/splits/real_life/split_summary.json`

Luu y methodology:

- Dataset khong co subject ID tieu chuan.
- `group_id` la proxy theo trial/person name, tot hon random split theo video.
- Can ghi ro trong bao cao: group-disjoint split duoc dung de giam leakage, nhung khong hoan hao bang subject ID chuan.
- Thong nhat thuat ngu: trong plan/bao cao dung **group**; trong mot so file code/output cu, key `subjects` thuc chat dang chua danh sach `group_id`.
- Known fix: README bi lap `trial_truth_008.mp4` va thieu `trial_truth_009.mp4`; pipeline da gan `trial_truth_009.mp4` vao group `Jodi Arias` dua tren ngu canh bang README.

## 4. Giai doan 2: Baseline khong hoc sau va metadata baseline

Trang thai: **da chay xong**.

Baselines:

- Majority baseline.
- Random stratified baseline.
- Metadata Logistic Regression.
- Metadata SVM.

Metadata features dang dung:

- `duration_sec`
- `fps`
- `num_frames`
- `width`
- `height`
- `file_size_bytes`

Ket qua test hien tai:

| Model | AUC-ROC | Macro F1 | Lie Recall | EER |
|---|---:|---:|---:|---:|
| Majority | 0.500 | 0.314 | 0.000 | 0.500 |
| Random stratified | 0.717 | 0.708 | 0.615 | 0.283 |
| Metadata Logistic Regression | 0.503 | 0.497 | 0.538 | 0.458 |
| Metadata SVM | 0.601 | 0.497 | 0.538 | 0.503 |

Canh bao:

- Metadata baseline co the hoc bias ky thuat cua video nhu file size, frame count, resolution.
- Chi dung lam moc tham khao, khong xem la model deception chinh.
- Metadata SVM sau khi sua split dat test AUC-ROC `0.601`; neu visual model khong vuot moc nay, can phan tich ky kha nang metadata dang hoc technical bias theo source video.
- Random baseline co AUC cao bat thuong do test set nho va mot seed ngau nhien; khong dien giai random baseline nhu model co nang luc.

## 5. Giai doan 3: Face crop bang MediaPipe

Trang thai: **da chay xong lan 1**.

Muc tieu:

- Cat khuon mat tu moi video.
- Luu video/frames face crop vao `data/processed/faces/real_life/`.
- Tao metadata moi tro den face-crop files.

Thiet lap:

- Chay trong `ai_env`.
- Dung MediaPipe Face Detection hoac Face Mesh.
- Output nen resize ve `128x128` de train co random crop xuong `112x112`.

Can luu:

- `data/processed/faces/real_life/<video_id>.mp4`
- `outputs/metrics/real_life_face_check/face_metadata.csv`
- ti le frame detect duoc face cho tung video
- sample preview 20-30 video de kiem tra chat luong

Ket qua lan 1:

- 121/121 video crop thanh cong.
- Mean face crop rate: `0.9975`.
- Median face crop rate: `1.0`.
- Video duoi 70%: `0`.
- Video duoi 95%: `2`.
- Contact sheet preview: `outputs/figures/real_life_face_previews/contact_sheet.jpg`.
- Can xem thu cong: `trial_truth_041.mp4` crop vao tay o mot so frame; `trial_lie_053.mp4` chat luong kem va co 2 khuon mat.

Rui ro:

- Video courtroom co nhieu goc quay, mat nho, blur, occlusion.
- Neu nhieu video co face crop rate < 70%, chay song song full-frame baseline de khong bi chan tien do.
- Neu face detection kem, fallback tam thoi:
  - frame-based CNN tren full frame
  - center crop/body crop
  - chi lay frame co face confidence cao

## 6. Giai doan 4: Visual baselines

Thu tu uu tien:

1. Frame-based CNN baseline.
2. RGB video model nhe.
3. 3D-CNN pretrained neu frame baseline da on.

Frame-based CNN:

- Model: ResNet18 pretrained ImageNet.
- Input: face crop hoac full frame neu face crop chua on.
- Lay 8-16 frame moi video.
- Gop frame predictions thanh video-level probability bang average probability.

RGB video model:

- Input: clip 8 hoac 16 frames, `112x112`.
- RTX 3060 12GB:
  - batch size goi y: 4-12 voi clip 16
  - batch size co the cao hon voi clip 8
  - bat AMP
  - gradient accumulation `2-4` neu can

3D-CNN:

- Uu tien `r3d_18` hoac `mc3_18` pretrained Kinetics-400.
- Khong train from scratch.
- Warm-up: freeze backbone, train head 2-5 epoch.
- Fine-tune: unfreeze mot phan/toan bo backbone voi LR nho.

Sanity checks bat buoc:

- Overfit 5-10 video nho truoc khi train that.
- Kiem tra label mapping `0=truthful`, `1=deceptive`.
- Kiem tra tensor shape `[B, C, T, H, W]`.
- Kiem tra DataLoader khong doc nham split.
- Kiem tra video-level aggregation gom dung clip theo `video_id`.

## 7. Giai doan 5: Audio va transcript baseline

Real-life dataset co transcript day du, nen baseline textual/audio dang gia tri hon optical flow.

Uu tien:

1. Transcript baseline:
   - TF-IDF + Logistic Regression/SVM.
   - Bao cao rieng text-only.
2. Audio baseline:
   - Trich xuat MFCC, energy, pitch neu kip.
   - Train SVM/MLP nho.
3. Late fusion:
   - Average probability cua visual + text/audio.

Canh bao:

- Audio/text co the hoc shortcut theo speaker/trial/source.
- Van phai group-disjoint.
- Ghi ro cac baseline nay la exploratory neu chua kiem soat confounders day du.

## 8. Giai doan 6: Optical flow

Do dataset nho va video courtroom khong dam bao micro-expression ro, optical flow de sau.

Chi lam khi:

- Face crop on.
- Frame CNN va RGB video baseline da co ket qua.
- Con du thoi gian.

Neu lam:

- Farneback truoc.
- TV-L1 chi thu neu can.
- Overlap clip chi dung cho train.
- Val/test dung non-overlapping clips, aggregate ve video-level truoc khi tinh metric.

## 9. Danh gia va bao cao

Metric can bao cao:

- AUC-ROC.
- AUC-PR.
- EER.
- Accuracy.
- Macro F1.
- Precision/Recall/F1 cho lop deceptive.
- Confusion matrix.

Confidence interval:

- Bootstrap 95% CI tren test video predictions.
- Resample test videos 1000 lan.
- Bao cao CI cho AUC-ROC, Macro F1, Lie Recall neu kip.

Error analysis can co:

- Loi do face crop sai, mat nho, blur hoac occlusion.
- Loi do video qua ngan hoac chat luong hinh anh kem.
- Loi do group/trial domain shift.
- Kiem tra technical bias: duration, resolution, file size, frame count, source video co tuong quan voi nhan hay khong.
- Neu visual model thua metadata baseline, phai phan tich kha nang metadata baseline hoc shortcut.
- ResNet18 frame baseline trung binh 3 seeds vuot nhe Metadata SVM ve AUC-ROC (`0.622` vs `0.601`), nhung variance con lon nen can phan tich threshold, crop quality va shortcut cua metadata baseline.
- Error analysis hien tai cho thay loi tap trung theo group/trial (`Andrea Sneiderman`, `Crystal Mangum`, `Jamie Hood`) hon la do face crop, vi test error co mean `face_crop_rate = 1.0`.

Expected result table:

| Model | Modality | Split | AUC-ROC | Macro F1 | Lie Recall | EER |
|---|---|---|---:|---:|---:|---:|
| Majority baseline | Label only | group-disjoint | 0.500 | 0.314 | 0.000 | 0.500 |
| Random baseline | Label only | group-disjoint | 0.717 | 0.708 | 0.615 | 0.283 |
| Metadata LR | Metadata | group-disjoint | 0.503 | 0.497 | 0.538 | 0.458 |
| Metadata SVM | Metadata | group-disjoint | 0.601 | 0.497 | 0.538 | 0.503 |
| ResNet18 frame | Visual | group-disjoint, 3 seeds | 0.622 +/- 0.086 | 0.571 +/- 0.075 | 0.487 +/- 0.247 | 0.402 +/- 0.097 |
| R3D-18 RGB | Visual | group-disjoint | 0.476 | 0.486 | 0.308 | 0.542 |
| Transcript TF-IDF LR | Text | group-disjoint | 0.643 | 0.571 | 0.385 | 0.458 |
| Transcript TF-IDF SVM | Text | group-disjoint | 0.622 | 0.344 | 0.077 | 0.336 |
| Late fusion | Visual ensemble + Text LR | group-disjoint | 0.678 | 0.619 | 0.462 | 0.252 |

Negative result policy:

- Neu RGB video model khong hon frame-based CNN, van bao cao.
- Neu optical flow khong cai thien, van bao cao neu da lam dung protocol.
- Khong loai bo ket qua am neu thiet ke thuc nghiem hop ly.

## 10. MVP bao cao

Bat buoc:

- [x] Dataset moi hop le va giai nen duoc.
- [x] Metadata sach.
- [x] Group-disjoint split.
- [x] Majority/random baseline.
- [x] Metadata baseline.
- [x] Face crop hoac full-frame visual input.
- [x] Frame-based CNN baseline.
- [x] Video-level metrics tren validation/test.
- [x] Error analysis.
- [x] Config/version/seed de tai lap.

Nen co:

- [ ] Grad-CAM hoac visualization cho 5 video.
- [x] Transcript baseline.
- [ ] Audio baseline.
- [x] RGB 3D-CNN pretrained.
- [ ] Bootstrap confidence interval.
- [x] Late fusion baseline.
- [x] Error analysis ban dau.
- [x] Final result table.
- [x] Threshold tuning tren validation.
- [x] So sanh voi paper goc Real-life Trial Dataset.
- [x] Visualization examples.
- [x] Artifact index / cleanup.

## 11. Thu tu thuc hien tiep theo

1. Viet bao cao chinh thuc tu `docs/report_notes.md`.
2. Chen bang tu `outputs/metrics/final_report/final_results_summary.md`.
3. Chen visualization tu `outputs/figures/report_examples/visual_examples_contact_sheet.jpg`.
4. Neu con thoi gian, them Grad-CAM that cho ResNet18 seed42.

## 12. Ket luan hien tai

Dataset moi da san sang cho bao cao: metadata, group-disjoint split, non-learning baselines, face crops, visual/text/fusion baselines, RGB 3D-CNN negative result, error analysis, final result table, threshold tuning, visualization examples, paper comparison va artifact index deu da co. Viec tiep theo nen uu tien la viet bao cao chinh thuc.
