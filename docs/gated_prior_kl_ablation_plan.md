# Gated Prior-KL Ablation Plan

Muc tieu moi cua paper:

- Lay **Gated logits Prior-KL with temporal face-valid mask** lam phuong phap chinh.
- Final source hien tai: `outputs/metrics/final_report_clean_temporal_mask_soft_cross/`.
- Khi viet paper, gated Prior-KL la main model; ensemble chi nen duoc nhac nhu phan phu/secondary neu can.

## 1. Cau hoi thuc nghiem

### Q1. Tung modality dong gop gi?

Chay single-stream tren Fold1 va Fold2:

- Static/spatial face only.
- Optical flow only.
- Audio only.

Bao cao trung binh Fold1-Fold2 cho:

- AUC-ROC.
- Balanced accuracy tai nguong 0.5.
- Calibrated balanced accuracy.
- F1 Lie.
- Macro F1.

Ly do chi Fold1/Fold2:

- Theo yeu cau hien tai.
- Dung de chung minh tung luong tin hieu, khong dung thay ket qua chinh 3-fold.
- Neu reviewer hoi, co the bo sung Fold3 vao supplementary sau.

### Q2. Prior-KL co dong gop that khong?

So sanh hai mo hinh gated cung protocol:

- `gated_no_prior`: gated logits, aux stream losses, temporal face-valid mask, **khong co KL prior**.
- `gated_prior_kl`: cung cau hinh, them `gate_prior_weight=0.2` va prior `[0.10, 0.45, 0.45]`.

Metric chinh:

- Mean AUC 3-fold.
- Mean calibrated balanced accuracy 3-fold.
- Mean calibrated F1 Lie 3-fold.
- Mean/std qua 3 seed.

Dong gop can chung minh:

- Prior-KL cai thien mean hoac giam variance so voi no-prior.
- Neu mean khong tang o moi metric, claim an toan hon la Prior-KL on dinh hoa gate va cai thien metric chinh da chon.

### Q3. Ket qua co on dinh theo random seed khong?

Chay 3 seed:

- `42`
- `123`
- `2025`

Ap dung cho:

- Gated Prior-KL main model.
- Gated no-prior baseline.

Bao cao:

- Mean +- std over seeds.
- Neu du kha nang tinh toan, tinh theo tat ca fold x seed.
- Neu thoi gian/GPU han che, uu tien Fold1-Fold3 seed 42 cho no-prior truoc, sau do them seed 123/2025.

## 2. Protocol co dinh

Tat ca thuc nghiem dung:

- Dataset: DOLOS.
- Split dir: `data/processed/splits/dolos/cache_filtered`.
- Faces: `data/processed/faces_224_clean`.
- Optical flow: `data/processed/optflow_clean`.
- Face-valid root: `data/processed/faces_224_clean/dolos/face_valid`.
- Face-valid mode: `ratio`.
- Min window face-valid ratio: `0.75`.
- Window length: `2.0s`.
- Max windows per clip at evaluation: `16`.
- Checkpoint selection: validation AUC-ROC.
- Threshold calibration: validation balanced accuracy.
- Test set is only used for final reporting.

## 3. Run Matrix

### 3.1 Single-stream ablation

| Experiment | Config | Folds | Seeds | Output |
| --- | --- | --- | --- | --- |
| Static only | `configs/paper_ablation_static.yaml` | fold1, fold2 | 42 | `outputs/metrics/paper_ablation_static` |
| Flow only | `configs/paper_ablation_flow.yaml` | fold1, fold2 | 42 | `outputs/metrics/paper_ablation_flow` |
| Audio only | `configs/paper_ablation_audio.yaml` | fold1, fold2 | 42 | `outputs/metrics/paper_ablation_audio` |

### 3.2 Prior-KL contribution

| Experiment | Config | Folds | Seeds | Output |
| --- | --- | --- | --- | --- |
| Gated no-prior | `configs/paper_gated_no_prior.yaml` | fold1, fold2, fold3 | 42, 123, 2025 | `outputs/metrics/paper_gated_no_prior` |
| Gated Prior-KL | `configs/paper_gated_prior_kl.yaml` | fold1, fold2, fold3 | 42, 123, 2025 | `outputs/metrics/paper_gated_prior_kl` |

## 4. Thu tu chay uu tien

1. Chay single-stream Fold1/Fold2 de co bang modality ablation nhanh.
2. Chay gated no-prior seed 42 Fold1-Fold3 de co baseline truc tiep voi ket qua Prior-KL seed 42 hien co.
3. Chay gated Prior-KL seed 123/2025 Fold1-Fold3.
4. Chay gated no-prior seed 123/2025 Fold1-Fold3.
5. Tong hop bang final mean/std.

## 5. Lenh chay

Chay toan bo ma tran:

```bash
bash scripts/run_paper_ablation.sh
```

Tong hop ket qua:

```bash
python scripts/summarize_paper_ablation.py
```

## 6. Bang du kien dua vao paper

### Table A. Single-stream ablation, Fold1-Fold2 mean

| Method | Folds | AUC | BA@0.5 | Cal. BA | F1 Lie | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Static only | 1-2 | TBD | TBD | TBD | TBD | TBD |
| Flow only | 1-2 | TBD | TBD | TBD | TBD | TBD |
| Audio only | 1-2 | TBD | TBD | TBD | TBD | TBD |
| Gated Prior-KL | 1-2 | TBD | TBD | TBD | TBD | TBD |

### Table B. Prior-KL contribution, 3 folds x 3 seeds

| Method | Seeds | Folds | AUC mean+-std | Cal. BA mean+-std | F1 Lie mean+-std | Macro F1 mean+-std |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Gated no-prior | 3 | 3 | TBD | TBD | TBD | TBD |
| Gated Prior-KL | 3 | 3 | TBD | TBD | TBD | TBD |

## 7. Claim sau khi co ket qua

Neu Prior-KL thang ro:

> The Prior-KL regularizer improves the gated fusion model over an otherwise identical no-prior gated baseline, indicating that weak prior guidance helps stabilize modality weighting under noisy face/audio-visual evidence.

Neu Prior-KL thang o AUC nhung khong thang o F1:

> The Prior-KL regularizer improves ranking performance and/or stability, while threshold-level metrics remain sensitive to validation calibration.

Neu Prior-KL khong thang:

> Prior-KL is retained only if it improves stability or interpretability; otherwise paper should frame it as a tested regularization rather than the core contribution.
