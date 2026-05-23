# Report Outline: Micro-Expression for Lie Detection

> Bao cao hoan chinh da duoc viet tai `docs/course_report_micro_expression_lie_detection.md`.

> Pham vi thuc nghiem: tat ca mo hinh duoc train/validation/test tren DOLOS theo 3-fold. Bao cao chot ket qua tren DOLOS-only, dung clean protocol voi `faces_224_clean`, `optflow_clean` va `face_valid`.

> Cap nhat final moi nhat: ket qua chinh dung `outputs/metrics/final_report_clean_temporal_mask_soft_cross`, final method `ensemble_raw_balanced_accuracy`, AUC `65.43`, calibrated BA `60.54`, calibrated F1 Lie `61.39`.

## 0. Trang bia va tom tat

### Trang bia
- Ten bao cao: **Micro-Expression for Lie Detection**
- Mon hoc, giang vien, nhom/sinh vien, MSSV, ngay nop.

### Abstract / Tom tat
- 150-250 tu.
- Noi dung can co:
  - Bai toan: phan loai video noi that/noi doi.
  - Tin hieu: face micro-expression, facial motion/optical flow, audio.
  - Dataset: train/validation/test tren DOLOS theo 3-fold.
  - Ket qua chinh strict clean rerun: DOLOS clean best ensemble raw-BA AUC 61.97, calibrated BA 57.92.

**[Table 0.1 - Key Results Snapshot]**

| Evaluation | Method | AUC | BA | F1 Lie | Note |
| --- | --- | ---: | ---: | ---: | --- |
| DOLOS 3-fold mean | Clean final ensemble raw-BA | 61.97 | 57.92 | 57.05 | Strict clean in-domain best |
| DOLOS 3-fold mean | Clean final ensemble raw-AUC | 61.30 | 57.21 | 58.96 | Direct raw-AUC comparable rerun |
| DOLOS 3-fold mean | Clean gated logits prior-KL | 61.42 | 57.51 | 56.50 | Best single trained model |
| DOLOS 3-fold mean | Clean cross-attention AUC baseline | 60.59 | 56.70 | 48.06 | Baseline fusion model |

Nguon: `outputs/metrics/final_report_clean_cross_raw_ba/final_results_summary.md`, `outputs/metrics/final_report_clean_cross/final_results_summary.md`.

---

## 1. Gioi thieu

### 1.1 Boi canh va dong co
- Phat hien noi doi tu video la bai toan quan trong trong an ninh, phong van, phap ly va tuong tac nguoi-may.
- Micro-expression la nhung bieu hien thoang qua, kho kiem soat, co the phan anh cam xuc that.
- Tuy nhien, lie detection khong chi phu thuoc vao khuon mat; am thanh, cach noi, chuyen dong dau/mat va boi canh cung anh huong.

**[Figure 1.1 - Minh hoa bai toan]**
- De xuat noi dung: anh pipeline tu video dau vao -> crop face/audio/flow -> model -> truth/lie.
- Dat ngay sau phan dat van de.

### 1.2 Bai toan nghien cuu
- Input: mot clip video ngan.
- Output: nhan nhi phan `truth` hoac `lie`.
- Muc tieu hoc may: hoc ham `f(video, audio) -> P(lie)`.
- Rang buoc bao cao:
  - DOLOS la dataset chinh de train, validation va test.
  - Tat ca model selection, threshold calibration va ensemble weight search deu thuc hien tren cac split DOLOS.

### 1.3 Muc tieu
- Xay dung pipeline tien xu ly cho face, optical flow va audio.
- Xay dung mo hinh multimodal 3 stream.
- So sanh cac chien luoc fusion: cross-attention, gated prior-KL, prediction-level ensemble.
- Phan tich tinh nhat quan cua ket qua DOLOS theo fold, host, episode va threshold.

### 1.4 Dong gop cua bao cao
- De xuat pipeline 3-stream gom spatial, flow va audio.
- Cai tien tien xu ly face bang dominant face track va face_valid mask.
- Danh gia in-domain tren DOLOS 3-fold voi strict clean protocol.
- Phan tich loi theo host, episode va contamination heuristic.

---

## 2. Co so ly thuyet va cong trinh lien quan

### 2.1 Micro-expression trong lie detection
- Khai niem micro-expression: cuong do nho, thoi gian ngan, kho quan sat bang mat thuong.
- Lien he voi deception: cam xuc that co the ro ri qua bieu cam mat.
- Han che: micro-expression khong phai bang chung truc tiep cua noi doi; can ket hop nhieu tin hieu.

**[Figure 2.1 - Vi du micro-expression / facial action regions]**
- Co the dung anh crop face minh hoa cac vung mat, mieng, chan may.
- Neu khong dua anh ca nhan, dung diagram tu preprocessing output hoac ve schematic.

### 2.2 Deception detection bang audio-visual learning
- Visual stream: facial appearance, head pose, eye/mouth motion.
- Temporal stream: optical flow bat chuyen dong vi mo.
- Audio stream: prosody, pause, stress, cognitive load.
- Multimodal fusion: early fusion, late fusion, attention fusion, gated fusion.

### 2.3 DOLOS paper va PECL
- Tom tat DOLOS dataset va muc tieu cua bai goc.
- DOLOS paper dung parameter-efficient crossmodal learning va multi-task learning.
- Bao cao nay khong dung annotation multi-task nhu bai goc, tap trung vao pipeline kha thi trong mon hoc.

**[Table 2.1 - Ket qua tham chieu tu DOLOS paper]**

| Source | Method | ACC | F1 Lie | AUC |
| --- | --- | ---: | ---: | ---: |
| DOLOS paper | Visual | 61.44 | 69.42 | 58.89 |
| DOLOS paper | Audio | 59.19 | 73.46 | 52.54 |
| DOLOS paper | Concatenation | 61.62 | 70.20 | 60.50 |
| DOLOS paper | PAVF | 64.75 | 71.20 | 62.71 |
| DOLOS paper | PAVF + Multi-task | 66.84 | 73.35 | 64.58 |

Nguon: `outputs/metrics/final_report_clean_cross_raw_ba/dolos_paper_comparison.csv`.

---

## 3. Dataset va chia tap

### 3.0 Data usage compliance
- DOLOS duoc ROSE Lab phat hanh cho **academic research only** va mien phi cho nha nghien cuu thuoc educational/research institutes voi muc dich **non-commercial**.
- Bao cao mon hoc nay chi su dung DOLOS cho muc dich hoc thuat, khong thuong mai.
- Khong dua kem, upload, hoac phan phoi lai:
  - video goc
  - file annotation goc
  - face crops/processed frames
  - optical-flow files
  - bat ky processed dataset nao co the duoc xem la dataset phai sinh.
- Neu can minh hoa hinh anh trong bao cao:
  - uu tien dung diagram/schematic tu chinh pipeline.
  - neu dung frame/crop that, chi dung mot so hinh nho trong pham vi bao cao lop hoc, co citation ro rang, va khong dua public repository.
- Khi nop code/artifact:
  - chi nop source code, config, bang metric va do thi tong hop.
  - khong nop `data/DOLOS`, `data/raw/dolos`, `data/processed/faces_224_clean/dolos`, `data/processed/optflow_clean/dolos`.
- Tat ca cong bo/bao cao dua tren DOLOS phai cite paper DOLOS ICCV 2023.

**[Table 3.0 - DOLOS usage compliance checklist]**

| Requirement | Compliance in this report |
| --- | --- |
| Academic/non-commercial use only | Course report, no commercial use |
| No redistribution | Do not attach raw videos, annotations, crops, flow files, or processed dataset |
| No derived dataset release | Only report aggregate metrics/plots; processed files stay local |
| YouTube copyright limitation | Do not redistribute gameshow video frames/clips publicly |
| Required citation | Cite Guo et al., ICCV 2023 |

Nguon quy dinh: ROSE Lab DOLOS Terms and Conditions, `https://rose1.ntu.edu.sg/dataset/DOLOS/`.

### 3.1 DOLOS dataset
- Nguon: video chuong trinh "Would I Lie To You?"
- Don vi mau: clip ngan co nhan truth/lie.
- DOLOS duoc dung cho:
  - Train
  - Validation
  - Test
  - Model selection
  - Threshold calibration
  - Ensemble weight search

**[Table 3.1 - Thong ke DOLOS theo fold]**
- Cot de xuat:
  - Fold
  - Train clips
  - Val clips
  - Test clips
  - Truth/Lie ratio
  - So host/episode
- Nguon: `data/processed/splits/dolos/cache_filtered/fold*_train.csv`, `fold*_val.csv`, `fold*_test.csv`.

**[Graph 3.1 - Label distribution tren DOLOS]**
- Bar chart truth vs lie cho train/val/test theo tung fold.
- Dat sau Table 3.1.

### 3.2 Protocol DOLOS-only
- Tat ca ket qua chinh su dung DOLOS theo 3-fold.
- Train split: hoc tham so mo hinh.
- Validation split: chon checkpoint, calibrate threshold, grid search ensemble weight.
- Test split: bao cao ket qua chinh.
- Khong bo sung dataset ngoai DOLOS vao ket qua chinh.

**[Table 3.2 - DOLOS evaluation protocol]**

| Stage | Data | Purpose |
| --- | --- | --- |
| Training | DOLOS train folds | Learn model parameters |
| Validation | DOLOS val folds | Checkpoint selection, threshold calibration, ensemble weight search |
| Test | DOLOS test folds | Main reported result |

### 3.3 Bien thien noi bo DOLOS
- DOLOS co nhieu host, episode, shot composition va dieu kien anh sang san khau.
- Day la nguon variance quan trong ngay ca trong in-domain evaluation.
- Bao cao vi vay phan tich them error theo host/episode thay vi chi bao cao mean 3-fold.

**[Figure 3.1 - DOLOS internal variation schematic]**
- Minh hoa cac nguon bien thien trong DOLOS:
  - host/person identity
  - episode/stage setting
  - face crop style
  - audio environment
  - clip length / shot composition

---

## 4. Tien xu ly du lieu

### 4.1 Tong quan pipeline

**[Figure 4.1 - Preprocessing pipeline]**

Video dau vao -> extract audio -> face detection/crop -> dominant face track -> face_valid mask -> optical flow -> dataset windows.

Dat Figure 4.1 o dau chuong 4.

### 4.2 Audio extraction
- Script: `src/data/extract_audio.py`.
- Chuyen video thanh WAV mono 16 kHz.
- Audio duoc cat theo window 2s khi train/evaluate.

**[Table 4.1 - Audio preprocessing setting]**

| Item | Value |
| --- | --- |
| Sample rate | 16 kHz |
| Channel | Mono |
| Window length | 2s |
| Missing audio handling | Zero waveform fallback |

### 4.3 Face preprocessing
- Script: `src/data/preprocess_faces_mediapipe.py`.
- Dung MediaPipe Face Mesh de phat hien/crop face 224x224.
- Clean pipeline:
  - Bo/lam giam anh huong dau clip neu face chua on dinh.
  - Face-track clustering/chon track xuat hien lau nhat sau doan dau.
  - Neu clip/window khong chac chan thi `face_valid=false`.
- Output:
  - `data/processed/faces_224_clean/dolos`
  - `face_valid/*.csv`

**[Figure 4.2 - Face preprocessing example]**
- 3 panel:
  - original frame
  - detected face crop
  - face_valid timeline / invalid windows.

**[Table 4.2 - Face validity fields]**

| Field | Meaning |
| --- | --- |
| frame_index | Index cua frame |
| face_valid | 1 neu face duoc xem la hop le |
| face_valid_ratio | Ty le hop le trong window |
| clip_face_valid | Danh gia tin cay cua clip |

### 4.4 Optical flow extraction
- Script: `src/data/extract_optical_flow.py`.
- Optical flow capture motion cua face/head.
- Luu thanh `.npz`, 2 channels, resize 224x224.
- Neu missing flow thi dataset fallback zero tensor.

**[Figure 4.3 - Optical flow visualization]**
- Minh hoa 1 frame RGB, flow-x, flow-y hoac HSV flow.

### 4.5 Windowing strategy
- Moi clip duoc chia thanh cac window 2s.
- Train: lay nhieu random windows/clip.
- Validation/test: sliding windows, aggregate score theo clip.

**[Figure 4.4 - Clip-to-window evaluation]**
- Diagram: clip -> windows -> scores -> mean score per clip.

---

## 5. Phuong phap de xuat

### 5.1 Tong quan kien truc 3 stream

**[Figure 5.1 - Three-stream architecture]**

Nen ve gom:
- Spatial stream: face crop RGB -> ViT micro-expression -> feature sequence.
- Flow stream: optical flow -> CNN/ResNet stream -> feature sequence.
- Audio stream: waveform -> Wav2Vec2 -> temporal attention pooling.
- Fusion block -> logits truth/lie.

### 5.2 Spatial stream
- Input: `T x 3 x 224 x 224` face frames.
- Backbone: ViT pretrained cho facial expression, frozen.
- Projection ve hidden dimension dung chung.
- Muc tieu: bat appearance va micro-expression static cues.

**[Table 5.1 - Spatial stream configuration]**

| Component | Setting |
| --- | --- |
| Backbone | `LaurenGurgiolo/vit-micro-facial-expressions` |
| Input | 224x224 face RGB |
| Freeze | True |
| Output dim | 256 |

### 5.3 Flow stream
- Input: optical flow 2-channel.
- Muc tieu: bat temporal motion cua mat, mieng, dau.
- Flow co the giup khi appearance static khong du.

**[Table 5.2 - Flow stream configuration]**

| Component | Setting |
| --- | --- |
| Input | 2-channel optical flow |
| Size | 224x224 |
| Output dim | 256 |
| Missing flow | Zero tensor fallback |

### 5.4 Audio stream
- Input: waveform 16 kHz.
- Backbone: Wav2Vec2-base frozen.
- Temporal pooling/attention de tao audio tokens.
- Muc tieu: bat prosody/cognitive-load cues.

**[Table 5.3 - Audio stream configuration]**

| Component | Setting |
| --- | --- |
| Backbone | `facebook/wav2vec2-base` |
| Input | 16 kHz waveform |
| Freeze | True |
| Output dim | 256 |

### 5.5 Fusion strategy 1: Cross-attention
- Visual token = concat spatial + flow.
- Audio query attends visual key/value.
- Residual + LayerNorm de on dinh scale.
- Output qua BiLSTM/head de tao logits.

**[Figure 5.2 - Cross-attention fusion block]**
- Ve Q tu audio, K/V tu visual, attention output, residual, LayerNorm, classifier.

### 5.6 Fusion strategy 2: Gated prior-KL logit fusion
- Moi stream co classifier rieng:
  - `spatial_head -> logits_s`
  - `flow_head -> logits_f`
  - `audio_head -> logits_a`
- Gate network sinh weight:
  - `gate = softmax(MLP([pooled_s, pooled_f, pooled_a]))`
- Final logits:
  - `logits = gate_s*logits_s + gate_f*logits_f + gate_a*logits_a`
- Loss:
  - CE final logits
  - auxiliary CE cho tung stream
  - KL prior de huong gate khong phu thuoc qua manh vao spatial.

**[Figure 5.3 - Gated logit fusion block]**
- Ve 3 logits rieng, gate softmax va weighted sum.

### 5.7 Prediction-level ensemble
- Sau khi co prediction cua `cross_attention_auc` va `gated_prior_kl`, grid-search weight tren DOLOS validation.
- Ap dung weight va threshold do vao DOLOS test.
- Final DOLOS method: `ensemble_raw_balanced_accuracy`.

**[Figure 5.4 - Prediction-level ensemble]**
- Model A score + Model B score -> weighted score -> threshold.

---

## 6. Thiet lap thuc nghiem

### 6.1 Nguyen tac thuc nghiem
- Train/validation/test: DOLOS only.
- Model selection: validation AUC.
- Threshold calibration: DOLOS validation balanced accuracy.
- Ensemble weight search: DOLOS validation.
- Tat ca bao cao metric chinh deu lay tu DOLOS test folds.

**[Table 6.1 - Experimental protocol]**

| Stage | Dataset | Used for |
| --- | --- | --- |
| Train | DOLOS train folds | Learn model parameters |
| Validation | DOLOS val folds | Select checkpoint, threshold, ensemble weights |
| Test | DOLOS test folds | Main in-domain evaluation |

### 6.2 Training setting
- Optimizer: AdamW.
- Learning rate head: 1e-3.
- Scheduler: cosine warmup.
- Early stopping: patience 10.
- AMP: enabled.
- Seed: 42.
- Hardware: RTX 3060 12GB.

**[Table 6.2 - Hyperparameters]**

| Hyperparameter | Value |
| --- | --- |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Max epochs | 50 |
| Early stopping metric | Val AUC-ROC |
| Window length | 2s |
| Frames per clip/window | 16 |
| Image size | 224 |

Nguon: `configs/retrain_clean_dolos_three_stream_gated_logits_prior_kl.yaml`, `configs/retrain_clean_dolos_three_stream_auc.yaml`.

### 6.3 Metrics
- Accuracy.
- Balanced accuracy (BA): quan trong khi label imbalance/threshold sensitivity.
- F1 Lie.
- Macro-F1.
- AUC-ROC.
- AUC-PR.

**[Table 6.3 - Metric definitions]**
- Dat cong thuc ngan cho Precision, Recall, F1, BA, AUC.

---

## 7. Ket qua tren DOLOS

### 7.1 Ket qua chinh 3-fold

**[Table 7.1 - DOLOS 3-fold mean results]**

| Method | AUC | BA@0.5 | Calibrated BA | Calibrated F1 Lie | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean cross-attention AUC baseline | 60.59 | 55.24 | 56.70 | 48.06 | 53.02 |
| Clean gated logits prior-KL | 61.42 | 58.39 | 57.51 | 56.50 | 57.08 |
| Clean final ensemble raw-AUC | 61.30 | 56.63 | 57.21 | 58.96 | 57.08 |
| Clean final ensemble raw-BA | 61.97 | 58.30 | 57.92 | 57.05 | 57.61 |

Nguon: `outputs/metrics/final_report_clean_cross/final_results_summary.md`, `outputs/metrics/final_report_clean_cross_raw_ba/final_results_summary.md`.

**[Graph 7.1 - Bar chart DOLOS 3-fold mean]**
- X-axis: method.
- Y-axis: AUC, calibrated BA, F1 Lie.
- Dat ngay sau Table 7.1.

### 7.2 Ket qua final ensemble theo fold

**[Table 7.2 - Final ensemble per fold]**

| Fold | Ensemble weights | Threshold | AUC | BA@0.5 | Calibrated BA | F1 Lie |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fold1 | cross=0.00; gated=1.00 | 0.5165 | 61.20 | 57.18 | 55.26 | 55.98 |
| fold2 | cross=0.21; gated=0.79 | 0.5147 | 67.01 | 62.50 | 62.92 | 59.87 |
| fold3 | cross=0.26; gated=0.74 | 0.4939 | 57.70 | 55.24 | 55.57 | 55.29 |

**[Graph 7.2 - Per-fold AUC/BA line chart]**
- Moi fold la 1 diem.
- 2 duong: AUC va calibrated BA.

### 7.3 So sanh voi DOLOS paper

**[Table 7.3 - Ours vs DOLOS paper]**

| Source | Method | ACC | F1 Lie | AUC | BA |
| --- | --- | ---: | ---: | ---: | ---: |
| Paper | Visual | 61.44 | 69.42 | 58.89 | - |
| Paper | Audio | 59.19 | 73.46 | 52.54 | - |
| Paper | PAVF + Multi-task | 66.84 | 73.35 | 64.58 | - |
| Ours clean | Cross-attention AUC baseline | 56.15 | 48.06 | 60.59 | 56.70 |
| Ours clean | Gated logits prior-KL | 57.26 | 56.50 | 61.42 | 57.51 |
| Ours clean | Final ensemble raw-AUC | 57.30 | 58.96 | 61.30 | 57.21 |
| Ours clean | Final ensemble raw-BA | 57.75 | 57.05 | 61.97 | 57.92 |

Nguon: `outputs/metrics/final_report_clean_cross_raw_ba/dolos_paper_comparison.csv`, `outputs/metrics/final_report_clean_cross_raw_ba/final_results_summary.md`.

**[Graph 7.3 - AUC comparison with DOLOS paper]**
- Bar chart AUC cua paper methods va ours.
- Caption nen nhan manh: paper co PAVF + multi-task, bao cao nay khong dung multi-task annotations.

### 7.4 Ablation theo stream

**[Table 7.4 - Fold3 stream ablation]**
- Lay tu `outputs/metrics/fold3_stream_ablation/fold3_stream_ablation_summary.md`.
- Nen co cac dong:
  - Spatial only
  - Flow only
  - Audio only
  - Three-stream/cross-attention
  - Gated fusion neu co

**[Graph 7.4 - Stream ablation bar chart]**
- X-axis: stream/model.
- Y-axis: Val AUC, Test AUC, calibrated BA.

**[Graph 7.5 - Final ensemble score distribution]**
- Histogram/density cua `P(lie)` tren truth vs lie.
- Dung de quan sat muc do overlap giua hai lop va vi tri threshold.

**[Graph 7.6 - Final ensemble ROC curve]**
- ROC all-fold va per-fold.
- Dat canh bang AUC final de minh hoa trade-off TPR/FPR.

**[Graph 7.7 - Final ensemble precision-recall curve]**
- PR curve cho lop lie.
- Bao gom baseline lie prior va Average Precision.

**[Graph 7.8 - Threshold sweep]**
- X-axis: decision threshold.
- Y-axis: BA va F1 Lie.
- Dung de giai thich vi sao can calibrated threshold thay vi mac dinh 0.5.

### 7.5 Thao luan ket qua DOLOS
- Trong strict clean rerun, final ensemble raw-BA dat AUC cao nhat: 61.97.
- Gated prior-KL la single model tot nhat tren DOLOS clean rerun ve AUC: 61.42.
- Ensemble raw-BA cai thien nho so voi gated prior-KL ve AUC/BA, nhung clean rerun thap hon mixed-cache final cu.
- F1 Lie cua ours thap hon paper do:
  - thresholding khac
  - khong dung multi-task supervision
  - chi dung single seed
  - data preprocessing va face contamination anh huong.

---

## 8. Error analysis tren DOLOS

### 8.1 Loi theo host

**[Table 8.1 - Worst hosts by calibrated BA]**

| Host | N | BA | Accuracy | AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SB | 183 | 50.86 | 50.82 | 53.51 | 42 | 40 | 50 | 51 |
| BRI | 269 | 54.05 | 56.88 | 60.14 | 111 | 47 | 69 | 42 |
| LS | 374 | 58.12 | 58.02 | 59.14 | 98 | 68 | 89 | 119 |

Nguon: `outputs/metrics/final_report_clean_cross_raw_ba/final_error_by_host.csv`.

**[Graph 8.1 - Error/BA by host]**
- Bar chart BA theo host.
- Highlight host SB co BA thap.

**[Graph 8.2 - Score distribution by host]**
- Boxplot `P(lie)` theo host va nhan truth/lie.
- Dung de kiem tra host nao co score bias hoac overlap cao.

### 8.2 Loi theo episode

**[Table 8.2 - Worst episodes]**
- Lay top 10-12 episode tu `outputs/metrics/final_report_clean_cross_raw_ba/final_error_by_episode.csv`.

**[Graph 8.3 - Episode error heatmap]**
- X-axis: host.
- Y-axis: episode.
- Color: BA hoac error rate.

**[Graph 8.4 - Prediction outcome counts]**
- Bar chart TN, TP, FP, FN cua final ensemble.
- Dung de nhin nhanh model dang loi nhieu ve false positive hay false negative.

### 8.3 Face contamination

**[Table 8.3 - Contamination heuristic]**

| Suspect contamination | N | BA | Accuracy | AUC | Error rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| False | 180 | 60.07 | 60.00 | 61.07 | 40.00 |
| True | 1250 | 57.59 | 57.36 | 61.46 | 42.64 |

Nguon: `outputs/metrics/final_report_clean_cross_raw_ba/final_error_by_contamination.csv`.

**[Graph 8.5 - Contamination metric comparison]**
- So sanh BA, AUC va error rate giua clip clean heuristic va suspect contamination.
- Dung de chung minh preprocessing/face tracking anh huong den ket qua.

**[Figure 8.1 - Vi du contamination]**
- Uu tien schematic/metadata timeline, khong dua raw frame/crop DOLOS vao ban nop public.
- Neu can minh hoa trong lop, chi dung anh nho noi bo va citation ro rang.

### 8.4 Confusion matrix

**[Figure 8.2 - Confusion matrices per fold]**
- 3 confusion matrix cho fold1/fold2/fold3 final ensemble.
- Nguon: `outputs/metrics/final_report_clean_cross_raw_ba/final_per_fold_metrics.csv`.

### 8.5 Nhan xet
- Loi tap trung theo host/episode cho thay dataset shift noi bo DOLOS.
- Contamination lam giam BA va tang error rate.
- Can cai tien face tracking neu tiep tuc nghien cuu.

---

## 9. Thao luan tong hop

### 9.1 Cau hoi nghien cuu 1: multimodal co giup DOLOS khong?
- Co, final ensemble vuot cross-attention va gated single model ve AUC/BA.
- Nhung do loi/variance, muc cai thien khong lon.

### 9.2 Cau hoi nghien cuu 2: micro-expression co du khong?
- Khong nen ket luan chi micro-expression la du.
- Spatial/face cues co ich, nhung flow/audio va fusion cung dong vai tro.
- Face preprocessing co anh huong lon.

### 9.3 Cau hoi nghien cuu 3: mo hinh co hoc tin hieu on dinh tren DOLOS khong?
- Tren DOLOS strict clean: gated prior-KL va final ensemble raw-BA tot nhat.
- Ket qua co tin hieu phan biet nhung con yeu, chenh lech giua cac fold/host/episode van ro.
- Ket luan: mo hinh hoc duoc mot phan tin hieu in-domain, nhung chua the khang dinh da hoc duoc cue micro-expression/deception ben vung ngoai DOLOS.

### 9.4 So sanh voi bai goc
- Strict clean final ensemble raw-BA AUC 61.97, thap hon PAVF + Multi-task AUC 64.58.
- Accuracy/F1 Lie cua ours thap hon paper.
- Ly do kha di:
  - khong dung multi-task labels
  - backbone/fusion khac
  - threshold strategy khac
  - single seed
  - preprocessing contamination

---

## 10. Ket luan va huong phat trien

### 10.1 Ket luan
- Da xay dung pipeline 3-stream cho lie detection tu video.
- Ket qua DOLOS:
  - Strict clean final ensemble raw-BA AUC 61.97, BA 57.92.
  - Thap hon AUC cua PAVF + Multi-task trong DOLOS paper, F1/ACC con thap.
  - Ket qua nay cho thay pipeline co tin hieu in-domain, nhung van thap hon PECL/PAVF + multi-task cua bai DOLOS goc.

### 10.2 Han che
- Single seed.
- Face contamination va face tracking chua hoan hao.
- Chua dung transcript/text modality.
- Chua dung multi-task annotations nhu bai DOLOS goc.

### 10.3 Huong phat trien
- Hoan thien face-track clustering va visual quality mask.
- Them text/transcript stream.
- Multi-seed va confidence interval.
- Danh gia kha nang generalization bang mot protocol ngoai mien rieng neu co them thoi gian.
- Self-supervised pretraining tren unlabeled face videos.

---

## 11. Tai lieu tham khao

1. Guo et al. "Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient Crossmodal Learning." ICCV 2023.
2. Ekman. "Telling Lies." 1985.
3. Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
4. Baevski et al. "wav2vec 2.0." NeurIPS 2020.
5. He et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
6. Lugaresi et al. "MediaPipe: A Framework for Building Perception Pipelines." 2019.

---

## 12. Phu luc de xuat

### Appendix A. Configs
- `configs/retrain_clean_dolos_three_stream_gated_logits_prior_kl.yaml`
- `configs/retrain_clean_dolos_three_stream_auc.yaml`

### Appendix B. Main DOLOS scripts
- `src/data/preprocess_faces_mediapipe.py`
- `src/data/extract_audio.py`
- `src/data/extract_optical_flow.py`
- `src/train_multimodal.py`
- `src/evaluate_prediction_ensemble.py`
- `src/report_final_ensemble.py`
- `src/report_figures.py`

### Appendix C. Artifact index

| Artifact | Usage in report |
| --- | --- |
| `outputs/metrics/final_report_clean_cross_raw_ba/final_results_summary.md` | Section 7 clean raw-BA final |
| `outputs/metrics/final_report_clean_cross/final_results_summary.md` | Section 7 clean raw-AUC comparable report |
| `outputs/metrics/final_report_clean_cross_raw_ba/dolos_paper_comparison.csv` | Section 7.3 |
| `outputs/metrics/final_report_clean_cross_raw_ba/final_error_analysis.md` | Section 8 |
| `outputs/metrics/final_report_clean_cross_raw_ba/final_error_by_host.csv` | Section 8.1 |
| `outputs/metrics/final_report_clean_cross_raw_ba/final_error_by_episode.csv` | Section 8.2 |
| `outputs/metrics/final_report_clean_cross_raw_ba/final_error_by_contamination.csv` | Section 8.3 |
| `outputs/metrics/face_contamination_audit_clean/face_contamination_audit.md` | Section 8.3 clean contamination audit |
| `outputs/metrics/prediction_level_ensemble_retrain_clean_cross_final/prediction_level_ensemble_results.md` | Section 7 |

---

## 13. Checklist bang bieu, hinh anh, do thi

### Tables
- Table 0.1: Key results snapshot.
- Table 2.1: DOLOS paper reference results.
- Table 3.1: DOLOS fold statistics.
- Table 3.2: DOLOS evaluation protocol.
- Table 4.1: Audio preprocessing setting.
- Table 4.2: Face validity fields.
- Table 5.1-5.3: Stream configurations.
- Table 6.1: Experimental protocol.
- Table 6.2: Hyperparameters.
- Table 6.3: Metric definitions.
- Table 7.1: DOLOS 3-fold mean results.
- Table 7.2: Final ensemble per fold.
- Table 7.3: Ours vs DOLOS paper.
- Table 7.4: Stream ablation.
- Table 8.1: Error by host.
- Table 8.2: Error by episode.
- Table 8.3: Contamination analysis.

### Figures
- Figure 1.1: Problem pipeline.
- Figure 2.1: Micro-expression/facial regions.
- Figure 3.1: DOLOS internal variation.
- Figure 4.1: Preprocessing pipeline.
- Figure 4.2: Face crop and face_valid example.
- Figure 4.3: Optical flow visualization.
- Figure 4.4: Clip-to-window aggregation.
- Figure 5.1: Three-stream architecture.
- Figure 5.2: Cross-attention block.
- Figure 5.3: Gated logit fusion block.
- Figure 5.4: Prediction-level ensemble.
- Figure 8.1: Contamination examples.
- Figure 8.2: Confusion matrices per fold.

### Graphs
- Graph 3.1: DOLOS label distribution.
- Graph 7.1: DOLOS method comparison.
- Graph 7.2: Per-fold AUC/BA.
- Graph 7.3: Ours vs DOLOS paper AUC.
- Graph 7.4: Stream ablation.
- Graph 7.5: Final ensemble score distribution.
- Graph 7.6: Final ensemble ROC curve.
- Graph 7.7: Final ensemble precision-recall curve.
- Graph 7.8: Threshold sweep.
- Graph 8.1: BA/error by host.
- Graph 8.2: Score distribution by host.
- Graph 8.3: Episode error heatmap.
- Graph 8.4: Prediction outcome counts.
- Graph 8.5: Contamination metric comparison.

---

## 14. Generated figure files

Tat ca file da duoc sinh tu `src/report_figures.py`. Moi hinh co ca `.png` va `.svg` trong `docs/figures/report/`.

| Report item | Generated file |
| --- | --- |
| Figure 1.1 | `docs/figures/report/fig01_problem_pipeline.png` |
| Figure 2.1 | `docs/figures/report/fig02_micro_expression_regions.png` |
| Figure 3.1 | `docs/figures/report/fig03_dolos_internal_variation.png` |
| Figure 4.1 | `docs/figures/report/fig04_preprocessing_pipeline.png` |
| Figure 4.2 | `docs/figures/report/fig05_face_valid_timeline.png` |
| Figure 4.3 | `docs/figures/report/fig06_optical_flow_schematic.png` |
| Figure 4.4 | `docs/figures/report/fig07_window_aggregation.png` |
| Figure 5.1 | `docs/figures/report/fig08_three_stream_architecture.png` |
| Figure 5.2 | `docs/figures/report/fig09_cross_attention_block.png` |
| Figure 5.3 | `docs/figures/report/fig10_gated_logit_fusion.png` |
| Figure 5.4 | `docs/figures/report/fig11_prediction_level_ensemble.png` |
| Figure 8.2 | `docs/figures/report/graph08_confusion_matrices.png` |
| Graph 3.1 | `docs/figures/report/graph01_dolos_label_distribution.png` |
| Graph 7.1 | `docs/figures/report/graph02_dolos_method_comparison.png` |
| Graph 7.2 | `docs/figures/report/graph03_dolos_per_fold_auc_ba.png` |
| Graph 7.3 | `docs/figures/report/graph04_ours_vs_paper_auc.png` |
| Graph 7.4 | `docs/figures/report/graph05_stream_ablation.png` |
| Graph 7.5 | `docs/figures/report/graph11_dolos_score_distribution.png` |
| Graph 7.6 | `docs/figures/report/graph12_dolos_roc_curve.png` |
| Graph 7.7 | `docs/figures/report/graph13_dolos_pr_curve.png` |
| Graph 7.8 | `docs/figures/report/graph14_dolos_threshold_sweep.png` |
| Graph 8.1 | `docs/figures/report/graph06_error_by_host.png` |
| Graph 8.2 | `docs/figures/report/graph17_dolos_score_by_host.png` |
| Graph 8.3 | `docs/figures/report/graph07_episode_error_heatmap.png` |
| Graph 8.4 | `docs/figures/report/graph15_dolos_error_type_counts.png` |
| Graph 8.5 | `docs/figures/report/graph16_contamination_metric_comparison.png` |

Luu y compliance: cac hinh schematic khong dung raw frame/crop tu DOLOS. `fig05_face_valid_timeline` chi hien thi metadata face-valid, khong hien thi anh video.

---

## 15. De cuong slide neu can thuyet trinh

1. Title + motivation.
2. Dataset and protocol: DOLOS 3-fold strict clean.
3. Preprocessing pipeline.
4. Three-stream model.
5. Fusion strategies.
6. DOLOS results.
7. Error analysis.
8. Limitations and future work.
9. Conclusion.
