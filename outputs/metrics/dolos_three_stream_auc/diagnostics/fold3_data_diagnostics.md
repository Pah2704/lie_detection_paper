# Fold3 Data Diagnostics

## train
- n=835, labels={'1': 432, '0': 403}
- frames missing=0, short<16=0, frame_count min/median=51/121.0
- audio missing=0, sampled windows=64, audio_abs_mean median=0.036928, audio_zero median=0.0014, zero>=0.99=0
- flow missing=0, flow_abs_mean median=0.630754, flow_zero median=0.0000, zero>=0.99=0

## val
- n=130, labels={'1': 78, '0': 52}
- frames missing=0, short<16=0, frame_count min/median=76/126.0
- audio missing=0, sampled windows=64, audio_abs_mean median=0.043670, audio_zero median=0.0010, zero>=0.99=0
- flow missing=0, flow_abs_mean median=0.627152, flow_zero median=0.0000, zero>=0.99=0

## test
- n=464, labels={'1': 245, '0': 219}
- frames missing=0, short<16=0, frame_count min/median=51/125.0
- audio missing=0, sampled windows=64, audio_abs_mean median=0.040384, audio_zero median=0.0009, zero>=0.99=0
- flow missing=0, flow_abs_mean median=0.621082, flow_zero median=0.0000, zero>=0.99=0

## Fold3 Test Score By Host

| host | n | lie | truth | score_mean | score_std | pred_lie_rate | balanced_accuracy | f1_lie | auc_roc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YW | 120 | 68 | 52 | 0.4811 | 0.0118 | 0.0750 | 0.5492 | 0.2078 | 0.6439 |
| LS | 116 | 72 | 44 | 0.4836 | 0.0138 | 0.1552 | 0.4602 | 0.2000 | 0.5748 |
| AN | 86 | 39 | 47 | 0.4823 | 0.0129 | 0.1047 | 0.4981 | 0.1667 | 0.5516 |
| BRI | 84 | 34 | 50 | 0.4796 | 0.0130 | 0.0833 | 0.5041 | 0.1463 | 0.5176 |
| SB | 58 | 32 | 26 | 0.4784 | 0.0127 | 0.0517 | 0.5469 | 0.1714 | 0.5288 |

## Fold3 Test Score By Episode

| host | episode_group_id | n | lie | truth | score_mean | score_std | pred_lie_rate | balanced_accuracy | f1_lie | auc_roc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | SB_EP42 | 17 | 8 | 9 | 0.4715 | 0.0107 | 0.0000 | 0.5000 | 0.0000 | 0.5556 |
| YW | YW_EP45 | 14 | 11 | 3 | 0.4877 | 0.0126 | 0.2143 | 0.4242 | 0.2857 | 0.5455 |
| BRI | BRI_EP63 | 13 | 2 | 11 | 0.4678 | 0.0097 | 0.0000 | 0.5000 | 0.0000 | 0.2273 |
| YW | YW_EP53 | 12 | 8 | 4 | 0.4769 | 0.0076 | 0.0000 | 0.5000 | 0.0000 | 0.6562 |
| LS | LS_EP2 | 12 | 7 | 5 | 0.4932 | 0.0121 | 0.3333 | 0.4429 | 0.3636 | 0.5143 |
| BRI | BRI_EP58 | 12 | 7 | 5 | 0.4818 | 0.0181 | 0.2500 | 0.2000 | 0.0000 | 0.0000 |
| LS | LS_EP9 | 11 | 8 | 3 | 0.4862 | 0.0150 | 0.2727 | 0.4583 | 0.3636 | 0.3750 |
| YW | YW_EP55 | 11 | 7 | 4 | 0.4865 | 0.0103 | 0.1818 | 0.6429 | 0.4444 | 0.7143 |
| BRI | BRI_EP61 | 10 | 5 | 5 | 0.4814 | 0.0121 | 0.1000 | 0.6000 | 0.3333 | 0.8400 |
| LS | LS_EP4 | 10 | 8 | 2 | 0.4773 | 0.0139 | 0.1000 | 0.5625 | 0.2222 | 0.8125 |
| BRI | BRI_EP60 | 10 | 8 | 2 | 0.4793 | 0.0072 | 0.0000 | 0.5000 | 0.0000 | 0.7500 |
| AN | AN_EP28 | 10 | 4 | 6 | 0.4753 | 0.0159 | 0.1000 | 0.6250 | 0.4000 | 0.7917 |
| LS | LS_EP72 | 10 | 7 | 3 | 0.4833 | 0.0139 | 0.1000 | 0.3333 | 0.0000 | 0.4762 |
| YW | YW_EP50 | 9 | 3 | 6 | 0.4820 | 0.0051 | 0.0000 | 0.5000 | 0.0000 | 0.2222 |
| YW | YW_EP44 | 9 | 5 | 4 | 0.4805 | 0.0129 | 0.1111 | 0.6000 | 0.3333 | 0.6500 |
| LS | LS_EP13 | 9 | 6 | 3 | 0.4854 | 0.0158 | 0.1111 | 0.5833 | 0.2857 | 0.7222 |
| BRI | BRI_EP57 | 9 | 3 | 6 | 0.4816 | 0.0082 | 0.0000 | 0.5000 | 0.0000 | 0.5556 |
| AN | AN_EP18 | 9 | 5 | 4 | 0.4777 | 0.0070 | 0.0000 | 0.5000 | 0.0000 | 0.7000 |
| LS | LS_EP12 | 9 | 8 | 1 | 0.4839 | 0.0104 | 0.1111 | 0.5625 | 0.2222 | 0.1250 |
| AN | AN_EP15 | 8 | 3 | 5 | 0.4788 | 0.0069 | 0.0000 | 0.5000 | 0.0000 | 0.3333 |
