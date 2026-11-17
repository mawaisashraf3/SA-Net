# Original manuscript tables

Transcribed from user-supplied SA.docx. **Reported manuscript findings, not results produced by this implementation.** Original typos, counts and values retained; see the [critical audit](../research-audit.md).

## Table 1

| Dataset | Resolution | Training | Validation | Testing | Lanes | Road Type | Data Source | Weather |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CULane | 1640x590 Px. | 88.9K | 9.7K | 34.7K | 2-4 | Urban, Highway | Real World | Multi-weather |
| TuSimple | 1280x720 Px. | 6.4K | 0.4K | 2.8K | 2-4 | Highway | Real World | Multi-weather |
| BDD100K | 1280x720 Px. | 70K | 10K | 20K | 2-5 | Urban, Highway | Real World | Multi-weather |
| FoggyLane | 1280x720 Px. | 3.3k | 0.35k | 2.8k | 2-4 | Urban, Highway | Real World | Multi-weather |
| ONCE-3DLanes | 1920x720 Px. | 200K | 3K | 8k | 2-6 | Urban, Highway | Real World | Multi-weather |

## Table 2

| Models | Backbone | F1@50 | F1@75  | Normal | Crowd | Dazzle | Shadow | Noline | Arrow | Curve | Cross | Night | FSP | GFlops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCNN [4] | VGG16 | 71.60 | 39.84 | 90.60 | 69.70 | 58.50 | 66.90 | 43.40 | 84.10 | 64.40 | 1990 | 66.10 | 7 | 328.4 |
| CondLane [12] | Resnet101 | 79.48 | 61.23 | 93.47 | 77.44 | 70.93 | 80.91 | 45.13 | 90.16 | 75.21 | 1201 | 74.80 | 47 | 44.8 |
| CRLNet [15] | DLA34 | 80.47 | 62.78 | 93.73 | 79.59 | 75.30 | 82.51 | 54.58 | 90.62 | 74.13 | 1155 | 75.37 | 71 | 18.4 |
| CondLSTR [5] | Resnet101 | 80.77 |    - | 94.17 | 79.90 | 75.43 | 80.99 | 55.00 | 90.97 | 76.87 | 1047 | 75.11 | 45 | 50.2 |
| CLRerNet [16] | DLA34 | 81.12 | 64.07 | 94.02 | 80.20 | 74.41 | 83.71 | 56.27 | 90.39 | 74.67 | 1161 | 76.53 | 71 | 18.4 |
| SRLane [26] | Resnet34 | 79.92 |    - | 93.85  | 78.31 | 73.58 | 82.09 | 55.47 | 90.08 | 76.09 | 1287 | 75.19 | 129 | 18.8 |
| FLAMNet [27] | Resnet34 | 80.15 |    - | 93.61 | 78.35 | 73.64 | 81.31 | 53.68 | 90.67 | 73.38 | 982 | 75.50 | 93 | 30.1 |
| BSNet [28] | Resnet34 | 79.89 |    - | 93.75 | 78.01 | 76.65 | 79.55 | 54.69 | 90.72 | 73.99 | 1445 | 75.28 | 133 | 34.5 |
| GSENet [29] | Resnet34 | 80.58 | 63.37 | 93.80 | 79.42 | 75.34 | 82.27 | 54.83 | 90.67 |     - | 1072 | 76.07 | 85 | 43.9 |
| DLNet [30] | DLA34 | 81.23 | 64.75 | 94.10 | 80.13 | 76.24 | 83.97 | 56.77 | 90.69 | 74.15 | 1098 | 76.18 | 74 | 18.8 |
| BANet [20] | DLA34 | 81.15 | 64.55 | 94.19 | 80.41 | 76.74 | 84.16 | 56.69 | 90.72 | 76.88 | 1151 | 76.64 | 92 | 23.9 |
| SA-Net (ours) | Resnet34 | 81.78  | 64.91  | 96.14 | 80.87 | 77.12 | 85.58 | 56.75 | 90.85 | 77.57 | 1140 | 77.50 | 72 | 21.5 |

## Table 3

| Models | Backbone | Acc (%) | F1 (%)  | FP (%) | FN (%) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| SCNN [4] | VGG16 | 96.53 | 95.97 | 6.17 | 1.80 | 88.9 |
| UFLDv2 [14] | Resnet34 | 95.56 | 96.22 | 3.18 | 4.23 | 90.1 |
| FOLOLane [10] | Resnet50 | 96.92 | 96.59 | 4.48 | 2.28 | 82.4 |
| CondLane [12] | Resnet101 | 95.54 | 97.24 | 2.88 | 3.50 | 91.2 |
| GANet [31] | Resnet101 | 96.46 | 97.45 | 2.63 | 2.47 | 89.5 |
| CLRNet [15] | Resnet101 | 96.83 | 97.62 | 2.57  | 2.38 | 84.7 |
| CenLaneNet | EfficientNet | 96.48 | 97.14 | 3.10 | 2.70 | 80.4 |
| DLNet [30] | DLA34 | 96.97 | 97.57 | 2.37 | 2.18 | 77.8 |
| BANet [20] | Resnet34 | 96.93 | 97.93 | 2.21 | 1.96 | 74.9 |
| SA-Net (ours) | Resnet34 | 96.98  | 97.96  | 2.14 | 1.87 | 72.5 |

## Table 4

| Models | Backbone | Acc (%) | F1 (%)  | P (%) | R (%) | IoU (%) |
| --- | --- | --- | --- | --- | --- | --- |
| MRDALane | ResNet-34 | 91.3  | 91.4  | 90.1  | 92.8  | 83.9  |
| RESA-50 [11] | ResNet-34 | 92.4 | 92.3 | 93.4  | 91.2 | 85.8 |
| ENet [32] | ResNet-101 | 91.1 | 91.8 | 95.1 | 88.8 | 85.3 |
| ENet-SAD [33] | ResNet-50 | 94.8 | 94.8 | 94.1 | 95.5 | 90.1 |
| SCNN [4] | Resnet-50 | 84.1  | 85.3  | 79.2  | 92.5 | 72.4  |
| DRN-D | ResNet  | 88.7 | 88.8 | 87.5  | 90.3  | 79.7 |
| PSPNet [34] | Resnet34 | 94.7 | 77.8 | 78.9 | 76.6 | 65.7 |
| DeepLabv3 [34] | Resnet50 | 95.3 | 76.8 | 78.2 | 75.5 | 64.9 |
| GatedCNN+Spatial [34] | DeepLabV3+ | 94.3 | 92.2 | 92.3 | 92.9 | 85.4 |
| SA-Net (ours)  | Resnet34 | 95.8 | 93.3 | 92.9 | 93.8 | 87.8 |

## Table 5

| Models | Backbone | Acc (%) | F1 (%)  | FP (%) | FN (%) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| SCNN [4] | VGG16  | 94.64 | 95.89 | 5.87 | 4.78 | 88.9 |
| LaneATT [13] | ResNet18 | 95.77  | 95.78  | 3.83  | 3.53 | 86.8 |
| LSTR [35] | ResNet34 | 94.70  | 95.58  | 5.74  | 4.81 | 95.1 |
| CondLane [12] | ResNet18 | 95.41  | 94.81  | 3.80 | 5.46 | 90.1 |
| CLRNet [15] | Resnet101 | 96.13  | 95.64  | 4.15  | 3.58 | 93.5 |
| FLAMNet [27] | ResNet18  | 96.52  | 95.65  | 3.84  | 3.09 | 95.8 |
| FLAMNet [27] | Resnet101 | 96.34  | 95.98  | 4.45  | 2.82 | 83.1 |
| DBNet [36] | Resnet34 | 95.61  | 93.72  | 3.48  | 5.38 | 81.9 |
| LEEM [24] | SwinGFFM-t | 96.63  | 95.86  | 2.77  | 4.04 | 77.8 |
| SA-Net (ours) | Resnet34 | 96.78 | 96.58 | 2.69  | 3.98 | 75.2 |

## Table 6

| Models | Backbone | Acc (%) | F1 (%)  | P (%) | R (%) | CD error(m) |
| --- | --- | --- | --- | --- | --- | --- |
| LATR [37] | ResNet-50 | 88.62 | 80.59 | 86.12 | 75.73 | 0.052 |
| 3D-LaneNet [38] | ResNet18  | 75.40 | 44.73 | 61.46 | 35.16 | 0.127 |
| Gen-LaneNet [39] | ERFNet  | 75.43 | 45.59 | 63.95 | 35.42 | 0.121 |
| SALAD [25] | ERFNet | 78.78 | 64.07 | 75.90 | 55.42 | 0.098 |
| PersFormer [40] | ResNet | 86.46 | 74.33 | 80.30 | 69.18 | 0.074 |
| WS-3D-Lane [41] | ResNet50  | 88.33 | 77.02 | 84.51 | 70.75 | 0.058 |
| Anchor3DLane [42] | ERFNet | 85.86 | 74.87 | 80.85 | 69.71 | 0.060 |
| Anchor3DLane++ [17] | ResNet18 | 88.61 | 79.55 | 82.67 | 76.67 | 0.059 |
| MambaVision-S-1K [43]   | MambaVision | 89.73 | 82.39 | 85.09 | 79.85 | 0.055 |
| SA-Net (ours)  | Resnet34 | 94.43 | 85.23 | 87.05 | 83.48 | 0.055 |

## Table 7

| Baseline | +SFM | +AMPM  | +ERA | SA-Net (ours) | F1@50 (%) | Precision | Recall | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ✓ | – | – | – | – | 80.47 | 85.00 | 76.40 | 75.37 |
| ✓ | ✓ | – | – | – | 80.68 | 85.47 | 76.40 | 76.08 |
| ✓ | ✓ | ✓ | – | – | 80.87 | 89.66 | 73.65 | 76.02 |
| ✓ | ✓ | ✓ | ✓ | – | 81.17 | 85.00 | 77.67 | 76.14 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 81.78 | 90.00 | 74.94 | 77.50 |
