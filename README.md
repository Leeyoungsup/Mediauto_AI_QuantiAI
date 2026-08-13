# Mediauto_AI_QuantiAI

PD-L1 stomach cell detection 실험 코드입니다. 기본 데이터 위치는
`../../data/mediautoAI_PDL1_Stomach/`이며 각 슬라이드 폴더 안의
`patch_N.jpeg`/`patch_N.json` 쌍을 읽습니다.

## 데이터 파이프라인

- 원본 patch를 RGB 1024×1024로 한 번 리사이즈하고 RAM에 `uint8`로 캐시합니다.
- 학습 시 1024 이미지에서 512×512 영역을 랜덤 크롭합니다.
- 검증 시 1024 patch를 겹치지 않는 512×512 사분면 4장으로 나눕니다.
- validation은 랜덤 slide split 대신 전체 3-slide 조합을 비교해, 모든 클래스의
  validation 포함 비율 중 최솟값이 가장 큰 조합을 고정 선택합니다. Train과
  validation 사이의 slide leakage는 허용하지 않습니다.
- crop 밖에 중심점이 있는 객체는 제외하고, 경계에 걸친 bbox는 crop에 맞춰 자릅니다.
- `class_name`이 아래 7개에 정확히 대응하는 라벨만 사용하며 대소문자는 구분하지 않습니다.

| ID | Class |
|---:|---|
| 0 | Negative Epithelial |
| 1 | Negative Lymphocyte |
| 2 | Negative Macrophage |
| 3 | Positive Epithelial |
| 4 | Positive Lymphocyte |
| 5 | Positive Macrophage |
| 6 | other |

현재 데이터 51장의 1024 RGB 캐시는 약 153 MiB를 사용합니다. 메모리 캐시를
끄려면 `IHCHierarchicalDataset(..., cache_images=False)`로 생성하면 됩니다.
사분면 validation은 Dataset index만 4배로 확장하므로 1024 캐시 메모리를
추가로 복제하지 않습니다. 예를 들어 validation 원본이 5장이면 512 타일은
20장입니다.

학습은 `Precise_PDL1_ST_hierarchical_train.ipynb`, 평가는
`Precise_PDL1_ST_hierarchical_test.ipynb`에서 실행합니다.

## 모델

- 원본 [Precise_BC_cell_scoring](https://github.com/Leeyoungsup/Precise_BC_cell_scoring)의
  일반 YOLOv11 학습 구조를 기준으로 합니다.
- `nn.yolo_v11_m(num_classes=7)`의 단일 detection head가 7개 클래스를 직접
  예측합니다. Epithelial/other gate나 계층형 분기 모델은 사용하지 않습니다.
- 학습 loss는 `util.ComputeLoss`의 Box + Classification + DFL 합입니다.
- 한 epoch에서 같은 학습 원본을 50회 사용하되, 매 접근마다 1024 캐시
  이미지에서 독립적인 512 랜덤 크롭과 augmentation을 적용합니다.
- 학습 crop의 80%는 class 1~5 가운데 하나를 먼저 선택해 해당 세포가 crop
  중앙부에 포함되도록 합니다. 풍부한 Negative Epithelial과 other는 일반
  crop 및 희소 클래스 주변 문맥에서 계속 학습됩니다.
- Classification BCE에는 이미 학습이 잘 되는 Negative Epithelial을 1로 둔
  `sqrt(reference_count / class_count)` positive weight를 적용합니다. 가중치는
  0.5~8배로 제한해 other는 낮추고 희소 클래스는 높입니다.
- 체크포인트는
  `../../model/mediautoAI_PDL1_Stomach/pdl1_flat_7class_balanced_yolov11/`에
  저장합니다. 기존 `pdl1_flat_7class_yolov11/best_model.pt`가 있으면 모델
  가중치만 warm-start하고 optimizer와 scheduler는 새로 시작합니다.
- early stopping은 사용하지 않습니다. validation은 매 epoch, 시각화와 학습
  곡선 저장은 10 epoch마다 수행합니다.
