# Waymo 평가 준비 상태

현재 구현은 **Waymo-v2 parquet 검증, FRONT RGB 추출, 논문 계열 split, PSNR/SSIM/LPIPS 평가 manifest 생성**까지 지원한다. 아직 Waymo에서 ArmGS를 학습시키는 전체 어댑터는 아니다.

| 항목 | 상태 |
|---|---|
| 7개 parquet component 존재/스키마 검증 | 준비됨 |
| FRONT 이미지 1600×1066(W×H), lossless PNG 추출 | 준비됨 |
| training-view reconstruction / held-out novel-view 분리 | 준비됨 |
| PSNR·SSIM·LPIPS-Alex 계산 | 준비됨 |
| camera/ego pose를 ArmGS manifest로 변환 | 미구현 |
| LiDAR + COLMAP SfM 좌표 정합·융합 초기화 | 미구현 |
| actor track/box canonicalization과 pose 보간 | 미구현 |
| LiDAR depth 및 Grounded-SAM sky mask 연결 | 미구현 |
| Waymo end-to-end trainer/render export | 미구현 |

따라서 지금 생성되는 파일로 GT 이미지 및 split을 고정하고 외부 렌더 결과를 평가할 수는 있지만, 이 단계만으로 `train_armgs*.py`를 Waymo에 실행하면 안 된다.

## 논문 계열 프로토콜

- ArmGS는 Waymo 8개 시퀀스에서 학습/평가하고, StreetGaussians의 실험 설정을 따른다.
- 공식 StreetGaussians 설정은 FRONT(camera 0), `split_test: 4`를 사용한다. 실제 held-out 상대 frame은 `4, 8, 12, ...`이고 상대 frame 0은 training에 남는다.
- ArmGS 보고 해상도는 `1066 × 1600`(H×W)이다.
- 공식 전처리와 맞춰 source width 기준 단일 비율 `5/6`을 이미지와 intrinsic 두 행에 동일하게 적용하고, `1600×1066`으로 BILINEAR resize한 lossless PNG를 쓴다. 높이에서 별도 scale을 계산하지 않는다.
- 최종 결과는 training view의 reconstruction과 held-out testing view의 novel-view synthesis를 별도로 집계한다.
- 논문은 RGB PSNR, SSIM, LPIPS를 보고하지만 LPIPS backbone, crop, averaging 세부는 명시하지 않는다. 현재 코드는 재현 가능한 로컬 계약으로 SSIM(11×11 Gaussian, σ=1.5)과 LPIPS-Alex를 이미지별 계산 후 평균한다.

장면 목록과 StreetGaussians의 inclusive frame 범위는 [`configs/waymo_streetgs_sequences.txt`](../configs/waymo_streetgs_sequences.txt)에 고정했다. [`configs/armgs_waymo_streetgs.yaml`](../configs/armgs_waymo_streetgs.yaml)은 이 목표 프로토콜을 기록하는 reference config이며, 아직 실행 가능한 Waymo trainer config가 아니다.

## 로컬 데이터 확인 결과

`/workspace/data/waymo_v2/training`에는 필요한 7개 component가 모두 있는 context 13개가 있다. 그러나 이 13개와 공식 8개 validation context의 교집합은 **0개**다. 현재 로컬 context로 파이프라인 smoke test는 가능하지만 ArmGS Waymo 표를 직접 재현할 수는 없다. 공식 평가에는 Waymo validation의 아래 8개 context parquet를 별도로 준비해야 한다.

```text
10448102132863604198_472_000_492_000
12374656037744638388_1412_711_1432_711
17612470202990834368_2800_000_2820_000
1906113358876584689_1359_560_1379_560
2094681306939952000_2972_300_2992_300
4246537812751004276_1560_000_1580_000
5372281728627437618_2005_000_2025_000
8398516118967750070_3958_000_3978_000
```

## 준비 실행

환경은 현재 `/venv/camosplat`을 쓴다. index/split만 검증하는 빠른 local smoke test:

```bash
cd /workspace/projects/camosplat/dohyun/ArmGS
EXTRACT_IMAGES=0 scripts/prepare_waymo_evaluation.sh \
  12251442326766052580_1840_000_1860_000
```

GT를 추출하고 metric manifest까지 만드는 실행:

```bash
EXTRACT_IMAGES=1 scripts/prepare_waymo_evaluation.sh \
  12251442326766052580_1840_000_1860_000
```

준비 스크립트의 시작과 끝도 StreetGaussians `selected_frames`와 동일한 inclusive 범위다. 공식 scene 006은 그대로 `[0, 85]`를 준다.

```bash
PARQUET_DIR=validation \
START_FRAME=0 \
END_FRAME=85 \
scripts/prepare_waymo_evaluation.sh \
  10448102132863604198_472_000_492_000
```

출력은 기본적으로 `data/waymo_prepared/<context>/` 아래에 생긴다.

- `waymo_evaluation_setup.json`: component 경로, calibration, split, frame 메타데이터
- `targets/FRONT/*.png`: 1600×1066(W×H) lossless GT
- `reconstruction_manifest.json`: training-view metric pair
- `novel_view_manifest.json`: held-out metric pair

`--no-extract-images`인 경우 setup JSON만 쓰며 metric manifest는 만들지 않는다.

## 렌더와 metric

준비 manifest가 예고하는 prediction 위치는 다음과 같다.

```text
renders/reconstruction/FRONT/<source_frame_index:06d>.png
renders/novel_view/FRONT/<source_frame_index:06d>.png
```

Waymo renderer가 해당 PNG들을 만든 뒤 각각 평가한다.

```bash
scripts/evaluate_waymo_rgb.sh \
  data/waymo_prepared/<context>/reconstruction_manifest.json

scripts/evaluate_waymo_rgb.sh \
  data/waymo_prepared/<context>/novel_view_manifest.json
```

두 split을 섞어 하나의 평균으로 보고하지 않는다. periodic held-out 평가는 개발 모니터링일 뿐 논문 최종 표의 별도 protocol이 아니므로, 고정 30k iteration 결과에서 두 manifest를 각각 평가하는 것을 기본으로 한다.

## 다음 구현 게이트

Waymo 학습 어댑터는 최소한 다음을 모두 만족해야 한다.

1. camera calibration, camera pose, vehicle pose의 좌표계·축 convention을 테스트로 고정한다.
2. LiDAR 점과 COLMAP SfM 점을 같은 world frame으로 정합한 뒤 background seed로 **함께** 사용한다.
3. 움직이는 객체 점을 background에서 제거하고 actor box 좌표로 canonicalize한다.
4. projected LiDAR depth, Grounded-SAM sky mask, actor alpha supervision을 frame manifest에 연결한다.
5. local refinement → background/actor 단일 depth-order rasterization → sky → global refinement 전체 backward를 검증한다.
6. 30k iteration 후 reconstruction/novel-view 렌더를 고정 manifest에 내보내고 PSNR·SSIM·LPIPS를 각각 집계한다.

특히 현재 production initialization은 LiDAR만 사용하고 COLMAP parser가 학습 경로에 연결되지 않았다. 두 point cloud를 단순 concatenate하면 좌표 불일치와 동적 객체 ghost가 생길 수 있으므로, 좌표 정합·중복 제거·actor 분리 규약을 먼저 구현해야 한다.

## 근거 자료

- [ArmGS 논문, Methodology 및 Experiments](https://arxiv.org/html/2507.03886)
- [StreetGaussians 공식 Waymo scene 006 설정](https://github.com/zju3dv/street_gaussians/blob/main/configs/experiments_waymo/waymo_val_006.yaml)
- [StreetGaussians 공식 split 구현](https://github.com/zju3dv/street_gaussians/blob/main/lib/utils/data_utils.py)
- [StreetGaussians 공식 dynamic validation 장면 목록](https://github.com/zju3dv/street_gaussians/blob/main/script/waymo/waymo_splits/val_dynamic.txt)
