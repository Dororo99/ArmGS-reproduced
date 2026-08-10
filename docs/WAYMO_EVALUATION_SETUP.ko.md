# ArmGS Waymo 평가 세팅

## 현재 평가 경로

Waymo trainer는 학습 마지막 checkpoint에서 FRONT camera의 두 split을 따로 렌더하고 평가한다.

| split | frame | appearance embedding |
|---|---|---|
| `reconstruction` | training source positions | 각 training row의 정확한 learned embedding |
| `novel_view` | relative source positions 4, 8, 12, … | 같은 camera의 nearest training-frame embedding |

공식 scene 006의 0–85 범위는 train 65장, novel-view 21장이다. 두 split의 점수를 섞지 않으며 최종 표에도 별도로 기록한다.

학습·asset 준비와 paper-mode 계약은 [Waymo 학습 세팅](WAYMO_TRAINING_SETUP.ko.md)을 먼저 참조한다.

## Metric 계약

평가는 clamp된 RGB `[0,1]`에서 이미지별 metric을 구한 뒤 split 안에서 산술 평균한다. FRONT-only이므로 aggregate와 `FRONT` 평균은 같다.

| metric | 구현 계약 |
|---|---|
| PSNR | 이미지별 RGB MSE, data range 1, 이미지별 PSNR의 평균 |
| SSIM | 3DGS 계열 Gaussian 11×11, σ=1.5, data range 1, 이미지별 평균 |
| LPIPS | 공식 `lpips` v0.1, RGB를 `[-1,1]`로 변환, Alex backbone, 이미지별 평균 |
| actor PSNR | CAStrack cuboid projection union 안의 RGB PSNR; actor pixel이 있는 이미지만 평균 |

ArmGS 논문은 PSNR/SSIM/LPIPS를 보고하지만 LPIPS backbone, crop, 이미지 평균 방식을 모두 공개하지 않는다. 따라서 Alex와 위 평균 규칙은 재현 가능한 로컬 계약이며 결과의 `evaluation_policy.json`과 W&B config에 함께 기록된다.

## 기본 실행

공식 launcher는 `EVAL_INTERVAL=0`, `--eval-at-end`, `--eval-reconstruction-at-end`, `--eval-lpips`, `--eval-lpips-net alex`를 전달한다. 즉 30k 도중 전체 held-out 렌더는 생략하고 마지막에 두 split을 평가한다.

~~~bash
cd /workspace/projects/camosplat/dohyun/ArmGS

SEQ=10448102132863604198_472_000_492_000
GPU_ID=0 \
COLMAP_DIR="$PWD/data/waymo_prepared/colmap_castrack_centered/$SEQ" \
OUTPUT_DIR="$PWD/outputs/waymo/scene_006/paper" \
scripts/train_armgs_waymo.sh "$SEQ" 0 85
~~~

필요하면 개발 중 periodic novel-view 평가를 켤 수 있다.

~~~bash
SEQ=10448102132863604198_472_000_492_000
EVAL_INTERVAL=5000 \
COLMAP_DIR="$PWD/data/waymo_prepared/colmap_castrack_centered/$SEQ" \
scripts/train_armgs_waymo.sh "$SEQ" 0 85
~~~

이는 학습을 모니터링하기 위한 추가 실행이며 기본 paper-oriented protocol은 final-only다.

## Checkpoint만 다시 평가

같은 `OUTPUT_DIR`, dataset identity, CAStrack, masks와 COLMAP provenance를 사용해야 한다.

~~~bash
cd /workspace/projects/camosplat/dohyun/ArmGS

SEQ=10448102132863604198_472_000_492_000
OUT="$PWD/outputs/waymo/scene_006/paper"
GPU_ID=0 \
COLMAP_DIR="$PWD/data/waymo_prepared/colmap_castrack_centered/$SEQ" \
OUTPUT_DIR="$OUT" \
RESUME="$OUT/checkpoints/final.pt" \
scripts/train_armgs_waymo.sh "$SEQ" 0 85 -- --eval-only
~~~

`--eval-only`는 checkpoint write나 추가 optimization 없이 두 split을 렌더한다.

## 로컬 결과

각 split의 JSON과 첫 FRONT GT/render preview는 다음 위치에 저장된다.

~~~text
<OUTPUT_DIR>/evaluation_policy.json
<OUTPUT_DIR>/evaluation/reconstruction/step_00030000.json
<OUTPUT_DIR>/evaluation/reconstruction/step_00030000_FRONT_gt_render.png
<OUTPUT_DIR>/evaluation/novel_view/step_00030000.json
<OUTPUT_DIR>/evaluation/novel_view/step_00030000_FRONT_gt_render.png
~~~

각 JSON에는 다음 필드가 있다.

- `aggregate`: `psnr`, `ssim`, `lpips`, `actor_psnr`, image/pixel count
- `per_camera.FRONT`: 같은 metric과 count
- `previews`: 저장된 GT/render 비교 이미지 경로
- `policy`: resolution, LPIPS net, metric/actor-mask protocol

모델이 마지막 step에 도달하면 checkpoint 경로와 split aggregate도 W&B summary에 기록된다.

## W&B key

Waymo는 split-qualified namespace를 사용한다.

~~~text
reconstruction/psnr
reconstruction/ssim
reconstruction/lpips
reconstruction/actor_psnr
reconstruction/FRONT/psnr
reconstruction/FRONT/ssim
reconstruction/FRONT/lpips
reconstruction/FRONT/actor_psnr
reconstruction/FRONT/gt_vs_render

novel_view/psnr
novel_view/ssim
novel_view/lpips
novel_view/actor_psnr
novel_view/FRONT/psnr
novel_view/FRONT/ssim
novel_view/FRONT/lpips
novel_view/FRONT/actor_psnr
novel_view/FRONT/gt_vs_render
~~~

NuScenes run의 `eval/CAM_FRONT/...` key와 다르므로 W&B dashboard query도 위 이름을 사용해야 한다.

학습 중에는 별도로 다음을 기록한다.

- `train/psnr`, `train/ssim`: `LOG_INTERVAL` 동안 sampled train images의 평균
- `train/gt_vs_render`: `IMAGE_LOG_INTERVAL=500`마다 현재 sampled FRONT GT/render

이 train metric은 전체 reconstruction 평가가 아니다. 논문 수치 비교에는 반드시 마지막 `reconstruction/*`와 `novel_view/*`를 사용한다.

## 독립 PNG manifest 평가

[`scripts/prepare_waymo_evaluation.sh`](../scripts/prepare_waymo_evaluation.sh)과 [`scripts/evaluate_waymo_rgb.sh`](../scripts/evaluate_waymo_rgb.sh)은 외부 renderer가 만든 PNG를 평가하는 보조 경로다. 통합 trainer의 final evaluation이 기본 경로이며, 독립 경로를 쓸 때 prediction 파일은 manifest가 지정한 위치에 둔다.

~~~text
renders/reconstruction/FRONT/<source_frame_index:06d>.png
renders/novel_view/FRONT/<source_frame_index:06d>.png
~~~

~~~bash
scripts/evaluate_waymo_rgb.sh \
  data/waymo_prepared/<context>/reconstruction_manifest.json

scripts/evaluate_waymo_rgb.sh \
  data/waymo_prepared/<context>/novel_view_manifest.json
~~~

통합 trainer와 외부 PNG 경로의 resize, color range, mask 및 metric protocol이 같아야 두 결과를 직접 비교할 수 있다.

## 최종 확인 항목

1. `run_metadata.json`의 `paper_protocol_compliant`와 `paper_protocol_deviations`를 확인한다.
2. initialization에 LiDAR와 SfM point count가 모두 0보다 큰지 확인한다.
3. coordinate frame이 `waymo_world_centered`이고 COLMAP world center가 runtime context와 같은지 확인한다.
4. final JSON의 `num_images`가 scene split 수와 일치하는지 확인한다.
5. W&B의 `train/gt_vs_render`로 500-step 단위 최적화 상태를 보고, 논문 비교는 final split metric으로만 한다.
6. 8개 scene 평균을 낼 때 scene별 평균을 어떤 방식으로 다시 평균했는지 별도 표에 명시한다. 논문은 cross-scene weighting 세부를 공개하지 않는다.

## 근거

- [ArmGS 논문: Experiments](https://arxiv.org/html/2507.03886#S4)
- [StreetGaussians 공식 저장소](https://github.com/zju3dv/street_gaussians)
- [`src/armgs/evaluation.py`](../src/armgs/evaluation.py)
- [`scripts/train_armgs_waymo.py`](../scripts/train_armgs_waymo.py)
