# ArmGS Waymo 학습 세팅

## 현재 기준

현재 Waymo 경로는 공식 StreetGaussians 동적 validation 8개 scene을 대상으로 다음 단계를 연결한다.

1. Waymo-v2 parquet에서 FRONT RGB, camera pose, 5개 LiDAR의 first return을 읽는다.
2. 전체 context vehicle translation 평균을 빼 모든 camera, LiDAR, actor와 SfM을 `waymo_world_centered` 좌표계에 둔다.
3. StreetGaussians가 사용하는 CAStrack actor track과 scene별 planar box scale을 적용한다.
4. Grounded-SAM sky mask와 CAStrack cuboid union actor mask를 만든다.
5. training RGB만 사용한 known-pose COLMAP과 LiDAR를 StreetGaussians 순서로 전처리해 background Gaussian을 초기화한다.
6. background, actor, sky를 한 depth-order로 합성해 30,000 step 학습한다.
7. 마지막 checkpoint에서 reconstruction과 novel view의 PSNR, SSIM, LPIPS-Alex를 따로 평가한다.

통합 진입점은 다음 두 개다.

- [`scripts/prepare_waymo_streetgs_scene.sh`](../scripts/prepare_waymo_streetgs_scene.sh): sky mask → centered known-pose COLMAP → 학습
- [`scripts/train_armgs_waymo.sh`](../scripts/train_armgs_waymo.sh): 준비된 sky/COLMAP/CAStrack으로 학습만 실행

최신 옵션은 항상 아래 도움말로 확인한다.

~~~bash
scripts/prepare_waymo_streetgs_scene.sh --help
scripts/train_armgs_waymo.sh --help
/venv/camosplat/bin/python scripts/prepare_waymo_colmap.py --help
/venv/camosplat/bin/python scripts/train_armgs_waymo.py --help
~~~

## 환경과 기본 경로

이 workspace는 아래 실행 파일을 기본으로 쓴다.

~~~text
ArmGS/Waymo:   /venv/camosplat/bin/python
Grounded-SAM:  /venv/armgs-gsam/bin/python
COLMAP:        /usr/bin/colmap
~~~

W&B 기본값은 다음과 같다.

~~~text
entity:   CamoSplat_ICLR_2027
project:  Ours-ArmGS-Waymo
mode:     online
local:    /workspace/projects/camosplat/dohyun/ArmGS/wandb
~~~

주요 asset과 결과의 기본 구조는 다음과 같다.

~~~text
data/waymo_v2/validation/<component>/<context>.parquet
data/waymo_prepared/
  cache/<context>/...
  manifests/<context>_<start>_<end>.json
  tracking/castrack/<context>.json
  sky_masks/<context>/FRONT/<source-index:08d>.png
  colmap/<context>/...
  colmap_castrack_centered/<context>/...
outputs/waymo/<context>/paper/
  prepared_actor_masks/
  checkpoints/
  evaluation/{reconstruction,novel_view}/
  evaluation_policy.json
  resolved_config.yaml
  run_metadata.json
  wandb_run.json
~~~

`wandb_run.json`은 동일 `OUTPUT_DIR`에서 checkpoint resume할 때 기존 W&B run ID를 복구하는 sidecar다.

## 공식 8개 scene

범위는 inclusive이며, 값은 [`configs/waymo_streetgs_sequences.txt`](../configs/waymo_streetgs_sequences.txt)에 고정되어 있다. `box scale`은 actor-local length/width에만 적용하고 height는 원래 값을 유지한다.

| ID | Waymo context | 범위 | box scale |
|---|---|---:|---:|
| 006 | `10448102132863604198_472_000_492_000` | 0–85 | 1.0 |
| 026 | `12374656037744638388_1412_711_1432_711` | 0–100 | 2.0 |
| 090 | `17612470202990834368_2800_000_2820_000` | 0–102 | 1.0 |
| 105 | `1906113358876584689_1359_560_1379_560` | 20–186 | 1.0 |
| 108 | `2094681306939952000_2972_300_2992_300` | 20–115 | 1.0 |
| 134 | `4246537812751004276_1560_000_1580_000` | 106–198 | 1.0 |
| 150 | `5372281728627437618_2005_000_2025_000` | 96–197 | 1.5 |
| 181 | `8398516118967750070_3958_000_3978_000` | 0–160 | 1.0 |

각 context에는 `validation` 아래 7개 component가 모두 있어야 한다.

~~~text
camera_image  camera_calibration  lidar  lidar_pose
lidar_box     lidar_calibration   vehicle_pose
~~~

2026-08-05 현재 로컬에는 8개 context의 7개 parquet와 scene별 CAStrack JSON이 모두 있다. Sky mask와 centered COLMAP은 scene 006이 준비됐고 나머지 scene은 실행 전에 생성해야 한다.

CAStrack 원본 multi-scene JSON을 다시 준비하는 경우, 학습 때 396 MB 파일을 매번 파싱하지 않도록 scene별 파일로 한 번 분리한다.

~~~bash
/venv/camosplat/bin/python scripts/extract_waymo_castrack.py \
  data/waymo_prepared/tracking/castrack_validation_result.json \
  data/waymo_prepared/tracking/castrack/10448102132863604198_472_000_492_000.json \
  --sequence 10448102132863604198_472_000_492_000
~~~

공식 launcher는 `CAS_TRACK_PATH`를 지정하지 않으면 `data/waymo_prepared/tracking/castrack/<context>.json`을 사용하며 paper mode에서는 non-empty 파일을 필수로 요구한다.

## scene 006 검증 asset

기존 `data/waymo_prepared/colmap/<context>` 결과는 보존되어 있다. CAStrack dynamic mask와 centered known pose를 명시적으로 사용한 현재 scene 006 결과는 아래 경로다.

~~~text
data/waymo_prepared/colmap_castrack_centered/
  10448102132863604198_472_000_492_000/
    mapping.json
    triangulated_text/points3D.txt
~~~

`mapping.json` 검증 결과는 다음과 같다.

- `status=complete`
- source 86장, train 65장, held-out 21장
- CAStrack 8개 track으로 만든 COLMAP dynamic mask
- 22,866개 SfM point
- `world_frame=waymo_world_centered`
- full-context world center `[-344.759818, -32.237161, 8.326038]` m
- pose/intrinsic refinement 비활성화, OpenCV PINHOLE known pose

scene 006 reference run에서는 과거 absolute-world 결과가 자동 재사용되지 않도록 `COLMAP_DIR`을 반드시 위 새 경로로 지정한다.

## 실행 방법

### 1. scene 006 준비 상태 확인

전체 파이프라인을 실행하지 않고 명령만 확인한다.

~~~bash
cd /workspace/projects/camosplat/dohyun/ArmGS

DRY_RUN=1 \
COLMAP_DIR="$PWD/data/waymo_prepared/colmap_castrack_centered/10448102132863604198_472_000_492_000" \
scripts/prepare_waymo_streetgs_scene.sh 006
~~~

### 2. 준비된 scene 006 학습

~~~bash
cd /workspace/projects/camosplat/dohyun/ArmGS

SEQ=10448102132863604198_472_000_492_000
GPU_ID=0 \
COLMAP_DIR="$PWD/data/waymo_prepared/colmap_castrack_centered/$SEQ" \
OUTPUT_DIR="$PWD/outputs/waymo/scene_006/paper" \
WANDB_ENTITY=CamoSplat_ICLR_2027 \
WANDB_PROJECT=Ours-ArmGS-Waymo \
WANDB_RUN_NAME=armgs_waymo_scene_006_paper \
scripts/train_armgs_waymo.sh "$SEQ" 0 85
~~~

이 명령은 30,000이 전체 목표 step이다. `ITERATIONS`는 resume 후 추가 step 수가 아니다.

### 3. 다른 공식 scene의 전체 준비와 학습

각 scene에도 centered/CAStrack COLMAP 경로를 명시하는 것을 권장한다.

~~~bash
cd /workspace/projects/camosplat/dohyun/ArmGS

SCENE=026
SEQ=12374656037744638388_1412_711_1432_711
GPU_ID=0 \
COLMAP_DIR="$PWD/data/waymo_prepared/colmap_castrack_centered/$SEQ" \
OUTPUT_DIR="$PWD/outputs/waymo/scene_${SCENE}/paper" \
scripts/prepare_waymo_streetgs_scene.sh "$SCENE"
~~~

준비 asset을 재사용하려면 다음과 같이 sky/COLMAP 단계를 끈다.

~~~bash
RUN_SKY=0 RUN_COLMAP=0 GPU_ID=0 \
COLMAP_DIR="$PWD/data/waymo_prepared/colmap_castrack_centered/$SEQ" \
scripts/prepare_waymo_streetgs_scene.sh "$SCENE"
~~~

`REUSE_COLMAP=1`은 `points3D.txt`와 `mapping.json`이 모두 있는 complete 결과만 재사용한다. 불완전한 디렉터리는 자동 삭제하지 않는다.

### 4. Resume와 eval-only

~~~bash
SEQ=10448102132863604198_472_000_492_000
OUT="$PWD/outputs/waymo/scene_006/paper"

GPU_ID=0 \
COLMAP_DIR="$PWD/data/waymo_prepared/colmap_castrack_centered/$SEQ" \
OUTPUT_DIR="$OUT" \
RESUME="$OUT/checkpoints/final.pt" \
scripts/train_armgs_waymo.sh "$SEQ" 0 85

# final checkpoint를 다시 학습하지 않고 두 split만 재평가
GPU_ID=0 \
COLMAP_DIR="$PWD/data/waymo_prepared/colmap_castrack_centered/$SEQ" \
OUTPUT_DIR="$OUT" \
RESUME="$OUT/checkpoints/final.pt" \
scripts/train_armgs_waymo.sh "$SEQ" 0 85 -- --eval-only
~~~

Checkpoint resume은 config와 dataset/split identity를 엄격히 비교한다. 다른 CAStrack, mask, frame range 또는 COLMAP asset으로 같은 checkpoint를 조용히 이어서 학습할 수 없다.

## Paper mode가 강제하는 계약

`PAPER_MODE=1`은 기본값이다. bash launcher와 Python trainer가 GPU 학습 전에 다음을 검증한다.

| 항목 | 강제 값 |
|---|---|
| dataset | 공식 8개 validation context 중 하나, 표의 inclusive range |
| camera / resolution | FRONT, 1600×1066(W×H) |
| split | relative source position 4, 8, 12, … held-out |
| tracker | non-empty scene CAStrack JSON |
| actor box scale | 공식 scene 표 값, length/width only |
| initialization | all-selected LiDAR first return + train-only known-pose COLMAP |
| COLMAP provenance | runtime train/eval row와 일치하는 `mapping.json`, PINHOLE/OpenCV fixed pose |
| coordinate frame | 새 asset은 full-context-mean `waymo_world_centered`; center 일치 확인 |
| masks | 모든 source의 sky mask, 모든 training row의 actor mask(빈 mask 포함) |
| model/config | SH degree 1, scene extent 20 m, merge 후 re-voxel 없음 |
| density | group별 opacity/world-scale pruning, actor bbox pruning, screen-radius pruning 비활성화 |
| optimization | 정확히 30,000 step |
| final evaluation | reconstruction + novel view, PSNR/SSIM/LPIPS-Alex |

Background seed의 실제 전처리 순서는 다음과 같다.

1. LiDAR background를 0.15 m voxel downsample
2. radius outlier 제거: 10 neighbors, radius 0.5 m
3. LiDAR AABB 중심과 half-diagonal의 2배 sphere 안에 있는 SfM point 유지
4. LiDAR와 SfM을 concatenate하고 다시 voxelize하지 않음

Actor seed는 box 안 LiDAR를 canonicalize한다. Actor별 LiDAR가 2,000점 이상이면 cap이나 subsampling 없이 모든 점을 사용하고, 2,000점 미만이면 희소 LiDAR 대신 20³ bbox grid를 사용한다. Densification 중에는 StreetGaussians식 bbox containment pruning을 적용한다.

Waymo density-control의 pruning 정책은 다음과 같다. `max_screen_radius: null`이므로 일반 3DGS의 화면상 반경 기준 pruning은 background와 actor 모두 사용하지 않는다.

| Gaussian group | 적용하는 pruning |
|---|---|
| background | opacity가 0.005 미만인 Gaussian; step 3,000 이후 per-group extent의 0.1보다 world-space 최대 scale이 큰 Gaussian |
| actor | opacity와 같은 world-scale 조건; step 3,000 이후 StreetGaussians의 actor-box 2-sample containment test를 통과하지 못한 Gaussian |

따라서 Waymo에서는 background에 opacity+world-scale만, actor에 opacity+world-scale+bbox만 적용하며 screen-radius는 prune 조건에 포함하지 않는다.

## W&B 로깅

기본 scalar interval은 100 step, train image interval은 500 step이다.

주요 train scalar:

- `train/loss`, `train/rgb_l1`, `train/ssim_loss`
- `train/depth_loss`, `train/sky_loss`, `train/foreground_loss`
- `train/psnr`, `train/ssim`: 해당 100-step 창에서 실제 sampled train frame의 이미지 단위 값 평균
- `train/gaussians/{total,background,actors}`와 actor별 count
- `train/lr/*`, densification/pruning count, step time, CUDA memory

이미지는 500 step마다 현재 sampled FRONT frame의 좌측 GT/우측 render를 `train/gt_vs_render`에 기록한다. `IMAGE_LOG_INTERVAL=0`이면 image logging을 끌 수 있다.

기본 `EVAL_INTERVAL=0`이므로 중간 held-out 전체 렌더는 하지 않는다. 마지막 step에 다음 namespace로 PSNR/SSIM/LPIPS와 FRONT preview를 기록한다.

~~~text
reconstruction/{psnr,ssim,lpips}
reconstruction/FRONT/{psnr,ssim,lpips,gt_vs_render}
novel_view/{psnr,ssim,lpips}
novel_view/FRONT/{psnr,ssim,lpips,gt_vs_render}
~~~

Waymo key에는 nuScenes에서 사용했던 `eval/` prefix가 붙지 않는다. 자세한 평균 규칙과 로컬 결과 파일은 [Waymo 평가 세팅](WAYMO_EVALUATION_SETUP.ko.md)을 참조한다.

## 논문에 공개되지 않아 고정한 가정

현재 구현은 공개된 ArmGS 수식과 StreetGaussians Waymo recipe에 최대한 맞췄지만, 아래는 ArmGS 논문만으로 exact 값을 결정할 수 없다.

| 항목 | 현재 선택 | 근거/주의 |
|---|---|---|
| local/global/actor hidden width와 frequency | config 값(주로 64, frequency 6) | 논문 비공개 구현 가정 |
| hash-grid level/resolution/hash size | config의 8-level PyTorch grid | 논문 비공개; fused CUDA 구현과 성능 차이 가능 |
| sky cubemap resolution | 1024 | StreetGaussians Waymo 기반 가정 |
| initial opacity/scale/KNN | 0.1/0.05/3-NN | 논문 비공개 구현 가정 |
| actor-pose LR | translation 0.005→0.00005, rotation 0.001→0.00001, 3k warmup | ArmGS 비공개; StreetGaussians profile 기반 |
| sky LR | 0.01→0.0001 | ArmGS 비공개; StreetGaussians profile 기반 |
| foreground regularizer 시작 | 15,000 | ArmGS 비공개; StreetGaussians의 densification 종료 시점 기반 |
| Grounded-SAM 세부 | prompt `sky`, source top-100-pixel box rule | 논문은 전체 checkpoint/prompt 후처리를 명시하지 않음 |
| LPIPS/SSIM 세부 | Alex, 이미지별 평균, SSIM 11×11/σ1.5 | 논문은 backbone·평균·crop 세부를 명시하지 않음 |

따라서 `paper_protocol_compliant=true`는 이 저장소가 정의한 공개 논문 + StreetGaussians 기반 계약을 모두 통과했다는 뜻이며, 저자 비공개 코드와 bit-exact하다는 뜻은 아니다. 각 run의 실제 가정과 point count, scale 진단, 좌표계는 `run_metadata.json`과 W&B config에 남는다.

## 근거

- [ArmGS 논문: Methodology 및 Experiments](https://arxiv.org/html/2507.03886)
- [StreetGaussians 공식 저장소](https://github.com/zju3dv/street_gaussians)
- [`configs/armgs_waymo_streetgs.yaml`](../configs/armgs_waymo_streetgs.yaml)
- [`scripts/train_armgs_waymo.py`](../scripts/train_armgs_waymo.py)
