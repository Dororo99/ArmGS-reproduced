# ArmGS research implementation

이 저장소는 **ArmGS: Composite Gaussian Appearance Refinement for Modeling
Dynamic Urban Environments**의 재현 구현을 단계적으로 구성합니다.

현재 milestone은 논문의 Eq. (3)–(9), background·actor·sky의 paper-ordered
forward, gsplat CUDA end-to-end backward에 더해 KITTI와 nuScenes canonical
data path, 학습·평가 CLI까지 연결한 상태입니다. LiDAR 초기화,
actor/background scene 구축, adaptive density control, dataloader를 포함한 exact
checkpoint resume도 독립 ArmGS 패키지에서 제공합니다. Waymo 등 추가 데이터셋과
실제 SplatAD/Nerfstudio 삽입, 논문 표 수치 재현은 다음 milestone입니다.

COLMAP `points3D.txt` 파서와 단위 테스트는 있지만, SfM 좌표 정렬·LiDAR 병합 및
학습 CLI 연결은 아직 없습니다. 따라서 현재 KITTI/nuScenes 학습 초기화는
LiDAR-only이며, 논문의 LiDAR+SfM 초기화를 완료한 것으로 간주하면 안 됩니다.

- 논문 분석: [docs/PAPER_ANALYSIS.ko.md](docs/PAPER_ANALYSIS.ko.md)
- 구현 계획: [docs/IMPLEMENTATION_PLAN.ko.md](docs/IMPLEMENTATION_PLAN.ko.md)
- 검증 현황: [docs/VALIDATION.ko.md](docs/VALIDATION.ko.md)
- 기본 설정: [configs/armgs_default.yaml](configs/armgs_default.yaml)
- SplatAD 통합 설계: [docs/SPLATAD_INTEGRATION.ko.md](docs/SPLATAD_INTEGRATION.ko.md)
- Waymo 평가 준비: [docs/WAYMO_EVALUATION_SETUP.ko.md](docs/WAYMO_EVALUATION_SETUP.ko.md)
- W&B 로깅 계약: [docs/WANDB_LOGGING.ko.md](docs/WANDB_LOGGING.ko.md)

## 개발 환경

Python 3.10 이상과 PyTorch 2.0 이상을 대상으로 합니다. 이 Vast 인스턴스에서는
CUDA 11.8/gsplat과 맞는 기존 환경으로 전체 suite를 실행할 수 있습니다.

~~~bash
PYTHONPATH=src /venv/camosplat/bin/python -m pytest
~~~

일반 환경에서는 uv venv 이후 다음처럼 설치합니다. LPIPS가 필요하지 않으면
`evaluation` extra를 생략할 수 있습니다.

~~~bash
uv pip install -e '.[dev,integration,data,evaluation,tracking]'
~~~

공개 논문은 네트워크 폭, 임베딩 차원, hash-grid 세부값, exact SSIM/depth 설정 등
일부 재현 파라미터를 명시하지 않습니다. 그런 값은 YAML의 구현 가정으로
분리했으며 논문에 명시된 값에는 paper 주석을 붙였습니다.

## 현재 구현 범위

- Eq. (3)–(4): frame embedding + multi-resolution hash-grid local affine
- Eq. (5)–(6): 카메라 위치/시선 기반 global image affine
- Eq. (7)–(8): actor spatial-temporal position/SH deformation
- Eq. (9): RGB, DSSIM, LiDAR depth, sky BCE, foreground entropy 결합 손실
- learnable background/actor Gaussian parameterization과 actor pose trajectory
- explicit differentiable cubemap sky
- actor deformation → pose → composite → local → single raster → sky → global 순서
- Gaussian parameter별 Adam group과 mean exponential LR decay
- stateful shuffle sampler를 포함한 exact mid-epoch resume
- gsplat gradient/radius statistics 기반 duplicate/split/prune/opacity reset
- topology 변경 시 Adam state migration과 topology-aware checkpoint resume
- KITTI camera/pose/timestamp/tracklet/Velodyne canonical loader와 lazy image batch
- nuScenes key-sample/6-camera/LIDAR_TOP/동적 actor canonical loader
- scene-0061 trainer의 training-loss 및 periodic held-out W&B metric logging
- 요청 카메라에 투영된 LiDAR point union만 보관하는 memory-safe 기본 경로
- capture 및 actor pose sample 단위 leak-free train/eval split
- KITTI bottom-center box를 centered canonical actor pose로 변환
- colored LiDAR Gaussian 초기화와 tracked actor/background scene 구축
- COLMAP points3D parser (LiDAR+SfM fusion 및 학습 연결은 미구현)
- dataset stat/content fingerprint를 포함한 checkpoint resume 검증
- standalone metric CLI와 nuScenes held-out PSNR/SSIM/optional LPIPS·actor-PSNR 평가
- RGB/expected-depth/aggregate actor-alpha gsplat CUDA adapter
- OpenCV/OpenGL, wxyz quaternion, log-scale, opacity-logit 계약
- same-camera nearest embedding과 float64 기반 timestamp 정규화
- gsplat near/far/eps2d 동기화 culling, rolling-shutter fallback, chunked reference hash-grid
- 설정 가능한 identity-centered local/global affine bounds

## 실행 예시

~~~bash
PYTHONPATH=src /venv/camosplat/bin/python scripts/train_armgs.py --help
PYTHONPATH=src /venv/camosplat/bin/python scripts/evaluate_armgs.py --help
~~~

nuScenes v1.0-trainval의 scene-0061은 다음 launcher로 실행합니다. 기본 W&B
대상은 `CamoSplat/ArmGS-nuScenes`입니다.

~~~bash
scripts/train_nuscenes_scene_0061.sh 0
~~~
scene-0061 launcher는 기본적으로 training scalar를 100 step 구간 평균으로,
training `GT | render` 이미지를 500 step마다 서로 독립적으로 기록한다. 여기서
`train/ssim_loss`는 Eq. (9)의 학습용 `1 - SSIM` 항이며 held-out 평가의 SSIM
값이 아니다. 별도로 capture-safe eval split 전체를 기본 1,000 step마다 그리고
학습 종료 시 렌더링해 PSNR, SSIM, projected 3D actor cuboid-silhouette PSNR을
aggregate/per-camera로 계산한다. launcher에서는 LPIPS도 기본 활성화한다.

평가 결과는 `<OUTPUT_DIR>/evaluation/step_XXXXXXXX.json`과 카메라별
`GT | render` PNG에 원자 저장되고 동일 step의 `eval/*` W&B key로 기록된다.
training 이미지는 `IMAGE_LOG_INTERVAL=0`, 주기 평가는 `EVAL_INTERVAL=0`,
LPIPS는 `EVAL_LPIPS=0`으로 끌 수 있으며,
`EVAL_AT_END=0`을 명시하지 않으면 마지막 평가는 항상 수행된다. 기존 checkpoint만
평가할 때는 학습이나 checkpoint 쓰기 없이 다음처럼 실행한다.

~~~bash
EVAL_ONLY=1 RESUME=/path/to/final.pt OUTPUT_DIR=/path/to/run \
  scripts/train_nuscenes_scene_0061.sh 0
~~~

LPIPS는 `lpips==0.1.4`의 AlexNet v0.1을 사용하고 RGB를 공식 입력 범위
`[-1,1]`로 변환한다. ArmGS 논문은 LPIPS backbone과 정확한 전처리를 공개하지
않았으므로 이 선택은 결과 JSON의 `policy.metric_protocols`에 명시한다.

W&B 통신 오류는 기본값 `WANDB_FAIL_FAST=0`에서 경고 후 로컬 학습과 checkpoint를
계속하며, 연구 추적을 필수 조건으로 만들 때만 `WANDB_FAIL_FAST=1`을 사용한다.
checkpoint는 기본적으로 로컬에만 저장한다. `WANDB_LOG_CHECKPOINT=1`은 periodic
checkpoint 전부가 아니라 final checkpoint 하나만 Artifact로 올린다. resume가
같은 `OUTPUT_DIR`을 사용하면 `wandb_run.json`의 run ID를 자동 복구한다.
`WANDB_RUN_ID`는 이 자동 선택을 덮어쓸 때, 또는 sidecar가 없는 기존 30K run을
명시적으로 이어갈 때 사용한다. 전체 key와 주기·resume 계약은
[W&B 로깅 문서](docs/WANDB_LOGGING.ko.md)에 정리했다.

## 검증 요약

SH degree 0–3은 공식 3DGS/gsplat과 parity를 검증했고, synthetic scene와 실제
gsplat 모두에서 local → rasterizer → sky → global → Eq. (9) 전체 backward를
통과합니다. Waymo 평가 준비 경로를 포함한 전체 suite 결과는 **269 passed,
2 skipped, 5 warnings**입니다.

Reference hash-grid의 RTX 4090 측정값은 다음과 같습니다.

| 전체 Gaussian | hash 적용 Gaussian | peak allocated |
|---:|---:|---:|
| 100,000 | 100,000 | 241.6 MiB |
| 1,000,000 | 1,000,000 | 2,313.1 MiB |
| 1,000,000 | 100,000 visible | 260.7 MiB |

재실행 명령은 scripts/benchmark_hash_grid.py에 있습니다.

## 남은 재현 범위

- Waymo/NOTR/VKITTI2 dataset adapter
- 실제 SplatAD/Nerfstudio model·dataparser 삽입과 rolling-shutter metadata 생산
- fused hash-grid 및 논문 규모 FPS/quality benchmark
- 매우 긴 sequence용 streaming projection/voxel fusion
- class-wise actor hash-grid/hypernetwork ablation
- 논문 표 수치 재현 및 차이를 기록한 report

논문이 공개한 density schedule은 500–15,000 step, 100-step 간격으로 분리해
기록했습니다. split/prune threshold, split 배율, opacity-reset 값·동작은 논문에
공개되지 않았으므로 YAML의 **구현 가정**이며 논문 설정으로 간주하지 않습니다.
