# ArmGS research implementation

이 저장소는 **ArmGS: Composite Gaussian Appearance Refinement for Modeling
Dynamic Urban Environments**의 재현 구현을 단계적으로 구성합니다.

현재 milestone은 논문의 Eq. (3)–(9)뿐 아니라 background·actor·sky를 하나의
paper-ordered forward로 연결하고, gsplat CUDA에서 end-to-end backward와
one-view trainer/checkpoint resume까지 검증한 상태입니다. Waymo/KITTI 실제 데이터
converter와 논문 수치 재현은 다음 milestone입니다.

- 논문 분석: [docs/PAPER_ANALYSIS.ko.md](docs/PAPER_ANALYSIS.ko.md)
- 구현 계획: [docs/IMPLEMENTATION_PLAN.ko.md](docs/IMPLEMENTATION_PLAN.ko.md)
- 검증 현황: [docs/VALIDATION.ko.md](docs/VALIDATION.ko.md)
- 기본 설정: [configs/armgs_default.yaml](configs/armgs_default.yaml)
- SplatAD 통합 설계: [docs/SPLATAD_INTEGRATION.ko.md](docs/SPLATAD_INTEGRATION.ko.md)

## 개발 환경

Python 3.10 이상과 PyTorch 2.0 이상을 대상으로 합니다. 이 Vast 인스턴스에서는
CUDA 11.8/gsplat과 맞는 기존 환경으로 전체 suite를 실행할 수 있습니다.

~~~bash
PYTHONPATH=src /venv/camosplat/bin/python -m pytest
~~~

일반 환경에서는 uv venv 이후 uv pip install -e '.[dev,integration]'로 설치합니다.

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
- one-view trainer 및 optimizer/model/CPU·CUDA RNG checkpoint resume
- RGB/expected-depth/aggregate actor-alpha gsplat CUDA adapter
- OpenCV/OpenGL, wxyz quaternion, log-scale, opacity-logit 계약
- same-camera nearest embedding과 float64 기반 timestamp 정규화
- gsplat near/far/eps2d 동기화 culling, rolling-shutter fallback, chunked reference hash-grid
- 설정 가능한 identity-centered local/global affine bounds

## 검증 요약

현재 전체 suite는 CPU와 CUDA를 포함해 74 passed입니다. SH degree 0–3은
공식 3DGS/gsplat과 parity를 검증했고, synthetic scene와 실제 gsplat 모두에서
local → rasterizer → sky → global → Eq. (9) 전체 backward를 통과합니다.

Reference hash-grid의 RTX 4090 측정값은 다음과 같습니다.

| 전체 Gaussian | hash 적용 Gaussian | peak allocated |
|---:|---:|---:|
| 100,000 | 100,000 | 241.6 MiB |
| 1,000,000 | 1,000,000 | 2,313.1 MiB |
| 1,000,000 | 100,000 visible | 260.7 MiB |

재실행 명령은 scripts/benchmark_hash_grid.py에 있습니다.

## 남은 재현 범위

- KITTI/Waymo/NOTR/VKITTI2 dataset converter와 tracked actor 초기화
- LiDAR/COLMAP Gaussian initialization
- 저자 코드에만 있을 densification/pruning/opacity-reset threshold
- SplatAD rolling-shutter metadata 연결
- class-wise actor hash-grid/hypernetwork ablation
- PSNR/SSIM/LPIPS evaluator와 논문 표 재현 report
