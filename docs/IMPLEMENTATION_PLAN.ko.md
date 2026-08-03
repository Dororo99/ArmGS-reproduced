# ArmGS 구현 계획

## 원칙

- 논문에 명시된 동작과 재현을 위한 가정을 코드/설정에서 구분한다.
- 모든 appearance/deformation head는 초기 상태에서 identity/no-op이어야 한다.
- background와 actors는 하나의 rasterization에서 깊이 정렬한다.
- 좌표계, quaternion, scale/opacity, timestamp 계약을 API에서 강제한다.
- dataset별 변환은 canonical intermediate schema로 정규화한다.

## Phase 0 — 패키지와 수치 계약 (완료)

- [x] 논문 분석 및 미공개 세부사항 목록화
- [x] Python package, configuration, test 구조
- [x] Gaussian/camera/actor tensor validation과 near-zero quaternion 거부
- [x] quaternion transform 및 SLERP
- [x] module dtype cast에도 보존되는 float64 absolute timestamp와 normalized actor-time 계약
- [x] SH degree 0–3 공식 3DGS/gsplat numerical parity
- [x] OpenCV/OpenGL 및 wxyz convention 고정

## Phase 1 — 세 refinement 모듈 (완료)

- [x] frame embedding 및 same-camera nearest-frame lookup
- [x] pure-PyTorch multi-resolution hash-grid
- [x] visible-index selection과 chunked hash evaluation
- [x] local Gaussian affine learner (Eq. 3–4)
- [x] viewpoint encoder와 global image affine learner (Eq. 5–6)
- [x] actor spatial-temporal encoder와 position/SH heads (Eq. 7–8)
- [x] identity/no-deformation initialization
- [x] optional identity-centered affine output bounds

## Phase 2 — rendering vertical slice (완료)

- [x] activated learnable background/actor Gaussian containers
- [x] actor deformation과 pose interpolation/transform
- [x] background/actor 단일 composite Gaussian set
- [x] actor track lifecycle과 범위 밖 ghost 제거
- [x] gsplat near/far/eps2d 동기화 culling과 rolling-shutter correctness fallback
- [x] local refinement 이전 SH RGB 평가
- [x] single depth-order gsplat rasterization
- [x] O(HW) aggregate actor alpha, backend 지원 부재 감지, per-group diagnostic mode
- [x] explicit cubemap sky 및 residual transmittance 합성
- [x] sky 합성 이후 global refinement
- [x] CPU exact compositor ordering test
- [x] 실제 gsplat CUDA 전체 forward/backward test

완료 조건인 RGB/depth/sky/actor alpha의 end-to-end gradient를 synthetic CUDA
scene에서 확인했다.

## Phase 3 — loss와 trainer (진행 중)

- [x] Eq. (9) weighted objective와 auxiliary input 강제
- [x] Gaussian parameter별 finite-positive Adam LR group
- [x] mean exponential LR decay
- [x] pose/refinement/sky parameter groups
- [x] one-view trainer
- [x] model/optimizer/CPU·CUDA RNG checkpoint resume와 GPU 수 변화 안전성
- [ ] densification/pruning/opacity reset lifecycle
- [ ] adaptive topology 변경 이후 optimizer state migration
- [ ] deterministic multi-frame dataloader resume

논문은 densification 주기만 공개하고 split/prune threshold와 opacity reset 규칙을
공개하지 않았으므로, 이 부분은 clean SplatAD baseline 전략을 명시적으로 선택한 뒤
연결한다.

## Phase 4 — dataset adapters

- [ ] canonical camera/frame/actor/LiDAR schema
- [ ] KITTI + tracklets converter를 첫 end-to-end target으로 구현
- [ ] Waymo + StreetGS tracked boxes
- [ ] NOTR, VKITTI2
- [ ] COLMAP/LiDAR Gaussian initialization
- [ ] train/test split 및 nearest embedding metadata 검증
- [ ] rolling-shutter metadata 보존

KITTI를 먼저 선택하는 이유는 데이터·해상도·tracklet 형식이 비교적 작아 전체
파이프라인 검증 비용이 낮기 때문이다.

## Phase 5 — evaluation/reproduction

- [ ] PSNR/SSIM/LPIPS와 actor-mask PSNR
- [ ] reconstruction / novel-view CLI
- [ ] ablation flags: no-local, no-global, no-actor, no-depth, no-sky, no-pose-opt
- [x] reference hash-grid 100k/1M VRAM benchmark
- [ ] FPS/VRAM/quality benchmark
- [ ] 논문 표와 차이를 기록한 reproduction report

## 현재 코딩 범위

독립 ArmGS 패키지 안에서는 논문의 composite forward와 one-view training vertical
slice가 동작한다. 다음 구현 단위는 clean SplatAD/KITTI 데이터 경로를 연결하고
adaptive density control을 선택·검증하는 것이다. 실제 데이터 converter가 없으므로
아직 논문 표를 재현할 수 있는 완성된 dataset CLI는 아니다.
