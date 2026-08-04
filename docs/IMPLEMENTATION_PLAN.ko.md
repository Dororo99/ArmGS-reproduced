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

## Phase 3 — loss와 trainer (완료)

- [x] Eq. (9) weighted objective와 auxiliary input 강제
- [x] Gaussian parameter별 finite-positive Adam LR group
- [x] mean exponential LR decay
- [x] pose/refinement/sky parameter groups
- [x] one-view trainer와 multi-frame stateful sampling loop
- [x] model/optimizer/CPU·CUDA RNG checkpoint resume와 GPU 수 변화 안전성
- [x] dataset stat/content fingerprint 기반 resume identity 검증
- [x] packed/unpacked gsplat metadata의 projected gradient/radius 통계 누적
- [x] duplicate/split/prune/opacity reset lifecycle
- [x] adaptive topology 변경 이후 Adam state migration
- [x] topology 변경을 포함한 strict checkpoint resume
- [x] stateful shuffle sampler의 deterministic mid-epoch resume
- [x] KITTI training CLI, periodic checkpoint와 resume

논문에 공개된 density schedule은 500–15,000 step, interval 100이다. 반면
duplicate/split/prune threshold, split 배율과 opacity reset 값·동작은 공개되지
않았다. 현재 값과 reset-to-configured-probability 동작은 YAML에 구현 가정으로 표시하며,
논문 고유 설정 또는 clean baseline과의 exact parity로 주장하지 않는다.

## Phase 4 — dataset adapters (부분 완료)

- [x] canonical camera/frame/actor/LiDAR schema
- [x] KITTI camera/calibration/pose/timestamp/tracklet/Velodyne loader
- [x] lazy RGB/mask와 sparse projected LiDAR depth training batch
- [x] 요청 카메라 projected LiDAR union만 보관하는 memory-safe 기본 경로
- [x] capture/frame 단위 leak-free train/eval split
- [x] split별 actor pose sample 분리와 eval-only actor의 train scene 제외
- [x] KITTI bottom-center box를 centered canonical actor pose로 변환
- [x] training camera/timestamp와 source-row nearest-embedding metadata 검증
- [x] colored LiDAR Gaussian initialization
- [x] COLMAP `points3D.txt` parser와 단위 테스트
- [ ] COLMAP-to-world 정렬, train-view SfM 생성, LiDAR+SfM 병합 및 학습 CLI 연결
- [x] tracked actor-local point와 background scene 구축
- [ ] Waymo + StreetGS tracked boxes
- [ ] NOTR, VKITTI2
- [ ] 실제 dataparser에서 rolling-shutter velocity/shutter metadata 생산

KITTI를 먼저 선택하는 이유는 데이터·해상도·tracklet 형식이 비교적 작아 전체
파이프라인 검증 비용이 낮기 때문이다.

## Phase 5 — evaluation/reproduction (부분 완료)

- [x] PSNR/SSIM/optional LPIPS와 actor-mask PSNR
- [x] `.pt` pair/manifest metric evaluator CLI와 JSON summary
- [x] uint8 [0,255] → [0,1] 정규화와 비지원 integer dtype 거부
- [x] nuScenes trainer의 periodic/final held-out novel-view 평가와 eval-only resume
- [ ] 다른 dataset용 범용 reconstruction / novel-view rendering CLI
- [ ] ablation flags: no-local, no-global, no-actor, no-depth, no-sky, no-pose-opt
- [x] reference hash-grid 100k/1M VRAM benchmark
- [ ] FPS/VRAM/quality benchmark
- [ ] 논문 표와 차이를 기록한 reproduction report

## 현재 코딩 범위

독립 ArmGS 패키지 안에서는 composite forward, KITTI canonical loader/batching,
scene 초기화, adaptive density control, exact-resume trainer와 metric evaluator가
연결된다. 다음 구현 단위는 Waymo/NOTR/VKITTI2 adapter, 실제 SplatAD/Nerfstudio
삽입과 rolling metadata 생산, fused hash-grid/FPS·quality 측정이다. class-wise actor
ablation과 논문 표 재현은 그 통합 이후 단계이며, 공개되지 않은 density 설정은 계속
구현 가정으로 분리한다.
