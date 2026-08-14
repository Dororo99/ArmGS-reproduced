# ArmGS 구현 검증 현황

검증일: 2026-08-03

## 사용자 지적 사항 처리 결과

| 항목 | 상태 | 구현/검증 |
|---|---|---|
| background/actor/sky/trainer 부재 | 해결 | learnable composite scene, cubemap sky, paper-ordered renderer, one-view trainer |
| 통합 순서와 gradient 미검증 | 해결 | exact CPU ordering test와 실제 gsplat CUDA Eq. (9) backward |
| 전체 Gaussian hash-grid 메모리 | 해결 경로 구현 | conservative frustum selection + visible/chunked hash API |
| frustum/backend visibility | 해결 | near/far/eps2d 동기화와 CUDA edge parity; nonzero rolling shutter는 culling 해제 |
| SH backend parity | 해결 | 공식 3DGS CPU와 gsplat CPU/CUDA degree 0–3 비교 |
| 좌표/파라미터 convention | 해결 | OpenCV/OpenGL 변환, wxyz quaternion, activated renderer contract, raw learnable wrapper |
| near-zero quaternion | 해결 | 초기값과 매 forward에서 거부해 invisible splat/NaN gradient 방지 |
| actor별 H×W×A alpha | 해결 | 기본 aggregate O(HW), per-group은 명시적 진단 옵션 |
| actor alpha 지원 부재 | 해결 | active actor의 alpha=None을 숨기지 않고 strict foreground loss에서 실패 |
| actor lifecycle | 해결 | track 범위 밖 actor는 composite에서 제외하고 alpha zero map 제공 |
| timestamp aliasing/정밀도 | 해결 | module dtype cast 후에도 absolute time float64 보존, conditioning만 정규화 |
| affine 폭주 | 완화 | YAML에서 켜는 identity-centered tanh bounds |
| checkpoint/LR 안전성 | 해결 | CUDA map-location/GPU 수 변화 RNG 복원과 finite-positive LR 강제 |
| multi-frame exact resume | 해결 | stateful shuffle sampler의 순열/cursor와 topology 변경 모델·Adam state 복원 |
| dataset 경로 부재 | KITTI/nuScenes 경로 해결 | canonical loader, lazy RGB/mask batch, projected LiDAR union 보관 |
| split/pose leakage | 해결 | capture와 actor sample을 함께 분할하고 eval-only actor를 train scene에서 제거 |
| KITTI actor box 원점 | 해결 | bottom-center를 rotated local +z·h/2만큼 이동해 centered pose로 변환 |
| scene 초기화 부재 | 해결 | colored LiDAR/COLMAP 초기화와 tracked actor-local/background point 구축 |
| density control 부재 | 해결(가정 격리) | packed/unpacked gsplat 통계, duplicate/split/prune/reset, Adam migration, topology checkpoint |
| checkpoint dataset identity | 해결 | 모든 입력 stat digest와 작은 metadata content SHA-256으로 변경 감지 |
| trainer/evaluator CLI 부재 | 해결 | standalone evaluator와 nuScenes periodic/final/eval-only held-out 평가 |
| evaluator integer RGB | 해결 | uint8을 [0,1]로 정규화하고 그 외 정수 dtype은 명시적으로 거부 |
| SSIM/depth 정의 불명확 | 가정 격리 | training `ssim_loss=1-SSIM`과 held-out SSIM을 분리하고 window/sigma/range를 고정 |
| class-wise actor hash | 미구현 | 기본 sinusoidal 경로에는 불필요하며 관련 ablation만 남음 |

## 수치 검증

SH 4,096개 무작위 샘플의 최대 절대 오차:

| backend | degree 0–3 최대 오차 범위 |
|---|---:|
| 공식 CPU reference | 2.22e-16 – 1.78e-15 |
| gsplat CUDA | 0 – 1.43e-6 |

Hash-grid training forward/backward benchmark, RTX 4090, 8 levels × 2 features,
chunk size 65,536:

| 전체 Gaussian | visible/hash 적용 | peak allocated | elapsed |
|---:|---:|---:|---:|
| 100,000 | 100,000 | 241.6 MiB | 207.9 ms |
| 1,000,000 | 1,000,000 | 2,313.1 MiB | 142.9 ms |
| 1,000,000 | 100,000 | 260.7 MiB | 229.9 ms |

시간은 CUDA warm-up 순서에 민감하므로 메모리 비교가 주 목적이다.

전체 테스트:

~~~text
256 passed, 2 skipped, 5 warnings
~~~

경고 중 4개는 설치된 gsplat 환경의 `pkg_resources` deprecation이고, 1개는
checkpoint 테스트 중 발생한 PyTorch `TypedStorage` deprecation이다. skip 2개는
LPIPS가 이미 설치된 환경에서 missing-dependency 전용 테스트를 생략한 것이며 모두
ArmGS 테스트 실패는 아니다.

실제 nuScenes v1.0-trainval scene-0061 smoke:

- 39 captures, 6 cameras, 234 rows; train 204 / eval 30 rows
- 모든 camera row에 2,807개 이상의 projected LiDAR sample
- 동적 actor 66개, background 390,300 + actor 6,036 Gaussians 초기화
- CUDA 1-step forward/backward와 step/final checkpoint 저장 성공
- eval 30장 중 25장에 StreetGS-style projected actor cuboid silhouette
  (총 1,442,482 pixel) 존재
- per-camera/weighted PSNR·SSIM·LPIPS·actor-PSNR JSON/PNG/W&B 경로 검증
- 기존 30k `final.pt` eval-only 실측: PSNR 19.8258, SSIM 0.6864,
  LPIPS 0.4794, actor-PSNR 19.1210

LPIPS는 `lpips==0.1.4`, AlexNet v0.1, 공식 `[-1,1]` 입력 규약을 사용한다.
논문은 backbone과 전처리 세부값을 공개하지 않았으므로 paper-number exact parity가
아니라 명시적인 구현 프로토콜로 JSON/W&B config에 기록한다.

## 아직 남은 blocker

현재 코드는 독립 synthetic scene뿐 아니라 canonical KITTI 및 nuScenes 입력에서
학습 준비와 checkpoint 저장을 수행할 수 있다. 다만 다음 항목이 없어 모든 논문
데이터셋의 최종 수치를 바로 재현할 수는 없다.

1. Waymo/NOTR/VKITTI2 adapter
2. 실제 SplatAD/Nerfstudio model·dataparser 삽입과 rolling-shutter metadata 생산
3. 논문 규모용 fused hash-grid와 FPS/quality 검증
4. 매우 긴 sequence용 streaming projection/voxel fusion
5. class-wise actor hash-grid/hypernetwork ablation
6. 논문 표 수치 재현과 자동 차이 report

Density control에서 논문이 공개한 것은 500–15,000 step, interval 100의
schedule이다. duplicate/split/prune threshold, split 배율, opacity-reset 값과
정확한 semantics는 공개되지 않았다. 현재 구현은 이 값들을 YAML의 구현 가정으로
노출하고 opacity를 configured probability로 reset한 뒤 내부 logit으로 변환하며,
이를 “논문 설정”이나 clean SplatAD exact parity로 표시하지 않는다.
